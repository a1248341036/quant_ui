"""Persistent, evidence-backed memory for AlphaAgent factor research."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOPWORDS = {"add", "subtract", "multiply", "divide", "ts", "cs", "mean", "std", "rank", "expr", "factor"}

# 肯定（正向证据）verdict：已有明确有效性证据，应鼓励模型在其相似邻域继续探索
_POSITIVE_VERDICTS = {"production_approved", "validated", "candidate_approved", "promising"}
# 否定 verdict：已证明无效的路径，不应机械重复
_NEGATIVE_VERDICTS = {"rejected", "revise_required", "weak"}


def _neg_str(s: Any) -> str:
    """Return a string that sorts *before* all normal strings, so that when
    used as a secondary sort key in ascending order the *most recent*
    (lexicographically largest) timestamp comes first."""
    # Invert by prefixing with a high char — simple but effective for
    # ISO-8601 timestamps which are all ASCII.
    return "\uffff" + str(s)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None

_jieba_cut = None
try:  # 可选依赖：装 jieba 时用标准中文分词，否则回退 bigram
    import jieba  # type: ignore
    _jieba_cut = jieba.lcut_for_search
except Exception:  # pragma: no cover - 环境无 jieba 时走 bigram
    pass


def _cjk_tokens(text: str) -> set[str]:
    """抽取中文字段为可检索 token。

    优先使用 jieba（安装后生效），否则按连续中文串生成 bigram。
    """
    if not text:
        return set()
    if _jieba_cut is not None:
        words = {w.strip().lower() for w in _jieba_cut(text) if len(w.strip()) >= 2 and _CJK_RE.fullmatch(w.strip())}
        if words:
            return words
    out: set[str] = set()
    for seq in _CJK_RE.findall(text):
        if len(seq) == 2:
            out.add(seq)
            continue
        out.update(seq[i:i + 2] for i in range(len(seq) - 1))
    return out


def _tokens(*values: Any) -> list[str]:
    """从英文表达式/字段与中文结论中抽取去重 token，供 BM25 检索。"""
    found: set[str] = set()
    for value in values:
        text = str(value or "")
        for token in _TOKEN_RE.findall(text):
            token = token.lower()
            found.add(token)
            # 下划线连接的复合词拆开（momentum_reversal -> momentum + reversal），
            # 否则查询侧 "momentum reversal"（空格分词）与条目侧对不上
            found.update(part for part in token.split("_") if len(part) >= 2)
        found.update(_cjk_tokens(text))
    return sorted(found - _STOPWORDS)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class ResearchMemoryStore:
    """Stores compact research conclusions, never raw model thinking or prompts."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_entries (
        id TEXT PRIMARY KEY,
        factor_name TEXT NOT NULL,
        expression TEXT NOT NULL,
        conclusion TEXT,
        verdict TEXT NOT NULL,
        stage TEXT,
        profile_id TEXT,
        profile_hash TEXT,
        candidate_id TEXT,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        interaction_json TEXT,
        error TEXT,
        failure_code TEXT,
        last_run_id TEXT,
        attempts INTEGER NOT NULL DEFAULT 1,
        tokens_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS store_meta (
        k TEXT PRIMARY KEY,
        v TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_memory_entries_verdict
        ON memory_entries(verdict);
    CREATE INDEX IF NOT EXISTS idx_memory_entries_stage
        ON memory_entries(stage);
    CREATE INDEX IF NOT EXISTS idx_memory_entries_updated_at
        ON memory_entries(updated_at);
    CREATE INDEX IF NOT EXISTS idx_memory_entries_failure_code
        ON memory_entries(failure_code);

    CREATE TABLE IF NOT EXISTS memory_observations (
        entry_id TEXT NOT NULL,
        run_id TEXT,
        observed_at TEXT,
        stage TEXT,
        verdict TEXT,
        failure_code TEXT,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (entry_id, observed_at, run_id),
        FOREIGN KEY (entry_id) REFERENCES memory_entries(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_memory_observations_entry
        ON memory_observations(entry_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        entry_id UNINDEXED,
        factor_name,
        expression,
        conclusion,
        failure_code,
        search_tokens,
        tokenize='unicode61'
    );

    CREATE TRIGGER IF NOT EXISTS memory_entries_after_delete
    AFTER DELETE ON memory_entries
    BEGIN
        DELETE FROM memory_fts WHERE entry_id = old.id;
    END;
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() == ".json":
            # 存储已迁至 SQLite；容忍调用方传旧 JSON 路径（迁移逻辑仍读该 JSON）。
            self.path = self.path.with_suffix(".db")
        self._schema_ready = False

    @contextmanager
    def _open(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_schema(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.executescript(self._SCHEMA)
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_entries)").fetchall()
        }
        if "interaction_json" not in columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN interaction_json TEXT")
        self._migrate_legacy_json(conn)
        conn.commit()
        self._schema_ready = True

    def _migrate_legacy_json(self, conn: sqlite3.Connection) -> None:
        """One-time import from the original JSON store when switching to SQLite."""
        legacy_path = self.path.with_suffix(".json")
        if legacy_path == self.path or not legacy_path.is_file():
            return
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
            entries = raw.get("entries", []) if isinstance(raw, dict) else []
        except (OSError, json.JSONDecodeError):
            return
        if not any(isinstance(item, dict) and item.get("id") for item in entries):
            return

        count = conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        if count:
            return
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id"):
                self._write_entry(conn, self._normalize_entry(entry))

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(entry)
        observations = normalized.get("observations", [])
        if isinstance(observations, int):
            normalized["observations"] = [{
                "at": normalized.get("updated_at"),
                "verdict": normalized.get("verdict"),
            }]
        elif not isinstance(observations, list):
            normalized["observations"] = []
        return normalized

    def _load(self) -> list[dict[str, Any]]:
        with self._open() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_entries ORDER BY updated_at DESC"
            ).fetchall()
            return self._hydrate_entries(conn, rows)

    @staticmethod
    def _metrics_json(metrics: dict[str, Any]) -> str:
        return json.dumps(metrics, ensure_ascii=False, separators=(",", ":"))

    def _write_entry(self, conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO memory_entries (
                id, factor_name, expression, conclusion, verdict, stage,
                profile_id, profile_hash, candidate_id, metrics_json, interaction_json, error,
                failure_code, last_run_id, attempts, tokens_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                factor_name=excluded.factor_name,
                expression=excluded.expression,
                conclusion=excluded.conclusion,
                verdict=excluded.verdict,
                stage=excluded.stage,
                profile_id=excluded.profile_id,
                profile_hash=excluded.profile_hash,
                candidate_id=excluded.candidate_id,
                metrics_json=excluded.metrics_json,
                interaction_json=excluded.interaction_json,
                error=excluded.error,
                failure_code=excluded.failure_code,
                last_run_id=excluded.last_run_id,
                attempts=excluded.attempts,
                tokens_json=excluded.tokens_json,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at
            """,
            (
                entry["id"], entry.get("factor_name"), entry.get("expression"),
                entry.get("conclusion"), entry.get("verdict"), entry.get("stage"),
                entry.get("profile_id"), entry.get("profile_hash"),
                entry.get("candidate_id"), self._metrics_json(entry.get("metrics", {})),
                json.dumps(entry.get("interaction"), ensure_ascii=False) if entry.get("interaction") is not None else None,
                entry.get("error"), entry.get("failure_code"),
                entry.get("last_run_id"), int(entry.get("attempts", 1)),
                json.dumps(entry.get("tokens", []), ensure_ascii=False),
                entry.get("created_at"), entry.get("updated_at"),
            ),
        )
        conn.execute("DELETE FROM memory_observations WHERE entry_id = ?", (entry["id"],))
        conn.executemany(
            """
            INSERT OR IGNORE INTO memory_observations (
                entry_id, run_id, observed_at, stage, verdict, failure_code, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry["id"], obs.get("run_id"), obs.get("at"), obs.get("stage"),
                    obs.get("verdict"), obs.get("failure_code"),
                    self._metrics_json(obs.get("metrics", {})),
                )
                for obs in entry.get("observations", [])
                if isinstance(obs, dict)
            ],
        )
        conn.execute("DELETE FROM memory_fts WHERE entry_id = ?", (entry["id"],))
        conn.execute(
            """
            INSERT INTO memory_fts (
                entry_id, factor_name, expression, conclusion, failure_code, search_tokens
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry["id"], entry.get("factor_name") or "", entry.get("expression") or "",
                entry.get("conclusion") or "", entry.get("failure_code") or "",
                " ".join(entry.get("tokens", [])),
            ),
        )

    def _hydrate_entries(
        self,
        conn: sqlite3.Connection,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        obs_rows = conn.execute(
            f"""
            SELECT * FROM memory_observations
            WHERE entry_id IN ({placeholders})
            ORDER BY observed_at
            """,
            ids,
        ).fetchall()
        observations: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in rows}
        for row in obs_rows:
            observations.setdefault(row["entry_id"], []).append({
                "run_id": row["run_id"],
                "at": row["observed_at"],
                "stage": row["stage"],
                "verdict": row["verdict"],
                "failure_code": row["failure_code"],
                "metrics": json.loads(row["metrics_json"] or "{}"),
            })

        output: list[dict[str, Any]] = []
        for row in rows:
            item = {
                "id": row["id"],
                "factor_name": row["factor_name"],
                "expression": row["expression"],
                "conclusion": row["conclusion"],
                "verdict": row["verdict"],
                "stage": row["stage"],
                "profile_id": row["profile_id"],
                "profile_hash": row["profile_hash"],
                "candidate_id": row["candidate_id"],
                "metrics": json.loads(row["metrics_json"] or "{}"),
                "error": row["error"],
                "failure_code": row["failure_code"],
                "last_run_id": row["last_run_id"],
                "attempts": row["attempts"],
                "tokens": json.loads(row["tokens_json"] or "[]"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "observations": observations.get(row["id"], []),
            }
            output.append(item)
        return output

    def record_tool_result(self, *, run_id: str, row: dict[str, Any]) -> dict[str, Any] | None:
        name = str(row.get("name") or "")
        if name not in {"evaluate_factor", "eval_on_train_set", "eval_on_val_set", "submit_factor"}:
            return None
        args = _parse_args(row.get("arguments_raw"))
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        expression = str(args.get("multi_line_expr") or "").strip()
        factor_name = str(args.get("factor_name") or result.get("factor_name") or "expr").strip()
        if not expression:
            return None

        profile_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else profile_metrics.get("cross_sectional_core", {})
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else summary
        if name == "evaluate_factor":
            metrics = summary
        error = str(result.get("error") or result.get("skipped_reason") or "")
        verdict, conclusion = self._classify(name, result, metrics, error)
        canonical_expression = "\n".join(line.strip() for line in expression.splitlines() if line.strip())
        signature = hashlib.sha256(canonical_expression.encode("utf-8")).hexdigest()[:20]
        failure_code = self._failure_code(name, result, error, verdict)
        with self._open() as conn:
            previous_row = conn.execute(
                "SELECT attempts, created_at FROM memory_entries WHERE id = ?",
                (signature,),
            ).fetchone()
        previous_attempts = int(previous_row["attempts"]) if previous_row else 0
        observation = {
            "run_id": run_id,
            "at": _now(),
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "verdict": verdict,
            "failure_code": failure_code,
            "metrics": self._compact_metrics(metrics),
            "interaction": _parse_args(args.get("interaction")),
        }
        entry = {
            "id": signature,
            "factor_name": factor_name,
            "expression": expression,
            "tokens": _tokens(factor_name, expression, conclusion),
            "verdict": verdict,
            "conclusion": conclusion,
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "profile_id": result.get("profile", {}).get("profile_id") if isinstance(result.get("profile"), dict) else None,
            "profile_hash": result.get("profile_hash"),
            "candidate_id": result.get("candidate", {}).get("candidate_id") if isinstance(result.get("candidate"), dict) else None,
            "metrics": self._compact_metrics(metrics),
            "error": error[:500],
            "failure_code": failure_code,
            "last_run_id": run_id,
            "attempts": previous_attempts + 1,
            "updated_at": _now(),
            "observations": [observation],
        }

        if previous_row:
            entry["created_at"] = previous_row["created_at"] or entry["updated_at"]
            with self._open() as conn:
                old_rows = conn.execute(
                    """
                    SELECT * FROM memory_observations
                    WHERE entry_id = ?
                    ORDER BY observed_at
                    """,
                    (signature,),
                ).fetchall()
            old_observations = [
                {
                    "run_id": row["run_id"],
                    "at": row["observed_at"],
                    "stage": row["stage"],
                    "verdict": row["verdict"],
                    "failure_code": row["failure_code"],
                    "metrics": json.loads(row["metrics_json"] or "{}"),
                }
                for row in old_rows
            ]
            entry["observations"] = [*old_observations[-98:], observation]
        else:
            entry["created_at"] = entry["updated_at"]

        with self._open() as conn:
            self._write_entry(conn, entry)
        return entry

    @staticmethod
    def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        keys = (
            "ic", "icir", "rank_ic", "factor_coverage", "coverage",
            "long_group_annual_excess_return", "winsorized_abs_ic_decay",
            "annualized_return", "annualized_excess_return", "sharpe",
            "max_drawdown", "annual_turnover",
        )
        return {key: value for key in keys if (value := _safe_float(metrics.get(key))) is not None}

    @staticmethod
    def _failure_code(name: str, result: dict[str, Any], error: str, verdict: str) -> str | None:
        if not error and verdict not in {"weak", "rejected", "revise_required"}:
            return None
        text = error.lower()
        if "timeout" in text:
            return "model_timeout" if "model" in text else "eval_timeout"
        if "corr" in text or "similar" in text or "duplicate" in text:
            return "correlation_duplicate"
        if "sign" in text:
            return "sign_flip"
        if name == "submit_factor":
            if "engine_gate" in text:
                return "backtest_failed"
            return "stage_one_failed" if not result.get("candidate_stored") else "stage_two_failed"
        if name == "eval_on_train_set":
            return "train_threshold"
        if name == "eval_on_val_set":
            return "val_threshold"
        return "tool_error" if error else "metric_threshold"

    @staticmethod
    def _classify(name: str, result: dict[str, Any], metrics: dict[str, Any], error: str) -> tuple[str, str]:
        review = result.get("factor_review") if isinstance(result.get("factor_review"), dict) else {}
        review_verdict = review.get("verdict")
        canonical = str(review.get("canonical_form") or "因子结构审核")
        reasons = review.get("reasons") if isinstance(review.get("reasons"), list) else []
        if review_verdict == "reject":
            return "rejected", f"{canonical}：Reviewer 拒绝；{str(reasons[0]) if reasons else '不得重复同构表达式。'}"
        if review_verdict == "revise":
            return "revise_required", f"{canonical}：Reviewer 要求结构性改造后再评估。"
        if error:
            return "rejected", f"{name} 被否定：{error}"
        if name == "submit_factor":
            if result.get("stored"):
                return "production_approved", "已通过精筛并正式入库，应保留其经济机制并避免重复。"
            if result.get("candidate_stored"):
                return "candidate_approved", "通过海选进入候选池，尚未满足精筛条件，应针对失败项改进。"
            return "rejected", "提交未通过，避免在未改变机制或拒绝原因的情况下重复提交。"
        ic = _safe_float(metrics.get("ic"))
        icir = _safe_float(metrics.get("icir"))
        coverage = _safe_float(metrics.get("factor_coverage", metrics.get("coverage")))
        is_val = name == "eval_on_val_set" or result.get("split") == "val"
        if is_val and (result.get("sign_check", {}).get("matches_expected_sign") is not False) and abs(ic or 0) >= 0.015:
            return "validated", "训练外方向一致且具有可用相关性，可扩展为相邻但不重复的机制。"
        if abs(ic or 0) >= 0.015 and (icir or 0) > 0.2 and (coverage or 0) > 0.85:
            return "promising", "训练阶段指标有潜力，优先进行训练外验证或独立性改造。"
        return "weak", "当前指标不足；除非改变变量、经济机制或处理方式，否则不要机械重试。"

    # Verdict 分类
    _POSITIVE_VERDICTS = frozenset({"production_approved", "validated", "candidate_approved", "promising"})
    _NEGATIVE_VERDICTS = frozenset({"rejected", "revise_required", "weak"})

    def delete_entry(self, entry_id: str) -> bool:
        with self._open() as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    @staticmethod
    def entry_signature(expr: str) -> str:
        """与 record_tool_result 一致的表达式签名（规范换行后 sha256 前 20 位）。"""
        canonical = "\n".join(line.strip() for line in (expr or "").splitlines() if line.strip())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    def purge_factor(
        self,
        *,
        factor_names: list[str] | tuple[str, ...] = (),
        expressions: list[str] | tuple[str, ...] = (),
    ) -> int:
        """删除与指定因子相关的全部记忆条目（含 FTS 与观察记录，经级联/触发器）。

        匹配规则：表达式签名精确命中，或 factor_name 精确命中。返回删除条数。
        """
        ids = {self.entry_signature(e) for e in expressions if e}
        names = [n for n in factor_names if n]
        if not ids and not names:
            return 0
        deleted = 0
        with self._open() as conn:
            for eid in ids:
                cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (eid,))
                deleted += cursor.rowcount
            for name in names:
                cursor = conn.execute("DELETE FROM memory_entries WHERE factor_name = ?", (name,))
                deleted += cursor.rowcount
        return deleted

    def context_for(
        self,
        research_goal: str,
        *,
        limit: int = 12,
        include_rejected: bool = True,
        prefer_orthogonal: bool = True,
        include_expression: bool = True,
        max_expression_chars: int | None = None,
    ) -> str:
        """Build a compact, retrieval-based research memory context block.

        Retrieval uses a local BM25 scoring over each entry's token set (zero
        API cost).  Entries are split into **positive** (validated, promising…)
        and **negative** (rejected, weak…) pools, scored independently, then
        merged with a guaranteed minimum quota for positive entries — so that
        known-good factor families always surface even when negatives vastly
        outnumber them (which is the common case after many dead-end attempts).
        Output is split into two sections — positive then negative — to make
        the contrast immediately legible to the LLM.
        """
        entries = self._retrieval_candidates(research_goal, include_rejected)
        if not include_rejected:
            entries = [entry for entry in entries if entry.get("verdict") not in self._NEGATIVE_VERDICTS]
        if not entries:
            return ""
        verdict_rank = {
            "production_approved": 5, "validated": 4, "candidate_approved": 3,
            "promising": 2, "revise_required": 1, "rejected": 1, "weak": 0,
        }

        def _key(entry: dict[str, Any], bm: float) -> tuple[float, int, int, str]:
            observations = entry.get("observations", [])
            n = observations if isinstance(observations, int) else len(observations)
            return (bm, verdict_rank.get(str(entry.get("verdict")), 0), n, str(entry.get("updated_at", "")))

        # Split into positive / negative pools and score independently so
        # that a large negative pool can't crowd out all positives.
        positive_pool = [(e, float(e.pop("_bm25", 0.0))) for e in entries if e.get("verdict") in self._POSITIVE_VERDICTS]
        negative_pool = [(e, float(e.pop("_bm25", 0.0))) for e in entries if e.get("verdict") in self._NEGATIVE_VERDICTS]

        positive_pool.sort(key=lambda pair: _key(pair[0], pair[1]), reverse=True)
        negative_pool.sort(key=lambda pair: _key(pair[0], pair[1]), reverse=True)

        # Guarantee at least 40% of slots go to positive entries (when available),
        # but don't waste slots on irrelevant positives with zero BM25 overlap.
        positive_quota = max(1, int(limit * 0.4)) if positive_pool else 0
        # Only include positive entries with non-zero BM25 relevance, or fall
        # back to top verdict-only entries if none overlap (still useful as
        # "here's what worked before" even if off-topic).
        positive_relevant = [(e, b) for e, b in positive_pool if b > 0]
        if not positive_relevant and positive_pool:
            positive_relevant = positive_pool[:positive_quota]
        positive_selected = positive_relevant[:positive_quota]
        n_pos = len(positive_selected)
        n_neg = limit - n_pos
        negative_selected = negative_pool[:n_neg]
        # If we didn't fill all positive slots (fewer available), give
        # remaining slots to negatives.
        if n_pos < positive_quota:
            negative_selected = negative_pool[:limit - n_pos]

        positive = positive_selected
        negative = negative_selected

        lines = [
            "# 长期研究记忆（来自真实评估与提交结果）",
            "以下结论必须作为实验先验。",
        ]

        # ── 肯定段 ──
        if positive:
            lines.append("")
            lines.append("## 已验证 / 有潜力的因子（优先在其邻近空间继续挖掘相似机制）")
            lines.append(
                "这些因子在训练集或验证集上表现可用。**鼓励**基于它们的经济逻辑，"
                "通过更换窗口、算子族或原始字段，在相似但不重复的方向上继续探索。"
            )
            if prefer_orthogonal:
                lines.append("扩展时优先引入正交变量，避免仅改窗口长度的同质微调。")
            for entry, _ in positive:
                lines.append(self._format_entry(entry, include_expression, max_expression_chars))

        # ── 否定段 ──
        if negative:
            lines.append("")
            lines.append("## 已否定 / 不足的因子（避免机械重复同一死路）")
            lines.append(
                "以下路径已被评估否定。除非改变了核心变量、经济机制或处理方式，"
                "否则不要重复尝试相同结构。"
            )
            for entry, _ in negative:
                lines.append(self._format_entry(entry, include_expression, max_expression_chars))

        return "\n".join(lines)

    @staticmethod
    def _fts_match_query(tokens: set[str]) -> str:
        return " OR ".join(
            '"' + str(token).replace('"', '""') + '"'
            for token in sorted(tokens)
        )

    def _retrieval_candidates(
        self,
        research_goal: str,
        include_rejected: bool,
        *,
        scan_limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Fetch a small FTS-ranked plus recency candidate set, never all history."""
        filters = "" if include_rejected else (
            f"AND e.verdict NOT IN ({','.join('?' for _ in self._NEGATIVE_VERDICTS)})"
        )
        filter_args = tuple() if include_rejected else tuple(sorted(self._NEGATIVE_VERDICTS))
        candidates: dict[str, dict[str, Any]] = {}

        with self._open() as conn:
            recent_rows = conn.execute(
                f"""
                SELECT e.*, 0.0 AS relevance
                FROM memory_entries e
                WHERE 1=1 {filters}
                ORDER BY e.updated_at DESC
                LIMIT ?
                """,
                (*filter_args, int(scan_limit)),
            ).fetchall()
            hydrated_recent = self._hydrate_entries(conn, recent_rows)
            for row, entry in zip(recent_rows, hydrated_recent):
                entry["_bm25"] = float(row["relevance"])
                candidates[entry["id"]] = entry

            tokens = set(_tokens(research_goal))
            match_query = self._fts_match_query(tokens)
            if match_query:
                matched_rows = conn.execute(
                    f"""
                    SELECT e.*, -bm25(memory_fts) AS relevance
                    FROM memory_fts
                    JOIN memory_entries e ON e.id = memory_fts.entry_id
                    WHERE memory_fts MATCH ? {filters}
                    ORDER BY relevance DESC
                    LIMIT ?
                    """,
                    (match_query, *filter_args, int(scan_limit)),
                ).fetchall()
                hydrated_matched = self._hydrate_entries(conn, matched_rows)
                for row, entry in zip(matched_rows, hydrated_matched):
                    existing = candidates.get(entry["id"])
                    entry["_bm25"] = float(row["relevance"])
                    if existing is None:
                        candidates[entry["id"]] = entry
                    else:
                        existing["_bm25"] = max(
                            float(existing.get("_bm25", 0.0)),
                            float(row["relevance"]),
                        )

        return list(candidates.values())

    def query_for_attempts(
        self,
        research_goal: str,
        recent_rows: list[dict[str, Any]],
        *,
        max_recent_attempts: int = 8,
        max_expression_chars: int = 1200,
    ) -> str:
        """Build a retrieval query from the goal plus the current run's latest work."""
        parts = [str(research_goal or "A股日频因子挖掘")]
        for row in list(recent_rows)[-max(0, int(max_recent_attempts)):]:
            args = _parse_args(row.get("arguments_raw"))
            factor_name = str(args.get("factor_name") or "").strip()
            expression = str(args.get("multi_line_expr") or "").strip()
            error = str(row.get("error") or "").strip()
            if factor_name:
                parts.append(factor_name)
            if expression:
                parts.append(expression[:max_expression_chars])
            if error:
                parts.append(error[:300])
            interaction = args.get("interaction")
            if isinstance(interaction, dict):
                parts.append(str(interaction.get("interaction_type") or ""))
                parts.append(str(interaction.get("base_signal") or ""))
                parts.append(str(interaction.get("condition_signal") or ""))
                parts.append(str(interaction.get("economic_mechanism") or ""))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _format_entry(
        entry: dict[str, Any],
        include_expression: bool,
        max_expression_chars: int | None = None,
    ) -> str:
        m = entry.get("metrics", {})
        metric_text = " ".join(
            f"{key}={value:.4g}" for key, value in m.items() if isinstance(value, (int, float))
        )
        interaction = entry.get("interaction") if isinstance(entry.get("interaction"), dict) else {}
        interaction_text = ""
        if interaction:
            interaction_text = f"；交互类型={interaction.get('interaction_type')}；条件={interaction.get('condition_signal')}"
        expr = entry.get("expression")
        if include_expression and expr and max_expression_chars is not None:
            text = str(expr)
            if int(max_expression_chars) <= 0:
                expr = None
            elif len(text) > int(max_expression_chars):
                text = text[:max(int(max_expression_chars) - 3, 0)] + "..."
            expr = text
        expr_tail = f"；表达式：{expr}" if (include_expression and expr) else ""
        return (
            f"- [{entry.get('verdict')}] {entry.get('factor_name')}: {entry.get('conclusion')} "
            f"指标({metric_text or '无'}){interaction_text}{expr_tail}"
        )

    # Verdict display order — positive first, then negative.
    _VERDICT_ORDER = {
        "production_approved": 0,
        "validated": 1,
        "candidate_approved": 2,
        "promising": 3,
        "revise_required": 4,
        "rejected": 5,
        "weak": 6,
    }

    def recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return entries sorted by verdict priority (positive first) then
        recency, so the frontend always shows validated/promising factors
        even when recent runs produced mostly rejections.
        """
        case_parts = " ".join(
            f"WHEN '{verdict}' THEN {rank}"
            for verdict, rank in self._VERDICT_ORDER.items()
        )
        with self._open() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_entries
                ORDER BY CASE verdict {case_parts} ELSE 99 END,
                         updated_at DESC
                LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()
            return self._hydrate_entries(conn, rows)

    def statistics(self) -> dict[str, Any]:
        with self._open() as conn:
            entry_count = int(conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0])
            obs_rows = conn.execute(
                """
                SELECT e.verdict AS verdict, COUNT(*) AS n
                FROM memory_observations o
                JOIN memory_entries e ON e.id = o.entry_id
                GROUP BY e.verdict
                """
            ).fetchall()
        counts = {str(row["verdict"] or "unknown"): int(row["n"]) for row in obs_rows}
        attempts = sum(counts.values())
        return {
            "entries": entry_count,
            "observations": attempts,
            "verdict_counts": counts,
            "train_to_validated_rate": round(counts.get("validated", 0) / attempts, 4) if attempts else None,
            "production_rate": round(counts.get("production_approved", 0) / attempts, 4) if attempts else None,
        }

    def backfill_from_logs(self, log_root: Path) -> int:
        """Populate an empty memory store from prior UI JSONL event logs once.

        以 store_meta.backfill_done 标记防止「删空记忆后重启又被全量复活」：
        只要标记存在（无论库内是否还有条目），就不再回放历史日志。
        """
        with self._open() as conn:
            existing = conn.execute("SELECT 1 FROM memory_entries LIMIT 1").fetchone()
            done = conn.execute("SELECT v FROM store_meta WHERE k = 'backfill_done'").fetchone()
        if existing or done:
            if not done:
                with self._open() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done', '1')"
                    )
            return 0
        count = 0
        for log_path in sorted(Path(log_root).glob("*/run_*.jsonl")):
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "tool_results":
                    continue
                for row in event.get("results") or []:
                    if isinstance(row, dict) and self.record_tool_result(run_id=log_path.parent.name, row=row):
                        count += 1
        with self._open() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done', '1')"
            )
        return count

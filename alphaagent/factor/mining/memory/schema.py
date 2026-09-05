# -*- coding: utf-8 -*-
"""SQLite schema、连接管理和基础指标规范化。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .calibration import APV_TAU_C_DEFAULT, APV_TAU_V_DEFAULT, EQ7_KAPPA_DEFAULT, _parent_bucket
from .constants import (
    DATA_VERSION,
    EDIT_PRIOR_HARD_CONF_DEFAULT,
    EDIT_PRIOR_RECOMMEND_CONF_DEFAULT,
    EDIT_PRIOR_VETO_CONF_DEFAULT,
    POSITIVE_VERDICTS,
)
from .diagnostics import _failure_code, _now, _safe_float
from .expressions import (
    _structure_fingerprint,
    classify_family,
    expr_facets,
    extract_edit_motif,
    is_cross_group_fusion,
)


def _compute_motif(parent_expr: str, child_expr: str) -> str:
    """从父本→子本表达式对提取 motif（迁移用）。"""
    try:
        return extract_edit_motif(parent_expr or "", child_expr or "")
    except Exception:
        return "other"


class SchemaMixin:
    """SQLite 存储底座：v3 schema、连接生命周期和指标扁平化。"""

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
        fail_detail TEXT,
        mechanism TEXT,
        family TEXT,
        stage_metrics_json TEXT NOT NULL DEFAULT '{}',
        last_run_id TEXT,
        attempts INTEGER NOT NULL DEFAULT 1,
        tokens_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT,
        updated_at TEXT,
        structure_fingerprint TEXT,
        operator_list_json TEXT,
        window_params_json TEXT,
        parent_id TEXT,
        parent_origin TEXT,
        intended_motif TEXT,
        edit_note TEXT
    );

    CREATE TABLE IF NOT EXISTS store_meta (
        k TEXT PRIMARY KEY,
        v TEXT
    );

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

    CREATE TABLE IF NOT EXISTS memory_patterns (
        id TEXT PRIMARY KEY,
        layer TEXT NOT NULL,
        category TEXT,
        content TEXT NOT NULL,
        evidence_json TEXT,
        success_rate REAL,
        total_attempts INTEGER NOT NULL DEFAULT 1,
        success_count INTEGER NOT NULL DEFAULT 0,
        saturation_score REAL NOT NULL DEFAULT 0,
        confidence REAL NOT NULL DEFAULT 0.5,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS edit_patterns (
        id TEXT PRIMARY KEY,
        parent_expression TEXT,
        child_expression TEXT NOT NULL,
        parent_fingerprint TEXT,
        child_fingerprint TEXT NOT NULL,
        edit_type TEXT NOT NULL,
        edit_detail_json TEXT,
        parent_ic REAL,
        child_ic REAL,
        delta_ic REAL,
        parent_icir REAL,
        child_icir REAL,
        delta_icir REAL,
        verdict TEXT,
        confidence REAL NOT NULL DEFAULT 0.5,
        total_uses INTEGER NOT NULL DEFAULT 0,
        success_count INTEGER NOT NULL DEFAULT 0,
        vetoed INTEGER NOT NULL DEFAULT 0,
        family TEXT,
        parent_entry_id TEXT,
        child_entry_id TEXT,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        entry_id UNINDEXED,
        factor_name,
        expression,
        conclusion,
        failure_code,
        search_tokens,
        tokenize='unicode61'
    );

    CREATE TABLE IF NOT EXISTS memory_experience (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        template TEXT,
        evidence_json TEXT,
        example_factors_json TEXT NOT NULL DEFAULT '[]',
        correlated_json TEXT NOT NULL DEFAULT '[]',
        typical_correlation REAL,
        occurrence_count INTEGER NOT NULL DEFAULT 1,
        run_id TEXT,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TRIGGER IF NOT EXISTS memory_entries_after_delete
    AFTER DELETE ON memory_entries
    BEGIN
        DELETE FROM memory_fts WHERE entry_id = old.id;
    END;
    """

    def __init__(
        self,
        path: Path,
        *,
        apv_tau_c: float = APV_TAU_C_DEFAULT,
        apv_tau_v: float = APV_TAU_V_DEFAULT,
        eq7_kappa: float = EQ7_KAPPA_DEFAULT,
        max_inject_chars: int = 2400,
        hard_block_duplicates: bool = False,
        edit_prior_hard_conf: float = EDIT_PRIOR_HARD_CONF_DEFAULT,
        edit_prior_recommend_conf: float = EDIT_PRIOR_RECOMMEND_CONF_DEFAULT,
        edit_prior_veto_conf: float = EDIT_PRIOR_VETO_CONF_DEFAULT,
        suggest_slots: int = 2,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() == ".json":
            # 存储已迁至 SQLite；容忍调用方传旧 JSON 路径（迁移逻辑仍读该 JSON）。
            self.path = self.path.with_suffix(".db")
        self.apv_tau_c = apv_tau_c
        self.apv_tau_v = apv_tau_v
        self.eq7_kappa = eq7_kappa
        self.max_inject_chars = int(max_inject_chars or 2400)
        self.hard_block_duplicates = bool(hard_block_duplicates)
        self.edit_prior_hard_conf = float(edit_prior_hard_conf)
        self.edit_prior_recommend_conf = float(edit_prior_recommend_conf)
        self.edit_prior_veto_conf = float(edit_prior_veto_conf)
        self.suggest_slots = max(0, int(suggest_slots))
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

        # v1 存量库（遗留结构）先建表，再补列，最后建索引——避免 executescript
        # 中"先建 idx 索引、后补列"导致 "no such column"。
        # 先执行"仅建表"的 DDL：逐条执行以 CREATE TABLE 开头的完整语句，
        # 触发器/VIRTUAL TABLE 内部的分号会产生碎片，单独在后面处理。
        for stmt in self._SCHEMA.split(";"):
            s = stmt.strip()
            if s.upper().startswith("CREATE TABLE"):
                conn.execute(s)

        # memory_entries 列迁移（v1→v3 逐列补齐）
        entry_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_entries)").fetchall()
        }
        _ENTRY_COLUMNS = {
            "interaction_json": "TEXT",
            "structure_fingerprint": "TEXT",
            "operator_list_json": "TEXT",
            "window_params_json": "TEXT",
            "parent_id": "TEXT",
            "fail_detail": "TEXT",
            "mechanism": "TEXT",
            "family": "TEXT",
            "stage_metrics_json": "TEXT NOT NULL DEFAULT '{}'",
            "parent_origin": "TEXT",
            "intended_motif": "TEXT",
            "edit_note": "TEXT",
            "facets_json": "TEXT",
        }
        for col, ddl in _ENTRY_COLUMNS.items():
            if col not in entry_columns:
                conn.execute(f"ALTER TABLE memory_entries ADD COLUMN {col} {ddl}")

        # memory_patterns 列（v1 兼容）
        pattern_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_patterns)").fetchall()
        }
        _PATTERN_COLUMNS = {
            "structure_fingerprint": "TEXT",
            "edit_types_json": "TEXT",
            "operator_combo_json": "TEXT",
            "parent_context_json": "TEXT",
        }
        for col, ddl in _PATTERN_COLUMNS.items():
            if col not in pattern_columns:
                conn.execute(f"ALTER TABLE memory_patterns ADD COLUMN {col} {ddl}")

        # memory_cells：v2 结构（cell_key 主键）→ v3 结构（family/motif/parent_bucket 主键 + 四列计数）
        self._ensure_cells_schema(conn)

        # 补索引（v1 建表时无索引；此处再建）
        for stmt in self._SCHEMA.split(";"):
            s = stmt.strip()
            if s.upper().startswith("CREATE INDEX"):
                conn.execute(s)

        # FTS 表与触发器（v1 库缺）
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                entry_id UNINDEXED,
                factor_name,
                expression,
                conclusion,
                failure_code,
                search_tokens,
                tokenize='unicode61'
            )
            """
        )
        # memory_experience（v1/v2 存量库可能缺；CREATE TABLE IF NOT EXISTS 兜底）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_experience (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                template TEXT,
                evidence_json TEXT,
                example_factors_json TEXT NOT NULL DEFAULT '[]',
                correlated_json TEXT NOT NULL DEFAULT '[]',
                typical_correlation REAL,
                occurrence_count INTEGER NOT NULL DEFAULT 1,
                run_id TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_entries_after_delete
            AFTER DELETE ON memory_entries
            BEGIN
                DELETE FROM memory_fts WHERE entry_id = old.id;
            END;
            """
        )

        self._migrate_legacy_json(conn)
        self._ensure_data_version(conn)
        conn.commit()
        self._schema_ready = True

    def _ensure_cells_schema(self, conn: sqlite3.Connection) -> None:
        """memory_cells 表升级到 v3 结构。

        v3 cells 键 = (family, motif, parent_bucket)，列含四类加权计数与残差。
        v1/v2 旧表为 cell_key 主键 + successes/failures 列 → 重建为 v3 结构。
        """
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_cells'"
        ).fetchall()
        if not rows:
            conn.execute(
                """
                CREATE TABLE memory_cells (
                    family TEXT NOT NULL,
                    motif TEXT NOT NULL,
                    parent_bucket TEXT NOT NULL,
                    explicit_s REAL NOT NULL DEFAULT 0,
                    explicit_f REAL NOT NULL DEFAULT 0,
                    implicit_s REAL NOT NULL DEFAULT 0,
                    implicit_f REAL NOT NULL DEFAULT 0,
                    residuals_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT,
                    PRIMARY KEY (family, motif, parent_bucket)
                )
                """
            )
            return
        # 旧表存在：确认列；若缺 v3 列则重建（保留数据会由 _backfill_v3_conn 重放）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_cells)").fetchall()}
        needs = {"family", "motif", "parent_bucket", "explicit_s", "explicit_f", "implicit_s", "implicit_f", "residuals_json"}
        if needs <= cols:
            # 确保主键为 v3 组合键（旧 v2 表主键是 cell_key）——不强制重建，但补齐缺失列
            return
        conn.execute("DROP TABLE IF EXISTS memory_cells")
        conn.execute(
            """
            CREATE TABLE memory_cells (
                family TEXT NOT NULL,
                motif TEXT NOT NULL,
                parent_bucket TEXT NOT NULL,
                explicit_s REAL NOT NULL DEFAULT 0,
                explicit_f REAL NOT NULL DEFAULT 0,
                implicit_s REAL NOT NULL DEFAULT 0,
                implicit_f REAL NOT NULL DEFAULT 0,
                residuals_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT,
                PRIMARY KEY (family, motif, parent_bucket)
            )
            """
        )

    def _ensure_data_version(self, conn: sqlite3.Connection) -> None:
        """幂等升级 data_version 到当前版（在索引建好后执行，避免 v1 库炸）。

        v3→v4 仅加列（facets_json，见 _ensure_schema 的逐列补齐），无需重放
        cells：family 新口径只影响新写入，存量行由读取侧现算兜底。
        """
        row = conn.execute("SELECT v FROM store_meta WHERE k='data_version'").fetchone()
        if row is None:
            # v1 存量库（无 store_meta 行）：同样重放 cells 并补 parent_origin
            self._backfill_v3_conn(conn)
            conn.execute("INSERT OR REPLACE INTO store_meta(k, v) VALUES ('data_version', ?)", (DATA_VERSION,))
            return
        if str(row["v"]) == DATA_VERSION:
            return
        # v2/v3 → 当前版：重放 cells 并补 parent_origin（v4 的 facets_json
        # 已在 _ensure_schema 逐列补齐，这里不重复处理）
        self._backfill_v3_conn(conn)
        conn.execute("INSERT OR REPLACE INTO store_meta(k, v) VALUES ('data_version', ?)", (DATA_VERSION,))

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

    @staticmethod
    def _metrics_json(metrics: dict[str, Any]) -> str:
        return json.dumps(metrics, ensure_ascii=False, separators=(",", ":"))

    def _write_entry(self, conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO memory_entries (
                id, factor_name, expression, conclusion, verdict, stage,
                profile_id, profile_hash, candidate_id, metrics_json, interaction_json, error,
                failure_code, fail_detail, mechanism, family, stage_metrics_json,
                last_run_id, attempts, tokens_json,
                created_at, updated_at,
                structure_fingerprint, operator_list_json, window_params_json, parent_id,
                parent_origin, intended_motif, edit_note, facets_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                fail_detail=excluded.fail_detail,
                mechanism=excluded.mechanism,
                family=excluded.family,
                stage_metrics_json=excluded.stage_metrics_json,
                last_run_id=excluded.last_run_id,
                attempts=excluded.attempts,
                tokens_json=excluded.tokens_json,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                structure_fingerprint=excluded.structure_fingerprint,
                operator_list_json=excluded.operator_list_json,
                window_params_json=excluded.window_params_json,
                parent_id=excluded.parent_id,
                parent_origin=excluded.parent_origin,
                intended_motif=excluded.intended_motif,
                edit_note=excluded.edit_note,
                facets_json=excluded.facets_json
            """,
            (
                entry["id"], entry.get("factor_name"), entry.get("expression"),
                entry.get("conclusion"), entry.get("verdict"), entry.get("stage"),
                entry.get("profile_id"), entry.get("profile_hash"),
                entry.get("candidate_id"), self._metrics_json(entry.get("metrics", {})),
                json.dumps(entry.get("interaction"), ensure_ascii=False) if entry.get("interaction") is not None else None,
                entry.get("error"), entry.get("failure_code"),
                entry.get("fail_detail"), entry.get("mechanism"), entry.get("family"),
                self._metrics_json(entry.get("stage_metrics", {})),
                entry.get("last_run_id"), int(entry.get("attempts", 1)),
                json.dumps(entry.get("tokens", []), ensure_ascii=False),
                entry.get("created_at"), entry.get("updated_at"),
                entry.get("structure_fingerprint"),
                entry.get("operator_list_json"),
                entry.get("window_params_json"),
                entry.get("parent_id"),
                entry.get("parent_origin"),
                entry.get("intended_motif"),
                entry.get("edit_note"),
                entry.get("facets_json"),
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
                "fail_detail": row["fail_detail"],
                "mechanism": row["mechanism"],
                "family": row["family"],
                "stage_metrics": json.loads(row["stage_metrics_json"] or "{}"),
                "last_run_id": row["last_run_id"],
                "attempts": row["attempts"],
                "tokens": json.loads(row["tokens_json"] or "[]"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "observations": observations.get(row["id"], []),
                "parent_origin": row["parent_origin"],
                "intended_motif": row["intended_motif"],
                "edit_note": row["edit_note"],
            }
            # 数据面标签（2026-09-03）：读 facets_json，老行为空时按表达式现算兜底
            facets: set[str] = set()
            row_keys = row.keys()
            if "facets_json" in row_keys and row["facets_json"]:
                try:
                    facets = set(json.loads(row["facets_json"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    facets = set()
            if not facets:
                facets = expr_facets(str(row["expression"] or ""))
            item["facets"] = sorted(facets)
            item["is_fusion"] = is_cross_group_fusion(facets)
            output.append(item)
        return output

    @staticmethod
    def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        keys = (
            "ic", "icir", "rank_ic", "factor_coverage", "coverage",
            "cs_pearson_autocorr",
            "long_group_annual_excess_return", "winsorized_abs_ic_decay",
            "annualized_return", "annualized_excess_return", "sharpe",
            "max_drawdown", "annual_turnover", "daily_overlap",
            "excess_sharpe", "monotonicity",
        )
        return {key: value for key in keys if (value := _safe_float(metrics.get(key))) is not None}

    @staticmethod
    def _flatten_backtest_metrics(metrics: dict[str, Any], profile_metrics: dict[str, Any]) -> dict[str, Any]:
        """把 quantile_portfolio / topn_portfolio / engine_backtest 的嵌套回测指标
        提取到扁平 metrics，供 _compact_metrics 统一捕获。"""
        flat = dict(metrics)
        qp = profile_metrics.get("quantile_portfolio") if isinstance(profile_metrics, dict) else None
        if isinstance(qp, dict):
            flat.setdefault("annualized_return", qp.get("top_group_annualized_return"))
            flat.setdefault("annualized_excess_return", qp.get("top_group_annualized_excess_return"))
            flat.setdefault("sharpe", qp.get("top_group_sharpe"))
            flat.setdefault("excess_sharpe", qp.get("top_group_excess_sharpe"))
            flat.setdefault("max_drawdown", qp.get("top_group_max_drawdown"))
            flat.setdefault("annual_turnover", qp.get("avg_daily_side_turnover"))
            flat.setdefault("monotonicity", qp.get("monotonicity"))
        tp = profile_metrics.get("topn_portfolio") if isinstance(profile_metrics, dict) else None
        if isinstance(tp, dict):
            flat.setdefault("annualized_return", tp.get("annualized_return"))
            flat.setdefault("annualized_excess_return", tp.get("annualized_excess_return"))
            flat.setdefault("sharpe", tp.get("sharpe"))
            flat.setdefault("max_drawdown", tp.get("max_drawdown"))
            flat.setdefault("daily_overlap", tp.get("daily_overlap"))
            flat.setdefault("excess_sharpe", tp.get("excess_sharpe"))
        eb = profile_metrics.get("engine_backtest") if isinstance(profile_metrics, dict) else None
        if isinstance(eb, dict):
            m = eb.get("metrics") if isinstance(eb.get("metrics"), dict) else eb
            flat.setdefault("annualized_return", m.get("annual_return"))
            flat.setdefault("annualized_excess_return", m.get("excess_annual"))
            flat.setdefault("sharpe", m.get("sharpe"))
            flat.setdefault("max_drawdown", m.get("max_drawdown"))
            flat.setdefault("daily_overlap", m.get("daily_overlap"))
        return flat

    @staticmethod
    def entry_signature(expr: str) -> str:
        """与 record_tool_result 一致的表达式签名（规范换行后 sha256 前 20 位）。"""
        canonical = "\n".join(line.strip() for line in (expr or "").splitlines() if line.strip())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _classify(name: str, result: dict[str, Any], metrics: dict[str, Any], error: str) -> tuple[str, str]:
        """从工具结果重建 verdict 与 conclusion（纯规则，确定性）。

        入库事实优先于 Reviewer 意见：revise 不阻断提交（候选池照常收纳），
        已入库的 submit 即使携带 revise/gate 错误码也按 production_approved /
        candidate_approved 记账，Reviewer 意见并入结论文本——避免「进了候选池
        却记负证据」的口径错位（cells / 检索极性 / 父本质量排序都受影响）。
        """
        review = result.get("factor_review") if isinstance(result.get("factor_review"), dict) else {}
        if not review and isinstance(result.get("review"), dict):
            # submit payload 的审查意见存于 "review" 键（evaluate 工具用 "factor_review"）
            review = result.get("review")
        review_verdict = review.get("verdict")
        canonical = str(review.get("canonical_form") or "因子结构审核")
        reasons = review.get("reasons") if isinstance(review.get("reasons"), list) else []
        review_note = ""
        if review_verdict == "revise":
            first_reason = str(reasons[0]) if reasons else canonical
            review_note = f" Reviewer 意见：{canonical}（{first_reason}）。"
        if review_verdict == "reject":
            return "rejected", f"{canonical}：Reviewer 拒绝；{str(reasons[0]) if reasons else '不得重复同构表达式。'}"
        if name == "submit_factor":
            if result.get("stored"):
                return "production_approved", "已通过精筛并正式入库，应保留其经济机制并避免重复。" + review_note
            if result.get("candidate_stored"):
                return "candidate_approved", "通过海选进入候选池，尚未满足精筛条件，应针对失败项改进。" + review_note
        if review_verdict == "revise":
            return "revise_required", f"{canonical}：Reviewer 要求结构性改造后再评估。"
        if error:
            snippet = error if len(error) <= 500 else error[:497] + "..."
            return "rejected", f"{name} 被否定：{snippet}"
        if name == "submit_factor":
            return "rejected", "提交未通过，避免在未改变机制或拒绝原因的情况下重复提交。"
        ic = _safe_float(metrics.get("ic"))
        icir = _safe_float(metrics.get("icir"))
        coverage = _safe_float(metrics.get("factor_coverage", metrics.get("coverage")))
        is_val = name == "eval_on_val_set" or result.get("split") == "val"
        ic_str = f"IC={ic:+.4f}" if ic is not None else "IC=N/A"
        icir_str = f"ICIR={icir:+.3f}" if icir is not None else "ICIR=N/A"
        cov_str = f"Coverage={coverage:.2f}" if coverage is not None else ""
        if is_val and (result.get("sign_check", {}).get("matches_expected_sign") is not False) and abs(ic or 0) >= 0.015:
            return "validated", f"训练外验证通过：{ic_str} {icir_str} {cov_str}。方向一致且有可用相关性，可在相邻但不重复的机制上扩展。"
        # 海选线 2026-09-01 对齐 0.02（与 CandidateCriteria.min_abs_ic 同步）；
        # 阈值按档位区分（2026-09-05）：fundamental 档 0.015。
        th = 0.015 if str(metrics.get("research_mode")) == "fundamental" else 0.02
        if abs(ic or 0) >= th and (icir or 0) > 0.2 and (coverage or 0) > 0.85:
            return "promising", f"训练阶段有潜力：{ic_str} {icir_str} {cov_str}。优先进行训练外验证或独立性改造。"
        # P0-2 near_miss（2026-09-05）：IC 达门槛 80%、ICIR/coverage 达标但未过线——
        # 不再直接记 weak 死档，给"窗口微调/推 val"的二次机会（记忆分析：technical
        # 档 239 个 near-miss 无一获得二次评估）。
        if abs(ic or 0) >= 0.8 * th and (icir or 0) > 0.2 and (coverage or 0) > 0.85:
            return "near_miss", (
                f"接近海选线：{ic_str} {icir_str} {cov_str}（IC 距 {th} 门槛 <20%）。"
                "建议窗口微调后重评（传 parent_factor/edit_note），或机制置信度高时直接 eval_on_val_set。"
            )
        return "weak", f"指标不足：{ic_str} {icir_str} {cov_str}。除非改变变量、经济机制或处理方式，否则不要机械重试。"

    @classmethod
    def _classify_family(cls, factor_name: str, expression: str) -> str:
        """根据因子名和表达式启发式分类到信号族（规则式）。"""
        from .expressions import classify_family as _cf
        return _cf(factor_name, expression)
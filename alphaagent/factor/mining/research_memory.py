"""Persistent, evidence-backed memory for AlphaAgent factor research."""

from __future__ import annotations

import hashlib
import json
import math
import re
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

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        entries = raw.get("entries", []) if isinstance(raw, dict) else []
        return [item for item in entries if isinstance(item, dict)]

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps({"version": 1, "updated_at": _now(), "entries": entries[-1000:]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)

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
        entries = self._load()
        previous = next((item for item in entries if item.get("id") == signature), None)
        observation = {
            "run_id": run_id,
            "at": _now(),
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "verdict": verdict,
            "failure_code": failure_code,
            "metrics": self._compact_metrics(metrics),
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
            "attempts": int(previous.get("attempts", 0)) + 1 if 'previous' in locals() and previous is not None else 1,
            "updated_at": _now(),
            "observations": [observation],
        }
        entries = self._load()
        previous = next((item for item in entries if item.get("id") == signature), None)
        if previous is not None:
            old_observations = previous.get("observations", [])
            if isinstance(old_observations, int):
                old_observations = [{"at": previous.get("updated_at"), "verdict": previous.get("verdict")}]
            entry["observations"] = [*old_observations[-99:], observation]
            entry["created_at"] = previous.get("created_at", entry["updated_at"])
            entries = [item for item in entries if item.get("id") != signature]
        else:
            entry["created_at"] = entry["updated_at"]
        entries.append(entry)
        self._save(entries)
        return entry

    @staticmethod
    def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        keys = (
            "ic", "icir", "rank_ic", "factor_coverage", "coverage",
            "long_group_annual_excess_return", "winsorized_abs_ic_decay",
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

    def context_for(
        self,
        research_goal: str,
        *,
        limit: int = 12,
        include_rejected: bool = True,
        prefer_orthogonal: bool = True,
        include_expression: bool = True,
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
        entries = self._load()
        if not include_rejected:
            entries = [entry for entry in entries if entry.get("verdict") not in self._NEGATIVE_VERDICTS]
        if not entries:
            return ""
        query_tokens = set(_tokens(research_goal))

        verdict_rank = {
            "production_approved": 5, "validated": 4, "candidate_approved": 3,
            "promising": 2, "revise_required": 1, "rejected": 1, "weak": 0,
        }

        bm25 = self._bm25_scores(entries, query_tokens)

        def _key(entry: dict[str, Any], bm: float) -> tuple[float, int, int, str]:
            observations = entry.get("observations", [])
            n = observations if isinstance(observations, int) else len(observations)
            return (bm, verdict_rank.get(str(entry.get("verdict")), 0), n, str(entry.get("updated_at", "")))

        # Split into positive / negative pools and score independently so
        # that a large negative pool can't crowd out all positives.
        positive_pool = [(e, b) for e, b in zip(entries, bm25) if e.get("verdict") in self._POSITIVE_VERDICTS]
        negative_pool = [(e, b) for e, b in zip(entries, bm25) if e.get("verdict") in self._NEGATIVE_VERDICTS]

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
                lines.append(self._format_entry(entry, include_expression))

        # ── 否定段 ──
        if negative:
            lines.append("")
            lines.append("## 已否定 / 不足的因子（避免机械重复同一死路）")
            lines.append(
                "以下路径已被评估否定。除非改变了核心变量、经济机制或处理方式，"
                "否则不要重复尝试相同结构。"
            )
            for entry, _ in negative:
                lines.append(self._format_entry(entry, include_expression))

        return "\n".join(lines)

    @staticmethod
    def _format_entry(entry: dict[str, Any], include_expression: bool) -> str:
        m = entry.get("metrics", {})
        metric_text = " ".join(
            f"{key}={value:.4g}" for key, value in m.items() if isinstance(value, (int, float))
        )
        expr = entry.get("expression")
        expr_tail = f"；表达式：{expr}" if (include_expression and expr) else ""
        return (
            f"- [{entry.get('verdict')}] {entry.get('factor_name')}: {entry.get('conclusion')} "
            f"指标({metric_text or '无'}){expr_tail}"
        )

    @staticmethod
    def _entry_token_set(entry: dict[str, Any]) -> set[str]:
        """Return the token set for an entry, re-splitting compound tokens on
        the fly so that legacy entries written before underscore-splitting was
        added still match sub-word queries (e.g. ``reversal_5`` → ``reversal``).
        """
        stored = set(entry.get("tokens", []))
        if not stored:
            # Fallback: re-tokenise from factor_name + expression + conclusion
            stored = set(_tokens(
                entry.get("factor_name", ""),
                entry.get("expression", ""),
                entry.get("conclusion", ""),
            ))
        # Ensure underscore-split sub-words are present
        extra: set[str] = set()
        for token in stored:
            parts = token.split("_")
            if len(parts) > 1:
                extra.update(p for p in parts if len(p) >= 2)
        return stored | extra

    @staticmethod
    def _bm25_scores(entries: list[dict[str, Any]], query_tokens: set[str]) -> list[float]:
        """BM25 IDF/term-frequency scoring over entry token sets.

        k1=1.5, b=0.75. tf is 1 for present tokens (we store deduplicated token
        sets). Without query overlap a small negative floor avoids zero scores
        dominating ordering; priority still acts as the tie-breaker.
        """
        if not query_tokens:
            return [0.0] * len(entries)
        docs = [ResearchMemoryStore._entry_token_set(entry) for entry in entries]
        n = len(docs)
        if n == 0:
            return []
        from collections import Counter
        df = Counter()
        for doc in docs:
            df.update(doc)
        avgdl = sum(len(doc) for doc in docs) / n
        k1, b = 1.5, 0.75
        out = []
        for doc in docs:
            dl = len(doc)
            score = 0.0
            for q in query_tokens:
                if q not in doc:
                    continue
                idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
                tf = 1.0
                norm = k1 * (1 - b + b * dl / avgdl) if avgdl else 1.0
                score += idf * (tf * (k1 + 1)) / (tf + norm)
            out.append(score if score > 0 else -1.0)
        return out

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
        entries = self._load()
        # Two-pass sort: first by updated_at descending, then stable sort by
        # verdict priority ascending — Python's sort is stable, so entries
        # with the same verdict rank keep their recency order.
        entries.sort(key=lambda e: str(e.get("updated_at", "")), reverse=True)
        entries.sort(key=lambda e: self._VERDICT_ORDER.get(str(e.get("verdict")), 99))
        return entries[:limit]

    def statistics(self) -> dict[str, Any]:
        entries = self._load()
        counts: dict[str, int] = {}
        attempts = 0
        for entry in entries:
            verdict = str(entry.get("verdict") or "unknown")
            observations = entry.get("observations", [])
            n = observations if isinstance(observations, int) else len(observations)
            attempts += n
            counts[verdict] = counts.get(verdict, 0) + n
        return {
            "entries": len(entries),
            "observations": attempts,
            "verdict_counts": counts,
            "train_to_validated_rate": round(counts.get("validated", 0) / attempts, 4) if attempts else None,
            "production_rate": round(counts.get("production_approved", 0) / attempts, 4) if attempts else None,
        }

    def backfill_from_logs(self, log_root: Path) -> int:
        """Populate an empty memory store from prior UI JSONL event logs once."""
        if self._load():
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
        return count

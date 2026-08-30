# -*- coding: utf-8 -*-
"""混合检索与提示上下文构建。"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime
from typing import Any

from .calibration import _apv_gate, _eq7_confidence
from .constants import NEGATIVE_VERDICTS, POSITIVE_VERDICTS, VERDICT_WEIGHT
from .expressions import (
    _structure_fingerprint,
    _tokens,
    classify_family,
    expression_features,
    expression_ops,
)


class RetrievalMixin:
    """混合检索与提示上下文构建（v3 四层注入：经验 → 编辑先验 → 证据 → 多样性）。"""

    # ── FTS / 候选 ──

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
            f"AND e.verdict NOT IN ({','.join('?' for _ in NEGATIVE_VERDICTS)})"
        )
        filter_args = tuple() if include_rejected else tuple(sorted(NEGATIVE_VERDICTS))
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

    # ── 条目格式化 ──

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
        """将 memory_entries 行转为检索条目 dict（兼容两种列布局）。"""
        family = row["family"] if "family" in row.keys() else None
        metrics = json.loads(row["metrics_json"] or "{}")
        if "attempts" in row.keys():
            return {
                "id": row["id"],
                "factor_name": row["factor_name"],
                "expression": row["expression"],
                "conclusion": row["conclusion"],
                "verdict": row["verdict"],
                "stage": row["stage"],
                "metrics": metrics,
                "failure_code": row["failure_code"] if "failure_code" in row.keys() else None,
                "fail_detail": row["fail_detail"] if "fail_detail" in row.keys() else None,
                "family": family or classify_family(row["factor_name"], row["expression"]),
                "updated_at": row["updated_at"],
                "attempts": row["attempts"],
            }
        # 兼容退化布局：列错位
        return {
            "id": None,
            "factor_name": row["id"],
            "expression": row["factor_name"],
            "conclusion": row["expression"],
            "verdict": row["conclusion"],
            "stage": row["verdict"],
            "metrics": json.loads("{}"),
            "failure_code": json.loads("{}"),
            "fail_detail": row["failure_code"] if "failure_code" in row.keys() else None,
            "family": row["fail_detail"] if "fail_detail" in row.keys() else None,
            "updated_at": classify_family(row["factor_name"], row["expression"]),
            "attempts": row["updated_at"],
        }

    @staticmethod
    def _hybrid_score(
        entry: dict[str, Any],
        *,
        query_ops: set[str],
        focus_families: set[str],
        now_ts: float,
    ) -> float:
        """BM25 + 族亲和 + 算子 Jaccard + verdict 权重 + 时间衰减。"""
        bm25 = float(entry.get("_bm25", 0))
        family_affinity = 0.3 if entry.get("family") in focus_families else 0
        entry_ops = set(expression_ops(str(entry.get("expression") or "")))
        op_jaccard = len(query_ops & entry_ops) / len(query_ops | entry_ops) if (query_ops | entry_ops) else 0
        verdict_w = VERDICT_WEIGHT.get(str(entry.get("verdict")), 0.2)
        try:
            age_days = max(0, (now_ts - datetime.fromisoformat(str(entry.get("updated_at"))).timestamp()) / 86400)
            recency = math.exp(-age_days / 90)
        except (ValueError, TypeError):
            recency = math.exp(-30 / 90)
        return bm25 + family_affinity + 0.2 * op_jaccard + 0.3 * verdict_w + 0.2 * recency

    @staticmethod
    def _select_diverse(
        scored: list[tuple[float, dict[str, Any]]],
        *,
        limit: int,
        max_per_family: int = 2,
    ) -> list[tuple[float, dict[str, Any]]]:
        """(family ≤2, 结构指纹 =1) 双重去重 + 贪心选择。"""
        family_count: dict[str, int] = {}
        seen_fp: set[str] = set()
        out: list[tuple[float, dict[str, Any]]] = []
        for score, entry in scored:
            family = str(entry.get("family") or "other")
            fp = _structure_fingerprint(str(entry.get("expression") or ""))
            if fp and fp in seen_fp:
                continue
            if family_count.get(family, 0) >= max_per_family:
                continue
            if fp:
                seen_fp.add(fp)
            family_count[family] = family_count.get(family, 0) + 1
            out.append((score, entry))
            if len(out) >= limit:
                return out
        return out

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
        family = str(entry.get("family") or "other")
        fail = f"；失效: {entry['fail_detail']}" if entry.get("fail_detail") else ""
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
            f"指标({metric_text or '无'}){expr_tail}{fail}"
        )

    # ── 注入块 ──

    def _evidence_block(
        self,
        research_goal: str,
        *,
        limit: int = 8,
        include_rejected: bool = True,
        prefer_orthogonal: bool = True,
        include_expression: bool = True,
        max_expression_chars: int | None = None,
        focus_families: set[str] | None = None,
        query_ops: set[str] | None = None,
    ) -> str:
        """构建单因子证据块（正/负池独立排序 + 各自多样性去重 + 40% 正池配额）。"""
        entries = self._retrieval_candidates(research_goal, include_rejected)
        if not entries:
            return ""
        focus_families = focus_families or set()
        query_ops = query_ops or set()
        now_ts = time.time()
        scored = [
            (self._hybrid_score(e, query_ops=query_ops, focus_families=focus_families, now_ts=now_ts), e)
            for e in entries
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # 正负池分离，各自独立做多样性去重，避免指纹去重跨 verdict 误杀
        pos_pool = [(s, e) for s, e in scored if e.get("verdict") in POSITIVE_VERDICTS]
        neg_pool = [(s, e) for s, e in scored if e.get("verdict") in NEGATIVE_VERDICTS]

        pos_quota = max(1, int(limit * 0.4)) if pos_pool else 0
        pos_selected = self._select_diverse(pos_pool, limit=pos_quota, max_per_family=2)
        neg_quota = limit - len(pos_selected)
        neg_selected = self._select_diverse(neg_pool, limit=neg_quota, max_per_family=2) if neg_quota > 0 else []

        positive = pos_selected
        negative = neg_selected

        block_lines: list[str] = []
        if positive:
            block_lines.append("")
            block_lines.append("## 已验证 / 有潜力的因子（优先在其邻近空间继续挖掘相似机制）")
            block_lines.append(
                "这些因子在训练集或验证集上表现可用。**鼓励**基于它们的经济逻辑，"
                "通过更换窗口、算子族或原始字段，在相似但不重复的方向上继续探索。"
            )
            if prefer_orthogonal:
                block_lines.append("扩展时优先引入正交变量，避免仅改窗口长度的同质微调。")
            for _, entry in positive:
                block_lines.append(self._format_entry(entry, include_expression, max_expression_chars))

        if negative:
            block_lines.append("")
            block_lines.append("## 已否定 / 不足的因子（避免机械重复同一死路）")
            block_lines.append(
                "以下路径已被评估否定。除非改变了核心变量、经济机制或处理方式，"
                "否则不要重复尝试相同结构。"
            )
            for _, entry in negative:
                block_lines.append(self._format_entry(entry, include_expression, max_expression_chars))

        return "\n".join(block_lines)

    def _experience_block(self) -> str:
        """经验块：成功模式 / 禁忌方向 / 战略洞察（按 occurrence 排序）。"""
        try:
            with self._open() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM memory_experience
                    WHERE kind IN ('success_pattern', 'forbidden', 'insight')
                    ORDER BY occurrence_count DESC, updated_at DESC
                    """
                ).fetchall()
        except sqlite3.Error:
            return ""
        if not rows:
            return ""
        block_lines: list[str] = []
        block_lines.append("")
        block_lines.append("## 经验记忆（跨因子成功模式 / 禁忌方向 / 战略洞察）")
        for row in rows:
            kind = str(row["kind"])
            content = str(row["content"] or "")
            if kind == "success_pattern":
                block_lines.append(f"- 成功模式：{content}")
            elif kind == "forbidden":
                block_lines.append(f"- 禁忌方向：{content}")
            else:
                block_lines.append(f"- 洞察：{content}")
        return "\n".join(block_lines)

    def _edit_prior_block(self, focus_families: set[str] | None = None) -> str:
        """AlphaMemo 单元注入（v3 的门控残差记忆）。

        硬推荐 c>0.7 且 s>0 / 软推荐 0.4<c<0.7 且 s>0 / APV 双向硬否决 /
        无显式胜率不注入；推荐封顶不超 "优先尝试"（positive_boost_cap 语义）。
        """
        focus_families = focus_families or set()
        try:
            with self._open() as conn:
                rows = conn.execute("SELECT * FROM memory_cells").fetchall()
        except sqlite3.Error:
            return ""
        if not rows:
            return ""
        lines: list[str] = []
        lines.append("")
        lines.append("## 编辑统计记忆（基于 cells 的显式/隐式编辑观测）")
        for row in rows:
            family = str(row["family"] or "other")
            motif = str(row["motif"] or "other")
            bucket = str(row["parent_bucket"] or "low")
            s_w, f_w = self._weighted_counts(row)
            residuals = json.loads(row["residuals_json"] or "[]")
            conf = _eq7_confidence(residuals) if residuals else 0.0
            verdict_mark = ""
            if s_w > 0 and conf > 0.7:
                verdict_mark = "硬推荐"
            elif s_w > 0 and conf > 0.4:
                verdict_mark = "软推荐"
            elif f_w > 0 and conf > 0.7:
                verdict_mark = "硬否决"
            elif f_w > 0 and conf > 0.4:
                verdict_mark = "软否决"
            if verdict_mark:
                lines.append(
                    f"- [{verdict_mark}] {family}/{motif}（桶 {bucket}）："
                    f"成功 {s_w:.2g} / 失败 {f_w:.2g}，置信 {conf:.0%}"
                )
        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _diversity_block(self, recent_batch: list[dict[str, Any]] | None = None) -> str:
        """多样性块：词法集中 + 重复空格 + 族新频度（占位实现）。"""
        return ""

    # ── 主入口 ──

    def context_for(
        self,
        research_goal: str,
        *,
        limit: int = 12,
        include_rejected: bool = True,
        prefer_orthogonal: bool = True,
        include_expression: bool = True,
        max_expression_chars: int | None = None,
        enable_factor_retrieval: bool = False,
        enable_edit_patterns: bool = False,
        recent_batch: list[dict[str, Any]] | None = None,
        max_inject_chars: int | None = None,
    ) -> str:
        """构建注入上下文。顺序：经验 → 编辑先验 → 证据 → 多样性。

        v3 预算：超过 max_inject_chars 时按优先级截断（编辑统计 > 经验 > 多样性 > 证据），
        先整体注入再按预算截断，显示顺序不变；防止上下文被低信息量条目挤占。
        """
        max_inject_chars = self.max_inject_chars if max_inject_chars is None else int(max_inject_chars)
        focus_families: set[str] = set()
        query_ops: set[str] = set()
        if research_goal:
            for expr in re.split(r"[\n;]", str(research_goal)):
                if expression_features(expr):
                    fam = classify_family("", expr)
                    if fam != "other":
                        focus_families.add(fam)
                query_ops |= set(expression_ops(expr))

        lines: list[str] = []
        header_added = False

        def _ensure_header() -> None:
            nonlocal header_added
            if not header_added:
                lines.append("# 长期研究记忆")
                lines.append("以下结论必须作为实验先验。")
                header_added = True

        # ① 经验块（跨因子，最高优先级）
        exp_block = self._experience_block()
        if exp_block:
            _ensure_header()
            lines.append(exp_block)

        # ② 编辑先验块（cells 统计）
        if enable_edit_patterns:
            edit_block = self._edit_prior_block(focus_families)
            if edit_block:
                _ensure_header()
                lines.append(edit_block)

        # ③ 多样性块（占位）
        div_block = self._diversity_block(recent_batch)
        if div_block:
            _ensure_header()
            lines.append(div_block)

        # ④ 证据块（单因子检索，可选）
        if enable_factor_retrieval:
            factor_block = self._evidence_block(
                research_goal,
                limit=limit,
                include_rejected=include_rejected,
                prefer_orthogonal=prefer_orthogonal,
                include_expression=include_expression,
                max_expression_chars=max_expression_chars,
                focus_families=focus_families,
                query_ops=query_ops,
            )
            if factor_block:
                _ensure_header()
                lines.append(factor_block)

        text = "\n".join(lines)
        if max_inject_chars > 0 and len(text) > max_inject_chars:
            text = text[:max_inject_chars].rsplit("\n", 1)[0]
        return text
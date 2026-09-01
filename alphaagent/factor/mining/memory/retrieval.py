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

# 注入文本自解释化：信号族 / 编辑类型的中文展开（缺省回退原文）
FAMILY_LABELS = {
    "volume": "量价/成交量",
    "volatility": "波动率",
    "momentum": "动量",
    "reversal": "反转",
    "vwap": "VWAP 偏离",
    "gap_overnight": "隔夜跳空",
    "chip": "筹码分布",
    "liquidity": "流动性",
    "fundamental": "基本面",
    "correlation": "量价相关",
    "breadth": "市场宽度",
    "sentiment": "情绪",
    "other": "其他",
}
MOTIF_LABELS = {
    "window_rescale": "调整窗口参数",
    "feature_swap": "替换输入字段",
    "operator_substitute": "替换核心算子",
    "operator_swap": "替换核心算子",
    "condition_gate": "加条件门控",
    "composition_add": "叠加外层修饰",
    "normalize_change": "更换标准化方式",
    "decorrelation_add": "加去相关处理",
    "interaction_add": "新增交互/乘除项",
    "compound_edit": "复合编辑（同时改动多处）",
    "normalization_change": "更换标准化方式",
    "other": "其他改动",
}


class RetrievalMixin:
    """混合检索与提示上下文构建（v3 六层注入：经验 → 模式 → 编辑先验 → 饱和度 → 多样性 → 证据）。"""

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

    def _guaranteed_positives(self, k: int) -> list[dict[str, Any]]:
        """跨族保底正向父本：全库正向条目按（验证档位, |IC|）全局排序，同族轮转取最优。

        与 BM25 命中无关 —— 通用 query 打不中 FTS 时正向通道也不断供，
        保证每轮证据块里一定有"值得作为父本"的正样本（2026-09-01）。
        """
        if k <= 0:
            return []
        tier = {
            "production_approved": 0, "validated": 1,
            "candidate_approved": 2, "promising": 3,
        }
        try:
            with self._open() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_entries WHERE verdict IN "
                    "('production_approved','validated','candidate_approved','promising')"
                ).fetchall()
        except sqlite3.Error:
            return []
        ranked: list[tuple[tuple[int, float], dict[str, Any], str]] = []
        for row in rows:
            e = self._row_to_entry(row)
            m = e.get("metrics") if isinstance(e.get("metrics"), dict) else {}
            ic = m.get("ic") if m else None
            abs_ic = abs(float(ic)) if isinstance(ic, (int, float)) else 0.0
            fam = str(e.get("family") or "other")
            ranked.append(((tier.get(str(e.get("verdict")), 9), -abs_ic), e, fam))
        ranked.sort(key=lambda t: t[0])
        # 家族按其最优条目的全局排名排序；同族轮转（先各家一条，再补第二条）
        fam_order: list[str] = []
        for _, _, fam in ranked:
            if fam not in fam_order:
                fam_order.append(fam)
        by_fam: dict[str, list[dict[str, Any]]] = {f: [] for f in fam_order}
        for _, e, fam in ranked:
            by_fam[fam].append(e)
        out: list[dict[str, Any]] = []
        while len(out) < k:
            progressed = False
            for f in fam_order:
                if by_fam[f] and len(out) < k:
                    out.append(by_fam[f].pop(0))
                    progressed = True
            if not progressed:
                break
        return out

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

        pos_quota = max(1, int(limit * 0.4))
        # 正向保底（2026-09-01）：跨族最优正向因子无条件入选约 2/3 正向配额，
        # BM25 正池只补充剩余 —— 通用 query 打不中 FTS 时正向通道不断供。
        guaranteed = self._guaranteed_positives(max(0, pos_quota - 1))
        g_names = {e.get("factor_name") for e in guaranteed}
        pos_pool_rest = [(s, e) for s, e in pos_pool
                         if e.get("factor_name") not in g_names]
        bm25_pos = (self._select_diverse(pos_pool_rest, limit=pos_quota - len(guaranteed),
                                         max_per_family=2)
                    if pos_quota > len(guaranteed) else [])
        pos_selected = [(0.0, e) for e in guaranteed] + bm25_pos
        neg_quota = limit - len(pos_selected)
        neg_selected = self._select_diverse(neg_pool, limit=neg_quota, max_per_family=2) if neg_quota > 0 else []

        positive = pos_selected
        negative = neg_selected

        block_lines: list[str] = []
        if positive:
            block_lines.append("")
            block_lines.append("## 已验证 / 有潜力的因子（优先在其邻近空间继续挖掘相似机制）")
            block_lines.append(
                "这些因子在训练集或验证集上表现可用，排在最前的是全库最强正向因子"
                "（跨族保底，与检索无关）。**鼓励**基于它们的经济逻辑，"
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
        """经验块：FactorMiner 式可执行条目。

        success_pattern = 机制说明 + 参数槽模板 + 真实示例表达式（可照抄骨架）；
        forbidden = DO NOT 禁令（模板 + 死路/被拒因子名单）；insight = 战略洞察。
        """
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
        block_lines.append("## 经验记忆（成功模式可照抄模板变异；DO NOT 行本轮不得违反）")
        success_rows = [r for r in rows if str(r["kind"]) == "success_pattern"]
        forbidden_rows = [r for r in rows if str(r["kind"]) == "forbidden"]
        insight_rows = [r for r in rows if str(r["kind"]) == "insight"]
        for row in forbidden_rows:
            content = str(row["content"] or "")
            block_lines.append(f"- 【禁止】{content}")
        for row in success_rows:
            content = str(row["content"] or "")
            template = str(row["template"] or "") if "template" in row.keys() else ""
            try:
                evidence = json.loads(row["evidence_json"] or "{}")
            except (TypeError, ValueError):
                evidence = {}
            examples = [e for e in (evidence.get("examples") or []) if e]
            line = f"- 成功模式：{content}"
            if template:
                line += f"\n  模板: `{template}`"
            for ex in examples[:2]:
                line += f"\n  示例: {ex}"
            block_lines.append(line)
        for row in insight_rows:
            block_lines.append(f"- 洞察：{row['content']}")
        return "\n".join(block_lines)

    # ── 记忆出题（AlphaMemo 口径：记忆参与搜索决策）──

    def recommend_edits(self, k: int = 2) -> list[dict[str, Any]]:
        """按 cells 残差×置信度给 (family, motif) 出题，附族内最优父本。

        与 _edit_prior_block 的区别：那边是"阅卷"（事后统计表），这里是
        "出题"——每轮前 k 个评估名额直接由记忆推荐（父本 × 编辑类型），
        run 循环把它们写进任务消息。排序键 = 平均残差 × Eq.7 置信度，
        APV 双门否决的 (family, motif) 不推荐；无正 cells 时回退
        「出过正信号且未拥挤族 + 族内最优父本」的族级推荐。
        """
        if k <= 0:
            return []
        try:
            with self._open() as conn:
                cell_rows = conn.execute("SELECT * FROM memory_cells").fetchall()
        except sqlite3.Error:
            return []
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for row in cell_rows:
            family = str(row["family"] or "other")
            motif = str(row["motif"] or "other")
            s_w, f_w = self._weighted_counts(row)
            try:
                residuals = [float(v) for v in json.loads(row["residuals_json"] or "[]")]
            except (TypeError, ValueError):
                residuals = []
            cell = agg.setdefault(
                (family, motif),
                {"s_w": 0.0, "f_w": 0.0, "residuals": []},
            )
            cell["s_w"] += s_w
            cell["f_w"] += f_w
            cell["residuals"].extend(residuals)

        scored: list[dict[str, Any]] = []
        for (family, motif), cell in agg.items():
            if not cell["residuals"]:
                continue
            conf = _eq7_confidence(cell["residuals"])
            delta = sum(cell["residuals"]) / len(cell["residuals"])
            vetoed, _, _ = _apv_gate(cell["s_w"], cell["f_w"], conf)
            if vetoed or delta <= 0 or conf < 0.3:
                continue
            scored.append({
                "family": family, "motif": motif,
                "delta": delta, "conf": conf,
                "score": delta * conf, "n": len(cell["residuals"]),
            })
        scored.sort(key=lambda c: -c["score"])

        picks: list[dict[str, Any]] = []
        seen_families: set[str] = set()
        for cand in scored:
            if cand["family"] in seen_families:
                continue
            parent = self._best_family_parent(cand["family"])
            if parent is None:
                continue
            picks.append({
                **cand,
                **parent,
                "family_cn": FAMILY_LABELS.get(cand["family"], cand["family"]),
                "motif_cn": MOTIF_LABELS.get(cand["motif"], cand["motif"]),
                "reason": (
                    f"历史 {cand['n']} 次该编辑平均残差 {cand['delta']:+.4f}、"
                    f"置信 {cand['conf']:.0%}"
                ),
            })
            seen_families.add(cand["family"])
            if len(picks) >= k:
                return picks

        # 回退：无可用正 cells → 出过正信号且未拥挤的族级推荐
        saturation = self.compute_saturation()
        open_fams = sorted(
            (
                (f, d) for f, d in saturation.items()
                if d.get("saturation_score", 0) <= 0.6 and d.get("n_promising", 0) > 0
            ),
            key=lambda x: (-x[1].get("n_promising", 0), x[1].get("saturation_score", 0)),
        )
        for family, data in open_fams:
            if family in seen_families:
                continue
            parent = self._best_family_parent(family)
            if parent is None:
                continue
            picks.append({
                "family": family, "motif": None,
                "family_cn": FAMILY_LABELS.get(family, family),
                "motif_cn": None,
                "n": int(data.get("n_promising", 0)),
                **parent,
                "reason": (
                    f"该族出过 {int(data.get('n_promising', 0))} 个正信号且未拥挤"
                    f"（饱和 {data.get('saturation_score', 0):.0%}）"
                ),
            })
            seen_families.add(family)
            if len(picks) >= k:
                break
        return picks

    def _best_family_parent(self, family: str) -> dict[str, Any] | None:
        """族内最优正向父本（验证档位优先，其次 |IC|）。"""
        tier = {"production_approved": 0, "validated": 1,
                "candidate_approved": 2, "promising": 3}
        try:
            with self._open() as conn:
                rows = conn.execute(
                    "SELECT factor_name, expression, verdict, metrics_json "
                    "FROM memory_entries WHERE family = ? AND verdict IN "
                    "('production_approved','validated','candidate_approved','promising')",
                    (family,),
                ).fetchall()
        except sqlite3.Error:
            return None
        best: tuple[tuple[int, float], str, str, float] | None = None
        for row in rows:
            try:
                m = json.loads(row["metrics_json"] or "{}")
                ic = float(m.get("ic") or 0.0)
            except (TypeError, ValueError):
                ic = 0.0
            key = (tier.get(str(row["verdict"]), 9), -abs(ic))
            if best is None or key < best[0]:
                best = (key, str(row["factor_name"] or ""),
                        str(row["expression"] or ""), ic)
        if best is None or not best[2]:
            return None
        return {
            "parent_factor": best[1],
            "parent_expression": best[2],
            "parent_ic": best[3],
        }

    def _edit_prior_block(self, focus_families: set[str] | None = None) -> str:
        """AlphaMemo 单元注入（v3 的门控残差记忆）。

        硬推荐 c>edit_prior_hard_conf 且 s>0 / 软推荐 c>edit_prior_recommend_conf 且 s>0 /
        硬否决 c>edit_prior_hard_conf 且 f>0 / 软否决 c>edit_prior_veto_conf 且 f>0 /
        无显式胜率不注入；推荐封顶不超 "优先尝试"（positive_boost_cap 语义）。
        阈值来自 memory_policy（edit_prior_*_conf），否决向默认放宽到 0.3 以放行避坑证据。
        2026-08-30 起行文本自解释化：中文展开族/编辑类型/父本桶，并给出行动指令，
        保证 LLM 无需约定术语即可理解与执行。
        """
        focus_families = focus_families or set()
        try:
            with self._open() as conn:
                rows = conn.execute("SELECT * FROM memory_cells").fetchall()
        except sqlite3.Error:
            return ""
        if not rows:
            return ""
        hard_conf = getattr(self, "edit_prior_hard_conf", 0.7)
        recommend_conf = getattr(self, "edit_prior_recommend_conf", 0.4)
        veto_conf = getattr(self, "edit_prior_veto_conf", 0.3)

        bucket_text = {
            "low": "弱父本",
            "medium": "中等父本",
            "high": "强父本",
        }
        # 档位 → (短标签, 行内行动指令)；档位语义在头部图例统一解释一次
        tier_text = {
            "hard_recommend": ("优先采用", "可作为优先的变异方向"),
            "soft_recommend": ("优先尝试", "候选方向接近时可优先选它"),
            "hard_veto": ("禁止", "本轮不得对该族因子使用此编辑"),
            "soft_veto": ("谨慎避开", "改用其他编辑类型或换族"),
        }

        lines: list[str] = []
        lines.append("")
        lines.append("## 编辑方向先验（供 A/B/C 轨选择编辑类型时参考）")
        lines.append(
            "每行 = 一个历史场景「信号族 × 编辑类型 × 父本质量桶」的成败统计，"
            "仅当该行场景与你正要做的变异匹配时才生效。"
        )
        lines.append(
            "档位含义：【禁止】= 一致失败且结论可靠，不得使用；【谨慎避开】= 失败为主，"
            "按行尾指令处理；【优先尝试/优先采用】= 历史偏成功，可优先选。"
            "父本桶 = 统计所基于的父本强弱（弱: |IC|<0.015，中: 0.015~0.025，强: ≥0.025），"
            "在弱父本上变异只看弱父本行。"
        )
        focus_lines: list[str] = []
        other_lines: list[str] = []
        for row in rows:
            family = str(row["family"] or "other")
            motif = str(row["motif"] or "other")
            bucket = str(row["parent_bucket"] or "low")
            s_w, f_w = self._weighted_counts(row)
            residuals = json.loads(row["residuals_json"] or "[]")
            conf = _eq7_confidence(residuals) if residuals else 0.0
            tier = ""
            if s_w > 0 and conf > hard_conf:
                tier = "hard_recommend"
            elif s_w > 0 and conf > recommend_conf:
                tier = "soft_recommend"
            elif f_w > 0 and conf > hard_conf:
                tier = "hard_veto"
            elif f_w > 0 and conf > veto_conf:
                tier = "soft_veto"
            if not tier:
                continue
            fam_cn = FAMILY_LABELS.get(family, family)
            motif_cn = MOTIF_LABELS.get(motif, motif)
            tag, action = tier_text[tier]
            line = (
                f"- 【{tag}】{fam_cn}类 × 「{motif_cn}」× {bucket_text.get(bucket, bucket)}："
                f"成功 {s_w:.2g} / 失败 {f_w:.2g}（失败率 {f_w / (s_w + f_w):.0%}），置信 {conf:.0%}"
                f" → {action}。"
            )
            (focus_lines if family in focus_families else other_lines).append(line)
        lines.extend(focus_lines)
        lines.extend(other_lines)
        if len(lines) <= 3:
            return ""
        return "\n".join(lines)

    # ── 模式层注入 ──

    def _pattern_block(self) -> str:
        """模式层注入：跨因子经验提炼（recommend / forbid / insight 三段）。"""
        patterns = self.query_patterns(min_confidence=0.3, limit=10)
        if not patterns:
            return ""
        block_lines: list[str] = []
        block_lines.append("")
        block_lines.append("## 研究模式记忆（跨因子经验提炼）")
        block_lines.append("以下模式来自历史多轮挖掘的统计提炼，优先级高于单因子记忆。")

        recommends = [p for p in patterns if p["layer"] == "recommend" and (p.get("success_rate") or 0) >= 0]
        forbids = [p for p in patterns if p["layer"] == "forbid"]
        insights = [p for p in patterns if p["layer"] == "insight"]

        if recommends:
            block_lines.append("")
            block_lines.append("### 推荐方向（成功率 > 0，在其邻近空间继续探索）")
            for p in recommends:
                rate = p.get("success_rate") or 0
                conf = p.get("confidence") or 0
                block_lines.append(
                    f"- [{p['category']}] {p['content']} "
                    f"(成功率 {rate:.0%}，置信度 {conf:.0%})"
                )

        if forbids:
            block_lines.append("")
            block_lines.append("### 禁止方向（已验证无效，除非改变核心机制否则不要重复）")
            for p in forbids:
                conf = p.get("confidence") or 0
                n = p.get("total_attempts") or 0
                block_lines.append(
                    f"- [{p['category']}] {p['content']} "
                    f"(已尝试 {n} 次，置信 {conf:.0%})"
                )

        if insights:
            block_lines.append("")
            block_lines.append("### 战略洞察")
            for p in insights:
                block_lines.append(f"- {p['content']}")

        return "\n".join(block_lines)

    # ── 饱和度层 ──

    def compute_saturation(self) -> dict[str, dict[str, float]]:
        """计算各因子族的饱和度。

        saturation_score = min(1.0, min(n_promising, 8) / 40 + n_candidate / 5
                                        + n_validated / 3)
        > 0.6 时标记为"拥挤"，建议切换方向。

        2026-09-01 修正：promising（仅过训练集海选，多数未通过验证/晋升）
        不再与 candidate 同权计入拥挤 —— 出过正信号 ≠ 已入库拥挤。
        promising 只轻度计入（上限 8 个 → 贡献 ≤0.2），拥挤度以真实幸存者
        （candidate / validated / production）为主，避免把整个出信号的族封禁。
        """
        families: dict[str, dict[str, float]] = {}
        with self._open() as conn:
            rows = conn.execute(
                "SELECT factor_name, expression, verdict FROM memory_entries"
            ).fetchall()

        for row in rows:
            family = classify_family(row["factor_name"], row["expression"])
            fam = families.setdefault(
                family, {"n_entries": 0, "n_promising": 0, "n_candidate": 0,
                         "n_validated": 0}
            )
            fam["n_entries"] += 1
            if row["verdict"] == "promising":
                fam["n_promising"] += 1
            elif row["verdict"] == "candidate_approved":
                fam["n_candidate"] += 1
            elif row["verdict"] in ("validated", "production_approved"):
                fam["n_validated"] += 1

        for fam_data in families.values():
            fam_data["saturation_score"] = round(min(
                1.0,
                min(fam_data["n_promising"], 8) / 40.0
                + fam_data["n_candidate"] / 5.0
                + fam_data["n_validated"] / 3.0,
            ), 3)

        return families

    def _saturation_block(self) -> str:
        """饱和度层注入：拥挤族警告 + 出过正信号且未拥挤族的定向建议。"""
        saturation = self.compute_saturation()
        crowded = {
            f: d for f, d in saturation.items()
            if d.get("saturation_score", 0) > 0.6
        }
        if not crowded:
            return ""
        block_lines: list[str] = []
        block_lines.append("")
        block_lines.append("### 饱和度警告")
        block_lines.append("以下因子族已拥挤（相似因子已入库/多次验证），继续微调边际收益低：")
        for family, data in sorted(crowded.items(), key=lambda x: -x[1].get("saturation_score", 0)):
            block_lines.append(
                f"- {family}: {int(data['n_promising'])} 个有潜力 + "
                f"{int(data['n_validated'])} 个已验证，"
                f"饱和度 {data['saturation_score']:.0%}"
            )
        open_fams = [
            (f, d) for f, d in saturation.items()
            if d.get("saturation_score", 0) <= 0.6 and d.get("n_promising", 0) > 0
        ]
        if open_fams:
            open_fams.sort(key=lambda x: (-x[1].get("n_promising", 0),
                                          x[1].get("saturation_score", 0)))
            sug = ", ".join(
                f"{f}（{int(d['n_promising'])} 个潜力, 饱和 {d['saturation_score']:.0%}）"
                for f, d in open_fams[:5]
            )
            block_lines.append(f"出过正信号且未拥挤的族（优先深挖）：{sug}。")
        else:
            block_lines.append("建议切换到饱和度 < 0.3 的未探索族。")
        return "\n".join(block_lines)

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
        """构建注入上下文。显示顺序：经验 → 编辑先验 → 饱和度 → 多样性 → 证据。

        v3 预算策略（2026-08-30 起）：核心块（经验、编辑先验）始终保留；
        次级块按 证据 > 饱和度 > 多样性 的优先级用剩余预算填充——证据块承载
        具体死路因子清单，不再被尾部一刀切截掉。所有截断均在行边界。
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

        def _clip(text: str, budget: int) -> str:
            """行边界截断到 budget 字符内。"""
            if budget <= 0 or len(text) <= budget:
                return text
            return text[:budget].rsplit("\n", 1)[0]

        # 核心块：无条件保留（超预算时行边界截断）
        core: list[str] = []
        exp_block = self._experience_block()
        if exp_block:
            core.append(exp_block)
        if enable_edit_patterns:
            edit_block = self._edit_prior_block(focus_families)
            if edit_block:
                core.append(edit_block)

        core_text = "\n\n".join(core)
        if max_inject_chars > 0:
            header_len = len("# 长期研究记忆\n以下结论必须作为实验先验。\n\n")
            core_text = _clip(core_text, max(0, max_inject_chars - header_len))
            remaining = max_inject_chars - len(core_text) - header_len
        else:
            remaining = 10 ** 9

        # 次级块：按优先级（证据 > 饱和度 > 多样性）填充剩余预算
        secondary: list[str] = []  # (priority, text)
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
                secondary.append((0, factor_block))
        sat_block = self._saturation_block()
        if sat_block:
            secondary.append((1, sat_block))
        div_block = self._diversity_block(recent_batch)
        if div_block:
            secondary.append((2, div_block))

        kept: list[str] = []
        for _, text in sorted(secondary, key=lambda p: p[0]):
            if len(text) <= remaining:
                kept.append(text)
                remaining -= len(text) + 2
            elif "## 已验证" in text or "## 已否定" in text:
                # 证据块超预算：截到剩余空间（行边界），保证在场
                clipped = _clip(text.strip(), remaining)
                if clipped.strip():
                    kept.append(clipped)
                    remaining = 0

        header = "# 长期研究记忆\n以下结论必须作为实验先验。"
        parts = [header] + ([core_text] if core_text else []) + kept
        return "\n\n".join(p for p in parts if p)
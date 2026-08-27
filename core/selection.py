"""组合候选筛选与目标权重构建。

该模块不读取行情、不执行订单，只把信号截面转换为目标持仓。

选股层以 :class:`SelectionPolicy` 为中心解耦：数量模式、绝对信号门控、
数量约束、行业分组上限、弱市降仓叠加全部收拢在策略对象里，
引擎主流程只调用 ``PortfolioBuilder.build_targets`` 一个入口。
新的选股类型优先扩展本模块，而不是往 ``run_backtest`` 加散装参数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionPolicy:
    """组合构建策略：描述"从因子截面到目标持仓"的全部规则。

    维度与流水线：
      1. 绝对门控   ``min_score``：质量分下限。质量分 = score（买大）或
                    -score（买小），即"越大越好"的方向归一化。
                    例：brk20 买大配 min_score=0 表示只买创20日新高的票；
                    vol20 买小配 min_score=0.05 表示只买波动率<=5%的票。
                    无票过线则空仓等待——排名制不凑数。
      2. 计数模式   ``count_mode``：top_n 固定只数 / top_pct 按候选池比例。
      3. 数量约束   ``min_positions`` / ``max_positions``。
      4. 行业分组   ``industry_cap``：每行业最多 N 只（依赖 builder 的 map）。
      5. 权重       当前等权；扩展点。
      6. 弱市叠加   ``regime_adx``/``regime_scale``：市场 ADX 中位数低于
                    阈值时目标权重乘以 regime_scale。
    """

    count_mode: str = "top_n"          # top_n | top_pct
    top_n: int = 3
    pct: float = 0.10
    min_positions: int = 1
    max_positions: int | None = None
    ascending: bool = False            # 门控方向语义需要
    min_score: float | None = None
    industry_cap: int | None = None
    regime_adx: float | None = None
    regime_scale: float = 0.5

    def quality(self, scores) -> np.ndarray:
        """方向归一化的质量分：越大越好（买小时取负）。"""
        s = np.asarray(scores, dtype=float)
        return -s if self.ascending else s

    def gate(self, candidates, scores) -> np.ndarray:
        """绝对阈值门控：保留质量分 >= min_score 的候选（不过线宁缺毋滥）。"""
        cand = np.asarray(candidates)
        if self.min_score is None or len(cand) == 0:
            return cand
        q = self.quality(scores)[:len(cand)]
        keep = np.isfinite(q) & (q >= float(self.min_score))
        return cand[keep]

    def scale_for_regime(self, market_adx) -> float:
        """弱势环境下的权重缩放系数（1.0 = 不减仓）。"""
        if (self.regime_adx is None or market_adx is None
                or not np.isfinite(market_adx)):
            return 1.0
        return float(self.regime_scale) if market_adx < self.regime_adx else 1.0


@dataclass
class PortfolioBuilder:
    codes: list[str]
    industry_map: dict[str, str] | None = None
    industry_cap: int | None = None

    def rank_select(
        self,
        candidates: np.ndarray,
        scores: np.ndarray,
        ascending: bool,
        top_n: int,
        selection_mode: str = "top_n",
        selection_pct: float = 0.10,
        min_positions: int = 1,
        max_positions: int | None = None,
        limit_count: int | None = None,
    ) -> list[int]:
        """按信号排序并应用行业约束，返回原始列索引。"""
        if len(candidates) == 0:
            return []
        order = np.argsort(scores, kind="mergesort")
        if not ascending:
            order = order[::-1]
        ordered = [int(candidates[o]) for o in order]

        if self.industry_cap and self.industry_map:
            selected: list[int] = []
            counts: dict[str, int] = {}
            for k in ordered:
                industry = self.industry_map.get(str(self.codes[k]), "?")
                if counts.get(industry, 0) >= self.industry_cap:
                    continue
                counts[industry] = counts.get(industry, 0) + 1
                selected.append(k)
                if limit_count is not None and len(selected) >= limit_count:
                    break
            ordered = selected

        count = limit_count if limit_count is not None else self.selection_count(
            len(ordered), top_n, selection_mode, selection_pct,
            min_positions, max_positions)
        return ordered[:count] if count > 0 else []

    @staticmethod
    def selection_count(
        candidate_count: int,
        top_n: int,
        selection_mode: str = "top_n",
        selection_pct: float = 0.10,
        min_positions: int = 1,
        max_positions: int | None = None,
    ) -> int:
        if candidate_count <= 0:
            return 0
        if selection_mode == "top_pct":
            pct = min(max(float(selection_pct), 0.001), 1.0)
            count = int(np.ceil(candidate_count * pct))
        else:
            count = int(top_n)
        count = max(int(min_positions), count)
        if max_positions is not None and int(max_positions) > 0:
            count = min(count, int(max_positions))
        return min(candidate_count, count)

    @staticmethod
    def equal_weights(selected: list[int], scale: float = 1.0) -> dict[int, float]:
        if not selected:
            return {}
        weight = float(scale) / len(selected)
        return {k: weight for k in selected}

    def build_targets(
        self,
        policy: SelectionPolicy,
        candidates,
        scores,
        market_adx=None,
    ) -> tuple[list[int], dict[int, float]]:
        """选股流水线唯一入口：门控 → 排名计数 → 等权 → 弱市叠加。

        candidates/scores 为同序数组（scores 对应候选的因子值）。
        返回 (chosen_list, targets)；无票过门控时返回 ([], {})。
        """
        cand = np.asarray(candidates)
        sc = np.asarray(scores, dtype=float)
        gated = policy.gate(cand, sc)
        if len(gated) == 0:
            return [], {}
        g_scores = sc[np.isin(cand, gated)]
        chosen = self.rank_select(
            gated, g_scores, policy.ascending,
            policy.top_n, policy.count_mode, policy.pct,
            policy.min_positions, policy.max_positions,
        )
        targets = self.equal_weights(chosen)
        scale = policy.scale_for_regime(market_adx)
        if scale != 1.0 and targets:
            targets = {k: v * scale for k, v in targets.items()}
        return chosen, targets

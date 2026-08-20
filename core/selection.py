"""组合候选筛选与目标权重构建。

该模块不读取行情、不执行订单，只把信号截面转换为目标持仓。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

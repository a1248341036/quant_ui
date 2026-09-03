"""Runtime context shared by transforms and metric plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.factor.evaluation.profile import EvaluationProfile
from alphaagent.factor.metrics import cross_sectional_ic, cross_sectional_rank_ic


@dataclass
class EvaluationContext:
    panel: pd.DataFrame
    factor: pd.Series
    label: pd.Series
    profile: EvaluationProfile
    factor_name: str
    cache: dict[str, Any] = field(default_factory=dict)
    transforms_applied: list[str] = field(default_factory=list)
    # label 名义持有期（label_20d → 20）：组合回测需按此节奏采样复利，
    # 否则 N 日收益被逐日重叠计入 N 次，年化/回撤/夏普全部失真。
    label_holding_days: int = 1

    def daily_ic(self) -> pd.Series:
        if "daily_ic" not in self.cache:
            self.cache["daily_ic"] = cross_sectional_ic(self.factor, self.label, min_pairs=5)
        return self.cache["daily_ic"]

    def daily_rank_ic(self) -> pd.Series:
        if "daily_rank_ic" not in self.cache:
            self.cache["daily_rank_ic"] = cross_sectional_rank_ic(self.factor, self.label, min_pairs=5)
        return self.cache["daily_rank_ic"]

    def replace_factor(self, values: np.ndarray, *, transform_name: str) -> None:
        self.factor = pd.Series(values, index=self.panel.index, name=self.factor_name, dtype=np.float32)
        self.cache.clear()
        self.transforms_applied.append(transform_name)

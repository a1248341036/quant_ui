"""股票因子挖掘评估上下文：panel 路径与 train/val 日期切分。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alphaagent.factor.types import DEFAULT_LABEL_COL


@dataclass
class StockEvalContext:
    """一次挖掘会话的数据与标签配置。"""

    panel_path: Path
    train_start: str = "2019-01-01"
    train_end: str = "2021-12-31"
    val_start: str = "2022-01-01"
    val_end: str = "2023-12-31"
    label_col: str = DEFAULT_LABEL_COL
    include_fundamentals: bool = True
    """是否载入基本面列（``funda_*``）。挖价量因子时可关闭以省内存。"""

    def split_range(self, split: str) -> tuple[str, str]:
        if split == "train":
            return self.train_start, self.train_end
        if split == "val":
            return self.val_start, self.val_end
        raise ValueError(f"未知 split: {split!r}")

    def coverage_range(self) -> tuple[str, str]:
        """train ∪ val 日期并集。"""
        return (
            min(self.train_start, self.val_start),
            max(self.train_end, self.val_end),
        )

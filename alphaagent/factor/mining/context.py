"""股票因子挖掘评估上下文：panel 路径与 train/val/test 日期切分。

三层时间隔离：
- train（2020-2022）：统计门槛
- val（2023-2024）：样本外保留率 + engine_gate
- test（2025-01 ~ 数据最新）：留出测试段，入库前终审（只报告不拦截）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alphaagent.factor.types import (
    DEFAULT_LABEL_COL,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)


def _test_end_default() -> str:
    """测试段右端默认：动态解析数据源最新交易日。"""
    from alphaagent.factor.window_config import resolve_test_end

    return resolve_test_end()


@dataclass
class StockEvalContext:
    """一次挖掘会话的数据与标签配置。"""

    panel_path: Path
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    val_start: str = DEFAULT_VAL_START
    val_end: str = DEFAULT_VAL_END
    test_start: str = DEFAULT_TEST_START
    test_end: str | None = DEFAULT_TEST_END
    """测试段右端；None = 动态解析数据源最新交易日。"""
    label_col: str = DEFAULT_LABEL_COL
    include_fundamentals: bool = True
    """是否载入基本面列（``funda_*``）。挖价量因子时可关闭以省内存。"""

    def resolved_test_end(self) -> str:
        """解析后的测试段右端（None → 数据源最新交易日）。"""
        return self.test_end or _test_end_default()

    def split_range(self, split: str) -> tuple[str, str]:
        if split == "train":
            return self.train_start, self.train_end
        if split == "val":
            return self.val_start, self.val_end
        if split == "test":
            return self.test_start, self.resolved_test_end()
        raise ValueError(f"未知 split: {split!r}")

    def coverage_range(self) -> tuple[str, str]:
        """train ∪ val ∪ test 日期并集（panel 加载范围）。"""
        test_end = self.resolved_test_end()
        return (
            min(self.train_start, self.val_start, self.test_start),
            max(self.train_end, self.val_end, test_end),
        )

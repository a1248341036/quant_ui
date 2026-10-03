# -*- coding: utf-8 -*-
"""字段别名 × 面板裁剪的顺序回归（2026-10-04）。

背景（run 61ded42e3ff8 实测）：`eval_multi_line_factor` 原先**先按原始表达式裁剪面板**、
再在编译期做别名改写。于是 `$turnover`（别名目标 `turnover_rate`）的目标列因为
"没被原始表达式引用"而被裁掉 → 编译期 known 集里没有它 → 误报

    symbol 阶段失败: 表达式引用了不可用字段: $turnover_rate

同 run 里直接写 `$turnover_rate` 的 10 次调用全部正常，只有写 `$turnover`（或 `$turnover`
出现在被裁剪路径上）的 3 次白烧。修法：别名改写提前到裁剪之前，并按**完整面板列**判定
（别名表本身是"目标列存在才改写"）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaagent.dsl.eval import eval_multi_line_factor


def _panel(cols: dict[str, np.ndarray], n_days: int = 30) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-01", periods=n_days, freq="D", name="datetime"),
            pd.Index(["A", "B", "C"], name="instrument"),
        ],
    )
    return pd.DataFrame(cols, index=idx)


def _cols(n: int, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "turnover": rng.uniform(0.5, 5.0, n),
        "turnover_rate": rng.uniform(0.5, 5.0, n),
        "amount": rng.uniform(1e6, 1e7, n),
        "close": 10.0 + np.cumsum(rng.normal(0, 0.1, n)),
    }


def test_alias_survives_panel_pruning():
    """`$turnover` 不得因"目标列没被原始表达式引用"而被裁掉后误报不可用字段。"""
    n = 90
    df = _panel(_cols(n))
    out = eval_multi_line_factor("TS_MEAN($turnover, 5)", df)
    assert out is not None
    assert len(out) == len(df)


def test_alias_only_target_when_target_present():
    """面板**只有** `turnover`（没有 `turnover_rate`）时：不改写、表达式照旧可用。"""
    n = 90
    cols = _cols(n)
    cols.pop("turnover_rate")
    df = _panel(cols)
    out = eval_multi_line_factor("TS_MEAN($turnover, 5)", df)
    assert out is not None


def test_alias_rewrite_happens_before_prune_multi_column_expr():
    """多列表达式里夹带别名：裁剪按改写后的引用集做，别名目标列必须留下。"""
    n = 90
    df = _panel(_cols(n))
    out = eval_multi_line_factor(
        "to = TS_MEAN($turnover, 5)\namt = TS_MEAN($amount, 5)\nDIVIDE(to, amt)",
        df,
    )
    assert out is not None

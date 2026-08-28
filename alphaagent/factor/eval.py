"""因子评估：DSL 求值 + IC/MLS 指标。

``evaluate_factor`` 提供全量 panel 上的快速评估（研究脚本用）。
按 train/val 切分的 split 评估已收敛到 ``EvaluationEngine``
（alphaagent.factor.evaluation.engine），由 mining/service.py 统一调用，
本模块不再保留第二套 split 评估实现。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from alphaagent.dsl import eval_factor
from alphaagent.factor.align import align_series_to_panel
from alphaagent.factor.metrics import evaluate_on_panel
from alphaagent.factor.types import DEFAULT_LABEL_COL
from alphaagent.data.panel import slice_panel


def evaluate_factor(
    expr: str,
    panel: pd.DataFrame,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    start: str | None = None,
    end: str | None = None,
    min_pairs: int = 5,
) -> dict[str, Any]:
    """在全量 panel 上求值 DSL，再按日期窗计算 IC/ICIR/RANKIC/MLS 等指标。"""
    panel_full = panel.sort_index()
    if label_col not in panel_full.columns:
        raise KeyError(f"panel 缺少标签列: {label_col}")
    out = eval_factor(expr, panel_full)
    if not isinstance(out, pd.Series):
        raise TypeError(f"因子输出须为 Series，得到 {type(out)!r}")
    values = align_series_to_panel(out, panel_full)
    eval_panel = slice_panel(panel_full, start=start, end=end)
    if eval_panel.empty:
        raise ValueError(f"评估切片为空: start={start!r} end={end!r}")
    pos = panel_full.index.isin(eval_panel.index)
    metrics = evaluate_on_panel(
        values[pos], eval_panel, label_col=label_col, min_ic_pairs=min_pairs
    )
    metrics["eval_start"] = start
    metrics["eval_end"] = end
    metrics["label_col"] = label_col
    return metrics
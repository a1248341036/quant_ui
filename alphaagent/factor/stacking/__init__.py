"""ML 因子组合（stacking）：因子池 → 截面特征 → walk-forward 模型 → 组合分数。

时间隔离契约：组合模型的训练/OOS 只允许使用 ``mining_end`` 之后的样本
（挖掘循环对 train/val 窗口的反复反馈会让入库因子 IC 幸存者偏差），
``mining_end`` 之前的窗口仅用于衰减对照表。
"""
from .dataset import (
    FactorEntry,
    StackingDataset,
    build_dataset_from_values,
    build_stacking_dataset,
    collect_factor_entries,
    daily_spearman_ic,
    decay_table,
    forward_return_label,
    materialize_entries,
    transform_factor_values,
)
from .model import (
    make_model,
    fit_predict_walkforward,
    walk_forward_splits,
)

__all__ = [
    "FactorEntry",
    "StackingDataset",
    "build_dataset_from_values",
    "build_stacking_dataset",
    "collect_factor_entries",
    "daily_spearman_ic",
    "decay_table",
    "forward_return_label",
    "materialize_entries",
    "transform_factor_values",
    "make_model",
    "fit_predict_walkforward",
    "walk_forward_splits",
]

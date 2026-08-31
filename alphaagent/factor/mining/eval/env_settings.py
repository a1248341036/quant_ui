"""挖掘评估相关的环境变量解析。"""

from __future__ import annotations

import os

ENV_MAX_PARALLEL_EVAL = "MAX_PARALLEL_EVAL"
DEFAULT_MAX_PARALLEL_EVAL = 1


def parse_max_parallel_eval(raw: str | None = None) -> int:
    """解析最大并行 train/val 评估数；未设置或空字符串时返回 1。"""
    if raw is None:
        raw = os.environ.get(ENV_MAX_PARALLEL_EVAL, "")
    text = str(raw).strip()
    if not text:
        return DEFAULT_MAX_PARALLEL_EVAL
    value = int(text)
    if value < 1:
        raise ValueError(f"{ENV_MAX_PARALLEL_EVAL} 须为正整数，得到 {value!r}")
    return value


def resolve_max_parallel_eval(override: int | None = None) -> int:
    """MiningConfig 显式值优先，否则读环境变量。"""
    if override is not None:
        return parse_max_parallel_eval(str(override))
    return parse_max_parallel_eval()

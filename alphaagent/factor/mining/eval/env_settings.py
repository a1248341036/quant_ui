"""挖掘评估相关的环境变量解析。"""

from __future__ import annotations

import os

ENV_MAX_PARALLEL_EVAL = "MAX_PARALLEL_EVAL"
# 2026-09-06：1 → 6。对比实验（docs/alphaagent_对比结果_run1.md）实测 CLI 默认 1
# 时评估完全串行，单轮 8 个 tool_calls 排队等待占墙钟 ~50%；Web 端 StartRequest
# 默认早已是 6，此处对齐。metrics 重活在 numba nogil 内核，6 并发可真多核。
DEFAULT_MAX_PARALLEL_EVAL = 6


def parse_max_parallel_eval(raw: str | None = None) -> int:
    """解析最大并行 train/val 评估数；未设置或空字符串时返回默认 6。"""
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

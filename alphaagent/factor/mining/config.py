"""因子挖掘配置：评估上下文 + LLM/循环参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphaagent.factor.mining.context import StockEvalContext


@dataclass
class MiningConfig:
    """一次挖掘运行的全部配置。"""

    eval: StockEvalContext
    model: str = "gpt-4o-mini"
    temperature: float | None = None
    max_tokens: int = 16384  # hy3 推理模型 thinking 约耗 8K，8192 会截断 tool_calls
    model_max_retries: int = 10
    """单次 LLM 调用的重试次数（框架默认 3）。抖动型代理/上游需要更大韧性。"""
    model_retry_delay: float = 5.0
    """重试间隔秒数（框架默认 1.0）；配合指数外的大间隔穿透上游抖动。"""
    population_max: int = 24
    """种群批量筛选（propose_population）单轮候选上限；0 = 关闭路径 B。"""
    max_turns: int = 16
    max_tool_calls_per_round: int = 8
    max_tool_workers: int = 4
    max_parallel_eval: int | None = None
    """同时进行的 train/val 评估上限；None 时读环境变量 MAX_PARALLEL_EVAL。"""
    min_tool_call_rounds_before_allow_stop: int = 3
    factorlib_path: Path | None = None
    enable_submit: bool = True  # 始终启用，已移除关闭开关
    enable_reviewer: bool = True
    research_spec: dict[str, Any] | None = None
    max_cs_corr: float = 0.8
    similar_top_k: int = 3
    ingest_overwrite: bool = False
    auto_realign_panel: bool = True
    # Keep delivery metadata with the FactorZoo unless a caller explicitly
    # overrides either path. A relative default here used to split successful
    # submissions across two different FactorZoo roots.
    registry_path: Path | None = None
    expr_dir: Path | None = None

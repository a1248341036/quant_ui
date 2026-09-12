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
    max_tokens: int = 16384  # 2026-09-12 曾下探到 12288，2026-09-12 退回：
    # run 42254d3990f0 实测 37.3% 调用打满 12288（中位 9.4K、p90 封顶），thinking
    # 平均约 2 万字符，深思考型模型输出时长——降档只会把截断点提前，反而引入
    # tool_calls JSON 被硬切作废的重试风暴（15fac33 的"16K 档只有截断浪费"结论
    # 与本次 run 数据不符）。16384 保留 thinking + 正文 + 并行 tool_calls 余量；
    # 8192 已验证会截断 tool_calls，仍是最低下限。
    model_max_retries: int = 10
    """单次 LLM 调用的重试次数（框架默认 3）。抖动型代理/上游需要更大韧性。"""
    model_retry_delay: float = 5.0
    """重试间隔秒数（框架默认 1.0）；配合指数外的大间隔穿透上游抖动。"""
    population_max: int = 0
    """种群批量筛选（propose_population）单轮候选上限；0 = 关闭路径 B。
    2026-09-03 默认关：并行批量扩容（12~20/轮）后常规路径已覆盖其吞吐，
    种群仅保留参数敏感性扫描用途，需要时显式开启。"""
    max_turns: int = 16
    max_tool_calls_per_round: int = 20
    max_tool_workers: int = 12
    max_parallel_eval: int | None = None
    """同时进行的 train/val 评估上限；None 时读环境变量 MAX_PARALLEL_EVAL。"""
    min_tool_call_rounds_before_allow_stop: int = 3
    factorlib_path: Path | None = None
    enable_submit: bool = True  # 始终启用，已移除关闭开关
    enable_reviewer: bool = True
    research_spec: dict[str, Any] | None = None
    focus_facets: list[str] | None = None
    """数据面聚焦（跨面融合）：用户在前端多选的面名（FACET_DEFS 键）。
    非空时每轮记忆注入附带聚焦提醒块，持续引导跨面融合。"""
    max_cs_corr: float = 0.8
    similar_top_k: int = 3
    ingest_overwrite: bool = False
    auto_realign_panel: bool = True
    # Keep delivery metadata with the FactorZoo unless a caller explicitly
    # overrides either path. A relative default here used to split successful
    # submissions across two different FactorZoo roots.
    registry_path: Path | None = None
    expr_dir: Path | None = None

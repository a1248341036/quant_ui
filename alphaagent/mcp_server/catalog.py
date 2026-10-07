# -*- coding: utf-8 -*-
"""AlphaAgent MCP server · 工具目录（JSON Schema + 处理器映射）。

`DESCRIPTORS` 直接作为 MCP `tools/list` 的返回值；`HANDLERS[name](ctx, **arguments)` 为处理器。
工具命名与 `docs/review/agent_tool_mcp_inventory_20261007.md` 的 E 类清单一致。
"""
from __future__ import annotations

from typing import Any, Callable

from alphaagent.mcp_server import tools

_MODE = {
    "type": "string",
    "enum": ["technical", "technical_weekly", "technical_monthly", "fundamental"],
    "default": "technical",
    "description": "研究档位（决定评估 label 与交付调仓频率）",
}


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or [],
            "additionalProperties": False}


_EXPR = {"type": "string", "description": "多行 DSL 表达式（末行为输出，如 `z = TS_MEAN($ret, 20)\\nCS_ZSCORE(z)`）"}

DESCRIPTORS: list[dict[str, Any]] = [
    {
        "name": "get_thresholds",
        "description": "查某档位的**生效门槛**（候选/正式库/engine_gate/换手硬门）。写因子前先看这个，别猜数值。",
        "inputSchema": _obj({"mode": _MODE}),
    },
    {
        "name": "release_session",
        "description": "释放评估会话与跨进程会话锁。多客户端（DSH/Codex/…）共用一台机器时，"
                       "重工具用完建议调用，让其它客户端能接管 panel（每会话 6–8GB）。",
        "inputSchema": _obj({}),
    },
    {
        "name": "list_fields",
        "description": "列出面板可用列（写 DSL 前必需）。可用 prefixes 过滤，如 'funda_,holder_'。",
        "inputSchema": _obj({"prefixes": {"type": "string", "default": ""},
                             "limit": {"type": "integer", "default": 0}}),
    },
    {
        "name": "describe_operator",
        "description": "算子目录：不传 name 返回全量 Markdown；传 name 只回该算子的签名行（如 TS_CORR）。",
        "inputSchema": _obj({"name": {"type": "string", "default": ""}}),
    },
    {
        "name": "precheck_expression",
        "description": "DSL 结构风险静态预检（纯 AST，不耗评估预算）。提交/评估前先跑。",
        "inputSchema": _obj({"multi_line_expr": _EXPR}, ["multi_line_expr"]),
    },
    {
        "name": "list_runs",
        "description": "最近 run 列表（outcome / 评估数 / 候选入库数）。",
        "inputSchema": _obj({"limit": {"type": "integer", "default": 10}}),
    },
    {
        "name": "run_summary",
        "description": "单个 run 的汇总 + steps.log 尾部（run_id 省略则取最新）。",
        "inputSchema": _obj({"run_id": {"type": "string", "default": ""}}),
    },
    {
        "name": "memory_search",
        "description": "检索跨 run 研究记忆（经验/死路证据/饱和度/多样性）——即真实 run 每轮注入的记忆块。",
        "inputSchema": _obj({"research_goal": {"type": "string"},
                             "limit": {"type": "integer", "default": 12}}, ["research_goal"]),
    },
    {
        "name": "memory_stats",
        "description": "记忆层统计：面/算子饱和度 + 研究漏斗。",
        "inputSchema": _obj({}),
    },
    {
        "name": "eval_batch",
        "description": "★批量评估：一次调用评多个表达式（≤60），返回逐因子指标 + 逐门 PASS/FAIL（真源门槛）。"
                       "比逐个工具调用快一个数量级；过门后再 dry_run_delivery / submit_factor。",
        "inputSchema": _obj({
            "exprs": {"type": "array", "description": "[{name, expr, why?}] 或 ['表达式', ...]",
                      "items": {"type": ["object", "string"]}},
            "mode": _MODE,
            "split": {"type": "string", "enum": ["train", "val"], "default": "train"},
            "fundamentals": {"type": "boolean", "default": False,
                             "description": "会话是否载入 funda_* 列（用基本面字段时必须 true）"},
            "quantile_n": {"type": "integer", "default": 10},
        }, ["exprs"]),
    },
    {
        "name": "eval_val",
        "description": "样本外（val 段）评估 + 与 train 的 IC 保留比 / 方向一致性。",
        "inputSchema": _obj({"multi_line_expr": _EXPR, "mode": _MODE,
                             "fundamentals": {"type": "boolean", "default": False},
                             "quantile_n": {"type": "integer", "default": 10}}, ["multi_line_expr"]),
    },
    {
        "name": "library_similarity",
        "description": "★与**候选池已有因子**的截面相关性（stage_two 相关性墙提前预警；>max_abs_corr 会挡住正式库晋升）。",
        "inputSchema": _obj({"multi_line_expr": _EXPR, "mode": _MODE,
                             "top_k": {"type": "integer", "default": 3}}, ["multi_line_expr"]),
    },
    {
        "name": "dry_run_delivery",
        "description": "★只读交付预检：train+val 评估 + 库内相关性 + stage_one/stage_two 判定（不写库）。"
                       "submit 之前必跑。",
        "inputSchema": _obj({"multi_line_expr": _EXPR, "mode": _MODE,
                             "fundamentals": {"type": "boolean", "default": False},
                             "quantile_n": {"type": "integer", "default": 10}}, ["multi_line_expr"]),
    },
    {
        "name": "submit_factor",
        "description": "**写**：真实交付（stage_one → 盲测 → stage_two → engine_gate）。stage_one 过即入候选池"
                       "（candidate_stored=true）；正式库需再过 stage_two 与 engine_gate。必须 confirm=true。",
        "inputSchema": _obj({"multi_line_expr": _EXPR, "factor_name": {"type": "string"},
                             "comment": {"type": "string", "description": "机制与经济直觉（必填）"},
                             "mode": _MODE, "confirm": {"type": "boolean", "default": False},
                             "fundamentals": {"type": "boolean", "default": False}},
                            ["multi_line_expr", "factor_name", "comment"]),
    },
    {
        "name": "memory_record",
        "description": "**写**：把一次评估写回 research_memory（跨 run 学习；直驱挖掘不写回会让系统记忆看不见）。"
                       "必须 confirm=true。",
        "inputSchema": _obj({"multi_line_expr": _EXPR, "factor_name": {"type": "string", "default": "expr"},
                             "mode": _MODE,
                             "split": {"type": "string", "enum": ["train", "val"], "default": "train"},
                             "run_id": {"type": "string", "default": "mcp-session"},
                             "confirm": {"type": "boolean", "default": False}},
                            ["multi_line_expr"]),
    },
]

HANDLERS: dict[str, Callable[..., Any]] = {
    "get_thresholds": tools.get_thresholds,
    "release_session": tools.release_session,
    "list_fields": tools.list_fields,
    "describe_operator": tools.describe_operator,
    "precheck_expression": tools.precheck_expression,
    "list_runs": tools.list_runs,
    "run_summary": tools.run_summary,
    "memory_search": tools.memory_search,
    "memory_stats": tools.memory_stats,
    "eval_batch": tools.eval_batch,
    "eval_val": tools.eval_val,
    "library_similarity": tools.library_similarity,
    "dry_run_delivery": tools.dry_run_delivery,
    "submit_factor": tools.submit_factor,
    "memory_record": tools.memory_record,
}

assert set(HANDLERS) == {d["name"] for d in DESCRIPTORS}, "DESCRIPTORS 与 HANDLERS 必须一一对应"

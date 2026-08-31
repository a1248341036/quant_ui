"""LLM 工具 JSON schema 参数定义。"""
from __future__ import annotations

from typing import Any

from alphaagent.factor.mining.interactions import INTERACTION_TYPES


_INTERACTION_PARAMETER: dict[str, Any] = {
    "type": "object",
    "description": "多因子交互契约；结构化交互算子和 MULTIPLY 必须提供。",
    "properties": {
        "interaction_type": {
            "type": "string",
            "enum": sorted(INTERACTION_TYPES),
        },
        "base_signal": {"type": "string"},
        "condition_signal": {"type": "string"},
        "economic_mechanism": {
            "type": "string",
            "description": ">=20字的因果机制，而非指标复述。",
        },
        "expected_subgroup_pattern": {},
        "ablation_required": {"type": "boolean", "default": True},
    },
    "required": ["interaction_type", "base_signal", "condition_signal", "economic_mechanism"],
}


_EVAL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "multi_line_expr": {
            "type": "string",
            "description": "多行因子 DSL：可含赋值行，最后一行为因子值；列用 $列名 引用，算子大写。",
        },
        "factor_name": {"type": "string", "description": "因子列逻辑名，默认 expr。"},
        "include_detail_tables": {
            "type": "boolean",
            "description": "true 时额外返回 by_month / by_symbol 明细；默认 false 仅返回 summary。",
            "default": False,
        },
        "label_quantile_n": {
            "type": "integer",
            "description": "按因子值等频分位分桶，输出每桶 label 均值；0 表示不计算。默认 10。",
            "default": 10,
        },
        "interaction": _INTERACTION_PARAMETER,
        "parent_factor": {
            "type": "string",
            "description": "变异父本的因子逻辑名（A/B/C 变异轨必传；D 新族留空）。研究记忆按此建立父子观测。",
        },
        "edit_note": {
            "type": "string",
            "description": "意向编辑说明，格式：edit=<motif> <参数变化>，如 edit=window_rescale 10→20；motif 取 window_rescale / operator_substitute / normalization_change。",
        },
    },
    "required": ["multi_line_expr"],
    "additionalProperties": False,
}


_VAL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        **_EVAL_PARAMETERS["properties"],
        "expected_sign": {
            "type": "integer",
            "description": "train summary.ic 的符号（1=正、-1=负）；传入后返回 sign_check。",
            "enum": [1, -1],
        },
        "interaction": _INTERACTION_PARAMETER,
    },
    "required": ["multi_line_expr"],
    "additionalProperties": False,
}

_PROFILE_EVAL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "multi_line_expr": _EVAL_PARAMETERS["properties"]["multi_line_expr"],
        "factor_name": _EVAL_PARAMETERS["properties"]["factor_name"],
        "profile_id": {
            "type": "string",
            "description": "冻结的 EvaluationProfile ID；决定 split、transform、metrics 与 rule gate。",
        },
        "interaction": _INTERACTION_PARAMETER,
        "parent_factor": _EVAL_PARAMETERS["properties"]["parent_factor"],
        "edit_note": _EVAL_PARAMETERS["properties"]["edit_note"],
    },
    "required": ["multi_line_expr", "profile_id"],
    "additionalProperties": False,
}


_SUBMIT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "multi_line_expr": {
            "type": "string",
            "description": "与 train/val 评估一致的多行因子 DSL。",
        },
        "factor_name": {
            "type": "string",
            "description": "因子唯一逻辑名（蛇形英文），将写入因子库 factor_id。",
        },
        "comment": {
            "type": "string",
            "description": "因子含义说明：经济直觉、关键算子与窗口、预期 IC 方向等，供后续查阅。",
        },
        "interaction": _INTERACTION_PARAMETER,
        "rebalance_freq": {
            "type": "string",
            "enum": ["daily", "weekly", "monthly"],
            "description": "交付调仓频率：对比 evaluate 结果中 topn_portfolio.by_freq 三种频率的收益/换手/重合率后选择；缺省 daily。",
        },
        "parent_factor": {
            "type": "string",
            "description": "变异父本的因子逻辑名（A/B/C 变异轨必传；D 新族留空）。",
        },
        "edit_note": {
            "type": "string",
            "description": "意向编辑说明，格式：edit=<motif> <参数变化>。",
        },
    },
    "required": ["multi_line_expr", "factor_name", "comment"],
    "additionalProperties": False,
}


_SCREEN_FACTORS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "factor_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "要筛选的因子名列表（正式库名）。为空时自动用正式库全部因子。"
            ),
        },
        "signal_date": {
            "type": "string",
            "description": (
                "信号日（YYYY-MM-DD）。为空时用 val 段最后一天。Screener 在此日"
                "检测 regime + 回看 lookback 天 Rank IC 评分。"
            ),
        },
    },
    "required": [],
    "additionalProperties": False,
}


TOOL_NAMES = ("evaluate_factor", "eval_on_train_set", "eval_on_val_set", "submit_factor", "screen_factors")

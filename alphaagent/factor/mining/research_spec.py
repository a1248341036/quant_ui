"""Versioned, runtime research policy for AlphaAgent factor mining."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from alphaagent.factor.evaluation.profile import default_evaluation_profiles, resolve_profiles
from alphaagent.factor.mining.interactions import INTERACTION_TYPES


def _default_evaluation_profile_spec() -> dict[str, Any]:
    # Rules are compiled from evaluation_policy unless the user explicitly overrides them.
    profiles: dict[str, Any] = {}
    for profile_id, profile in default_evaluation_profiles().items():
        value = profile.as_dict()
        value.pop("rules", None)
        profiles[profile_id] = value
    return profiles


RESEARCH_MODES = ("technical", "fundamental")


DEFAULT_RESEARCH_SPEC: dict[str, Any] = {
    "version": 1,
    "research_mode": "technical",
    # 信息性提示：该模式建议的评估 label，前端/调用方可据此设置 --label-col。
    "recommended_label_col": "label_1d_open_to_open",
    "search_policy": {
        "allowed_signal_families": ["volume_price", "volatility", "chip", "momentum_reversal"],
        "forbidden_signal_families": ["pure_size"],
        "min_distinct_raw_fields": 2,
        "require_time_series_structure": False,
        "max_candidates_per_round": 5,
    },
    "evaluation_policy": {
        "min_train_abs_ic": 0.02,
        "min_train_icir": 0.25,
        "min_train_coverage": 0.85,
        "min_val_abs_ic": 0.01,
        "min_val_ic_retention_ratio": 0.5,
        "require_sign_consistency": True,
        # 换手率约束：因子排名日度自相关低于此值的候选不入池（高换手→高交易成本）
        "min_cs_autocorr": 0.18,
    },
    "evaluation_profiles": _default_evaluation_profile_spec(),
    "review_policy": {
        "enabled": True,
        "review_on": ["validation", "pre_submit"],
        "block_classic_transforms": True,
        "minimum_novelty": "medium",
    },
    "interaction_policy": {
        # 用户约束：尽量不用乘法——默认禁用 MULTIPLY 交互（含契约形式）。
        # 确需乘法时，须在运行 ResearchSpec 中显式把 "multiplication" 加回本列表，
        # 并按 reviewer 要求提供 base-only / condition-only / combined 消融证据。
        "allowed_interaction_types": [t for t in INTERACTION_TYPES if t != "multiplication"],
        "block_undeclared_multiply": True,
        "require_contract_for_typed_interactions": True,
        "require_ablation_for_multiplication": True,
    },
    "memory_policy": {
        "retrieve_limit": 12,
        "dynamic_retrieve_limit": 6,
        "max_expression_chars": 320,
        "include_rejected_paths": True,
        "prefer_orthogonal_to_approved": True,
        "include_expression": True,
    },
    "delivery_policy": {
        "candidate": {
            "min_abs_ic": 0.015,
            "min_icir": 0.2,
            "min_coverage": 0.85,
            "max_abs_corr": 0.5,
        },
        "production": {
            "min_abs_ic": 0.035,
            "min_icir": 0.5,
            "min_fmb_t_stat": 2.5,
            "min_long_group_annual_excess_return": 0.03,
            "max_winsorized_abs_ic_decay": 0.10,
            "max_abs_corr": 0.4,
            # 进正式库前的最后一道门：旧交易引擎完整约束回测
            # （T+1/整手/涨跌停/停牌/费率/滑点/流动性），纯内存，不落盘。
            # TopN 可交易性（超额/Sharpe/回撤/尾部稳定）全部由 engine_gate
            # 用完整引擎裁决，不再使用简化模拟代理指标。
            # 尾部稳定性说明：全局截面自相关高不代表尾部稳定——极端前 N 名
            # 可能每天几乎全换（IC 有效但不可持仓）。
            "engine_gate": {
                "enabled": True,
                "top_n": 30,
                "freq": "daily",
                # LLM 可在 submit 时选择交付调仓频率，必须属于本列表；
                # 选择会记录进审计轨迹，引擎按该频率复检。
                "allowed_freqs": ["daily", "weekly", "monthly"],
                "min_annual_return": 0.0,
                "min_excess_annual": 0.05,
                "min_sharpe": 0.5,
                "max_drawdown": 0.35,
                "min_daily_overlap": 0.5,
            },
        },
    },
}


def default_research_spec(mode: str = "technical") -> dict[str, Any]:
    spec = copy.deepcopy(DEFAULT_RESEARCH_SPEC)
    if mode == "technical":
        return spec
    if mode != "fundamental":
        raise ValueError("research_spec.research_mode_invalid")
    spec["research_mode"] = mode
    # 基本面为慢因子：财报 PIT 阶跃数据对 1d label 几乎无预测力，评估须用 10d 持有期 label。
    spec["recommended_label_col"] = "label_10d_close_to_close"
    spec["search_policy"].update({
        "allowed_signal_families": ["fundamental_quality", "fundamental_growth", "fundamental_value", "fundamental_revision"],
        "forbidden_signal_families": ["pure_size", "pure_price_momentum"],
        "min_distinct_raw_fields": 2,
        "require_time_series_structure": True,
    })
    return spec


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"research_spec.{name}_must_be_object")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"research_spec.{name}_must_be_boolean")
    return value


def _bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= float(value) <= upper:
        raise ValueError(f"research_spec.{name}_must_be_between_{lower}_and_{upper}")
    return float(value)


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"research_spec.{name}_must_be_string_list")
    return [item.strip() for item in value]


def normalize_research_spec(value: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a user policy with defaults and validate fields used by the engine."""
    if value is not None and not isinstance(value, dict):
        raise ValueError("research_spec_must_be_object")
    mode = str((value or {}).get("research_mode", "technical"))
    if mode not in RESEARCH_MODES:
        raise ValueError("research_spec.research_mode_invalid")
    spec = _deep_merge(default_research_spec(mode), value or {})
    if int(spec.get("version", 0)) != 1:
        raise ValueError("research_spec.unsupported_version")
    spec["version"] = 1
    spec["research_mode"] = mode

    search = _require_dict(spec.get("search_policy"), "search_policy")
    search["allowed_signal_families"] = _string_list(search.get("allowed_signal_families"), "search_policy.allowed_signal_families")
    search["forbidden_signal_families"] = _string_list(search.get("forbidden_signal_families"), "search_policy.forbidden_signal_families")
    search["min_distinct_raw_fields"] = int(_bounded_number(search.get("min_distinct_raw_fields"), "search_policy.min_distinct_raw_fields", 1, 10))
    search["require_time_series_structure"] = _require_bool(search.get("require_time_series_structure"), "search_policy.require_time_series_structure")
    search["max_candidates_per_round"] = int(_bounded_number(search.get("max_candidates_per_round"), "search_policy.max_candidates_per_round", 1, 16))

    evaluation = _require_dict(spec.get("evaluation_policy"), "evaluation_policy")
    for key in ("min_train_abs_ic", "min_val_abs_ic"):
        evaluation[key] = _bounded_number(evaluation.get(key), f"evaluation_policy.{key}", 0, 1)
    for key in ("min_train_icir",):
        evaluation[key] = _bounded_number(evaluation.get(key), f"evaluation_policy.{key}", -10, 20)
    evaluation["min_train_coverage"] = _bounded_number(evaluation.get("min_train_coverage"), "evaluation_policy.min_train_coverage", 0, 1)
    evaluation["min_val_ic_retention_ratio"] = _bounded_number(evaluation.get("min_val_ic_retention_ratio"), "evaluation_policy.min_val_ic_retention_ratio", 0, 2)
    evaluation["require_sign_consistency"] = _require_bool(evaluation.get("require_sign_consistency"), "evaluation_policy.require_sign_consistency")
    evaluation["min_cs_autocorr"] = _bounded_number(
        evaluation.get("min_cs_autocorr", 0), "evaluation_policy.min_cs_autocorr", 0, 1
    )

    review = _require_dict(spec.get("review_policy"), "review_policy")
    review["enabled"] = _require_bool(review.get("enabled"), "review_policy.enabled")
    review["review_on"] = _string_list(review.get("review_on"), "review_policy.review_on")
    invalid_hooks = set(review["review_on"]) - {"validation", "pre_submit"}
    if invalid_hooks:
        raise ValueError("research_spec.review_policy.review_on_invalid")
    review["block_classic_transforms"] = _require_bool(review.get("block_classic_transforms"), "review_policy.block_classic_transforms")
    if review.get("minimum_novelty") not in {"low", "medium", "high"}:
        raise ValueError("research_spec.review_policy.minimum_novelty_invalid")

    interaction = _require_dict(spec.get("interaction_policy"), "interaction_policy")
    allowed_types = _string_list(
        interaction.get("allowed_interaction_types"),
        "interaction_policy.allowed_interaction_types",
    )
    invalid_types = set(allowed_types) - set(INTERACTION_TYPES)
    if invalid_types:
        raise ValueError(f"interaction_policy.unknown_types:{','.join(sorted(invalid_types))}")
    interaction["allowed_interaction_types"] = allowed_types
    interaction["block_undeclared_multiply"] = _require_bool(
        interaction.get("block_undeclared_multiply"),
        "interaction_policy.block_undeclared_multiply",
    )
    interaction["require_contract_for_typed_interactions"] = _require_bool(
        interaction.get("require_contract_for_typed_interactions"),
        "interaction_policy.require_contract_for_typed_interactions",
    )
    interaction["require_ablation_for_multiplication"] = _require_bool(
        interaction.get("require_ablation_for_multiplication"),
        "interaction_policy.require_ablation_for_multiplication",
    )

    memory = _require_dict(spec.get("memory_policy"), "memory_policy")
    memory["retrieve_limit"] = int(_bounded_number(memory.get("retrieve_limit"), "memory_policy.retrieve_limit", 0, 100))
    memory["dynamic_retrieve_limit"] = int(_bounded_number(
        memory.get("dynamic_retrieve_limit"), "memory_policy.dynamic_retrieve_limit", 0, 100))
    memory["max_expression_chars"] = int(_bounded_number(
        memory.get("max_expression_chars"), "memory_policy.max_expression_chars", 0, 4000))
    memory["include_rejected_paths"] = _require_bool(memory.get("include_rejected_paths"), "memory_policy.include_rejected_paths")
    memory["prefer_orthogonal_to_approved"] = _require_bool(memory.get("prefer_orthogonal_to_approved"), "memory_policy.prefer_orthogonal_to_approved")
    memory["include_expression"] = _require_bool(memory.get("include_expression"), "memory_policy.include_expression")

    delivery = _require_dict(spec.get("delivery_policy"), "delivery_policy")
    candidate = _require_dict(delivery.get("candidate"), "delivery_policy.candidate")
    candidate["min_abs_ic"] = _bounded_number(candidate.get("min_abs_ic"), "delivery_policy.candidate.min_abs_ic", 0, 1)
    candidate["min_icir"] = _bounded_number(candidate.get("min_icir"), "delivery_policy.candidate.min_icir", -10, 20)
    candidate["min_coverage"] = _bounded_number(candidate.get("min_coverage"), "delivery_policy.candidate.min_coverage", 0, 1)
    candidate["max_abs_corr"] = _bounded_number(candidate.get("max_abs_corr"), "delivery_policy.candidate.max_abs_corr", 0, 1)
    production = _require_dict(delivery.get("production"), "delivery_policy.production")
    for key in ("min_abs_ic", "min_long_group_annual_excess_return", "max_winsorized_abs_ic_decay", "max_abs_corr"):
        production[key] = _bounded_number(production.get(key), f"delivery_policy.production.{key}", 0, 1)
    production["min_icir"] = _bounded_number(production.get("min_icir"), "delivery_policy.production.min_icir", -10, 20)
    production["min_fmb_t_stat"] = _bounded_number(production.get("min_fmb_t_stat", 0), "delivery_policy.production.min_fmb_t_stat", 0, 20)
    profiles = resolve_profiles(spec)
    spec["evaluation_profiles"] = {profile_id: profile.as_dict() for profile_id, profile in profiles.items()}

    # 注入换手率约束：低自相关(高换手)因子不入候选池
    ac_min = float(evaluation.get("min_cs_autocorr", 0))
    if ac_min > 0:
        ac_rule = {"metric": "cross_sectional_core.cs_pearson_autocorr", "op": "gte", "value": ac_min}
        for pid in ("train_screen",):
            prof = spec["evaluation_profiles"].get(pid)
            if prof is None:
                continue
            rules = prof.setdefault("rules", [])
            if not any(r.get("metric") == ac_rule["metric"] for r in rules):
                rules.append(dict(ac_rule))

    return spec


def load_research_spec(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_research_spec()
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"research_spec_load_failed: {exc}") from exc
    return normalize_research_spec(value)


def research_policy_prompt(spec: dict[str, Any]) -> str:
    """A concise policy contract consumed by the primary mining agent."""
    search = spec["search_policy"]
    evaluation = spec["evaluation_policy"]
    review = spec["review_policy"]
    interaction = spec.get("interaction_policy", {})
    profiles = spec.get("evaluation_profiles", {})
    fundamental = spec.get("research_mode") == "fundamental"
    mode_line = (
        "研究模式：基本面因子。信号以 `$funda_*` 财务字段为主（PIT 日频阶跃数据，注意披露生效日与低换手），"
        f"本次评估 label 为 {spec.get('recommended_label_col', 'label_10d_close_to_close')}（10 日持有期）。"
        if fundamental
        else "研究模式：日线技术因子。信号以价量/波动/筹码等行情字段为主。"
    )
    return "\n".join(
        [
            "# 本次运行的 ResearchSpec（高优先级研究约束）",
            mode_line,
            f"允许的信号族：{', '.join(search['allowed_signal_families']) or '不限制'}。",
            f"禁止作为独立候选的信号族：{', '.join(search['forbidden_signal_families']) or '不限制'}。",
            f"每个保留候选至少使用 {search['min_distinct_raw_fields']} 个彼此独立的原始字段。",
            "候选必须包含时序结构。" if search["require_time_series_structure"] else "允许纯截面候选，但必须说明其独立经济机制。",
            f"每轮最多提出 {search['max_candidates_per_round']} 个候选。",
            f"本次允许的 EvaluationProfile：{', '.join(sorted(profiles))}。只能选择已有 profile_id，不得自定义临时参数。",
            "进入验证的最低 train 要求："
            f"abs(IC)>={evaluation['min_train_abs_ic']:.4g}，abs(ICIR)>={evaluation['min_train_icir']:.4g}，"
            f"Coverage>={evaluation['min_train_coverage']:.4g}。",
            "验证要求："
            f"abs(IC)>={evaluation['min_val_abs_ic']:.4g}，val/train abs(IC) 保留比例>="
            f"{evaluation['min_val_ic_retention_ratio']:.4g}，"
            + ("方向必须一致。" if evaluation["require_sign_consistency"] else "方向一致性仅作诊断。"),
            "换手率约束："
            f"cs_pearson_autocorr>={evaluation.get('min_cs_autocorr', 0):.4g}——低于此值的因子排名日度变化过快、"
            "实际交易成本会吃掉全部 alpha，将被自动拦截不入候选池。",
            "Reviewer 策略："
            f"在 {', '.join(review['review_on'])} 阶段审查；最低新颖性={review['minimum_novelty']}；"
            + ("经典单调变换必须阻断。" if review["block_classic_transforms"] else "经典变换只提示，不自动阻断。"),
            "交互策略："
            f"允许 interaction_type={', '.join(interaction.get('allowed_interaction_types', []))}；"
            + ("未声明契约的 MULTIPLY 直接拦截。" if interaction.get("block_undeclared_multiply") else "MULTIPLY 仅提示。")
            + "所有结构化多因子交互都必须传完整 interaction 契约。",
            "交付策略："
            "通过 validation 的因子自动进入 candidate 候选池；Reviewer approve 后进入 production 正式库。",
        ]
    )

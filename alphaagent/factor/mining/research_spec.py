"""Versioned, runtime research policy for AlphaAgent factor mining."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from alphaagent.factor.evaluation.profile import default_evaluation_profiles, resolve_profiles
from alphaagent.factor.mining.interactions import INTERACTION_TYPES
from alphaagent.core.paths import RESEARCH_SPECS_DIR
from core import trading_config
from core.research_modes import RESEARCH_MODES as _MODE_REGISTRY, get_research_mode


def _default_evaluation_profile_spec() -> dict[str, Any]:
    # Rules are compiled from evaluation_policy unless the user explicitly overrides them.
    profiles: dict[str, Any] = {}
    for profile_id, profile in default_evaluation_profiles().items():
        value = profile.as_dict()
        value.pop("rules", None)
        profiles[profile_id] = value
    return profiles


RESEARCH_MODES = tuple(_MODE_REGISTRY.keys())


# ── 用户门槛覆盖持久化（每模式一个 JSON 文件）────────────────────────
# 存储的是"相对注册表默认值的增量覆盖"（compute_spec_overrides 的 diff 结果），
# 而非整份 spec：代码默认值演进时，用户未改过的键自动跟随，改过的键保持覆盖。
# 消费方：
# - effective_research_spec(mode)  → 默认 + 覆盖（normalize 后），前端编辑/展示用
# - build_run_research_spec(spec)  → 运行口径：注册表默认 < 保存覆盖 < 显式 spec
# - core.factor_categories 不涉及门槛；全链路（CLI/Web/晋升）统一走上述两个入口。
def _spec_overrides_path(mode: str) -> Path:
    if mode not in RESEARCH_MODES:
        raise ValueError(f"research_mode_invalid:{mode}")
    return RESEARCH_SPECS_DIR / f"{mode}.json"


def load_saved_overrides(mode: str) -> dict[str, Any]:
    """读取某模式的用户门槛覆盖；无文件/损坏时返回空 dict（回落默认）。"""
    try:
        value = json.loads(_spec_overrides_path(mode).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_research_spec_overrides(mode: str, overrides: dict[str, Any]) -> Path:
    """保存某模式的用户门槛覆盖（默认值无需保存，丢键即回落默认）。"""
    if not isinstance(overrides, dict):
        raise ValueError("research_spec.overrides_must_be_object")
    path = _spec_overrides_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def reset_research_spec_overrides(mode: str) -> bool:
    """删除某模式的覆盖文件，恢复注册表默认门槛。返回是否真的有文件被删。"""
    path = _spec_overrides_path(mode)
    if not path.exists():
        return False
    path.unlink(missing_ok=True)
    return True


def compute_spec_overrides(defaults: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    """默认 spec 与前端编辑后 spec 的增量 diff（递归 dict；list/标量整体替换）。

    只保留与默认值不同的键 → 持久化的"门槛文件"最小、可读且随代码默认演进。
    """
    out: dict[str, Any] = {}
    for key, value in edited.items():
        if key not in defaults:
            out[key] = value
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            sub = compute_spec_overrides(defaults[key], value)
            if sub:
                out[key] = sub
        elif value != defaults[key]:
            out[key] = value
    # 编辑版显式删掉的键 = 恢复默认，无需写回
    return out


def effective_research_spec(mode: str = "technical") -> dict[str, Any]:
    """注册表默认 + 用户保存覆盖（normalize 后的完整 spec）。

    前端编辑/展示、以及"不显式传 spec"的运行路径都以此为准；
    它不会改变 default_research_spec 的纯默认语义（测试/调用方依旧可取纯默认）。
    """
    merged = _deep_merge(default_research_spec(mode), load_saved_overrides(mode))
    return normalize_research_spec(merged)


def build_run_research_spec(explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    """运行口径研究规范：注册表默认 < 保存覆盖 < 显式 spec（如前端 JSON / CLI 文件）。

    显式 spec 已含保存值时幂等（保存值再合并一次不改变结果）；
    显式 spec 缺键时从保存覆盖/默认补齐——保证任何入口都不会绕过用户改过的门槛。
    """
    explicit = dict(explicit) if isinstance(explicit, dict) else {}
    mode = str(explicit.get("research_mode") or "technical")
    base = default_research_spec(mode)
    merged = _deep_merge(_deep_merge(base, load_saved_overrides(mode)), explicit)
    return normalize_research_spec(merged)


def _default_delivery_policy() -> dict[str, Any]:
    """两阶段交付门槛默认值，唯一真源在 delivery_criteria（含设计注释）。

    运行时以 research_spec 注入为准（agentscope/run 均传 delivery_policy）；
    delivery_criteria.DeliveryCriteria.defaults() 集中定义数值与设计依据，
    prompt 渲染（tools.py / prompts.py）也从同一对象取数，杜绝硬编码漂移。
    """
    from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria

    return DeliveryCriteria.defaults().to_spec_dict()


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
        # 新颖度门槛暂关（2026-08）：先积累一批统计有效的候选，再回头筛新颖性。
        # 恢复严格筛选时改回 "medium"/"high"；Reviewer 仍输出 novelty 供参考。
        "minimum_novelty": "low",
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
    "delivery_policy": _default_delivery_policy(),
}


def default_research_spec(mode: str = "technical") -> dict[str, Any]:
    """按研究模式生成默认 ResearchSpec（门槛/信号族/label 来自 core.research_modes 注册表）。

    加新模式只需在 core/research_modes.RESEARCH_MODES 增加一项，本函数零改动。
    """
    spec = copy.deepcopy(DEFAULT_RESEARCH_SPEC)
    if mode == "technical":
        return spec
    mode_spec = get_research_mode(mode)
    spec["research_mode"] = mode
    spec["recommended_label_col"] = mode_spec.recommended_label_col
    spec["search_policy"].update({
        "allowed_signal_families": list(mode_spec.signal_families),
        "forbidden_signal_families": list(mode_spec.forbidden_families),
        "min_distinct_raw_fields": 2,
        "require_time_series_structure": True,
    })
    if mode_spec.evaluation_overrides:
        spec["evaluation_policy"].update(mode_spec.evaluation_overrides)
    if mode_spec.candidate_overrides:
        spec["delivery_policy"]["candidate"].update(mode_spec.candidate_overrides)
    if mode_spec.production_overrides:
        spec["delivery_policy"]["production"].update(mode_spec.production_overrides)
    if mode_spec.engine_gate_overrides:
        spec["delivery_policy"]["production"]["engine_gate"].update(
            mode_spec.engine_gate_overrides
        )
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
    if review.get("reviewer_max_tokens") is not None:
        raw_tokens = review["reviewer_max_tokens"]
        try:
            coerced = int(raw_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("research_spec.review_policy.reviewer_max_tokens_invalid") from exc
        review["reviewer_max_tokens"] = int(
            _bounded_number(coerced, "review_policy.reviewer_max_tokens", 256, 200_000)
        )

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
    # 换手可行性与样本外保留比（0.18 / 0.5 由 DEFAULT_RESEARCH_SPEC 提供默认值）
    candidate["min_cs_autocorr"] = _bounded_number(candidate.get("min_cs_autocorr"), "delivery_policy.candidate.min_cs_autocorr", 0, 1)
    candidate["min_val_ic_retention"] = _bounded_number(candidate.get("min_val_ic_retention"), "delivery_policy.candidate.min_val_ic_retention", 0, 1)
    production = _require_dict(delivery.get("production"), "delivery_policy.production")
    # 双窗口统计门槛（2026-08 重构：混合窗口稀释 val 衰减，已弃用单口径 min_abs_ic/min_icir）
    for key in ("min_train_abs_ic", "min_val_abs_ic", "max_winsorized_abs_ic_decay", "max_abs_corr"):
        production[key] = _bounded_number(production.get(key), f"delivery_policy.production.{key}", 0, 1)
    production["min_train_icir"] = _bounded_number(production.get("min_train_icir"), "delivery_policy.production.min_train_icir", -10, 20)
    production["min_val_ic_retention"] = _bounded_number(production.get("min_val_ic_retention"), "delivery_policy.production.min_val_ic_retention", 0, 2)
    if production.get("min_val_long_excess") is not None:
        production["min_val_long_excess"] = _bounded_number(
            production.get("min_val_long_excess"), "delivery_policy.production.min_val_long_excess", -1, 1
        )
    # 兼容旧 spec 的遗留键：存在则归一化（新默认值不再生成它们）
    for key, lo, hi in (
        ("min_abs_ic", 0, 1), ("min_icir", -10, 20),
        ("min_fmb_t_stat", 0, 20), ("min_ls_t_stat", 0, 20),
        ("min_quantile_excess_return", -1, 1), ("min_quantile_sharpe", -10, 20),
        ("min_monotonicity", -1, 1), ("min_long_group_annual_excess_return", -1, 1),
    ):
        if production.get(key) is not None:
            production[key] = _bounded_number(production.get(key), f"delivery_policy.production.{key}", lo, hi)
    eg = _require_dict(production.get("engine_gate"), "delivery_policy.production.engine_gate")
    production["engine_gate"] = eg
    for key in ("min_excess_annual",):
        eg[key] = _bounded_number(eg.get(key), f"engine_gate.{key}", -1, 5)
    for key in ("min_excess_sharpe", "max_drawdown", "min_daily_overlap"):
        if eg.get(key) is not None:
            eg[key] = _bounded_number(eg.get(key), f"engine_gate.{key}", 0, 10)
    if eg.get("min_invested_ratio") is not None:
        eg["min_invested_ratio"] = _bounded_number(eg.get("min_invested_ratio"), "engine_gate.min_invested_ratio", 0, 1)
    if eg.get("capital") is not None:
        eg["capital"] = _bounded_number(eg.get("capital"), "engine_gate.capital", 10_000, 1_000_000_000)
    if eg.get("min_am20_yuan") is not None:
        eg["min_am20_yuan"] = _bounded_number(eg.get("min_am20_yuan"), "engine_gate.min_am20_yuan", 0, 1_000_000_000_000)
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
    """从文件加载 ResearchSpec（None 时取默认+保存覆盖）。

    显式文件内容作"显式 spec"与注册表默认/保存覆盖合并（build_run_research_spec），
    保证 CLI 直跑也不会绕过用户在前端保存的门槛修改。
    """
    if path is None:
        return build_run_research_spec()
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"research_spec_load_failed: {exc}") from exc
    return build_run_research_spec(value)


def research_policy_prompt(spec: dict[str, Any]) -> str:
    """A concise policy contract consumed by the primary mining agent."""
    search = spec["search_policy"]
    evaluation = spec["evaluation_policy"]
    review = spec["review_policy"]
    interaction = spec.get("interaction_policy", {})
    profiles = spec.get("evaluation_profiles", {})
    mode = str(spec.get("research_mode") or "technical")
    mode_spec = get_research_mode(mode)
    needs_funda = mode_spec.needs_fundamentals
    mode_desc = (
        f"研究模式：{mode_spec.label}。信号以 `$funda_*` 财务字段为主"
        "（PIT 日频阶跃数据，注意披露生效日与低换手），"
        f"本次评估 label 为 {spec.get('recommended_label_col', mode_spec.recommended_label_col)}。"
        if needs_funda
        else f"研究模式：{mode_spec.label}。信号以价量/波动/筹码等行情字段为主。"
    )
    return "\n".join(
        [
            "# 本次运行的 ResearchSpec（高优先级研究约束）",
            mode_desc,
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

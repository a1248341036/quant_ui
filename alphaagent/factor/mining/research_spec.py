"""Versioned, runtime research policy for AlphaAgent factor mining."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from alphaagent.factor.evaluation.profile import default_evaluation_profiles, resolve_profiles
from alphaagent.factor.mining.interactions import INTERACTION_TYPES
from core import trading_config


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
    "delivery_policy": {
        "candidate": {
            "min_abs_ic": 0.015,
            # 海选 ICIR 与 train_screen 规则对齐（0.20 偏松，弱稳定因子堆积候选池）
            "min_icir": 0.25,
            "min_coverage": 0.85,
            "max_abs_corr": 0.5,
            # 换手可行性硬门槛：低于此值的因子截面排名日度剧变，不可交付
            "min_cs_autocorr": 0.18,
            # 样本外保留比：val_ic / train_ic 的绝对比值下限（方向反转直接拦截）
            "min_val_ic_retention": 0.5,
        },
        "production": {
            # ── 统计族：双窗口各自达标（混合窗口会稀释 val 衰减，2026-08 重构）──
            # train 窗口统计门槛
            "min_train_abs_ic": 0.025,
            "min_train_icir": 0.30,
            # val 窗口：绝对水平 + 相对 train 的保留比
            "min_val_abs_ic": 0.015,
            "min_val_ic_retention": 0.60,
            # val 多头端毛值超额（方向自适应十分组，复利年化）：
            # IC 为正不代表多头端赚钱——alpha 可能全在空头端/中段排名，
            # 纯多头可交易组合必须单独为正（2026-08 审计发现的 IC 盲区）
            "min_val_long_excess": 0.0,
            # 截尾 IC 衰减（全窗口口径）与去重
            "max_winsorized_abs_ic_decay": 0.10,
            "max_abs_corr": 0.4,
            # 已移除的摆设/失真门槛（研究结论 2026-08）：
            # - min_fmb_t_stat / min_ls_t_stat：t ≈ ICIR×√N，700+ 天下永不拦截
            # - min_quantile_excess_return / min_quantile_sharpe /
            #   min_monotonicity / min_long_group_annual_excess_return：
            #   毛值十分组口径系统性高估可交易性（同因子净值 weekly 净超额仅 ~4%）
            #   → 组合可行性全部交给 engine_gate 净值裁决。
            #
            # 进正式库前的最后一道门：旧交易引擎完整约束回测
            # （T+1/整手/涨跌停/停牌/费率/滑点/流动性），纯内存，不落盘。
            # 口径要点（2026-08）：alpha 的意义在超额而非绝对收益——样本含熊市时
            # 绝对年化/绝对夏普门会把所有因子拒之门外，故只设净值超额门 +
            # 超额夏普门；回撤与尾部稳定照旧。
            "engine_gate": {
                "enabled": True,
                # 动态百分比选股：自动适配停牌/涨跌停导致的候选池缩放。
                # 散户口径：top 0.4% ≈ 20只，统一配置在 core/trading_config.py。
                "selection_mode": trading_config.SELECTION_MODE,
                "selection_pct": trading_config.GATE_SELECTION_PCT,
                # weekly 为默认交付调仓频率：daily 全约束调仓的摩擦远大于因子超额。
                "freq": trading_config.GATE_FREQ,
                "allowed_freqs": ["daily", "weekly", "monthly"],
                # 净值超额年化下限（vs 全市场等权基准，扣全费）
                "min_excess_annual": trading_config.GATE_MIN_EXCESS_ANNUAL,
                # 超额夏普下限（active NAV 口径）
                "min_excess_sharpe": trading_config.GATE_MIN_EXCESS_SHARPE,
                "max_drawdown": trading_config.GATE_MAX_DRAWDOWN,
                "min_daily_overlap": trading_config.GATE_MIN_DAILY_OVERLAP,
                # 散户口径：10万资金，统一配置在 core/trading_config.py
                "capital": trading_config.GATE_CAPITAL,
                # 仓位利用率硬门：平均投入占比低于此值 = 执行不可行
                "min_invested_ratio": trading_config.GATE_MIN_INVESTED_RATIO,
                # 候选流动性下限：散户口径 500万
                "min_am20_yuan": trading_config.GATE_MIN_AM20_YUAN,
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
    # ── 基本面专属门槛（2026-08）：慢因子评估口径与 technical 差异化 ──
    # 季频 PIT 阶跃数据对 10d label 的信号强度天然弱于日频价量对 1d label，
    # 若沿用 technical 门槛会把真实基本面 alpha 全部拒之门外。
    # 设计原则：统计门槛适度放宽，但可交易性（engine_gate）只小幅放松，
    # 防止"统计有效但实盘不赚钱"的假因子进正式库。
    spec["evaluation_policy"].update({
        "min_train_abs_ic": 0.015,   # technical 0.02 → 基本面 0.015（慢因子弱信号）
        "min_train_icir": 0.22,      # technical 0.25 → 0.22（季频横截面少，ICIR 天然低）
        "min_val_abs_ic": 0.008,     # technical 0.01 → 0.008
        "min_val_ic_retention_ratio": 0.5,  # 保留比不变（防方向反转）
    })
    spec["delivery_policy"]["candidate"].update({
        "min_abs_ic": 0.012,         # technical 0.015 → 0.012（海选放宽松，让 reviewer 筛）
        "min_icir": 0.20,            # technical 0.25 → 0.20
        "min_cs_autocorr": 0.18,     # 保留通用换手性硬门（防排名日度剧变）
    })
    spec["delivery_policy"]["production"].update({
        "min_train_abs_ic": 0.020,   # technical 0.025 → 0.020
        "min_train_icir": 0.28,      # technical 0.30 → 0.28
        "min_val_abs_ic": 0.012,     # technical 0.015 → 0.012
        "min_val_ic_retention": 0.60,  # 保留
        "min_val_long_excess": 0.0,  # 保留（纯多头必须为正）
        "max_winsorized_abs_ic_decay": 0.12,  # technical 0.10 → 0.12（慢因子 IC 衰减本来就快）
    })
    # engine_gate：慢因子按周调仓摩擦过大 → 默认月频；超额/夏普门槛小幅放松，
    # 但回撤与仓位利用率保持原样（这些是执行可行性硬约束，与因子频率无关）。
    spec["delivery_policy"]["production"]["engine_gate"].update({
        "freq": "monthly",           # technical weekly → 基本面 monthly
        "min_excess_annual": 0.02,   # technical 0.03 → 0.02
        "min_excess_sharpe": 0.4,    # technical 0.5 → 0.4
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

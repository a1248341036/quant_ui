"""Immutable, versioned evaluation profiles."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationProfile:
    profile_id: str
    split: str
    transforms: tuple[dict[str, Any], ...] = ()
    metrics: tuple[dict[str, Any], ...] = ()
    rules: tuple[dict[str, Any], ...] = ()
    version: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "profile_id": self.profile_id,
            "split": self.split,
            "transforms": [copy.deepcopy(item) for item in self.transforms],
            "metrics": [copy.deepcopy(item) for item in self.metrics],
            "rules": [copy.deepcopy(item) for item in self.rules],
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


# 统一预处理：去极值 → 标准化（主口径）。
# 注意：**不做市值残差化**——主口径保持"能看见完整信号"，市值风险由
# size_neutral_decay 诊断字段暴露（见 _BASE_METRICS），市值中性化只存在于
# size_neutral_validation 诊断 profile。这保证因子实验室 / LLM 评估 /
# 入库指标三者同口径（与提交入门的 raw IC 门槛一致）。
_BASE_TRANSFORMS = (
    {"plugin": "cross_sectional_winsorize", "params": {"lower_pct": 1, "upper_pct": 99}},
    {"plugin": "cross_sectional_zscore"},
)

_BASE_METRICS = (
    {"plugin": "cross_sectional_core"},
    {"plugin": "monthly_robustness"},
    {"plugin": "mls_fmb"},
    {"plugin": "ic_series_diagnostics"},
    {"plugin": "long_short_portfolio", "params": {"groups": 10, "cost_bps": 0}},
    # 市值中性衰减诊断：只报告不设门槛，辅助判断因子是否靠小市值暴露
    {"plugin": "size_neutral_decay", "params": {"field": "float_cap"}},
    # 筛选阶段 quantile 组合用毛值口径（cost_bps=0，成本按换手计）；
    # 真实费率/滑点/约束由 delivery 的 engine_gate 完整回测把关。
    {"plugin": "quantile_portfolio", "params": {"groups": 10, "cost_bps": 0.0}},
)


def default_evaluation_profiles() -> dict[str, EvaluationProfile]:
    return {
        "train_screen": EvaluationProfile(
            "train_screen", "train",
            transforms=_BASE_TRANSFORMS, metrics=_BASE_METRICS,
        ),
        "validation": EvaluationProfile(
            "validation", "val",
            transforms=_BASE_TRANSFORMS, metrics=_BASE_METRICS,
        ),
        # 市值中性诊断 profile：真正做市值残差化，用于对比主口径，不参与门槛
        "size_neutral_validation": EvaluationProfile(
            "size_neutral_validation",
            "val",
            transforms=(
                {"plugin": "cross_sectional_winsorize", "params": {"lower_pct": 1, "upper_pct": 99}},
                {"plugin": "size_residualize", "params": {"field": "float_cap", "log": True}},
                {"plugin": "cross_sectional_zscore"},
            ),
            metrics=_BASE_METRICS,
        ),
        "production_delivery": EvaluationProfile(
            "production_delivery",
            "full",
            transforms=_BASE_TRANSFORMS,
            metrics=_BASE_METRICS,
        ),
    }


def _profile_from_mapping(profile_id: str, value: dict[str, Any]) -> EvaluationProfile:
    if not isinstance(value, dict):
        raise ValueError(f"evaluation_profile.{profile_id}_must_be_object")
    split = value.get("split")
    if split not in {"train", "val", "full"}:
        raise ValueError(f"evaluation_profile.{profile_id}.split_invalid")
    transforms = value.get("transforms", [])
    metrics = value.get("metrics", [])
    rules = value.get("rules", [])
    if not all(isinstance(row, dict) and isinstance(row.get("plugin"), str) for row in transforms):
        raise ValueError(f"evaluation_profile.{profile_id}.transforms_invalid")
    if not all(isinstance(row, dict) and isinstance(row.get("plugin"), str) for row in metrics):
        raise ValueError(f"evaluation_profile.{profile_id}.metrics_invalid")
    if not isinstance(rules, list) or not all(isinstance(row, dict) for row in rules):
        raise ValueError(f"evaluation_profile.{profile_id}.rules_invalid")
    return EvaluationProfile(
        profile_id=profile_id,
        split=split,
        transforms=tuple(copy.deepcopy(transforms)),
        metrics=tuple(copy.deepcopy(metrics)),
        rules=tuple(copy.deepcopy(rules)),
        version=int(value.get("version", 1)),
    )


def resolve_profiles(spec: dict[str, Any] | None = None) -> dict[str, EvaluationProfile]:
    """Merge `evaluation_profiles` overrides from ResearchSpec onto registered defaults."""
    defaults = default_evaluation_profiles()
    configured = (spec or {}).get("evaluation_profiles", {})
    if configured is None:
        return defaults
    if not isinstance(configured, dict):
        raise ValueError("research_spec.evaluation_profiles_must_be_object")
    raw_defaults = {key: value.as_dict() for key, value in defaults.items()}
    evaluation = (spec or {}).get("evaluation_policy", {})
    # 缺失键回落 canonical 默认（DEFAULT_RESEARCH_SPEC 唯一真源），避免本模块
    # 出现第二份门槛数值；懒加载打破 research_spec ↔ profile 的模块级循环依赖。
    from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC
    canonical_evaluation = DEFAULT_RESEARCH_SPEC["evaluation_policy"]
    canonical_production = DEFAULT_RESEARCH_SPEC["delivery_policy"]["production"]
    train_rules = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": evaluation.get("min_train_abs_ic", canonical_evaluation["min_train_abs_ic"])},
        {"metric": "cross_sectional_core.icir", "op": "abs_gte", "value": evaluation.get("min_train_icir", canonical_evaluation["min_train_icir"])},
        {"metric": "cross_sectional_core.factor_coverage", "op": "gte", "value": evaluation.get("min_train_coverage", canonical_evaluation["min_train_coverage"])},
    ]
    val_rules = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": evaluation.get("min_val_abs_ic", canonical_evaluation["min_val_abs_ic"])},
    ]
    raw_defaults["train_screen"]["rules"] = train_rules
    raw_defaults["validation"]["rules"] = val_rules
    raw_defaults["size_neutral_validation"]["rules"] = val_rules
    delivery = (spec or {}).get("delivery_policy", {})
    production = delivery.get("production", {})
    # production_delivery profile 只做 train 口径粗筛；train/val 双窗终审在
    # submit._check_stage_two，组合可行性净值裁决在 engine_gate。
    # 已移除 fmb/ls t 值规则（t≈ICIR×√N 在长样本上永不拦截）与毛值 quantile
    # 三项规则（十分组毛超额系统性高估可交易性）——研究结论见 research_spec。
    raw_defaults["production_delivery"]["rules"] = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": production.get("min_train_abs_ic", canonical_production["min_train_abs_ic"])},
        {"metric": "cross_sectional_core.icir", "op": "abs_gte", "value": production.get("min_train_icir", canonical_production["min_train_icir"])},
    ]
    for profile_id, override in configured.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("research_spec.evaluation_profiles_id_invalid")
        if not isinstance(override, dict):
            raise ValueError(f"research_spec.evaluation_profiles.{profile_id}_must_be_object")
        base = raw_defaults.get(profile_id, {"profile_id": profile_id, "version": 1})
        raw_defaults[profile_id] = {**base, **copy.deepcopy(override), "profile_id": profile_id}
    return {profile_id: _profile_from_mapping(profile_id, value) for profile_id, value in raw_defaults.items()}

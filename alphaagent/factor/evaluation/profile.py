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


_BASE_METRICS = (
    {"plugin": "cross_sectional_core"},
    {"plugin": "monthly_robustness"},
    {"plugin": "mls_fmb"},
    {"plugin": "ic_series_diagnostics"},
    {"plugin": "long_short_portfolio", "params": {"groups": 10, "cost_bps": 0}},
)


def default_evaluation_profiles() -> dict[str, EvaluationProfile]:
    return {
        "train_screen": EvaluationProfile("train_screen", "train", metrics=_BASE_METRICS),
        "validation": EvaluationProfile("validation", "val", metrics=_BASE_METRICS),
        "size_neutral_validation": EvaluationProfile(
            "size_neutral_validation",
            "val",
            transforms=(
                {"plugin": "size_residualize", "params": {"field": "float_cap", "log": True}},
                {"plugin": "cross_sectional_zscore"},
            ),
            metrics=_BASE_METRICS,
        ),
        "production_delivery": EvaluationProfile(
            "production_delivery",
            "full",
            transforms=(
                {"plugin": "cross_sectional_winsorize", "params": {"lower_pct": 1, "upper_pct": 99}},
                {"plugin": "size_residualize", "params": {"field": "float_cap", "log": True}},
            ),
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
    train_rules = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": evaluation.get("min_train_abs_ic", 0.015)},
        {"metric": "cross_sectional_core.icir", "op": "gte", "value": evaluation.get("min_train_icir", 0.2)},
        {"metric": "cross_sectional_core.factor_coverage", "op": "gte", "value": evaluation.get("min_train_coverage", 0.85)},
    ]
    val_rules = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": evaluation.get("min_val_abs_ic", 0.01)},
    ]
    raw_defaults["train_screen"]["rules"] = train_rules
    raw_defaults["validation"]["rules"] = val_rules
    raw_defaults["size_neutral_validation"]["rules"] = val_rules
    delivery = (spec or {}).get("delivery_policy", {})
    production = delivery.get("production", {})
    raw_defaults["production_delivery"]["rules"] = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": production.get("min_abs_ic", 0.035)},
        {"metric": "cross_sectional_core.icir", "op": "gte", "value": production.get("min_icir", 0.5)},
        {"metric": "mls_fmb.nw_t_ls", "op": "abs_gte", "value": production.get("min_fmb_t_stat", 2.5)},
        {"metric": "long_short_portfolio.long_group_annual_excess_return", "op": "gte", "value": production.get("min_long_group_annual_excess_return", 0.03)},
    ]
    for profile_id, override in configured.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("research_spec.evaluation_profiles_id_invalid")
        if not isinstance(override, dict):
            raise ValueError(f"research_spec.evaluation_profiles.{profile_id}_must_be_object")
        base = raw_defaults.get(profile_id, {"profile_id": profile_id, "version": 1})
        raw_defaults[profile_id] = {**base, **copy.deepcopy(override), "profile_id": profile_id}
    return {profile_id: _profile_from_mapping(profile_id, value) for profile_id, value in raw_defaults.items()}

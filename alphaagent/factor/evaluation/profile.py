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
        # 海选轻筛 profile：只跑 cross_sectional_core（门槛规则唯一数据源），
        # 供挖掘 eval_train 两段式第一段使用——没过线的因子跳过 fmb/组合回测/
        # 月度稳健性等纯诊断件（实测占单次评估 GIL 时间的 ~2/3，是并发吞吐
        # 瓶颈）。规则与 train_screen 完全同源；decile_mean_label 仍在 core 内，
        # 预测对账不受影响。诊断数据由第二段全量评估补齐。
        "train_screen_lite": EvaluationProfile(
            "train_screen_lite", "train",
            transforms=_BASE_TRANSFORMS,
            metrics=({"plugin": "cross_sectional_core"},),
        ),
        # 换手率预筛 profile：lite 过线后、全量之前单独跑，只算 quantile_portfolio
        # 的 avg_daily_side_turnover（depth_ks=[] 关掉深度曲线、tradable_domain=False
        # 关掉可成交域透镜第二路调用，只保留主口径 ~2-3s）。换手率 > 0.5 直接短路，
        # 省掉全量 fmb/monthly/long_short 等剩余 ~7s 诊断件。实测 lite 过线因子 89%
        # 换手率 > 0.5，此预筛能短路绝大多数全量评估。
        "train_screen_turnover": EvaluationProfile(
            "train_screen_turnover", "train",
            transforms=_BASE_TRANSFORMS,
            metrics=({"plugin": "quantile_portfolio", "params": {"groups": 10, "cost_bps": 0.0, "depth_ks": [], "tradable_domain": False}},),
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
    # 缺失键回落 canonical 默认（evaluation/defaults.py 单一真源，与
    # research_spec.DEFAULT_RESEARCH_SPEC 同值），避免本模块出现第二份门槛数值，
    # 也避免 evaluation → mining 的分层倒置。
    from alphaagent.factor.evaluation.defaults import (
        DEFAULT_EVALUATION_POLICY,
        DEFAULT_PRODUCTION_THRESHOLDS,
    )

    canonical_evaluation = DEFAULT_EVALUATION_POLICY
    canonical_production = DEFAULT_PRODUCTION_THRESHOLDS
    train_rules = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": evaluation.get("min_train_abs_ic", canonical_evaluation["min_train_abs_ic"])},
        {"metric": "cross_sectional_core.icir", "op": "abs_gte", "value": evaluation.get("min_train_icir", canonical_evaluation["min_train_icir"])},
        {"metric": "cross_sectional_core.factor_coverage", "op": "gte", "value": evaluation.get("min_train_coverage", canonical_evaluation["min_train_coverage"])},
    ]
    val_rules = [
        {"metric": "cross_sectional_core.ic", "op": "abs_gte", "value": evaluation.get("min_val_abs_ic", canonical_evaluation["min_val_abs_ic"])},
    ]
    raw_defaults["train_screen"]["rules"] = train_rules
    # lite 与 train_screen 同一套规则（规则只依赖 cross_sectional_core）
    raw_defaults["train_screen_lite"]["rules"] = train_rules
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

    # ── 行业/风格轮动 P1 诊断 metrics（2026-10-12）──────────────────────────
    # rotation_policy.enabled 为前提：report_industry_metrics 挂行业组指标；
    # report_state_metrics 挂状态条件 IC（独立开关）。只追加不覆盖（用户
    # override 的 metrics 保留），train_screen_lite 不挂（吞吐敏感）。
    # 这些 metric 只写评估输出，不参与任何准入规则。
    rotation = (spec or {}).get("rotation_policy") or {}
    if rotation.get("enabled"):
        # state_conditional_ic 只消费 canonical style_state 列 + min_cover，
        # 不读 field/days；industry_field/days 仅行业组指标消费，故只在
        # report_industry_metrics 分支赋值，避免条件性死代码。
        _rcover = int(rotation.get("min_industry_cover") or 15)
        _rotation_metrics: list[dict[str, Any]] = []
        if rotation.get("report_industry_metrics"):
            _rfield = str(rotation.get("industry_field") or "industry_sw_l1")
            _rdays = int(rotation.get("min_industry_days") or 20)
            _rotation_metrics += [
                {"plugin": "industry_within_rank_ic", "params": {"field": _rfield, "min_cover": _rcover}},
                # exposure_r2 不做 min_cover 过滤（暴露分解需计入全部股票，见插件 docstring）
                {"plugin": "industry_exposure_r2", "params": {"field": _rfield}},
                {"plugin": "industry_ic_dispersion", "params": {"field": _rfield, "min_cover": _rcover, "min_days": _rdays}},
            ]
        if rotation.get("report_state_metrics"):
            # 状态条件 IC：只消费 panel 恒载 canonical style_state 列（§3.4.1），
            # 只报告、不参与准入（§4.2）。panel 缺列时 metric 自报告 missing，不报错。
            _rotation_metrics.append(
                {"plugin": "state_conditional_ic", "params": {"field": "style_state", "min_cover": _rcover}}
            )
        if _rotation_metrics:
            for _pid in ("train_screen", "validation", "production_delivery"):
                if _pid in raw_defaults:
                    prev = list(raw_defaults[_pid].get("metrics") or [])
                    # 幂等：生产链路会把前端回传/续跑冻结的 evaluation_profiles 再走一遍
                    # build_run_research_spec → resolve_profiles，此时 prev 已含轮动项；
                    # 按 plugin 名去重，避免反复追加导致 profile 指纹漂移。
                    existing = {m.get("plugin") for m in prev if isinstance(m, dict)}
                    raw_defaults[_pid]["metrics"] = prev + [
                        m for m in _rotation_metrics if m.get("plugin") not in existing
                    ]
    return {profile_id: _profile_from_mapping(profile_id, value) for profile_id, value in raw_defaults.items()}

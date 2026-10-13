# -*- coding: utf-8 -*-
"""rotation_strategy 产物 schema 与结构校验（Phase 1）。

「LLM 交付结构化策略设计」的产物 schema 定义 + `validate_design`。
这里只做**结构**校验（字段存在性、类型、白名单、范围）；状态枚举全集 /
carrier 数据体检 / mode 因子族校验 / 最坏换手预门禁属于编译期判定，在
``core/rotation/compiler.py``。校验失败的报错信息一律附带合法全集，供 LLM 自纠。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# ── 合法枚举全集（schema 文档的代码真源；compiler 报错时引同一常量）─────
DIMS: frozenset[str] = frozenset(
    {"size", "value_growth", "momentum", "market_regime"}
)
CARRIERS: frozenset[str] = frozenset(
    {
        "style_factor",
        "index_spread",
        "industry_momentum",
        "margin_chg",
        "northbound_chg",
        "macro_level",
    }
)
THRESHOLD_MODES: frozenset[str] = frozenset({"sign", "quantile", "level"})
# 分位参考窗：rolling 用 window_days，expanding 全历史——均只含 T-1 及之前。
QUANTILE_REFS: frozenset[str] = frozenset({"rolling", "expanding"})

# 参数范围（spec §3.2；research_spec 三件套风格，编译期与这里共用同一边界）
WINDOW_DAYS_MIN, WINDOW_DAYS_MAX = 20, 500
BOOST_MIN, BOOST_MAX = 1 / 1.25, 1.25
MIN_STATE_DURATION_MIN = 2
MAX_DAILY_TURNOVER_MIN, MAX_DAILY_TURNOVER_MAX = 0.05, 0.5
QUANTILE_Q_MIN, QUANTILE_Q_MAX = 0.05, 0.95
LEVEL_SIGNAL_MIN, LEVEL_SIGNAL_MAX = -1e6, 1e6

# level 模式只允许用于跨期可比量（变化率/分位），禁用于指数水平等水平量。
LEVEL_ALLOWED_CARRIERS: frozenset[str] = frozenset(
    {"margin_chg", "northbound_chg"}
)


def legal_values() -> dict[str, set[str]]:
    """合法枚举全集（CompiledStrategy / 报错信息 / rotation_guide 共用）。"""
    return {
        "dim": set(DIMS),
        "carrier": set(CARRIERS),
        "threshold_mode": set(THRESHOLD_MODES),
        "quantile_benchmark": set(QUANTILE_REFS),
    }


@dataclass
class ValidationReport:
    """validate_design 的完整结果（ok/errors/warnings + 合法全集）。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    legal: dict[str, set[str]] = field(default_factory=legal_values)

    def message(self) -> str:
        if self.ok:
            return "design ok"
        lines = list(self.errors)
        lines.append("合法取值全集: " + _format_legal(self.legal))
        return "\n".join(lines)


def _format_legal(legal: dict[str, set[str]]) -> str:
    return "; ".join(
        f"{key}={{{','.join(sorted(vals))}}}" for key, vals in sorted(legal.items())
    )


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _in_range(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def validate_design(obj: Any) -> ValidationReport:
    """结构校验一个 rotation_strategy 设计 JSON（不做数据面判定）。"""
    rep = ValidationReport(ok=True)
    if not isinstance(obj, dict):
        rep.ok = False
        rep.errors.append("design 必须是 JSON 对象")
        return rep

    # version
    version = obj.get("version", SCHEMA_VERSION)
    if not _is_number(version) or version != int(version) or int(version) < 1:
        rep.errors.append(f"version 必须为 >=1 的整数，收到 {version!r}")
    elif int(version) > SCHEMA_VERSION:
        rep.errors.append(
            f"version={version} 高于当前编译器支持的 {SCHEMA_VERSION}；"
            "请迁移到 v2（见 schema 演进：旧设计保留可编译，新字段走新编译器）"
        )

    # thesis
    thesis = obj.get("thesis", "")
    if not isinstance(thesis, str) or not thesis.strip():
        rep.errors.append("thesis 必填：描述状态区有效的经济直觉（供评审与 memory）")

    # evidence：不接受空串/玄学描述
    evidence = obj.get("evidence", "")
    if not isinstance(evidence, str) or not evidence.strip():
        rep.errors.append("evidence 必填且非空：可验证的支撑数据段或回测观察")
    elif len(evidence) < 8:
        rep.errors.append("evidence 过短：请给出可验证的支撑描述（>=8 字符）")

    # allocation（Phase 3 预留，本期必须为 null/缺失）
    allocation = obj.get("allocation")
    if allocation not in (None,):
        rep.warnings.append("allocation 为 P3 预留字段，本期忽略（不参与编译/状态/门禁）")

    # state_dimensions
    dims = obj.get("state_dimensions")
    if not isinstance(dims, list) or not dims:
        rep.errors.append("state_dimensions 必须是非空数组")
    else:
        seen_dims: set[str] = set()
        for i, dim_spec in enumerate(dims):
            _validate_dimension(dim_spec, seen_dims, rep, index=i)

    # quantile_benchmark：任何维度 threshold_mode=quantile 时必填
    if isinstance(dims, list) and any(
        isinstance(d, dict) and d.get("threshold_mode") == "quantile" for d in dims
    ):
        qref = obj.get("quantile_benchmark")
        if not isinstance(qref, str) or qref not in QUANTILE_REFS:
            rep.errors.append(
                "threshold_mode=quantile 必须声明量化参考窗 "
                f"quantile_benchmark ∈ {sorted(QUANTILE_REFS)}"
                "（禁止全期分位；rolling/expanding 均只含 T-1 及之前）"
            )
    else:
        qref = obj.get("quantile_benchmark")
        if qref is not None and (not isinstance(qref, str) or qref not in QUANTILE_REFS):
            rep.errors.append(
                f"quantile_benchmark 必须是 {sorted(QUANTILE_REFS)} 之一，收到 {qref!r}"
            )

    # mapping
    mapping = obj.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        rep.errors.append("mapping 必须是非空对象（状态 -> 因子族软权重）")
    else:
        for state, boosts in mapping.items():
            if not isinstance(state, str) or not state:
                rep.errors.append(f"mapping 状态键必须是非空字符串，收到 {state!r}")
                continue
            if not isinstance(boosts, dict):
                rep.errors.append(
                    f"mapping[{state}] 必须是对象 {{family: boost}}"
                    "（可为空 = 该状态不调制，回退 baseline 1.0）"
                )
                continue
            for family, boost in boosts.items():
                if not isinstance(family, str) or not family:
                    rep.errors.append(f"mapping[{state}] 因子族键必须是非空字符串（收到 {family!r}）")
                    continue
                if not _is_number(boost) or not _in_range(
                    float(boost), BOOST_MIN, BOOST_MAX
                ):
                    rep.errors.append(
                        f"mapping[{state}][{family}] boost 必须在 "
                        f"[{BOOST_MIN:.4f}, {BOOST_MAX}] 内，收到 {boost!r}"
                    )

    # guardrails
    guard = obj.get("guardrails")
    if not isinstance(guard, dict):
        rep.errors.append("guardrails 必须是非空对象")
    else:
        md = guard.get("min_state_duration_days")
        if not _is_number(md) or int(md) < MIN_STATE_DURATION_MIN:
            rep.errors.append(
                f"guardrails.min_state_duration_days 必须 >= {MIN_STATE_DURATION_MIN}，"
                f"收到 {md!r}"
            )
        mto = guard.get("max_daily_turnover")
        if not _is_number(mto) or not _in_range(
            float(mto), MAX_DAILY_TURNOVER_MIN, MAX_DAILY_TURNOVER_MAX
        ):
            rep.errors.append(
                "guardrails.max_daily_turnover 必须在 "
                f"[{MAX_DAILY_TURNOVER_MIN}, {MAX_DAILY_TURNOVER_MAX}] 内，收到 {mto!r}"
            )
        if guard.get("no_lookahead", True) is not True:
            rep.errors.append("guardrails.no_lookahead 必须为 true（编译器强制无前视）")

    rep.ok = not rep.errors
    return rep


def _validate_dimension(
    dim_spec: Any, seen_dims: set[str], rep: ValidationReport, *, index: int
) -> None:
    tag = f"state_dimensions[{index}]"
    if not isinstance(dim_spec, dict):
        rep.errors.append(f"{tag} 必须是对象")
        return
    dim = dim_spec.get("dim")
    if not isinstance(dim, str) or dim not in DIMS:
        rep.errors.append(f"{tag}.dim 必须是 {sorted(DIMS)} 之一，收到 {dim!r}")
    elif dim in seen_dims:
        rep.errors.append(f"{tag}.dim={dim!r} 重复定义（每维只允许一个）")
    else:
        seen_dims.add(dim)
    carrier = dim_spec.get("carrier")
    if not isinstance(carrier, str) or carrier not in CARRIERS:
        rep.errors.append(f"{tag}.carrier 必须是 {sorted(CARRIERS)} 之一，收到 {carrier!r}")
    wd = dim_spec.get("window_days")
    if not _is_number(wd) or not _in_range(wd, WINDOW_DAYS_MIN, WINDOW_DAYS_MAX):
        rep.errors.append(
            f"{tag}.window_days 必须在 [{WINDOW_DAYS_MIN}, {WINDOW_DAYS_MAX}] 内，"
            f"收到 {wd!r}"
        )
    mode = dim_spec.get("threshold_mode")
    if not isinstance(mode, str) or mode not in THRESHOLD_MODES:
        rep.errors.append(f"{tag}.threshold_mode 必须是 {sorted(THRESHOLD_MODES)} 之一")
    params = dim_spec.get("params")
    if not isinstance(params, dict):
        rep.errors.append(f"{tag}.params 必须是对象")
    else:
        if mode == "quantile":
            q = params.get("q", 0.5)
            if not _is_number(q) or not _in_range(q, QUANTILE_Q_MIN, QUANTILE_Q_MAX):
                rep.errors.append(
                    f"{tag}.params.q 必须在 [{QUANTILE_Q_MIN}, {QUANTILE_Q_MAX}] 内，"
                    f"收到 {q!r}"
                )
        elif mode == "level":
            lv = params.get("threshold", 0.0)
            if not _is_number(lv) or not _in_range(
                float(lv), LEVEL_SIGNAL_MIN, LEVEL_SIGNAL_MAX
            ):
                rep.errors.append(
                    f"{tag}.params.threshold 必须在 [{LEVEL_SIGNAL_MIN}, "
                    f"{LEVEL_SIGNAL_MAX}] 内，收到 {lv!r}"
                )
            if carrier not in LEVEL_ALLOWED_CARRIERS:
                rep.errors.append(
                    f"{tag} level 模式只允许用于平稳量（变化率/分位），carrier 必须是 "
                    f"{sorted(LEVEL_ALLOWED_CARRIERS)} 之一，收到 {carrier!r}；"
                    "index_spread 等水平量请先强制分位化（改用 quantile）"
                )

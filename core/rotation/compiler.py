# -*- coding: utf-8 -*-
"""确定性编译器/校验器：rotation_strategy 设计 -> CompiledStrategy（Phase 1）。

职责（spec §3.2 / §2）：
- 结构校验之上补编译期判定：状态枚举全集、mode 因子族、level 限平稳量（与 schema
  双保险）、carrier 数据体检（注入式，不硬编码数据面）、最坏换手预门禁；
- 产出 canonical JSON（键排序 + 浮点规范化）→ 内容 SHA-256 → strategy_id；
- compile_fp = hash(compiler_version, schema_version, style_regime_version)，
  状态层/DSL 变化即失效（§3.10 三层指纹链的 design/compile 两指纹先行）。

无前视纪律（§3.2）：quantile 只允许 rolling/expanding 参考窗（均只含 T-1 及以前），
禁止全期分位；状态判定本身在 style_regime 层实现，同样不越窗。
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from alphaagent.factor.mining.rotation.schema import SCHEMA_VERSION, validate_design

COMPILER_VERSION = "1"
STYLE_REGIME_VERSION = "1"

# carrier 数据体检版本（数据面判定逻辑变化时 bump，进入 compile_fp）
CARRIER_PROBE_VERSION = "1"


# ── (dim, carrier) → 载体信号列 解析表（Phase 1 可编译组合全集）──────────
# 信号列名与 style_regime.build_canonical_signals 产出的列一致；未出现在
# 载体信号面板中的列由 carrier 体检判定 unavailable（数据面断供）拒绝。
_SIGNAL_RESOLUTION: dict[tuple[str, str], str] = {
    ("size", "style_factor"): "size_proxy",
    ("size", "index_spread"): "size_index_spread",
    ("value_growth", "style_factor"): "value_growth_spread",
    ("value_growth", "index_spread"): "value_growth_index_spread",
    ("momentum", "style_factor"): "momentum_beta",
    ("momentum", "industry_momentum"): "industry_momentum",
    ("market_regime", "style_factor"): "market_index_momentum",
    ("market_regime", "index_spread"): "index_spread",
    ("market_regime", "industry_momentum"): "industry_momentum",
    ("market_regime", "margin_chg"): "margin_chg",
    ("market_regime", "northbound_chg"): "northbound_chg",
    ("market_regime", "macro_level"): "macro_level",
}


# ── 细因子族 -> research_modes.signal_families 词汇（mode 因子族校验）────
_FINE_TO_SIGNAL: dict[str, str] = {
    "momentum": "momentum_reversal",
    "reversal": "momentum_reversal",
    "gap_overnight": "momentum_reversal",
    "volatility": "volatility",
    "volume": "volume_price",
    "liquidity": "volume_price",
    "breadth": "volume_price",
    "vwap": "volume_price",
    "size": "volume_price",
    "crowd": "volume_price",
    "correlation": "correlation",
    "chip": "chip",
    "value": "fundamental_value",
    "quality": "fundamental_quality",
    "growth": "fundamental_growth",
    "revision": "fundamental_revision",
}


# 每个状态维度的高/低桶名（枚举命名确定性，spec §3.2）。low=阈值以下，
# high=阈值及以上；多维权组合名按 state_dimensions 顺序拼接。
DIM_BUCKETS: dict[str, tuple[str, str]] = {
    "size": ("SMALL", "LARGE"),
    "value_growth": ("GROWTH", "VALUE"),
    "momentum": ("SLOW", "MOMENTUM"),
    "market_regime": ("RISK_OFF", "RISK_ON"),
}


@dataclass(frozen=True)
class CompiledDimension:
    dim: str
    carrier: str
    signal: str
    window_days: int
    threshold_mode: str
    params: dict[str, Any]
    quantile_benchmark: str = ""


@dataclass(frozen=True)
class CompiledStrategy:
    strategy_id: str
    design_fp: str
    compile_fp: str
    version: int
    mode: str
    thesis: str
    evidence: str
    dimensions: tuple[CompiledDimension, ...]
    mapping: Mapping[str, Mapping[str, float]]
    guardrails: Mapping[str, Any]
    quantile_benchmark: str
    state_enums: tuple[str, ...]
    families_used: frozenset[str]
    worst_case_turnover: float
    carrier_health_reported: bool
    blind_start: str | None
    canonical: dict[str, Any] = field(repr=False)

    @property
    def fingerprints(self) -> dict[str, str]:
        return {
            "strategy_id": self.strategy_id,
            "design": self.design_fp,
            "compile": self.compile_fp,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "fingerprints": self.fingerprints,
            "version": self.version,
            "mode": self.mode,
            "thesis": self.thesis,
            "evidence": self.evidence,
            "dimensions": [
                {
                    "dim": d.dim,
                    "carrier": d.carrier,
                    "signal": d.signal,
                    "window_days": d.window_days,
                    "threshold_mode": d.threshold_mode,
                    "params": d.params,
                    "quantile_benchmark": d.quantile_benchmark,
                }
                for d in self.dimensions
            ],
            "quantile_benchmark": self.quantile_benchmark,
            "mapping": {k: dict(v) for k, v in self.mapping.items()},
            "guardrails": dict(self.guardrails),
            "state_enums": list(self.state_enums),
            "families_used": sorted(self.families_used),
            "worst_case_turnover": self.worst_case_turnover,
            "carrier_health_reported": self.carrier_health_reported,
            "blind_start": self.blind_start,
            "canonical": self.canonical,
        }


def default_blind_start() -> str:
    """盲测段起点 = val 段结束次日（源自 window_config 单一真源，不硬编码日期）。"""

    from alphaagent.factor.window_config import DEFAULT_VAL_END

    return (pd.Timestamp(DEFAULT_VAL_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# ── canonical 规范化 ───────────────────────────────────────────────────
def _canonicalize(value: Any, *, _depth: int = 0) -> Any:
    """递归规范化 JSON：dict 键排序、float 12 位取整、tuple->list，保证哈希稳定。"""

    if isinstance(value, Mapping):
        if _depth > 50:
            raise ValueError("design 嵌套过深")
        return {
            str(k): _canonicalize(v, _depth=_depth + 1)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v, _depth=_depth + 1) for v in value]
    if isinstance(value, float):
        return round(float(value), 12)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    return value


def _content_hash(canonical: dict[str, Any]) -> str:
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_strategy_id(design_fp: str) -> str:
    """设计指纹派生 strategy_id：改任一设计字段 => 新 hash => 新 id（同设计永远同 id）。"""

    return f"rot_{design_fp[:8]}"


def compile_fingerprint() -> str:
    payload = ":".join(
        [COMPILER_VERSION, str(SCHEMA_VERSION), STYLE_REGIME_VERSION, CARRIER_PROBE_VERSION]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_signal(dim: str, carrier: str) -> str | None:
    """(dim, carrier) -> 载体信号列名；无匹配返回 None（编译期直接拒）。"""

    return _SIGNAL_RESOLUTION.get((dim, carrier))


# carrier -> 其可解析的信号列全集（carrier 数据体检用：任一列可用即算可用）
CARRIER_SIGNALS: dict[str, tuple[str, ...]] = {
    "style_factor": (
        "size_proxy",
        "value_growth_spread",
        "momentum_beta",
        "market_index_momentum",
    ),
    "index_spread": (
        "size_index_spread",
        "value_growth_index_spread",
        "index_spread",
    ),
    "industry_momentum": ("industry_momentum",),
    "margin_chg": ("margin_chg",),
    "northbound_chg": ("northbound_chg",),
    "macro_level": ("macro_level",),
}


def probe_carrier_health(
    signals: pd.DataFrame,
    blind_start: str,
    *,
    min_valid_days: int = 20,
) -> dict[str, dict[str, Any]]:
    """由载体信号面板生成 carrier 数据体检（动态白名单，spec §3.2）。

    - available：该 carrier 至少一个信号列存在且全期有效日 >= min_valid_days；
    - blind_coverage：盲测段（>= blind_start）内该 carrier 存在非 NaN 观测。
    缺列/断供（如北向 2024-08 后停更、margin T+1 延迟）在此暴露并被编译期拒绝。
    """

    idx = (signals.index if hasattr(signals, "index") else None)
    blind_mask = None
    if blind_start and idx is not None:
        try:
            blind_dates = pd.to_datetime(idx)
            blind_mask = blind_dates >= pd.Timestamp(blind_start)
        except Exception:  # noqa: BLE001 — 日期索引不规范则盲测段检查跳过
            blind_mask = None

    out: dict[str, dict[str, Any]] = {}
    for carrier, sig_cols in CARRIER_SIGNALS.items():
        present = [c for c in sig_cols if c in signals.columns]
        ok_cols = [
            c for c in present
            if int(signals[c].notna().sum()) >= max(int(min_valid_days), 1)
        ]
        available = bool(ok_cols)
        health: dict[str, Any] = {"available": available}
        if not available:
            health["reason"] = (
                f"载体信号 {present or list(sig_cols)} 全期有效日 "
                f"< {min_valid_days} 或不存在（数据面断供/未接入）"
            )
        if available and blind_mask is not None:
            segments = [
                signals[c].to_numpy(copy=False)[blind_mask] for c in ok_cols
            ]
            nonempty = [seg for seg in segments if len(seg) > 0]
            health["blind_coverage"] = (
                any(bool(pd.Series(seg).notna().any()) for seg in nonempty)
                if nonempty
                else False
            )
        out[carrier] = health
    return out


# ── 状态枚举命名（确定性）───────────────────────────────────────────────
def state_enum_names(dims: tuple[CompiledDimension, ...]) -> tuple[str, ...]:
    """状态枚举全集 = 各维高/低桶名的笛卡尔积（按 state_dimensions 顺序拼接）。"""

    buckets: list[tuple[str, str]] = []
    for d in dims:
        try:
            buckets.append(DIM_BUCKETS[d.dim])
        except KeyError as exc:
            raise ValueError(f"未知状态维度 {d.dim!r}（无法命名枚举）") from exc
    if not buckets:
        return ("UNKNOWN",)
    enums: list[str] = []
    for combo in itertools.product(*buckets):
        enums.append("_".join(combo))
    return tuple(sorted(enums))


# ── mode 因子族校验 ─────────────────────────────────────────────────────
def available_families(mode: str) -> frozenset[str]:
    """目标 mode 可解析的细因子族集合（从 research_modes.signal_families 派生）。

    mapping 引用的族必须是该 mode composite 能解析的族，缺族即报错
    （不静默忽略、不权重退化）。同一设计换 mode = 不同可编译产物。
    """

    from core.research_modes import get_research_mode

    allowed = set(get_research_mode(mode).signal_families)
    out = {
        fine
        for fine, signal_family in _FINE_TO_SIGNAL.items()
        if signal_family in allowed
    }
    return frozenset(out)


# ── 最坏换手预门禁（spec §3.2 / A3）─────────────────────────────────────
def worst_case_turnover(
    *,
    min_state_duration_days: int,
    mapping: Mapping[str, Mapping[str, float]],
    state_enums: tuple[str, ...],
) -> float:
    """最坏日均换手上界 = 切换频率上界 x 单次切换最大权重变化。

    切换频率上界 = 1 / min_state_duration_days（最短持续 => 最频繁切换；
    下界恒为 0 无预门禁意义）；max_pair_delta = 任意状态对权重差向量 L1 的最大值。
    """

    max_pair_delta = 0.0
    for s1, s2 in itertools.combinations(state_enums, 2):
        b1 = mapping.get(s1) or {}
        b2 = mapping.get(s2) or {}
        families = set(b1) | set(b2)
        delta = sum(abs(float(b2.get(f, 1.0)) - float(b1.get(f, 1.0))) for f in families)
        max_pair_delta = max(max_pair_delta, delta)
    return (1.0 / int(min_state_duration_days)) * max_pair_delta


# ── carrier 数据体检（动态白名单；注入式，数据面判定不硬编码）───────────
def check_carrier_health(
    dims: tuple[CompiledDimension, ...],
    *,
    carrier_health: dict[str, dict[str, Any]] | None,
    blind_start: str | None,
) -> tuple[bool, list[str], bool]:
    """返回 (ok, errors, reported)。carrier_health=None 时跳过数据面判定。

    carrier_health 每项：{"available": bool, "reason": str, "blind_coverage": bool}。
    已知数据面陷阱（spec §3.2）：northbound_chg 2024-08 后日度停更、
    margin_chg T+1 披露延迟、macro_level 发布滞后/PIT——均在该体检层暴露。
    """

    if carrier_health is None:
        return True, [], False
    used = {d.carrier for d in dims}
    missing = used - set(carrier_health)
    if missing:
        return (
            False,
            [f"carrier 无体检数据: {sorted(missing)}（请先跑 carrier 数据体检并注入可用集）"],
            True,
        )
    errors: list[str] = []
    for carrier in sorted(used):
        health = carrier_health[carrier] or {}
        if not health.get("available", True):
            errors.append(
                f"carrier {carrier} 不可用: {health.get('reason', '数据源断供')}"
            )
            continue
        if blind_start and health.get("blind_coverage") is False:
            errors.append(
                f"carrier {carrier} 在盲测段（{blind_start} 起）无数据覆盖，"
                "该状态在后半段会退化为平线，编译期拒绝"
            )
    return (not errors), errors, True


# ── 主入口 ──────────────────────────────────────────────────────────────
def compile_design(
    design: dict[str, Any],
    *,
    mode: str = "technical",
    carrier_health: dict[str, dict[str, Any]] | None = None,
    blind_start: str | None = None,
) -> CompiledStrategy:
    """编译一个 rotation_strategy 设计为确定性规则对象（spec §3.2）。

    失败抛 ValueError，错误信息附带合法枚举全集（LLM 可自纠）。
    carrier_health 由调用方（submit 工具 / harness）先做数据体检后注入；
    为 None 表示纯设计校验路径，数据面判定标记为未报告。
    """

    report = validate_design(design)
    if not report.ok:
        raise ValueError(f"rotation_strategy 校验失败:\n{report.message()}")

    if blind_start is None:
        blind_start = default_blind_start()

    version = int(design.get("version", SCHEMA_VERSION))
    dims: list[CompiledDimension] = []
    for spec in design["state_dimensions"]:
        dim = spec["dim"]
        carrier = spec["carrier"]
        signal = _SIGNAL_RESOLUTION.get((dim, carrier))
        if signal is None:
            legal_pairs = ", ".join(
                f"{d}×{c}" for d, c in sorted(_SIGNAL_RESOLUTION)
            )
            raise ValueError(
                f"(dim={dim}, carrier={carrier}) 无可用载体信号；"
                f"Phase 1 可编译组合: {legal_pairs}"
            )
        qref = ""
        if spec["threshold_mode"] == "quantile":
            qref = str(design.get("quantile_benchmark") or "")
        dims.append(
            CompiledDimension(
                dim=dim,
                carrier=carrier,
                signal=signal,
                window_days=int(spec["window_days"]),
                threshold_mode=spec["threshold_mode"],
                params=dict(spec.get("params") or {}),
                quantile_benchmark=qref,
            )
        )

    state_enums = state_enum_names(tuple(dims))

    # mapping：键必须在状态枚举全集内；族必须在目标 mode 可解析集合内
    mapping: dict[str, dict[str, float]] = {}
    families_used: set[str] = set()
    mode_families = available_families(mode)
    for state, boosts in design["mapping"].items():
        if state not in state_enums:
            raise ValueError(
                f"mapping 状态 {state!r} 不在状态枚举全集 {state_enums} 内；"
                "mapping 只允许覆盖全集子集，空态按 boost=1.0 处理"
            )
        for family, boost in boosts.items():
            if family not in mode_families:
                raise ValueError(
                    f"mapping[{state}].{family} 在 mode={mode} 下不可解析；"
                    f"该 mode 可用因子族: {sorted(mode_families)}"
                )
            families_used.add(family)
        mapping[state] = {f: float(b) for f, b in boosts.items()}

    # carrier 数据体检（编译期硬门禁；断供/盲测段无覆盖 => 拒）
    ok, health_errors, reported = check_carrier_health(
        tuple(dims), carrier_health=carrier_health, blind_start=blind_start
    )
    if not ok:
        raise ValueError("rotation_strategy carrier 数据体检失败:\n" + "\n".join(health_errors))

    guard = dict(design["guardrails"])
    worst = worst_case_turnover(
        min_state_duration_days=int(guard["min_state_duration_days"]),
        mapping=mapping,
        state_enums=state_enums,
    )
    if worst > float(guard["max_daily_turnover"]):
        raise ValueError(
            f"编译期最坏换手 {worst:.4f} 超 max_daily_turnover="
            f"{guard['max_daily_turnover']}；公式=(1/min_state_duration="
            f"{guard['min_state_duration_days']})×max_pair_delta。降低 boost 幅度/拉长"
            "状态最短持续，或放宽换手上限"
        )

    canonical = _canonicalize(
        {
            "version": version,
            "mode": mode,
            "state_dimensions": [
                {
                    "dim": d.dim,
                    "carrier": d.carrier,
                    "signal": d.signal,
                    "window_days": d.window_days,
                    "threshold_mode": d.threshold_mode,
                    "params": d.params,
                    "quantile_benchmark": d.quantile_benchmark,
                }
                for d in dims
            ],
            "quantile_benchmark": design.get("quantile_benchmark", ""),
            "mapping": mapping,
            "guardrails": guard,
        }
    )
    # canonical 只含参与哈希的语义字段（排除 thesis/evidence 等非执行字段，
    # 避免改措辞换 id）；thesis/evidence 单独落 design.json 供评审。
    design_fp = _content_hash(canonical)

    return CompiledStrategy(
        strategy_id=make_strategy_id(design_fp),
        design_fp=design_fp,
        compile_fp=compile_fingerprint(),
        version=version,
        mode=mode,
        thesis=str(design.get("thesis", "")),
        evidence=str(design.get("evidence", "")),
        dimensions=tuple(dims),
        mapping={k: dict(v) for k, v in mapping.items()},
        guardrails=guard,
        quantile_benchmark=str(design.get("quantile_benchmark", "")),
        state_enums=state_enums,
        families_used=frozenset(families_used),
        worst_case_turnover=worst,
        carrier_health_reported=reported,
        blind_start=blind_start,
        canonical=canonical,
    )


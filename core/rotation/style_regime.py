# -*- coding: utf-8 -*-
"""状态层运行时 StyleRegime（Phase 1，spec §3.3）。

输入：已编译策略（``CompiledStrategy``）+ 载体信号面板（``carrier_signals``）。
输出：确定性状态序列（枚举 + status：normal / degraded / unknown）。

无前视纪律（不越窗）：T 日状态只用 ``(T-window_days, T-1]``（rolling q 参考窗）
或 expanding（仅 T-1 及以前）数据，判定一律不含 T 日；信号平滑与分位参考窗
都作用于 ``signal.shift(1)`` 序列，保证只用历史。

canonical 内置两组（size / value_growth）通过 ``canonical_design()`` +
``compile_design`` 走与 LLM 设计同一路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.rotation.compiler import DIM_BUCKETS, CompiledStrategy, compile_design

# ── canonical 内置状态（size / value_growth 两维）───────────────────────
CANONICAL_DIMENSIONS: tuple[dict[str, Any], ...] = (
    {
        "dim": "size",
        "carrier": "style_factor",
        "window_days": 120,
        "threshold_mode": "quantile",
        "params": {"q": 0.5},
    },
    {
        "dim": "value_growth",
        "carrier": "style_factor",
        "window_days": 60,
        "threshold_mode": "quantile",
        "params": {"q": 0.5},
    },
)
CANONICAL_GUARDRAILS: dict[str, Any] = {
    "max_daily_turnover": 0.20,
    "min_state_duration_days": 5,
    "no_lookahead": True,
}
# canonical mapping 用 technical 族（fundamental 模式调用方需自行编译替代设计）
CANONICAL_MAPPING: dict[str, dict[str, float]] = {
    "SMALL_VALUE": {"liquidity": 1.15, "reversal": 1.10, "momentum": 0.90},
    "SMALL_GROWTH": {"momentum": 1.10, "volume": 1.05, "reversal": 1.02},
    "LARGE_VALUE": {"volatility": 1.05, "reversal": 1.10, "momentum": 0.90},
    "LARGE_GROWTH": {"momentum": 1.20, "volume": 1.10, "reversal": 0.85},
}


def canonical_design(mode: str = "technical") -> dict[str, Any]:
    """构造 canonical 内置状态设计（size/value_growth，供快速验证与基线对照）。"""

    return {
        "version": 1,
        "thesis": (
            "canonical 内置：size（市值中枢扩张/收缩）× value_growth（价值-成长价差）"
            "两维判定市场风格状态；小市值/价值阶段偏好反转与流动性，大市值/成长阶段"
            "偏好动量。作公共基线与 LLM 自设计对照。"
        ),
        "evidence": "内置基线，不携带支撑数据段；作为等标尺对照与随机状态对照的对象",
        "state_dimensions": [dict(d) for d in CANONICAL_DIMENSIONS],
        "quantile_benchmark": "expanding",
        "mapping": {k: dict(v) for k, v in CANONICAL_MAPPING.items()},
        "allocation": None,
        "guardrails": dict(CANONICAL_GUARDRAILS),
    }


@dataclass(frozen=True)
class StateSeries:
    """确定性状态序列输出。

    states：每交易日状态枚举（UNKNOWN=从头无数据）；status：
    normal / degraded（某连续 run < min_state_days）/ unknown；
    buckets：每维高/低桶（'H'/'L'/NaN）。
    """

    states: pd.Series
    status: pd.Series
    buckets: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({"state": self.states, "status": self.status})
        for col in self.buckets.columns:
            out[f"bucket_{col}"] = self.buckets[col]
        return out


# ── 载体信号面板构建（Phase 1；spec §3.4 数据面）────────────────────────
def build_canonical_signals(
    panel: pd.DataFrame | None,
    style_beta: pd.DataFrame | None,
    *,
    index_bars: pd.DataFrame | None = None,
    aux: dict[str, pd.Series] | None = None,
    industry_field: str | None = "industry_sw_l1",
) -> pd.DataFrame:
    """从原始数据面构建 carrier 信号面板（datetime 索引、每列一信号）。

    约定：各输入 Series/DataFrame 需使用同一交易日历（调用方负责对齐；
    本函数不重采样）。缺失来源 => 对应信号列不产出 => carrier 体检判断供。
    实现（Phase 1 轻量代理，确定性、可解释）：
    - size_proxy：float_cap 横截面 log 中位数的日变化率（市值中枢扩张/收缩，
      大/小盘阶段的轻量代理；后续可换 SMB）；
    - value_growth_spread：风格日收益 value - growth（价值-成长价差）；
    - momentum_beta：风格日收益 momentum；
    - market_index_momentum：index_bars 首列收益率；
    - industry_momentum：行业等权收益截面差（最强-最弱行业），panel 需行业+收益列；
    - margin_chg / northbound_chg / macro_rate_level：aux 透传。
    """

    out: dict[str, pd.Series] = {}
    if panel is not None and "float_cap" in panel.columns:
        logcap = pd.Series(
            np.log(pd.to_numeric(panel["float_cap"], errors="coerce").to_numpy(dtype=float)),
            index=panel.index,
        )
        med = logcap.groupby(level="datetime", sort=False).median()
        out["size_proxy"] = med.pct_change()
    if style_beta is not None and not style_beta.empty:
        if {"value", "growth"} <= set(style_beta.columns):
            out["value_growth_spread"] = (
                style_beta["value"].astype(float) - style_beta["growth"].astype(float)
            )
        if "momentum" in style_beta.columns:
            out["momentum_beta"] = style_beta["momentum"].astype(float)
    if index_bars is not None and not index_bars.empty:
        first = str(index_bars.columns[0])
        out["market_index_momentum"] = pd.to_numeric(index_bars[first], errors="coerce").pct_change()
    if index_bars is not None and not index_bars.empty and panel is not None:
        out.update(_build_index_spreads(panel, index_bars))
    industry = _build_industry_momentum(panel, industry_field)
    if industry is not None:
        out["industry_momentum"] = industry
    for key in ("margin_chg", "northbound_chg", "macro_rate_level"):
        if aux is not None and key in aux and aux[key] is not None:
            seq = aux[key]
            out[key] = seq if isinstance(seq, pd.Series) else pd.Series(seq)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)


def _build_index_spreads(
    panel: pd.DataFrame, index_bars: pd.DataFrame
) -> dict[str, pd.Series]:
    """宽基价差信号：沪深300/中证500/中证1000/上证50 若都提供则算规模价差变化率。

    指数水平不可跨期比，故输出**价差的日变化率**（平稳量）。仅当至少两列可用。
    约定列名（与 CNE 宽基一致）：300=000300.SH、500=000905.SH、1000=000852.SH、
    50=000016.SH。
    """

    cols = {
        "hs300": "000300.SH",
        "zz500": "000905.SH",
        "zz1000": "000852.SH",
        "sz50": "000016.SH",
    }
    have = {k: v for k, v in cols.items() if v in index_bars.columns}
    out: dict[str, pd.Series] = {}
    spread_specs = [
        ("size_index_spread", ("zz1000", "hs300")),
        ("value_growth_index_spread", ("zz500", "sz50")),
        ("index_spread", ("zz500", "hs300")),
    ]
    for name, (a, b) in spread_specs:
        if a in have and b in have:
            spread = pd.to_numeric(index_bars[cols[a]], errors="coerce") - pd.to_numeric(
                index_bars[cols[b]], errors="coerce"
            )
            out[name] = spread.pct_change()
    return out


def _build_industry_momentum(
    panel: pd.DataFrame | None, industry_field: str | None
) -> pd.Series | None:
    """行业动量：每日各行业等权收益的截面差（最强-最弱行业）。panel 需收益列。"""

    if panel is None or not industry_field or industry_field not in panel.columns:
        return None
    ret_col = "ret" if "ret" in panel.columns else ("close" if "close" in panel.columns else None)
    if ret_col is None:
        return None
    if ret_col == "close":
        ret = (
            pd.to_numeric(panel["close"], errors="coerce")
            .groupby(level="instrument", sort=False)
            .pct_change()
        )
        ret.name = "ret"
    else:
        ret = pd.to_numeric(panel[ret_col], errors="coerce")
    ind = panel[industry_field]
    df = pd.DataFrame({"d": panel.index.get_level_values("datetime"), "g": ind, "ret": ret})
    df = df.dropna(subset=["ret", "g"])
    if df.empty:
        return None
    ind_mean = df.groupby(["d", "g"], sort=False)["ret"].mean()
    per_day = ind_mean.groupby(level=0, sort=False)
    spread = per_day.max() - per_day.min()
    return spread.rename("industry_momentum")


# ── 状态判定（无前视核心）────────────────────────────────────────────────
def _bucket_series(
    signal: pd.Series,
    *,
    window_days: int,
    threshold_mode: str,
    params: dict[str, Any],
    quantile_benchmark: str,
) -> pd.Series:
    """单维高/低桶：'H' >= 阈值、'L' < 阈值；NaN 保留（维持前态/UNKNOWN 由调用方处理）。

    只用 ``signal.shift(1)``（T-1 及以前）；平滑 rolling mean 与 quantile 参考窗
    都作用在该历史序列上，T 日状态不含 T 日信息。
    """

    prev = signal.astype(float).shift(1)
    w = int(window_days)
    smoothed = prev.rolling(w, min_periods=2).mean()
    threshold: np.ndarray | pd.Series | float
    if threshold_mode == "sign":
        threshold = 0.0
    elif threshold_mode == "level":
        threshold = float(params.get("threshold", 0.0))
    elif threshold_mode == "quantile":
        q = float(params.get("q", 0.5))
        if quantile_benchmark == "rolling":
            threshold = prev.rolling(w, min_periods=w).quantile(q)
        elif quantile_benchmark == "expanding":
            threshold = prev.expanding(min_periods=w).quantile(q)
        else:
            raise ValueError(
                f"quantile 参考窗未声明或非法: {quantile_benchmark!r} "
                "（编译器应已拒绝；rolling/expanding 均只含 T-1 及以前）"
            )
    else:
        raise ValueError(f"未知 threshold_mode {threshold_mode!r}")

    buckets = pd.Series(np.nan, index=signal.index, dtype=object)
    mask = smoothed.notna()
    if isinstance(threshold, (pd.Series, np.ndarray)):
        mask = mask & pd.Series(threshold, index=signal.index).notna()
    s = smoothed[mask]
    t = pd.Series(threshold, index=signal.index)[mask] if not isinstance(threshold, float) else threshold
    buckets.loc[mask] = np.where(s.to_numpy(dtype=float) >= (t if isinstance(t, float) else t.to_numpy(dtype=float)), "H", "L")
    return buckets


def _combine_buckets(
    compiled: CompiledStrategy, bucket_frames: list[pd.Series]
) -> tuple[pd.Series, pd.DataFrame]:
    """逐维 ffill（缺失维持前状态）+ H/L 桶映射到维度枚举名后拼接。

    例：size 维 H->LARGE/L->SMALL、value_growth 维 H->VALUE/L->GROWTH =>
    SMALL_VALUE / LARGE_GROWTH 等（与 compiler.state_enum_names 一致）。
    """

    per_dim = pd.DataFrame(index=bucket_frames[0].index)
    for dim, bucket in zip(compiled.dimensions, bucket_frames):
        low_name, high_name = DIM_BUCKETS[dim.dim]
        # 全 NaN 时 ffill 是无操作，且 pandas 2.3 对 object 全 NaN 列 ffill 会
        # 抛弃用警告（未来会 downcast）；直接保持 NaN 列语义等价
        if bucket.notna().any():
            per_dim[dim.dim] = bucket.ffill().map({"H": high_name, "L": low_name})
        else:
            per_dim[dim.dim] = bucket.astype(object)
    combined = per_dim.agg(lambda row: "_".join(row.dropna().astype(str)), axis=1)
    combined = combined.where(per_dim.notna().all(axis=1))
    states = combined.fillna("UNKNOWN").astype(str)
    states.index = per_dim.index
    states.name = "state"
    return states, per_dim


def _mark_status(
    states: pd.Series, *, min_state_days: int
) -> pd.Series:
    """状态状态标记：unknown / degraded（连续 run < min_state_days）/ normal。"""

    status = pd.Series("normal", index=states.index, dtype=object)
    status[states == "UNKNOWN"] = "unknown"
    if min_state_days > 1:
        changed = states != states.shift(1)
        run_id = changed.cumsum()
        run_len = states.groupby(run_id).transform("size")
        degraded = (run_len < int(min_state_days)) & (states != "UNKNOWN")
        status[degraded] = "degraded"
    return status


def compute_states(
    compiled: CompiledStrategy,
    carrier_signals: pd.DataFrame,
    *,
    min_state_days: int | None = None,
) -> StateSeries:
    """按编译后的确定性规则计算状态序列（spec §3.3；全确定性、无前视）。

    缺失语义：某维某日缺数据 => 维持前一状态；从头无数据 => UNKNOWN；
    UNKNOWN 日组合权重回退 baseline=1.0（在组合层消费）;
    degraded 状态照常调制、仅评估端不单独汇报（报告端语义，见回测 harness）。
    """

    if min_state_days is None:
        min_state_days = int(compiled.guardrails.get("min_state_duration_days", 5))
    bucket_frames: list[pd.Series] = []
    for dim in compiled.dimensions:
        if dim.signal not in carrier_signals.columns:
            # 信号列缺失 + carrier 体检未在上游拦截：该维整段无值 => UNKNOWN
            # （缺失维维持前态/全维开头 UNKNOWN 由 _combine_buckets 处理）
            bucket_frames.append(pd.Series(np.nan, index=carrier_signals.index, dtype=object))
            continue
        bucket_frames.append(
            _bucket_series(
                carrier_signals[dim.signal],
                window_days=dim.window_days,
                threshold_mode=dim.threshold_mode,
                params=dim.params,
                quantile_benchmark=dim.quantile_benchmark,
            )
        )
    states, per_dim = _combine_buckets(compiled, bucket_frames)
    status = _mark_status(states, min_state_days=int(min_state_days))
    return StateSeries(states=states, status=status, buckets=per_dim)


def boost_for_state(
    compiled: CompiledStrategy, state: str
) -> dict[str, float]:
    """某一状态的因子族软权重（mapping 子集外的状态/UNKNOWN => 空表=baseline 1.0）。"""

    if state == "UNKNOWN":
        return {}
    return {f: float(b) for f, b in (compiled.mapping.get(state) or {}).items()}

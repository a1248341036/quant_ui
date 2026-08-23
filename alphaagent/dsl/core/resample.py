"""多周期变换层：支持把细粒度行情聚合到更粗 bar，并提供 60m 无前视广播工具。

设计要点
--------
- 输入输出既支持 ``MultiIndex(datetime, instrument)`` 面板，也支持长表 ``datetime`` / ``dominant_id``。
- 按**自然整点**切桶，不处理夜盘/午休边界（与仓库现状一致，后续可另起 session-aware 版本）。
- 对离线合成更粗粒度行情时，可选 ``strict_complete_bars=True`` 要求每个桶内必须凑齐完整 bar 数。
- 粗周期 ``adj_vwap`` 为桶内根 bar 上 ``adj_vwap`` 的成交量加权平均；未调整 ``vwap`` 仍为成交额/成交量。
- 60m 面板的 ``datetime`` 索引使用桶起点时刻（``HH:00``），便于在 60m 上继续使用现有按
  行滚动的算子。广播阶段独立完成“右闭/左开 + 向后移 1 小时”的无前视规则。
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .intervals import bar_interval_to_minutes, normalize_bar_interval


# 不同字段的默认聚合规则；未列出的列使用 ``last``。
_AGG_FIRST = {"open", "adj_open", "today_open"}
_AGG_MAX = {"high", "adj_high", "today_high"}
_AGG_MIN = {"low", "adj_low", "today_low"}
_AGG_LAST = {"close", "adj_close"}
_AGG_SUM = {"volume", "total_turnover"}


def _select_numeric_columns(
    panel_1m: pd.DataFrame,
    include: Optional[Iterable[str]] = None,
) -> list[str]:
    if include is not None:
        cols = [c for c in include if c in panel_1m.columns]
    else:
        cols = list(panel_1m.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(panel_1m[c])]
    return numeric


def _aggregation_rule_for(col: str) -> str:
    if col in _AGG_FIRST:
        return "first"
    if col in _AGG_MAX:
        return "max"
    if col in _AGG_MIN:
        return "min"
    if col in _AGG_SUM:
        return "sum"
    if col in _AGG_LAST:
        return "last"
    # 默认 VWAP/价格型字段保留最后一笔；成交类字段已在上面覆盖。
    return "last"


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = pd.to_numeric(num, errors="coerce").to_numpy(dtype=float, copy=False)
    den_arr = pd.to_numeric(den, errors="coerce").to_numpy(dtype=float, copy=False)
    out = np.full(len(num_arr), np.nan, dtype=float)
    mask = np.isfinite(num_arr) & np.isfinite(den_arr) & (den_arr != 0.0)
    out[mask] = num_arr[mask] / den_arr[mask]
    return pd.Series(out, index=num.index, dtype=float)


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="datetime"), pd.Index([], name="instrument")]
        )
    )


def _expected_rows_per_bucket(base_interval: str | int, target_interval: str | int) -> int:
    base_minutes = bar_interval_to_minutes(base_interval)
    target_minutes = bar_interval_to_minutes(target_interval)
    if target_minutes % base_minutes != 0:
        raise ValueError(
            f"目标周期 {target_interval!r} 不是基础周期 {base_interval!r} 的整数倍"
        )
    return int(target_minutes // base_minutes)


def _pandas_floor_rule(interval: str | int) -> str:
    tag = normalize_bar_interval(interval)
    if tag.endswith("m"):
        return f"{tag[:-1]}min"
    if tag.endswith("d"):
        return f"{tag[:-1]}D"
    return tag


def build_timeframe_panel(
    panel: pd.DataFrame,
    *,
    target_interval: str = "60m",
    base_interval: str = "1m",
    columns: Optional[Iterable[str]] = None,
    strict_complete_bars: bool = False,
) -> pd.DataFrame:
    """把面板聚合为更粗周期（按自然边界切桶）。

    Parameters
    ----------
    panel:
        ``MultiIndex(datetime, instrument)`` 面板；列为特征列。
    target_interval:
        目标周期，如 ``5m`` / ``60m``。
    base_interval:
        当前面板的基础周期；用于 ``strict_complete_bars`` 时判断桶是否完整。
    columns:
        若指定，则只聚合这些列（可用于减轻内存压力）；非数值列会被跳过。
    strict_complete_bars:
        为 True 时，仅保留包含完整基础 bar 数的桶。

    Notes
    -----
    粗周期 ``adj_vwap``：桶内分钟（根 bar）``adj_vwap`` 的**成交量加权平均**
    ``sum(adj_vwap * volume) / sum(volume)``，仅计入 ``adj_vwap`` 与 ``volume`` 均有限且
    ``volume > 0`` 的行；若分母为 0 则为 NaN。未调整 ``vwap`` 仍为 ``sum(turnover)/sum(volume)``。

    Returns
    -------
    panel_target:
        ``MultiIndex(datetime, instrument)`` 的聚合面板，``datetime`` 为桶起点。
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel 必须是 (datetime, instrument) MultiIndex 面板")
    if panel.index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("panel 的索引层必须依次为 datetime、instrument")

    target_rule = normalize_bar_interval(target_interval)
    floor_rule = _pandas_floor_rule(target_rule)
    expected_rows = _expected_rows_per_bucket(base_interval, target_rule)
    use_cols = _select_numeric_columns(panel, include=columns)
    if not use_cols:
        return _empty_panel()
    requested_cols = list(use_cols)
    needs_adj_close_for_ret = "ret" in use_cols and "adj_close" in panel.columns
    if needs_adj_close_for_ret and "adj_close" not in use_cols:
        use_cols.append("adj_close")

    df = panel[use_cols].reset_index()
    df["__bucket__"] = df["datetime"].dt.floor(floor_rule)

    agg_map = {c: _aggregation_rule_for(c) for c in use_cols}
    if strict_complete_bars:
        counts = (
            df.groupby(["instrument", "__bucket__"], sort=True)["datetime"]
            .size()
            .rename("__n_rows__")
        )
        valid = counts[counts >= expected_rows].index
        if len(valid) == 0:
            return _empty_panel()
        df = (
            df.set_index(["instrument", "__bucket__"])
            .loc[valid]
            .reset_index()
        )
    grouped = df.groupby(["instrument", "__bucket__"], sort=True).agg(agg_map)

    # VWAP（未调整）：桶内成交额 / 成交量。
    if {"volume", "total_turnover"}.issubset(df.columns):
        vol_sum = grouped["volume"] if "volume" in grouped.columns else None
        turnover_sum = grouped["total_turnover"] if "total_turnover" in grouped.columns else None
        if vol_sum is not None and turnover_sum is not None:
            vwap_agg = _safe_divide(turnover_sum, vol_sum)
            if "vwap" in use_cols:
                grouped["vwap"] = vwap_agg

    # 调整后 VWAP：桶内分钟 adj_vwap 的成交量加权平均（不依赖未调整 vwap 列）。
    if "adj_vwap" in use_cols and "volume" in df.columns:
        av = pd.to_numeric(df["adj_vwap"], errors="coerce").to_numpy(dtype=float, copy=False)
        vol = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float, copy=False)
        mask = np.isfinite(av) & np.isfinite(vol) & (vol > 0.0)
        vw_num = np.where(mask, av * vol, 0.0)
        vw_den = np.where(mask, vol, 0.0)
        acc = df[["instrument", "__bucket__"]].copy()
        acc["__adj_vw_n"] = vw_num
        acc["__adj_vw_d"] = vw_den
        vw_part = acc.groupby(["instrument", "__bucket__"], sort=True)[
            ["__adj_vw_n", "__adj_vw_d"]
        ].sum()
        grouped["adj_vwap"] = _safe_divide(vw_part["__adj_vw_n"], vw_part["__adj_vw_d"])
    if "ret" in grouped.columns and "adj_close" in grouped.columns:
        grouped["ret"] = grouped["adj_close"].groupby(level="instrument").pct_change()
    if needs_adj_close_for_ret:
        grouped = grouped[requested_cols]
    # grouped.index 是 (instrument, bucket)；改回 (datetime, instrument) 并排序。
    grouped.index = grouped.index.set_names(["instrument", "datetime"])
    grouped = grouped.swaplevel("instrument", "datetime").sort_index()
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped


def build_60m_panel(
    panel: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    base_interval: str = "1m",
    strict_complete_bars: bool = False,
) -> pd.DataFrame:
    """把**主条行情**面板聚合为 60m 面板（按自然整点切桶）。

    ``panel`` 的 bar 可以来自 1m、5m 等与数据集一致的任一格律；`datetime` 仍按根 bar
    起点对齐。聚合规则见 ``build_timeframe_panel``（OHLCV 等按桶 first/max/min/last/sum）。

    参数 ``base_interval`` 须与 ``panel`` 实际根周期一致。它**仅**在
    ``strict_complete_bars=True`` 时参与「每桶是否凑满预期根数」的过滤；**当前默认**
    ``strict_complete_bars=False`` 时，无论写成 ``1m`` 还是 ``5m``，聚合路径都只对行做
    ``floor`` 到 60m 桶再 ``groupby``，**不因写错 base_interval 而改变数值**。

    若主数据是 5m 而不是 1m，60m K 线是由 5m bar 聚出来的；与从底层 1m 再聚 60m 相比，
    高低价等在理论上可能更粗（缺分钟内极值）。本仓库评估管线里应对 ``base_interval`` 与
    数据集的 ``bar_interval`` 保持一致，以便日后若开启「整桶才保留」时语义正确。"""
    return build_timeframe_panel(
        panel,
        target_interval="60m",
        base_interval=base_interval,
        columns=columns,
        strict_complete_bars=strict_complete_bars,
    )


def resample_universe_long(
    df: pd.DataFrame,
    *,
    target_interval: str,
    base_interval: str = "1m",
    strict_complete_bars: bool = False,
    symbol_col: str = "dominant_id",
) -> pd.DataFrame:
    """把长表行情聚合到更粗粒度。

    返回列仍为长表风格：至少包含 ``datetime`` 与 ``dominant_id``，便于直接写 parquet / sqlite。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", symbol_col])
    if "datetime" not in df.columns:
        raise ValueError("行情长表需含 datetime 列")
    if symbol_col not in df.columns:
        raise ValueError(f"行情长表需含 {symbol_col} 列")

    frame = df.copy()
    frame["instrument"] = frame[symbol_col].astype(str)
    idx_cols = ["datetime", "instrument"]
    feature_cols = [c for c in frame.columns if c not in idx_cols and c != symbol_col]
    panel = frame.set_index(idx_cols)[feature_cols].sort_index()
    out = build_timeframe_panel(
        panel,
        target_interval=target_interval,
        base_interval=base_interval,
        strict_complete_bars=strict_complete_bars,
    )
    long = out.reset_index().rename(columns={"instrument": symbol_col})
    return long


def broadcast_timeframe_to_main_freq(
    values: pd.DataFrame,
    target_index: pd.MultiIndex,
    target_interval: str,
) -> pd.DataFrame:
    """把辅周期 ``values`` 无前视地广播到**主频率** ``target_index``（与 ``broadcast_60m_to_main_freq`` 同
    一语义，但完成时间 = 桶起点 + 该辅周期整段长度）。

    Parameters
    ----------
    values:
        ``MultiIndex(datetime, instrument)`` 面板，``datetime`` 为辅周期桶**起点**（与
        ``build_timeframe_panel`` 输出一致）。
    target_index:
        主面板的 ``MultiIndex(datetime, instrument)``。
    target_interval:
        辅周期，如 ``5m`` / ``10m`` / ``60m`` / ``1h``（经 ``normalize_bar_interval`` 归一）。
    """
    if not isinstance(target_index, pd.MultiIndex):
        raise ValueError("target_index 必须是 (datetime, instrument) MultiIndex")
    if target_index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("target_index 的索引层必须依次为 datetime、instrument")

    tag = normalize_bar_interval(target_interval)
    bar_minutes = int(bar_interval_to_minutes(tag))
    completion = pd.Timedelta(minutes=bar_minutes)

    cols = list(values.columns)
    if values.empty or not cols:
        return pd.DataFrame(
            np.nan,
            index=target_index,
            columns=cols,
        )

    # 右表：桶起点 + 整段 bar 长 = 该桶「完成时间」，作为 merge_asof 的 on 键（无前视）。
    src = values.reset_index()
    src["datetime"] = src["datetime"] + completion
    src = src.sort_values(["datetime", "instrument"], kind="mergesort").reset_index(
        drop=True
    )

    # 左表：把 target_index 展开并附带原始行号，以便合并后恢复原顺序。
    tgt = pd.DataFrame(
        {"__row__": np.arange(len(target_index), dtype=np.int64)},
        index=target_index,
    ).reset_index()
    tgt_sorted = tgt.sort_values(
        ["datetime", "instrument"], kind="mergesort"
    ).reset_index(drop=True)

    # 按 instrument 分组的 asof backward：在同一 instrument 内找最近一根"完成时间 <= t"的 bar。
    merged = pd.merge_asof(
        tgt_sorted,
        src,
        on="datetime",
        by="instrument",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("__row__", kind="mergesort")

    out = pd.DataFrame(
        merged[cols].to_numpy(),
        index=target_index,
        columns=cols,
    )
    return out


def broadcast_60m_to_main_freq(
    values_60m: pd.DataFrame,
    target_index: pd.MultiIndex,
) -> pd.DataFrame:
    """兼容旧名：等价于 ``broadcast_timeframe_to_main_freq(..., target_interval=\"60m\")``。"""
    return broadcast_timeframe_to_main_freq(values_60m, target_index, "60m")


__all__ = [
    "broadcast_60m_to_main_freq",
    "broadcast_timeframe_to_main_freq",
    "build_60m_panel",
    "build_timeframe_panel",
    "resample_universe_long",
]

"""股票日历混频：日频 panel 聚合为 @1d / @1w，并无前视广播回日频。

改编自 AQRA dsl_core/resample.py，适配 A 股日频 panel 与 W-FRI 周线。
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alphaagent.dsl.stock.intervals import bar_interval_to_timedelta, normalize_bar_interval

# 字段聚合规则（OHLCV + 股票扩展列）
_AGG_FIRST = {"open", "adj_open"}
_AGG_MAX = {"high", "adj_high"}
_AGG_MIN = {"low", "adj_low"}
_AGG_LAST = {
    "close",
    "adj_close",
    "is_trade",
    "not_st",
    "float_cap",
    "tot_cap",
    "label_1d_close_to_close",
    "label_1d_open_to_open",
    "label_10d_close_to_close",
    "label_20d_close_to_close",
}
_AGG_SUM = {"volume", "amount", "total_turnover"}


def _select_numeric_columns(
    panel: pd.DataFrame,
    include: Optional[Iterable[str]] = None,
) -> list[str]:
    if include is not None:
        cols = [c for c in include if c in panel.columns]
    else:
        cols = list(panel.columns)
    return [c for c in cols if pd.api.types.is_numeric_dtype(panel[c])]


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


def _bucket_datetime(dt: pd.Series, interval: str) -> pd.Series:
    """日历切桶；1w 用 to_period（pandas 3 不支持 dt.floor('W-FRI')）。"""
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt)
    tag = normalize_bar_interval(interval)
    if tag == "1d":
        return dt.dt.floor("1D")
    if tag == "1w":
        # W-FRI：自然周，周五为周期锚点
        return dt.dt.to_period("W-FRI").dt.start_time
    raise ValueError(f"不支持的 bucket 周期: {tag!r}")


def build_timeframe_panel(
    panel: pd.DataFrame,
    *,
    target_interval: str = "1w",
    base_interval: str = "1d",
    columns: Optional[Iterable[str]] = None,
    strict_complete_bars: bool = False,
) -> pd.DataFrame:
    """把日频 panel 聚合为更粗日历周期（1d / 1w）。"""
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel 必须是 (datetime, instrument) MultiIndex 面板")
    if panel.index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("panel 的索引层必须依次为 datetime、instrument")

    target_rule = normalize_bar_interval(target_interval)
    normalize_bar_interval(base_interval)

    use_cols = _select_numeric_columns(panel, include=columns)
    if not use_cols:
        return _empty_panel()
    requested_cols = list(use_cols)

    # ret 需在聚合后按 instrument 重算
    needs_adj_close_for_ret = "ret" in use_cols and "adj_close" in panel.columns
    if needs_adj_close_for_ret and "adj_close" not in use_cols:
        use_cols.append("adj_close")

    df = panel[use_cols].reset_index()
    # 按日历边界切桶
    df["__bucket__"] = _bucket_datetime(df["datetime"], target_rule)

    agg_map = {c: _aggregation_rule_for(c) for c in use_cols}

    # 股票 1d→1w 不按固定交易日数过滤桶；strict 模式暂不启用
    if strict_complete_bars and target_rule != "1w":
        raise ValueError("股票 resample 暂不支持 strict_complete_bars")

    grouped = df.groupby(["instrument", "__bucket__"], sort=True).agg(agg_map)

    # vwap = 桶内 sum(amount) / sum(volume)
    if {"volume", "amount"}.issubset(df.columns) and "vwap" in use_cols:
        vol_sum = grouped["volume"] if "volume" in grouped.columns else None
        amt_sum = grouped["amount"] if "amount" in grouped.columns else None
        if vol_sum is not None and amt_sum is not None:
            grouped["vwap"] = _safe_divide(amt_sum, vol_sum)

    # 聚合后重算 ret
    if "ret" in grouped.columns and "adj_close" in grouped.columns:
        grouped["ret"] = grouped["adj_close"].groupby(level="instrument").pct_change(
            fill_method=None
        )

    if needs_adj_close_for_ret:
        grouped = grouped[requested_cols]

    # (instrument, bucket) → (datetime, instrument)
    grouped.index = grouped.index.set_names(["instrument", "datetime"])
    grouped = grouped.swaplevel("instrument", "datetime").sort_index()
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped


def broadcast_timeframe_to_main_freq(
    values: pd.DataFrame,
    target_index: pd.MultiIndex,
    target_interval: str,
) -> pd.DataFrame:
    """辅周期 values 无前视广播到主频 target_index（merge_asof backward）。"""
    if not isinstance(target_index, pd.MultiIndex):
        raise ValueError("target_index 必须是 (datetime, instrument) MultiIndex")
    if target_index.names[:2] != ["datetime", "instrument"]:
        raise ValueError("target_index 的索引层必须依次为 datetime、instrument")

    tag = normalize_bar_interval(target_interval)
    # 完成时刻 = 桶起点 + 整段 bar 长（与 AQRA 60m 语义一致）
    completion = bar_interval_to_timedelta(tag)

    cols = list(values.columns)
    if values.empty or not cols:
        return pd.DataFrame(np.nan, index=target_index, columns=cols)

    # 右表：桶起点 + bar 全长 → 完成时间
    src = values.reset_index()
    src["datetime"] = src["datetime"] + completion
    src = src.sort_values(["datetime", "instrument"], kind="mergesort").reset_index(
        drop=True
    )

    # 左表带行号，合并后恢复原顺序
    tgt = pd.DataFrame(
        {"__row__": np.arange(len(target_index), dtype=np.int64)},
        index=target_index,
    ).reset_index()
    tgt_sorted = tgt.sort_values(["datetime", "instrument"], kind="mergesort").reset_index(
        drop=True
    )

    merged = pd.merge_asof(
        tgt_sorted,
        src,
        on="datetime",
        by="instrument",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("__row__", kind="mergesort")

    return pd.DataFrame(
        merged[cols].to_numpy(),
        index=target_index,
        columns=cols,
    )

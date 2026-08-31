"""Panel 构建与持久化（离线）。

本模块**不联网**：panel 由本地 hq 缓存（`artifacts/market/daily_hq.parquet`）
离线构建。行情拉取见 `alphaagent.data.market_fetch`。

- 主入口：build_panel_from_hq（读 hq 缓存离线构建，plugins/cnequity 链路消费）
- 衍生列逻辑与 AlphaAgent-Stock 保持一致
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.core.paths import PANEL_PATH
from alphaagent.core.types import OUTPUT_COLUMNS
from alphaagent.data.universe import filter_universe

DEFAULT_PANEL_PATH = PANEL_PATH

# label_{N}d_close_to_close：T+1 收盘 → T+(N+1) 收盘
CLOSE_TO_CLOSE_LABEL_HOLD_DAYS = (1, 10, 20)


def close_to_close_label_name(hold_days: int) -> str:
    return f"label_{hold_days}d_close_to_close"


_DERIVED_COLUMNS = (
    "ret",
    "label_1d_open_to_open",
    *(close_to_close_label_name(n) for n in CLOSE_TO_CLOSE_LABEL_HOLD_DAYS),
)


def _coerce_datetime_index(panel: pd.DataFrame) -> pd.DataFrame:
    """确保 MultiIndex datetime 层为 DatetimeIndex。"""
    if not isinstance(panel.index, pd.MultiIndex):
        return panel
    if panel.index.names[0] != "datetime":
        return panel

    dt = panel.index.get_level_values("datetime")
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt)
        inst = panel.index.get_level_values("instrument")
        panel = panel.copy()
        panel.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
    return panel.sort_index()


def slice_panel(
    panel: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """按 datetime 闭区间 [start, end] 切片。"""
    if start is None and end is None:
        return panel

    dt = panel.index.get_level_values("datetime")
    mask = pd.Series(True, index=panel.index)
    if start is not None:
        mask &= dt >= pd.Timestamp(start)
    if end is not None:
        mask &= dt <= pd.Timestamp(end)
    return panel.loc[mask]


def _calc_label_1d_open_to_open(adj_open: pd.Series) -> pd.Series:
    open_t1 = adj_open.shift(-1)
    open_t2 = adj_open.shift(-2)
    denom = open_t1.replace(0, np.nan)
    return (open_t2 - open_t1) / denom


def _calc_label_nd_close_to_close(adj_close: pd.Series, hold_days: int) -> pd.Series:
    """T+1 收盘 → T+(hold_days+1) 收盘。例：hold_days=10 即 T+1 close 到 T+11 close。"""
    entry = adj_close.shift(-1)
    exit_ = adj_close.shift(-(hold_days + 1))
    denom = entry.replace(0, np.nan)
    return (exit_ - entry) / denom


def _derive_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """从原始行情宽表衍生 adj_*、vwap 等（不含 ret / label）。

    资产类型兼容：
    - ETF 等无复权因子的数据源缺 ``adjfactor``，兜底补 1.0（qfq 前复权价）。
    - 缺 ``float_cap`` / ``tot_cap`` 时补 NaN（评估 profile 会跳过市值类指标）。
    """
    df = df.copy()
    df = df.rename_axis(index={"code": "instrument"})

    if "adjfactor" not in df.columns:
        df["adjfactor"] = 1.0

    for col in ("open", "high", "low", "close"):
        df[f"adj_{col}"] = df[col] * df["adjfactor"]

    if "float_cap" not in df.columns:
        df["float_cap"] = np.nan
    if "tot_cap" not in df.columns:
        df["tot_cap"] = np.nan

    if "isTrade" in df.columns:
        df = df.rename(columns={"isTrade": "is_trade", "notST": "not_st"})

    vol = df["volume"].replace(0, np.nan)
    df["vwap"] = df["amount"] / vol
    df["adj_vwap"] = df["vwap"] * df["adjfactor"]
    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """在完整时间序列上计算 ret / label。"""
    df = df.copy()
    df["ret"] = df.groupby(level="instrument", sort=False)["adj_close"].pct_change(fill_method=None)

    g_close = df.groupby(level="instrument", sort=False)["adj_close"]
    for hold_days in CLOSE_TO_CLOSE_LABEL_HOLD_DAYS:
        col = close_to_close_label_name(hold_days)
        df[col] = g_close.transform(lambda s, d=hold_days: _calc_label_nd_close_to_close(s, d))

    df["label_1d_open_to_open"] = df.groupby(level="instrument", sort=False)[
        "adj_open"
    ].transform(_calc_label_1d_open_to_open)
    return df


def _finalize_panel(df: pd.DataFrame, *, dtype: str = "float32") -> pd.DataFrame:
    # 兼容缺列的旧 hq 缓存 / 合成数据：缺失的输出列置 NaN
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # 保留 OUTPUT_COLUMNS + 任何插件带来的额外列
    extra_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    final_cols = list(OUTPUT_COLUMNS) + extra_cols
    panel = df[final_cols].copy()
    # 数值列 downcast（排除标记列）
    non_numeric = {"is_trade", "not_st"}
    for col in panel.columns:
        if col in non_numeric:
            continue
        if pd.api.types.is_numeric_dtype(panel[col]):
            panel[col] = panel[col].astype(dtype)

    panel = panel.sort_index()
    panel = _coerce_datetime_index(panel)

    assert panel.index.names == ["datetime", "instrument"]
    assert not panel.index.duplicated().any()
    return panel


def _panel_base_from_hq(
    hq: pd.DataFrame,
    *,
    universe_mask: bool = True,
    dtype: str = "float32",
) -> pd.DataFrame:
    """hq → panel 基础列（ret / label 置 NaN，供增量 merge 后统一重算）。"""
    df = hq.copy()
    if universe_mask:
        df = filter_universe(df)
    if df.empty:
        return df

    df = _derive_base_columns(df)
    for col in _DERIVED_COLUMNS:
        df[col] = np.nan

    return _finalize_panel(df, dtype=dtype)


def _ensure_derived_columns(panel: pd.DataFrame, *, dtype: str = "float32") -> pd.DataFrame:
    """补齐缺失的 ret / label 列（panel schema 升级时用）。"""
    panel = panel.copy()
    for col in _DERIVED_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
            panel[col] = panel[col].astype(dtype)
    return panel


def _rederive_since(panel: pd.DataFrame, since: pd.Timestamp, *, dtype: str = "float32") -> pd.DataFrame:
    """基于 panel 内 adj 列，从 since 起重算 ret / label（用全历史 groupby，避免前视缺失）。"""
    if panel.empty:
        return panel

    panel = _ensure_derived_columns(panel, dtype=dtype)
    since = pd.Timestamp(since)
    dt = panel.index.get_level_values("datetime")
    mask = dt >= since
    if not mask.any():
        return panel

    full_ret = panel.groupby(level="instrument", sort=False)["adj_close"].pct_change(fill_method=None)

    g_close = panel.groupby(level="instrument", sort=False)["adj_close"]
    full_labels_c2c = {
        close_to_close_label_name(hold_days): g_close.transform(
            lambda s, d=hold_days: _calc_label_nd_close_to_close(s, d)
        )
        for hold_days in CLOSE_TO_CLOSE_LABEL_HOLD_DAYS
    }

    full_label_o = panel.groupby(level="instrument", sort=False)["adj_open"].transform(
        _calc_label_1d_open_to_open
    )

    panel.loc[mask, "ret"] = full_ret.loc[mask].astype(dtype)
    for col, series in full_labels_c2c.items():
        panel.loc[mask, col] = series.loc[mask].astype(dtype)
    panel.loc[mask, "label_1d_open_to_open"] = full_label_o.loc[mask].astype(dtype)
    return panel


def build_panel_from_hq(
    hq: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    universe_mask: bool = True,
    dtype: str = "float32",
) -> pd.DataFrame:
    """从 (datetime, code) 行情宽表构建 panel。"""
    df = hq.copy()
    if start is not None or end is not None:
        dt = pd.to_datetime(df.index.get_level_values(0))
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= dt >= pd.Timestamp(start)
        if end is not None:
            mask &= dt <= pd.Timestamp(end)
        df = df.loc[mask]

    if universe_mask:
        df = filter_universe(df)

    if df.empty:
        return df

    df = _derive_base_columns(df)
    df = _add_derived_columns(df)
    return _finalize_panel(df, dtype=dtype)


def _enrich_panel(
    panel: pd.DataFrame,
    *,
    with_fundamentals: bool,
    quarterly_path,
    disclosure_path,
    include_disclosure_features: bool,
    with_industry: bool,
    industry_path,
    refresh_industry: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    """离线 enrich：从本地缓存并入 funda_* / industry_sw_l1 列。"""
    if with_fundamentals:
        from alphaagent.core.paths import DISCLOSURE_CALENDAR_PATH, FUNDAMENTAL_QUARTERLY_PATH
        from alphaagent.data.fundamental import enrich_panel_fundamentals

        panel = enrich_panel_fundamentals(
            panel,
            quarterly_path=quarterly_path or FUNDAMENTAL_QUARTERLY_PATH,
            disclosure_path=disclosure_path or DISCLOSURE_CALENDAR_PATH,
            include_disclosure_features=include_disclosure_features,
        )

    if with_industry:
        from alphaagent.data.industry import enrich_panel_industry

        ind_kwargs: dict = {"refresh": refresh_industry, "verbose": verbose}
        if industry_path is not None:
            ind_kwargs["membership_path"] = industry_path
        panel = enrich_panel_industry(panel, **ind_kwargs)

    return panel


def load_panel(path: Path | str = DEFAULT_PANEL_PATH) -> pd.DataFrame:
    """加载 panel parquet。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"panel 不存在: {p}")
    panel = pd.read_parquet(p)
    if "instrument" not in panel.index.names and "code" in panel.index.names:
        panel = panel.rename_axis(index={"code": "instrument"})
    return _coerce_datetime_index(panel)


def save_panel(panel: pd.DataFrame, path: Path | str) -> Path:
    """写出 panel parquet。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    return out


# ---------------------------------------------------------------------------
# adjfactor 诊断（纯函数，不联网；修补见 market_fetch.repair_panel_adjfactor）
# ---------------------------------------------------------------------------
def find_suspect_adjfactor_instruments(
    panel: pd.DataFrame,
    *,
    min_real_factor: float = 1.5,
) -> list[str]:
    """宽口径候选：曾有 adjfactor>min_real_factor，且仍存在 adjfactor≈1 的行。

    新股上市初期 adjfactor=1 也符合此条件，**误报多**；修补请用 find_adjfactor_jump_instruments。
    """
    if panel.empty:
        return []

    inst_max = panel.groupby(level="instrument")["adjfactor"].max()
    candidates = inst_max[inst_max > min_real_factor].index
    suspects: list[str] = []
    for inst in candidates:
        s = panel.xs(inst, level="instrument")["adjfactor"]
        if (s <= 1.0 + 1e-6).any():
            suspects.append(str(inst))
    return sorted(suspects)


def find_adjfactor_jump_instruments(
    panel: pd.DataFrame,
    *,
    low: float = 1.01,
    high: float = 1.5,
    max_close_move: float = 0.25,
) -> list[str]:
    """窄口径候选：相邻交易日 adjfactor 从≈1 跳到≥high（或反向），且 raw close 涨跌幅不大。

    对应 merge 失败导致的尺度断层（如 600601 的 1.0 → 5764）；正常上市/除权不会命中。
    """
    if panel.empty:
        return []

    suspects: list[str] = []
    for inst in panel.index.get_level_values("instrument").unique():
        s = panel.xs(inst, level="instrument").sort_index()
        adj = s["adjfactor"].to_numpy(dtype=float, copy=False)
        close = s["close"].to_numpy(dtype=float, copy=False)
        if len(adj) < 2:
            continue
        for i in range(len(adj) - 1):
            if close[i] <= 0:
                continue
            if abs(close[i + 1] / close[i] - 1.0) > max_close_move:
                continue
            if adj[i] <= low and adj[i + 1] >= high:
                suspects.append(str(inst))
                break
            if adj[i] >= high and adj[i + 1] <= low:
                suspects.append(str(inst))
                break
    return sorted(set(suspects))


def count_suspect_adjfactor_rows(panel: pd.DataFrame, instruments: list[str]) -> int:
    """指定股票列表中 adjfactor≈1 的行数。"""
    if not instruments:
        return 0
    inst_idx = panel.index.get_level_values("instrument")
    mask = inst_idx.isin(instruments) & (panel["adjfactor"] <= 1.0 + 1e-6)
    return int(mask.sum())


def _rederive_adj_price_columns(panel: pd.DataFrame, *, dtype: str = "float32") -> pd.DataFrame:
    """按 adjfactor 重算 adj_* / adj_vwap。"""
    panel = panel.copy()
    for col in ("open", "high", "low", "close"):
        panel[f"adj_{col}"] = (panel[col] * panel["adjfactor"]).astype(dtype)
    panel["adj_vwap"] = (panel["vwap"] * panel["adjfactor"]).astype(dtype)
    return panel

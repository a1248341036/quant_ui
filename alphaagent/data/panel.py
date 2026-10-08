"""Panel 构建与持久化（离线）。

本模块**不联网**：panel 由本地 hq 缓存（`artifacts/market/daily_hq.parquet`）
离线构建。行情拉取见 `alphaagent.data.market_fetch`。

- 主入口：build_panel_from_hq（读 hq 缓存离线构建，plugins/cnequity 链路消费）
- 衍生列逻辑与 AlphaAgent-Stock 保持一致
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.core.paths import PANEL_PATH
from alphaagent.core.types import OUTPUT_COLUMNS
from alphaagent.data.universe import filter_universe

DEFAULT_PANEL_PATH = PANEL_PATH

# 切片视图快路径开关（0 = 回退布尔掩码复制路径；排查问题时可用）
_SLICE_VIEW_ENABLED = os.environ.get("ALPHA_PANEL_SLICE_VIEW", "1") not in {"0", "false", "False"}

# label_{N}d_close_to_close：T+1 收盘 → T+(N+1) 收盘的**点到点累计收益**
# （不是"后 N 天日均收益"，也不除以 N）；逐日滚动 → 相邻样本重叠 N-1 天，
# 所以拿它算 ICIR 必须按持有期去重叠（factor/metrics/ic.py:cs_ic_summary）。
# 各档用途（**本家族是 close→close**；主档 technical 的研究口径是另一个列
# label_1d_open_to_open，见下方 _DERIVED_COLUMNS，勿与本家族的 1d 混用）：
#   1  = 收盘到收盘 1 日口径（提示词推荐表与部分脚本/基线使用）
#   5  = weekly 三对齐子档（technical_weekly，持有 5 个交易日）
#   10 = 历史遗留（现无任何档位推荐，基本面档走 20d），保留仅为老 panel / 老条目兼容
#   20 = 慢线档（fundamental / technical_monthly）
# label 期与调仓频率刻意解耦，口径说明见 core/research_modes.py 顶部区块。
CLOSE_TO_CLOSE_LABEL_HOLD_DAYS = (1, 5, 10, 20)


def close_to_close_label_name(hold_days: int) -> str:
    return f"label_{hold_days}d_close_to_close"


_DERIVED_COLUMNS = (
    "ret",
    "label_1d_open_to_open",
    *(close_to_close_label_name(n) for n in CLOSE_TO_CLOSE_LABEL_HOLD_DAYS),
)


def _coerce_datetime_index(panel: pd.DataFrame) -> pd.DataFrame:
    """确保索引为 ``(datetime, instrument)`` 且 datetime 层是 DatetimeIndex。

    兼容两种落盘形态（2026-10-08 补）：
    1. **索引已是** ``(datetime, instrument)``：CNE adapter 实时构建的面板，只需保证
       datetime 层是 DatetimeIndex；
    2. ``datetime``/``instrument``（或 ``date``/``code``）只是**普通列**：面板缓存
       parquet 以 ``reset_index()`` 落盘时即如此 —— 此前直接返回 RangeIndex，
       导致 ``--panel <缓存文件>`` 在 ``slice_panel`` 处 KeyError；现按列重建索引。
    """
    if not isinstance(panel.index, pd.MultiIndex):
        cols = set(panel.columns)
        if {"datetime", "instrument"} <= cols:
            return _coerce_datetime_index(panel.set_index(["datetime", "instrument"]))
        if {"date", "code"} <= cols:
            return _coerce_datetime_index(
                panel.set_index(["date", "code"]).rename_axis(
                    index={"date": "datetime", "code": "instrument"})
            )
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
    """按 datetime 闭区间 [start, end] 切片。

    **视图快路径（2026-09-10）**：panel 的索引是 ``(datetime, instrument)`` 且
    已按 datetime 排序，因此日期区间对应的行是**连续区间**——用 ``iloc[a:b]``
    切片返回与父面板共享数据块的视图（实测 ``np.shares_memory`` 为 True），
    不再为 train/val 各复制一份（8.2M 行 × 124 列面板上实测省 2.8 GiB）。
    布尔掩码路径保留为回退（索引非单调/非 DatetimeIndex 时）。

    只读语义：评估链路对切片的读取都是只读（因子值由 DSL 重新计算，transform
    写入的是因子数组而非 panel），故共享数据块安全。需要可写副本时显式 ``.copy()``。
    ``ALPHA_PANEL_SLICE_VIEW=0`` 可关闭该快路径。
    """
    if start is None and end is None:
        return panel

    if _SLICE_VIEW_ENABLED:
        dt_level = panel.index.get_level_values("datetime")
        if isinstance(dt_level, pd.DatetimeIndex) and dt_level.is_monotonic_increasing:
            lo = dt_level.searchsorted(pd.Timestamp(start), side="left") if start is not None else 0
            hi = (
                dt_level.searchsorted(pd.Timestamp(end), side="right")
                if end is not None
                else len(panel)
            )
            if lo <= 0 and hi >= len(panel):
                return panel  # 全区间：直接返回父面板，连视图都不用建
            if hi <= lo:
                return panel.iloc[0:0]
            return panel.iloc[lo:hi]

    dt = panel.index.get_level_values("datetime")
    mask = pd.Series(True, index=panel.index)
    if start is not None:
        mask &= dt >= pd.Timestamp(start)
    if end is not None:
        mask &= dt <= pd.Timestamp(end)
    return panel.loc[mask]


def ensure_sorted(panel: pd.DataFrame) -> pd.DataFrame:
    """已按索引排序则原样返回，否则返回排序副本（热路径专用）。

    pandas 的 ``sort_index()`` 对**已排序**面板也无条件 ``take`` 整表拷贝：
    实测 7,782,511 行 × 176 列面板一次 ``sort_index()`` 瞬时新增 ~15GB
    （整表拷贝 + MultiIndex 排序机械）、耗时 ~5s。submit/评估热路径上该调用
    曾出现 3 份全量拷贝同时存活（compute_ingest_metrics 内 153/227/214 三处），
    是 31GB 机器上 21GB 峰值的主因。session 面板加载时本就有序（切片视图亦然），
    经此短路后排序退化为一次 O(n) 单调性检查（索引对象级缓存）。
    """
    if panel.index.is_monotonic_increasing:
        return panel
    return panel.sort_index()


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

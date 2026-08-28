"""季频基本面 PIT 展开与披露日历特征（对齐 AlphaAgent-Stock 语义）。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.core.paths import DISCLOSURE_CALENDAR_PATH, FUNDAMENTAL_QUARTERLY_PATH

# 财报科目中文列名 → DSL 列名（与 AlphaAgent 一致，供未来三大表接入）
FUNDAMENTAL_STATEMENT_COLUMN_MAP: dict[str, str] = {
    "总资产": "funda_fs_total_assets",
    "总负债": "funda_fs_total_liabilities",
    "流动资产": "funda_fs_current_assets",
    "流动负债": "funda_fs_current_liabilities",
    "非流动负债": "funda_fs_noncurrent_liabilities",
    "股东权益合计": "funda_fs_total_equity",
    "股东权益合计含少数": "funda_fs_total_equity_incl_mi",
    "股东权益及负债总计": "funda_fs_total_liab_equity",
    "未分配利润": "funda_fs_retained_earnings",
    "息税前利润": "funda_fs_ebit",
    "留存收益": "funda_fs_retained_income",
    "应交税费": "funda_fs_taxes_payable",
    "其他应付款": "funda_fs_other_payables",
    "应付职工薪酬": "funda_fs_payroll_payable",
    "在建工程": "funda_fs_construction_in_progress",
    "盈余公积金": "funda_fs_surplus_reserve",
    "其他综合收益": "funda_fs_other_comprehensive_income",
    "营业收入_季度": "funda_fs_oper_revenue_q",
    "总营业成本_季度": "funda_fs_oper_cost_q",
    "所得税_季度": "funda_fs_income_tax_q",
    "营业税金及附加_季度": "funda_fs_tax_surcharge_q",
    "归母净利润_季度": "funda_fs_net_profit_parent_q",
    "持续经营净利润_季度": "funda_fs_net_profit_continuing_q",
    "母公司综合收益_季度": "funda_fs_comprehensive_income_parent_q",
    "利息费用_季度": "funda_fs_interest_expense_q",
    "销售费用_季度": "funda_fs_selling_expense_q",
    "稀释每股收益_季度": "funda_fs_eps_diluted_q",
    "基本每股收益_季度": "funda_fs_eps_basic_q",
    "少数股东损益_季度": "funda_fs_minority_interest_q",
    "经营现金净流_季度": "funda_fs_ocf_net_q",
    "经营现金流入_季度": "funda_fs_ocf_inflow_q",
    "投资现金净流_季度": "funda_fs_icf_net_q",
    "综合收益_季度": "funda_fs_comprehensive_income_q",
    "其他收入_季度": "funda_fs_other_income_q",
    "购建长期资产支付现金_季度": "funda_fs_capex_cash_q",
    "期末现金等价物_季度": "funda_fs_cash_equiv_end_q",
    "支付职工现金_季度": "funda_fs_cash_paid_employees_q",
    "支付税费_季度": "funda_fs_cash_paid_taxes_q",
    "固定资产折旧_季度累计": "funda_fs_depreciation_q_ytd",
    "间接法经营活动现金流量净额_季度累计": "funda_fs_ocf_indirect_q_ytd",
    "营运资本": "funda_fs_working_capital",
}

DISCLOSURE_DISTANCE_COLUMNS = (
    "funda_days_since_disclose",
    "funda_days_since_quarter_start",
)

_QUARTER_PERIOD_START_MD = {
    (3, 31): (1, 1),
    (6, 30): (4, 1),
    (9, 30): (7, 1),
    (12, 31): (10, 1),
}


def _load_quarterly_fundamentals(path: Path | str) -> pd.DataFrame:
    """读取季末基本面宽表，索引 (report_end, instrument)。"""
    raw_path = Path(path).expanduser()
    if not raw_path.is_file():
        raise FileNotFoundError(f"基本面文件不存在: {raw_path}")

    raw = pd.read_parquet(raw_path)
    if "instrument" not in raw.index.names and "code" in raw.index.names:
        raw = raw.rename_axis(index={"code": "instrument", "datetime": "report_end"})
    elif raw.index.names[0] == "datetime":
        raw = raw.rename_axis(index={"datetime": "report_end"})

    rename = {
        col: FUNDAMENTAL_STATEMENT_COLUMN_MAP[col]
        for col in raw.columns
        if col in FUNDAMENTAL_STATEMENT_COLUMN_MAP
    }
    if rename:
        raw = raw.rename(columns=rename)

    allowed = set(FUNDAMENTAL_STATEMENT_COLUMN_MAP.values())
    unknown = [
        c
        for c in raw.columns
        if not str(c).startswith("funda_") and c not in allowed
    ]
    if unknown:
        raise ValueError(f"{raw_path} 含未识别列: {sorted(unknown)}")

    report_end = pd.to_datetime(raw.index.get_level_values("report_end"))
    instrument = raw.index.get_level_values("instrument")
    raw.index = pd.MultiIndex.from_arrays(
        [report_end, instrument],
        names=["report_end", "instrument"],
    )
    return raw.sort_index()


def load_disclosure_calendar(path: Path | str) -> pd.DataFrame:
    """宽表 (report_end × instrument) → long: report_end, instrument, disclosure。"""
    wide = pd.read_parquet(path)
    wide.index = pd.to_datetime(wide.index)
    long = wide.stack(future_stack=True).rename("disclosure").reset_index()
    long.columns = ["report_end", "instrument", "disclosure"]
    long["disclosure"] = pd.to_datetime(long["disclosure"], errors="coerce")
    return long.dropna(subset=["disclosure"])


def validate_quarter_report_ends(path: Path | str) -> None:
    """确认披露映射表行索引均为标准 A 股季报季末。"""
    ends = pd.to_datetime(pd.read_parquet(path).index)
    bad = [d for d in ends if (d.month, d.day) not in _QUARTER_PERIOD_START_MD]
    if bad:
        sample = ", ".join(str(x.date()) for x in bad[:5])
        raise ValueError(f"非标准季报季末 index: {sample} ...")


def quarter_period_start(trade_day: pd.Timestamp) -> pd.Timestamp:
    """交易日所属报告区间的首日（严格 PIT，纯日历边界）。"""
    ts = pd.Timestamp(trade_day).normalize()
    month = ts.month
    if month <= 3:
        return pd.Timestamp(ts.year, 1, 1)
    if month <= 6:
        return pd.Timestamp(ts.year, 4, 1)
    if month <= 9:
        return pd.Timestamp(ts.year, 7, 1)
    return pd.Timestamp(ts.year, 10, 1)


def _map_disclosure_to_effective_trade_dates(
    events: pd.DataFrame,
    trade_dates_by_inst: dict[str, pd.DatetimeIndex],
) -> pd.DataFrame:
    """披露日历日 → 信息可交易的首个交易日（D 之后首个 bar）。"""
    chunks: list[pd.DataFrame] = []
    meta_cols = {"report_end", "instrument", "disclosure"}
    value_cols = [c for c in events.columns if c not in meta_cols]

    for inst, grp in events.groupby("instrument", sort=False):
        trade_dates = trade_dates_by_inst.get(inst)
        if trade_dates is None or len(trade_dates) == 0:
            continue

        td_arr = trade_dates.values.astype("datetime64[ns]")
        disc = pd.to_datetime(grp["disclosure"]).values.astype("datetime64[ns]")
        pos = np.searchsorted(td_arr, disc, side="right")
        ok = pos < len(td_arr)
        if not ok.any():
            continue

        part = grp.loc[ok, list(meta_cols | set(value_cols))].copy()
        part["datetime"] = trade_dates.take(pos[ok])
        chunks.append(part)

    if not chunks:
        return pd.DataFrame(columns=["datetime", "instrument", *value_cols])
    return pd.concat(chunks, ignore_index=True)


def expand_quarterly_fundamentals_pit(
    panel: pd.DataFrame,
    fundamentals_path: Path | str,
    disclosure_map_path: Path | str,
    *,
    dtype: str = "float32",
) -> pd.DataFrame:
    """季末基本面 + 披露映射 → 严格 PIT 日频宽表，left join 到 panel。"""
    if panel.index.names != ["datetime", "instrument"]:
        raise ValueError(f"panel 索引须为 (datetime, instrument)，当前: {panel.index.names}")

    raw = _load_quarterly_fundamentals(fundamentals_path)
    value_cols = list(raw.columns)
    overlap = set(panel.columns) & set(value_cols)
    if overlap:
        raise ValueError(f"panel 已含基本面列: {sorted(overlap)}")

    validate_quarter_report_ends(disclosure_map_path)

    cal = load_disclosure_calendar(disclosure_map_path)
    events = raw.reset_index().merge(cal, on=["report_end", "instrument"], how="inner")

    trade_dates_by_inst = {
        inst: grp.index.get_level_values("datetime").unique().sort_values()
        for inst, grp in panel.groupby(level="instrument", sort=False)
    }
    effective = _map_disclosure_to_effective_trade_dates(events, trade_dates_by_inst)
    if effective.empty:
        out = panel.copy()
        for col in value_cols:
            out[col] = np.nan
        return out

    effective = effective.sort_values(["instrument", "datetime", "report_end"])
    effective = effective.drop_duplicates(["instrument", "datetime"], keep="last")

    chunks: list[pd.DataFrame] = []
    for inst, grp in effective.groupby("instrument", sort=False):
        trade_dates = trade_dates_by_inst[inst]
        daily = grp.set_index("datetime")[value_cols].reindex(trade_dates).ffill()
        idx = pd.MultiIndex.from_product(
            [trade_dates, [inst]],
            names=["datetime", "instrument"],
        )
        chunks.append(daily.set_index(idx))

    expanded = pd.concat(chunks).sort_index()
    out = panel.join(expanded, how="left")
    for col in value_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(dtype)
    return out.sort_index()


def _disclosure_effective_trade_positions(
    trade_dates: pd.DatetimeIndex,
    disclosure_dates: np.ndarray,
) -> np.ndarray:
    """实际披露日历日 → 信息可交易的首个交易日位置（严格晚于披露日的下一交易日）。"""
    td = trade_dates.values.astype("datetime64[ns]")
    disc = np.sort(np.unique(disclosure_dates.astype("datetime64[ns]")))
    if disc.size == 0:
        return np.array([], dtype=np.int64)

    pos = np.searchsorted(td, disc, side="right")
    pos = pos[(pos > 0) & (pos < len(td))]
    return np.unique(pos.astype(np.int64))


def _disclosure_days_since(
    trade_dates: pd.DatetimeIndex,
    disclosure_dates: np.ndarray,
) -> np.ndarray:
    """按单只股票交易日历，计算距上一期实际披露生效日的交易日天数。"""
    n = len(trade_dates)
    if n == 0:
        return np.array([], dtype=float)

    disc_pos = _disclosure_effective_trade_positions(trade_dates, disclosure_dates)
    if disc_pos.size == 0:
        return np.full(n, np.nan)

    idx = np.arange(n, dtype=np.int64)
    prev_i = np.searchsorted(disc_pos, idx, side="right") - 1
    days_since = np.full(n, np.nan, dtype=float)
    has_prev = prev_i >= 0
    days_since[has_prev] = idx[has_prev] - disc_pos[prev_i[has_prev]]
    return days_since


def _days_since_quarter_start(trade_dates: pd.DatetimeIndex) -> np.ndarray:
    """各交易日距所属季报区间首日的交易日天数（区间内首日=0）。"""
    n = len(trade_dates)
    if n == 0:
        return np.array([], dtype=float)

    td_ns = trade_dates.values.astype("datetime64[ns]")
    starts = np.array(
        [np.datetime64(quarter_period_start(pd.Timestamp(d)), "ns") for d in trade_dates],
        dtype="datetime64[ns]",
    )
    idx = np.arange(n, dtype=np.int64)
    pos = np.searchsorted(td_ns, starts, side="left")
    out = np.full(n, np.nan, dtype=float)
    ok = pos <= idx
    out[ok] = (idx[ok] - pos[ok]).astype(float)
    return out


def append_disclosure_distance_features(
    panel: pd.DataFrame,
    disclosure_map_path: Path | str,
    *,
    dtype: str = "float32",
) -> pd.DataFrame:
    """并入距上一期财报实际披露生效日的交易日天数。"""
    if panel.index.names != ["datetime", "instrument"]:
        raise ValueError(f"panel 索引须为 (datetime, instrument)，当前: {panel.index.names}")

    overlap = set(panel.columns) & set(DISCLOSURE_DISTANCE_COLUMNS)
    if overlap:
        raise ValueError(f"panel 已含披露距离列: {sorted(overlap)}")

    path = Path(disclosure_map_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"披露日映射文件不存在: {path}")

    cal = load_disclosure_calendar(path)
    disc_by_inst = {
        inst: np.sort(grp["disclosure"].unique())
        for inst, grp in cal.groupby("instrument", sort=False)
    }

    since_chunks: list[pd.Series] = []
    for inst, grp in panel.groupby(level="instrument", sort=False):
        trade_dates = grp.index.get_level_values("datetime").unique().sort_values()
        disc = disc_by_inst.get(inst)
        if disc is None or len(disc) == 0:
            since = np.full(len(trade_dates), np.nan)
        else:
            since = _disclosure_days_since(trade_dates, disc)
        idx = pd.MultiIndex.from_product(
            [trade_dates, [inst]],
            names=["datetime", "instrument"],
        )
        since_chunks.append(pd.Series(since, index=idx, dtype=float))

    since_s = pd.concat(since_chunks).sort_index()
    out = panel.copy()
    out["funda_days_since_disclose"] = since_s.astype(dtype)
    return out.sort_index()


def append_quarter_period_features(
    panel: pd.DataFrame,
    disclosure_map_path: Path | str | None = None,
    *,
    dtype: str = "float32",
) -> pd.DataFrame:
    """并入距当前季报区间首日的交易日天数（与披露映射表季报划分一致，严格 PIT）。"""
    if panel.index.names != ["datetime", "instrument"]:
        raise ValueError(f"panel 索引须为 (datetime, instrument)，当前: {panel.index.names}")

    if "funda_days_since_quarter_start" in panel.columns:
        raise ValueError("panel 已含 funda_days_since_quarter_start")

    if disclosure_map_path is not None:
        validate_quarter_report_ends(disclosure_map_path)

    chunks: list[pd.Series] = []
    for inst, grp in panel.groupby(level="instrument", sort=False):
        trade_dates = grp.index.get_level_values("datetime").unique().sort_values()
        vals = _days_since_quarter_start(trade_dates)
        idx = pd.MultiIndex.from_product(
            [trade_dates, [inst]],
            names=["datetime", "instrument"],
        )
        chunks.append(pd.Series(vals, index=idx, dtype=float))

    out = panel.copy()
    out["funda_days_since_quarter_start"] = pd.concat(chunks).sort_index().astype(dtype)
    return out.sort_index()


def enrich_panel_fundamentals(
    panel: pd.DataFrame,
    *,
    quarterly_path: Path | str = FUNDAMENTAL_QUARTERLY_PATH,
    disclosure_path: Path | str = DISCLOSURE_CALENDAR_PATH,
    include_disclosure_features: bool = True,
    dtype: str = "float32",
) -> pd.DataFrame:
    """将季频基本面 PIT 展开并 left join 到 panel。

    幂等：先删除 panel 中已有的 ``funda_*`` 列再重新展开，因此重复 enrich、
    或缓存新增列（如 ``--with-statements`` 后的 ``funda_fs_*``）时都会全量刷新。
    """
    stale = [c for c in panel.columns if str(c).startswith("funda_")]
    if stale:
        panel = panel.drop(columns=stale)

    panel = expand_quarterly_fundamentals_pit(
        panel,
        quarterly_path,
        disclosure_path,
        dtype=dtype,
    )
    if include_disclosure_features:
        panel = append_disclosure_distance_features(
            panel,
            disclosure_path,
            dtype=dtype,
        )
        panel = append_quarter_period_features(
            panel,
            disclosure_path,
            dtype=dtype,
        )
    return panel


def list_funda_columns(columns: Sequence[str]) -> list[str]:
    """返回列名中的基本面相关列。"""
    return sorted(
        c
        for c in columns
        if str(c).startswith("funda_")
    )

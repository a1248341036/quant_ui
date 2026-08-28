"""从 Tushare 拉取季频财务指标并写入 quarterly / disclosure 缓存。"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from alphaagent.core.paths import DISCLOSURE_CALENDAR_PATH, FUNDAMENTAL_DIR, FUNDAMENTAL_QUARTERLY_PATH
from alphaagent.data.fundamental import validate_quarter_report_ends
from alphaagent.data.tushare_client import call_with_retry, get_pro

# fina_indicator 字段 → panel 列名
FINA_INDICATOR_COLUMN_MAP: dict[str, str] = {
    "roe": "funda_roe",
    "roa": "funda_roa",
    "debt_to_assets": "funda_debt_to_assets",
    "netprofit_yoy": "funda_netprofit_yoy",
    "or_yoy": "funda_or_yoy",
    "tr_yoy": "funda_tr_yoy",
    "bps": "funda_bps",
    "eps": "funda_eps",
    "grossprofit_margin": "funda_grossprofit_margin",
    "netprofit_margin": "funda_netprofit_margin",
    "ocfps": "funda_ocfps",
    "working_capital": "funda_fs_working_capital",
    "ebit": "funda_fs_ebit",
    "rd_exp": "funda_fs_rd_exp",
    "profit_dedt": "funda_profit_dedt",
    "current_ratio": "funda_current_ratio",
    "quick_ratio": "funda_quick_ratio",
}

FINA_INDICATOR_API_FIELDS = (
    "ts_code,ann_date,end_date,"
    + ",".join(FINA_INDICATOR_COLUMN_MAP.keys())
)

# ----------------------------------------------------------------------------
# 三大表（income / balancesheet / cashflow）字段 → panel 列名
#
# 口径说明（按 Tushare 原始值存储，不做单季差分）：
#   * 资产负债表：时点值（无后缀）。
#   * 利润表 / 现金流量表：Tushare 返回**年初至今累计值**，列名统一带 `_ytd`
#     后缀以示区分（Q1=当季，H1/Q3/年报为累计）；期末/期初现金余额为时点值。
# 仅取 report_type='1'（合并报表）；同 (ts_code, end_date) 保留 ann_date 最新一条。
# ----------------------------------------------------------------------------

INCOME_COLUMN_MAP: dict[str, str] = {
    "total_revenue": "funda_fs_total_revenue_ytd",
    "revenue": "funda_fs_oper_revenue_ytd",
    "total_cogs": "funda_fs_total_cogs_ytd",
    "oper_cost": "funda_fs_oper_cost_ytd",
    "sell_exp": "funda_fs_selling_expense_ytd",
    "admin_exp": "funda_fs_admin_expense_ytd",
    "fin_exp": "funda_fs_finance_expense_ytd",
    "int_exp": "funda_fs_interest_expense_ytd",
    "biz_tax_surchg": "funda_fs_tax_surcharge_ytd",
    "operate_profit": "funda_fs_operate_profit_ytd",
    "total_profit": "funda_fs_total_profit_ytd",
    "income_tax": "funda_fs_income_tax_ytd",
    "n_income": "funda_fs_net_profit_ytd",
    "n_income_attr_p": "funda_fs_net_profit_parent_ytd",
    "minority_gain": "funda_fs_minority_interest_ytd",
    "t_compr_income": "funda_fs_comprehensive_income_ytd",
    "compr_inc_attr_p": "funda_fs_comprehensive_income_parent_ytd",
    "basic_eps": "funda_fs_eps_basic_ytd",
    "diluted_eps": "funda_fs_eps_diluted_ytd",
}

BALANCESHEET_COLUMN_MAP: dict[str, str] = {
    "total_assets": "funda_fs_total_assets",
    "total_cur_assets": "funda_fs_current_assets",
    "total_nca": "funda_fs_noncurrent_assets",
    "total_liab": "funda_fs_total_liabilities",
    "total_cur_liab": "funda_fs_current_liabilities",
    "total_ncl": "funda_fs_noncurrent_liabilities",
    "total_hldr_eqy_exc_min_int": "funda_fs_total_equity",
    "total_hldr_eqy_inc_min_int": "funda_fs_total_equity_incl_mi",
    "total_liab_hldr_eqy": "funda_fs_total_liab_equity",
    "minority_int": "funda_fs_minority_interest_equity",
    "money_cap": "funda_fs_money_cap",
    "notes_receiv": "funda_fs_notes_receivable",
    "accounts_receiv": "funda_fs_accounts_receivable",
    "inventories": "funda_fs_inventories",
    "fix_assets": "funda_fs_fixed_assets",
    "cip": "funda_fs_construction_in_progress",
    "intan_assets": "funda_fs_intangible_assets",
    "goodwill": "funda_fs_goodwill",
    "r_and_d": "funda_fs_rd_capitalized",
    "st_borr": "funda_fs_short_term_borrow",
    "lt_borr": "funda_fs_long_term_borrow",
    "bond_payable": "funda_fs_bond_payable",
    "notes_payable": "funda_fs_notes_payable",
    "acct_payable": "funda_fs_accounts_payable",
    "adv_receipts": "funda_fs_advance_receipts",
    "taxes_payable": "funda_fs_taxes_payable",
    "payroll_payable": "funda_fs_payroll_payable",
    "oth_payable": "funda_fs_other_payables",
    "undistr_porfit": "funda_fs_retained_earnings",
    "surplus_rese": "funda_fs_surplus_reserve",
    "cap_rese": "funda_fs_capital_reserve",
    "total_share": "funda_fs_total_share",
    "oth_comp_income": "funda_fs_other_comprehensive_income",
}

CASHFLOW_COLUMN_MAP: dict[str, str] = {
    "c_fr_sale_sg": "funda_fs_cash_from_sales_ytd",
    "c_inf_fr_operate_a": "funda_fs_ocf_inflow_ytd",
    "c_paid_goods_s": "funda_fs_cash_paid_goods_ytd",
    "c_paid_to_for_empl": "funda_fs_cash_paid_employees_ytd",
    "c_paid_for_taxes": "funda_fs_cash_paid_taxes_ytd",
    "st_cash_out_act": "funda_fs_ocf_outflow_ytd",
    "n_cashflow_act": "funda_fs_ocf_net_ytd",
    "c_pay_acq_const_fiolta": "funda_fs_capex_ytd",
    "c_paid_invest": "funda_fs_cash_paid_invest_ytd",
    "n_cashflow_inv_act": "funda_fs_icf_net_ytd",
    "c_recp_borrow": "funda_fs_cash_from_borrow_ytd",
    "c_prepay_amt_borr": "funda_fs_cash_repay_debt_ytd",
    "n_cash_flows_fnc_act": "funda_fs_fcf_net_ytd",
    "free_cashflow": "funda_fs_free_cashflow_ytd",
    "n_incr_cash_cash_equ": "funda_fs_cash_net_incr_ytd",
    "depr_fa_coga_dpba": "funda_fs_depreciation_ytd",
    "amort_intang_assets": "funda_fs_amortization_intangible_ytd",
    "im_net_cashflow_oper_act": "funda_fs_ocf_indirect_ytd",
    "c_cash_equ_beg_period": "funda_fs_cash_equiv_beg",
    "c_cash_equ_end_period": "funda_fs_cash_equiv_end",
}


class StatementSpec:
    """一张财报表的拉取规格。"""

    __slots__ = ("name", "api", "vip_api", "column_map")

    def __init__(self, name: str, api: str, vip_api: str, column_map: dict[str, str]) -> None:
        self.name = name
        self.api = api
        self.vip_api = vip_api
        self.column_map = column_map

    @property
    def api_fields(self) -> str:
        return "ts_code,ann_date,end_date,report_type," + ",".join(self.column_map.keys())


STATEMENT_SPECS: tuple[StatementSpec, ...] = (
    StatementSpec("income", "income", "income_vip", INCOME_COLUMN_MAP),
    StatementSpec("balancesheet", "balancesheet", "balancesheet_vip", BALANCESHEET_COLUMN_MAP),
    StatementSpec("cashflow", "cashflow", "cashflow_vip", CASHFLOW_COLUMN_MAP),
)

_STANDARD_QUARTER_ENDS = ("0331", "0630", "0930", "1231")
_QUARTER_END_MD = ((3, 31), (6, 30), (9, 30), (12, 31))


def quarter_periods_between(start: str, end: str) -> list[str]:
    """返回 [start, end] 内所有标准 A 股季报季末（YYYYMMDD）。"""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if start_ts > end_ts:
        raise ValueError(f"start 不能晚于 end: {start} > {end}")

    periods: list[str] = []
    for year in range(start_ts.year, end_ts.year + 1):
        for month, day in _QUARTER_END_MD:
            qe = pd.Timestamp(year=year, month=month, day=day)
            if start_ts <= qe <= end_ts:
                periods.append(qe.strftime("%Y%m%d"))
    return periods


def _normalize_period(period: str) -> str:
    p = period.replace("-", "")
    if len(p) != 8:
        raise ValueError(f"period 须为 YYYYMMDD，收到: {period!r}")
    if p[4:] not in _STANDARD_QUARTER_ENDS:
        raise ValueError(f"非标准季报季末: {period!r}")
    return p


def _dedupe_fina_raw(df: pd.DataFrame) -> pd.DataFrame:
    """同一 (ts_code, end_date) 保留 ann_date 最新的一条。"""
    if df.empty:
        return df
    out = df.copy()
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out.dropna(subset=["ts_code", "end_date", "ann_date"])
    out = out.sort_values(["ts_code", "end_date", "ann_date"])
    return out.groupby(["ts_code", "end_date"], as_index=False).tail(1)


def fetch_fina_indicator_period(
    period: str,
    *,
    ts_codes: list[str] | None = None,
    sleep_sec: float = 0.35,
    verbose: bool = True,
    use_vip: bool = True,
) -> pd.DataFrame:
    """拉取单个报告期的 fina_indicator 原始表。

    use_vip=True 时用 ``fina_indicator_vip`` 拉**全市场**（每期 1 次请求，完整落盘）。
    use_vip=False 时按 ts_codes 逐股拉取（须指定股票列表）。
    """
    period = _normalize_period(period)
    pro = get_pro()

    if use_vip:
        if verbose:
            print(f"  fina_indicator_vip period={period}（全市场）")
        raw = call_with_retry(
            pro.fina_indicator_vip,
            period=period,
            fields=FINA_INDICATOR_API_FIELDS,
            label=f"fina_indicator_vip_{period}",
        )
        time.sleep(sleep_sec)
        return _dedupe_fina_raw(raw)

    if not ts_codes:
        raise ValueError("无 VIP 权限时须指定 ts_codes（--no-vip 且 --universe）")

    chunks: list[pd.DataFrame] = []
    n = len(ts_codes)
    for i, code in enumerate(ts_codes):
        if verbose and (i == 0 or (i + 1) % 50 == 0 or i + 1 == n):
            print(f"  fina_indicator [{i + 1}/{n}] {code} period={period}")
        part = call_with_retry(
            pro.fina_indicator,
            ts_code=code,
            period=period,
            fields=FINA_INDICATOR_API_FIELDS,
            label=f"fina_indicator_{code}_{period}",
        )
        if part is not None and not part.empty:
            chunks.append(part)
        time.sleep(sleep_sec)

    if not chunks:
        return pd.DataFrame()
    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        return pd.DataFrame()
    return _dedupe_fina_raw(pd.concat(non_empty, ignore_index=True))


def raw_fina_to_quarterly(raw: pd.DataFrame) -> pd.DataFrame:
    """fina_indicator 原始表 → (report_end, instrument) 索引的 panel 列。"""
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["instrument"] = df["ts_code"]
    df["report_end"] = pd.to_datetime(df["end_date"])
    rename = {k: v for k, v in FINA_INDICATOR_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    value_cols = list(rename.values())
    out = df.set_index(["report_end", "instrument"])[value_cols]
    return out.sort_index()


def fetch_statement_period(
    spec: StatementSpec,
    period: str,
    *,
    ts_codes: list[str] | None = None,
    sleep_sec: float = 0.35,
    verbose: bool = True,
    use_vip: bool = True,
) -> pd.DataFrame:
    """拉取单个报告期的一张三大表原始数据（income/balancesheet/cashflow）。

    use_vip=True 时用 ``*_vip`` 接口拉**全市场**（每期 1 次请求）。
    use_vip=False 时按 ts_codes 逐股拉取（须指定股票列表）。
    """
    period = _normalize_period(period)
    pro = get_pro()

    if use_vip:
        if verbose:
            print(f"  {spec.vip_api} period={period}（全市场）")
        raw = call_with_retry(
            getattr(pro, spec.vip_api),
            period=period,
            fields=spec.api_fields,
            label=f"{spec.vip_api}_{period}",
        )
        time.sleep(sleep_sec)
        return _dedupe_statement_raw(raw)

    if not ts_codes:
        raise ValueError("无 VIP 权限时须指定 ts_codes（--no-vip 且 --universe）")

    chunks: list[pd.DataFrame] = []
    n = len(ts_codes)
    for i, code in enumerate(ts_codes):
        if verbose and (i == 0 or (i + 1) % 50 == 0 or i + 1 == n):
            print(f"  {spec.api} [{i + 1}/{n}] {code} period={period}")
        part = call_with_retry(
            getattr(pro, spec.api),
            ts_code=code,
            period=period,
            fields=spec.api_fields,
            label=f"{spec.api}_{code}_{period}",
        )
        if part is not None and not part.empty:
            chunks.append(part)
        time.sleep(sleep_sec)

    non_empty = [c for c in chunks if not c.empty]
    if not non_empty:
        return pd.DataFrame()
    return _dedupe_statement_raw(pd.concat(non_empty, ignore_index=True))


def _dedupe_statement_raw(df: pd.DataFrame) -> pd.DataFrame:
    """三大表原始表：仅留合并报表(report_type='1')，同 (ts_code, end_date) 取 ann_date 最新。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "report_type" in out.columns:
        out = out[out["report_type"].astype(str) == "1"]
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out.dropna(subset=["ts_code", "end_date", "ann_date"])
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values(["ts_code", "end_date", "ann_date"])
    return out.groupby(["ts_code", "end_date"], as_index=False).tail(1)


def raw_statement_to_quarterly(raw: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """三大表原始表 → (report_end, instrument) 索引的 panel 列（保留原始值）。

    内部先做 report_type='1' 过滤 + 同 (ts_code, end_date) 取 ann_date 最新，
    因此对未去重的原始表也安全（幂等）。
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = _dedupe_statement_raw(raw)
    if df.empty:
        return pd.DataFrame()
    df["instrument"] = df["ts_code"]
    df["report_end"] = pd.to_datetime(df["end_date"])
    rename = {k: v for k, v in column_map.items() if k in df.columns}
    if not rename:
        return pd.DataFrame()
    df = df.rename(columns=rename)
    value_cols = list(rename.values())
    out = df.set_index(["report_end", "instrument"])[value_cols]
    out = out.apply(pd.to_numeric, errors="coerce")
    return out[~out.index.duplicated(keep="last")].sort_index()


def _join_quarterly_columns(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """按 (report_end, instrument) 索引列向合并（outer join）。"""
    if base.empty:
        return extra
    if extra.empty:
        return base
    dup = [c for c in extra.columns if c in base.columns]
    extra_clean = extra.drop(columns=dup) if dup else extra
    return base.join(extra_clean, how="outer")


def raw_fina_to_disclosure_events(raw: pd.DataFrame) -> pd.DataFrame:
    """从 fina_indicator 提取披露日历 long 表。"""
    if raw.empty:
        return pd.DataFrame(columns=["report_end", "instrument", "disclosure"])

    df = raw.copy()
    df["report_end"] = pd.to_datetime(df["end_date"])
    df["instrument"] = df["ts_code"]
    df["disclosure"] = pd.to_datetime(df["ann_date"], errors="coerce")
    return df[["report_end", "instrument", "disclosure"]].dropna()


def disclosure_events_to_wide(events: pd.DataFrame) -> pd.DataFrame:
    """long 披露表 → 宽表 (report_end × instrument)。"""
    if events.empty:
        return pd.DataFrame()
    wide = events.pivot_table(
        index="report_end",
        columns="instrument",
        values="disclosure",
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def merge_quarterly(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """合并季频缓存，同键以 new 为准。"""
    if existing.empty:
        return new.sort_index()
    if new.empty:
        return existing.sort_index()
    combined = pd.concat([existing, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def merge_disclosure_wide(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """合并披露宽表：新数据覆盖同 (report_end, instrument) 单元格。"""
    if existing.empty:
        return new.sort_index()
    if new.empty:
        return existing.sort_index()

    all_index = existing.index.union(new.index)
    all_cols = existing.columns.union(new.columns)
    base = existing.reindex(index=all_index, columns=all_cols)
    overlay = new.reindex(index=all_index, columns=all_cols)
    return base.combine_first(overlay).sort_index()


def save_quarterly(df: pd.DataFrame, path: Path | str = FUNDAMENTAL_QUARTERLY_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


def save_disclosure_calendar(wide: pd.DataFrame, path: Path | str = DISCLOSURE_CALENDAR_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(out)
    if not wide.empty:
        validate_quarter_report_ends(out)
    return out


def load_quarterly_cache(path: Path | str = FUNDAMENTAL_QUARTERLY_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if df.index.names != ["report_end", "instrument"]:
        if "datetime" in df.index.names:
            df = df.rename_axis(index={"datetime": "report_end"})
        if "code" in df.index.names:
            df = df.rename_axis(index={"code": "instrument"})
    return df.sort_index()


def load_disclosure_wide(path: Path | str = DISCLOSURE_CALENDAR_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame()
    wide = pd.read_parquet(p)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def fetch_and_save_periods(
    periods: list[str],
    *,
    ts_codes: list[str] | None = None,
    quarterly_path: Path | str = FUNDAMENTAL_QUARTERLY_PATH,
    disclosure_path: Path | str = DISCLOSURE_CALENDAR_PATH,
    sleep_sec: float = 0.35,
    verbose: bool = True,
    use_vip: bool = True,
    with_statements: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拉取多个报告期并增量写入缓存。

    with_statements=True 时额外拉取三大表（income/balancesheet/cashflow），
    按 (report_end, instrument) 列向合并进同一份季频缓存；PIT 展开逻辑通用，
    这些 ``funda_fs_*`` 列会随 fina 指标一起展开为日频。
    """
    quarterly_acc = load_quarterly_cache(quarterly_path)
    disclosure_acc = load_disclosure_wide(disclosure_path)

    for period in periods:
        period = _normalize_period(period)
        if verbose:
            print(f"拉取 fina_indicator: {period}")
        raw = fetch_fina_indicator_period(
            period,
            ts_codes=ts_codes,
            sleep_sec=sleep_sec,
            verbose=verbose,
            use_vip=use_vip,
        )

        q = raw_fina_to_quarterly(raw) if not raw.empty else pd.DataFrame()
        wide = disclosure_events_to_wide(raw_fina_to_disclosure_events(raw)) if not raw.empty else pd.DataFrame()

        if with_statements:
            for spec in STATEMENT_SPECS:
                if verbose:
                    print(f"拉取 {spec.name}: {period}")
                raw_s = fetch_statement_period(
                    spec,
                    period,
                    ts_codes=ts_codes,
                    sleep_sec=sleep_sec,
                    verbose=verbose,
                    use_vip=use_vip,
                )
                q = _join_quarterly_columns(q, raw_statement_to_quarterly(raw_s, spec.column_map))
                if wide.empty and not raw_s.empty:
                    wide = disclosure_events_to_wide(raw_fina_to_disclosure_events(raw_s))

        if q.empty:
            if verbose:
                print(f"  警告: {period} 无数据")
            continue

        quarterly_acc = merge_quarterly(quarterly_acc, q)
        if not wide.empty:
            disclosure_acc = merge_disclosure_wide(disclosure_acc, wide)
        save_quarterly(quarterly_acc, quarterly_path)
        save_disclosure_calendar(disclosure_acc, disclosure_path)
        if verbose:
            n_inst = q.index.get_level_values("instrument").nunique()
            print(
                f"  本期 +{len(q)} 条（{n_inst} 只股票，{q.shape[1]} 列）"
                f" → 已落盘 cumulative={quarterly_acc.shape}"
            )

    if verbose:
        print(f"季频缓存: {quarterly_path} shape={quarterly_acc.shape}")
        print(f"披露缓存: {disclosure_path} shape={disclosure_acc.shape}")
    return quarterly_acc, disclosure_acc


def ensure_fundamental_dir() -> Path:
    FUNDAMENTAL_DIR.mkdir(parents=True, exist_ok=True)
    return FUNDAMENTAL_DIR

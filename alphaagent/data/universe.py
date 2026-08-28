"""Universe：指数成分、ST/停牌过滤。"""



from __future__ import annotations



import time

import warnings



import numpy as np

import pandas as pd



# 常用指数 Tushare 代码

INDEX_CODES: dict[str, str] = {

    "zz1000": "000852.SH",  # 中证1000

    "zz500": "000905.SH",

    "hs300": "000300.SH",

}





def resolve_index_code(name: str) -> str:

    """解析指数别名或原始 ts_code。"""

    key = name.strip().lower()

    if key in INDEX_CODES:

        return INDEX_CODES[key]

    if name.endswith(".SH") or name.endswith(".SZ"):

        return name

    raise ValueError(f"未知指数: {name}，可选: {list(INDEX_CODES)}")





def _to_yyyymmdd(d: str) -> str:

    return d.replace("-", "")[:8]





def _members_from_index_member(

    df: pd.DataFrame,

    start: str,

    end: str,

) -> list[str]:

    """

    从 index_member 结果筛选与 [start, end] 有交集的成分股。

    字段: con_code, in_date, out_date（out_date 空表示仍在成分内）

    """

    start_s = _to_yyyymmdd(start)

    end_s = _to_yyyymmdd(end)

    out: set[str] = set()



    for _, row in df.iterrows():

        code = row.get("con_code")

        if not code or pd.isna(code):

            continue

        in_d = str(row.get("in_date", "")).replace("-", "")[:8]

        if not in_d or in_d == "nan":

            continue



        raw_out = row.get("out_date")

        if raw_out is None or (isinstance(raw_out, float) and pd.isna(raw_out)):

            out_d = "99991231"

        else:

            out_d = str(raw_out).replace("-", "")[:8]

            if not out_d or out_d == "nan":

                out_d = "99991231"



        # 区间有交集: in_date <= end 且 out_date >= start

        if in_d <= end_s and out_d >= start_s:

            out.add(str(code))



    return sorted(out)





def _fetch_members_index_member(

    pro,

    index_code: str,

    start: str,

    end: str,

) -> list[str]:

    """pro.index_member：含 in_date/out_date 的全历史成分。"""

    df = pro.index_member(index_code=index_code, is_new="")

    if df is None or df.empty:

        return []

    return _members_from_index_member(df, start, end)





def _fetch_members_index_weight_range(
    pro,
    index_code: str,
    start: str,
    end: str,
) -> list[str]:
    """pro.index_weight 区间查询，返回 con_code 并集。"""
    d0 = _to_yyyymmdd(start)
    d1 = _to_yyyymmdd(end)
    df = pro.index_weight(index_code=index_code, start_date=d0, end_date=d1)
    if df is None or df.empty:
        return []
    return sorted(df["con_code"].dropna().astype(str).unique())


def _fetch_members_index_weight_at(
    pro,
    index_code: str,
    trade_date: str,
    *,
    lookback_days: int = 60,
    sleep_sec: float = 0,
) -> list[str]:
    """
    取 trade_date 当日 index_weight；若无数据则向前 lookback_days 内取最近一日快照。
    """
    td = _to_yyyymmdd(trade_date)
    df = pro.index_weight(index_code=index_code, start_date=td, end_date=td)
    if df is not None and not df.empty:
        return sorted(df["con_code"].dropna().astype(str).unique())

    start = (pd.Timestamp(td) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    df = pro.index_weight(index_code=index_code, start_date=start, end_date=td)
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    if df is None or df.empty:
        return []

    latest = str(df["trade_date"].max())
    sub = df[df["trade_date"] == latest]
    return sorted(sub["con_code"].dropna().astype(str).unique())


def _fetch_members_index_weight_monthly(

    pro,

    index_code: str,

    start: str,

    end: str,

    *,

    sleep_sec: float = 0.35,

    verbose: bool = False,

) -> list[str]:

    """

    pro.index_weight 按月快照循环（宽区间查询有 ~7000 行上限，会丢历史）。

    """

    start_ts = pd.Timestamp(start)

    end_ts = pd.Timestamp(end)

    month_starts = pd.date_range(

        start_ts.replace(day=1),

        end_ts.replace(day=1),

        freq="MS",

    )



    seen: set[str] = set()

    for i, m in enumerate(month_starts):

        snap = min(m + pd.offsets.MonthEnd(0), end_ts)

        if snap < start_ts:

            continue

        d = snap.strftime("%Y%m%d")

        df = pro.index_weight(index_code=index_code, start_date=d, end_date=d)

        if df is not None and not df.empty:

            seen |= set(df["con_code"].dropna().astype(str))

        if verbose and (i + 1) % 12 == 0:

            print(f"    index_weight 进度 {i + 1}/{len(month_starts)} 月, 累计 {len(seen)} 只")

        if sleep_sec > 0:

            time.sleep(sleep_sec)



    return sorted(seen)





def fetch_index_members(

    pro,

    index: str,

    start: str,

    end: str,

    *,

    sleep_sec: float = 0.35,

    verbose: bool = True,

) -> list[str]:

    """

    获取 [start, end] 期间曾出现在指数中的全部股票（并集）。



    优先 index_member（纳入/剔除日期，适合长历史）；

    若无数据则回退 index_weight 按月循环。

    """

    index_code = resolve_index_code(index)



    members = _fetch_members_index_member(pro, index_code, start, end)

    source = "index_member"

    if not members:

        if verbose:

            print(f"  index_member 无数据，回退 index_weight...")

        span = (pd.Timestamp(end) - pd.Timestamp(start)).days

        if span <= 62:

            members = _fetch_members_index_weight_range(pro, index_code, start, end)

            source = "index_weight(range)"

            if not members:

                members = _fetch_members_index_weight_at(

                    pro, index_code, end, lookback_days=90, sleep_sec=sleep_sec

                )

                source = "index_weight(latest)"

        if not members:

            members = _fetch_members_index_weight_monthly(

                pro,

                index_code,

                start,

                end,

                sleep_sec=sleep_sec,

                verbose=verbose,

            )

            source = "index_weight(monthly)"



    if not members:

        raise ValueError(

            f"无法获取指数成分: {index_code} {start} ~ {end}，"

            "请检查 Tushare 积分（index_member / index_weight）"

        )



    if verbose:

        print(f"  成分来源: {source}, 共 {len(members)} 只")

    return members





def fetch_index_members_for_dates(

    pro,

    index: str,

    dates: list[str],

    *,

    lookback_days: int = 60,

    sleep_sec: float = 0.35,

    verbose: bool = True,

) -> set[str]:

    """

    增量更新用：对多个交易日分别取 index_weight 快照并集。

    单日无数据时向前 lookback_days 内取最近可用成分。

    """

    index_code = resolve_index_code(index)

    pool: set[str] = set()

    for d in sorted({_to_yyyymmdd(x) for x in dates}):

        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        members = _fetch_members_index_weight_at(

            pro,

            index_code,

            iso,

            lookback_days=lookback_days,

            sleep_sec=sleep_sec,

        )

        if members:

            pool |= set(members)

            if verbose:

                print(f"  {iso} 成分 {len(members)} 只")

        elif verbose:

            print(f"  警告: {iso} index_weight 无成分数据")

        if sleep_sec > 0:

            time.sleep(sleep_sec)



    if not pool:

        raise ValueError(

            f"无法获取指数成分: {index_code} dates={dates}，"

            "请检查 Tushare 积分（index_weight）或指定 --dates 为已有数据的交易日"

        )

    if verbose:

        print(f"  成分来源: index_weight(按日), 并集 {len(pool)} 只")

    return pool





def fetch_st_table(

    pro,

    *,

    trade_date: str | None = None,

    start_date: str | None = None,

    end_date: str | None = None,

) -> pd.DataFrame:

    """

    拉取 Tushare stock_st，返回 ts_code / trade_date / is_st(=1) 表。

    无 ST 记录时返回空表（列齐全）。

    """

    kwargs: dict[str, str] = {}

    if trade_date is not None:

        kwargs["trade_date"] = _to_yyyymmdd(trade_date)

    if start_date is not None:

        kwargs["start_date"] = _to_yyyymmdd(start_date)

    if end_date is not None:

        kwargs["end_date"] = _to_yyyymmdd(end_date)



    empty = pd.DataFrame(columns=["ts_code", "trade_date", "is_st"])

    if not kwargs:

        return empty



    try:

        df = pro.stock_st(fields="ts_code,trade_date", **kwargs)

    except Exception as exc:

        warnings.warn(f"stock_st 拉取失败: {exc}，当日 is_st 按 0 处理", stacklevel=2)

        return empty



    if df is None or df.empty:

        return empty



    out = df[["ts_code", "trade_date"]].drop_duplicates().copy()

    out["is_st"] = np.int8(1)

    return out





def apply_is_st(df: pd.DataFrame, st_table: pd.DataFrame) -> pd.DataFrame:

    """按日 stock_st 结果写入 is_st / not_st。"""

    out = df.copy()

    if st_table is None or st_table.empty:

        out["is_st"] = np.int8(0)

    else:

        st = st_table[["ts_code", "trade_date", "is_st"]].drop_duplicates()

        out = out.merge(st, on=["ts_code", "trade_date"], how="left")

        out["is_st"] = out["is_st"].fillna(0).astype(np.int8)

    out["not_st"] = (1 - out["is_st"]).astype(np.int8)

    return out





def mark_not_st(names: pd.Series) -> pd.Series:

    """根据股票名称标记 not_st（1=非ST，0=ST）。仅作兼容/测试，生产请用 apply_is_st。"""

    is_st = names.str.contains(r"ST", case=False, na=False)

    return (~is_st).astype("int8")





def filter_universe(df: pd.DataFrame, *, universe_mask: bool = True) -> pd.DataFrame:

    """过滤可交易、非 ST 样本（优先 is_st 日度标记）。"""

    if not universe_mask:

        return df

    if "is_st" in df.columns:

        st_ok = df["is_st"] == 0

    else:

        st_ok = df["not_st"] == 1

    return df[(df["is_trade"] == 1) & st_ok]



# -*- coding: utf-8 -*-
"""财务数据(聚宽文档类别: 数据获取函数 -> finance 数据查询)。

把 jqdata.finance 从占位桩升级为真实现:
- finance.STK_XR_XD   分红送转(pg_parquet/dividend.parquet 映射)
- finance.run_query(q) 通用财务表查询(过滤/选列/排序/limit)

聚宽字段 -> 本地口径映射(STK_XR_XD):
- code                 tushare ts_code 前 6 位
- company_name         stock_basic.name
- report_date          end_date(报告期)
- board_plan_pub_date  ann_date(公告日, JQ 为预案公告日, 近似)
- dividend_ratio       stk_div(送转比例)
- bonus_ratio_rmb      cash_div × 10(每 10 股股息, 元)
- bonus_amount_rmb     cash_div × 总股本 / 1e4 (万元; tushare 无总额字段,
                       总股本取 balancesheet.total_share 同期值)
- record_date/ex_date/pay_date  直接映射

同一次分红在 tushare 中有 预案/股东大会通过/实施 多阶段行:
保留每组 (code, 报告期) 内 cash_div>0 且公告最早的行, 防重复累计。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.event_engine.jq.query import _Col, _Query

_XR_XD_COLUMNS = [
    "code", "company_name", "report_date", "board_plan_pub_date",
    "dividend_ratio", "bonus_ratio_rmb", "bonus_amount_rmb",
    "record_date", "ex_date", "pay_date",
]


class _FinanceTable:
    """JQ finance 表视图: 属性访问返回 query DSL 列。"""

    def __init__(self, name: str, columns: list[str], loader):
        self._name = name
        self._columns = columns
        self._loader = loader
        self._df: pd.DataFrame | None = None

    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._loader()
        return self._df

    def __getattr__(self, name: str) -> _Col:
        if name in object.__getattribute__(self, "_columns"):
            return _Col(name)
        raise AttributeError(
            f"finance.{object.__getattribute__(self, '_name')}.{name} "
            f"未支持(可用列: {', '.join(self._columns)})")


def load_dividend() -> pd.DataFrame:
    """tushare dividend.parquet -> 聚宽 STK_XR_XD 口径(点时公告日)."""
    from scripts.jq_repro import jq_data  # noqa: F401  (确保 sys.path)
    import jq_data

    d = pd.read_parquet(
        jq_data.PG / "dividend.parquet",
        columns=["ts_code", "end_date", "ann_date", "div_proc", "stk_div",
                 "cash_div", "record_date", "ex_date", "pay_date"])
    d["code"] = d["ts_code"].str[:6]
    d["ann_date"] = pd.to_datetime(d["ann_date"], errors="coerce")
    d["end_date"] = pd.to_datetime(d["end_date"], errors="coerce")
    for c in ("record_date", "ex_date", "pay_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["cash_div"] = pd.to_numeric(d["cash_div"], errors="coerce")
    d = d[d["cash_div"] > 0].sort_values(["code", "end_date", "ann_date"],
                                         kind="stable")
    # 同一报告期的 预案/实施 多行只留公告最早的一条, 防止分红重复累计
    d = d.drop_duplicates(["code", "end_date"], keep="first")

    # 总股本(balancesheet.total_share): 优先同期, 回退该股最新一期
    b = pd.read_parquet(jq_data.PG / "balancesheet.parquet",
                        columns=["ts_code", "end_date", "report_type",
                                 "total_share"])
    b = b[pd.to_numeric(b["report_type"], errors="coerce") == 1]
    b["code"] = b["ts_code"].str[:6]
    b["end_date"] = pd.to_datetime(b["end_date"], errors="coerce")
    b = b.dropna(subset=["total_share"]).sort_values(
        ["code", "end_date"], kind="stable")
    b = b.drop_duplicates(["code", "end_date"], keep="last")
    same = b.set_index(["code", "end_date"])["total_share"]
    latest = b.groupby("code")["total_share"].last()
    d["total_share"] = [
        same.get((c, e), latest.get(c, np.nan))
        for c, e in zip(d["code"], d["end_date"])]

    # 公司名
    basic = pd.read_parquet(jq_data.PG / "stock_basic.parquet",
                            columns=["ts_code", "name"])
    basic["code"] = basic["ts_code"].str[:6]
    basic = basic.drop_duplicates("code").set_index("code")["name"]
    d["company_name"] = d["code"].map(basic).fillna("")

    out = pd.DataFrame({
        "code": d["code"].values,
        "company_name": d["company_name"].values,
        # 日期列保持 datetime.date 对象(与聚宽 run_query 返回一致,
        # 用户可直接与 dt.date 比较)
        "report_date": pd.DatetimeIndex(d["end_date"]).date,
        "board_plan_pub_date": pd.DatetimeIndex(d["ann_date"]).date,
        "dividend_ratio": pd.to_numeric(d["stk_div"], errors="coerce").values,
        "bonus_ratio_rmb": d["cash_div"].values * 10.0,
        "bonus_amount_rmb": (d["cash_div"].values
                             * d["total_share"].to_numpy(dtype=float) / 1e4),
        "record_date": pd.DatetimeIndex(d["record_date"]).date,
        "ex_date": pd.DatetimeIndex(d["ex_date"]).date,
        "pay_date": pd.DatetimeIndex(d["pay_date"]).date,
    })
    return out


def run_table_query(q: _Query, df: pd.DataFrame) -> pd.DataFrame:
    """在单张财务表上执行 query DSL: 过滤/选列/排序/limit。"""
    if not len(df):
        return pd.DataFrame(columns=[c.key if hasattr(c, "key") else str(c)
                                     for c in q.cols])
    mask = np.ones(len(df), dtype=bool)
    for expr in q.exprs:
        mask = mask & np.asarray(expr.fn(df), dtype=bool)
    out = df[mask]
    keys = [c.key if isinstance(c, _Col) else str(c) for c in q.cols]
    out = out[[k for k in keys if k in out.columns]]
    if q.order is not None:
        direction, col = q.order
        out = out.sort_values(col.key, ascending=(direction == "asc"),
                              kind="stable")
    if q._limit is not None:
        out = out.head(q._limit)
    return out.reset_index(drop=True)


def install(ns: dict, rt) -> None:
    from . import misc
    misc._setup_jq_modules()
    import sys

    xrxd = _FinanceTable("STK_XR_XD", _XR_XD_COLUMNS, load_dividend)
    _TABLES = {"STK_XR_XD": xrxd}

    def run_query(q: _Query, *args, **kwargs):
        used = set(q.columns)
        for name, table in _TABLES.items():
            if used and used.issubset(set(table._columns)):
                return run_table_query(q, table.df())
        raise NotImplementedError(
            "finance.run_query 暂只支持 STK_XR_XD(分红送转)表; "
            f"本次查询列: {sorted(used)}")

    finance_ns = types_compat_finance(_TABLES, run_query)
    jqdata = sys.modules.get("jqdata")
    if jqdata is not None:
        # `from jqdata import finance` 与 jqdata.finance.run_query 均可用
        jqdata.finance = finance_ns
        for _k, _v in (("finance", finance_ns),):
            setattr(jqdata, _k, _v)


def types_compat_finance(tables: dict, run_query):
    import types
    ns = {"run_query": run_query}
    for name, table in tables.items():
        ns[name] = table
    return types.SimpleNamespace(**ns)

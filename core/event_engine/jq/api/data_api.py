# -*- coding: utf-8 -*-
"""数据获取函数(聚宽文档类别): 行情/基本面/证券信息/交易日历/行业/ extras。

重型矩阵实现(get_price/history/attribute_history/get_current_data/
get_fundamentals/get_snapshot/get_factor)是 JQRuntime 的有状态方法
(依赖行情矩阵/截面/财务值缓存), 本模块负责:
1) 把运行时方法装配进策略命名空间;
2) 实现轻量的证券信息/交易日历/行业分类/extras 类 API;
3) 注入 query DSL(valuation/income)。
指数代码(399101.XSHE 等)行情走 CNE curated index_bars;
行业分类走 CNE curated industry_members(sw + eastmoney 月度快照)。

行业键映射(与聚宽键对齐, 口径差异见 _INDUSTRY_KEY_MAP):
- sw_l1/sw_l2/sw_l3 -> CNE 申万分类(6位层级 code 按前缀拆级, 名称缺表用 code)
- jq_l1/jq_l2/zjw   -> CNE eastmoney 分类(带中文名, 粒度近似聚宽二级/证监会)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import jq_data
from core.event_engine.jq.objects import _SecurityInfo
from core.event_engine.jq.query import income, indicator, query, valuation

# 聚宽分类键 -> (CNE classification_system, 级别拆分方式)
_INDUSTRY_KEY_MAP = {
    "sw_l1": ("sw", "l1"),
    "sw_l2": ("sw", "l2"),
    "sw_l3": ("sw", "l3"),
    "jq_l1": ("eastmoney", "name"),
    "jq_l2": ("eastmoney", "name"),
    "zjw": ("eastmoney", "name"),
}


class _PanelProxy:
    """聚宽旧式 Panel 近似: panel.close / panel.open -> DataFrame(日期x代码)。

    pandas >= 0.25 移除了 Panel, 这里用属性代理复现 `panel.<field>` 用法。
    """

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = frames

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        if key in self._frames:
            return self._frames[key]
        raise AttributeError(f"Panel 无字段: {key}")

    def __getitem__(self, key):
        return self._frames[key]

    def __contains__(self, key):
        return key in self._frames

    def __iter__(self):
        return iter(self._frames)


def _pivot_panel(rt, rows: list[tuple], fields: list[str],
                 fill_paused: bool, skip_paused: bool) -> _PanelProxy:
    df = pd.DataFrame(rows, columns=["time", "code", *fields])
    frames: dict[str, pd.DataFrame] = {}
    for f in fields:
        if f in ("time", "code"):
            continue
        wide = (df.pivot_table(index="time", columns="code", values=f,
                               aggfunc="last")
                if len(df) else pd.DataFrame())
        frames[f] = wide.sort_index() if len(wide) else wide
    return _PanelProxy(frames)


def _industry_entry(code: str, snap: pd.DataFrame) -> dict:
    """单只股票的 {分类键: {industry_code, industry_name}} 视图。

    分类键优先取对应系统; 系统无数据时回退 sw(CNE 的 eastmoney 快照只
    覆盖早期月份, sw 月度快照全期覆盖), 再缺给占位 'NA/未知'
    (聚宽研究脚本常以 `len(set(ind[k]) & set(cats))==len(cats)` 过滤,
    缺键会把整行剔空)。
    """
    sw_row = snap[(snap["code"] == code) & (snap["system"] == "sw")]
    out: dict[str, dict] = {}
    for key, (system, level) in _INDUSTRY_KEY_MAP.items():
        row = snap[(snap["code"] == code) & (snap["system"] == system)]
        if len(row):
            row = row.iloc[-1]
            icode = str(row["industry_code"])
            iname = str(row["industry_name"])
        elif len(sw_row):
            # 回退: 用申万 6 位代码作该分类键的分组粒度
            row = sw_row.iloc[-1]
            icode = str(row["industry_code"])
            iname = str(row["industry_code"])
        else:
            icode, iname = "NA", "未知"
        if level == "l1" and icode != "NA":
            icode, iname = icode[:2], iname[:2]
        elif level == "l3" and icode != "NA":
            icode, iname = icode, iname        # CNE 快照到 6 位, l3 与 l2 同粒度
        out[key] = {"industry_code": icode, "industry_name": iname}
    return out


def install(ns: dict, rt) -> None:
    def get_index_stocks(index_symbol, date=None):
        """点时成分近似。
        - 399101(中小板综): 原 002/003 段, 2021 年并入主板后成分冻结,
          前缀 + 点时上市过滤即可, 无未来泄漏。
        - 其余指数: CNE index_constituents 快照仅当快照日 >= 请求日
          (避免未来泄漏), 否则回落域内全池(旧行为)。
        """
        d = (pd.Timestamp(date) if date is not None
             else rt.context.previous_date)
        ldm = rt.ctx.list_date_map

        def _listed(cs):
            return [c for c in cs if c not in ldm or ldm[c] <= d]

        c6 = str(index_symbol).split(".")[0].strip().zfill(6)
        if c6 == "399101":
            return _listed(c for c in rt.ctx.codes
                           if c.startswith(("002", "003")))
        try:
            from core.event_engine.jq.datalake.base import CNE_CURATED
            root = CNE_CURATED / "index_constituents"
            parts = sorted(root.glob("as_of_date=*"))
            if parts:
                snap = pd.read_parquet(
                    parts[-1], columns=["index_symbol", "symbol",
                                        "as_of_date"])
                hit = snap[snap["index_symbol"].str[:6] == c6]
                if len(hit):
                    as_of = pd.Timestamp(hit["as_of_date"].iloc[0])
                    if as_of <= d:
                        members = {str(s)[:6] for s in hit["symbol"]}
                        out = _listed(members & set(rt.ctx.codes))
                        if out:
                            return out
        except Exception:
            pass
        return _listed(rt.ctx.codes)          # 兜底: 域内全池(旧行为)

    def get_security_info(code):
        code = str(code).split(".")[0].strip().zfill(6)
        return _SecurityInfo(code, rt.ctx.name_map.get(code, ""),
                             rt.ctx.list_date_map.get(code))

    def get_all_securities(types="stock", date=None):
        codes = list(rt.ctx.codes)
        # start_date 用 datetime.date(object dtype, 聚宽口径): 研究代码常
        # 直接与 .date() 比较(datetime64 与 date 比较在新版 pandas 会报错)
        default_ld = pd.Timestamp("1990-01-01").date()
        df = pd.DataFrame({
            "display_name": [rt.ctx.name_map.get(c, "") for c in codes],
            "start_date": [rt.ctx.list_date_map[c].date()
                           if c in rt.ctx.list_date_map else default_ld
                           for c in codes],
            "end_date": pd.Timestamp("2200-01-01").date(),
            "type": "stock",
        }, index=codes)
        if date is not None:
            # JQ 语义: 该日期已上市的证券
            df = df[df["start_date"] <= pd.Timestamp(date).date()]
        return df

    def get_trade_days(start_date=None, end_date=None, count=None):
        return rt._get_trade_days(start_date, end_date, count)

    def get_all_trade_days():
        return rt.ctx.tables.dates

    def normalize_code(code):
        s = str(code).strip().upper()
        if "." in s:
            return s
        c = s.zfill(6)
        return c + (".XSHG" if c.startswith(("5", "6", "9", "11", "13"))
                    else ".XSHE")

    def get_industry(security, date=None):
        """行业分类 {code: {分类键: {industry_code, industry_name}}}。

        点时口径(as_of_date <= date, 缺省信号日); 数据源 CNE
        industry_members(sw 申万 + eastmoney, 后者近似 jq/zjw 粒度)。
        """
        if security is None:
            stocks = list(rt.ctx.codes)
        elif isinstance(security, (list, tuple, set)):
            stocks = [str(s).zfill(6) for s in security]
        else:
            stocks = [str(security).zfill(6)]
        d = (pd.Timestamp(date) if date is not None
             else rt.context.previous_date)
        snap = jq_data.industry_asof(d)
        return {c: _industry_entry(c, snap) for c in stocks}

    def get_extras(tag, security_list, start_date=None, end_date=None,
                   count=None, **kwargs):
        """聚宽 get_extras 子集。

        - 'is_st': tables.is_st 点时矩阵, DataFrame(index=交易日,
          columns=security_list) 值 bool;
        - 'unit_net_value'/'acc_net_value'(基金净值) 暂未接入。
        """
        if tag != "is_st":
            raise NotImplementedError(
                f"get_extras 仅支持 'is_st'(基金净值类 extras 未接入): {tag}")
        codes = ([str(s).split(".")[0].zfill(6) for s in security_list]
                 if security_list else list(rt.ctx.codes))
        dates = rt.ctx.tables.dates
        if end_date is not None:
            hi = int(dates.searchsorted(pd.Timestamp(end_date), side="right")) - 1
        else:
            hi = len(dates) - 1
        n = int(count) if count is not None else 1
        lo = max(0, hi - n + 1)
        st = rt.ctx.tables.is_st
        out = {}
        for c in codes:
            k = rt.ctx._ci.get(c)
            col = (st[lo:hi + 1, k] if k is not None
                   else np.full(hi - lo + 1, False))
            out[c] = np.where(np.isfinite(col), col > 0, False)
        return pd.DataFrame(out,
                            index=dates[lo:hi + 1]).astype(bool)

    def get_billboard_list(tag=None, start_date=None, end_date=None,
                           **kwargs):
        """龙虎榜(CNE curated/dragon_tiger, eastmoney 源, 2016 起)。

        返回长表 DataFrame(index=day): code/trade_date/reason/buy_amount/
        sell_amount/net_amount(元)。tag 兼容聚宽 'total'(全部)/行业分类键
        ——行业过滤在龙虎榜原始数据中无对应口径, 传行业键时按全量返回。
        """
        from core.event_engine.jq import datalake
        df = datalake.load("dragon_tiger")
        if not len(df):
            return pd.DataFrame(columns=["code", "trade_date", "reason",
                                         "buy_amount", "sell_amount",
                                         "net_amount"])
        df = df.copy()
        df["code"] = df["symbol"].astype(str).str.split(".").str[0]
        df["code"] = df["code"].str.zfill(6)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if start_date is not None:
            df = df[df["trade_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["trade_date"] <= pd.Timestamp(end_date)]
        out = (df.sort_values(["trade_date", "code"], kind="stable")
                 .set_index("trade_date"))
        out.index.name = "day"
        return out[["code", "reason", "buy_amount", "sell_amount",
                    "net_amount"]]

    ns.update({
        # 有状态实现: JQRuntime 方法(矩阵/截面/财务缓存)
        "get_price": rt.get_price,
        "get_snapshot": rt.get_snapshot,
        "history": rt.history,
        "attribute_history": rt.attribute_history,
        "get_current_data": rt.get_current_data,
        "get_fundamentals": rt.get_fundamentals,
        "get_factor": rt.get_factor,
        # 本模块轻量实现
        "get_index_stocks": get_index_stocks,
        "get_security_info": get_security_info,
        "get_all_securities": get_all_securities,
        "get_trade_days": get_trade_days,
        "get_all_trade_days": get_all_trade_days,
        "normalize_code": normalize_code,
        "get_industry": get_industry,
        "get_extras": get_extras,
        "get_billboard_list": get_billboard_list,
        # query DSL(get_fundamentals 的查询构造器)
        "query": query,
        "valuation": valuation,
        "income": income,
        "indicator": indicator,
    })

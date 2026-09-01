# -*- coding: utf-8 -*-
"""聚宽策略复现 —— 数据层门面(具体加载实现已插件化)。

加载实现位于 core/event_engine/jq/datalake/ (插件注册中心 + plugins/ 目录);
本模块保留:
- 策略域构建器: load_panel/build_tables/fin_ok_matrix/ew_index
  (由插件原始帧派生的组合产物, 不属于单表插件)
- 向后兼容门面: load_income/load_index_bars/load_industry_members/industry_asof
  委托到注册中心(paper.py/_runtime.py/历史脚本零改动)

约定:
- code 统一 6 位数字字符串
- 面板价格前复权(每股按自身最后交易日 adj_factor 锚定), 撮合/净值用它
- 过滤用未复权原始值(市值/涨停价/价格上限), 消除复权对绝对价格的扭曲
- 全部点时口径
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from core.event_engine.jq import datalake

# 兼容别名(历史代码直接引用 jq_data.PG 等)
ROOT = datalake.ROOT
QDATA = datalake.QDATA
PG = datalake.PG
CNE_CURATED = datalake.CNE_CURATED

DEFAULT_PREFIXES = ("00", "60")   # 沪深主板(创业板30/科创68/北交8,4默认排除)


# ============================================================
# 指数日线 / 利润表 / 行业分类 —— 委托数据插件(门面)
# ============================================================
def load_index_bars() -> pd.DataFrame:
    """CNE curated 指数日线: symbol/trade_date/open/high/low/close/volume/amount."""
    return datalake.load("index_bars")


def load_income() -> pd.DataFrame:
    """利润表(合并报表): 按 (code, end_date) 去重保留最新公告, 按 (code, ann_date) 排序."""
    return datalake.load("income")


def load_industry_members() -> pd.DataFrame:
    """CNE curated industry_members 全量长表(约 42 万行, 进程级缓存)。"""
    return datalake.load("industry_members")


def industry_asof(date) -> pd.DataFrame:
    """点时行业快照: as_of_date <= date 的每个 (code, system) 最新成员行。"""
    return datalake.asof("industry_members", date)


def data_status() -> pd.DataFrame:
    """数据插件诊断表(行数/日期覆盖/内存) —— 排查"报错是否在数据"用。"""
    return datalake.status()


# ============================================================
# 分钟线(按股票分文件的 1min 年度 parquet, 未复权收盘)
# ============================================================
_MINUTE_CACHE: dict[tuple, dict[str, pd.DataFrame]] = {}


def load_minute_close(years, minutes, codes,
                      workers: int = 8, progress=None) -> dict[str, pd.DataFrame]:
    """调度时点的分钟收盘价(未复权): {HH:MM: DataFrame(index=date, columns=code6)}。

    - 数据源 QDATA/<年>/<年>/1min/<ts_code>.parquet(每股每年一文件,
      含 trade_time/close/adj_factor), 只扫 codes 域内股票
    - 单次扫描同时抽取全部请求分钟; 会话级缓存, 撮合/盘中价查询共用
    - 缺文件/停牌分钟 -> NaN(由调用方回落日线近似)
    - progress(done, total): 每完成一只股票回调一次; 抛异常即取消
      (未完成任务排队撤销, 不写缓存)
    """
    key = (tuple(sorted(set(years))), tuple(sorted(set(minutes))))
    if key in _MINUTE_CACHE:
        return _MINUTE_CACHE[key]
    from concurrent.futures import ThreadPoolExecutor

    want = {m: int(m.replace(":", "")) for m in minutes}   # '09:30' -> 930
    want_int = np.array(sorted(want.values()), dtype=np.int64)
    per_minute: dict[str, list] = {m: [] for m in minutes}

    def _one(code: str) -> dict[str, list]:
        suf = ".SH" if code.startswith("6") else ".SZ"
        out: dict[str, list] = {}
        for y in key[0]:
            f = QDATA / str(y) / str(y) / "1min" / f"{code}{suf}.parquet"
            if not f.exists():
                continue
            try:
                # pyarrow 不吃 pandas 索引元数据(2026 文件把 trade_date/
                # trade_time 写成 index columns, pd.read_parquet 会丢列)
                import pyarrow.parquet as pq
                tb = pq.read_table(f, columns=["trade_time", "close"])
                df = tb.to_pandas(ignore_metadata=True)
            except Exception:
                continue
            t = pd.DatetimeIndex(df["trade_time"])
            hm = t.hour.to_numpy() * 100 + t.minute.to_numpy()
            hit = np.isin(hm, want_int)
            if not hit.any():
                continue
            sub = df.loc[:, "close"].to_numpy(dtype=np.float64)[hit]
            dates = t.normalize()[hit]
            hmv = hm[hit]
            for m, mi in want.items():
                sel = hmv == mi
                if sel.any():
                    out.setdefault(m, []).extend(
                        zip(dates[sel], [code] * int(sel.sum()),
                            sub[sel]))
        return out

    codes = [c for c in dict.fromkeys(codes)
             if str(c)[:1] in ("0", "3", "5", "6")]
    from concurrent.futures import as_completed
    n_total = len(codes)
    done_n = 0
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = [ex.submit(_one, c) for c in codes]
        for fut in as_completed(futs):
            part = fut.result()
            done_n += 1
            for m, lst in part.items():
                per_minute[m].extend(lst)
            if progress is not None:
                progress(done_n, n_total)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)

    out: dict[str, pd.DataFrame] = {}
    for m in minutes:
        if per_minute[m]:
            df = pd.DataFrame(per_minute[m],
                              columns=["date", "code", "close"])
            out[m] = df.pivot_table(index="date", columns="code",
                                    values="close", aggfunc="last")
        else:
            out[m] = pd.DataFrame()
    _MINUTE_CACHE[key] = out
    return out


# ============================================================
# 行情面板(策略域构建器: stock_daily 插件原始帧 -> 引擎面板)
# ============================================================
def load_panel(start: str, end: str,
               prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
               buffer_days: int = 45) -> tuple[pd.DataFrame, pd.DataFrame,
                                               pd.DataFrame]:
    """返回 (engine_panel, meta, close_raw_df)。

    engine_panel: date/code/open/close/turnover/am20/amount (前复权, 引擎可用)
    meta:         ts_code/trade_date/pre_close/up_limit/total_mv/is_st/listed_days
                  (全市场未复权, 供过滤矩阵使用, 不限 prefixes)
    close_raw_df: date/code/close_raw/open_raw(/high_raw/low_raw) 未复权价格
    """
    raw, meta = datalake.load("stock_daily", start=start, end=end,
                              buffer_days=buffer_days, prefixes=prefixes)
    raw["code"] = raw["ts_code"].str[:6]
    raw = raw.sort_values(["code", "trade_date"], kind="stable")
    raw = raw.reset_index(drop=True)

    # 前复权: 每股按其最后一个交易日 adj_factor 锚定
    last_factor = raw.groupby("code")["adj_factor"].last()
    adj = raw["adj_factor"] / raw["code"].map(last_factor)
    raw["open"] = raw["open"] * adj
    raw["close"] = raw["close"] * adj

    # am20(元): 20日均成交额, 停牌日(amount<=0)滚动值置 NaN
    amt_yuan = raw["amount"] * 1e3
    raw["am20"] = amt_yuan.groupby(raw["code"]).transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    raw.loc[amt_yuan.fillna(0) <= 0, "am20"] = np.nan

    panel = pd.DataFrame({
        "date": raw["trade_date"], "code": raw["code"],
        "open": raw["open"], "close": raw["close"],
        "turnover": raw["turnover_rate"], "am20": raw["am20"],
        "amount": raw["amount"],
    })
    panel = panel.dropna(subset=["close"])
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)]  # 防 o2o=inf
    panel = panel.reset_index(drop=True)

    crd = {
        "date": raw["trade_date"], "code": raw["code"],
        "close_raw": raw["close"] / adj,
        "open_raw": raw["open"] / adj,
    }
    if "high" in raw.columns:
        crd["high_raw"] = raw["high"]
    if "low" in raw.columns:
        crd["low_raw"] = raw["low"]
    close_raw_df = pd.DataFrame(crd)
    return panel, meta, close_raw_df


# ============================================================
# 点时财务矩阵
# ============================================================
def fin_ok_matrix(inc: pd.DataFrame, calendar: pd.DatetimeIndex,
                  codes: list[str], pred: Callable[[pd.DataFrame], np.ndarray]
                  ) -> np.ndarray:
    """点时财务布尔矩阵: 信号日取 ann_date<=该日 的最新一期报表, pred(该股全部报表)为真则 True.

    pred: 接收单只股票的报表分组(按 ann_date 升序), 返回与行数等长的 bool ndarray.
    """
    cal_d = calendar.values.astype("datetime64[D]")
    out = np.zeros((len(calendar), len(codes)), dtype=bool)
    cmap = {c: i for i, c in enumerate(codes)}
    for code, g in inc.groupby("code", sort=False):
        j = cmap.get(code)
        if j is None:
            continue
        ann = g["ann_date"].values.astype("datetime64[D]")
        flag = np.asarray(pred(g), dtype=bool)
        pos = np.searchsorted(ann, cal_d, side="right") - 1
        valid = pos >= 0
        out[valid, j] = flag[pos[valid]]
    return out


def triple_positive_pred() -> Callable[[pd.DataFrame], np.ndarray]:
    """国九条"三正": 归母净利>0 且 净利润>0 且 营收>1亿."""
    def pred(g: pd.DataFrame) -> np.ndarray:
        return ((g["n_income_attr_p"].values > 0) &
                (g["n_income"].values > 0) &
                (g["revenue"].values > 1e8))
    return pred


# ============================================================
# 过滤矩阵容器
# ============================================================
@dataclass
class MarketTables:
    """逐日过滤矩阵(与策略无关, build_tables 一次性构建)。矩阵形状 (T, K)。"""
    dates: pd.DatetimeIndex
    codes: list[str]
    close_qfq: np.ndarray          # 前复权收盘
    close_raw: np.ndarray          # 未复权收盘
    open_raw: np.ndarray           # 未复权开盘
    mv: np.ndarray                 # 总市值, 万元
    up_limit: np.ndarray           # 涨停价(未复权, 精确值)
    down_limit: np.ndarray         # 跌停价(未复权, 精确值)
    is_st: np.ndarray              # bool
    paused: np.ndarray             # bool
    listed_ok: np.ndarray          # bool 上市满 listed_days
    hl: np.ndarray                 # bool 收盘涨停(精确涨停价)
    delist_name: np.ndarray        # (K,) bool 当前名称含"退"
    high_raw: np.ndarray | None = None    # 未复权最高价
    low_raw: np.ndarray | None = None     # 未复权最低价
    pre_close: np.ndarray | None = None   # 未复权昨收
    amount: np.ndarray | None = None      # 成交额, 千元

    def index_of(self, date) -> int:
        return int(self.dates.get_loc(pd.Timestamp(date)))

    def hl_codes(self, date) -> set:
        """信号日收盘涨停的代码集合(豁免卖出用, 覆盖全部股票而非仅候选)."""
        i = self.index_of(date)
        return {self.codes[j] for j in np.nonzero(self.hl[i])[0]}

    def hl_price_of(self, date) -> dict:
        """信号日涨停股 -> 精确涨停价(开板近似用)."""
        i = self.index_of(date)
        js = np.nonzero(self.hl[i])[0]
        return {self.codes[j]: float(self.up_limit[i, j]) for j in js}


def build_tables(panel: pd.DataFrame, meta: pd.DataFrame,
                 close_raw_df: pd.DataFrame,
                 listed_days: int = 375) -> MarketTables:
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    codes = sorted(panel["code"].unique())
    T, K = len(dates), len(codes)
    dmap = {d: i for i, d in enumerate(dates)}
    cmap = {c: j for j, c in enumerate(codes)}

    m = meta[meta["ts_code"].str[:6].isin(cmap)].copy()
    if "down_limit" not in m.columns:       # 部分年份 schema 缺跌停价
        m["down_limit"] = np.nan
    m["code"] = m["ts_code"].str[:6]
    m["di"] = m["trade_date"].map(dmap)
    m["ki"] = m["code"].map(cmap)
    m = m.dropna(subset=["di", "ki"])
    m["di"] = m["di"].astype(int)
    m["ki"] = m["ki"].astype(int)

    def scatter(col: str, dtype) -> np.ndarray:
        a = np.full((T, K), np.nan, dtype=dtype)
        a[m["di"].values, m["ki"].values] = m[col].astype(dtype).values
        return a

    mv = scatter("total_mv", np.float64)
    up_limit = scatter("up_limit", np.float64)
    down_limit = scatter("down_limit", np.float64)
    is_st = scatter("is_st", np.float64) == 1
    listed = scatter("listed_days", np.float64)

    def pivot_from(df: pd.DataFrame, col: str) -> np.ndarray:
        return (df.pivot_table(index="date", columns="code", values=col,
                               aggfunc="last")
                .reindex(index=dates, columns=codes).to_numpy())

    close_qfq = pivot_from(panel, "close")
    close_raw = pivot_from(close_raw_df, "close_raw")
    open_raw = pivot_from(close_raw_df, "open_raw")
    high_raw = (pivot_from(close_raw_df, "high_raw")
                if "high_raw" in close_raw_df.columns else None)
    low_raw = (pivot_from(close_raw_df, "low_raw")
               if "low_raw" in close_raw_df.columns else None)
    amount = pivot_from(panel, "amount") if "amount" in panel.columns else None
    pre_close = (scatter("pre_close", np.float64)
                 if "pre_close" in m.columns else None)

    paused = ~np.isfinite(close_qfq)
    hl = np.zeros((T, K), dtype=bool)
    ok = np.isfinite(close_raw) & np.isfinite(up_limit)
    hl[ok] = close_raw[ok] >= up_limit[ok] - 1e-4

    basic = pd.read_parquet(PG / "stock_basic.parquet",
                            columns=["ts_code", "name", "list_date"])
    basic["code"] = basic["ts_code"].str[:6]
    basic = basic.drop_duplicates("code").set_index("code")
    name = basic["name"].reindex(codes).fillna("")
    delist_name = name.str.contains("退").values
    ld = pd.to_datetime(basic["list_date"], errors="coerce").reindex(codes)
    ld = ld.values.astype("datetime64[D]")
    age_fb = ((dates.values.astype("datetime64[D]")[:, None] - ld[None, :]) /
              np.timedelta64(1, "D"))
    listed_ok = np.where(np.isfinite(listed), listed >= listed_days,
                         age_fb >= listed_days)

    return MarketTables(
        dates=dates, codes=codes, close_qfq=close_qfq, close_raw=close_raw,
        open_raw=open_raw, mv=mv, up_limit=up_limit, down_limit=down_limit,
        is_st=is_st, paused=paused,
        listed_ok=listed_ok, hl=hl, delist_name=delist_name,
        high_raw=high_raw, low_raw=low_raw, pre_close=pre_close, amount=amount)


# ============================================================
# 域等权指数(择时/基准用)
# ============================================================
def ew_index(tables: MarketTables, clip: float = 0.2,
             ma_window: int = 10) -> tuple[pd.Series, pd.Series]:
    """返回 (level, ma): 域等权指数与其均线。

    日收益裁剪 ±clip: 剔除前复权微价股/长期停牌复牌造成的伪影收益。
    """
    cl = pd.DataFrame(tables.close_qfq, index=tables.dates, columns=tables.codes)
    ret = cl.pct_change(fill_method=None).clip(-clip, clip)
    ew = ret.mean(axis=1, skipna=True).fillna(0.0)
    level = (1.0 + ew).cumprod()
    ma = level.rolling(ma_window, min_periods=1).mean()
    return level, ma

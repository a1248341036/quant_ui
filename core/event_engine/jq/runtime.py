# -*- coding: utf-8 -*-
"""JQRuntime: exec 用户代码 + 生命周期循环 + 撮合对接。

职责边界(策略 API 的实现按聚宽文档类别拆分在 api/ 子包, 见其 __init__):
- 本模块是编排层: exec 用户代码 -> 采集生命周期钩子 -> run_day 按
  生命周期(before_trading_start -> 定时任务 -> flush -> after_trading_end)驱动
- 下单: pending_orders 在用户函数跑完后统一 flush 到引擎
  (target_value -> order_target_shares 等)
- 有状态的数据实现(get_price/history/get_fundamentals 等矩阵/截面/缓存)
  也挂在运行时上, 由 api/data_api.py 装配进策略命名空间
- 定时任务按时间串排序执行('9:05' < '10:00' < '14:00' < '14:50',
  before_open 最先 / after_close 最后), 同时刻保持注册顺序
- 命名空间预载 datetime/timedelta/date/time 与 OrderStatus(聚宽全局注入)
"""
from __future__ import annotations

import datetime as _datetime
import inspect
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT / "strategies" / "event"),
           str(_ROOT / "scripts" / "jq_repro"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _runtime import JQContext  # noqa: E402  (facade: tables/snapshot/ew_level)

from core.event_engine.jq import api as _api  # noqa: E402
from core.event_engine.jq.api.framework import sched_sort_key  # noqa: E402
from core.event_engine.jq.objects import (_CodeData, _Context, _CurrentData,  # noqa: E402
                                          _G, _Log, _Order, _OrderStatus,
                                          _SecurityInfo, _Trade)
from core.event_engine.jq.query import _Col, _Query  # noqa: E402,F401

# 指数 get_price 字段 -> CNE index_bars 列
_IX_FIELD_MAP = {"close_adj": "close", "money": "amount"}


def _load_income():
    import jq_data  # scripts/jq_repro 已在 sys.path
    return jq_data.load_income()


class JQRuntime:
    """聚宽风格运行时。"""

    def __init__(self, code: str, ctx: JQContext, capital: float,
                 window_start: str | None = None):
        self.ctx = ctx
        self.capital = float(capital)
        # 回测窗口起点: 周/月内交易日序数从它起计数(聚宽自窗口首日驱动,
        # 残缺周从窗口首日重数); None = 全日历口径(旧行为)
        self.window_start = (pd.Timestamp(window_start) if window_start
                             else None)
        self.benchmark: str | None = None
        self.g = _G()
        self.log = _Log()
        self.scheduled: list[tuple[str, Callable, object, str, int]] = []
        self.pending_orders: list[tuple[str, str, float, _Order | None]] = []
        self._cost: dict[str, float] = {}
        self._engine_ctx = None
        self._snap_cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self._fin_value_mats: dict[str, np.ndarray] | None = None
        self._bal_value_mats: dict[str, np.ndarray] | None = None
        self._minute_frames: dict[str, pd.DataFrame] = {}
        self._warned: set[str] = set()
        self._sched_sorted = False
        self._month_map: dict[pd.Timestamp, tuple[int, int]] | None = None
        self._week_map: dict[pd.Timestamp, tuple[int, int]] | None = None
        self._money_cache: np.ndarray | None = None
        self._volume_cache: np.ndarray | None = None
        self._nan_mat_cache: np.ndarray | None = None
        self.cost_cfg: dict[str, float] = {}   # initialize 内采集的费率设置
        self._hooks: dict[str, Callable] = {}  # 生命周期钩子(用户定义才生效)
        self._day_orders: dict[int, _Order] = {}
        self._order_seq = 0
        self._day_trades: list[_Trade] = []
        self._placed_report: list[dict] = []
        self._hd_injected = False
        self._proc_init_done = False
        self.context = _Context(self)
        self._ns = self._build_namespace()
        from core.event_engine.jq.api.misc import legacy_exec_scope
        with legacy_exec_scope():
            # exec 期间启用旧版研究环境兼容(pandas 旧选项/time.clock)
            exec(compile(code, "<聚宽策略>", "exec"), self._ns)  # noqa: S102
        self._init_fn = self._ns.get("initialize")
        for _name in ("process_initialize", "before_trading_start",
                      "handle_data", "after_trading_end"):
            _fn = self._ns.get(_name)
            if callable(_fn):
                self._hooks[_name] = _fn

    # ================= 命名空间 =================
    def _build_namespace(self) -> dict:
        ns = {
            "__name__": "jq_strategy",
            "g": self.g, "log": self.log,
            "context": self.context,
            # 聚宽全局预载
            "datetime": _datetime, "timedelta": _datetime.timedelta,
            "date": _datetime.date, "time": _datetime.time,
            "OrderStatus": _OrderStatus,
            # 聚宽预载的 numpy 别名(不覆盖 max/min/sum/abs/round 等内建)
            "zeros": np.zeros, "ones": np.ones, "array": np.array,
            "arange": np.arange, "linspace": np.linspace, "full": np.full,
            "where": np.where, "mean": np.mean, "nanmean": np.nanmean,
            "std": np.std, "nanstd": np.nanstd,
            "exp": np.exp, "sqrt": np.sqrt, "dot": np.dot,
            "unique": np.unique, "median": np.median, "cumsum": np.cumsum,
            "diff": np.diff, "nanmax": np.nanmax, "nanmin": np.nanmin,
            "nansum": np.nansum, "corrcoef": np.corrcoef, "cov": np.cov,
            "power": np.power, "maximum": np.maximum, "minimum": np.minimum,
        }
        _api.install_all(ns, self)
        return ns

    # ================= 下单回执 =================
    def _mk_order(self, code: str, value: float | None = None,
                  shares: float | None = None):
        """构造下单回执; 无行情代码警告并返回 None(聚宽拒单语义)。"""
        code = str(code).split(".")[0].strip().zfill(6)
        if code not in self.ctx._ci:
            if code not in self._warned:
                self._warned.add(code)
                self.log.warn(f"[order] 无行情数据, 忽略下单: {code}")
            return None
        px = self._engine_ctx.last_close(code) if self._engine_ctx else None
        if shares is None:
            shares = (float(value) / px) if px else 0.0
        amt = float(value) if value is not None else float(shares)
        self._order_seq += 1
        o = _Order(security=code, amount=float(shares), is_buy=amt > 0,
                   add_time=self.context.current_dt)
        o.order_id = self._order_seq
        self._day_orders[o.order_id] = o
        return o

    # ================= 数据 API(矩阵/截面, 由 api/data_api.py 装配) =================
    def _row_index(self, end_date=None, count=None, start_date=None) -> list[int]:
        dates = self.ctx.tables.dates
        if end_date is not None:
            end = pd.Timestamp(end_date)
            hi = int(dates.searchsorted(end, side="right")) - 1
        else:
            hi = len(dates) - 1
        if start_date is not None:
            lo = int(dates.searchsorted(pd.Timestamp(start_date), side="left"))
        elif count is not None:
            lo = max(0, hi - int(count) + 1)
        else:
            lo = hi
        return list(range(max(0, lo), min(hi, len(dates) - 1) + 1))

    def _hi_row(self, d) -> int:
        dates = self.ctx.tables.dates
        i = int(dates.searchsorted(pd.Timestamp(d), side="right")) - 1
        return max(0, min(i, len(dates) - 1))

    def _field_matrix(self, field: str) -> np.ndarray:
        t = self.ctx.tables
        mapping = {
            "close": t.close_raw, "open": t.open_raw,
            "high_limit": t.up_limit, "low_limit": t.down_limit,
            "close_adj": t.close_qfq, "high": t.high_raw,
            "low": t.low_raw, "pre_close": t.pre_close,
        }
        if field == "volume":
            return self._volume_mat()
        if field == "money":
            return self._money_mat()
        if field not in mapping:
            raise NotImplementedError(
                f"get_price/history 字段未支持: {field}。"
                "支持: close/open/high/low/pre_close/high_limit/low_limit/"
                "close_adj/volume/money/turnover")
        mat = mapping[field]
        if mat is None:                      # 部分年份 schema 缺列 -> NaN 填充
            if self._nan_mat_cache is None:
                self._nan_mat_cache = np.full_like(t.close_raw, np.nan)
            mat = self._nan_mat_cache
        return mat

    def _money_mat(self) -> np.ndarray:
        """成交额矩阵(元), 来源 tables.amount(千元)。"""
        if self._money_cache is None:
            t = self.ctx.tables
            if t.amount is not None:
                self._money_cache = t.amount * 1e3
            else:
                self._money_cache = np.full_like(t.close_raw, np.nan)
        return self._money_cache

    def _volume_mat(self) -> np.ndarray:
        """成交量矩阵(手, 近似 = 成交额/收盘价/100)。"""
        if self._volume_cache is None:
            money = self._money_mat()
            close = self.ctx.tables.close_raw
            with np.errstate(divide="ignore", invalid="ignore"):
                self._volume_cache = (money /
                                      np.where(close > 0, close, np.nan)
                                      / 100.0)
        return self._volume_cache

    def get_price(self, security, start_date=None, end_date=None,
                  frequency="daily", fields=None, count=None,
                  panel=False, fill_paused=True, skip_paused=False,
                  fq="pre", **kwargs):
        """日线取数(真实价)。'1m' 盘中现价: 分钟收盘优先, 缺失回落当日开盘。

        指数代码(399101.XSHE/000300.XSHG 等)走 CNE index_bars;
        默认 end_date: 'daily'->前一交易日, '1m'->当前交易日。
        """
        if frequency not in ("daily", "1d", "1m"):
            raise NotImplementedError(f"频率未支持: {frequency}(日线引擎)")
        multi = isinstance(security, (list, tuple, set))
        # '601988.XSHG'/'601988' -> '601988' (用户常写聚宽风格后缀码)
        codes = [str(s).split(".")[0].strip().zfill(6)
                 for s in (security if multi else [security])]
        if fields is None:
            fields = ["close"]
        if isinstance(fields, str):
            fields = [fields]
        if end_date is None and start_date is None:
            end_date = (self.context.current_dt if frequency == "1m"
                        else self.context.previous_date)
        rows = self._row_index(end_date=end_date, count=count,
                               start_date=start_date)
        t = self.ctx.tables
        ci = self.ctx._ci
        dates = t.dates
        # '1m' 现价: end_date(缺省 current_dt)时点的分钟收盘(未复权)
        minute_hi = None
        minute_dt = None
        if frequency == "1m":
            if end_date is not None and pd.Timestamp(end_date).hour:
                minute_dt = pd.Timestamp(end_date)
            elif end_date is None:
                minute_dt = self.context.current_dt
            if minute_dt is not None:
                minute_hi = self._hi_row(minute_dt)
        long_rows: list[tuple] = []
        single_dates: list = []
        single_vals: list[dict] = []
        for i in rows:
            d = dates[i]
            for c in codes:
                k = ci.get(c)
                ix = None
                if k is None:
                    ix = self.ctx.index_frame(c)
                    if ix is None:
                        continue
                vals: dict[str, float] = {}
                ok = True
                for f in fields:
                    if f == "time":
                        continue
                    if ix is not None:
                        key = _IX_FIELD_MAP.get(f, f)
                        v = (ix[key].get(d, np.nan)
                             if key in ix.columns else np.nan)
                    elif f == "turnover":
                        v = self.ctx._turnover_col(d).get(c, np.nan)
                    else:
                        # 统一字段解析(close/open/high/low/pre_close/
                        # high_limit/low_limit/close_adj/volume/money)
                        v = self._field_matrix(f)[i, k]
                        if (f == "close" and i == minute_hi
                                and k is not None):
                            # 盘中现价: 分钟价优先; 缺失回落当日开盘
                            # (不可用当日收盘 -> 未来泄漏)
                            mv = self._minute_px_raw(
                                c, minute_dt if minute_dt is not None
                                else self.context.current_dt)
                            v = mv if mv is not None else t.open_raw[i, k]
                    vals[f] = v
                    if (not fill_paused or skip_paused) and not np.isfinite(v):
                        ok = False
                if not ok:
                    continue
                if panel or multi:
                    long_rows.append((d, c, *vals.values()))
                else:
                    single_dates.append(d)
                    single_vals.append(vals)
        if panel:
            # 聚宽旧式 Panel 近似: panel.close / panel.open -> 日期x代码 矩阵
            from core.event_engine.jq.api.data_api import _pivot_panel
            return _pivot_panel(self, long_rows, fields, fill_paused,
                                skip_paused)
        if multi:
            return pd.DataFrame(long_rows, columns=["time", "code", *fields])
        out = {f: [v[f] for v in single_vals] for f in fields if f != "time"}
        return pd.DataFrame(out, index=pd.DatetimeIndex(single_dates))

    def history(self, count, unit="1d", field="close",
                security_list=None, df=True, **kwargs):
        """最近 count 个有效交易日字段(真实价), 返回 {code: list}。

        '1d' 以信号日(前一交易日)为界; '1m' 的当日现价以开盘价代理(无未来泄漏)。
        """
        if unit not in ("1d", "1m"):
            raise NotImplementedError(f"history 单位未支持: {unit}(日线引擎)")
        codes = ([str(s).zfill(6) for s in security_list]
                 if security_list else list(self.ctx.codes))
        t = self.ctx.tables
        mat = self._field_matrix(field)   # 统一字段解析(含 high/low/volume/money)
        cur_mat = t.open_raw if (unit == "1m" and field == "close") else mat
        hi = self._hi_row(self.context.current_dt if unit == "1m"
                          else self.context.previous_date)
        out = {}
        ci = self.ctx._ci
        for c in codes:
            k = ci.get(c)
            if k is None:
                out[c] = []
                continue
            vals: list[float] = []
            i = hi
            while i >= 0 and len(vals) < int(count):
                m = cur_mat if (unit == "1m" and i == hi) else mat
                v = m[i, k]
                if (unit == "1m" and i == hi and field == "close"):
                    mv = self._minute_px_raw(c, self.context.current_dt)
                    if mv is not None:
                        v = mv
                if np.isfinite(v):
                    vals.append(float(v))
                i -= 1
            vals = vals[::-1]
            if len(vals) < int(count):
                # JQ 口径: 未上市/长期停牌的 K 线以 NaN 前置补齐而非空表,
                # 避免 last_prices[s][-1] 越界(聚宽此处返回 NaN)
                vals = [np.nan] * (int(count) - len(vals)) + vals
            out[c] = vals
        return out

    def attribute_history(self, security, count, unit="1d",
                          fields=("close",)):
        # JQ 语义: 截止前一交易日的 count 根K线(不含当日), get_price 默认已如此
        return self.get_price(security, count=count, fields=list(fields),
                              panel=False)

    def get_snapshot(self, date=None) -> pd.DataFrame:
        """信号日截面(JQ 单位): market_cap(亿元) + 财务值(元) 等。"""
        d = pd.Timestamp(date) if date is not None else self.context.previous_date
        key = pd.Timestamp(d).normalize()   # current_dt 可带盘中时点
        if key not in self._snap_cache:
            snap = self.ctx.snapshot(key)
            if len(snap):
                snap = snap.copy()
                snap["market_cap"] = snap["mv"] / 1e4        # 万元 -> 亿元
                snap["circulating_market_cap"] = np.nan
                snap["__code__"] = snap.index
                i = self.ctx._di.get(key)
                j = self.ctx._ci
                for col, mat in self._income_value_mats().items():
                    snap[col] = [mat[i, j[c]] if c in j else np.nan
                                 for c in snap.index]
                for col, mat in self._balance_value_mats().items():
                    snap[col] = [mat[i, j[c]] if c in j else np.nan
                                 for c in snap.index]
                snap["income_end_date"] = np.nan
                # 估值指标(JQ 口径): pe=市值/年化归母净利; pb=市值/归母净资产
                # 亏损/净资产为负 -> NaN(与 JQ 的 >0 过滤语义一致)
                mv_yuan = snap["mv"] * 1e4
                ann = snap["ann_netprofit"]
                snap["pe_ratio"] = np.where(ann > 0, mv_yuan / ann, np.nan)
                eq = snap["total_equity"]
                snap["pb_ratio"] = np.where(eq > 0, mv_yuan / eq, np.nan)
            self._snap_cache[key] = snap
        return self._snap_cache[key]

    def _income_value_mats(self) -> dict[str, np.ndarray]:
        """点时财务值矩阵(懒构建): 归母净利/净利/营收/年化归母净利, 元, 点时。

        ann_netprofit = 最新一期归母净利 × 12/累计月数(年化, PE 分母口径);
        累计月数按报告期末月份数(3/6/9/12)。
        """
        if self._fin_value_mats is not None:
            return self._fin_value_mats
        t = self.ctx.tables
        cal_d = t.dates.values.astype("datetime64[D]")
        inc = _load_income()
        inc = inc[inc["code"].isin(set(t.codes))]
        mats = {k: np.full((len(t.dates), len(t.codes)), np.nan, dtype=np.float64)
                for k in ("np_parent_company_owners", "net_profit",
                          "operating_revenue", "ann_netprofit")}
        for code, g in inc.groupby("code", sort=False):
            k = self.ctx._ci.get(code)
            if k is None:
                continue
            ann = g["ann_date"].values.astype("datetime64[D]")
            pos = np.searchsorted(ann, cal_d, side="right") - 1
            rows = np.where(pos >= 0)[0]
            values = {
                "np_parent_company_owners": g["n_income_attr_p"].to_numpy(),
                "net_profit": g["n_income"].to_numpy(),
                "operating_revenue": g["revenue"].to_numpy(),
            }
            months = (pd.DatetimeIndex(g["end_date"]).month
                      .to_numpy().astype(float))
            factor = 12.0 / np.clip(months, 1.0, None)
            values["ann_netprofit"] = values["np_parent_company_owners"] * factor
            for key, arr in values.items():
                mats[key][rows, k] = arr[pos[rows]]
        self._fin_value_mats = mats
        return mats

    def _balance_value_mats(self) -> dict[str, np.ndarray]:
        """点时归母净资产矩阵(懒构建): total_hldr_eqy_exc_min_int, 元, 点时。"""
        if self._bal_value_mats is not None:
            return self._bal_value_mats
        t = self.ctx.tables
        cal_d = t.dates.values.astype("datetime64[D]")
        bal = pd.read_parquet(
            _ROOT / "data" / "pg_parquet" / "balancesheet.parquet",
            columns=["ts_code", "f_ann_date", "ann_date", "end_date",
                     "report_type", "total_hldr_eqy_exc_min_int"])
        bal = bal[pd.to_numeric(bal["report_type"], errors="coerce") == 1]
        bal["code"] = bal["ts_code"].str[:6]
        ann = pd.to_datetime(bal["f_ann_date"], errors="coerce").fillna(
            pd.to_datetime(bal["ann_date"], errors="coerce"))
        bal["ann_date"] = ann
        bal = bal.dropna(subset=["ann_date"])
        bal = bal.sort_values(["code", "end_date", "ann_date"], kind="stable")
        bal = bal.drop_duplicates(["code", "end_date"], keep="last")
        bal = bal.sort_values(["code", "ann_date"], kind="stable")
        mat = np.full((len(t.dates), len(t.codes)), np.nan, dtype=np.float64)
        for code, g in bal.groupby("code", sort=False):
            k = self.ctx._ci.get(code)
            if k is None:
                continue
            ann = g["ann_date"].values.astype("datetime64[D]")
            pos = np.searchsorted(ann, cal_d, side="right") - 1
            rows = np.where(pos >= 0)[0]
            mat[rows, k] = pd.to_numeric(
                g["total_hldr_eqy_exc_min_int"], errors="coerce"
            ).to_numpy()[pos[rows]]
        self._bal_value_mats = {"total_equity": mat}
        return self._bal_value_mats

    def get_fundamentals(self, q: _Query, date=None):
        snap = self.get_snapshot(date)
        if not len(snap):
            return pd.DataFrame()
        mask = np.ones(len(snap), dtype=bool)
        for expr in q.exprs:
            mask = mask & np.asarray(expr.fn(snap), dtype=bool)
        out = snap[mask]
        col_keys = []
        for col in q.cols:
            key = col.key if isinstance(col, _Col) else str(col)
            col_keys.append(key)
        out = out[[k for k in col_keys if k in out.columns]]
        if q.order is not None:
            direction, col = q.order
            out = out.sort_values(col.key, ascending=(direction == "asc"),
                                  kind="stable")
        if q._limit is not None:
            out = out.head(q._limit)
        return out.reset_index().rename(columns={"index": "code"})

    def get_current_data(self) -> _CurrentData:
        # 当日截面: 涨跌停/停牌/ST 为 T 日口径(JQ 语义); last_price≈开盘价(无泄漏)
        return _CurrentData(self.get_snapshot(self.context.current_dt),
                            self.ctx.name_map)

    # ================= 因子桥 =================
    def get_factor(self, expr, date=None):
        """AlphaAgent DSL 因子表达式 -> 截面 Series(index=code)。

        语法与因子实验室一致: $close/$open/$amount... 引用列, TS_*/CS_* 算子,
        多行表达式(中间变量赋值)最后一行为输出。date 缺省= 信号日。
        面板口径: 前复权 OHLC + amount(千元) + turnover_rate(%) + volume(手)。
        """
        from core.event_engine.jq import factor_bridge
        ser = factor_bridge.factor_series(self.ctx, expr)
        d = (pd.Timestamp(date) if date is not None
             else self.context.previous_date)
        if isinstance(ser.index, pd.MultiIndex):
            lvl = pd.DatetimeIndex(ser.index.get_level_values(0))
            if d not in lvl:
                raise ValueError(
                    f"get_factor: {d.date()} 不在因子面板日期范围 "
                    f"({lvl[0].date()} ~ {lvl[-1].date()})")
            out = ser[lvl == d]
            idx = out.index.get_level_values(1)
        else:
            out = ser
            idx = out.index
        return pd.Series(
            out.to_numpy(dtype=float),
            index=[str(c).split(".")[0].zfill(6) for c in idx],
            name="factor").dropna().sort_index()

    # ================= 日循环 =================
    def bind(self, engine_ctx):
        self._engine_ctx = engine_ctx
        for c in list(self._cost):
            if c not in engine_ctx.positions:
                del self._cost[c]

    @staticmethod
    def _call_hook(fn, *args):
        """按用户函数签名长度传参(兼容 (context) 与 (context, data) 两种形态)."""
        try:
            n = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            n = len(args)
        if n <= 0:
            return fn()
        return fn(*args[:n])

    def _snapshot_fills(self):
        """上一执行日的引擎成交 -> 当日可查的 Trade 列表 + 回执日志。"""
        ctx = self._engine_ctx
        fills = list(getattr(ctx, "fills", None) or [])
        out = []
        d = self.context.previous_date
        for f in fills:
            shares = float(f.get("shares") or 0)
            side = str(f.get("side") or "")
            out.append(_Trade(
                security=f.get("code"),
                price=f.get("price") or 0.0,
                amount=shares if side == "buy" else -shares,
                time=d,
                fee=f.get("fee") or 0.0))
        self._day_trades = out
        # ---- 回执日志: 昨日委托的成交/未成交 + 日终快照(预热段不记) ----
        if (self.window_start is not None
                and d < self.window_start):
            self._placed_report = []
            return
        placed = getattr(self, "_placed_report", None) or []
        if placed or fills:
            filled_codes = set()
            for f in fills:
                code = str(f.get("code"))
                filled_codes.add(code)
                name = self.ctx.name_map.get(code, "")
                side = ("买入" if str(f.get("side")) == "buy" else "卖出")
                self.log.info(
                    f"[成交 {d.date()}] {side} {code} {name} "
                    f"{float(f.get('shares') or 0):.0f}股 "
                    f"@ {float(f.get('price') or 0):.3f} "
                    f"金额 {float(f.get('amount') or 0):,.0f}元 "
                    f"费 {float(f.get('fee') or 0):.2f}")
            for p in placed:
                if str(p["code"]) not in filled_codes:
                    self.log.warn(
                        f"[未成交 {d.date()}] {p['desc']} "
                        f"(涨停/跌停/停牌/资金不足)")
        self._placed_report = []
        if ctx is not None:
            pv = float(getattr(ctx, "portfolio_value", 0.0) or 0.0)
            cash = float(getattr(ctx, "cash", 0.0) or 0.0)
            pos = getattr(ctx, "positions", None) or {}
            if pos:
                items = " ".join(
                    f"{c}({sh:.0f})" for c, sh in sorted(pos.items()))
                self.log.info(
                    f"[日终 {d.date()}] 净值 {pv / self.capital:.4f} "
                    f"资产 {pv:,.0f} 现金 {cash:,.0f} 持仓 {items}")
            else:
                self.log.info(
                    f"[日终 {d.date()}] 净值 {pv / self.capital:.4f} "
                    f"资产 {pv:,.0f} 现金 {cash:,.0f} 持仓 (空仓)")

    def _get_trade_days(self, start_date=None, end_date=None, count=None):
        dates = self.ctx.tables.dates
        end = (pd.Timestamp(end_date) if end_date is not None
               else self.context.previous_date)
        hi = int(dates.searchsorted(end, side="right")) - 1
        if start_date is not None:
            lo = int(dates.searchsorted(pd.Timestamp(start_date), side="left"))
        elif count is not None:
            lo = max(0, hi - int(count) + 1)
        else:
            lo = 0
        return dates[max(0, lo):hi + 1]

    # ================= 分钟数据(盘中价撮合/查询) =================
    def prefetch_minutes(self, minutes) -> None:
        """预取调度时点的分钟收盘(未复权), 供盘中价撮合/查询。

        数据缺失时保持空 dict, 所有路径自动回落日线近似(旧行为)。
        """
        minutes = [str(m) for m in minutes if m]
        if not minutes:
            return
        try:
            import jq_data
            dates = self.ctx.tables.dates
            years = sorted({int(d.year) for d in dates})
            codes = [c for c in self.ctx.codes if str(c)[:1] in "036"]
            frames = jq_data.load_minute_close(years, minutes, codes)
            self._minute_frames = {m: f for m, f in frames.items()
                                   if f is not None and len(f)}
            n = sum(len(f.index) for f in self._minute_frames.values())
            self.log.info(f"[runtime] 分钟线就绪: {len(self._minute_frames)}"
                          f"个时点 x {n}股日")
        except Exception as exc:                  # noqa: BLE001 - 回落日线
            self._minute_frames = {}
            self.log.warn(f"[runtime] 分钟数据不可用, 盘中按日线近似: {exc}")

    def _minute_px_raw(self, code: str, dt) -> float | None:
        """(code, dt) 的分钟收盘价(未复权); 无数据返回 None。"""
        if not self._minute_frames:
            return None
        ts = pd.Timestamp(dt)
        frame = self._minute_frames.get(ts.strftime("%H:%M"))
        if frame is None or frame.empty:
            return None
        d = ts.normalize()
        if d not in frame.index:
            return None
        col = frame.get(str(code))
        if col is None:
            return None
        v = col.at[d] if d in col.index else np.nan
        return float(v) if np.isfinite(v) else None

    def _minute_px_panel(self, code: str, dt) -> float | None:
        """分钟收盘价换算到引擎面板(前复权)价空间; 无数据返回 None。"""
        v = self._minute_px_raw(code, dt)
        if v is None:
            return None
        k = self.ctx._ci.get(str(code))
        if k is None:
            return None
        i = self._hi_row(pd.Timestamp(dt).normalize())
        t = self.ctx.tables
        raw, adj = t.close_raw[i, k], t.close_qfq[i, k]
        if not np.isfinite(raw) or raw <= 0 or not np.isfinite(adj):
            return None
        return float(v) * float(adj) / float(raw)

    @staticmethod
    def _slot_dt(day, tk: str | None) -> pd.Timestamp:
        """交易日 + 调度时点字符串(如 '10:00') -> 带时点的 current_dt。"""
        day = pd.Timestamp(day).normalize()
        if tk:
            try:
                h, m = str(tk).split(":")
                return day + pd.Timedelta(hours=int(h), minutes=int(m))
            except (ValueError, AttributeError):
                return day
        return day

    def _ensure_schedule(self):
        if not self._hd_injected:
            # handle_data(context, data) 注入为 9:30 的 daily 任务(时间排序自动就位);
            # 包一层以补上 data 参数(当日截面, 开盘价代理现价)
            hd = self._hooks.get("handle_data")
            if hd is not None:
                def _hd_wrapper(ctx):
                    self._call_hook(hd, ctx, self.get_current_data())
                self.scheduled.append(("daily", _hd_wrapper, None, "9:30", -1))
            self._hd_injected = True
        if not self._sched_sorted:
            self.scheduled.sort(key=sched_sort_key)
            self._sched_sorted = True
        if self._month_map is None or self._week_map is None:
            dates = self.ctx.tables.dates
            # 窗口起点之前(预热段)的日期不参与周/月内计数: 聚宽自回测
            # 首日驱动, 残缺周/月从窗口首日重数(实证: 2025-01-02 起的
            # 回测, 聚宽把 01-03 当首周第 2 个交易日)
            if self.window_start is not None:
                keep = np.asarray(dates) >= self.window_start
            else:
                keep = np.ones(len(dates), dtype=bool)
            d_keep = dates[keep]
            if self._month_map is None:
                per = d_keep.to_period("M")
                s = pd.Series(1, index=d_keep)
                nth = s.groupby(per).cumsum()
                tot = s.groupby(per).transform("size")
                self._month_map = {d: (int(n), int(t))
                                   for d, n, t in zip(d_keep, nth, tot)}
            if self._week_map is None:
                iso = d_keep.isocalendar()
                wk = (iso["year"].to_numpy().astype(np.int64) * 100
                      + iso["week"].to_numpy().astype(np.int64))
                s = pd.Series(1, index=d_keep)
                wn = s.groupby(wk).cumsum()
                wt = s.groupby(wk).transform("size")
                self._week_map = {d: (int(n), int(t))
                                  for d, n, t in zip(d_keep, wn, wt)}

    def run_day(self, bar):
        self._ensure_schedule()
        self.context.current_dt = self._slot_dt(bar.exec_date, "9:00")
        self.context.previous_date = bar.date
        self._day_orders = {}
        self._snapshot_fills()
        if not self._proc_init_done:
            self._proc_init_done = True
            fn = self._hooks.get("process_initialize")
            if fn is not None:
                self._call_hook(fn, self.context)
        bts = self._hooks.get("before_trading_start")
        if bts is not None:
            self._call_hook(bts, self.context, self.get_current_data())
        monthly_first = bar.exec_date.month != bar.date.month
        for kind, func, arg, tk, _seq in self.scheduled:
            # 聚宽语义: 函数内 current_dt = 调度时点(盘中价查询/下单回执依赖)
            self.context.current_dt = self._slot_dt(bar.exec_date, tk)
            if kind == "daily":
                func(self.context)
            elif kind == "weekly":
                n, tot = self._week_map.get(bar.exec_date, (0, 0))
                hit = (n == arg) if arg > 0 else ((tot - n + 1) == -arg)
                if hit:
                    func(self.context)
            elif kind == "monthly":
                nth, tot = self._month_map.get(bar.exec_date, (0, 0))
                if arg is None:
                    hit = monthly_first
                elif arg > 0:
                    hit = nth == arg
                else:
                    hit = (tot - nth + 1) == -arg
                if hit:
                    func(self.context)
        self._flush_orders(bar)
        ate = self._hooks.get("after_trading_end")
        if ate is not None:
            self.context.current_dt = self._slot_dt(bar.exec_date, "15:00")
            self._call_hook(ate, self.context)

    _KIND_DESC = {"tv": "目标市值", "ot": "目标股数", "ov": "买入市值",
                  "os": "增减股数", "op": "目标占比"}

    def _flush_orders(self, bar):
        ctx = self._engine_ctx
        pending = self.pending_orders
        self.pending_orders = []
        self._placed_report = []
        warmup = (self.window_start is not None
                  and bar.exec_date < self.window_start)
        for kind, code, arg, order in pending:
            # 盘中价撮合: 下单时点(order.add_time)的分钟价(面板前复权空间),
            # 无分钟数据回落 signal 日收盘定价(引擎再按执行日开盘成交)
            pxm = self._minute_px_panel(code, order.add_time) if order is not None else None
            px = pxm or ctx.last_close(code)
            name = self.ctx.name_map.get(str(code), "")
            slot = (order.add_time.strftime("%H:%M")
                    if order is not None and order.add_time is not None
                    else "--:--")
            if not px:
                if code not in self._warned:
                    self._warned.add(code)
                    self.log.warn(f"[order] 无有效价格, 跳过下单: {code}")
                if order is not None:
                    order.status = _OrderStatus.rejected
                continue
            # 方向推断 + 拒单预判(引擎侧涨跌停/停牌检查)
            cur = float(ctx.position(code) or 0.0)
            if kind in ("tv", "ot"):
                target = (arg / px) if kind == "tv" else arg
                direction = "买入" if target > cur else "卖出"
            elif kind == "op":
                direction = "买入" if arg > 0 else "卖出"
            else:
                direction = "买入" if arg > 0 else "卖出"
            block = ""
            if direction == "买入" and not ctx.can_buy(code):
                block = "  [预计废单:涨停/停牌]"
            elif direction == "卖出" and not ctx.can_sell(code):
                block = "  [预计废单:跌停/停牌]"
            src = "分钟价" if pxm is not None else "前收盘"
            arg_desc = (f"{arg:,.0f}元" if kind in ("tv", "ov")
                        else (f"{arg:,.0f}股" if kind in ("ot", "os")
                              else f"{arg:.2%}"))
            if not warmup:
                self.log.info(
                    f"[委托 {bar.exec_date.date()} {slot}] {direction} {code} "
                    f"{name} {self._KIND_DESC.get(kind, kind)} {arg_desc} "
                    f"参考价 {px:.3f}({src}){block}")
                self._placed_report.append({"code": str(code),
                                            "desc": f"{direction} {code} "
                                                    f"{name} {arg_desc}"})
            if kind == "tv":          # 目标市值 -> 目标股数
                ctx.order_target_shares(code, arg / px, fill_price=pxm)
            elif kind == "ot":        # 目标股数
                ctx.order_target_shares(code, arg, fill_price=pxm)
            elif kind == "ov":        # 市值增量
                ctx.order_shares(code, arg / px, fill_price=pxm)
            elif kind == "os":        # 股数增量
                ctx.order_shares(code, arg, fill_price=pxm)
            elif kind == "op":        # 目标占比
                ctx.order_target_pct(code, arg, fill_price=pxm)
            if order is not None:
                # 提交即视为受理; 实际成交明细见 get_trades()(引擎逐笔)
                order.status = _OrderStatus.held
                order.filled = order.amount
                order.price = px
                order.avg_cost = px * 1.0001
            self._record_cost(code, bar, pxm or bar.open.get(code))

    def _record_cost(self, code: str, bar, px):
        # 成本价近似 = 实际成交基准价(分钟价), 缺省执行日开盘(与引擎撮合一致)
        if px:
            self._cost[code] = px * 1.0001

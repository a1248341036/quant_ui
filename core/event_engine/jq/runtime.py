# -*- coding: utf-8 -*-
"""JQRuntime: exec 用户代码 + 数据/下单 API + 每日循环适配。

职责边界:
- 本模块只做"翻译": 用户 API 调用 -> 数据层(_runtime.JQContext)查询 +
  事件引擎下单接口; 不含任何撮合/账户逻辑
- 下单: pending_orders 在用户函数跑完后统一 flush 到引擎
  (target_value -> order_target_shares 等)
- 定时任务按时间串排序执行('9:05' < '10:00' < '14:00' < '14:50',
  before_open 最先 / after_close 最后), 同时刻保持注册顺序;
  reference_security 等聚宽专有参数接受但忽略
- 命名空间预载 datetime/timedelta/date/time 与 OrderStatus(聚宽全局注入)
"""
from __future__ import annotations

import datetime as _datetime
import sys
import types
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

from core.event_engine.jq.objects import (_CodeData, _Context, _CurrentData,  # noqa: E402
                                          _FixedSlippage, _G, _Log, _Order,
                                          _OrderCost, _OrderStatus, _Position,
                                          _SecurityInfo)
from core.event_engine.jq.query import _Col, _Query, query  # noqa: E402,F401

# 指数 get_price 字段 -> CNE index_bars 列
_IX_FIELD_MAP = {"close_adj": "close", "money": "amount"}


def _load_income():
    import jq_data  # scripts/jq_repro 已在 sys.path
    return jq_data.load_income()


def _sched_sort_key(entry):
    """(kind, func, arg, time_key, seq) -> 排序键。"""
    t, seq = entry[3], entry[4]
    if t == "before_open":
        return (0, 0, seq)
    if t == "after_close":
        return (3, 0, seq)
    if t in ("open", "every_bar"):
        return (1, 9 * 60 + 30, seq)
    try:
        hh, mm = str(t).split(":")
        return (1, int(hh) * 60 + int(mm), seq)
    except Exception:
        return (1, 9 * 60 + 30, seq)


class JQRuntime:
    """聚宽风格运行时。"""

    def __init__(self, code: str, ctx: JQContext, capital: float):
        self.ctx = ctx
        self.capital = float(capital)
        self.g = _G()
        self.log = _Log()
        self.scheduled: list[tuple[str, Callable, object, str, int]] = []
        self.pending_orders: list[tuple[str, str, float]] = []  # (kind, code, arg)
        self._cost: dict[str, float] = {}
        self._engine_ctx = None
        self._snap_cache: dict[pd.Timestamp, pd.DataFrame] = {}
        self._fin_value_mats: dict[str, np.ndarray] | None = None
        self._warned: set[str] = set()
        self._sched_sorted = False
        self._month_map: dict[pd.Timestamp, tuple[int, int]] | None = None
        self._week_map: dict[pd.Timestamp, tuple[int, int]] | None = None
        self._money_cache: np.ndarray | None = None
        self._volume_cache: np.ndarray | None = None
        self._nan_mat_cache: np.ndarray | None = None
        self.cost_cfg: dict[str, float] = {}   # initialize 内采集的费率设置
        self.context = _Context(self)
        self._ns = self._build_namespace()
        exec(compile(code, "<聚宽策略>", "exec"), self._ns)  # noqa: S102
        self._init_fn = self._ns.get("initialize")

    # ================= 命名空间 =================
    def _build_namespace(self) -> dict:
        rt = self
        jqdata = types.ModuleType("jqdata")
        # 审计意见等 finance 表无本地数据: 返回空表(相关过滤逻辑恒通过)
        jqdata.finance = types.SimpleNamespace(
            STK_AUDIT_OPINION=types.SimpleNamespace(
                code=_Col("code"), pub_date=_Col("pub_date"),
                report_type=_Col("report_type")),
            run_query=lambda *a, **k: pd.DataFrame(
                columns=["code", "pub_date", "report_type"]))
        jqfactor = types.ModuleType("jqfactor")
        sys.modules.setdefault("jqdata", jqdata)
        sys.modules.setdefault("jqfactor", jqfactor)

        def _schedule(kind, func, arg, time_key):
            rt.scheduled.append((kind, func, arg, str(time_key),
                                 len(rt.scheduled)))

        def run_daily(func, time="9:30", **kwargs):
            _schedule("daily", func, None, time)

        def run_weekly(func, weekday=1, time="10:00", **kwargs):
            # JQ 语义: weekday=本周第 N 个交易日(1=周内首个交易日, 负数=倒数)
            _schedule("weekly", func, int(weekday), time)

        def run_monthly(func, monthday=None, time="9:30", **kwargs):
            # monthday: None=每月首个交易日, 正N=第N个交易日, 负N=倒数第N个
            _schedule("monthly", func, monthday, time)

        def _mk_and_enqueue(kind, code, value=None, shares=None):
            code = str(code).zfill(6)
            o = rt._mk_order(code, value=value, shares=shares)
            if o is not None:
                rt.pending_orders.append((kind, code,
                                          float(value if value is not None
                                                else shares)))
            return o

        def order_target_value(code, value):
            return _mk_and_enqueue("tv", code, value=float(value))

        def order_value(code, value):
            return _mk_and_enqueue("ov", code, value=float(value))

        def order_shares(code, shares):
            return _mk_and_enqueue("os", code, shares=float(shares))

        def order(code, amount):
            return order_shares(code, amount)

        def order_target(code, amount):
            return _mk_and_enqueue("ot", code, shares=float(amount))

        def order_target_percent(code, pct):
            # 回执按市值估算, 排队存 pct(引擎按占比撮合, flush 端不再换算)
            code = str(code).zfill(6)
            pv = (rt._engine_ctx.portfolio_value
                  if rt._engine_ctx is not None else None) or rt.capital
            o = rt._mk_order(code, value=float(pct) * float(pv))
            if o is not None:
                rt.pending_orders.append(("op", code, float(pct)))
            return o

        def set_order_cost(cost, type="stock", **kwargs):
            k = getattr(cost, "kwargs", None) or {}
            try:
                oc = (float(k.get("open_commission") or 0)
                      + float(k.get("open_tax") or 0))
                sc = (float(k.get("close_commission") or 0)
                      + float(k.get("close_tax") or 0))
                if oc > 0:
                    rt.cost_cfg["buy_cost"] = oc
                if sc > 0:
                    rt.cost_cfg["sell_cost"] = sc
                if k.get("min_commission"):
                    rt.log.info("[runtime] min_commission(最低佣金)暂不支持, "
                                "引擎按比例计费")
            except (TypeError, ValueError):
                pass

        def set_slippage(s, **kwargs):
            v = getattr(s, "value", s if isinstance(s, (int, float)) else None)
            if v:
                rt.cost_cfg["slippage_bps"] = float(v) * 1e4

        def get_index_stocks(index_symbol, date=None):
            # 全量池(域内全部股票), 点时口径: 剔除信号日尚未上市的代码;
            # 已退市代码无可靠退市标记, 保留(由 paused 过滤兜底)
            d = (pd.Timestamp(date) if date is not None
                 else rt.context.previous_date)
            ldm = rt.ctx.list_date_map
            return [c for c in rt.ctx.codes
                    if c not in ldm or ldm[c] <= d]

        def get_security_info(code):
            code = str(code).zfill(6)
            return _SecurityInfo(code, rt.ctx.name_map.get(code, ""),
                                 rt.ctx.list_date_map.get(code))

        def get_all_securities(types="stock", date=None):
            codes = list(rt.ctx.codes)
            return pd.DataFrame({
                "display_name": [rt.ctx.name_map.get(c, "") for c in codes],
                "start_date": [rt.ctx.list_date_map.get(c)
                               or pd.Timestamp("1990-01-01") for c in codes],
                "end_date": pd.Timestamp("2200-01-01"),
                "type": "stock",
            }, index=codes)

        ns = {
            "__name__": "jq_strategy",
            "g": rt.g, "log": rt.log,
            "context": rt.context,
            "run_daily": run_daily, "run_weekly": run_weekly,
            "run_monthly": run_monthly,
            "order_target_value": order_target_value,
            "order_value": order_value, "order_shares": order_shares,
            "order": order, "order_target": order_target,
            "order_target_percent": order_target_percent,
            "get_price": rt.get_price,
            "get_snapshot": rt.get_snapshot,
            "history": rt.history,
            "attribute_history": rt.attribute_history,
            "get_current_data": rt.get_current_data,
            "get_fundamentals": rt.get_fundamentals,
            "query": query, "valuation": _query_valuation,
            "income": _query_income,
            "get_index_stocks": get_index_stocks,
            "get_security_info": get_security_info,
            "get_all_securities": get_all_securities,
            "get_factor": rt.get_factor,
            # 聚宽全局预载
            "datetime": _datetime, "timedelta": _datetime.timedelta,
            "date": _datetime.date, "time": _datetime.time,
            "OrderStatus": _OrderStatus,
            "set_option": lambda *a, **k: None,
            "set_benchmark": lambda *a, **k: None,
            "set_universe": lambda *a, **k: None,
            "set_slippage": set_slippage,
            "set_order_cost": set_order_cost,
            "FixedSlippage": _FixedSlippage, "OrderCost": _OrderCost,
        }
        return ns

    # ================= 下单回执 =================
    def _mk_order(self, code: str, value: float | None = None,
                  shares: float | None = None):
        """构造下单回执; 无行情代码警告并返回 None(聚宽拒单语义)。"""
        if code not in self.ctx._ci:
            if code not in self._warned:
                self._warned.add(code)
                self.log.warn(f"[order] 无行情数据, 忽略下单: {code}")
            return None
        px = self._engine_ctx.last_close(code) if self._engine_ctx else None
        if shares is None:
            shares = (float(value) / px) if px else 0.0
        amt = float(value) if value is not None else float(shares)
        return _Order(security=code, amount=float(shares), is_buy=amt > 0)

    # ================= 数据 API =================
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
        """日线取数(真实价)。'1m' 按日线近似(当日请求以收盘价代理盘中价)。

        指数代码(399101.XSHE/000300.XSHG 等)走 CNE index_bars;
        默认 end_date: 'daily'->前一交易日, '1m'->当前交易日。
        """
        if frequency not in ("daily", "1d", "1m"):
            raise NotImplementedError(f"频率未支持: {frequency}(日线引擎)")
        multi = isinstance(security, (list, tuple, set))
        codes = [str(s).zfill(6) for s in
                 (security if multi else [security])]
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
                    elif f in ("close", "open", "high_limit", "low_limit",
                               "close_adj"):
                        v = self._field_matrix(f)[i, k]
                    elif f == "turnover":
                        v = self.ctx._turnover_col(d).get(c, np.nan)
                    else:
                        raise NotImplementedError(f"字段未支持: {f}")
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
        if panel or multi:
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
        key = pd.Timestamp(d)
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
                snap["income_end_date"] = np.nan
            self._snap_cache[key] = snap
        return self._snap_cache[key]

    def _income_value_mats(self) -> dict[str, np.ndarray]:
        """点时财务值矩阵(懒构建): 归母净利/净利/营收, 元, ann_date 点时。"""
        if self._fin_value_mats is not None:
            return self._fin_value_mats
        t = self.ctx.tables
        cal_d = t.dates.values.astype("datetime64[D]")
        inc = _load_income()
        inc = inc[inc["code"].isin(set(t.codes))]
        mats = {k: np.full((len(t.dates), len(t.codes)), np.nan, dtype=np.float64)
                for k in ("np_parent_company_owners", "net_profit",
                          "operating_revenue")}
        for code, g in inc.groupby("code", sort=False):
            k = self.ctx._ci.get(code)
            if k is None:
                continue
            ann = g["ann_date"].values.astype("datetime64[D]")
            pos = np.searchsorted(ann, cal_d, side="right") - 1
            rows = np.where(pos >= 0)[0]
            for key, col in (("np_parent_company_owners", "n_income_attr_p"),
                             ("net_profit", "n_income"),
                             ("operating_revenue", "revenue")):
                mats[key][rows, k] = g[col].to_numpy()[pos[rows]]
        self._fin_value_mats = mats
        return mats

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

    def _ensure_schedule(self):
        if not self._sched_sorted:
            self.scheduled.sort(key=_sched_sort_key)
            self._sched_sorted = True
        if self._month_map is None:
            dates = self.ctx.tables.dates
            per = dates.to_period("M")
            nth = pd.Series(1, index=dates).groupby(per).cumsum()
            tot = pd.Series(1, index=dates).groupby(per).transform("size")
            self._month_map = {d: (int(n), int(t))
                               for d, n, t in zip(dates, nth, tot)}
        if self._week_map is None:
            dates = self.ctx.tables.dates
            iso = dates.isocalendar()
            wk = (iso["year"].to_numpy().astype(np.int64) * 100
                  + iso["week"].to_numpy().astype(np.int64))
            s = pd.Series(1, index=dates)
            wn = s.groupby(wk).cumsum()
            wt = s.groupby(wk).transform("size")
            self._week_map = {d: (int(n), int(t))
                              for d, n, t in zip(dates, wn, wt)}

    def run_day(self, bar):
        self._ensure_schedule()
        self.context.current_dt = bar.exec_date
        self.context.previous_date = bar.date
        monthly_first = bar.exec_date.month != bar.date.month
        for kind, func, arg, _tk, _seq in self.scheduled:
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

    def _flush_orders(self, bar):
        ctx = self._engine_ctx
        pending = self.pending_orders
        self.pending_orders = []
        for kind, code, arg in pending:
            px = ctx.last_close(code)
            if not px:
                if code not in self._warned:
                    self._warned.add(code)
                    self.log.warn(f"[order] 无有效价格, 跳过下单: {code}")
                continue
            if kind == "tv":          # 目标市值 -> 目标股数
                ctx.order_target_shares(code, arg / px)
            elif kind == "ot":        # 目标股数
                ctx.order_target_shares(code, arg)
            elif kind == "ov":        # 市值增量
                ctx.order_shares(code, arg / px)
            elif kind == "os":        # 股数增量
                ctx.order_shares(code, arg)
            elif kind == "op":        # 目标占比
                ctx.order_target_pct(code, arg)
            self._record_cost(code, bar, px)

    def _record_cost(self, code: str, bar, px: float):
        # 成本价近似 = 执行日开盘(与引擎撮合价一致, 费用另计)
        op = bar.open.get(code)
        if op:
            self._cost[code] = op * 1.0001


# query.py 的 valuation/income 是模块级类; 命名空间直接引用
from core.event_engine.jq.query import income as _query_income  # noqa: E402
from core.event_engine.jq.query import valuation as _query_valuation  # noqa: E402

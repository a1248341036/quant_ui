# -*- coding: utf-8 -*-
"""聚宽风格策略运行时 —— 策略文件只写 select(), 其余全部在这层。

策略文件形态(strategies/event/xxx.py):

    from _runtime import build_context, make_event_strategy

    PARAMS = dict(stock_num=7, max_single_weight=0.12, max_exposure=0.70,
                  stoploss=0.07, take_profit=2.0, market_crash=0.05,
                  highest=60.0, rebalance_weekday=1, pass_months=(4,),
                  ma_timing=dict(window=10, base=7, span=3, scale=0.025, lo=4))

    def select(snap):
        \"\"\"返回按优先级排序的候选代码列表(已过策略自己的硬过滤)\"\"\"
        df = snap
        ok = df.mv.between(3e4, 1e7) & df.fin_三正 & ~df.st & df.listed_ok
        return df[ok].sort_values("mv").head(50).index.tolist()

    EVENT_STRATEGIES = {"我的策略": make_event_strategy(PARAMS, select)}

snapshot(=信号日截面 DataFrame, index=code) 可用列:
    close(前复权收盘) close_raw(未复权收盘) mv(万元) turnover(%)
    st paused listed_ok listed_days hl(收盘涨停) limit_price(涨停价)
    fin_<名字>(注入的财务谓词, 见 fin_preds)

骨架自动处理(与聚宽语义对齐): 周期调仓、持仓豁免价格/涨停过滤、
昨日涨停豁免卖出、个股止损/止盈、大盘惨跌清仓、涨停开板近似卖出、
空仓月清仓、暴露/单票权重上限、MA 连续仓位(可选)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "scripts" / "jq_repro"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jq_data  # noqa: E402
from template import ma_num_fn  # noqa: E402
from core.event_engine import EventStrategy, run_event_backtest  # noqa: E402

DEFAULT_PARAMS = dict(
    stock_num=7, max_single_weight=0.12, max_exposure=0.70,
    stoploss=0.07, take_profit=2.0, market_crash=0.05,
    highest=None,               # None=骨架不做价格过滤(由 select 自己管)
    rebalance_weekday=1, pass_months=(4,),
    open_seal_ratio=1.095,      # 昨收涨停后今开涨幅低于该值视为开板
    ma_timing=None,             # dict(window,base,span,scale,lo) 或 None
    buy_cost=0.0001, sell_cost=0.0011,
    top_keep=None,              # select 返回过长的截断(None=不截断)
)

# 聚宽空仓月常买货基ETF -> 注入合成行情(年化~2%), 使下单可撮合
MONEY_ETFS = ("511880.XSHG", "511990.XSHG")
METF_NAME = {"511880.XSHG": "银华日利", "511990.XSHG": "华宝添益"}


def _inject_money_etfs(panel: pd.DataFrame, close_raw_df: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """注入货基ETF合成行情(年化~2%直线): 空仓月买 511880.XSHG 等需有价可撮合。

    meta(涨跌停/市值)不注入 -> 天然不会被选股/涨停过滤误入股票池。
    """
    have = set(panel["code"].unique())
    todo = [c for c in MONEY_ETFS if c not in have]
    if not todo:
        return panel, close_raw_df
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    path = 100.0 * (1.0 + 0.02 / 250.0) ** np.arange(len(dates))
    pans = [pd.DataFrame({"date": dates, "code": c, "open": path,
                          "close": path, "turnover": 0.01, "am20": 1e9})
            for c in todo]
    crs = [pd.DataFrame({"date": dates, "code": c, "close_raw": path,
                         "open_raw": path}) for c in todo]
    print(f"[runtime] 注入货基ETF合成行情: {todo}", flush=True)
    return (pd.concat([panel, *pans], ignore_index=True),
            pd.concat([close_raw_df, *crs], ignore_index=True))


class JQContext:
    """数据上下文: 构建一次, snapshot(date) 给出信号日截面。"""

    def __init__(self, end: str | None = None, lookback_days: int = 800,
                 prefixes: tuple[str, ...] = ("00", "60"),
                 fin_preds: dict[str, Callable] | None = None,
                 listed_days: int = 375):
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
        start = (end_ts - pd.Timedelta(days=lookback_days)).date().isoformat()
        t0 = time.time()
        panel, meta, close_raw_df = jq_data.load_panel(
            start, end_ts.date().isoformat(), prefixes=prefixes)
        panel, close_raw_df = _inject_money_etfs(panel, close_raw_df)
        self.tables = jq_data.build_tables(panel, meta, close_raw_df,
                                           listed_days=listed_days)
        self.panel = panel
        self.codes = sorted(panel["code"].unique())
        self._fin_preds = dict(fin_preds or {})
        self._fin_mats: dict[str, np.ndarray] = {}
        if self._fin_preds:
            inc = jq_data.load_income()
            for name, pred in self._fin_preds.items():
                self._fin_mats[f"fin_{name}"] = jq_data.fin_ok_matrix(
                    inc, self.tables.dates, self.tables.codes, pred)
        if "三正" not in self._fin_mats:
            inc = jq_data.load_income()
            self._fin_mats["fin_三正"] = jq_data.fin_ok_matrix(
                inc, self.tables.dates, self.tables.codes,
                jq_data.triple_positive_pred())
        self._di = {d: i for i, d in enumerate(self.tables.dates)}
        self._ci = {c: i for i, c in enumerate(self.tables.codes)}
        self._close_series: dict[str, pd.Series] = {}
        self._ew_level: pd.Series | None = None
        self._index_df: pd.DataFrame | None = None
        self._index_cache: dict[str, pd.DataFrame] = {}
        basic = pd.read_parquet(jq_data.PG / "stock_basic.parquet",
                                columns=["ts_code", "name", "list_date"])
        basic["code"] = basic["ts_code"].str[:6]
        basic = basic.drop_duplicates("code").set_index("code")
        self.name_map: dict[str, str] = basic["name"].astype(str).to_dict()
        self.list_date_map: dict[str, pd.Timestamp] = {
            c: pd.Timestamp(d) for c, d in basic["list_date"].items()
            if pd.notna(d)}
        for c in MONEY_ETFS:
            self.name_map.setdefault(c, METF_NAME.get(c, ""))
            self.list_date_map.setdefault(c, pd.Timestamp("2012-01-01"))
        print(f"[runtime] 上下文就绪({max(self.tables.dates).date()}) "
              f"{time.time()-t0:.0f}s, {len(self.codes)} codes")

    def ew_level(self, clip: float = 0.2) -> pd.Series:
        """域等权指数(归一化 level, 裁剪伪影收益), 缓存。"""
        if self._ew_level is None:
            self._ew_level, _ = jq_data.ew_index(self.tables, clip=clip)
        return self._ew_level

    def _index_daily(self) -> pd.DataFrame:
        if self._index_df is None:
            self._index_df = jq_data.load_index_bars()
        return self._index_df

    def index_frame(self, code) -> pd.DataFrame | None:
        """指数日线(code 不在股票面板时), date 索引 + OHLCV 列。

        - 指数形态代码(000xxx.XSHG / 399xxx.XSHE / 932000.CSI 等):
          真实数据走 CNE index_bars; CNE 缺失时回退域等权指数(中位归一)。
        - 非指数形态(域内未覆盖的股票代码)返回 None, 由调用方按无数据处理。
        """
        code = str(code)
        sym, _, suf = code.partition(".")
        suf_n = {"XSHE": "SZ", "XSHG": "SH"}.get(suf.upper(), suf.upper())
        if suf_n == "SH":
            if not sym.startswith(("000", "880", "950")):
                return None
        elif suf_n == "SZ":
            if not sym.startswith("399"):
                return None
        elif suf_n != "CSI":
            if sym.startswith("399"):
                suf_n = "SZ"
            else:
                return None          # 无后缀且非 399: 按股票处理
        key = sym.zfill(6) + "." + suf_n
        if key in self._index_cache:
            return self._index_cache[key]
        frame = None
        idx = self._index_daily()
        if len(idx):
            sub = idx[idx["symbol"] == key]
            if len(sub):
                frame = (sub.set_index("trade_date")
                            [["open", "high", "low", "close",
                              "volume", "amount"]]
                            .sort_index())
                frame = frame[~frame.index.duplicated(keep="last")]
        if frame is None:
            lvl = self.ew_level()
            s = lvl * (11000.0 / float(lvl.median()))
            print(f"[index] CNE 无 {code} 指数日线, "
                  f"用域等权指数(中位归一≈11000点)替代", flush=True)
            frame = pd.DataFrame({"open": s, "high": s, "low": s, "close": s})
        self._index_cache[key] = frame
        return frame

    # ---- 截面 ----
    def snapshot(self, date) -> pd.DataFrame:
        """信号日截面 DataFrame(index=code, 只含当日有交易的股票)."""
        i = self._di.get(pd.Timestamp(date))
        if i is None:
            return pd.DataFrame()
        t = self.tables
        keep = np.isfinite(t.close_qfq[i])
        idx = [c for c, k in zip(t.codes, keep) if k]
        df = pd.DataFrame({
            "close": t.close_qfq[i][keep],
            "close_raw": t.close_raw[i][keep],
            "open_raw": t.open_raw[i][keep],
            "mv": t.mv[i][keep],
            "st": t.is_st[i][keep],
            "paused": t.paused[i][keep],
            "listed_ok": t.listed_ok[i][keep],
            "listed_days": self._listed_days_row(i)[keep],
            "hl": t.hl[i][keep],
            "limit_price": t.up_limit[i][keep],
            "low_limit": t.down_limit[i][keep],
        }, index=idx)
        df["name"] = [self.name_map.get(c, "") for c in idx]
        for name, mat in self._fin_mats.items():
            df[name] = mat[i][keep]
        tr = self._turnover_col(pd.Timestamp(date))
        df["turnover"] = tr.reindex(df.index)
        return df

    def _listed_days_row(self, i: int) -> np.ndarray:
        if not hasattr(self, "_listed_mat"):
            basic = pd.read_parquet(
                jq_data.PG / "stock_basic.parquet",
                columns=["ts_code", "list_date"])
            basic["code"] = basic["ts_code"].str[:6]
            basic = basic.drop_duplicates("code").set_index("code")
            ld = pd.to_datetime(basic["list_date"], errors="coerce")
            vals = ld.reindex(self.tables.codes).values.astype("datetime64[D]")
            age = ((self.tables.dates.values.astype("datetime64[D]")[:, None]
                    - vals[None, :]) / np.timedelta64(1, "D"))
            self._listed_mat = age
        return self._listed_mat[i]

    def _turnover_col(self, d: pd.Timestamp) -> pd.Series:
        if d not in self._close_series:
            day = self.panel[self.panel["date"] == d]
            self._close_series[d] = day.set_index("code")["turnover"]
        return self._close_series[d]

    # ---- 择时 ----
    def num_fn(self, ma_timing: dict) -> Callable:
        level, ma = jq_data.ew_index(self.tables,
                                     ma_window=ma_timing.get("window", 10))
        return ma_num_fn(level, ma, base=ma_timing.get("base", 7),
                         span=ma_timing.get("span", 3),
                         scale_rel=ma_timing.get("scale", 0.025),
                         lo=ma_timing.get("lo", 4))


def make_event_strategy(params: dict, select: Callable,
                        ctx: JQContext | None = None):
    """骨架 + 用户的 select() -> EventStrategy 类。"""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if ctx is None:
        ctx = build_context()
    num_fn = ctx.num_fn(p["ma_timing"]) if p.get("ma_timing") else None

    class JQFacade(EventStrategy):
        def init(self, c) -> None:
            self.cost: dict[str, float] = {}

        def _sell(self, c, code):
            c.order_target_pct(code, 0.0)
            self.cost.pop(code, None)

        def on_bar(self, c, bar) -> None:
            sig = bar.date
            for code in list(self.cost):
                if code not in c.positions:
                    del self.cost[code]

            if bar.exec_date.month in tuple(p["pass_months"]):
                for code in list(c.positions):
                    self._sell(c, code)
                return

            snap = ctx.snapshot(sig)
            ranking = select(snap) if len(snap) else []
            if p.get("top_keep"):
                ranking = ranking[:p["top_keep"]]
            hl_set = ctx.tables.hl_codes(sig)

            # 个股止损/止盈 + 涨停开板近似
            for code in list(c.positions):
                ac = self.cost.get(code)
                px = c.last_close(code)
                if ac and px and p["take_profit"] and px >= ac * p["take_profit"]:
                    self._sell(c, code)
                    continue
                if ac and px and p["stoploss"] and px < ac * (1 - p["stoploss"]):
                    self._sell(c, code)
                    continue
                if code in hl_set and p["open_seal_ratio"]:
                    op = bar.open.get(code)
                    if op is not None and px and op < px * p["open_seal_ratio"]:
                        self._sell(c, code)

            # 大盘惨跌
            if p["market_crash"] and bar.close and bar.open:
                drops = [1.0 - bar.close[s] / bar.open[s]
                         for s in bar.close if s in bar.open and bar.open[s] > 0]
                if drops and float(np.mean(drops)) >= p["market_crash"]:
                    for code in list(c.positions):
                        self._sell(c, code)
                    return

            if bar.exec_date.weekday() != p["rebalance_weekday"]:
                return

            num = num_fn(sig) if num_fn else p["stock_num"]
            target: list[str] = []
            for code in ranking:
                if len(target) >= num:
                    break
                if code in c.positions:
                    target.append(code)          # 持仓豁免价格/涨停过滤
                    continue
                if code in hl_set:               # 收盘涨停不入池
                    continue
                if p["highest"] is not None:
                    v = snap["close_raw"].get(code, np.nan)
                    if not (np.isfinite(v) and v <= p["highest"]):
                        continue
                target.append(code)
            for code in list(c.positions):
                if code not in target and code not in hl_set:
                    self._sell(c, code)
            buy_list = [x for x in target if x not in c.positions]
            if not buy_list:
                return
            pv = c.portfolio_value or 100000.0
            exposure = sum(c.position_value(s) for s in c.positions) / pv
            sold = sum(c.position_value(s) for s in c.positions
                       if s not in target) / pv
            per = min(p["max_single_weight"],
                      max(p["max_exposure"] - (exposure - sold), 0.0)
                      / len(buy_list))
            if per <= 0:
                return
            for code in buy_list:
                c.order_target_pct(code, per)
                op = bar.open.get(code)
                if op:
                    self.cost[code] = op * (1 + p["buy_cost"])

    return JQFacade


def build_context(end: str | None = None, **kw) -> JQContext:
    return JQContext(end=end, **kw)


def run_backtest(ctx: JQContext, strategy_cls, start: str, end: str,
                 capital: float = 100_000.0, warmup_days: int = 400):
    """同一策略类的快速回测(模拟盘面板口径)."""
    return run_event_backtest(
        panel=ctx.panel, codes=ctx.codes, strategy_class=strategy_cls,
        start=start, end=end, capital=capital, warmup_days=warmup_days,
        buy_cost=0.0001, sell_cost=0.0011, limit_flags=True,
    )

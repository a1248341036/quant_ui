from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .limit import build_limit_flags
from .metrics import compute_excess_metrics, compute_metrics, drawdown_series
from .assets import AssetExecutionProfile, STOCK_PROFILE
from . import trading_config


# ============================================================
# 事件驱动回测引擎
#
# 用法：
#   class MyStrategy(EventStrategy):
#       def on_bar(self, ctx, bar):
#           if bar.date.weekday() == 0:
#               for code in ctx.codes:
#                   if ctx.is_tradable(code):
#                       ctx.order_target_pct(code, 1.0 / 3)
#
#   res = run_event_backtest(panel, codes, MyStrategy, start, end, capital)
#
# 撮合语义（与 run_backtest 对齐）：
#   - 信号日 bar.date 收盘后生成订单，执行日 bar.exec_date 开盘价成交（T+1）
#   - 一手 100 股，买入/卖出均向下取整到整手
#   - 先卖后买；执行日停牌（开盘价无效）不可交易
#   - 执行日涨停不可买入、跌停不可卖出（保守近似，缺 ST 5% 标记）
#   - 可选滑点/冲击成本：成交价 = 开盘价 × (1 ± slippage)
#   - 可选流动性约束：单笔买入金额 <= 20日均成交额 × max_participation
#   - 可选限价单：limit_price 未触及则不成交（买需价<=限价，卖需价>=限价）
#   - 买/卖手续费分别按 buy_cost/sell_cost 从成交额中扣除
#   - 每日收盘按最新有效收盘价估值
# ============================================================


@dataclass
class Order:
    code: str
    target_pct: float | None = None
    target_shares: float | None = None
    delta_shares: float | None = None
    limit_price: float | None = None


@dataclass
class Bar:
    """策略回调看到的截面：date 是信号日（t-1），exec_date 是执行日（t）。"""

    date: pd.Timestamp
    exec_date: pd.Timestamp
    close: dict[str, float]
    open: dict[str, float]
    turnover: dict[str, float]
    am20: dict[str, float]
    tradable: set[str]


class EventStrategy:
    """事件驱动策略基类。用户子类实现 init / on_bar。"""

    def init(self, ctx: "Context") -> None:
        """回测开始前调用一次，可做预计算/状态初始化。"""

    def on_bar(self, ctx: "Context", bar: Bar) -> None:
        """每个交易日调用。通过 ctx.order_* 下单。"""
        raise NotImplementedError


class Context:
    def __init__(
        self,
        codes: list[str],
        close_mat: np.ndarray,
        open_mat: np.ndarray,
        valid_close: np.ndarray,
        valid_open: np.ndarray,
        turnover_mat: np.ndarray,
        am20_mat: np.ndarray,
        limit_up: np.ndarray | None,
        limit_down: np.ndarray | None,
        dates: pd.DatetimeIndex,
        capital: float,
        buy_cost: float,
        sell_cost: float,
        lot_size: int,
        amount_q: float,
        slippage: float = 0.0,
        max_participation: float = 0.0,
        field_mats: dict[str, np.ndarray] | None = None,
    ):
        self.codes = list(codes)
        self._idx = {c: i for i, c in enumerate(self.codes)}
        self._close = close_mat
        self._open = open_mat
        self._valid_close = valid_close
        self._valid_open = valid_open
        self._turnover = turnover_mat
        self._am20 = am20_mat
        self._field_mats: dict[str, np.ndarray] = dict(field_mats) if field_mats else {}
        for name, mat in (("close", close_mat), ("open", open_mat),
                          ("turnover", turnover_mat), ("am20", am20_mat)):
            self._field_mats.setdefault(name, mat)
        self.available_fields: list[str] = sorted(self._field_mats)
        self._limit_up = limit_up
        self._limit_down = limit_down
        self._dates = dates
        self.capital = float(capital)
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.lot_size = int(lot_size)
        self.amount_q = float(amount_q)
        self.slippage = float(slippage)
        self.max_participation = float(max_participation)

        self.t = 0
        self.sig = 0
        self.cash = float(capital)
        self.positions: dict[str, float] = {}
        self._last_close: dict[str, float] = {}
        self.orders: list[Order] = []
        self.fills: list[dict] = []  # 逐笔成交明细（供模拟盘落库/归因）

    # ---------- 只读信息 ----------

    @property
    def date(self) -> pd.Timestamp:
        return self._dates[self.sig]

    @property
    def portfolio_value(self) -> float:
        """按最近有效收盘价估算的组合总市值（策略信号期使用）。"""
        return self.cash + sum(sh * self._last_close.get(c, 0.0)
                               for c, sh in self.positions.items())

    def position(self, code: str) -> float:
        return float(self.positions.get(code, 0.0))

    def position_value(self, code: str) -> float:
        return self.position(code) * self._last_close.get(code, 0.0)

    def last_close(self, code: str) -> float | None:
        return self._last_close.get(code)

    def close_series(self, code: str, window: int) -> list[float]:
        """截至信号日最近 window 个有效收盘价（旧→新）。"""
        k = self._idx.get(code)
        if k is None or window <= 0:
            return []
        out: list[float] = []
        for i in range(self.sig, -1, -1):
            if self._valid_close[i, k]:
                out.append(float(self._close[i, k]))
                if len(out) >= window:
                    break
        out.reverse()
        return out

    def history(self, code: str, fields: str | list[str],
                window: int = 20) -> pd.DataFrame:
        """截至信号日（含）最近 window 个交易日的多字段历史。

        返回 DataFrame：行 = 交易日（旧→新，末行即信号日），列 = 字段。
        与 close_series 的差异：按日历交易日截窗，停牌/缺失保留 NaN，
        不做有效值跳过。fields 可为单个字符串或列表；
        未知字段抛 ValueError（可用字段见 ctx.available_fields）。
        """
        k = self._idx.get(code)
        if k is None:
            return pd.DataFrame()
        if isinstance(fields, str):
            fields = [fields]
        missing = [f for f in fields if f not in self._field_mats]
        if missing:
            raise ValueError(
                f"history 字段不可用: {missing}；可用字段: {self.available_fields}")
        if window <= 0 or self.sig < 0:
            return pd.DataFrame()
        start = max(0, self.sig - window + 1)
        data = {f: self._field_mats[f][start:self.sig + 1, k] for f in fields}
        return pd.DataFrame(data, index=self._dates[start:self.sig + 1])

    def is_tradable(self, code: str) -> bool:
        k = self._idx.get(code)
        if k is None:
            return False
        t, sig = self.t, self.sig
        if not (self._valid_close[sig, k] and self._valid_open[t, k]):
            return False
        v = self._turnover[sig, k]
        if not np.isfinite(v) or v <= 0:
            return False
        return True

    def can_buy(self, code: str) -> bool:
        """执行日能否买入：开盘有效且非涨停日（涨停买不进，保守处理）。"""
        k = self._idx.get(code)
        if k is None:
            return False
        if self._limit_up is not None and self._limit_up[self.t, k]:
            return False
        return self._valid_open[self.t, k]

    def can_sell(self, code: str) -> bool:
        """执行日能否卖出：开盘有效且非跌停日（跌停卖不出，保守处理）。"""
        k = self._idx.get(code)
        if k is None:
            return False
        if self._limit_down is not None and self._limit_down[self.t, k]:
            return False
        return self._valid_open[self.t, k]

    # ---------- 下单 ----------

    def order_target_pct(self, code: str, pct: float,
                         limit_price: float | None = None) -> None:
        self.orders.append(Order(code, target_pct=float(pct),
                                 limit_price=limit_price))

    def order_target_shares(self, code: str, shares: float,
                            limit_price: float | None = None) -> None:
        self.orders.append(Order(code, target_shares=float(shares),
                                 limit_price=limit_price))

    def order_shares(self, code: str, delta: float,
                     limit_price: float | None = None) -> None:
        self.orders.append(Order(code, delta_shares=float(delta),
                                 limit_price=limit_price))

    # ---------- 组合优化 ----------

    def _returns_window(self, codes: list[str], window: int) -> pd.DataFrame:
        """返回信号日前最近 window 个交易日收益（列=股票，行=日）。"""
        keep = [c for c in codes if c in self._idx]
        if not keep:
            return pd.DataFrame()
        cols_idx = [self._idx[c] for c in keep]
        start = max(0, self.sig - window + 1)
        mat = self._close[start:self.sig + 1][:, cols_idx]
        rets = np.diff(mat, axis=0) / mat[:-1]
        rets = rets[np.all(np.isfinite(rets), axis=1)]
        return pd.DataFrame(rets, columns=keep)

    def optimize_risk_parity(self, codes: list[str], window: int = 60,
                             max_weight: float = 0.4) -> dict[str, float]:
        """用最近 window 日收益做风险平价，返回 {code: 权重}。"""
        from .portfolio import weights_from_returns
        r = self._returns_window(codes, window)
        if r.shape[0] < 2 or r.shape[1] == 0:
            return {}
        return weights_from_returns(r, "risk_parity", max_weight=max_weight)

    def optimize_mean_variance(self, codes: list[str], window: int = 60,
                               gamma: float = 1.0,
                               max_weight: float = 0.4) -> dict[str, float]:
        """均值方差优化：最大化 w'μ - γ·w'Σw，返回 {code: 权重}。"""
        from .portfolio import weights_from_returns
        r = self._returns_window(codes, window)
        if r.shape[0] < 2 or r.shape[1] == 0:
            return {}
        return weights_from_returns(r, "mean_variance", gamma=gamma,
                                    max_weight=max_weight)

    def optimize_max_diversification(self, codes: list[str], window: int = 60,
                                     max_weight: float = 0.4) -> dict[str, float]:
        """最大化分散化权重：DR = Σ(w_i·σ_i) / sqrt(w'Σw)。"""
        from .portfolio import weights_from_returns
        r = self._returns_window(codes, window)
        if r.shape[0] < 2 or r.shape[1] == 0:
            return {}
        return weights_from_returns(r, "max_diversification",
                                    max_weight=max_weight)

    # ---------- 撮合 ----------

    def _open_price(self, code: str) -> float | None:
        k = self._idx.get(code)
        if k is None:
            return None
        t = self.t
        if not self._valid_open[t, k]:
            return None
        return float(self._open[t, k])

    def execute(self) -> dict:
        """先卖后买，按执行日开盘价成交。返回成交统计。"""
        self.fills = []
        if not self.orders:
            return {"buy": 0.0, "sell": 0.0, "trades": []}
        total_prev = self.portfolio_value or 1.0
        lot = self.lot_size
        exec_list: list[tuple[str, float, str]] = []  # (code, amount, "buy"/"sell")

        # ---- 第一步：计算每只股票的期望股数 ----
        target: dict[str, tuple[float | None, float | None]] = {}
        for o in self.orders:
            if o.code not in self._idx:
                continue
            cur = self.position(o.code)
            prev_px = self._last_close.get(o.code, 0.0)
            if o.target_pct is not None:
                pct = max(min(o.target_pct, 1.0), -1.0)
                val = total_prev * pct
                want = int(val / prev_px) if prev_px > 0 else 0
                target[o.code] = (float(want), o.limit_price)
            elif o.target_shares is not None:
                target[o.code] = (float(o.target_shares), o.limit_price)
            elif o.delta_shares is not None:
                target[o.code] = (cur + o.delta_shares, o.limit_price)

        # ---- 第二步：卖出（先） ----
        for code, (want, limit_price) in target.items():
            cur = self.position(code)
            if want is None:
                continue
            delta = cur - want
            if delta <= 0:
                continue
            px = self._open_price(code)
            if px is None:
                continue  # 停牌，不能卖，保留持仓
            if not self.can_sell(code):
                continue  # 跌停卖不出（含开空），保留持仓
            px = px * (1.0 - self.slippage)
            if limit_price is not None and px < limit_price:
                continue  # 限价卖出：成交价未达到限价，不成交
            shares = int(delta // lot) * lot
            if shares <= 0:
                continue
            amt = shares * px
            self.cash += amt * (1.0 - self.sell_cost)
            new_pos = cur - shares
            if abs(new_pos) <= 1e-9:
                del self.positions[code]
            else:
                self.positions[code] = new_pos
            exec_list.append((code, amt, "sell"))
            self.fills.append({
                "code": code, "side": "sell", "shares": shares,
                "price": px, "fee": amt * self.sell_cost,
                "amount": amt,
            })

        # ---- 第三步：买入（后） ----
        for code, (want, limit_price) in target.items():
            cur = self.position(code)
            if want is None:
                continue
            delta = want - cur
            if delta <= 0:
                continue
            px = self._open_price(code)
            if px is None:
                continue
            if not self.can_buy(code):
                continue  # 涨停买不进（含回补空头），跳过
            px = px * (1.0 + self.slippage)
            if limit_price is not None and px > limit_price:
                continue  # 限价买入：成交价高于限价，不成交
            need = delta
            gross = px * (1.0 + self.buy_cost)
            max_lots = int(self.cash // gross) // lot
            need_lots = int(need // lot)
            lots = min(need_lots, max_lots)
            if self.max_participation > 0:
                k = self._idx[code]
                liq = self._am20[self.sig, k]
                if not np.isfinite(liq) or liq <= 0:
                    continue
                liq_lots = int((liq * self.max_participation) / gross) // lot
                lots = min(lots, liq_lots)
            if lots <= 0:
                continue
            shares = lots * lot
            cost = shares * gross
            fee = shares * px * self.buy_cost
            self.cash -= cost
            new_pos = cur + shares
            if abs(new_pos) <= 1e-9:
                del self.positions[code]
            else:
                self.positions[code] = new_pos
            exec_list.append((code, shares * px, "buy"))
            self.fills.append({
                "code": code, "side": "buy", "shares": shares,
                "price": px, "fee": fee,
                "amount": shares * px,
            })

        buy_amt = sum(a for _, a, s in exec_list if s == "buy")
        sell_amt = sum(a for _, a, s in exec_list if s == "sell")
        return {"buy": buy_amt, "sell": sell_amt, "trades": exec_list}

    def mark_to_market(self, day: int, close_row: np.ndarray) -> None:
        for k, code in enumerate(self.codes):
            if self._valid_close[day, k]:
                self._last_close[code] = float(close_row[k])

    def reset(self) -> None:
        self.cash = float(self.capital)
        self.positions.clear()
        self._last_close.clear()
        self.orders.clear()


def run_event_backtest(
    panel: pd.DataFrame,
    codes: list[str],
    strategy_class: type[EventStrategy],
    start: str,
    end: str,
    capital: float,
    buy_cost: float = trading_config.BUY_COST,
    sell_cost: float = trading_config.SELL_COST,
    lot_size: int = 100,
    warmup_days: int | None = None,
    amount_q: float = 0.3,
    limit_flags: bool = True,
    slippage_bps: float = trading_config.SLIPPAGE_BPS,
    max_participation: float = trading_config.MAX_PARTICIPATION,
    short_rate: float = 0.0,
    execution_profile: AssetExecutionProfile | None = None,
) -> dict:
    """事件驱动回测。

    slippage_bps: 固定滑点（基点）。买入价=开盘×(1+bps/1e4)，卖出反向。
    max_participation: 流动性约束，单笔买入金额 <= 20日均成交额 × 该比例。
        0 表示不限。
    short_rate: 空头年化融券费率（占空头市值比例/年），每日按 short_rate/252 扣。
    """
    profile = execution_profile or STOCK_PROFILE
    if profile.asset_type == "etf":
        if buy_cost == STOCK_PROFILE.buy_cost:
            buy_cost = profile.buy_cost
        if sell_cost == STOCK_PROFILE.sell_cost:
            sell_cost = profile.sell_cost
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    calc_start = (start_ts - pd.Timedelta(days=warmup_days)
                  if warmup_days and warmup_days > 0 else start_ts)
    sub = panel[panel["code"].isin(codes)].copy()
    sub = sub[(sub["date"] >= calc_start) & (sub["date"] <= end_ts)]
    if sub.empty:
        raise ValueError("所选区间/股票池内没有数据")

    cal = pd.DatetimeIndex(sorted(sub["date"].unique()))

    for req_col in ("close", "open"):
        if req_col not in sub.columns:
            raise ValueError(f"事件回测需要列: {req_col}")

    # 排序/去重/建索引一次共享（这里逐数值列都 pivot，列数多，groupby-last 开销更显著）；
    # (date, code) 唯一时 unstack 与 pivot_table(aggfunc="last") 逐位等价，
    # 重复键保留最后一条，与 aggfunc="last" 语义一致。
    dup = sub.duplicated(["date", "code"], keep="last")
    if dup.any():
        sub = sub[~dup]
    sub = sub.sort_values(["date", "code"], kind="stable").set_index(["date", "code"])

    def pivot(col: str) -> pd.DataFrame:
        # 全 NaN 列丢弃（pivot_table 语义，codes_used 依赖这一点），
        # 全 NaN 行由 reindex(cal) 恢复交易日历
        return sub[col].unstack().dropna(axis=1, how="all").reindex(cal).sort_index()

    close_df = pivot("close")
    cols_ref = close_df.columns.tolist()
    codes_used = cols_ref
    # 除日期/代码外，全部数值列转成矩阵，供 ctx.history 多字段查询
    field_mats: dict[str, np.ndarray] = {}
    for col in sub.columns:
        if col in ("date", "code"):
            continue
        try:
            arr = pivot(col).to_numpy(dtype=np.float64)
        except Exception:
            continue
        if np.isfinite(arr).any():
            field_mats[col] = arr
    missing = [f for f in ("close", "open", "turnover", "am20") if f not in field_mats]
    if missing:
        raise ValueError(f"事件回测缺少字段: {missing}")

    close_mat = field_mats["close"]
    open_mat = field_mats["open"]
    turn_mat = field_mats["turnover"]
    am20_mat = field_mats["am20"]
    df_close = pd.DataFrame(close_mat, index=cal, columns=cols_ref)
    df_open = pd.DataFrame(open_mat, index=cal, columns=cols_ref)
    valid_close = ~np.isnan(close_mat)
    valid_open = ~np.isnan(open_mat)

    dates = close_df.index
    T, K = close_mat.shape
    if T < 5:
        raise ValueError("数据区间太短")

    open_ff = df_open.ffill()
    o2o = np.nan_to_num(open_ff.pct_change().values, nan=0.0)
    start_idx = int(np.argmax(dates >= start_ts)) if (dates >= start_ts).any() else 0

    limit_up = limit_down = None
    if limit_flags:
        limit_up, limit_down, _, _ = build_limit_flags(df_close, df_open)

    ctx = Context(
        codes=codes_used, close_mat=close_mat, open_mat=open_mat,
        valid_close=valid_close, valid_open=valid_open,
        turnover_mat=turn_mat, am20_mat=am20_mat,
        limit_up=limit_up, limit_down=limit_down,
        dates=dates, capital=capital, buy_cost=buy_cost,
        sell_cost=sell_cost, lot_size=lot_size, amount_q=amount_q,
        slippage=slippage_bps / 1e4, max_participation=max_participation,
        field_mats=field_mats,
    )
    strategy = strategy_class()
    strategy.init(ctx)

    nav = np.ones(T)
    bench = np.ones(T)
    holdings_history: list[dict[str, float]] = []
    weight_history: list[dict[str, float]] = []
    cash_history: list[float] = []
    trades: list[dict] = []
    trades_detail: list[dict] = []
    last_chosen: list[str] = []

    for t in range(1, T):
        prev = t - 1
        ctx.t = t
        ctx.sig = prev
        ctx.orders = []

        if t == start_idx and warmup_days:
            # 预热段只用于积累策略状态，窗口起点持仓清零、净值归 1
            ctx.reset()
            nav[t] = 1.0
            bench[t] = 1.0
            holdings_history.append({})
            weight_history.append({})
            cash_history.append(ctx.cash)
            continue

        # 基准：股票池等权（开盘到开盘收益）
        elig = valid_close[prev] & valid_open[prev] & valid_open[t]
        bench_ret = float(np.nanmean(np.where(elig, o2o[t], np.nan))) if elig.any() else 0.0

        # 构造信号日截面
        bar = Bar(
            date=dates[prev],
            exec_date=dates[t],
            close={c: float(close_mat[prev, k]) for k, c in enumerate(codes_used)
                   if valid_close[prev, k]},
            open={c: float(open_mat[t, k]) for k, c in enumerate(codes_used)
                  if valid_open[t, k]},
            turnover={c: float(turn_mat[prev, k]) for k, c in enumerate(codes_used)
                      if np.isfinite(turn_mat[prev, k])},
            am20={c: float(am20_mat[prev, k]) for k, c in enumerate(codes_used)
                  if np.isfinite(am20_mat[prev, k])},
            tradable={c for c in codes_used if ctx.is_tradable(c)},
        )

        # 估值基准更新为信号日收盘后，再让策略下单
        ctx.mark_to_market(prev, close_mat[prev])
        try:
            strategy.on_bar(ctx, bar)
        except Exception as exc:
            raise RuntimeError(f"策略在 {dates[prev].date()} 出错: {exc}") from exc

        stats = ctx.execute()
        hold_count = len(ctx.positions)
        if ctx.fills:
            for f in ctx.fills:
                trades_detail.append({
                    "date": dates[t], "signal_date": dates[prev],
                    **f,
                })
        if stats["trades"]:
            buys = [c for c, _, s in stats["trades"] if s == "buy"]
            sells = [c for c, _, s in stats["trades"] if s == "sell"]
            total_val = ctx.portfolio_value or 1.0
            turn = (stats["buy"] + stats["sell"]) / 2.0 / total_val
            trades.append({
                "date": dates[t],
                "signal_date": dates[prev],
                "num_hold": hold_count,
                "turnover": float(turn),
                "bought": ",".join(buys[:12]),
                "sold": ",".join(sells[:12]),
            })
            last_chosen = buys

        # 执行日收盘估值
        ctx.mark_to_market(t, close_mat[t])
        if short_rate > 0:
            short_val = sum(max(0.0, -sh * ctx._last_close.get(c, 0.0))
                            for c, sh in ctx.positions.items())
            ctx.cash -= short_val * short_rate / 252.0
        nav[t] = ctx.portfolio_value / capital
        bench[t] = bench[t - 1] * (1.0 + bench_ret)
        holdings_history.append(dict(ctx.positions))
        cash_history.append(ctx.cash)
        pv = ctx.portfolio_value or 1.0
        weight_history.append({c: sh * ctx._last_close.get(c, 0.0) / pv
                               for c, sh in ctx.positions.items()})

    if start_idx > 0:
        nav = nav[start_idx:]
        bench = bench[start_idx:]
        dates_out = dates[start_idx:]
        trades = [t for t in trades if t["date"] >= start_ts]
        trades_detail = [t for t in trades_detail if t["date"] >= start_ts]
        holdings_history = holdings_history[start_idx - 1:] if start_idx > 0 else holdings_history
        weight_history = weight_history[start_idx - 1:] if start_idx > 0 else weight_history
        cash_history = cash_history[start_idx - 1:] if start_idx > 0 else cash_history
    else:
        dates_out = dates

    nav_s = pd.Series(nav, index=dates_out, name="nav")
    bench_s = pd.Series(bench, index=dates_out, name="bench")
    trades_df = pd.DataFrame(trades)

    last_hold = ctx.positions
    if last_hold:
        last_close_vals = [ctx._last_close.get(c, np.nan) for c in last_hold]
        weights = [sh * px / nav_s.iloc[-1] / capital for sh, px in
                   zip(last_hold.values(), last_close_vals)]
        holdings_df = pd.DataFrame({
            "code": list(last_hold.keys()),
            "weight": weights,
            "price": last_close_vals,
            "direction": ["空" if sh < 0 else "多" for sh in last_hold.values()],
        })
        holdings_df["market_value"] = holdings_df["weight"] * nav_s.iloc[-1] * capital
        holdings_df["weight_pct"] = holdings_df["weight"] * 100
        holdings_df = holdings_df.sort_values("weight", ascending=False).reset_index(drop=True)
    else:
        holdings_df = pd.DataFrame(columns=["code", "weight", "price",
                                            "market_value", "weight_pct"])

    metrics = compute_metrics(nav_s)
    metrics.update(compute_excess_metrics(nav_s, bench_s))

    return {
        "nav": nav_s,
        "bench": bench_s,
        "drawdown": drawdown_series(nav_s),
        "metrics": metrics,
        "bench_metrics": compute_metrics(bench_s),
        "trades": trades_df,
        "trades_detail": trades_detail,
        "holdings": holdings_df,
        "last_signal_date": dates_out[-2] if len(dates_out) >= 2 else None,
        "capital": capital,
        "dates": dates_out,
        "weight_history": weight_history,
        "positions_history": holdings_history,
        "cash_history": cash_history,
        "last_chosen": last_chosen,
        "factor_quality": None,
    }


# ============================================================
# 内置示例：双均线金叉事件策略
# 金叉日买入（最多持 top_n 只，等权），死叉日清仓。
# ============================================================
class GoldenCrossStrategy(EventStrategy):
    short = 5
    long = 20
    top_n = 3
    max_weight = 0.5  # 单票目标权重上限，防止资金过度集中

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        # 清仓：持仓中当前出现死叉的
        for code in list(ctx.positions):
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            short_prev = sum(closes[-self.short - 1:-1]) / self.short
            long_prev = sum(closes[-self.long - 1:-1]) / self.long
            short_now = sum(closes[-self.short:]) / self.short
            long_now = sum(closes[-self.long:]) / self.long
            if short_prev >= long_prev and short_now < long_now:
                ctx.order_target_pct(code, 0.0)

        # 计算剩余可新增仓位（持仓数受 top_n 限制）
        held = [c for c, sh in ctx.positions.items() if sh > 0]
        slots = max(0, self.top_n - len(held))

        # 买入：出现金叉且当前无仓的，按金叉强度排序取 top_n
        scores: list[tuple[float, str]] = []
        for code in bar.tradable:
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            short_prev = sum(closes[-self.short - 1:-1]) / self.short
            long_prev = sum(closes[-self.long - 1:-1]) / self.long
            short_now = sum(closes[-self.short:]) / self.short
            long_now = sum(closes[-self.long:]) / self.long
            if short_prev <= long_prev and short_now > long_now:
                scores.append((short_now - long_now, code))

        scores.sort(reverse=True)
        w = min(self.max_weight, 1.0 / self.top_n)
        for _, code in scores:
            if slots <= 0:
                break
            if ctx.position(code) > 0:
                continue
            ctx.order_target_pct(code, w)
            slots -= 1


# ============================================================
# 内置示例：风险平价组合策略
# 每周一从可交易池中按 20 日均成交额取 top_n，用历史收益做风险平价定权重。
# 演示 ctx.optimize_risk_parity / optimize_mean_variance 的用法。
# ============================================================
class RiskParityStrategy(EventStrategy):
    top_n = 5
    window = 60
    max_weight = 0.4
    rebalance_weekday = 0  # 周一调仓

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        if bar.exec_date.weekday() != self.rebalance_weekday:
            return
        tradable = list(bar.tradable)
        if not tradable:
            return
        ranked = sorted(tradable, key=lambda c: -bar.am20.get(c, 0.0))[: self.top_n]
        weights = ctx.optimize_risk_parity(ranked, self.window, self.max_weight)
        if not weights:
            return
        # 清仓不在优化组合里的股票
        for code in list(ctx.positions):
            if code not in weights:
                ctx.order_target_pct(code, 0.0)
        for code, w in weights.items():
            ctx.order_target_pct(code, w)


# ============================================================
# 内置示例：多空动量对冲策略
# 每周一按 20 日动量取 TopN 多头 + 最弱 N 只空头（需支持融券）。
# 多头/空头各用 50% 名义资金，净敞口约 0。
# ============================================================
class LongShortMomentumStrategy(EventStrategy):
    mom_window = 20
    long_n = 3
    short_n = 3
    long_notional = 0.5
    short_notional = 0.5

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        if bar.exec_date.weekday() != 0:
            return
        scores: dict[str, float] = {}
        for code in bar.tradable:
            closes = ctx.close_series(code, self.mom_window + 1)
            if len(closes) < self.mom_window + 1:
                continue
            scores[code] = closes[-1] / closes[0] - 1.0
        if len(scores) < self.long_n + self.short_n:
            return
        ranked = sorted(scores, key=scores.get)  # 弱 → 强
        longs = ranked[-self.long_n:]
        shorts = ranked[:self.short_n]

        # 清掉所有旧持仓（多头卖出 / 空头回补）
        for code in list(ctx.positions):
            ctx.order_target_shares(code, 0)
        for code in longs:
            ctx.order_target_pct(code, self.long_notional / self.long_n)
        for code in shorts:
            ctx.order_target_pct(code, -self.short_notional / self.short_n)

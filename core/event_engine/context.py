from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ============================================================
# 事件驱动回测引擎 — 上下文与撮合
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
        from ..portfolio import weights_from_returns
        r = self._returns_window(codes, window)
        if r.shape[0] < 2 or r.shape[1] == 0:
            return {}
        return weights_from_returns(r, "risk_parity", max_weight=max_weight)

    def optimize_mean_variance(self, codes: list[str], window: int = 60,
                               gamma: float = 1.0,
                               max_weight: float = 0.4) -> dict[str, float]:
        """均值方差优化：最大化 w'μ - γ·w'Σw，返回 {code: 权重}。"""
        from ..portfolio import weights_from_returns
        r = self._returns_window(codes, window)
        if r.shape[0] < 2 or r.shape[1] == 0:
            return {}
        return weights_from_returns(r, "mean_variance", gamma=gamma,
                                    max_weight=max_weight)

    def optimize_max_diversification(self, codes: list[str], window: int = 60,
                                     max_weight: float = 0.4) -> dict[str, float]:
        """最大化分散化权重：DR = Σ(w_i·σ_i) / sqrt(w'Σw)。"""
        from ..portfolio import weights_from_returns
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

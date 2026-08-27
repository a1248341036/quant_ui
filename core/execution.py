"""资产执行层。

第一版只抽取股票/场内 ETF 共用的现金撮合语义。它不负责选股，输入是
组合层生成的目标权重；因此因子、组合和成交可以独立演进。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    cash: float
    positions: dict[int, float]
    buy_amount: float = 0.0
    sell_amount: float = 0.0
    bought_codes: list[str] = field(default_factory=list)
    sold_codes: list[str] = field(default_factory=list)
    trades_detail: list[dict] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)


class _CashExecutionAdapter:
    """现金/整手/开盘成交执行器。

    该类刻意保持当前 ``core.engine`` 的撮合顺序：先卖后买；目标权重只
    影响买入预算，卖不出时保留持仓。它不改变信号日和执行日的定义。
    """

    def __init__(
        self,
        codes: list[str],
        open_mat: np.ndarray,
        valid_open: np.ndarray,
        am20_mat: np.ndarray,
        turnover_mat: np.ndarray,
        limit_up: np.ndarray | None,
        limit_down: np.ndarray | None,
        dates: pd.DatetimeIndex,
        buy_cost: float,
        sell_cost: float,
        lot_size: int,
        slippage_bps: float,
        max_participation: float,
        spread_bps: float = 0.0,
        min_commission: float = 0.0,
        impact_coef: float = 0.0,
        impact_vol: float = 0.02,
    ) -> None:
        self.codes = codes
        self.open_mat = open_mat
        self.valid_open = valid_open
        self.am20_mat = am20_mat
        self.turnover_mat = turnover_mat
        self.limit_up = limit_up
        self.limit_down = limit_down
        self.dates = dates
        self.buy_cost = float(buy_cost)
        self.sell_cost = float(sell_cost)
        self.lot_size = int(lot_size)
        self.slippage = float(slippage_bps) / 1e4
        self.max_participation = float(max_participation)
        self.spread = float(spread_bps) / 1e4
        self.min_commission = max(float(min_commission), 0.0)
        # 平方根冲击模型：impact = impact_coef * impact_vol * sqrt(成交额/ADV)。
        # impact_coef=0 时关闭；impact_vol 为代表性日波动率（默认2%）。
        self.impact_coef = max(float(impact_coef), 0.0)
        self.impact_vol = max(float(impact_vol), 1e-6)

    def _impact(self, trade_value: float, adv_value: float | None) -> float:
        """按本单金额占20日均成交额的比例返回冲击率（小数）。"""
        if (self.impact_coef <= 0 or not np.isfinite(trade_value)
                or trade_value <= 0 or adv_value is None
                or not np.isfinite(adv_value) or adv_value <= 0):
            return 0.0
        return float(self.impact_coef * self.impact_vol
                     * np.sqrt(trade_value / adv_value))

    def _fee(self, amount: float, rate: float) -> float:
        return max(amount * rate, self.min_commission) if amount > 0 else 0.0

    def execute_stop_losses(
        self,
        cash: float,
        positions: dict[int, float],
        close_mat: np.ndarray,
        valid_close: np.ndarray,
        stop_mat: np.ndarray,
        signal_idx: int,
        exec_idx: int,
    ) -> ExecutionResult:
        """执行 Chandelier 止损；无效开盘/跌停时保持原持仓。"""
        result = ExecutionResult(float(cash), positions)
        for k in list(positions.keys()):
            shares = positions[k]
            if shares <= 0:
                continue
            if not (valid_close[signal_idx, k]
                    and np.isfinite(stop_mat[signal_idx, k])
                    and close_mat[signal_idx, k] < stop_mat[signal_idx, k]):
                continue
            if not self.valid_open[exec_idx, k]:
                continue
            if self.limit_down is not None and self.limit_down[exec_idx, k]:
                continue
            px = float(self.open_mat[exec_idx, k]) * (1.0 - self.slippage - self.spread)
            amount = shares * px
            fee = self._fee(amount, self.sell_cost)
            result.cash += amount - fee
            result.positions.pop(k)
            result.sell_amount += amount
            result.sold_codes.append(self.codes[k])
            result.trades_detail.append({
                "date": self.dates[exec_idx],
                "signal_date": self.dates[signal_idx],
                "code": self.codes[k], "side": "sell",
                "shares": float(shares), "price": px, "fee": fee,
                "amount": amount, "status": "filled", "reason": "chandelier",
            })
        return result


class StockExecutionAdapter(_CashExecutionAdapter):
    """A 股股票现金执行器。"""

    def execute_targets(
        self,
        cash: float,
        positions: dict[int, float],
        targets: dict[int, float],
        chosen_list: list[int],
        portfolio_value: float,
        amount_threshold: float,
        signal_idx: int,
        exec_idx: int,
        max_weight: float | None = None,
    ) -> ExecutionResult:
        """按目标权重在执行日开盘成交，返回新现金、持仓和明细。"""
        result = ExecutionResult(float(cash), positions)
        signal_date = self.dates[signal_idx].date().isoformat()
        exec_date = self.dates[exec_idx].date().isoformat()

        # 先卖出：未进入目标组合的仓位清仓；停牌/跌停保留。
        for k in list(result.positions.keys()):
            if targets.get(k, 0.0) > 0:
                continue
            shares = result.positions[k]
            if shares <= 0:
                continue
            if not self.valid_open[exec_idx, k]:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "sell",
                    "status": "rejected", "reason": "停牌/无开盘价",
                })
                continue
            if self.limit_down is not None and self.limit_down[exec_idx, k]:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "sell",
                    "status": "rejected", "reason": "跌停卖不出",
                })
                continue
            px = float(self.open_mat[exec_idx, k]) * (1.0 - self.slippage - self.spread)
            impact = self._impact(shares * px,
                                  float(self.am20_mat[signal_idx, k])
                                  if np.isfinite(self.am20_mat[signal_idx, k]) else None)
            px *= (1.0 - impact)
            amount = shares * px
            fee = self._fee(amount, self.sell_cost)
            result.cash += amount - fee
            result.sell_amount += amount
            result.positions.pop(k)
            result.sold_codes.append(self.codes[k])
            result.trades_detail.append({
                "date": self.dates[exec_idx], "signal_date": self.dates[signal_idx],
                "code": self.codes[k], "side": "sell", "shares": float(shares),
                "price": px, "fee": fee, "amount": amount, "status": "filled",
            })

        # 后买入：预算、现金、整手和流动性依次限制成交量。
        for k in chosen_list:
            if not self.valid_open[exec_idx, k]:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "停牌/无开盘价",
                })
                continue
            if self.limit_up is not None and self.limit_up[exec_idx, k]:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "涨停买不进",
                })
                continue
            am20v = self.am20_mat[signal_idx, k]
            if not np.isfinite(am20v) or (np.isfinite(amount_threshold)
                                          and am20v < amount_threshold):
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "流动性不足(am20分位)",
                })
                continue
            turnover = self.turnover_mat[signal_idx, k]
            if not np.isfinite(turnover) or turnover <= 0:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "无成交量",
                })
                continue
            pct = targets[k]
            if max_weight:
                pct = min(pct, float(max_weight))
            budget = portfolio_value * pct
            px = float(self.open_mat[exec_idx, k]) * (1.0 + self.slippage + self.spread)
            # 冲击按目标预算额近似（整手股数未定前的名义金额）
            impact = self._impact(budget,
                                  float(self.am20_mat[signal_idx, k])
                                  if np.isfinite(self.am20_mat[signal_idx, k]) else None)
            px *= (1.0 + impact)
            gross = px * (1.0 + self.buy_cost)
            want_lots = int(budget // gross // self.lot_size)
            cash_lots = int(result.cash // gross // self.lot_size)
            lots = min(want_lots, cash_lots)
            if self.max_participation > 0:
                liq_lots = int((am20v * self.max_participation) / gross) // self.lot_size
                lots = min(lots, liq_lots)
            if lots <= 0:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "现金不足/预算过小",
                })
                continue
            shares = lots * self.lot_size
            fee = self._fee(shares * px, self.buy_cost)
            amount = shares * px
            result.cash -= amount + fee
            result.buy_amount += amount
            result.positions[k] = result.positions.get(k, 0.0) + shares
            result.bought_codes.append(self.codes[k])
            result.trades_detail.append({
                "date": self.dates[exec_idx], "signal_date": self.dates[signal_idx],
                "code": self.codes[k], "side": "buy", "shares": float(shares),
                "price": px, "fee": fee, "amount": amount, "status": "filled",
            })
        return result


class ETFExecutionAdapter(StockExecutionAdapter):
    """场内 ETF 执行器。

    ETF 与股票共享开盘成交、现金、整手和流动性算法，但通过独立类保留
    ETF 规则的扩展点。当前涨跌停是否启用由上层传入；默认 ETF 入口关闭
    股票式涨跌停过滤，后续可加入 ETF 专用涨跌幅/申赎/折溢价模型。
    """

    asset_type = "etf"


# ---------- 场外基金申赎费率 ----------

# 持有期阶梯赎回费率（天数下限 → 费率）
_FUND_REDEEM_TIERS = [
    (0,   0.015),   # < 7 天：1.5%
    (7,   0.0075),  # 7~30 天：0.75%
    (30,  0.005),   # 30~365 天：0.5%
    (365, 0.0),     # >= 1 年：0%
]


def _redeem_fee_rate(holding_days: int) -> float:
    """按持有天数查阶梯赎回费率。"""
    rate = 0.0
    for threshold, r in _FUND_REDEEM_TIERS:
        if holding_days >= threshold:
            rate = r
    return rate


def detect_fund_share_class(name: str) -> str:
    """从基金简称识别 A/C 类。

    返回 'A' / 'C' / 'AC'（后端份额）/ ''（无后缀，按 A 类处理）。
    """
    if not name:
        return ""
    n = name.upper().strip()
    if "(后端)" in name or "后端" in name:
        return "AC"
    # 结尾是 A/C/AC/E 的识别（常见命名：xxx混合A / xxx混合C）
    for suffix in ("AC", "A", "C", "E"):
        if n.endswith(suffix):
            return "A" if suffix == "E" else suffix
    return ""


class FundNavExecutionAdapter(_CashExecutionAdapter):
    """场外基金净值执行器。

    与股票/ETF 的核心区别：
    - 份额计算：金额 → 扣申购费 → 按净值折算份额（向下取整 0.01 份）
    - 申购费：A 类前端扣费，C 类无申购费
    - 赎回费：按持有天数阶梯计算（<7天1.5% / <30天0.75% / <1年0.5% / >1年0%）
    - 持有期追踪：每笔买入记录日期，赎回时 FIFO 匹配
    - 无涨跌停/停牌/流动性过滤（基金申赎无这些约束）
    - lot_size 概念改为最小份额精度 0.01 份
    """

    asset_type = "fund_nav"

    def __init__(
        self,
        codes: list[str],
        open_mat: np.ndarray,
        valid_open: np.ndarray,
        am20_mat: np.ndarray,
        turnover_mat: np.ndarray,
        limit_up: np.ndarray | None,
        limit_down: np.ndarray | None,
        dates: pd.DatetimeIndex,
        buy_cost: float,
        sell_cost: float,
        lot_size: int,
        slippage_bps: float,
        max_participation: float,
        share_classes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            codes=codes, open_mat=open_mat, valid_open=valid_open,
            am20_mat=am20_mat, turnover_mat=turnover_mat,
            limit_up=limit_up, limit_down=limit_down, dates=dates,
            buy_cost=buy_cost, sell_cost=sell_cost, lot_size=lot_size,
            slippage_bps=slippage_bps, max_participation=max_participation,
        )
        # share_classes: {基金代码: 'A'/'C'/'AC'/''}，None 时全部按 A 类处理
        self.share_classes = share_classes or {}
        # 持仓的买入批次：{列索引: [(份额, 买入日期索引), ...]}
        self._lots: dict[int, list[tuple[float, int]]] = {}

    def _buy_fee_rate(self, k: int) -> float:
        """申购费率：A 类前端收费，C 类无申购费。"""
        cls = self.share_classes.get(str(self.codes[k]), "A")
        if cls == "C":
            return 0.0
        # A 类 / AC 类前端 / 无后缀：收申购费
        return self.buy_cost

    def _subscribe(self, amount: float, nav: float, k: int, exec_idx: int) -> tuple[float, float]:
        """金额 → 扣申购费 → 按净值折算份额。

        返回 (份额, 申购费)。份额向下取整到 0.01 份。
        """
        fee_rate = self._buy_fee_rate(k)
        fee = amount * fee_rate
        investable = amount - fee
        shares = float(int(investable / nav * 100)) / 100.0  # 0.01 份精度
        return shares, fee

    def _redeem(self, shares: float, nav: float, k: int, exec_idx: int) -> tuple[float, float, float]:
        """赎回份额 → 按持有期阶梯扣赎回费。

        FIFO 匹配买入批次计算持有天数。返回 (到账金额, 赎回费, 剩余份额)。
        """
        lots = self._lots.get(k, [])
        remaining = shares
        proceeds = 0.0
        fee_total = 0.0
        exec_date = self.dates[exec_idx]

        while remaining > 0 and lots:
            lot_shares, lot_idx = lots[0]
            if lot_shares <= 0:
                lots.pop(0)
                continue
            take = min(remaining, lot_shares)
            holding_days = (exec_date - self.dates[lot_idx]).days
            fee_rate = _redeem_fee_rate(holding_days)
            gross = take * nav
            fee = gross * fee_rate
            proceeds += gross - fee
            fee_total += fee
            remaining -= take
            lot_shares -= take
            if lot_shares <= 0.001:
                lots.pop(0)
            else:
                lots[0] = (lot_shares, lot_idx)

        # 如果没有批次记录（如预热段遗留持仓），按最长持有期 0% 处理
        if remaining > 0:
            gross = remaining * nav
            proceeds += gross
        return proceeds, fee_total, 0.0

    def execute_stop_losses(
        self,
        cash: float,
        positions: dict[int, float],
        close_mat: np.ndarray,
        valid_close: np.ndarray,
        stop_mat: np.ndarray,
        signal_idx: int,
        exec_idx: int,
    ) -> ExecutionResult:
        """基金止损：按 exec 日净值赎回。"""
        result = ExecutionResult(float(cash), positions)
        for k in list(positions.keys()):
            shares = positions[k]
            if shares <= 0:
                continue
            if not (valid_close[signal_idx, k]
                    and np.isfinite(stop_mat[signal_idx, k])
                    and close_mat[signal_idx, k] < stop_mat[signal_idx, k]):
                continue
            if not self.valid_open[exec_idx, k]:
                continue
            nav = float(self.open_mat[exec_idx, k])
            proceeds, fee, _ = self._redeem(float(shares), nav, k, exec_idx)
            result.cash += proceeds
            result.positions.pop(k)
            self._lots.pop(k, None)
            result.sell_amount += float(shares) * nav
            result.sold_codes.append(self.codes[k])
            result.trades_detail.append({
                "date": self.dates[exec_idx],
                "signal_date": self.dates[signal_idx],
                "code": self.codes[k], "side": "sell",
                "shares": float(shares), "price": nav, "fee": fee,
                "amount": float(shares) * nav, "status": "filled",
                "reason": "chandelier",
            })
        return result

    def execute_targets(
        self,
        cash: float,
        positions: dict[int, float],
        targets: dict[int, float],
        chosen_list: list[int],
        portfolio_value: float,
        amount_threshold: float,
        signal_idx: int,
        exec_idx: int,
        max_weight: float | None = None,
    ) -> ExecutionResult:
        """基金申赎执行：先赎回非目标持仓，后申购目标持仓。"""
        result = ExecutionResult(float(cash), positions)
        signal_date = self.dates[signal_idx].date().isoformat()
        exec_date = self.dates[exec_idx].date().isoformat()

        # 先赎回：未进入目标组合的仓位全部赎回
        for k in list(result.positions.keys()):
            if targets.get(k, 0.0) > 0:
                continue
            shares = result.positions[k]
            if shares <= 0:
                continue
            if not self.valid_open[exec_idx, k]:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "sell",
                    "status": "rejected", "reason": "无净值",
                })
                continue
            nav = float(self.open_mat[exec_idx, k])
            proceeds, fee, _ = self._redeem(float(shares), nav, k, exec_idx)
            result.cash += proceeds
            result.sell_amount += float(shares) * nav
            result.positions.pop(k)
            self._lots.pop(k, None)
            result.sold_codes.append(self.codes[k])
            result.trades_detail.append({
                "date": self.dates[exec_idx], "signal_date": self.dates[signal_idx],
                "code": self.codes[k], "side": "sell", "shares": float(shares),
                "price": nav, "fee": fee, "amount": float(shares) * nav,
                "status": "filled",
            })

        # 后申购：按目标权重分配预算
        for k in chosen_list:
            if not self.valid_open[exec_idx, k]:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "无净值",
                })
                continue
            nav = float(self.open_mat[exec_idx, k])
            pct = targets[k]
            if max_weight:
                pct = min(pct, float(max_weight))
            budget = portfolio_value * pct
            budget = min(budget, result.cash)
            if budget < nav * 0.01:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "现金不足/预算过小",
                })
                continue
            shares, fee = self._subscribe(budget, nav, k, exec_idx)
            if shares <= 0:
                result.rejections.append({
                    "date": exec_date, "signal_date": signal_date,
                    "code": self.codes[k], "side": "buy",
                    "status": "rejected", "reason": "份额不足0.01",
                })
                continue
            amount = shares * nav
            result.cash -= amount + fee
            result.buy_amount += amount
            result.positions[k] = result.positions.get(k, 0.0) + shares
            # 记录买入批次
            self._lots.setdefault(k, []).append((shares, exec_idx))
            result.bought_codes.append(self.codes[k])
            result.trades_detail.append({
                "date": self.dates[exec_idx], "signal_date": self.dates[signal_idx],
                "code": self.codes[k], "side": "buy", "shares": float(shares),
                "price": nav, "fee": fee, "amount": amount, "status": "filled",
            })
        return result

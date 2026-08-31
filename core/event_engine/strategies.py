from __future__ import annotations

from .context import Context, Bar, EventStrategy


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

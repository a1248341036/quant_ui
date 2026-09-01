# -*- coding: utf-8 -*-
"""日线策略模板 —— 代码优先、显式开启。

用法（用户只写两个类）：

    from strategies.template.daily import DailyStrategy, RiskPolicy

    class 稳健风控(RiskPolicy):
        def __init__(self):
            self.stoploss = 0.07           # 不写 = 没有止损
            self.take_profit = 2.0         # 止盈 100%
            self.cooldown_days = 1         # 止损后禁买天数
            self.market_crash = 0.05       # 大盘惨跌清仓
            self.pass_ranges = [("01-05", "02-05")]   # 空仓日期段
            self.pass_months = (4,)        # 空仓月
            self.max_single_nav = 0.25     # 底线: 单票占净值上限
            self.daily_loss_limit = 0.05   # 底线: 单日亏损熔断(较前一日净值)
            self.blacklist = ("600000",)   # 底线: 黑名单(只拦买入)
            self.limitup_open_sell = True  # 昨日涨停今开板卖出(默认开)
            self.pre_buy_hook = my_fn      # 可选: 自定义买入检查(ctx, code, pct)->pct|None

    class 小盘三正(DailyStrategy):
        data_ctx = _CTX                # 截面提供者(build_context())
        risk = 稳健风控                # def 了挂上就触发; 不写 = 无风控
        stock_num = 7                  # 不写 = 5
        rebalance_weekday = 1          # 不写 = 周一(0=周一)
        position_mode = "cash_equal"   # 不写 = "target_pct" 目标等权
        reserve_cash = 0
        min_buy_value = None
        top_keep = 50
        highest = None                 # 信号日收盘价上限(持仓豁免)

        def select(self, snap):
            ...
            return [code, ...]         # 按优先级排序的候选

    EVENT_STRATEGIES = {"小盘三正": 小盘三正}

原则:
- 所有能力默认关闭; 在 RiskPolicy/策略类上写了对应属性才触发。
- 引擎以 strategy_class() 无参实例化(runner.py), 用户 __init__ 正常执行。
- 不修改 core/event_engine、core/paper 与 strategies/event/_runtime.py。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.event_engine.context import Context, EventStrategy

__all__ = ["RiskPolicy", "DailyStrategy"]


class RiskPolicy:
    """风控配置基类：全部默认关，子类 __init__ 里写了才触发。"""

    # 类级默认(全关)。子类 __init__ 里的实例属性覆盖这里。
    stoploss: float | None = None
    take_profit: float | None = None
    cooldown_days: int = 0
    market_crash: float | None = None
    pass_ranges: list | None = None
    pass_months: tuple = ()
    max_single_nav: float | None = None
    daily_loss_limit: float | None = None
    blacklist: tuple = ()
    limitup_open_sell: bool = True
    pre_buy_hook = None


class DailyStrategy(EventStrategy):
    """日线策略基类：子类写 select()（必须）+ 仓位属性（可选）+ risk（可选）。"""

    # ── 策略类配置(类属性或 __init__ 里 super().__init__() 后设实例属性均可) ──
    data_ctx = None                 # 截面提供者(JQContext): .snapshot(date)
    risk: type[RiskPolicy] | RiskPolicy | None = None
    stock_num: int = 5
    rebalance_weekday: int = 1      # 0=周一
    position_mode: str = "target_pct"
    reserve_cash: float = 0.0
    min_buy_value: float | None = None
    top_keep: int = 50
    highest: float | None = None

    # ================= 引擎生命周期 =================
    def init(self, ctx: Context) -> None:
        self._cost: dict[str, float] = {}     # code -> 参考成本价
        self._cooldown_left: int = 0
        self._prev_nav: float | None = None
        self._day_breached: bool = False
        # 挂载风控: 类属性引用类或实例均可, 统一为实例
        risk = type(self).risk
        if risk is None:
            self.policy = None
        elif isinstance(risk, RiskPolicy):
            self.policy = risk
        elif isinstance(risk, type) and issubclass(risk, RiskPolicy):
            self.policy = risk()
        else:
            raise TypeError(
                f"risk 必须是 RiskPolicy 子类/实例, 得到 {type(risk)}")

    # ================= 子类必须实现 =================
    def select(self, snap: pd.DataFrame) -> list[str]:
        """信号日截面 -> 按优先级排序的候选 code 列表。"""
        raise NotImplementedError("策略类必须实现 select(snap)")

    # ================= 内部工具 =================
    def _snap(self, sig) -> pd.DataFrame:
        jq = getattr(type(self), "data_ctx", None)
        if jq is None:
            return pd.DataFrame()
        return jq.snapshot(sig)

    def _hl_set(self, sig) -> set[str]:
        jq = getattr(type(self), "data_ctx", None)
        tables = getattr(jq, "tables", None)
        fn = getattr(tables, "hl_codes", None) if tables is not None else None
        if callable(fn):
            try:
                return set(fn(sig))
            except Exception:
                return set()
        return set()

    def _p(self, name: str, default=None):
        """读风控配置(挂了 risk 用它的, 否则默认关)。"""
        if self.policy is None:
            return default
        return getattr(self.policy, name, default)

    def _s(self, name: str):
        """读策略配置(实例属性优先, 回落类属性)。"""
        v = getattr(self, name, None)
        if v is not None:
            return v
        return type(self).__dict__.get(name)

    def _in_pass_window(self, sig: pd.Timestamp) -> bool:
        md = sig.strftime("%m-%d")
        for lo, hi in (self._p("pass_ranges") or []):
            if lo <= md <= hi:
                return True
        months = self._p("pass_months") or ()
        if months and sig.month in months:
            return True
        return False

    def _sell(self, c: Context, code: str) -> None:
        c.order_target_pct(code, 0.0)
        self._cost.pop(code, None)

    # ================= 底线风控 =================
    def _gate_buy(self, c: Context, code: str, pct: float) -> float | None:
        """新买入前的底线检查。返回调整后的 pct; None = 拦下。"""
        if str(code) in set(self._p("blacklist") or ()):
            return None
        hook = self._p("pre_buy_hook")
        if callable(hook):
            pct = hook(c, code, pct)
            if pct is None:
                return None
        cap = self._p("max_single_nav")
        if cap:
            nav = c.portfolio_value or 0.0
            cur_pct = (c.position_value(code) / nav) if nav else 0.0
            pct = max(min(float(pct), float(cap) - cur_pct), 0.0)
            if pct <= 0:
                return None
        return pct

    def _update_daily_breach(self, c: Context) -> None:
        """单日亏损熔断: 今日净值较昨日回撤超阈值 -> 今日禁新买。"""
        self._day_breached = False
        limit = self._p("daily_loss_limit")
        nav = c.portfolio_value or 0.0
        if limit and self._prev_nav:
            if nav < self._prev_nav * (1.0 - float(limit)):
                self._day_breached = True
        self._prev_nav = nav

    # ================= 每日主流程 =================
    def on_bar(self, c: Context, bar) -> None:
        sig = bar.date
        # 熔断监测(每日都跑, 保持净值跟踪连续)
        self._update_daily_breach(c)

        # 清理失效成本
        for code in list(self._cost):
            if code not in c.positions:
                del self._cost[code]
        # 冷却倒计时
        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        # 空仓窗口: 清仓并跳过
        if self._in_pass_window(sig):
            for code in list(c.positions):
                self._sell(c, code)
            return

        snap = self._snap(sig)
        hl_set = self._hl_set(sig)

        # ---- 每日风控: 止损 / 止盈 / 涨停开板 ----
        for code in list(c.positions):
            ac = self._cost.get(code)
            px = c.last_close(code)
            if not ac or not px:
                continue
            tp = self._p("take_profit")
            if tp and px >= ac * float(tp):
                self._sell(c, code)                 # 止盈
                continue
            sl = self._p("stoploss")
            if sl and px < ac * (1.0 - float(sl)):
                self._sell(c, code)                 # 止损
                self._cooldown_left = max(self._cooldown_left,
                                          int(self._p("cooldown_days") or 0))
                continue
            if (self.policy is not None
                    and self._p("limitup_open_sell", True)
                    and code in hl_set):
                op = bar.open.get(code)
                if op is not None and op < px:
                    self._sell(c, code)             # 昨日涨停今开板

        # ---- 大盘惨跌清仓 ----
        mc = self._p("market_crash")
        if mc and bar.close and bar.open:
            drops = [1.0 - bar.close[s] / bar.open[s]
                     for s in bar.close if s in bar.open and bar.open[s] > 0]
            if drops and float(np.mean(drops)) >= float(mc):
                for code in list(c.positions):
                    self._sell(c, code)
                return

        # ---- 调仓日 ----
        if bar.exec_date.weekday() != int(self._s("rebalance_weekday") or 1):
            return
        target = list(self.select(snap)) if len(snap) else []
        # stock_num 是目标持仓数的硬约束: 已持仓的排前(豁免卖出),
        # 剩余名额给新目标, 超出部分截断
        held_in_pool = [x for x in target if x in c.positions]
        n = int(self._s("stock_num") or 5)
        slots = max(n - len(held_in_pool), 0)
        new_in_pool = [x for x in target if x not in c.positions][:slots]
        target = held_in_pool + new_in_pool
        top_keep = self._s("top_keep")
        if top_keep:
            target = target[: max(int(top_keep), len(held_in_pool))]

        # 股价上限过滤(持仓豁免)
        highest = self._s("highest")
        if highest and len(snap):
            raw_px = snap["close_raw"]
            target = [code for code in target
                      if code in c.positions
                      or not np.isfinite(raw_px.get(code, np.nan))
                      or raw_px.get(code) <= float(highest)]

        # 卖出: 不在目标且非昨日涨停(豁免)的持仓
        for code in list(c.positions):
            if code not in target and code not in hl_set:
                self._sell(c, code)

        # 买入池
        buy_list = [x for x in target if x not in c.positions]
        if not buy_list:
            return
        if self._cooldown_left > 0:      # 止损冷却: 暂缓新买
            return
        if self._day_breached:           # 单日熔断: 禁新买
            return

        self._position(c, bar, target, buy_list)

    # ================= 仓位 =================
    def _position(self, c: Context, bar, target: list[str],
                  buy_list: list[str]) -> None:
        mode = self._s("position_mode") or "target_pct"

        if mode == "cash_equal":
            # 剩余现金均分(只买新目标), 保留现金/最小金额可配
            cash = max(c.cash - float(self._s("reserve_cash") or 0.0), 0.0)
            per_val = cash / len(buy_list)
            min_buy = self._s("min_buy_value")
            lot = c.lot_size
            for code in buy_list:
                if min_buy and per_val < float(min_buy):
                    continue
                px = c.last_close(code)
                if not px:
                    continue
                shares = int(per_val / px // lot) * lot
                if shares > 0:
                    c.order_target_shares(code, shares)
                    self._record_cost(c, bar, code)
            return

        # target_pct(默认): 目标组合等权再平衡(含已持有), 底线单票上限截断
        n = len(target)
        if n == 0:
            return
        per = 1.0 / n
        cap = self._p("max_single_nav")
        if cap:
            per = min(per, float(cap))
        for code in target:
            if code in c.positions:
                c.order_target_pct(code, per)       # 已持有: 再平衡到目标
            else:
                gated = self._gate_buy(c, code, per)
                if gated is not None and gated > 0:
                    c.order_target_pct(code, gated)
                    self._record_cost(c, bar, code)

    def _record_cost(self, c: Context, bar, code: str) -> None:
        """记录买入参考成本(执行日开盘 x (1+买入费率), 与骨架同口径)。

        订单若撮合失败(涨停/停牌), 次日 on_bar 的失效清理会剔除。
        """
        op = bar.open.get(code)
        if op:
            self._cost[code] = float(op) * (1.0 + float(c.buy_cost or 0.0))

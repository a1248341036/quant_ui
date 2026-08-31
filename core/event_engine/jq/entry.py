# -*- coding: utf-8 -*-
"""聚宽模式回测入口。

流程: exec 用户代码(注册任务) -> 适配器(EventStrategy) -> run_event_backtest。
撮合/账户/净值全部由 core.event_engine 提供, 本模块不含撮合逻辑。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core.event_engine import run_event_backtest
from core.event_engine.jq.runtime import JQRuntime
from _runtime import JQContext  # noqa: E402  (facade: scripts+strategies 路径已在 runtime 注入)

_LAST_CTX: JQContext | None = None   # 调试/冒烟: 最近一次回测的数据上下文
_LAST_RT: JQRuntime | None = None


def run_jq_backtest(code: str, start: str, end: str | None = None,
                    capital: float = 100_000.0, warmup_days: int = 60,
                    lookback_buffer_days: int = 500,
                    buy_cost: float = 0.0001, sell_cost: float = 0.0011,
                    slippage_bps: float = 0.0) -> dict:
    """聚宽风格策略回测。返回 {metrics, nav, holdings, trades, logs}。"""
    global _LAST_CTX, _LAST_RT
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    lookback = max(800, (end_ts - pd.Timestamp(start)).days + lookback_buffer_days)
    ctx = JQContext(end=end_ts.date().isoformat(), lookback_days=lookback)
    t0 = time.time()
    rt = JQRuntime(code, ctx, capital=capital)
    init_fn = rt._init_fn
    # 干跑 initialize: 采集 set_order_cost/set_slippage(引擎费率需在建引擎前
    # 确定), 随后清空调度/挂单, 由引擎 init 正式注册
    if init_fn is not None:
        try:
            init_fn(rt.context)
        except Exception:
            pass
        rt.scheduled = []
        rt.pending_orders = []
    exec_logs = list(rt.log.buffer)
    cfg = dict(rt.cost_cfg)
    if cfg.get("buy_cost"):
        buy_cost = cfg["buy_cost"]
    if cfg.get("sell_cost"):
        sell_cost = cfg["sell_cost"]
    if cfg.get("slippage_bps"):
        slippage_bps = cfg["slippage_bps"]
    if cfg:
        rt.log.info(f"[runtime] 应用策略内费率: buy={buy_cost:.5f} "
                    f"sell={sell_cost:.5f} slippage={slippage_bps:.1f}bps")
    _LAST_CTX, _LAST_RT = ctx, rt

    class _JQAdapter:
        """事件引擎适配器: init=用户 initialize; on_bar=按序跑注册函数。"""

        def init(self, engine_ctx) -> None:
            rt.bind(engine_ctx)
            if init_fn is not None:
                init_fn(rt.context)

        def on_bar(self, engine_ctx, bar) -> None:
            rt.run_day(bar)

    res = run_event_backtest(
        panel=ctx.panel, codes=ctx.codes, strategy_class=_JQAdapter,
        start=start, end=end_ts.date().isoformat(), capital=capital,
        buy_cost=buy_cost, sell_cost=sell_cost, slippage_bps=slippage_bps,
        max_participation=0.0, lot_size=100, warmup_days=warmup_days,
        amount_q=0.2, limit_flags=True,
    )
    exec_logs += rt.log.buffer[len(exec_logs):]
    exec_logs.append(f"[runtime] 回测完成 {time.time()-t0:.0f}s, "
                     f"候选域 {len(ctx.codes)} 只")

    nav = res["nav"]
    metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in res["metrics"].items()}
    holdings = res["holdings"].copy()
    if len(holdings):
        holdings["name"] = [ctx.name_map.get(str(c), "") for c in holdings["code"]]
    trades = res["trades"].copy()
    if len(trades):
        trades["date"] = trades["date"].astype(str)
    return {
        "ok": True,
        "metrics": metrics,
        "nav": [{"date": str(pd.Timestamp(d).date()), "value": float(v)}
                for d, v in nav.items()],
        "drawdown": [{"date": str(pd.Timestamp(d).date()),
                      "value": float(v)} for d, v in res["drawdown"].items()],
        "holdings": holdings.to_dict(orient="records"),
        "trades": trades.to_dict(orient="records"),
        "logs": exec_logs,
        "codes_count": len(ctx.codes),
        "start": start, "end": end_ts.date().isoformat(),
        "capital": capital,
    }

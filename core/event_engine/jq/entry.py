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
from core.event_engine.runner import BacktestAborted
from core.metrics import compute_jq_panel
from _runtime import JQContext  # noqa: E402  (facade: scripts+strategies 路径已在 runtime 注入)

_LAST_CTX: JQContext | None = None   # 调试/冒烟: 最近一次回测的数据上下文
_LAST_RT: JQRuntime | None = None


def run_jq_backtest(code: str, start: str, end: str | None = None,
                    capital: float = 100_000.0, warmup_days: int = 60,
                    lookback_buffer_days: int = 500,
                    buy_cost: float = 0.0001, sell_cost: float = 0.0011,
                    slippage_bps: float = 0.0,
                    smoke: bool = False,
                    progress=None, cancel_event=None) -> dict:
    """聚宽风格策略回测。返回 {metrics, nav, holdings, trades, logs}。

    smoke=True: 冒烟模式(短窗口+跳过分钟预取), 用于快速验证策略能否跑通。
    回测前自动执行 API 预检(_preflight_api), 缺失 API 秒级报错。
    progress(ev: dict): 阶段/进度事件回调, ev 形如
        {"phase": "context"|"minutes"|"engine", "done": int, "total": int,
         "date": "YYYY-MM-DD", "nav": float, "in_window": bool}
        (date/nav/in_window 仅 engine 阶段); 抛 BacktestAborted 即取消。
    cancel_event: threading.Event, 置位后在下一个检查点终止并抛 BacktestAborted。
    """
    global _LAST_CTX, _LAST_RT

    def _emit(phase: str, **kw) -> None:
        if progress is not None:
            progress({"phase": phase, **kw})

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise BacktestAborted("已手动停止")

    _check_cancel()
    _emit("context", done=0, total=1)
    # ---- API 预检(秒级): 缺失 API 不再等 90 秒回测后才炸 ----
    try:
        from core.event_engine.jq.preflight import preflight as _preflight
        missing = _preflight(code)
        if missing:
            raise RuntimeError(
                "策略引用了兼容层尚未支持的 API: "
                + ", ".join(missing)
                + " (详见 docs/jq_compat_迁移评估.md)")
    except RuntimeError:
        raise
    except Exception:
        pass  # 预检自身异常不阻断回测
    if smoke:
        warmup_days = min(warmup_days, 10)
        lookback_buffer_days = 30
        end_ts_smoke = pd.Timestamp(start) + pd.Timedelta(days=90)
        if end is None or pd.Timestamp(end) > end_ts_smoke:
            end = end_ts_smoke.date().isoformat()
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    lookback = max(45 if smoke else 800,
                   (end_ts - pd.Timestamp(start)).days + lookback_buffer_days)
    ctx = JQContext(end=end_ts.date().isoformat(), lookback_days=lookback)
    _emit("context", done=1, total=1)
    t0 = time.time()
    rt = JQRuntime(code, ctx, capital=capital, window_start=start)
    init_fn = rt._init_fn
    # 干跑 initialize: 采集 set_order_cost/set_slippage(引擎费率需在建引擎前
    # 确定), 随后清空调度/挂单, 由引擎 init 正式注册
    if init_fn is not None:
        try:
            init_fn(rt.context)
        except Exception:
            pass
        if not smoke:
            try:
                # 聚宽语义: 盘中下单按下单时点真实价成交 -> 预取调度时点分钟线
                minutes = ({t[3] for t in rt.scheduled
                            if t[3] and ":" in str(t[3])} | {"9:30"})

                def _min_prog(done: int, total: int) -> None:
                    _check_cancel()
                    _emit("minutes", done=done, total=total)

                rt.prefetch_minutes(sorted(minutes), progress=_min_prog)
            except BacktestAborted:
                raise
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
    buy_tax = float(cfg.get("buy_tax") or 0.0)
    sell_tax = float(cfg.get("sell_tax") or 0.0)
    fixed_slippage = float(cfg.get("fixed_slippage") or 0.0)
    if fixed_slippage:
        slippage_bps = 0.0     # 绝对价差由运行时在 fill_price 上加, 引擎侧置 0
    min_commission = float(cfg.get("min_commission") or 0.0)
    if cfg:
        extra = f" min_commission={min_commission:.2f}" if min_commission else ""
        if fixed_slippage:
            extra += f" 固定滑点={fixed_slippage:.4f}元"
        rt.log.info(f"[runtime] 应用策略内费率: 佣金买={buy_cost:.5f} "
                    f"卖={sell_cost:.5f} 税买={buy_tax:.4f} 卖={sell_tax:.4f} "
                    f"slippage={slippage_bps:.1f}bps{extra}")
    _LAST_CTX, _LAST_RT = ctx, rt
    rt.fixed_slippage = fixed_slippage
    start_ts = pd.Timestamp(start)

    def _eng_prog(t_idx: int, t_total: int, d, nav_v: float) -> None:
        _check_cancel()
        d_ts = pd.Timestamp(d)
        _emit("engine", done=t_idx, total=t_total,
              date=d_ts.date().isoformat(), nav=nav_v,
              in_window=bool(d_ts >= start_ts))

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
        min_commission=min_commission,
        buy_tax=buy_tax, sell_tax=sell_tax,
        max_participation=0.0, lot_size=100, warmup_days=warmup_days,
        amount_q=0.2, limit_flags=True,
        progress=_eng_prog,
    )
    exec_logs += rt.log.buffer[len(exec_logs):]
    exec_logs.append(f"[runtime] 回测完成 {time.time()-t0:.0f}s, "
                     f"候选域 {len(ctx.codes)} 只")

    nav = res["nav"]
    # 聚宽回测详情面板同口径指标(见 core/metrics.compute_jq_panel);
    # 基准用 set_benchmark 的真实指数日线(CNE index_bars), 缺失时超额类为 NaN
    bench_curve = None
    bench_code = rt.benchmark or "000300.XSHG"   # 聚宽默认基准: 沪深300
    try:
        ix = ctx.index_frame(str(bench_code))
        if ix is not None and len(ix):
            b_all = ix["close"].astype(float).dropna().sort_index()
            # 聚宽口径: 基点 = 窗口首日之前的收盘(首日涨跌计入基准收益)
            loc = int(b_all.index.searchsorted(nav.index[0]))
            base = float(b_all.iloc[loc - 1]) if loc > 0 else float(b_all.iloc[0])
            b = b_all.reindex(nav.index).ffill().bfill()
            bench_curve = b / base
    except Exception:
        bench_curve = None
    jq_panel = compute_jq_panel(
        nav=nav.astype(float), bench=bench_curve,
        fills=res.get("trades_detail") or [])
    metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in jq_panel.items()}
    holdings = res["holdings"].copy()
    if len(holdings):
        holdings["name"] = [ctx.name_map.get(str(c), "") for c in holdings["code"]]
    trades = res["trades"].copy()
    if len(trades):
        trades["date"] = trades["date"].astype(str)
    deals = []
    for f in res.get("trades_detail") or []:
        deals.append({
            "date": str(pd.Timestamp(f["date"]).date()),
            "code": str(f.get("code")),
            "side": str(f.get("side")),
            "shares": float(f.get("shares") or 0),
            "price": float(f.get("price") or 0),
            "amount": float(f.get("amount") or 0),
            "fee": float(f.get("fee") or 0),
        })
    result = {
        "ok": True,
        "metrics": metrics,
        "nav": [{"date": str(pd.Timestamp(d).date()), "value": float(v)}
                for d, v in nav.items()],
        "drawdown": [{"date": str(pd.Timestamp(d).date()),
                      "value": float(v)} for d, v in res["drawdown"].items()],
        "holdings": holdings.to_dict(orient="records"),
        "trades": trades.to_dict(orient="records"),
        "trades_detail": deals,
        "logs": exec_logs,
        "codes_count": len(ctx.codes),
        "start": start, "end": end_ts.date().isoformat(),
        "capital": capital,
        "bench_code": str(bench_code),
    }
    if bench_curve is not None:
        result["benchmark"] = [{"date": str(pd.Timestamp(d).date()), "value": float(v)}
                               for d, v in bench_curve.items()]
    return result

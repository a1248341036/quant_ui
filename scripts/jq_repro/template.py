# -*- coding: utf-8 -*-
"""聚宽策略复现 —— 策略骨架模板。

复现一个新策略只需要提供:
1) ReproConfig: 参数(持仓数/权重上限/风控阈值/调仓日/空仓月/费率...)
2) cand_fn(sig) -> list[(code, passed, is_hl)]
   "信号日候选, 按优先级排序"。passed=是否通过全部非持仓豁免过滤;
   is_hl=信号日收盘是否涨停。骨架自动做"持仓豁免 passed/is_hl"。
3) 可选 num_fn(sig) -> int  仓位数量映射(默认固定 cfg.stock_num;
   jq_data.ew_index + ma_num_fn 可复刻原版 MA sigmoid 连续仓位)

骨架已实现(与本次小市值复现逐行同源):
- 空仓月清仓(4月)          - 个股止损/止盈(信号日收盘 vs 买入开盘成本)
- 大盘惨跌清仓              - 涨停开板近似卖出(昨收涨停+今开未封死)
- 周度调仓: 持仓豁免过滤、昨日涨停豁免卖出、暴露上限+单票上限买入
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jq_data  # noqa: E402

from core.event_engine import EventStrategy, run_event_backtest  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402


@dataclass
class ReproConfig:
    start: str
    end: str
    capital: float = 1_000_000.0
    # 持仓/权重
    stock_num: int = 7
    max_single_weight: float = 0.12
    max_exposure: float = 0.70
    # 风控(0 = 关闭)
    stoploss: float = 0.07          # 个股止损(收盘 vs 成本)
    take_profit_multiple: float = 2.0   # 止盈倍数
    market_crash: float = 0.05      # 域平均(1-收/开) >= 该值 -> 清仓
    limit_open_sell: bool = True    # 昨收涨停+今开未封死 -> 开盘卖出
    # 日历
    rebalance_weekday: int = 1      # 0=周一 ... 1=周二(原 run_weekly(2))
    pass_months: tuple = (4,)       # 空仓月(持币)
    # 费率/撮合
    buy_cost: float = 0.0001
    sell_cost: float = 0.0011
    slippage_bps: float = 0.0
    warmup_days: int = 45
    # 输出
    out_dir: Path | None = None


def ma_num_fn(level: pd.Series, ma: pd.Series, base: int = 7,
              span: int = 3, scale_rel: float = 0.025,
              lo: int = 4) -> Callable:
    """复刻原版 sigmoid 仓位映射: num = base - span*sigmoid(diff_rel/scale_rel).

    原版: 中小板综 diff/500, 指数约2万点 -> 相对幅度 2.5%。
    """
    frac = 1.0 / (1.0 + np.exp(-((level - ma) / ma) / scale_rel))
    raw = (base - span * frac).round()
    clipped = raw.clip(lo, base).astype(int)
    return lambda sig: int(clipped.get(pd.Timestamp(sig), base))


def score_cand_fn(score_matrix: pd.DataFrame, tables: "jq_data.MarketTables",
                  top_keep: int = 50,
                  filters: Callable | None = None) -> Callable:
    """分数矩阵 -> cand_fn 桥接：把 AlphaAgent/DSL 挖出的因子接进事件引擎。

    score_matrix: date×code 分数矩阵（无需对齐 tables 的完整网格，
                  缺失日/缺失股自动跳过 = 天然过滤停牌/未上市/数据缺口）。
                  **分数越大越优先买入**（小市值类因子请传 -log(mv) 这类
                  "越大越好"的分数）。
    filters:      可选 tables 行号 i -> np.ndarray[bool]（与 tables.codes 对齐），
                  True=该股当日通过风险筛选；None=全部通过。
                  三正/ST/次新等由 jq_data 矩阵组合而来，与排序解耦。
                  **语义是"先过滤后排序"**（与聚宽 get_fundamentals WHERE+ORDER
                  一致）：先掩掉不合格股票，再在合格池内按分数降序取 top_keep。

    返回 cand_fn(sig) -> [(code, passed, is_hl), ...]：
    - passed = 分数存在 且 filters 通过
    - is_hl  = 信号日收盘涨停（骨架用它做涨停豁免）
    持仓豁免/涨停豁免卖出/仓位约束等由 template 骨架统一处理。
    """
    ci = {c: j for j, c in enumerate(tables.codes)}
    ranked: dict = {}
    for i, sig in enumerate(tables.dates):
        if sig not in score_matrix.index:
            ranked[sig] = []
            continue
        s = score_matrix.loc[sig].reindex(tables.codes)   # 对齐全网格
        vals = np.asarray(s.values, dtype=np.float64).copy()
        if filters is not None:
            vals[~np.asarray(filters(i), dtype=bool)] = np.nan   # 掩码不删位
        # 升序键: 有效分数取负(分数大者在前); NaN/-inf 沉底, 网格位置不变
        key = np.where(np.isfinite(vals) & (vals > -np.inf), -vals, np.inf)
        order = np.argsort(key, kind="stable")
        order = [j for j in order if np.isfinite(vals[j])][:top_keep]
        ranked[sig] = [tables.codes[j] for j in order]

    def cand_fn(sig):
        codes = ranked.get(pd.Timestamp(sig), [])
        if not codes:
            return []
        i = tables.index_of(sig)
        ok = filters(i) if filters is not None else None
        out = []
        for c in codes:
            j = ci[c]
            passed = bool(ok[j]) if ok is not None else True
            out.append((c, passed, bool(tables.hl[i, j])))
        return out

    return cand_fn


def make_strategy(cfg: ReproConfig, tables: jq_data.MarketTables,
                  cand_fn: Callable, num_fn: Callable | None = None):
    """构建事件引擎策略类。cand_fn/num_fn 闭包捕获预计算结果。"""

    class JQRepro(EventStrategy):
        def init(self, ctx) -> None:
            self.cost: dict[str, float] = {}

        # ---- 工具 ----
        def _sell(self, ctx, code: str) -> None:
            ctx.order_target_pct(code, 0.0)
            self.cost.pop(code, None)

        # ---- 每日 ----
        def on_bar(self, ctx, bar) -> None:
            sig = bar.date
            # 清理已不存在的成本记录
            for c in list(self.cost):
                if c not in ctx.positions:
                    del self.cost[c]

            # 空仓月清仓持币
            if bar.exec_date.month in cfg.pass_months:
                for c in list(ctx.positions):
                    self._sell(ctx, c)
                return

            self._daily_risk(ctx, bar, sig)

            if bar.exec_date.weekday() == cfg.rebalance_weekday:
                self._rebalance(ctx, bar, sig)

        # ---- 止损/止盈/大盘惨跌/涨停开板 ----
        def _daily_risk(self, ctx, bar, sig) -> None:
            hl_price = tables.hl_price_of(sig) if cfg.limit_open_sell else {}
            for code in list(ctx.positions):
                ac = self.cost.get(code)
                px = ctx.last_close(code)
                if ac and px and cfg.take_profit_multiple > 0:
                    if px >= ac * cfg.take_profit_multiple:
                        self._sell(ctx, code)
                        continue
                if ac and px and cfg.stoploss > 0:
                    if px < ac * (1 - cfg.stoploss):
                        self._sell(ctx, code)
                        continue
                # 涨停开板近似: 昨收涨停 且 今日开盘未封死 -> 开盘卖出
                lim = hl_price.get(code)
                if lim is not None and code in bar.open and bar.open[code] < lim:
                    self._sell(ctx, code)

            if cfg.market_crash > 0 and bar.close and bar.open:
                drops = [1.0 - bar.close[s] / bar.open[s]
                         for s in bar.close if s in bar.open and bar.open[s] > 0]
                if drops and float(np.mean(drops)) >= cfg.market_crash:
                    for code in list(ctx.positions):
                        self._sell(ctx, code)

        # ---- 调仓 ----
        def _rebalance(self, ctx, bar, sig) -> None:
            num = num_fn(sig) if num_fn else cfg.stock_num
            target: list[str] = []
            for code, passed, is_hl in cand_fn(sig):
                if len(target) >= num:
                    break
                if code in ctx.positions:
                    target.append(code)      # 持仓豁免全部过滤(原版语义)
                elif passed and not is_hl:
                    target.append(code)

            hl_set = tables.hl_codes(sig)    # 昨日涨停豁免卖出(含全部股票)
            for code in list(ctx.positions):
                if code not in target and code not in hl_set:
                    self._sell(ctx, code)

            buy_list = [c for c in target if c not in ctx.positions]
            if not buy_list:
                return
            pv = ctx.portfolio_value or cfg.capital
            exposure = sum(ctx.position_value(s) for s in ctx.positions) / pv
            sold = sum(ctx.position_value(s) for s in ctx.positions
                       if s not in target) / pv
            avail = cfg.max_exposure - (exposure - sold)
            per = min(cfg.max_single_weight, max(avail, 0.0) / len(buy_list))
            if per <= 0:
                return
            for code in buy_list:
                ctx.order_target_pct(code, per)
                op = bar.open.get(code)
                if op:
                    self.cost[code] = op * (1 + cfg.buy_cost)

    return JQRepro


# ============================================================
# 一键运行: 数据 -> 表格 -> 回测 -> 干净基准超额 -> 落盘
# ============================================================
def run(cfg: ReproConfig, cand_fn: Callable, num_fn: Callable | None = None,
        tables: jq_data.MarketTables | None = None,
        panel: pd.DataFrame | None = None,
        bench_level: pd.Series | None = None) -> dict:
    """tables/panel/bench_level 可外部传入(多策略共享一次数据构建)."""
    if tables is None or panel is None:
        engine_panel, meta, close_raw_df = jq_data.load_panel(
            cfg.start, cfg.end)
        tables = jq_data.build_tables(engine_panel, meta, close_raw_df)
        panel = engine_panel
    strat = make_strategy(cfg, tables, cand_fn, num_fn)
    res = run_event_backtest(
        panel=panel, codes=tables.codes, strategy_class=strat,
        start=cfg.start, end=cfg.end, capital=cfg.capital,
        buy_cost=cfg.buy_cost, sell_cost=cfg.sell_cost,
        slippage_bps=cfg.slippage_bps, max_participation=0.0,
        lot_size=100, warmup_days=cfg.warmup_days, limit_flags=True,
    )
    metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in res["metrics"].items()}

    # 干净基准: 域等权指数(裁剪伪影), 重算超额
    if bench_level is None:
        bench_level, _ = jq_data.ew_index(tables)
    nav = res["nav"]
    bench_s = bench_level.reindex(nav.index).ffill()
    bench_s = bench_s / bench_s.iloc[0]
    metrics["bench"] = {k: float(v) for k, v in compute_metrics(bench_s).items()}
    er = (nav.pct_change() - bench_s.pct_change()).dropna()
    metrics["超额年化"] = float((nav.iloc[-1] / bench_s.iloc[-1])
                               ** (252 / max(len(nav), 1)) - 1.0)
    metrics["超额夏普"] = (float(er.mean() / er.std() * np.sqrt(252))
                          if er.std() > 0 else 0.0)

    if cfg.out_dir is not None:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        nav.to_frame("nav").assign(bench=bench_s).to_csv(out / "nav.csv")
        res["trades"].to_csv(out / "trades.csv", index=False)
        res["holdings"].to_csv(out / "holdings.csv", index=False)
        res["drawdown"].to_frame("drawdown").to_csv(out / "drawdown.csv")
        import json
        (out / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
    return {"res": res, "metrics": metrics, "bench": bench_s}

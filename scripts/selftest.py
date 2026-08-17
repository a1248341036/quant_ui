#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quant_ui 自测：引擎 + API + 记账 + 数据。"""
from __future__ import annotations

import http.cookiejar
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import (load_etf, load_etf_panel, load_fund, load_fund_nav,  # noqa: E402
                       load_index, load_panel, load_tech, load_universe)
from core.composites import delete_composite as api_delete_composite  # noqa: E402
from core.composites import load_composites as api_load_composites  # noqa: E402
from core.composites import save_composite as api_save_composite  # noqa: E402
from core.engine import build_composite_factor, latest_signals, run_backtest  # noqa: E402
from core.ledger import compute_equity, current_positions  # noqa: E402
from core.store import LEGACY_DATA_DIR  # noqa: E402
from strategies.registry import STRATEGIES  # noqa: E402


BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0

# 带 cookie 的 opener（后续如需恢复鉴权直接复用）
_cj = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_cj))


def raw_request(path, method="GET", body=None, use_cookie=True):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    opener = _opener if use_cookie else urllib.request.build_opener()
    with opener.open(req, timeout=60) as resp:
        return resp.status, json.loads(resp.read().decode())


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def api(path, method="GET", body=None, use_cookie=True):
    _, data = raw_request(path, method, body, use_cookie=use_cookie)
    return data


def main() -> int:
    print("== 数据加载 ==")
    panel = load_panel()
    uni = load_universe()
    tech = load_tech()
    index = load_index()
    check("panel 非空", len(panel) > 100000 and panel["code"].nunique() > 500,
          f"rows={len(panel)} codes={panel['code'].nunique()}")
    check("panel 字段完整", {"date", "open", "close", "turnover", "amount", "turn20", "am20"} <= set(panel.columns))
    check("universe 含 code/name", {"code", "name"} <= set(uni.columns))
    check("tech 含 industry", "industry" in tech.columns)
    check("index 含 date/close", {"date", "close"} <= set(index.columns))

    print("== ETF / 场外基金数据 ==")
    etf = load_etf()
    etf_panel = load_etf_panel()
    fund = load_fund()
    fund_nav = load_fund_nav()
    check("ETF 池非空", len(etf) > 100, f"n={len(etf)}")
    check("ETF 面板非空", len(etf_panel) > 100000
          and etf_panel["code"].nunique() > 100,
          f"rows={len(etf_panel)} codes={etf_panel['code'].nunique()}")
    check("场外基金池非空", len(fund) > 100, f"n={len(fund)}")
    check("场外基金池剔除场内 ETF",
          not fund["name"].astype(str).str.contains("ETF", na=False).any())
    check("基金净值非空", len(fund_nav) > 100000
          and fund_nav["code"].nunique() > 100,
          f"rows={len(fund_nav)} codes={fund_nav['code'].nunique()}")

    print("== ETF 回测 ==")
    etf_codes = sorted(set(etf["code"]) & set(etf_panel["code"].unique()))
    etf_res = run_backtest(etf_panel, etf_codes, "mom20", False,
                           "2025-01-02", "2026-08-14", 5000, 3, "monthly",
                           affordable=True, limit_flags=False)
    check("ETF 回测 NAV 终点有限", np.isfinite(etf_res["nav"].iloc[-1]))
    check("ETF 回测 指标无 NaN",
          all(np.isfinite(v) for v in etf_res["metrics"].values()))
    check("ETF 回测 有调仓记录", len(etf_res["trades"]) > 0)

    print("== 场外基金回测 ==")
    from core.fund_engine import run_fund_backtest
    fund_codes = sorted(set(fund["code"]) & set(fund_nav["code"].unique()))
    f_res = run_fund_backtest(fund_nav, fund_codes, "mom20", False,
                              "2025-01-02", "2026-08-14", 5000, 3, "monthly",
                              affordable=True)
    check("基金回测 NAV 终点有限", np.isfinite(f_res["nav"].iloc[-1]))
    check("基金回测 指标无 NaN",
          all(np.isfinite(v) for v in f_res["metrics"].values()))
    check("基金回测 有调仓记录", len(f_res["trades"]) > 0)

    print("== 引擎回测 ==")
    codes = sorted((set(tech["code"]) & set(panel["code"].unique())))
    codes = [c for c in codes if not c.startswith(("300", "301", "688", "689"))]
    for strat_name, strat in STRATEGIES.items():
        res = run_backtest(panel, codes, strat["factor"], strat["ascending"],
                           "2025-01-02", "2026-08-14", 5000, 3, "monthly", affordable=True)
        m = res["metrics"]
        check(f"回测[{strat_name}] NAV 起点=1 终点有限",
              abs(res["nav"].iloc[0] - 1) < 1e-9 and np.isfinite(res["nav"].iloc[-1]),
              f"end={res['nav'].iloc[-1]}")
        check(f"回测[{strat_name}] 指标无 NaN", all(np.isfinite(v) for v in m.values()))
        check(f"回测[{strat_name}] 调仓记录非空", len(res["trades"]) > 0)
        wsum = float(res["holdings"]["weight"].sum()) if not res["holdings"].empty else 0.0
        check(f"回测[{strat_name}] 持仓权重和∈[0,1]",
              res["holdings"].empty or -1e-6 <= wsum <= 1 + 1e-6,
              f"sum={wsum:.4f}")

    print("== 多因子自由组合 ==")
    close = panel[panel["code"].isin(codes)].pivot_table(
        index="date", columns="code", values="close", aggfunc="last")
    am20 = panel[panel["code"].isin(codes)].pivot_table(
        index="date", columns="code", values="am20", aggfunc="last")
    turn20 = panel[panel["code"].isin(codes)].pivot_table(
        index="date", columns="code", values="turn20", aggfunc="last")
    combo = build_composite_factor(close, am20, turn20, {"am20": 1.0, "mom20": 1.0},
                                   {"am20": False, "mom20": False})
    check("composite 得分矩阵形状", combo.shape == close.shape)
    check("composite 每行非全空", combo.dropna(how="all").shape[0] > 0)
    rank_am = am20.rank(axis=1, pct=True)
    c_high = build_composite_factor(close, am20, turn20, {"am20": 1.0}, {"am20": False})
    c_low = build_composite_factor(close, am20, turn20, {"am20": 1.0}, {"am20": True})
    check("composite 买高=rank", abs(float((c_high - rank_am).abs().max().max())) < 1e-9)
    check("composite 买低=1-rank",
          abs(float((c_low + rank_am - 1.0).abs().max().max())) < 1e-9)
    try:
        build_composite_factor(close, am20, turn20, {"no_such_factor": 1.0})
        check("composite 未知因子抛错", False)
    except ValueError:
        check("composite 未知因子抛错", True)
    res_c = run_backtest(panel, codes, "composite", False,
                         "2025-01-02", "2026-08-14", 5000, 3, "monthly",
                         affordable=True,
                         factor_weights={"am20": 1.0, "mom20": 1.0},
                         factor_directions={"am20": False, "mom20": False})
    check("组合回测 NAV 起点=1 终点有限",
          abs(res_c["nav"].iloc[0] - 1) < 1e-9 and np.isfinite(res_c["nav"].iloc[-1]))
    check("组合回测 指标无 NaN", all(np.isfinite(v) for v in res_c["metrics"].values()))
    check("组合回测 有调仓记录", len(res_c["trades"]) > 0)
    sig_c, _ = latest_signals(panel, codes, "composite", False, top_n=10,
                               factor_weights={"am20": 1.0, "mom20": 1.0},
                               factor_directions={"am20": False, "mom20": False})
    check("组合信号有结果", len(sig_c) > 0)
    check("组合持久化 save/load/delete",
          api_save_composite("__selftest__", {"am20": 1.0}, {"am20": False})
          .get("name") == "__selftest__"
          and "__selftest__" in api_load_composites()
          and api_delete_composite("__selftest__") is True)

    print("== 多空对冲 / 行业中性 / 组合优化 ==")
    res_ls = run_backtest(panel, codes, "mom20", False,
                          "2025-01-02", "2026-08-14", 50000, 3, "monthly",
                          affordable=True, long_short=True, short_n=3,
                          short_cost_rate=0.086)
    check("多空回测 NAV 终点有限", np.isfinite(res_ls["nav"].iloc[-1]))
    check("多空回测 指标无 NaN",
          all(np.isfinite(v) for v in res_ls["metrics"].values()))
    check("多空回测 有调仓记录", len(res_ls["trades"]) > 0)
    check("多空回测 含空头方向",
          len(res_ls["holdings"]) == 0 or "空" in set(res_ls["holdings"]["direction"]))
    sig_ls, _ = latest_signals(panel, codes, "mom20", False, top_n=10,
                               long_short=True, short_n=3)
    check("多空信号含多空两侧", set(sig_ls["side"]) == {"多", "空"})

    ind_map = {str(c): (tech.set_index("code")["industry"].to_dict().get(c, "?") if c in set(tech["code"]) else "?")
               for c in codes}
    res_in = run_backtest(panel, codes, "mom20", False,
                          "2025-01-02", "2026-08-14", 50000, 3, "monthly",
                          affordable=True, industry_map=ind_map,
                          industry_neutral=True)
    check("行业中性回测 NAV 终点有限", np.isfinite(res_in["nav"].iloc[-1]))
    check("行业中性回测 指标无 NaN",
          all(np.isfinite(v) for v in res_in["metrics"].values()))

    from core.portfolio import max_diversification_weights
    _cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.02, 0.003],
                     [0.005, 0.003, 0.01]])
    _w = max_diversification_weights(_cov)
    check("最大分散 权重和=1", abs(_w.sum() - 1) < 1e-6)
    check("最大分散 权重非负", (_w >= 0).all())

    print("== 事件驱动引擎 ==")
    from core.event_engine import EventStrategy, run_event_backtest

    class _WeeklyTop2(EventStrategy):
        """每周一按信号日收盘价最低的 2 只各买一手，其余清仓。"""

        def on_bar(self, ctx, bar):
            if bar.exec_date.weekday() != 0:
                return
            for code in list(ctx.positions):
                ctx.order_target_pct(code, 0.0)
            ranked = sorted(bar.tradable,
                            key=lambda c: bar.close.get(c, 1e9))[:2]
            for c in ranked:
                ctx.order_target_shares(c, 100)

    from core.event_engine import GoldenCrossStrategy

    res = run_event_backtest(panel, codes, GoldenCrossStrategy,
                             "2025-01-02", "2026-08-14", 5000, warmup_days=400)
    check("事件[金叉] NAV 起点=1 终点有限",
          abs(res["nav"].iloc[0] - 1) < 1e-9 and np.isfinite(res["nav"].iloc[-1]))
    check("事件[金叉] 指标无 NaN",
          all(np.isfinite(v) for v in res["metrics"].values()))
    check("事件[金叉] 有调仓记录", len(res["trades"]) > 0)
    # 事件引擎是真实股数+现金撮合，权重和允许 <=1（留有现金）
    check("事件[金叉] 持仓权重和<=1",
          res["holdings"].empty or res["holdings"]["weight"].sum() <= 1 + 1e-6)

    res2 = run_event_backtest(panel, codes, _WeeklyTop2,
                              "2025-01-02", "2026-08-14", 5000, warmup_days=400)
    check("事件[周频Top2] NAV 终点有限", np.isfinite(res2["nav"].iloc[-1]))
    check("事件[周频Top2] 有调仓记录", len(res2["trades"]) > 0)
    check("事件[周频Top2] 持仓权重和<=1",
          res2["holdings"].empty or res2["holdings"]["weight"].sum() <= 1 + 1e-6)

    from core.event_engine import LongShortMomentumStrategy
    res_ls2 = run_event_backtest(panel, codes, LongShortMomentumStrategy,
                                 "2025-01-02", "2026-08-14", 50000,
                                 warmup_days=400, short_rate=0.086)
    check("事件[多空动量] NAV 终点有限", np.isfinite(res_ls2["nav"].iloc[-1]))
    check("事件[多空动量] 有调仓记录", len(res_ls2["trades"]) > 0)

    print("== 撮合增强 / 归因 / 组合优化 ==")
    res3 = run_event_backtest(panel, codes, GoldenCrossStrategy,
                              "2025-01-02", "2026-08-14", 5000,
                              warmup_days=400, slippage_bps=10,
                              max_participation=0.1)
    check("事件[滑点+流动性] NAV 终点有限",
          np.isfinite(res3["nav"].iloc[-1]))

    class _LimitStrategy(EventStrategy):
        """限价单冒烟：限价设为收盘价×1.05（通常可成交）。"""

        def on_bar(self, ctx, bar):
            if bar.exec_date.weekday() != 0:
                return
            for code in list(ctx.positions):
                ctx.order_target_pct(code, 0.0)
            ranked = sorted(bar.tradable, key=lambda c: bar.close.get(c, 1e9))[:1]
            for c in ranked:
                ctx.order_target_shares(c, 100,
                                        limit_price=bar.close[c] * 1.05)

    res4 = run_event_backtest(panel, codes, _LimitStrategy,
                              "2025-01-02", "2026-08-14", 5000,
                              warmup_days=400)
    check("事件[限价单] NAV 终点有限", np.isfinite(res4["nav"].iloc[-1]))

    from core.attribution import brinson_attribution
    ind_map = {str(c).zfill(6): str(i)
               for c, i in zip(tech["code"], tech["industry"])}
    detail, summary = brinson_attribution(panel, codes, res["weight_history"],
                                          res["dates"], ind_map)
    check("归因 detail 非空", len(detail) > 0)
    check("归因 summary 非空", len(summary) > 0)
    check("归因效应合计有限", np.isfinite(summary["total"].sum()))

    from core.portfolio import mean_variance_weights, risk_parity_weights
    np.random.seed(1)
    _rr = np.random.randn(80, 3)
    _w1 = risk_parity_weights(np.cov(_rr, rowvar=False), max_weight=0.5)
    _w2 = mean_variance_weights(_rr, gamma=1.0, max_weight=0.5)
    check("风险平价权重和=1", abs(_w1.sum() - 1) < 1e-6)
    check("均值方差权重和=1", abs(_w2.sum() - 1) < 1e-6)

    print("== 参数稳健性 walk-forward ==")
    from core.walkforward import golden_cross_sweep, walk_forward_factor

    summary, heatmap, windows = golden_cross_sweep(
        panel, codes, "2023-01-03", "2026-08-14", 5000,
        short_list=[5, 10], long_list=[20, 60], n_folds=2, warmup_days=400,
    )
    check("walkforward summary 非空", len(summary) > 0)
    check("walkforward 窗口明细覆盖全部参数组合",
          len(windows) == len(summary) * 2)
    check("walkforward heatmap 有数值",
          len(heatmap) > 0 and bool(heatmap.notna().any().any()))
    wf = walk_forward_factor(panel, codes, "mom20", False,
                             "2023-01-03", "2026-08-14", 5000,
                             top_n=3, n_folds=2, warmup_days=400)
    check("因子策略 walk-forward 正常", len(wf) == 2 and np.isfinite(wf["total"].all()))

    print("== P0/P1：财务因子 / 风险模型 / 滚动训练-测试 ==")
    from core.financial import FINANCIAL_FACTORS, financial_factor_frames
    close_p = panel[panel["code"].isin(codes)].pivot_table(
        index="date", columns="code", values="close", aggfunc="last",
        observed=True)
    cal_p = pd.DatetimeIndex(sorted(panel[panel["code"].isin(codes)]["date"].unique()))
    fframes = financial_factor_frames(codes, cal_p, close_p)
    check("财务因子矩阵不报错",
          isinstance(fframes, dict) and set(fframes) <= set(FINANCIAL_FACTORS))
    fin_res = run_backtest(panel, codes, "roe", False, "2025-01-02", "2026-08-14",
                           50000, 3, "monthly", affordable=True, use_financial=True)
    check("use_financial 回测不报错",
          np.isfinite(fin_res["nav"].iloc[-1]) and len(fin_res["weight_history"]) == len(fin_res["dates"]))

    ind_map = {str(c).zfill(6): str(i)
               for c, i in zip(tech["code"], tech["industry"])}
    rn_res = run_backtest(panel, codes, "mom20", False, "2025-01-02", "2026-08-14",
                          50000, 3, "monthly", affordable=True,
                          industry_map=ind_map, risk_neutral=True)
    ra = rn_res.get("risk_attribution") or {}
    check("risk_neutral 回测有风险归因", len(ra) > 3 and "specific" in ra)
    check("risk_neutral 归因占比合计≈1",
          abs(sum(v for v in ra.values()) - 1.0) < 1e-6)
    check("weight_history 与日期对齐",
          len(rn_res["weight_history"]) == len(rn_res["dates"]))

    from core.walkforward import rolling_train_test_event, rolling_train_test_factor
    rw, rsum, rhist = rolling_train_test_factor(
        panel, codes, "mom20", False, "2025-06-01", "2026-08-14", 5000,
        top_n_list=[3, 5], freq_list=["monthly"], n_folds=2, warmup_days=400)
    check("滚动训练-测试(因子) 有输出",
          len(rw) == 2 and "chosen_top_n" in rw.columns and len(rhist) == 2)
    check("滚动训练-测试 训练窗口数>0",
          int(rsum.iloc[0]["trained_windows"]) >= 1)
    rew, _, _ = rolling_train_test_event(
        panel, codes, "2025-06-01", "2026-08-14", 5000,
        short_list=[3, 5], long_list=[10, 20], n_folds=2, warmup_days=400)
    check("滚动训练-测试(双均线) 有输出",
          len(rew) == 2 and "chosen_short" in rew.columns)

    from core.portfolio import shrink_covariance, weights_from_returns
    np.random.seed(2)
    _mat = np.random.randn(120, 5)
    _cov = shrink_covariance(_mat)
    check("收缩协方差 5x5", _cov.shape == (5, 5))
    w = weights_from_returns(pd.DataFrame(_mat), method="risk_parity",
                             max_weight=0.5)
    check("收缩协方差风险平价权重和=1", abs(sum(w.values()) - 1) < 1e-6)

    print("== 日级模拟盘 ==")
    from core.paper import (account_equity, account_events, account_orders,
                            account_positions, account_trades, account_summary,
                            create_account, delete_account, list_accounts,
                            reset_account, run_paper_trade)
    codes_by_universe = {
        "科技TMT": [c for c in codes],
        "沪深300+中证500+中证1000": [c for c in codes],
    }
    pname = "__selftest_paper__"
    for a in list_accounts():
        if a["name"] == pname:
            delete_account(a["id"])
    # 因子账户现在重放 run_backtest(cash_mode=True)，用短区间控制自测耗时
    pacc = create_account(pname, "动量 20 日", "mom20", False, "科技TMT",
                          50000, 3, "monthly", start_date="2025-01-02")
    check("模拟盘账户创建", pacc["id"] > 0 and pacc["status"] == "active")
    dr = run_paper_trade(panel, codes_by_universe, pacc["id"],
                         exec_date="2025-06-30", dry_run=True)
    check("模拟盘 dry-run 返回目标持仓",
          dr["accounts"][0]["processed"] == "ok"
          and len(dr["accounts"][0].get("targets", {})) > 0)
    r = run_paper_trade(panel, codes_by_universe, pacc["id"],
                        exec_date="2025-06-30")
    ar = r["accounts"][0]
    check("模拟盘执行成功", ar["processed"] == "ok" and ar["rebalanced"])
    r2 = run_paper_trade(panel, codes_by_universe, pacc["id"],
                         exec_date="2025-06-30")
    check("模拟盘幂等（重复执行跳过）", r2["accounts"][0]["processed"] == "already")
    check("模拟盘持仓/快照/订单落库",
          len(account_positions(pacc["id"])) >= 0
          and len(account_equity(pacc["id"])) >= 1
          and len(account_orders(pacc["id"])) >= 1
          and len(account_trades(pacc["id"])) >= 0
          and len(account_events(pacc["id"])) >= 1)
    summ = account_summary(pacc["id"], panel)
    check("模拟盘 summary 含最新快照", bool(summ and summ.get("latest")))
    reset_account(pacc["id"])
    check("模拟盘重置后清空", len(account_equity(pacc["id"])) == 0
          and len(account_orders(pacc["id"])) == 0)
    delete_account(pacc["id"])
    check("模拟盘账户删除", not any(a["name"] == pname for a in list_accounts()))

    print("== 日级模拟盘（事件策略） ==")
    import tempfile
    ev_module = Path(tempfile.gettempdir()) / "__selftest_event_strategy__.py"
    ev_module.write_text('''
from core.event_engine import EventStrategy

class QuickCross(EventStrategy):
    short = 3
    long = 10
    top_n = 2
    max_weight = 0.5

    def on_bar(self, ctx, bar):
        for code in list(ctx.positions):
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            sp = sum(closes[-self.short - 1:-1]) / self.short
            lp = sum(closes[-self.long - 1:-1]) / self.long
            sn = sum(closes[-self.short:]) / self.short
            ln = sum(closes[-self.long:]) / self.long
            if sp >= lp and sn < ln:
                ctx.order_target_pct(code, 0.0)
        held = [c for c, sh in ctx.positions.items() if sh > 0]
        slots = max(0, self.top_n - len(held))
        scores = []
        for code in bar.tradable:
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            sp = sum(closes[-self.short - 1:-1]) / self.short
            lp = sum(closes[-self.long - 1:-1]) / self.long
            sn = sum(closes[-self.short:]) / self.short
            ln = sum(closes[-self.long:]) / self.long
            if sp <= lp and sn > ln:
                scores.append((sn - ln, code))
        scores.sort(reverse=True)
        w = min(self.max_weight, 1.0 / self.top_n)
        for _, code in scores:
            if slots <= 0:
                break
            if ctx.position(code) > 0:
                continue
            ctx.order_target_pct(code, w)
            slots -= 1

EVENT_STRATEGIES = {"QuickCross": QuickCross}
''', encoding="utf-8")
    event_codes = codes[:40]
    ev_codes_by_universe = {
        "科技TMT": [c for c in event_codes],
        "沪深300+中证500+中证1000": [c for c in event_codes],
    }
    ename = "__selftest_paper_event__"
    for a in list_accounts():
        if a["name"] == ename:
            delete_account(a["id"])
    eacc = create_account(ename, "QuickCross", "", False, "科技TMT",
                          50000, 2, "daily", strategy_type="event",
                          module=str(ev_module), event_strategy="QuickCross")
    check("事件账户创建", eacc["id"] > 0 and eacc["strategy_type"] == "event")
    edr = run_paper_trade(panel, ev_codes_by_universe, eacc["id"], dry_run=True)
    check("事件账户 dry-run 成功",
          edr["accounts"][0]["processed"] == "ok")
    er = run_paper_trade(panel, ev_codes_by_universe, eacc["id"])
    check("事件账户执行成功", er["accounts"][0]["processed"] == "ok")
    check("事件账户快照/事件落库",
          len(account_equity(eacc["id"])) >= 1
          and len(account_events(eacc["id"])) >= 1)
    er2 = run_paper_trade(panel, ev_codes_by_universe, eacc["id"])
    check("事件账户幂等（重复执行跳过）",
          er2["accounts"][0]["processed"] == "already")
    delete_account(eacc["id"])
    check("事件账户删除",
          not any(a["name"] == ename for a in list_accounts()))
    ev_module.unlink(missing_ok=True)

    print("== 引擎边界 ==")
    res = run_backtest(panel, codes, "mom20", False, "2021-01-04", "2021-12-31",
                       1000, 3, "weekly", affordable=True)
    check("周频+小资金不报错", res["nav"].notna().all())
    res = run_backtest(panel, codes, "turn20", True, "2026-02-02", "2026-08-14",
                       5000, 3, "monthly", affordable=True)
    check("近半年区间正常", len(res["nav"]) > 100)
    try:
        run_backtest(panel, [], "turn20", True, "2026-02-02", "2026-08-14", 5000, 3, "monthly")
        check("空股票池抛错", False)
    except (ValueError, KeyError):
        check("空股票池抛错", True)

    print("== 与旧 backtest_5w 结果一致性 ==")
    legacy_panel = LEGACY_DATA_DIR / "panel/turn20/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet"
    legacy_tech = LEGACY_DATA_DIR / "panel/tech_universe_sw.csv"
    old_csv = LEGACY_DATA_DIR / "backtest/quant_3stocks/outputs/backtest_5w.csv"
    if legacy_panel.exists() and legacy_tech.exists() and old_csv.exists():
        old = pd.read_csv(old_csv)
        old = old[old["top_n"] == 5].set_index(["window", "strategy"])
        lp = pd.read_parquet(legacy_panel)
        lp["code"] = lp["code"].astype(str).str.zfill(6)
        lt = pd.read_csv(legacy_tech, dtype={"code": str})
        lt["code"] = lt["code"].astype(str).str.zfill(6)
        lcodes = sorted({c for c in set(lt["code"]) & set(lp["code"].unique())
                         if not c.startswith(("300", "301", "688", "689"))})
        old_map = {"低成交冷门": "cold", "高成交领涨": "leader", "动量 20 日": "mom20",
                   "动量 60 日": "mom60", "反转 20 日": "rev20", "低波动": "lowvol",
                   "复合因子": "composite"}
        n_ok = 0
        for wname, (ws, we) in {"长期(2020-2026)": ("2020-01-02", "2026-08-13"),
                                "近半年": ("2026-02-02", "2026-08-13")}.items():
            for strat_name, old_key in old_map.items():
                st = STRATEGIES[strat_name]
                # 旧 backtest_5w 对照走权重模型；cash_mode=True 是新的统一现金口径
                res = run_backtest(lp, lcodes, st["factor"], st["ascending"],
                                   ws, we, 50000, 5, "monthly", affordable=True,
                                   amount_q=0.2, warmup_days=9999,
                                   cash_mode=False)
                m = res["metrics"]
                ref = old.loc[(wname, old_key)]
                close_total = abs(m["总收益"] * 100 - ref["total"]) < 0.01
                close_mdd = abs(m["最大回撤"] * 100 - ref["mdd"]) < 0.01
                close_sharpe = abs(m["夏普"] - ref["sharpe"]) < 0.01
                if close_total and close_mdd and close_sharpe:
                    n_ok += 1
                else:
                    check(f"对照[{wname}/{old_key}]", False,
                          f"total {m['总收益']*100:.2f} vs {ref['total']:.2f}, "
                          f"sharpe {m['夏普']:.3f} vs {ref['sharpe']:.3f}")
        check("旧结果对照 14 组全一致", n_ok == 14, f"ok={n_ok}/14")
    else:
        check("旧结果对照（旧数据缺失，跳过）", True)

    print("== 信号 ==")
    for strat_name, strat in STRATEGIES.items():
        sig, d = latest_signals(panel, codes, strat["factor"], strat["ascending"], top_n=10)
        check(f"信号[{strat_name}] 有结果", len(sig) > 0 and d is not None)

    print("== 记账数学 ==")
    tx = pd.DataFrame([
        {"date": pd.Timestamp("2026-02-02"), "code": "601728", "name": "中国电信",
         "action": "buy", "shares": 800, "price": 6.10, "fee": 4.0, "note": ""},
        {"date": pd.Timestamp("2026-03-02"), "code": "601728", "name": "中国电信",
         "action": "sell", "shares": 300, "price": 6.55, "fee": 2.5, "note": ""},
    ])
    dep = pd.DataFrame([{"date": pd.Timestamp("2026-02-02"), "amount": 5000.0, "note": ""}])
    eq = compute_equity(panel, tx, dep)
    check("记账首日现金=116", abs(eq.iloc[0]["cash"] - 116.0) < 1e-6, str(eq.iloc[0]["cash"]))
    check("记账首日市值=4520", abs(eq.iloc[0]["market_value"] - 4520.0) < 1e-6,
          str(eq.iloc[0]["market_value"]))
    check("记账盈利>0", eq["pnl"].iloc[-1] > 0)
    pos = current_positions(panel, tx)
    check("持仓 500 股", abs(pos.iloc[0]["shares"] - 500) < 1e-6, str(pos.iloc[0]["shares"]))
    check("持仓成本 3050", abs(pos.iloc[0]["cost"] - 3050.0) < 1e-6, str(pos.iloc[0]["cost"]))

    print("== API ==")
    check("health", api("/api/health").get("status") == "ok")
    check("API 免登录可访问", isinstance(api("/api/strategies", use_cookie=False), list))
    info = api("/api/data/panel-info")
    check("panel-info", info["n_codes"] > 500 and bool(info["last_date"]))
    strats = api("/api/strategies")
    check("strategies 数量", len(strats) >= 5, str(len(strats)))
    q = urllib.parse.urlencode({"universe": "科技TMT", "strategy": "低换手冷门", "top_n": 5})
    sig = api("/api/signals?" + q)
    check("signals 有 items", len(sig.get("items", [])) > 0)
    body = {"universe": "科技TMT", "strategy": "动量 20 日", "top_n": 5, "capital": 10000,
            "freq": "monthly", "start": "2025-01-02", "end": "2026-08-14",
            "exclude_kechuang": True, "affordable": True}
    bt = api("/api/backtest", "POST", body)
    check("backtest 指标完整", all(k in bt["metrics"] for k in
                                   ["总收益", "年化收益", "夏普", "最大回撤", "卡玛", "胜率"]))
    check("backtest JSON 无 NaN", json.dumps(bt).find("NaN") < 0)
    bt_risk = api("/api/backtest", "POST", {
        "universe": "科技TMT", "strategy": "动量 20 日", "top_n": 3, "capital": 50000,
        "freq": "monthly", "start": "2025-01-02", "end": "2026-08-14",
        "exclude_kechuang": True, "affordable": True, "risk_neutral": True,
    })
    check("backtest risk_neutral API 返回风险归因",
          "specific" in (bt_risk.get("risk_attribution") or {}))
    check("backtest API 返回 Brinson 归因",
          bool(bt_risk.get("brinson") and bt_risk["brinson"].get("summary")))
    bt_fin = api("/api/backtest", "POST", {
        "universe": "科技TMT", "strategy": "动量 20 日", "top_n": 3, "capital": 50000,
        "freq": "monthly", "start": "2025-01-02", "end": "2026-08-14",
        "exclude_kechuang": True, "affordable": True, "use_financial": True,
    })
    check("backtest use_financial API 不报错",
          "总收益" in bt_fin.get("metrics", {}) and bt_fin.get("use_financial") is True)
    factors = api("/api/factors")
    check("factors 列表非空", len(factors) >= 5, str(len(factors)))
    comp_name = "__selftest_combo__"
    api("/api/composites", "POST", {
        "name": comp_name,
        "weights": {"am20": 1.0, "mom20": 1.0},
        "directions": {"am20": False, "mom20": False},
    })
    comps = api("/api/composites")
    check("composites 保存后可见", any(c["name"] == comp_name for c in comps))
    comp_bt = api("/api/backtest", "POST", {
        "universe": "科技TMT", "strategy": "组合策略", "top_n": 3, "capital": 10000,
        "freq": "monthly", "start": "2025-01-02", "end": "2026-08-14",
        "exclude_kechuang": True, "affordable": True,
        "composite_weights": {"am20": 1.0, "mom20": 1.0},
        "composite_directions": {"am20": False, "mom20": False},
    })
    check("backtest composite 指标完整",
          comp_bt.get("composite") is True and "总收益" in comp_bt.get("metrics", {}))
    comp_sig = api("/api/signals", "POST", {
        "universe": "科技TMT", "strategy": "组合策略", "top_n": 5,
        "composite_weights": {"am20": 1.0, "mom20": 1.0},
        "composite_directions": {"am20": False, "mom20": False},
    })
    check("signals composite 有 items",
          comp_sig.get("composite") is True and len(comp_sig.get("items", [])) > 0)
    q2 = urllib.parse.urlencode({
        "universe": "科技TMT", "strategy": "组合策略", "top_n": 5,
        "composite_weights": json.dumps({"am20": 1.0}),
        "composite_directions": json.dumps({"am20": False}),
    })
    comp_sig_get = api("/api/signals?" + q2)
    check("signals GET composite 有 items", len(comp_sig_get.get("items", [])) > 0)
    api("/api/composites/" + comp_name, "DELETE")
    comps2 = api("/api/composites")
    check("composites 删除生效", not any(c["name"] == comp_name for c in comps2))
    ls_bt = api("/api/backtest", "POST", {
        "universe": "科技TMT", "strategy": "多空动量 20 日", "top_n": 3, "capital": 50000,
        "freq": "monthly", "start": "2025-01-02", "end": "2026-08-14",
        "exclude_kechuang": True, "affordable": True,
    })
    check("backtest 多空策略 long_short=True",
          ls_bt.get("long_short") is True and "总收益" in ls_bt.get("metrics", {}))
    ls_sig = api("/api/signals", "POST", {
        "universe": "科技TMT", "strategy": "多空动量 20 日", "top_n": 10,
    })
    check("signals 多空含多空两侧",
          set(s.get("side") for s in ls_sig.get("items", [])) == {"多", "空"})
    sweep_body = {"mode": "event", "start": "2023-01-03", "end": "2026-08-14",
                  "capital": 5000, "folds": 2, "short_list": [3, 5],
                  "long_list": [10, 20], "top_n": 3}
    sw = api("/api/sweep", "POST", sweep_body)
    check("sweep API 汇总非空", len(sw.get("summary", [])) > 0)
    check("sweep API windows 非空", len(sw.get("windows", [])) > 0)
    sw_roll = api("/api/sweep", "POST", {
        "mode": "rolling", "strategy": "动量 20 日", "start": "2025-06-01",
        "end": "2026-08-14", "capital": 5000, "folds": 2, "top_n": 3,
        "top_n_list": [3, 5], "freq_list": ["monthly"],
    })
    check("sweep rolling API 有窗口与参数历史",
          len(sw_roll.get("windows", [])) == 2
          and len(sw_roll.get("param_history", [])) == 2)
    paper_api_name = "__selftest_paper_api__"
    for a in api("/api/paper/accounts"):
        if a["name"] == paper_api_name:
            api(f"/api/paper/accounts/{a['id']}", "DELETE")
    pacc = api("/api/paper/accounts", "POST", {
        "name": paper_api_name, "strategy_name": "动量 20 日", "capital": 50000,
        "top_n": 3, "freq": "monthly", "start_date": "2025-01-02",
    })
    check("paper API 创建账户", pacc.get("id") and not pacc.get("error"))
    pr = api("/api/paper/run", "POST", {
        "account_id": pacc["id"], "exec_date": "2025-06-30",
    })
    check("paper API 执行成功",
          pr.get("accounts") and pr["accounts"][0]["processed"] == "ok")
    ps = api(f"/api/paper/accounts/{pacc['id']}/summary")
    check("paper API summary 含最新快照", bool(ps.get("latest")))
    peq = api(f"/api/paper/accounts/{pacc['id']}/equity")
    check("paper API equity 快照非空", len(peq.get("items", [])) >= 1)
    api(f"/api/paper/accounts/{pacc['id']}", "DELETE")

    ev_api_mod = Path("labs") / "__selftest_paper_event_api__.py"
    ev_api_mod.write_text('''
from core.event_engine import EventStrategy

class ApiQuick(EventStrategy):
    short = 3
    long = 10
    top_n = 1
    max_weight = 0.5

    def on_bar(self, ctx, bar):
        for code in list(ctx.positions):
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            sp = sum(closes[-self.short - 1:-1]) / self.short
            lp = sum(closes[-self.long - 1:-1]) / self.long
            sn = sum(closes[-self.short:]) / self.short
            ln = sum(closes[-self.long:]) / self.long
            if sp >= lp and sn < ln:
                ctx.order_target_pct(code, 0.0)
        held = [c for c, sh in ctx.positions.items() if sh > 0]
        slots = max(0, self.top_n - len(held))
        scores = []
        for code in bar.tradable:
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            sp = sum(closes[-self.short - 1:-1]) / self.short
            lp = sum(closes[-self.long - 1:-1]) / self.long
            sn = sum(closes[-self.short:]) / self.short
            ln = sum(closes[-self.long:]) / self.long
            if sp <= lp and sn > ln:
                scores.append((sn - ln, code))
        scores.sort(reverse=True)
        w = min(self.max_weight, 1.0 / self.top_n)
        for _, code in scores:
            if slots <= 0:
                break
            if ctx.position(code) > 0:
                continue
            ctx.order_target_pct(code, w)
            slots -= 1

EVENT_STRATEGIES = {"ApiQuick": ApiQuick}
''', encoding="utf-8")
    ev_items = api("/api/paper/event-strategies")
    check("paper API 事件策略列表",
          any(m["name"] == ev_api_mod.stem for m in ev_items.get("items", [])))
    ev_acc = api("/api/paper/accounts", "POST", {
        "name": "__selftest_paper_event_api__", "strategy_type": "event",
        "strategy_name": "ApiQuick", "module": str(ev_api_mod),
        "event_strategy": "ApiQuick", "capital": 50000, "top_n": 1,
        "freq": "daily",
    })
    check("paper API 创建事件账户",
          ev_acc.get("id") and ev_acc.get("strategy_type") == "event")
    ev_run = api("/api/paper/run", "POST", {"account_id": ev_acc["id"]})
    check("paper API 事件执行成功",
          ev_run.get("accounts") and ev_run["accounts"][0]["processed"] == "ok")
    ev_trades = api(f"/api/paper/accounts/{ev_acc['id']}/trades")
    check("paper API 事件成交可读", isinstance(ev_trades, list))
    api(f"/api/paper/accounts/{ev_acc['id']}", "DELETE")
    ev_api_mod.unlink(missing_ok=True)

    eq_api = api("/api/ledger/equity")
    check("ledger equity 可序列化", "items" in eq_api and "summary" in eq_api)
    pos_api = api("/api/ledger/positions")
    check("ledger positions 可序列化", isinstance(pos_api, list))
    st = api("/api/data/status")
    check("data status 含 meta", "meta" in st and "panel" in st)

    print("== 前端资源 ==")
    for path, name in [("/", "index.html"), ("/vendor/vue.global.prod.js", "vue"),
                       ("/vendor/echarts.min.js", "echarts")]:
        with urllib.request.urlopen(BASE + path, timeout=10) as resp:
            check(f"前端 {name} 200", resp.status == 200, str(resp.status))

    print(f"\n结果: {PASS} 通过, {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

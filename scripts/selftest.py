#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quant_ui 自测：引擎 + API + 记账 + 数据。"""
from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import load_panel, load_universe, load_tech, load_index  # noqa: E402
from core.engine import latest_signals, run_backtest  # noqa: E402
from core.ledger import compute_equity, current_positions  # noqa: E402
from strategies.registry import STRATEGIES  # noqa: E402


BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0

# 带 cookie 的 opener：登录后 API 调用共享会话
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
        check(f"回测[{strat_name}] 持仓权重和≈1", abs(res["holdings"]["weight"].sum() - 1) < 1e-6
              or res["holdings"].empty)

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
    legacy_panel = Path("/tmp/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet")
    legacy_tech = Path("/tmp/tech_universe_sw.csv")
    old_csv = Path("/home/ubuntu/quant_3stocks/outputs/backtest_5w.csv")
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
                res = run_backtest(lp, lcodes, st["factor"], st["ascending"],
                                   ws, we, 50000, 5, "monthly", affordable=True,
                                   amount_q=0.2, warmup_days=9999)
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
    # 登录鉴权
    try:
        api("/api/strategies", use_cookie=False)
        check("未登录 API 返回 401", False)
    except urllib.error.HTTPError as e:
        check("未登录 API 返回 401", e.code == 401, str(e.code))
    try:
        api("/api/auth/login", "POST", {"username": "root", "password": "wrong"})
        check("错误密码返回 401", False)
    except urllib.error.HTTPError as e:
        check("错误密码返回 401", e.code == 401, str(e.code))
    pw = os.environ.get("QUANT_UI_PASSWORD", "ZBW207060")
    me = api("/api/auth/login", "POST", {"username": "root", "password": pw})
    check("登录成功", me.get("username") == "root", str(me))
    me = api("/api/auth/me")
    check("me 返回 root", me.get("username") == "root", str(me))
    info = api("/api/data/panel-info")
    check("panel-info", info["n_codes"] > 500 and bool(info["last_date"]))
    strats = api("/api/strategies")
    check("strategies 数量", len(strats) >= 5, str(len(strats)))
    q = urllib.parse.urlencode({"universe": "科技行业", "strategy": "低换手冷门", "top_n": 5})
    sig = api("/api/signals?" + q)
    check("signals 有 items", len(sig.get("items", [])) > 0)
    body = {"universe": "科技行业", "strategy": "动量 20 日", "top_n": 5, "capital": 10000,
            "freq": "monthly", "start": "2025-01-02", "end": "2026-08-14",
            "exclude_kechuang": True, "affordable": True}
    bt = api("/api/backtest", "POST", body)
    check("backtest 指标完整", all(k in bt["metrics"] for k in
                                   ["总收益", "年化收益", "夏普", "最大回撤", "卡玛", "胜率"]))
    check("backtest JSON 无 NaN", json.dumps(bt).find("NaN") < 0)
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

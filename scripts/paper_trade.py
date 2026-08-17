#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日级模拟盘 CLI：账户管理 / 手动执行 / 明细查询。

用法示例：
  python scripts/paper_trade.py --create --name 模拟盘A --strategy "动量 20 日" \
      --capital 100000 --top-n 3 --freq monthly
  python scripts/paper_trade.py --list
  python scripts/paper_trade.py --run                 # 所有启用账户，最新交易日
  python scripts/paper_trade.py --run --account 1 --dry-run
  python scripts/paper_trade.py --status --account 1
  python scripts/paper_trade.py --orders --account 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import (load_etf, load_etf_panel, load_panel, load_tech,  # noqa: E402
                       load_universe)
from core.paper import (  # noqa: E402
    account_events, account_equity, account_orders, account_positions,
    account_trades, create_account, delete_account, enrich_positions,
    list_accounts, reset_account, run_paper_trade,
)
from core.strategy_pool import resolve_strategy  # noqa: E402
from strategies.registry import STRATEGIES  # noqa: E402


def build_codes_by_universe(exclude_kechuang: bool = True) -> dict[str, list[str]]:
    panel = load_panel()
    tech = load_tech()
    uni = load_universe()
    etf = load_etf()
    etf_panel = load_etf_panel()
    out = {}
    for name, pool in (("科技TMT", tech), ("沪深300+中证500+中证1000", uni)):
        codes = set(pool["code"]) & set(panel["code"].unique())
        if exclude_kechuang:
            codes = {c for c in codes if not c.startswith(("300", "301", "688", "689"))}
        out[name] = sorted(codes)
    if len(etf) and len(etf_panel):
        etf_codes = sorted(set(etf["code"]) & set(etf_panel["code"].unique()))
        # 模拟盘同样过滤上市不足 60 个交易日的次新 ETF，避免上市初期炒作
        if len(etf_panel):
            cnt = etf_panel.groupby("code", observed=True).size()
            keep = cnt[cnt >= 60].index
            etf_codes = [c for c in etf_codes if c in set(keep)]
        out["ETF"] = etf_codes
    return out


def pprint(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser(description="日级模拟盘")
    ap.add_argument("--create", action="store_true", help="创建账户")
    ap.add_argument("--name", help="账户名称")
    ap.add_argument("--strategy", default="动量 20 日", help="策略名")
    ap.add_argument("--strategy-type", default="factor", dest="strategy_type",
                    choices=["factor", "event"], help="策略类型")
    ap.add_argument("--module", help="事件策略代码模块路径")
    ap.add_argument("--event-strategy", dest="event_strategy", help="事件策略名")
    ap.add_argument("--start-date", dest="start_date",
                    help="事件账户模拟起始日（默认首次执行日）")
    ap.add_argument("--universe", default="科技TMT", help="股票池")
    ap.add_argument("--capital", type=float, default=100000.0, help="初始资金")
    ap.add_argument("--top-n", type=int, default=3, dest="top_n", help="TopN")
    ap.add_argument("--freq", default="monthly", help="daily/weekly/monthly")
    ap.add_argument("--adx-filter", type=float, default=0,
                    help="ADX 趋势强度过滤阈值（信号日 ADX>=阈值才可买，0=不过滤）")
    ap.add_argument("--max-weight", type=float, default=0.5, dest="max_weight")
    ap.add_argument("--amount-q", type=float, default=0.2, dest="amount_q")
    ap.add_argument("--list", action="store_true", help="列出账户")
    ap.add_argument("--run", action="store_true", help="执行模拟盘")
    ap.add_argument("--account", type=int, help="账户 ID")
    ap.add_argument("--date", help="执行日 YYYY-MM-DD（默认最新交易日）")
    ap.add_argument("--dry-run", action="store_true", help="只预览目标持仓，不落库")
    ap.add_argument("--status", action="store_true", help="账户持仓/权益")
    ap.add_argument("--orders", action="store_true", help="订单明细")
    ap.add_argument("--trades", action="store_true", help="成交明细")
    ap.add_argument("--events", action="store_true", help="事件日志")
    ap.add_argument("--reset", action="store_true", help="重置账户")
    ap.add_argument("--delete", action="store_true", help="删除账户")
    args = ap.parse_args()

    if args.create:
        if args.strategy_type == "event":
            if not args.module or not args.event_strategy:
                print("事件账户需要 --module 与 --event-strategy", file=sys.stderr)
                return 1
            mp = Path(args.module)
            if not mp.exists():
                print(f"代码模块不存在: {args.module}", file=sys.stderr)
                return 1
            from core.paper import _load_module
            try:
                mod = _load_module(str(mp))
            except Exception as exc:
                print(f"加载代码模块失败: {exc}", file=sys.stderr)
                return 1
            ev = getattr(mod, "EVENT_STRATEGIES", None) or {}
            if args.event_strategy not in ev:
                print(f"模块中没有事件策略: {args.event_strategy}（可用: "
                      f"{', '.join(ev.keys()) or '空'}）", file=sys.stderr)
                return 1
            acc = create_account(
                name=args.name or input("账户名称: ").strip(),
                strategy_name=args.event_strategy, factor="", ascending=False,
                universe=args.universe, capital=args.capital,
                top_n=args.top_n, freq=args.freq,
                risk_config={"max_weight": args.max_weight, "amount_q": args.amount_q},
                strategy_type="event", module=str(mp),
                event_strategy=args.event_strategy, start_date=args.start_date,
            )
        else:
            try:
                s = STRATEGIES[args.strategy]
            except KeyError:
                try:
                    s = resolve_strategy(args.strategy)
                except KeyError:
                    print(f"未知策略: {args.strategy}（不在注册表/配置池）",
                          file=sys.stderr)
                    return 1
            risk_cfg = {"max_weight": args.max_weight, "amount_q": args.amount_q,
                        "adx_filter": args.adx_filter or None}
            for k in ("adx_filter", "chandelier_mult", "chandelier_period",
                      "regime_adx", "regime_scale"):
                if s.get(k) not in (None, ""):
                    risk_cfg[k] = (int(s[k]) if k == "chandelier_period"
                                   else float(s[k]))
            acc = create_account(
                name=args.name or input("账户名称: ").strip(),
                strategy_name=args.strategy, factor=s["factor"],
                ascending=s["ascending"], universe=args.universe,
                capital=args.capital, top_n=args.top_n, freq=args.freq,
                risk_config=risk_cfg,
            )
        pprint(acc)
        return 0

    if args.list:
        pprint(list_accounts())
        return 0

    if args.status:
        if args.account is None:
            print("--status 需要 --account", file=sys.stderr)
            return 1
        panel = load_panel()
        acc = next((a for a in list_accounts() if a["id"] == args.account), None)
        if acc and acc.get("universe") == "ETF":
            panel = load_etf_panel()
        from core.paper import account_summary
        pprint(account_summary(args.account, panel))
        return 0

    if args.orders or args.trades or args.events:
        if args.account is None:
            print("查询明细需要 --account", file=sys.stderr)
            return 1
        if args.orders:
            pprint(account_orders(args.account))
        if args.trades:
            pprint(account_trades(args.account))
        if args.events:
            pprint(account_events(args.account))
        return 0

    if args.reset:
        if args.account is None:
            print("--reset 需要 --account", file=sys.stderr)
            return 1
        reset_account(args.account)
        print("已重置")
        return 0

    if args.delete:
        if args.account is None:
            print("--delete 需要 --account", file=sys.stderr)
            return 1
        print("已删除" if delete_account(args.account) else "账户不存在")
        return 0

    if args.run:
        panel = load_panel()
        accounts = list_accounts()
        if args.account is not None:
            accounts = [a for a in accounts if a["id"] == args.account]
        stock_ids = [a["id"] for a in accounts if a.get("universe") != "ETF"]
        etf_ids = [a["id"] for a in accounts if a.get("universe") == "ETF"]
        out_accounts = []
        run_date = args.date
        if stock_ids:
            r = run_paper_trade(
                panel, build_codes_by_universe(),
                account_ids=stock_ids, exec_date=args.date, dry_run=args.dry_run,
            )
            run_date = r.get("run_date")
            out_accounts += r.get("accounts", [])
        if etf_ids:
            etf_panel = load_etf_panel()
            etf_codes = {"ETF": sorted(set(load_etf()["code"])
                                       & set(etf_panel["code"].unique()))}
            r = run_paper_trade(
                etf_panel, etf_codes,
                account_ids=etf_ids, exec_date=args.date, dry_run=args.dry_run,
            )
            run_date = r.get("run_date")
            out_accounts += r.get("accounts", [])
        if not out_accounts:
            print("未找到可执行账户", file=sys.stderr)
            return 1
        pprint({"run_date": run_date, "accounts": out_accounts})
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

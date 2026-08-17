#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0 绩效报告：QuantStats HTML + IC/分组分析，直接消费现有回测引擎。

用法：
    python scripts/performance_report.py --strategy "低换手冷门" \
        --start 2023-01-03 --end 2026-08-14 --top-n 3 --capital 5000
    python scripts/performance_report.py --all  # 全策略跑一遍

输出（默认 results/performance/）：
    {strategy}_quantstats.html   QuantStats 绩效报告
    {strategy}_report.md         指标 + IC + 分组 Markdown
    {strategy}_factor_ic.csv     逐日 Spearman IC
    {strategy}_group.csv         分组收益明细
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data import load_index, load_panel, load_tech, load_universe  # noqa: E402
from core.store import normalize_universe  # noqa: E402
from core.engine import run_backtest  # noqa: E402
from core.performance import build_md_report, quantstats_html  # noqa: E402
from strategies.registry import STRATEGIES, get_strategy  # noqa: E402


def slug(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "_", name).strip("_") or "strategy"


def build_codes(universe: str, exclude_kechuang: bool) -> list[str]:
    panel = load_panel()
    uni = load_universe()
    tech = load_tech()
    if normalize_universe(universe) == "科技TMT":
        codes = set(tech["code"])
    else:
        codes = set(uni["code"])
    codes &= set(panel["code"].unique())
    if exclude_kechuang:
        codes = {c for c in codes if not c.startswith(("300", "301", "688", "689"))}
    return sorted(codes)


def run_one(name: str, out_dir: Path, universe: str, start: str, end: str,
            top_n: int, capital: float, freq: str, exclude_kechuang: bool) -> dict:
    panel = load_panel()
    codes = build_codes(universe, exclude_kechuang)
    strat = get_strategy(name)
    res = run_backtest(
        panel=panel, codes=codes, factor=strat["factor"],
        ascending=strat["ascending"], start=start, end=end,
        capital=capital, top_n=top_n, freq=freq,
        affordable=True, amount_q=0.2, warmup_days=400,
        industry_map=None, industry_cap=strat.get("industry_cap"),
        analyze=True,
    )
    q = res.get("factor_quality")
    nav, bench = res["nav"], res["bench"]

    tag = slug(name)
    html_path = quantstats_html(nav, bench, title=f"{name} · {start}~{end}",
                                out_path=out_dir / f"{tag}_quantstats.html")
    md = build_md_report(name, res["metrics"], res["bench_metrics"], q, nav, bench,
                         ascending=strat["ascending"])
    (out_dir / f"{tag}_report.md").write_text(md, encoding="utf-8")
    if q is not None:
        q["ic_series"].rename("ic").to_frame().to_csv(out_dir / f"{tag}_factor_ic.csv",
                                                      encoding="utf-8-sig")
        q["group_table"].to_csv(out_dir / f"{tag}_group.csv",
                                index=False, encoding="utf-8-sig")
        sign = -1.0 if strat["ascending"] else 1.0
        q_adj = {
            "ic": q["ic"]["mean_ic"] * sign if q["ic"]["mean_ic"] is not None else None,
            "spread": q["group"]["spread"] * sign if q["group"]["spread"] is not None else None,
        }
    else:
        q_adj = None
    return {
        "name": name, "factor": strat["factor"], "ascending": strat["ascending"],
        "metrics": res["metrics"], "q": q, "q_adj": q_adj, "html": html_path,
        "md": out_dir / f"{tag}_report.md",
    }


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="绩效报告 + IC/分组分析")
    ap.add_argument("--strategy", default=None, help="策略名；不传则用默认策略")
    ap.add_argument("--all", action="store_true", help="全部策略跑一遍")
    ap.add_argument("--universe", default="科技TMT", choices=["科技TMT", "沪深300+中证500+中证1000"])
    ap.add_argument("--start", default="2023-01-03")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--freq", default="monthly", choices=["monthly", "weekly"])
    ap.add_argument("--keep-kechuang", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "performance"))
    args = ap.parse_args()

    names = list(STRATEGIES) if args.all else [args.strategy or "低换手冷门"]
    if not args.all and names[0] not in STRATEGIES:
        print(f"未知策略: {names[0]}；可选: {', '.join(STRATEGIES)}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        r = run_one(name, out_dir, args.universe, args.start, args.end,
                    args.top_n, args.capital, args.freq,
                    exclude_kechuang=not args.keep_kechuang)
        m, q, q_adj = r["metrics"], r["q"], r["q_adj"]
        rows.append({
            "策略": name, "因子": r["factor"],
            "总收益%": fmt(m["总收益"] * 100),
            "夏普": fmt(m["夏普"]),
            "最大回撤%": fmt(m["最大回撤"] * 100),
            "IC均值": fmt(q["ic"]["mean_ic"]) if q else "-",
            "ICIR": fmt(q["ic"]["icir"]) if q else "-",
            "多空价差%": fmt(q["group"]["spread"] * 100) if q and q["group"]["spread"] is not None else "-",
            "方向调整IC": fmt(q_adj["ic"]) if q_adj else "-",
            "方向调整价差%": fmt(q_adj["spread"] * 100) if q_adj and q_adj["spread"] is not None else "-",
            "HTML": str(r["html"]) if r["html"] else "未生成",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    (out_dir / "summary.csv").write_text(df.to_csv(index=False, encoding="utf-8-sig"),
                                         encoding="utf-8-sig")
    print(f"\nsaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

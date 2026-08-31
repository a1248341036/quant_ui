# -*- coding: utf-8 -*-
"""与聚宽回测明细逐日对拍: result_1.csv (2021-01-04 ~ 2026-08-21, 10万资金)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import jq_data  # noqa: E402
from run_smallcap import build_cand_fn  # noqa: E402
from template import ReproConfig, ma_num_fn, run  # noqa: E402

JQ_CSV = Path(r"C:\Users\zhoubw\Downloads\result_1.csv")


def main() -> None:
    jq = pd.read_csv(JQ_CSV, encoding="gbk", parse_dates=["时间"])
    jq["date"] = jq["时间"].dt.normalize()
    jq_nav = (1 + jq["策略收益"] / 100).set_axis(jq["date"])
    jq_bench = (1 + jq["基准收益"] / 100).set_axis(jq["date"])
    start, end = str(jq["date"].iloc[0].date()), str(jq["date"].iloc[-1].date())

    # 同区间 + 10万资金重跑
    engine_panel, meta, close_raw_df = jq_data.load_panel(start, end)
    tables = jq_data.build_tables(engine_panel, meta, close_raw_df)
    inc = jq_data.load_income()
    fin_ok = jq_data.fin_ok_matrix(inc, tables.dates, tables.codes,
                                   jq_data.triple_positive_pred())
    cand_fn = build_cand_fn(tables, fin_ok)
    level, ma = jq_data.ew_index(tables)
    num_fn = ma_num_fn(level, ma)
    cfg = ReproConfig(start=start, end=end, capital=100_000.0,
                      out_dir=HERE / "out_vs_jq")
    out = run(cfg, cand_fn, num_fn, tables=tables, panel=engine_panel)
    my_nav = out["res"]["nav"].rename("mine")

    # 对齐
    cmp = pd.DataFrame({"jq": jq_nav, "mine": my_nav}).dropna()
    cmp["jq_ret"] = cmp["jq"].pct_change()
    cmp["my_ret"] = cmp["mine"].pct_change()
    corr = cmp[["jq_ret", "my_ret"]].corr().iloc[0, 1]

    print(f"窗口: {start} ~ {end}  对齐交易日: {len(cmp)}")
    print(f"总收益: JQ {cmp['jq'].iloc[-1]-1:+.1%}  mine {cmp['mine'].iloc[-1]-1:+.1%}")
    print(f"终值比: {cmp['mine'].iloc[-1]/cmp['jq'].iloc[-1]:.3f}")
    print(f"日收益相关系数: {corr:.3f}")
    for name, col in (("JQ", "jq"), ("mine", "mine")):
        dd = cmp[col] / cmp[col].cummax() - 1
        print(f"MDD {name}: {dd.min():+.2%} @ {dd.idxmin().date()}")

    cmp["y"] = cmp.index.year
    print("\n分年度对比:")
    print(f"{'年':<6}{'JQ':>9}{'mine':>9}{'差':>9}")
    for y, g in cmp.groupby("y"):
        rj = g["jq"].iloc[-1] / (cmp.loc[:g.index[0], "jq"].iloc[-2]
                                 if g.index[0] != cmp.index[0] else 1.0) - 1
        rm = g["mine"].iloc[-1] / (cmp.loc[:g.index[0], "mine"].iloc[-2]
                                   if g.index[0] != cmp.index[0] else 1.0) - 1
        print(f"{y:<6}{rj:>9.1%}{rm:>9.1%}{rm-rj:>9.1%}")

    # 月度差异 top
    m_jq = cmp["jq"].resample("ME").last().pct_change()
    m_my = cmp["mine"].resample("ME").last().pct_change()
    mdiff = (m_my - m_jq).dropna().sort_values()
    print("\n月度差异最极端 6 个月 (mine - JQ):")
    for d, v in list(mdiff.head(3).items()) + list(mdiff.tail(3).items()):
        print(f"  {d.date()}  {v:+.1%}   (jq {m_jq[d]:+.1%} / my {m_my[d]:+.1%})")

    # 保存对齐净值
    cmp[["jq", "mine"]].to_csv(HERE / "out_vs_jq" / "overlay.csv",
                               encoding="utf-8-sig")


if __name__ == "__main__":
    main()

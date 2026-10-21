# -*- coding: utf-8 -*-
"""执行摩擦瀑布分解：周频 ML 组合的超额是被哪一层杀死的（2026-10-20）。

**背景**：同一份 purged 分数，简化口径周频 +21.8%/年，引擎口径 −0.56%/年。
差距 ~22pp 的构成不明。本脚本在统一框架内逐层加摩擦：

    A  close-to-close，T-1 信号 → T 收盘买，10bp            （原简化口径，复现）
    A' A + 真实成本（卖出 12.5bp）                            （尾盘竞价可执行 + 真实费率）
    B  open-to-open，周五信号 → 下周一开盘买，无成本          （隔离「隔夜跳空+时点」效应）
    C  B + 12.5bp 卖出 / 2.5bp 买入                           （B + 引擎费率）
    D  引擎口径（已有结果 −0.56%，含整手/流动性/涨跌停）       （全摩擦）

    A→A' 隔离费率；A'→B 隔离成交时点/跳空；C→D 隔离整手/流动性/涨跌停。

**数据口径**：purged walk-forward 分数，盲测纪律不触碰（全程 ≤2026-09-30）。
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS = 244.0


def _metrics(nav: pd.Series, ppy: float) -> dict:
    if len(nav) < 4:
        return {}
    ret = nav.pct_change().dropna()
    n_per = max(1, len(ret))
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    ann = float((1 + total) ** (ppy / n_per) - 1) if total > -1 else float("nan")
    vol = float(ret.std() * np.sqrt(ppy))
    return {"annual_return": ann, "sharpe": float(ann / vol) if vol > 0 else float("nan"),
            "max_drawdown": float(((nav - nav.cummax()) / nav.cummax()).min()),
            "total_return": total}


def _run(score_wide, buy_px: pd.DataFrame, ok_wide,
         sig_days, buy_days, sell_days, top_n: int, buy_cost: float, sell_cost: float) -> pd.Series:
    """sig_days[k] 的分数 → buy_days[k] 成交 → sell_days[k] 平仓。

    对齐纪律（2026-10-20 修正：首版买入日比信号日晚一整个调仓期，信号多衰减 5 天，
    凭空损失约 7pp/年）：
      A/A' 尾盘竞价：sig=buy=T 收盘（T 收盘分 T 收盘价买，卖 T+5 收盘）
      B/C  开盘口径：sig=T 收盘（周五）、buy=T+1 开盘、sell=T+6 开盘
    """
    prev_w, nav, dates = None, [1.0], [buy_days[0]]
    n = min(len(sig_days), len(buy_days), len(sell_days))
    for k in range(n - 1):
        d_sig, d_buy, d_sell = sig_days[k], buy_days[k], sell_days[k + 1]
        if d_sig not in score_wide.index or d_buy not in buy_px.index or d_sell not in buy_px.index:
            continue
        s = score_wide.loc[d_sig].dropna()
        ok = ok_wide.loc[d_sig].reindex(s.index).fillna(False).astype(bool)
        pb = buy_px.loc[d_buy].reindex(s.index)
        valid = ok & pb.notna() & np.isfinite(s) & (pb > 0)
        s, pb = s[valid], pb[valid]
        if len(s) < top_n:
            continue
        sel = s.nlargest(top_n).index
        w = pd.Series(1.0 / len(sel), index=sel)
        ps = buy_px.loc[d_sell].reindex(sel)  # 同价表：买入/卖出同口径（close 或 open）
        ret = (ps / pb.reindex(sel) - 1).fillna(0.0)
        turn = 0.5 * float((w.sub(prev_w.reindex(w.index).fillna(0.0), fill_value=0.0)).abs().sum()) \
            if prev_w is not None else 0.5
        prev_w = w
        net_ret = float((w * ret).sum()) - turn * (buy_cost + sell_cost)
        nav.append(nav[-1] * (1 + net_ret))
        dates.append(d_buy)
    return pd.Series(nav, index=pd.to_datetime(dates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache", default=str(
        ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment" / "full_period_scores_purged.parquet"))
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    sdf = pd.read_parquet(args.score_cache)
    dcol = "date" if "date" in sdf.columns else "datetime"
    sdf[dcol] = pd.to_datetime(sdf[dcol])
    score_wide = sdf.set_index([dcol, "instrument"])["score"].sort_index().unstack("instrument")

    panel = load_panel_from_cne(start="2020-11-01", end=args.end,
                                include_fundamentals=False,
                                keep_columns=("close", "open", "adj_close",
                                              "is_trade", "not_st"))
    close_wide = panel["close"].astype("float64").unstack("instrument").ffill()
    open_wide = panel["open"].astype("float64").unstack("instrument").ffill()
    adj_wide = panel["adj_close"].astype("float64").unstack("instrument").ffill()
    # ⚠ 停牌股在卖出日 reindex 得 NaN——必须 ffill 延续前价（视为无法成交、继续持有），
    # 否则 fillna(-1) 会把停牌记成 -100%（2026-10-20 首跑 bug，费率层凭空掉 21pp）
    ok_wide = (panel["is_trade"].fillna(False).astype(bool)
               & panel["not_st"].fillna(False).astype(bool)).unstack("instrument")
    common = score_wide.index.unique().intersection(close_wide.index)
    day_list = np.array(sorted(common))
    ppy = TRADING_DAYS / 5.0

    results = {}

    # ── A / A'：尾盘竞价口径——T 收盘分 T 收盘买（sig=buy=sell 基准），持有 5 日 ──
    reb = day_list[::5]
    for tag, sell_cost in (("A_cc_10bp", 0.0010), ("Ap_cc_12.5bp", 0.00125)):
        nav = _run(score_wide, adj_wide, ok_wide, reb, reb, reb, args.top_n, 0.00025, sell_cost)
        m = _metrics(nav, ppy)
        results[tag] = m
        print(f"[ waterfall] {tag:16s}: 年化 {m['annual_return']:+.2%}  "
              f"夏普 {m['sharpe']:.2f}  回撤 {m['max_drawdown']:.2%}", flush=True)

    # ── B / C：开盘口径——周五收盘分 → 下周一开盘买 → 再下周五开盘卖 ────────
    fri = np.array([d for d in day_list if pd.Timestamp(d).weekday() == 4])
    for tag, buy_cost, sell_cost in (("B_oo_nocost", 0.0, 0.0),
                                     ("C_oo_fees", 0.00025, 0.00125)):
        nav = _run(score_wide, open_wide, ok_wide, fri, fri, fri, args.top_n, buy_cost, sell_cost)
        m = _metrics(nav, ppy)
        results[tag] = m
        print(f"[ waterfall] {tag:16s}: 年化 {m['annual_return']:+.2%}  "
              f"夏普 {m['sharpe']:.2f}  回撤 {m['max_drawdown']:.2%}", flush=True)

    # 换手与成本诊断
    sel_sets, prev = [], None
    for k in range(len(reb) - 1):
        d = reb[k]
        if d not in score_wide.index or d not in adj_wide.index:
            continue
        s = score_wide.loc[d].dropna()
        o = ok_wide.loc[d].reindex(s.index).fillna(False).astype(bool)
        p = adj_wide.loc[d].reindex(s.index)
        v = o & p.notna() & np.isfinite(s)
        sel_sets.append(set(s[v].nlargest(args.top_n).index))
    ovs = [len(a & b) / args.top_n for a, b in zip(sel_sets[1:], sel_sets[:-1])]
    ann_turn = float(np.mean([1 - x for x in ovs]) * (TRADING_DAYS / 5))
    cost_real = ann_turn * 0.0015
    print(f"\n[ waterfall] 换手诊断: 平均单期换手 {1-np.mean(ovs):.2%} → 年单边 {ann_turn:.1f} 倍 → "
          f"真实费率成本 ≈ {cost_real:.1%}/年", flush=True)

    print("""
[ waterfall] ===== 分解（每层吃掉多少）=====
  A→A'  真实费率（卖出 10bp→12.5bp）
  A'→B  成交时点：T 收盘买（尾盘竞价可执行）→ T+1 开盘买（隔夜跳空税）
  B→C   费率叠回
  C→D   整手/流动性/涨跌停（引擎 D 已知 -0.56%/年）
""", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
        print(f"[ waterfall] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

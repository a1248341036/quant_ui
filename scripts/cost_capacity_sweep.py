"""可执行口径扫描：真实成本 × 真实资金规模。

回答两个问题：
  1. 把价差/滑点/冲击算进去之后，超额还剩多少（现在这些参数全是 0）
  2. 在 1万~10万 这个量级，容量约束会不会卡住（max_participation=10%）

窗口与参数完全对齐 engine_gate（2024-07-01~2024-12-27、Top10、日频、amount_q=0）。
"""
from __future__ import annotations

import sys
import time

import pandas as pd

sys.path.insert(0, ".")

from alphaagent.data.adapters import cnequity as cne  # noqa: E402
from alphaagent.factor.mining.delivery.engine_gate import (  # noqa: E402
    panel_to_engine_frame,
)
from core.data.panel import load_pred_scores  # noqa: E402
from core.engine import run_backtest  # noqa: E402

START, END = "2024-07-01", "2024-12-27"
TOP_N = 10
FREQ = "daily"

CAPITALS = [10_000, 30_000, 50_000, 100_000]
FREQS = ["daily", "weekly", "monthly"]

# 成本档：现行 = 引擎默认（只有佣金+印花税，价差/滑点/冲击全 0）
COST_MODES = {
    "现行(价差0/滑点0/冲击0)": dict(spread_bps=0.0, slippage_bps=0.0, impact_coef=0.0),
    "真实(价差22bp/滑点5bp/冲击0.2)": dict(spread_bps=22.0, slippage_bps=5.0, impact_coef=0.2),
}


def main() -> int:
    t0 = time.time()
    print("加载面板（缓存命中）…", flush=True)
    panel = cne.load_panel_from_cne(start="2024-05-01", end="2024-12-31",
                                    include_fundamentals=True)
    print("面板 %d 行 × %d 列，%.1fs" % (len(panel), panel.shape[1], time.time() - t0), flush=True)

    ef = panel_to_engine_frame(panel, asset_type="stock")

    wide = load_pred_scores(channel="demo")
    if wide is None:
        print("pred 通道 demo 读取失败", flush=True)
        return 1
    # pred 文件里 code 是 6 位（load_pred_scores 归一化过），engine_frame 里是
    # '000001.SZ' 形态；不对齐会零交集 → 回测直接报"所选区间/股票池内没有数据"。
    ef["code"] = ef["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    ef = ef.dropna(subset=["code"])
    codes = [str(c) for c in wide.columns]
    print("分数矩阵 %s，覆盖 %s ~ %s" % (wide.shape, wide.index.min().date(), wide.index.max().date()), flush=True)

    rows = []
    for cname, ccfg in COST_MODES.items():
        for freq in FREQS:
            for cap in CAPITALS:
                t = time.time()
                try:
                    res = run_backtest(
                        ef, codes=codes, factor="pred", ascending=False,
                        start=START, end=END, capital=float(cap), top_n=TOP_N,
                        freq=freq, warmup_days=25, external_scores=wide,
                        amount_q=0.0, max_participation=0.10,
                        limit_flags=True, min_commission=5.0,
                        spread_bps=ccfg["spread_bps"],
                        slippage_bps=ccfg["slippage_bps"],
                        impact_coef=ccfg["impact_coef"],
                    )
                    m = res.get("metrics") or {}
                    rows.append({
                        "成本档": cname, "频率": freq, "资金": cap,
                        "年化": m.get("年化收益"), "超额年化": m.get("超额年化"),
                        "夏普": m.get("夏普"), "超额夏普": m.get("超额夏普"),
                        "最大回撤": m.get("最大回撤"),
                    })
                    print("  %-28s %-8s %7d  年化=%6.2f%% 超额=%7.2f%% 夏普=%.3f 超额夏普=%.3f (%.0fs)" % (
                        cname, freq, cap,
                        (m.get("年化收益") or 0) * 100, (m.get("超额年化") or 0) * 100,
                        (m.get("夏普") or 0), (m.get("超额夏普") or 0), time.time() - t), flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("  失败 %s %s %d: %s: %s" % (
                        cname, freq, cap, type(exc).__name__, str(exc)[:100]), flush=True)
                    rows.append({"成本档": cname, "频率": freq, "资金": cap, "年化": None})

    df = pd.DataFrame(rows)
    print("\n=== 汇总 ===")
    print(df.to_string(index=False))
    out = "artifacts/alphaagent/stacking/cost_capacity_sweep.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n已保存:", out, " 总耗时 %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

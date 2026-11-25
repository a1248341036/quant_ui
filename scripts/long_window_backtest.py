"""长窗口回测：频率 × 成本 × 资金。

默认跑 2024-07-01 ~ 2026-10-29（pred 覆盖区间，含 2026 下跌），
可用 --start/--end 改成更长窗口（但需先有覆盖该区间的面板缓存与 pred 分数）。

用法：
  python scripts/long_window_backtest.py
  python scripts/long_window_backtest.py --start 2020-11-24 --end 2026-11-24 --freqs monthly
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from alphaagent.data.adapters import cnequity as cne  # noqa: E402
from alphaagent.factor.mining.delivery.engine_gate import panel_to_engine_frame  # noqa: E402
from core.data.panel import load_pred_scores  # noqa: E402
from core.engine import run_backtest  # noqa: E402

COSTS = {
    "现行": dict(spread_bps=0.0, slippage_bps=0.0, impact_coef=0.0),
    "真实": dict(spread_bps=22.0, slippage_bps=5.0, impact_coef=0.2),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-07-01")
    ap.add_argument("--end", default="2026-10-29")
    ap.add_argument("--freqs", default="monthly,weekly,daily")
    ap.add_argument("--capitals", default="10000,50000")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out", default="artifacts/alphaagent/stacking/long_window_backtest.csv")
    args = ap.parse_args()

    freqs = [f.strip() for f in args.freqs.split(",") if f.strip()]
    caps = [int(c) for c in args.capitals.split(",") if c.strip()]

    # 面板起点留足 warmup（因子窗口 + 平滑）
    panel_start = (pd.Timestamp(args.start) - pd.Timedelta(days=420)).date().isoformat()
    panel = cne.load_panel_from_cne(start=panel_start, end=args.end,
                                    include_fundamentals=True)
    ef = panel_to_engine_frame(panel, asset_type="stock")
    # pred 文件里 code 是 6 位，engine_frame 是 '000001.SZ'；不对齐会零交集
    ef["code"] = ef["code"].astype(str).str.extract(r"(\d{6})", expand=False)
    ef = ef.dropna(subset=["code"])

    wide = load_pred_scores(channel="demo")
    if wide is None:
        print("pred 通道 demo 读取失败", flush=True)
        return 1
    codes = [str(c) for c in wide.columns]
    print("面板 %d 行，分数 %s，窗口 %s~%s" % (len(ef), wide.shape, args.start, args.end), flush=True)

    rows = []
    for cname, cfg in COSTS.items():
        for freq in freqs:
            for cap in caps:
                try:
                    res = run_backtest(
                        ef, codes=codes, factor="pred", ascending=False,
                        start=args.start, end=args.end, capital=float(cap),
                        top_n=args.top_n, freq=freq, warmup_days=25,
                        external_scores=wide, amount_q=0.0, max_participation=0.10,
                        limit_flags=True, min_commission=5.0, **cfg)
                    m = res.get("metrics") or {}
                    rows.append({"成本": cname, "频率": freq, "资金": cap,
                                 "年化": m.get("年化收益"), "超额年化": m.get("超额年化"),
                                 "夏普": m.get("夏普"), "超额夏普": m.get("超额夏普"),
                                 "回撤": m.get("最大回撤")})
                    print("  %-4s %-8s %6d 年化=%7.2f%% 超额=%8.2f%% 夏普=%6.3f 超额夏普=%6.3f 回撤=%6.2f%%" % (
                        cname, freq, cap,
                        (m.get("年化收益") or 0) * 100, (m.get("超额年化") or 0) * 100,
                        (m.get("夏普") or 0), (m.get("超额夏普") or 0),
                        (m.get("最大回撤") or 0) * 100), flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("  失败 %s %s %d: %s" % (cname, freq, cap, str(exc)[:100]), flush=True)

    df = pd.DataFrame(rows)
    print("\n=== 汇总（%s ~ %s）===" % (args.start, args.end))
    print(df.to_string(index=False))
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("\n已保存:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

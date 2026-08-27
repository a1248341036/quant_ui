"""导出「动量 20 日 · 科技TMT · Top3 月度」的本地引擎基准结果，供与米筐回测对比。

产出（默认写到 strategies/ricequant/baseline/）：
- nav.csv             每日净值/基准/回撤
- fills.csv           本地引擎逐笔成交（含费用），与平台成交记录逐笔核对
- monthly_targets.csv 每个调仓日收盘后应持有的目标组合（与平台持仓对比的核心文件）
- rejections.csv      被拒成交（停牌/涨跌停/流动性不足）
- summary.json        收益指标

用法：
    .venv\\Scripts\\python.exe scripts\\ricequant_picks_export.py --start 2024-01-01 --end 2025-12-31

参数默认值与本系统回测页 BacktestRequest 默认一致：
universe=科技TMT(剔科创) / top_n=3 / capital=5000 / freq=monthly /
amount_q=0.2 / buy_cost=0.0008 / sell_cost=0.0013 / warmup_days=400。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.engine import run_backtest
from core.store import PANEL_FILE, STOCK_DIR


def build_codes(universe: str = "tech",
                exclude_kechuang: bool = True) -> list[str]:
    """universe=tech 走科技TMT池；all 走中证800全量池（与回测页 else 分支同口径）。"""
    if universe == "all":
        from core.data import load_universe
        codes = set(load_universe()["code"].astype(str).str.zfill(6))
    else:
        tech = pd.read_csv(STOCK_DIR / "tech.csv", dtype={"code": str})
        codes = set(tech["code"].astype(str).str.zfill(6))
    panel_codes = set(pd.read_parquet(PANEL_FILE, columns=["code"])["code"]
                      .astype(str).str.zfill(6))
    codes &= panel_codes
    if exclude_kechuang:
        codes = {c for c in codes
                 if not c.startswith(("300", "301", "688", "689"))}
    return sorted(codes)


def load_panel(codes: list[str], start: str, end: str,
               warmup_days: int = 450) -> pd.DataFrame:
    calc_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
                  ).strftime("%Y-%m-%d")
    df = pd.read_parquet(
        PANEL_FILE,
        columns=["date", "open", "high", "low", "close", "turnover",
                 "amount", "code", "turn20", "am20", "volume"],
        filters=[("code", "in", codes)],
    )
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= calc_start) & (df["date"] <= end)].copy()


def monthly_targets(trades_detail: list[dict]) -> pd.DataFrame:
    """按调仓日重放成交，得到每个调仓日收盘后的持仓快照。"""
    shares: dict[str, float] = {}
    rows: list[dict] = []
    for tr in sorted(trades_detail, key=lambda x: (x["date"], x["side"] != "sell")):
        code = str(tr["code"]).zfill(6)
        if tr["side"] == "sell":
            shares.pop(code, None)
        else:
            shares[code] = shares.get(code, 0.0) + float(tr["shares"])
        day = pd.Timestamp(tr["date"]).strftime("%Y-%m-%d")
        if rows and rows[-1]["exec_date"] == day:
            rows[-1]["target_codes"] = ",".join(sorted(shares))
            rows[-1]["n_target"] = len(shares)
        else:
            rows.append({"exec_date": day,
                         "signal_date": pd.Timestamp(tr["signal_date"]).strftime("%Y-%m-%d"),
                         "target_codes": ",".join(sorted(shares)),
                         "n_target": len(shares)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--amount-q", type=float, default=0.2)
    # 费用默认值镜像米筐默认佣金模型（佣金倍率=1）：
    # 买入佣金万三 + 卖出佣金万三 + 印花税千一，双边最低 5 元。
    # 回测页面口径是买 0.0008 / 卖 0.0013（最低佣金已默认 5 元），用参数显式指定。
    ap.add_argument("--buy-cost", type=float, default=0.0003)
    ap.add_argument("--sell-cost", type=float, default=0.0013)
    ap.add_argument("--min-commission", type=float, default=5.0)
    ap.add_argument("--factor", default="mom20",
                    help="mom20/mom60/vol20/ma_cross20_30 ...")
    ap.add_argument("--ascending", action="store_true",
                    help="True=买因子最小的一批(反转/低波)")
    ap.add_argument("--freq", default="monthly", choices=["monthly", "weekly", "daily"])
    ap.add_argument("--min-score", type=float, default=None,
                    help="绝对信号门控：质量分下限，方向感知（买小=−score）。无票过线则空仓")
    ap.add_argument("--impact-coef", type=float, default=0.0,
                    help="平方根冲击系数，0=关闭。大资金建议 0.5~1.0")
    ap.add_argument("--universe", default="tech", choices=["tech", "all"],
                    help="tech=科技TMT池  all=中证800全量池")
    ap.add_argument("--out", default=str(ROOT / "strategies/ricequant/baseline"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = build_codes(args.universe)
    panel = load_panel(codes, args.start, args.end)
    res = run_backtest(
        panel, codes, factor=args.factor, ascending=args.ascending,
        start=args.start, end=args.end, capital=args.capital,
        top_n=args.top_n, freq=args.freq, amount_q=args.amount_q,
        warmup_days=400, cash_mode=True, limit_flags=True,
        slippage_bps=0.0, buy_cost=args.buy_cost, sell_cost=args.sell_cost,
        min_commission=args.min_commission, min_score=args.min_score,
        impact_coef=args.impact_coef,
    )

    res["nav"].to_frame("nav").join(res["bench"].rename("bench")).to_csv(
        out_dir / "nav.csv")
    fills = pd.DataFrame(res["trades_detail"])
    if not fills.empty:
        fills["code"] = fills["code"].astype(str).str.zfill(6)
        fills.to_csv(out_dir / "fills.csv", index=False)
    targets = monthly_targets(res["trades_detail"])
    targets.to_csv(out_dir / "monthly_targets.csv", index=False)
    rej = pd.DataFrame(res["rejections"])
    rej.to_csv(out_dir / "rejections.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"params": vars(args), "pool_size": len(codes),
                   "metrics": {k: v for k, v in res["metrics"].items()}},
                  f, ensure_ascii=False, indent=2, default=str)

    print(f"池大小: {len(codes)}  区间: {args.start}~{args.end}")
    print(f"调仓次数: {len(targets)}")
    print(targets.to_string(index=False))
    for k in ("总收益", "年化收益", "夏普", "最大回撤"):
        v = res["metrics"].get(k)
        if v is not None:
            print(f"  {k}: {float(v):.4f}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()

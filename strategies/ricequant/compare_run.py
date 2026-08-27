"""米筐回测产物 ↔ 本地基线 三层对比。

用法：
    .venv\\Scripts\\python.exe strategies\\ricequant\\compare_run.py \
        --dir "C:\\Users\\zhoubw\\Downloads" \
        --run run_id_8140303_202608261428

输入（--dir 目录下按 --run 前缀匹配）：
    *_trade.csv / *_account.csv / *_log.txt
    以及 strategies/ricequant/baseline/{fills,monthly_targets,nav}.csv

输出三层结论：
    A. 逐月目标组合对照（本地 after-exec vs 平台日志目标）
    B. 平台所选票在本地因子截面中的排名（区分数据源洗牌 vs 口径错误）
    C. 逐日总权益曲线偏差统计
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from core.store import PANEL_FILE, STOCK_DIR

POOL = None  # 从 tech.csv 现算，保持与导出脚本同口径


def load_pool() -> list[str]:
    tech = pd.read_csv(STOCK_DIR / "tech.csv", dtype={"code": str})
    codes = set(tech["code"].astype(str).str.zfill(6))
    pc = pd.read_parquet(PANEL_FILE, columns=["code"])
    codes &= set(pc["code"].astype(str).str.zfill(6))
    return sorted(c for c in codes
                  if not c.startswith(("300", "301", "688", "689")))


def parse_log_picks(log_path: Path) -> dict[str, list[str]]:
    """从平台日志提取 [日期] ... 目标(组合): xxx,yyy,zzz（兼容新旧格式）。"""
    pat = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]\s*(?:\S+\s*)?目标(?:组合)?\s*[:：]\s*([^|]*)")
    out: dict[str, list[str]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            picks = [c.strip()[:6] for c in m.group(2).split(",") if c.strip()]
            out[m.group(1)] = picks
    return out


_NEED = {"mom20": 21, "mom60": 61, "vol20": 21, "ma_cross20_30": 30,
         "brk20": 21}


def local_cross_section(sig_date: str, closes_wide: pd.DataFrame,
                        am20_wide: pd.DataFrame,
                        factor: str = "mom20",
                        amount_q: float = 0.2,
                        ascending: bool = False):
    """复现引擎信号日截面：因子排名 + am20 分位过滤。"""
    c = closes_wide.loc[closes_wide.index <= sig_date]
    need = _NEED.get(factor)
    if need is None or len(c) < need:
        return pd.DataFrame()
    w = c.iloc[-need:]
    if factor in ("mom20", "mom60"):
        score = w.iloc[-1] / w.iloc[0] - 1.0
    elif factor == "ma_cross20_30":
        score = w.iloc[-20:].mean() / w.mean() - 1.0
    elif factor == "vol20":
        score = w.pct_change().iloc[1:].std(ddof=1)
    elif factor == "brk20":
        # 收盘 / 前20日最高收盘 - 1（w 的最后一行是信号日）
        score = w.iloc[-1] / w.iloc[:-1].max() - 1.0
    else:
        raise ValueError(factor)
    am = am20_wide.loc[sig_date]
    ok = am.dropna()
    thr = np.quantile(ok.values, amount_q) if len(ok) else np.nan
    valid = [s for s in score.dropna().index
             if s in ok.index and ok[s] > 0 and ok[s] >= thr]
    df = pd.DataFrame({"score": score[valid]}).sort_values(
        "score", ascending=ascending)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def prev_trading_date(closes_wide: pd.DataFrame, d: str) -> str | None:
    idx = closes_wide.index[closes_wide.index <= d]
    if len(idx) < 21:
        return None
    i = idx.get_indexer([pd.Timestamp(d)])
    return str(idx[max(i[0] - 1, 0)].date())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=str(ROOT / "strategies/ricequant/baseline"))
    ap.add_argument("--factor", default="mom20")
    ap.add_argument("--ascending", action="store_true")
    args = ap.parse_args()

    d = Path(args.dir)
    log_file = next(d.glob(f"{args.run}*log*.txt"))
    picks = parse_log_picks(log_file)

    base = Path(args.baseline)
    local_targets = pd.read_csv(base / "monthly_targets.csv")
    local_nav = (pd.read_csv(base / "nav.csv", index_col=0, parse_dates=True)
                 .rename(columns=lambda c: c.strip()))
    local_nav.index.name = "date"
    fills = pd.read_csv(base / "fills.csv")

    pool = load_pool()
    pf = pd.read_parquet(PANEL_FILE, columns=["date", "close", "am20", "code"],
                         filters=[("code", "in", pool)])
    pf["code"] = pf["code"].astype(str).str.zfill(6)
    pf["date"] = pd.to_datetime(pf["date"])
    start = min(pd.Timestamp(k) for k in picks) - pd.Timedelta(days=60)
    pf = pf[pf["date"] >= start]
    closes = pf.pivot_table(index="date", columns="code", values="close",
                            aggfunc="last").sort_index()
    am20w = pf.pivot_table(index="date", columns="code", values="am20",
                           aggfunc="last").reindex(closes.index).sort_index()

    # ---------- A+B：逐月组合对照 + 本地排名反查 ----------
    print("=" * 88)
    print("A/B. 逐月目标组合对照 与 平台所选票的本地因子排名")
    print("=" * 88)
    rows = []
    exec_items = sorted(picks.items())
    detail_idx = set(range(len(exec_items))) if len(exec_items) <= 30 \
        else set(range(5)) | set(range(len(exec_items) - 5, len(exec_items)))
    for i, (exec_d, plat) in enumerate(exec_items):
        sig = prev_trading_date(closes, exec_d)
        cs = (local_cross_section(sig, closes, am20w, factor=args.factor,
                                  ascending=args.ascending)
              if sig else pd.DataFrame())
        loc_row = local_targets[local_targets["exec_date"] == exec_d]
        loc = (loc_row["target_codes"].iloc[0].split(",")
               if not loc_row.empty and isinstance(loc_row["target_codes"].iloc[0], str)
               else [])
        overlap = len(set(loc) & set(plat))
        rank_info = []
        for s in plat:
            if s in cs.index:
                r = cs.loc[s]
                rank_info.append(f"{s}#{int(r['rank'])}({r['score']:+,.4f})")
            else:
                rank_info.append(f"{s}#无效")
        rows.append({"exec_date": exec_d, "signal_date": sig,
                     "local_n": len(loc), "plat_n": len(plat),
                     "overlap": overlap})
        if i not in detail_idx:
            continue
        print(f"\n{exec_d} (信号 {sig})  重合 {overlap}/{len(plat)}")
        print(f"  本地: {','.join(loc) or '(空仓)'}")
        print(f"  平台: {','.join(plat) or '(空仓)'}")
        if rank_info:
            print(f"  平台所选票在本地截面的排名: {' '.join(rank_info)}")
            if cs.shape[0] >= 3:
                top3 = cs.head(3)
                print("  本地截面Top3: " + " ".join(
                    f"{s}({r['score']:+.4f})" for s, r in top3.iterrows()))

    ov = pd.DataFrame(rows)
    print("\n汇总: 平均重合 {:.1f}/{}  完全一致月份数 {}/{}".format(
        ov["overlap"].mean(), ov["plat_n"].max(), int((ov["overlap"] == ov["plat_n"]).sum()),
        len(ov)))

    # ---------- C：逐日权益曲线（有 account 文件才做） ----------
    acct_files = list(d.glob(f"{args.run}*account*.csv"))
    print("\n" + "=" * 88)
    if not acct_files:
        print("C. 未找到 account 文件，跳过权益曲线对比")
        return
    print("C. 逐日总权益对比（起点归一）")
    print("=" * 88)
    acct = pd.read_csv(acct_files[0])
    acct["date"] = pd.to_datetime(acct["日期"].str[:10])
    eq = acct.set_index("date")["总权益"].astype(float)
    eq = eq / eq.iloc[0]
    ln = local_nav["nav"] / local_nav["nav"].iloc[0]
    joined = pd.DataFrame({"local": ln, "rq": eq}).dropna()
    ret_diff = (joined["local"].pct_change() - joined["rq"].pct_change()).abs()
    gap = (joined["local"] - joined["rq"]).abs()
    print(f"共同交易日数: {len(joined)}")
    print(f"期末净值: 本地 {joined['local'].iloc[-1]:.4f} | 米筐 {joined['rq'].iloc[-1]:.4f}")
    print(f"日收益差均值(bps): {ret_diff.mean()*1e4:.1f}  中位数: {ret_diff.median()*1e4:.1f}")
    print(f"净值差最大: {gap.max():.4f} @ {gap.idxmax().date()}  期末差: {gap.iloc[-1]:.4f}")
    m = joined.copy()
    m["ym"] = m.index.strftime("%Y-%m")
    mgap = (m.groupby("ym").last()["local"] - m.groupby("ym").last()["rq"]).abs()
    print("月末净值差最大的5个月:")
    for ymv, v in mgap.sort_values(ascending=False).head(5).items():
        print(f"  {ymv}: {v:.4f}")


if __name__ == "__main__":
    main()

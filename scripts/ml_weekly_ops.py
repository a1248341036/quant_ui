#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML 周更一条龙（周五晚）：训练 → 出名单 → 风格暴露归因 → 漂移台账 → 通知。

分工边界：本脚本只负责**周五晚**这一段；周一开盘执行由既有计划任务
QuantUIPaperRun + scripts/paper_offline_run.py 兜底（core.paper 对同一 exec_date
幂等），不在此重复。

  [1/4] train   : train_ml_composite.py（透传 --train-args；--skip-train 可跳过，
                  直接用最新 stacking run 复跑名单/归因）
  [2/4] 名单    : 取该 run scores.parquet 最新截日 Top-N（--top-n，缺省 20），
                  写 data/stock/ml_weekly_list_<runid>.md/.csv——**只读快照供人工核对**，
                  模拟盘实际执行走 pred_demo.parquet 因子通道，不依赖此文件
  [3/4] 归因    : 复用 analyze_ml_score_exposure 的画像函数算本 run 平均风格暴露
                  → 追加 artifacts/alphaagent/stacking/exposure_ledger.jsonl
  [4/4] 通知    : ops_notify；与台账上一条对比，任一风格 |Δ暴露| ≥ --drift-threshold
                  （缺省 0.05，量纲=截面相关系数差；调大更迟钝/调小更敏感）→ level=warn

用法：
  python scripts/ml_weekly_ops.py                          # 全链（计划任务用 --no-notify 反例）
  python scripts/ml_weekly_ops.py --skip-train             # 复跑名单/归因（不重训）
  python scripts/ml_weekly_ops.py --train-args "--model both --scheme ml"
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STACKING_DIR = ROOT / "artifacts" / "alphaagent" / "stacking"
DATA_DIR = ROOT / "data" / "stock"
EXPOSURE_LEDGER = STACKING_DIR / "exposure_ledger.jsonl"

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# 复用 risk-attribution 归因脚本的画像实现（零口径漂移）；panel 加载器留在模块级
# 便于单测替换合成面板
from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402


def latest_stacking_run(stacking_dir: Path = STACKING_DIR) -> tuple[str, Path] | None:
    """扫 stacking 目录，返回最新一个带 scores.parquet 的 (run_id, 路径)。"""
    if not stacking_dir.is_dir():
        return None
    for d in sorted((x for x in stacking_dir.iterdir() if x.is_dir()), reverse=True):
        scores = d / "scores.parquet"
        if scores.is_file():
            return d.name, scores
    return None


def build_list(scores_path: Path, top_n: int = 20) -> tuple[pd.DataFrame, str]:
    """长表 date/code/score → 最新截日 Top-N（剔 NaN，score 降序）。"""
    df = pd.read_parquet(scores_path)
    latest = str(pd.Timestamp(df["date"].max()).date())
    sub = (df[df["date"] == df["date"].max()]
           .dropna(subset=["score"])
           .nlargest(top_n, "score")[["code", "score"]]
           .reset_index(drop=True))
    return sub, latest


def write_list(top: pd.DataFrame, latest_date: str, run_id: str,
               out_dir: Path = DATA_DIR) -> Path:
    """名单落盘（md 为主，csv 同名备用）。返回 md 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"ml_weekly_list_{run_id}.md"
    lines = [f"# ML 周更名单：{run_id}（截日 {latest_date}，Top {len(top)}）", "",
             "| # | code | score |", "|---|---|---|"]
    for i, r in top.iterrows():
        lines.append(f"| {i + 1} | {r['code']} | {float(r['score']):.4f} |")
    lines += ["", "> 只读快照供人工核对；模拟盘执行走 pred_demo 因子通道，周一开盘由 "
              "QuantUIPaperRun / paper_offline_run 兜底执行。"]
    md.write_text("\n".join(lines), encoding="utf-8")
    top.to_csv(out_dir / f"ml_weekly_list_{run_id}.csv", index=False, encoding="utf-8-sig")
    return md


def exposure_snapshot(scores_path: Path, window: int = 20) -> dict:
    """本 run 的平均风格暴露画像（复用 analyze_ml_score_exposure 的函数，只读）。"""
    import analyze_ml_score_exposure as amx
    from core.risk_model import build_exposures

    score_w = amx._load_score_wide(scores_path)
    d0, d1 = score_w.index.min(), score_w.index.max()
    pad_front = timedelta(days=max(45, window * 2))
    panel = load_panel_from_cne(start=str((d0 - pad_front).date()), end=str(d1.date()),
                                include_fundamentals=False)
    mats = amx.build_style_matrices(panel, window)
    close = mats["close"]
    ind_map = amx._industry_map_from_panel(panel)
    codes = list(close.columns)
    X, names = build_exposures(
        close.values, mats["am20"].values, mats["turn20"].values,
        mom20=mats["mom20"].values, vol20=mats["vol20"].values,
        industry_map=ind_map or None, codes=codes,
    )
    n_style = sum(1 for n in names if not n.startswith("ind_"))
    dates = close.index
    S = score_w.reindex(index=dates, columns=codes).values.astype(float)
    score_mask = np.isfinite(S).sum(axis=1) >= 50
    profile = amx.exposure_profile(S, X, dates, names, n_style, score_mask)
    coef = profile["coef"].mean()
    return {
        "style_exposure": {n: round(float(coef[n]), 4) for n in names[:n_style]},
        "r2_style": round(float(profile["r2_style"].mean()), 4),
        "n_days": int(profile["r2_style"].shape[0]),
    }


def load_exposure_ledger(path: Path = EXPOSURE_LEDGER) -> list[dict]:
    try:
        raw = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in raw:
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def append_exposure_ledger(entry: dict, path: Path = EXPOSURE_LEDGER) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_drift(new_exp: dict[str, float],
                  prev_exp: dict[str, float] | None) -> dict[str, float]:
    """周间风格暴露漂移：仅对两期共有的风格取差；无上一期 → 空（首跑不告警）。"""
    if not prev_exp:
        return {}
    return {k: round(new_exp[k] - prev_exp[k], 4) for k in new_exp.keys() & prev_exp.keys()}


def run_train(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(SCRIPTS / "train_ml_composite.py")]
    if args.train_args:
        cmd += shlex.split(args.train_args)
    print(f"$ {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="ML 周更一条龙（详见模块头注释）")
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过训练，直接用最新 stacking run 复跑名单/归因")
    ap.add_argument("--train-args", default="", help="train_ml_composite 附加参数（shlex 透传）")
    ap.add_argument("--top-n", type=int, default=20, help="名单条数（缺省 20）")
    ap.add_argument("--window", type=int, default=20, help="风格暴露滚动窗口（交易日）")
    ap.add_argument("--drift-threshold", type=float, default=0.05,
                    help="风格暴露周间漂移告警阈值（截面相关差；缺省 0.05）")
    ap.add_argument("--no-notify", dest="notify", action="store_false", default=True)
    args = ap.parse_args()

    if not args.skip_train:
        rc = run_train(args)
        if rc != 0:
            print(f"[abort] 训练失败 rc={rc}")
            if args.notify:
                try:
                    import ops_notify

                    ops_notify.send("ML 周更失败", "训练阶段非零退出", level="error")
                except Exception as exc:  # noqa: BLE001
                    print(f"通知失败（忽略）: {exc}")
            return rc

    run = latest_stacking_run()
    if run is None:
        print("[abort] artifacts/alphaagent/stacking 下没有带 scores.parquet 的 run")
        return 1
    run_id, scores = run
    print(f"stacking run: {run_id}")

    top, latest = build_list(scores, args.top_n)
    md = write_list(top, latest, run_id)
    print(f"名单: {md}（Top {len(top)}，截日 {latest}）")

    snap = exposure_snapshot(scores, window=args.window)
    prev_entries = [e for e in load_exposure_ledger() if e.get("run_id") != run_id]
    prev = prev_entries[-1] if prev_entries else None
    drift = compute_drift(snap["style_exposure"],
                          (prev or {}).get("style_exposure"))
    append_exposure_ledger({
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "run_id": run_id, "latest_date": latest, "top_n": args.top_n,
        "style_exposure": snap["style_exposure"],
        "r2_style": snap["r2_style"], "n_days": snap["n_days"],
    })
    drifted = {k: v for k, v in drift.items() if abs(v) >= args.drift_threshold}
    print(f"风格暴露: {snap['style_exposure']}（R² style-only={snap['r2_style']}）")
    print(f"周间漂移: {drift or '首跑无对照'}"
          + (f" ⚠ 超阈值: {drifted}" if drifted else ""))

    if args.notify:
        try:
            import ops_notify

            top3 = " ".join(top["code"].astype(str).head(3))
            body = (f"run={run_id} 名单 Top{len(top)}: {top3}…\n"
                    f"风格暴露 {snap['style_exposure']}\n"
                    f"漂移 {drifted or '无/首跑'}\n名单: {md}")
            ops_notify.send("ML 周更完成", body, level="warn" if drifted else "info")
        except Exception as exc:  # noqa: BLE001
            print(f"通知失败（忽略）: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

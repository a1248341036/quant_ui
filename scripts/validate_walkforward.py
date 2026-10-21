# -*- coding: utf-8 -*-
"""Walk-forward 验证：剔除选择偏差后，ML 增量库还剩多少真增量（2026-10-20）。

**要回答的问题**：`ml_driven_screen.py` 的筛选和验证都用了 2024 ——
"在 2024 上挑表现好的因子，然后 2024 变好"含**选择效应**，+0.0042 不能全信。
本脚本把两段完全分离：

    阶段①（筛选，只用 ≤2023）：
        训练 2020-2022 → 在 2023 上对每个新因子做 permutation → 选出"有用"名单
    阶段②（验证，2024 从未参与筛选）：
        模型 A：基线因子（候选池），train 2020-2023 → 2024 OOS IC
        模型 B：基线 + 阶段①选出的因子，train 2020-2023 → 2024 OOS IC
        （对照 C：不筛选全塞，同口径）

    真增量 = IC_B − IC_A（这次筛选与验证零重叠）。

**纪律**：全程 ≤2024-12-31，不碰 2025+ 盲测段。固定 seed，确定性可复现。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL = "label_20d_close_to_close"
INC_DIR = ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment"


def _rank_ic_arr(d: np.ndarray, p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """向量化按日截面 rank IC（同 ml_driven_screen）。

    **不接收 y_rank 缓存**：缓存是全局行号索引，子集切片后索引不连续，
    `reindex(日期数组)` 会全部错位成 NaN（2026-10-20 踩过）。label rank 每次重算，
    相比 predict 的耗时可忽略。
    """
    df = pd.DataFrame({"d": d, "p": p, "y": y}).dropna()
    if len(df) < 1000:
        return float("nan"), float("nan")
    df["pr"] = df.groupby("d")["p"].rank()
    df["yr"] = df.groupby("d")["y"].rank()
    g = df.groupby("d")
    n = g["pr"].size().astype(float)
    sp, sy = g["pr"].sum(), g["yr"].sum()
    spy = (df["pr"] * df["yr"]).groupby(df["d"]).sum()
    sp2 = (df["pr"] ** 2).groupby(df["d"]).sum()
    sy2 = (df["yr"] ** 2).groupby(df["d"]).sum()
    corr = ((spy - sp * sy / n) / np.sqrt((sp2 - sp**2/n) * (sy2 - sy**2/n)))\
        .replace([np.inf, -np.inf], np.nan).dropna()
    if not len(corr):
        return float("nan"), float("nan")
    ic = float(corr.mean())
    icir = float(ic / corr.std() * np.sqrt(252)) if len(corr) > 1 and corr.std() > 0 else float("nan")
    return ic, icir


def _load_factors() -> tuple[list[dict], list[dict]]:
    from alphaagent.factor.mining.eval.combo_increment import load_candidate_exprs

    base = load_candidate_exprs()
    inc: dict[str, dict] = {}
    for p in sorted(INC_DIR.glob("batch*_rows.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("pass_increment") and r.get("expr"):
                inc[r["name"]] = {"name": r["name"], "expr": r["expr"]}
    base_names = {b["name"] for b in base}
    return base, [v for k, v in inc.items() if k not in base_names]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--sel-train-end", default="2022-12-31", help="阶段①训练终点")
    ap.add_argument("--sel-end", default="2023-12-31", help="阶段① permutation 段终点（=筛选 OOS）")
    ap.add_argument("--confirm-end", default="2024-12-31", help="阶段② 验证段终点")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--min-contribution", type=float, default=0.0005)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lightgbm as lgb
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import cs_zscore

    base, inc = _load_factors()
    print(f"[wf] 基线 {len(base)} / 增量轨待筛 {len(inc)}", flush=True)

    need: set[str] = {LABEL}
    for c in base + inc:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass
    need = {c for c in need if not str(c).startswith(("funda_", "fsi_"))}

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start=args.start, end=args.confirm_end,
                                include_fundamentals=False,
                                keep_columns=tuple(sorted(need)))
    print(f"[wf] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
    lab = panel[LABEL].astype("float32")
    dates = pd.Series(panel.index.get_level_values(0), index=panel.index)

    def _eval(exprs: list[dict]) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for c in exprs:
            try:
                v = eval_multi_line_factor(c["expr"], panel)
                if isinstance(v, pd.DataFrame):
                    v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
                s = pd.Series(v).astype("float32").reindex(panel.index)
                if s.notna().mean() >= 0.5:
                    out[c["name"]] = cs_zscore(s)
            except Exception:  # noqa: BLE001
                continue
        return out

    print("[wf] 求值...", flush=True)
    zb = _eval(base)
    zi = _eval(inc)
    Xb = pd.DataFrame(zb)
    Xi = pd.DataFrame(zi)
    X = pd.concat([Xb, Xi], axis=1)
    print(f"[wf] 可用：基线 {Xb.shape[1]} / 新因子 {Xi.shape[1]} / 合计 {X.shape[1]}", flush=True)

    def _fit(Xa: pd.DataFrame, tr: pd.Series, seed=42):
        m = lgb.LGBMRegressor(
            n_estimators=args.max_rounds, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            n_jobs=args.n_jobs, verbose=-1)
        m.fit(Xa[tr], lab[tr])
        return m



    # ── 阶段①：筛选（train 2020-2022，permutation 在 2023）──────────────
    sel_tr = dates <= pd.Timestamp(args.sel_train_end)
    sel_oo = (dates > pd.Timestamp(args.sel_train_end)) & (dates <= pd.Timestamp(args.sel_end))
    print(f"[wf] 阶段① 筛选：train ≤{args.sel_train_end}，permutation 段 {args.sel_train_end}~{args.sel_end}",
          flush=True)
    m1 = _fit(X, sel_tr)
    d_oo1 = dates[sel_oo].to_numpy()
    y_oo1 = lab[sel_oo].to_numpy()
    base_ic1, _ = _rank_ic_arr(d_oo1, m1.predict(X[sel_oo]), y_oo1)
    print(f"[wf] 筛选段全模型 IC={base_ic1:+.4f}", flush=True)

    Xoo1 = X[sel_oo].copy()
    rng = np.random.default_rng(42)
    keep: list[str] = []
    contrib = {}
    for i, feat in enumerate(Xi.columns, 1):
        orig = Xoo1[feat].to_numpy().copy()
        drops = []
        for _ in range(args.repeats):
            Xoo1[feat] = rng.permutation(orig)
            ic1, _ = _rank_ic_arr(d_oo1, m1.predict(Xoo1), y_oo1)
            drops.append(base_ic1 - ic1)
        Xoo1[feat] = orig
        c = float(np.mean(drops))
        contrib[feat] = c
        if c > args.min_contribution:
            keep.append(feat)
        if i % 20 == 0:
            print(f"  [{i}/{Xi.shape[1]}] 累计 keep={len(keep)}", flush=True)
    print(f"[wf] 阶段① 选出 {len(keep)} 个（阈值 > {args.min_contribution}）", flush=True)

    # ── 阶段②：验证（train 2020-2023，OOS 2024，筛选段零重叠）────────────
    tr2 = dates <= pd.Timestamp(args.sel_end)
    oos2 = (dates > pd.Timestamp(args.sel_end)) & (dates <= pd.Timestamp(args.confirm_end))
    d_oo2 = dates[oos2].to_numpy()
    y_oo2 = lab[oos2].to_numpy()
    print(f"[wf] 阶段② 验证：train ≤{args.sel_end}，OOS 2024", flush=True)

    mA = _fit(Xb, tr2)
    icA, icirA = _rank_ic_arr(d_oo2, mA.predict(Xb[oos2]), y_oo2)
    print(f"[wf] A 基线（{Xb.shape[1]} 特征）    : IC={icA:+.4f} ICIR={icirA:+.2f}", flush=True)

    if keep:
        # **必须并上基线**：B 是「基线 + 筛选出的新因子」，不是"只有新因子"。
        # 初版写成 X[keep] 等于丢掉全部基线（B 只剩 24 特征，对比无意义）。
        Xc = pd.concat([Xb, X[keep]], axis=1)
        mB = _fit(Xc, tr2)
        icB, icirB = _rank_ic_arr(d_oo2, mB.predict(Xc[oos2]), y_oo2)
        print(f"[wf] B 基线+筛选因子（{Xc.shape[1]} 特征）: IC={icB:+.4f} ICIR={icirB:+.2f}", flush=True)
    else:
        icB, icirB = float("nan"), float("nan")

    mC = _fit(X, tr2)
    icC, icirC = _rank_ic_arr(d_oo2, mC.predict(X[oos2]), y_oo2)
    print(f"[wf] C 不筛选全塞（{X.shape[1]} 特征） : IC={icC:+.4f} ICIR={icirC:+.2f}", flush=True)

    print(f"\n[wf] ===== 结论（2024 OOS，筛选与验证零重叠）=====")
    print(f"[wf] A 基线            : {icA:+.4f}")
    print(f"[wf] B 基线+筛选因子   : {icB:+.4f}   真增量 = {icB-icA:+.5f}")
    print(f"[wf] C 全塞（不筛选）  : {icC:+.4f}   vs 基线 = {icC-icA:+.5f}")
    verdict = ("有效" if icB - icA > 0.002 else
               ("边际（接近零，需更多样本确认）" if icB - icA > -0.001 else "无真增量"))
    print(f"[wf] 判定: {verdict}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ic_baseline": icA, "ic_selected": icB, "ic_all": icC,
            "true_gain": icB - icA, "all_gain": icC - icA,
            "kept": keep, "contribution": contrib, "verdict": verdict,
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[wf] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

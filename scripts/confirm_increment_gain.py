# -*- coding: utf-8 -*-
"""给定「因子名单」，在彻底干净的 OOS 段确认增量（2026-10-20）。

**用途**：`ml_driven_screen.py` 的筛选和验证都用同一年（2024），含选择效应；
`validate_walkforward.py` 已实现分段隔离，但每次都要重跑筛选（十几分钟）。
本脚本只做**确认**：读一份已定的名单（哪个年份筛的都行），在指定 OOS 段比对

    A 基线（候选池）           → OOS IC
    B 基线 + 名单内因子         → OOS IC   ← 真增量 = B - A
    C 基线 + 全部新因子（不筛）  → OOS IC

**纪律**：名单的筛选段与这里的 OOS 段必须不重叠（脚本会打印日期区间提醒）。

用法：
    python scripts/confirm_increment_gain.py \
        --kept-json artifacts/.../walkforward.json --kept-key kept \
        --train-end 2023-12-31 --oos-end 2024-12-31
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
try:
    from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC
    LABEL = str((DEFAULT_RESEARCH_SPEC.get("ml_increment_policy") or {}).get("label")
                or "label_20d_close_to_close")
except Exception:  # 脚本独立运行 / 包不可导入时回落字面量（2026-10-26 收口）
    LABEL = "label_20d_close_to_close"
INC_DIR = ROOT / "artifacts" / "alphaagent" / "harness_runs" / "2026-10-20_increment"


def _rank_ic_arr(d: np.ndarray, p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """向量化按日截面 rank IC（不接收 y_rank 缓存：子集索引不连续会错位成 NaN）。"""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept-json", required=True, help="含名单的 json（ml_screen.json 的 keep / walkforward.json 的 kept）")
    ap.add_argument("--kept-key", default="kept", choices=("kept", "keep"))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--oos-end", default="2024-12-31")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lightgbm as lgb
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import cs_zscore, load_candidate_exprs

    payload = json.loads(Path(args.kept_json).read_text(encoding="utf-8"))
    wanted = set(payload.get(args.kept_key) or [])
    print(f"[confirm] 名单 {len(wanted)} 个（来自 {Path(args.kept_json).name}:{args.kept_key}）")

    base = load_candidate_exprs()
    inc: dict[str, dict] = {}
    for p in sorted(INC_DIR.glob("batch*_rows.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("expr"):
                inc[r["name"]] = {"name": r["name"], "expr": r["expr"]}
    base_names = {b["name"] for b in base}
    new = [v for k, v in inc.items() if k not in base_names]
    kept = [v for v in new if v["name"] in wanted]
    print(f"[confirm] 基线 {len(base)} / 新因子池 {len(new)} / 命中名单 {len(kept)}")

    need: set[str] = {LABEL}
    for c in base + new:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass
    need = {c for c in need if not str(c).startswith(("funda_", "fsi_"))}

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start=args.start, end=args.oos_end,
                                include_fundamentals=False,
                                keep_columns=tuple(sorted(need)))
    print(f"[confirm] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
    lab = panel[LABEL].astype("float32")
    dates = pd.Series(panel.index.get_level_values(0), index=panel.index)

    def _eval(exprs):
        out = {}
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

    print("[confirm] 求值...", flush=True)
    Xb = pd.DataFrame(_eval(base))
    Xn = pd.DataFrame(_eval(new))
    Xk = Xn[[c for c in Xn.columns if c in wanted]]
    print(f"[confirm] 可用：基线 {Xb.shape[1]} / 全池 {Xn.shape[1]} / 名单 {Xk.shape[1]}", flush=True)

    tr = dates <= pd.Timestamp(args.train_end)
    oo = (dates > pd.Timestamp(args.train_end)) & (dates <= pd.Timestamp(args.oos_end))
    d_oo = dates[oo].to_numpy()
    y_oo = lab[oo].to_numpy()

    def _fit(Xa):
        m = lgb.LGBMRegressor(
            n_estimators=args.max_rounds, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
            n_jobs=args.n_jobs, verbose=-1)
        m.fit(Xa[tr], lab[tr])
        return m

    def _score(Xa):
        m = _fit(Xa)
        return _rank_ic_arr(d_oo, m.predict(Xa[oo]), y_oo)

    icA, icirA = _score(Xb)
    print(f"[confirm] A 基线（{Xb.shape[1]}）              : IC={icA:+.4f} ICIR={icirA:+.2f}", flush=True)
    icB = icirB = float("nan")
    if Xk.shape[1]:
        icB, icirB = _score(pd.concat([Xb, Xk], axis=1))
        print(f"[confirm] B 基线+名单（{Xb.shape[1]+Xk.shape[1]}）      : IC={icB:+.4f} ICIR={icirB:+.2f}", flush=True)
    icC, icirC = _score(pd.concat([Xb, Xn], axis=1))
    print(f"[confirm] C 基线+全池（{Xb.shape[1]+Xn.shape[1]}）      : IC={icC:+.4f} ICIR={icirC:+.2f}", flush=True)

    print(f"\n[confirm] ===== OOS {args.train_end} ~ {args.oos_end} =====")
    print(f"[confirm] 名单增量 B-A = {icB-icA:+.5f}")
    print(f"[confirm] 全池增量 C-A = {icC-icA:+.5f}")
    verdict = ("真增量" if icB - icA > 0.002 else
               ("边际（接近零）" if icB - icA > -0.001 else "无真增量"))
    print(f"[confirm] 判定: {verdict}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "ic_baseline": icA, "ic_named": icB, "ic_all": icC,
            "gain_named": icB - icA, "gain_all": icC - icA,
            "n_base": int(Xb.shape[1]), "n_named": int(Xk.shape[1]), "n_all": int(Xn.shape[1]),
            "train_end": args.train_end, "oos_end": args.oos_end, "verdict": verdict,
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[confirm] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

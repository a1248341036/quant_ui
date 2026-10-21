# -*- coding: utf-8 -*-
"""ML 驱动的因子筛选（2026-10-20）：以 **ML 的 OOS 表现** 为唯一标准，不用固定 IC 门。

**为什么不沿用固定 IC 门**
2026-10-20 实测：用"候选池等权组合"做控制变量算出的增量（`combo_increment`），
捞回 87 个因子后喂进 lgbm，OOS IC 反而 **下降 0.0036**（0.0864 → 0.0828），
且这些因子在模型里拿了 17.86% 的 gain —— 不是没被用，是用了更差。
根因：**等权平均代表不了 lgbm 的组合能力**，代理指标把方向带偏了。

**本脚本的做法（以 ML 结果为标准）**
1. 一次训练全模型（基线因子 ∪ 新因子），得到 OOS 基线 IC；
2. 对每个**待评估的新因子**做 permutation importance：把该列在 OOS 段打乱，
   重新预测并算 IC —— **IC 降幅就是该因子对模型的真实贡献**；
3. 按降幅排序：正值 = 有用；接近 0 = 冗余；负值 = **有害**（打乱它反而更好）；
4. 输出建议保留/剔除名单，可选 `--prune` 做一轮"剔除有害因子后重训"的验证。

成本：一次训练（~3 分钟）+ 每因子 3 次预测（~2s）→ 100 个因子约 10 分钟。
对比"逐因子重训"（3-5 分钟/因子）快 20 倍以上，且判据就是 ML 本身。

**纪律**：只用 ≤2024 的数据（train 2020-2023 / OOS 2024），**不碰 2025+ 盲测段**。

用法：
    # 只诊断：给出每个新因子的 ML 贡献
    python scripts/ml_driven_screen.py --rows <batch*_rows.jsonl ...>

    # 诊断后剔除有害因子重训，验证是否真的更好
    python scripts/ml_driven_screen.py --rows <...> --prune
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


def _rank_ic_arr(d: np.ndarray, p: np.ndarray, y: np.ndarray,
                 y_rank_cache: pd.Series | None = None) -> tuple[float, float]:
    """按日截面 rank IC + 年化 ICIR（向量化，无 groupby.apply —— 后者在 400 万行上极慢）。

    ``y_rank_cache`` 可传入预先算好的 label 截面 rank（permutation 循环里 label 不变，
    预计算一次即可），省掉重复排序。
    """
    df = pd.DataFrame({"d": d, "p": p, "y": y}).dropna()
    if len(df) < 1000:
        return float("nan"), float("nan")
    df["pr"] = df.groupby("d")["p"].rank()
    df["yr"] = y_rank_cache.reindex(df.index).to_numpy() if y_rank_cache is not None \
        else df.groupby("d")["y"].rank()
    g = df.groupby("d")
    n = g["pr"].size().astype(float)
    sp, sy = g["pr"].sum(), g["yr"].sum()
    spy = (df["pr"] * df["yr"]).groupby(df["d"]).sum()
    sp2 = (df["pr"] ** 2).groupby(df["d"]).sum()
    sy2 = (df["yr"] ** 2).groupby(df["d"]).sum()
    cov = spy - sp * sy / n
    vp = sp2 - sp ** 2 / n
    vy = sy2 - sy ** 2 / n
    corr = (cov / np.sqrt(vp * vy)).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(corr):
        return float("nan"), float("nan")
    ic = float(corr.mean())
    icir = float(ic / corr.std() * np.sqrt(252)) if len(corr) > 1 and corr.std() > 0 else float("nan")
    return ic, icir


def _load_new_factors(rows_paths: list[str]) -> list[dict]:
    seen, out = set(), []
    for p in rows_paths:
        path = Path(p)
        if not path.is_file():
            print(f"[screen] 跳过（不存在）: {p}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("expr") or r["name"] in seen:
                continue
            seen.add(r["name"])
            out.append({"name": r["name"], "expr": r["expr"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True,
                    help="harness_mine 产出的 rows.jsonl（取其 expr 作为待评估因子）")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--oos-end", default="2024-12-31")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3, help="每因子置换次数（取均值）")
    ap.add_argument("--prune", action="store_true", help="剔除负贡献因子后重训对比")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import cs_zscore, load_candidate_exprs

    base = load_candidate_exprs()
    new = _load_new_factors(args.rows)
    base_names = {b["name"] for b in base}
    new = [n for n in new if n["name"] not in base_names]
    print(f"[screen] 基线因子 {len(base)} / 待评估新因子 {len(new)}", flush=True)

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
    print(f"[screen] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
    lab = panel[LABEL].astype("float32")
    dates = pd.Series(panel.index.get_level_values(0), index=panel.index)
    train_mask = dates <= pd.Timestamp(args.train_end)
    oos_mask = (dates > pd.Timestamp(args.train_end)) & (dates <= pd.Timestamp(args.oos_end))

    def _eval(exprs: list[dict], tag: str) -> dict[str, pd.Series]:
        from alphaagent.dsl.eval import eval_multi_line_factor as _ev

        out: dict[str, pd.Series] = {}
        for i, c in enumerate(exprs, 1):
            try:
                v = _ev(c["expr"], panel)
                if isinstance(v, pd.DataFrame):
                    v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
                s = pd.Series(v).astype("float32").reindex(panel.index)
                if s.notna().mean() < 0.5:
                    continue
                out[c["name"]] = cs_zscore(s)
            except Exception:  # noqa: BLE001
                continue
            if i % 20 == 0:
                print(f"  [{tag} {i}/{len(exprs)}] 可用 {len(out)}", flush=True)
        return out

    import lightgbm as lgb

    print("[screen] 求值基线因子...", flush=True)
    zb = _eval(base, "base")
    print(f"[screen] 基线可用 {len(zb)}", flush=True)
    print("[screen] 求值新因子...", flush=True)
    zn = _eval(new, "new")
    print(f"[screen] 新因子可用 {len(zn)}", flush=True)

    Xb = pd.DataFrame(zb)
    Xn = pd.DataFrame(zn)
    X = pd.concat([Xb, Xn], axis=1)

    def _fit(X_all: pd.DataFrame, y: pd.Series, tr, oo, seed=42):
        m = lgb.LGBMRegressor(
            n_estimators=args.max_rounds, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
            n_jobs=args.n_jobs, verbose=-1,
        )
        m.fit(X_all[tr], y[tr])
        return m

    print("[screen] 训练全模型...", flush=True)
    model = _fit(X, lab, train_mask, oos_mask)
    d_oos = dates[oos_mask].to_numpy()
    yoo = lab[oos_mask].to_numpy()
    pred = model.predict(X[oos_mask])
    base_ic, base_icir = _rank_ic_arr(d_oos, pred, yoo)
    # label 的截面 rank 预计算一次（permutation 里 label 不变，省 87×3 次重复排序）
    _ydf = pd.DataFrame({"d": d_oos, "y": yoo})
    y_rank_cache = _ydf.groupby("d")["y"].rank()
    print(f"[screen] 全模型 OOS: IC={base_ic:+.4f} ICIR={base_icir:+.2f} "
          f"特征={X.shape[1]}", flush=True)

    # permutation importance：**原地替换单列**（不复制整表 —— 134 列 × 400 万行整表
    # copy 一次就是 2GB，87×3 次会直接打爆内存，2026-10-20 实测吃到 13.2GB）
    print(f"[screen] permutation importance（每因子 {args.repeats} 次）...", flush=True)
    Xoo = X[oos_mask].copy()
    rng = np.random.default_rng(42)
    results = []
    for i, feat in enumerate(Xn.columns, 1):
        orig = Xoo[feat].to_numpy().copy()
        drops = []
        for _ in range(args.repeats):
            Xoo[feat] = rng.permutation(orig)
            p = model.predict(Xoo)
            ic, _icir = _rank_ic_arr(d_oos, p, yoo, y_rank_cache)
            drops.append(base_ic - ic)  # 正值 = 该因子真的 contributes
        Xoo[feat] = orig  # 还原，供下一个因子使用
        results.append({"name": feat, "ml_contribution": float(np.mean(drops)),
                        "std": float(np.std(drops))})
        if i % 10 == 0:
            print(f"  [{i}/{len(Xn.columns)}] {feat[:28]} 贡献={results[-1]['ml_contribution']:+.5f}",
                  flush=True)

    results.sort(key=lambda r: -r["ml_contribution"])
    print(f"\n{'因子':36s} {'ML贡献(OOS IC降幅)':>18s} 判定")
    print("-" * 72)
    keep, drop = [], []
    for r in results:
        c = r["ml_contribution"]
        verdict = "有用" if c > 0.0005 else ("冗余" if c > -0.0005 else "有害")
        (keep if c > 0.0005 else drop).append(r["name"])
        print(f"{r['name'][:36]:36s} {c:+18.5f} {verdict}")

    print(f"\n[screen] 有用 {len(keep)} 个 / 冗余或有害 {len(drop)} 个", flush=True)

    if args.prune and keep:
        print(f"\n[screen] --prune: 仅保留 {len(keep)} 个有用因子重训对比...", flush=True)
        Xp2 = X[[c for c in X.columns if (c in Xb.columns or c in keep)]]
        m2 = _fit(Xp2, lab, train_mask, oos_mask)
        p2 = m2.predict(Xp2[oos_mask])
        ic2, icir2 = _rank_ic_arr(d_oos, p2, yoo, y_rank_cache)
        print(f"[screen] 剪枝后 OOS: IC={ic2:+.4f} ICIR={icir2:+.2f} 特征={Xp2.shape[1]}", flush=True)
        print(f"[screen] === 相对全模型 IC {base_ic:+.4f} -> {ic2:+.4f} "
              f"({ic2-base_ic:+.5f}) ===", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"base_oos_ic": base_ic, "base_icir": base_icir,
             "n_base": Xb.shape[1], "n_new": Xn.shape[1],
             "contributions": results, "keep": keep, "drop": drop},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[screen] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

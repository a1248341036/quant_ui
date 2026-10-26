# -*- coding: utf-8 -*-
"""增量轨价值验证：把增量轨因子加入 lgbm，看 OOS IC 是否真提升。

**要回答的问题**：增量轨捞回 87 个因子（单因子盲测 retention 0.11–0.30，多数过不了
盲测门），它们对 **ML 组合**到底有没有增量？如果 lgbm 加了这些特征后 OOS IC 不动，
那增量轨就是白搭；如果有提升，说明"单因子不行但对组合有用"这个假设成立。

**方法（对齐但简化，只做归因不做交付）**
1. 因子集 A（基线）= 候选池 51 个里能在 train 段求值的（funda 类跳过）；
2. 因子集 B（+增量轨）= A ∪ 增量轨捞回的因子；
3. 同一 panel、同一 label、同一日期切分（train 2020–2023 / OOS 2024）；
4. 两套 LightGBM 同参数各训一次，对比 **OOS 段 rank IC / ICIR**。
5. 另外报「增量因子在 B 模型里的总 gain 占比」——直接看模型有没有用它们。

**纪律**：只用 train 段（≤2024-12-31），**不碰 2025+ 盲测段**。这是归因实验，
不产出交付结论。全流程确定性（固定 seed、不调参）。

用法：
    python scripts/validate_increment_ml.py
    python scripts/validate_increment_ml.py --n-jobs 4 --max-rounds 300
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


def _load_factor_sets() -> tuple[list[dict], list[dict]]:
    """返回 (基线因子, 增量轨因子)，都已按"能在 train 段求值"过滤。"""
    from alphaagent.factor.mining.eval.combo_increment import load_candidate_exprs

    base = load_candidate_exprs()
    inc: dict[str, dict] = {}
    for p in sorted(INC_DIR.glob("batch*_rows.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("pass_increment") or not r.get("expr"):
                continue
            inc[r["name"]] = {"name": r["name"], "expr": r["expr"]}
    base_names = {b["name"] for b in base}
    inc_list = [v for k, v in inc.items() if k not in base_names]
    return base, inc_list


def _eval_all(exprs: list[dict], panel, *, tag: str) -> dict[str, pd.Series]:
    from alphaagent.dsl.eval import eval_multi_line_factor
    from alphaagent.factor.mining.eval.combo_increment import cs_zscore

    out: dict[str, pd.Series] = {}
    for i, c in enumerate(exprs, 1):
        try:
            v = eval_multi_line_factor(c["expr"], panel)
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
            s = pd.Series(v).astype("float32").reindex(panel.index)
            if s.notna().mean() < 0.5:
                continue
            out[c["name"]] = cs_zscore(s)
            if i % 10 == 0:
                print(f"  [{tag} {i}/{len(exprs)}] 累计 {len(out)} 个可用", flush=True)
        except Exception:  # noqa: BLE001
            continue
    return out


def _fit_eval(X: pd.DataFrame, y: pd.Series, dates: pd.Series, args) -> dict:
    import lightgbm as lgb

    train_mask = dates <= pd.Timestamp(args.train_end)
    oos_mask = (dates > pd.Timestamp(args.train_end)) & (dates <= pd.Timestamp(args.oos_end))
    Xtr, ytr = X[train_mask], y[train_mask]
    Xoo, yoo = X[oos_mask], y[oos_mask]
    if len(Xtr) < 10000 or len(Xoo) < 5000:
        return {"error": f"样本不足 train={len(Xtr)} oos={len(Xoo)}"}

    model = lgb.LGBMRegressor(
        n_estimators=args.max_rounds, learning_rate=0.05, num_leaves=31,
        min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
        n_jobs=args.n_jobs, verbose=-1,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xoo)
    oos_dates = dates[oos_mask].to_numpy()
    df = pd.DataFrame({"d": oos_dates, "p": pred, "y": yoo.to_numpy()})
    ics = df.dropna().groupby("d").apply(
        lambda g: g["p"].corr(g["y"], method="spearman") if len(g) > 30 else np.nan
    ).dropna()
    ic = float(ics.mean()) if len(ics) else float("nan")
    icir = float(ic / ics.std() * np.sqrt(252)) if len(ics) > 1 and ics.std() > 0 else float("nan")
    gain = model.booster_.feature_importance(importance_type="gain")
    feats = list(X.columns)
    return {"ic": ic, "icir": icir, "n_days": int(len(ics)),
            "n_train": int(len(Xtr)), "n_oos": int(len(Xoo)),
            "gain": dict(zip(feats, (gain / gain.sum()).tolist()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--oos-end", default="2024-12-31")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import referenced_column_names

    base, inc = _load_factor_sets()
    print(f"[val] 基线因子 {len(base)} 个 / 增量轨因子 {len(inc)} 个", flush=True)

    all_exprs = base + inc
    need: set[str] = {LABEL}
    for c in all_exprs:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass
    need = {c for c in need if not str(c).startswith(("funda_", "fsi_"))}

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start=args.start, end=args.oos_end,
                                include_fundamentals=False,
                                keep_columns=tuple(sorted(need)))
    print(f"[val] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
    lab = panel[LABEL].astype("float32")
    dates = pd.Series(panel.index.get_level_values(0), index=panel.index)

    print("[val] 求值基线因子...", flush=True)
    zb = _eval_all(base, panel, tag="base")
    print(f"[val] 基线可用 {len(zb)}", flush=True)
    print("[val] 求值增量轨因子...", flush=True)
    zi = _eval_all(inc, panel, tag="inc")
    print(f"[val] 增量轨可用 {len(zi)}", flush=True)

    Xb = pd.DataFrame(zb)
    Xi = pd.DataFrame(zi)
    res_a = _fit_eval(Xb, lab, dates, args)
    print(f"[val] A 基线   : IC={res_a.get('ic'):+.4f} ICIR={res_a.get('icir'):+.2f} "
          f"n_feat={Xb.shape[1]}", flush=True)

    Xc = pd.concat([Xb, Xi], axis=1)
    res_b = _fit_eval(Xc, lab, dates, args)
    print(f"[val] B +增量轨: IC={res_b.get('ic'):+.4f} ICIR={res_b.get('icir'):+.2f} "
          f"n_feat={Xc.shape[1]}", flush=True)

    if "gain" in res_b:
        inc_names = set(Xi.columns)
        inc_gain = sum(v for k, v in res_b["gain"].items() if k in inc_names)
        print(f"\n[val] 增量轨因子在 B 模型里的总 gain 占比: {inc_gain*100:.2f}% "
              f"（{len(inc_names)}/{Xc.shape[1]} 个特征）", flush=True)
        top = sorted(((k, v) for k, v in res_b["gain"].items() if k in inc_names),
                     key=lambda kv: -kv[1])[:10]
        print("[val] 增量轨因子 gain top10:", flush=True)
        for k, v in top:
            print(f"   {v*100:6.2f}%  {k}", flush=True)

    if "ic" in res_a and "ic" in res_b:
        d = res_b["ic"] - res_a["ic"]
        print(f"\n[val] === OOS IC 提升: {d:+.5f} "
              f"({res_a['ic']:+.4f} -> {res_b['ic']:+.4f}) ===", flush=True)
        print(f"[val] 判定: {'增量轨有效' if d > 0.002 else ('无显著增量' if d > -0.002 else '增量轨有害')}",
              flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({"baseline": res_a, "with_increment": res_b},
                                             ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
        print(f"[val] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

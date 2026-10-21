# -*- coding: utf-8 -*-
"""ML 增量挖掘 pipeline（增量学习式，2026-10-20）。

**做什么**：像深度学习增量学习那样，每批新因子都在**当前基线**上接受检验——
有效就累积入库、基线随之上移；无效就整批作废，不留垃圾。

    ┌─ 当前基线 = 候选池(candidate_main) + 已入库增量库(ml_increment_main)
    │
    ├─ ① 新批次因子 → 求值
    ├─ ② 训练 A（基线）            → judge 段日度 IC 序列
    ├─ ③ 训练 B（基线 + 本批全量）  → judge 段日度 IC 序列
    ├─ ④ paired t 检验 ΔIC(t) = IC_B(t) - IC_A(t)
    │     显著为正 → 全批入库，基线扩大
    │     不显著 → ⑤ permutation 挑 top-k 再试，仍不显著 → 整批作废
    └─ 下一批以此为新基线

**为什么必须 paired 检验**：单年 OOS 的绝对 IC 标准误 ≈ 0.011（244 天、ICIR≈8），
而因子增量通常只有 0.002 量级 —— 直接比均值等于拿噪声当信号。paired 检验消掉
A/B 共同的日间波动（两者因子高度重叠），标准误可降到 0.003 级，才谈得上判别。

**防过拟合的三条硬约束（对应深度学习里的 validation leakage）**：
1. judge 段固定为 `2024`（`--judge-end`），**不许为了好看改**；2025+ 盲测段永不触碰；
2. 同一批因子**只判定一次**；不过就是不过，不许"改一版再试同一批"直到通过；
3. permutation 只在 train 内部子段做（`--screen-end`），judge 段只用于最终判定。

用法：
    # 只判定，不入库
    python scripts/ml_incremental_mine.py --rows batch5_rows.jsonl
    # 判定通过则自动入库并更新基线
    python scripts/ml_incremental_mine.py --rows batch5_rows.jsonl --apply
    # 挖掘流水线标准用法（基线缓存 → 判定 → 过门自动入库 → 缓存基线上移）
    python scripts/ml_incremental_mine.py --rows <环路产出的 rows.jsonl> --apply \
        --feature-cache artifacts/alphaagent/harness_runs/base_features_cache.parquet
"""
from __future__ import annotations

import argparse
import json
import math
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
INC_LIB = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "ml_increment_main" / "mining_ml_increment_registry.json"


def daily_rank_ic(d: np.ndarray, p: np.ndarray, y: np.ndarray) -> pd.Series:
    """返回**逐日**截面 rank IC 序列（索引=交易日），供 paired 检验用。"""
    df = pd.DataFrame({"d": d, "p": p, "y": y}).dropna()
    if len(df) < 1000:
        return pd.Series(dtype="float64")
    df["pr"] = df.groupby("d")["p"].rank()
    df["yr"] = df.groupby("d")["y"].rank()
    g = df.groupby("d")
    n = g["pr"].size().astype(float)
    sp, sy = g["pr"].sum(), g["yr"].sum()
    spy = (df["pr"] * df["yr"]).groupby(df["d"]).sum()
    sp2 = (df["pr"] ** 2).groupby(df["d"]).sum()
    sy2 = (df["yr"] ** 2).groupby(df["d"]).sum()
    corr = ((spy - sp * sy / n) / np.sqrt((sp2 - sp**2 / n) * (sy2 - sy**2 / n)))
    return corr.replace([np.inf, -np.inf], np.nan).dropna()


def paired_test(ic_a: pd.Series, ic_b: pd.Series) -> dict:
    """对同一组交易日做 paired t 检验：H1 mean(ΔIC) > 0。"""
    j = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    if len(j) < 30:
        return {"n": int(len(j)), "mean_delta": float("nan"), "t": float("nan"),
                "p_one_sided": float("nan"), "se": float("nan")}
    delta = (j["b"] - j["a"]).to_numpy()
    n = len(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else float("inf")
    t = mean / se if se > 0 and math.isfinite(se) else 0.0
    # 单侧 p 值（正态近似；n>200 时与 t 分布几乎一致）
    p = 0.5 * math.erfc(t / math.sqrt(2))
    return {"n": n, "mean_delta": mean, "se": se, "t": t, "p_one_sided": p,
            "ic_a_mean": float(j["a"].mean()), "ic_b_mean": float(j["b"].mean())}


def load_base_factors(include_ml_increment: bool = True) -> list[dict]:
    from alphaagent.factor.mining.eval.combo_increment import load_candidate_exprs

    base = list(load_candidate_exprs())
    if include_ml_increment and INC_LIB.is_file():
        reg = json.loads(INC_LIB.read_text(encoding="utf-8"))
        seen = {b["name"] for b in base}
        for name, rec in reg.items():
            if name in seen:
                continue
            if rec.get("expr"):
                base.append({"name": name, "expr": rec["expr"]})
    return base


def load_new_factors(paths: list[str]) -> list[dict]:
    seen, out = set(), []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            print(f"[inc] 跳过（不存在）: {p}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("name") and r.get("expr") and r["name"] not in seen:
                seen.add(r["name"])
                out.append({"name": r["name"], "expr": r["expr"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="本批新因子的 rows.jsonl")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--train-end", default="2023-12-31", help="训练终点（judge 段起点）")
    ap.add_argument("--judge-end", default="2024-12-31",
                    help="judge 段终点。**固定为 2024，不许为好看而改**；2025+ 盲测永不触碰")
    ap.add_argument("--screen-end", default="2022-12-31",
                    help="permutation 筛选子段终点（须 < train_end，避免污染 judge）")
    ap.add_argument("--t-threshold", type=float, default=1.28,
                    help="paired t 单侧阈值（1.28≈90%% 置信）")
    ap.add_argument("--min-delta", type=float, default=0.001,
                    help="ΔIC 均值下限（t 显著但幅度过小也不收）")
    ap.add_argument("--top-k", type=int, default=15, help="整批不过时，permutation 挑前 k 个再试")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--apply", action="store_true", help="判定通过则自动入库（自动构造 screen 并调 promote_ml_increment）")
    ap.add_argument("--feature-cache", default="",
                    help="基线特征矩阵缓存 parquet 路径；命中则跳过基线求值（挖掘流水线高频复用，"
                         "入库后自动追加列=基线上移；基线名单或 start/judge_end 变化时自动重建）")
    ap.add_argument("--rebuild-cache", action="store_true", help="忽略并重建特征缓存")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lightgbm as lgb
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import cs_zscore

    base = load_base_factors()
    new = load_new_factors(args.rows)
    base_names = {b["name"] for b in base}
    new = [n for n in new if n["name"] not in base_names]
    print(f"[inc] 当前基线 {len(base)} 个 / 本批新因子 {len(new)} 个", flush=True)
    if not new:
        print("[inc] 本批没有新因子")
        return 0

    need: set[str] = {LABEL}
    for c in base + new:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass
    need = {c for c in need if not str(c).startswith(("funda_", "fsi_"))}

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start=args.start, end=args.judge_end,
                                include_fundamentals=False, keep_columns=tuple(sorted(need)))
    print(f"[inc] panel {panel.shape} load={time.perf_counter()-t0:.0f}s", flush=True)
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

    cache_path = Path(args.feature_cache) if args.feature_cache else None
    Xb = None
    if cache_path and not args.rebuild_cache and cache_path.is_file():
        meta_p = cache_path.with_suffix(".meta.json")
        if meta_p.is_file():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                cur_names = sorted({b["name"] for b in base})
                if (meta.get("start") == args.start and meta.get("judge_end") == args.judge_end
                        and meta.get("base_names_all") == cur_names):
                    df = pd.read_parquet(cache_path)
                    idx = pd.MultiIndex.from_arrays(
                        [pd.to_datetime(df["__date__"]), df["__code__"]])
                    lab = df["__label__"].astype("float32")
                    lab.index = idx
                    Xb = df.drop(columns=["__date__", "__code__", "__label__"])
                    Xb.index = idx
                    print(f"[inc] 基线缓存命中 {Xb.shape[1]} 列（{cache_path.name}）", flush=True)
                else:
                    print("[inc] 缓存元数据不匹配（基线名单/日期段变化），重建", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[inc] 缓存读取失败，重建: {type(e).__name__} {e}", flush=True)
                Xb = None
    if Xb is None:
        print("[inc] 求值基线...", flush=True)
        Xb = pd.DataFrame(_eval(base))
        if cache_path:
            buf = Xb.copy()
            buf["__label__"] = lab
            buf["__date__"] = panel.index.get_level_values(0)
            buf["__code__"] = panel.index.get_level_values(1)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            buf.reset_index(drop=True).to_parquet(cache_path)
            cache_path.with_suffix(".meta.json").write_text(json.dumps({
                "start": args.start, "judge_end": args.judge_end,
                "base_names_all": sorted({b["name"] for b in base})}), encoding="utf-8")
            print(f"[inc] 基线缓存已写 {cache_path.name}（{Xb.shape[1]} 列）", flush=True)
    print("[inc] 求值新因子...", flush=True)
    Xn = pd.DataFrame(_eval(new))
    if not Xn.shape[1]:
        print("[inc] 新因子全部求值失败")
        return 1
    print(f"[inc] 可用：基线 {Xb.shape[1]} / 新 {Xn.shape[1]}", flush=True)

    tr = dates <= pd.Timestamp(args.train_end)
    jd = (dates > pd.Timestamp(args.train_end)) & (dates <= pd.Timestamp(args.judge_end))
    d_jd = dates[jd].to_numpy()
    y_jd = lab[jd].to_numpy()

    def _fit(Xa, mask):
        m = lgb.LGBMRegressor(
            n_estimators=args.max_rounds, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
            n_jobs=args.n_jobs, verbose=-1)
        m.fit(Xa[mask], lab[mask])
        return m

    def _ic_seq(Xa):
        m = _fit(Xa, tr)
        return daily_rank_ic(d_jd, m.predict(Xa[jd]), y_jd)

    print("[inc] ② 训练基线模型...", flush=True)
    ic_a = _ic_seq(Xb)
    print(f"[inc] 基线 judge IC={ic_a.mean():+.4f} ({len(ic_a)} 天)", flush=True)

    def _report(tag, ic_b):
        r = paired_test(ic_a, ic_b)
        passed = (math.isfinite(r["t"]) and r["t"] >= args.t_threshold
                  and math.isfinite(r["mean_delta"]) and r["mean_delta"] >= args.min_delta)
        print(f"[inc] {tag}: judge IC={r['ic_b_mean']:+.4f} "
              f"ΔIC={r['mean_delta']:+.5f} SE={r['se']:.5f} t={r['t']:+.2f} "
              f"p={r['p_one_sided']:.3f} n={r['n']} -> {'PASS' if passed else 'fail'}",
              flush=True)
        return r, passed

    print("[inc] ③ 训练 +全批...", flush=True)
    ic_full = _ic_seq(pd.concat([Xb, Xn], axis=1))
    rep_full, pass_full = _report("全批", ic_full)

    chosen: list[str] = []
    permutation_drop: dict = {}
    rep = rep_full
    passed = pass_full
    if not pass_full:
        # ④ 整批不过 → 在 train 内部子段做 permutation 挑 top-k（judge 段零污染）
        print(f"[inc] 整批未过，转 permutation 挑选 top-{args.top_k}"
              f"（筛选子段 ≤{args.screen_end}）", flush=True)
        scr_tr = dates <= pd.Timestamp(args.screen_end)
        scr_oo = (dates > pd.Timestamp(args.screen_end)) & (dates <= pd.Timestamp(args.train_end))
        d_s = dates[scr_oo].to_numpy()
        y_s = lab[scr_oo].to_numpy()
        m_s = _fit(pd.concat([Xb, Xn], axis=1), scr_tr)
        Xs = pd.concat([Xb, Xn], axis=1)[scr_oo].copy()
        base_s = daily_rank_ic(d_s, m_s.predict(Xs), y_s)
        base_s_mean = base_s.mean()
        rng = np.random.default_rng(42)
        contrib = {}
        for i, feat in enumerate(Xn.columns, 1):
            orig = Xs[feat].to_numpy().copy()
            drops = []
            for _ in range(2):
                Xs[feat] = rng.permutation(orig)
                drops.append(base_s_mean - daily_rank_ic(d_s, m_s.predict(Xs), y_s).mean())
            Xs[feat] = orig
            contrib[feat] = float(np.mean(drops))
            if i % 20 == 0:
                print(f"  [{i}/{Xn.shape[1]}]", flush=True)
        chosen = [k for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0][: args.top_k]
        permutation_drop = dict(contrib)
        print(f"[inc] 挑出 {len(chosen)} 个: {chosen[:8]}{'...' if len(chosen) > 8 else ''}", flush=True)
        if chosen:
            ic_sel = _ic_seq(pd.concat([Xb, Xn[chosen]], axis=1))
            rep, passed = _report(f"top{len(chosen)}", ic_sel)

    print(f"\n[inc] ===== 判定 =====")
    print(f"[inc] 基线 judge IC = {ic_a.mean():+.4f}")
    print(f"[inc] ΔIC={rep['mean_delta']:+.5f}  t={rep['t']:+.2f}  "
          f"阈值 t≥{args.t_threshold} 且 ΔIC≥{args.min_delta}")
    verdict = "本批有效，可累积入库" if passed else "本批无显著增量，作废（不改基线）"
    print(f"[inc] 结论：{verdict}", flush=True)

    if passed and args.apply:
        keep_names = [n["name"] for n in new
                      if n["name"] in (chosen or [x["name"] for x in new])]
        run_dir = Path(args.out).parent if args.out else (
            ROOT / "artifacts" / "alphaagent" / "harness_runs" / "increment_pipeline")
        run_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.rows[0]).stem or "batch"
        # per-factor 贡献：permutation 场景记各自 drop；全批直过场景统一记整批 ΔIC
        screen_payload = {
            "keep": keep_names,
            "contributions": [{"name": n,
                               "ml_contribution": float(permutation_drop.get(n, rep["mean_delta"]))}
                              for n in keep_names],
        }
        screen_p = run_dir / f"{stem}.ml_screen.json"
        screen_p.write_text(json.dumps(screen_payload, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts" / "promote_ml_increment.py"),
               "--screen", str(screen_p), "--rows", *args.rows, "--apply"]
        print(f"[inc] 自动入库：{' '.join(str(c) for c in cmd)}", flush=True)
        _pr = subprocess.run(cmd, check=False)
        # 入库成功 → 缓存追加通过列（基线上移），下一批自动以新基线判定
        if _pr.returncode == 0 and cache_path is not None and Xb is not None:
            add = {n: Xn[n] for n in keep_names if n in Xn.columns}
            if add:
                buf = pd.concat([Xb, pd.DataFrame(add)], axis=1)
                buf = buf.loc[:, ~buf.columns.duplicated()]
                buf["__label__"] = lab
                buf["__date__"] = panel.index.get_level_values(0)
                buf["__code__"] = panel.index.get_level_values(1)
                buf.reset_index(drop=True).to_parquet(cache_path)
                cache_path.with_suffix(".meta.json").write_text(json.dumps({
                    "start": args.start, "judge_end": args.judge_end,
                    "base_names_all": sorted({b["name"] for b in base} | set(add))}),
                    encoding="utf-8")
                print(f"[inc] 缓存已追加 {len(add)} 列（基线上移至 {buf.shape[1]}）", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps({
            "n_base": int(Xb.shape[1]), "n_new": int(Xn.shape[1]),
            "ic_base": float(ic_a.mean()), "full": rep_full, "final": rep,
            "chosen": chosen, "passed": passed, "verdict": verdict,
            "train_end": args.train_end, "judge_end": args.judge_end,
        }, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[inc] 报告写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""组合增量审计（只读）：算每个因子的 raw IC 与「对现有因子库组合的边际增量」。

**为什么**
现有候选门是按"单因子能否独立交易"定的（月频 20d：IC>=0.053 / ICIR>=0.65 /
正交<0.5）。但候选池的实际用途是 **ML 样本源**——lgbm 需要的是"组合解释不掉的
增量信息"，不是"单因子够强"。两者会错配：
  - 强但与库内高相关的因子（如 vsi 骨架变体，corr 0.95+）→ 单因子门放行、对组合零增量
  - 弱但正交的因子（如事件面 IC 0.02）→ 单因子门拦死、对组合可能有增量

**本脚本不改任何判定逻辑**，只把两个口径的差异算出来给人看。

**指标**
对每个交易日截面做单变量 OLS（闭式解，不是重训）：
    candidate_t = α_t + β_t · combo_score_t + ε_t
    raw_ic        = corr(candidate_t, label_t)        截面相关，按日平均
    resid_ic      = corr(ε_t, label_t)                去组合后剩余 alpha
    increment     = resid_ic / |raw_ic|               组合解释不掉的比例
    combo_corr    = corr(candidate_t, score_t)        与组合的相关性

**combo_score 怎么来（关键设计）**
**不用** `data/stock/pred_demo.parquet`：它只有 2025-01 起（盲测段），在盲测段上
做因子筛选就是数据窥视。这里用**候选池因子自身在 train 段等权合成**（各因子截面
zscore 后取均值），语义 = "现有因子库的公共信息"。
评估库内某个因子时用 **leave-one-out**（从合成里剔除它自己），否则它作为控制变量
会把残差压扁、系统性低估自己的增量。

成本：每因子一次截面回归 + 两次截面相关（全向量化），0.5-1s；不重训模型。

用法：
    python scripts/audit_combo_increment.py                       # 审计候选池
    python scripts/audit_combo_increment.py --expr-file x.json    # 审计指定因子（新因子用全库合成）
    python scripts/audit_combo_increment.py --limit 10
    python scripts/audit_combo_increment.py --out result.json
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


def _xs_corr(a: pd.Series, b: pd.Series) -> float:
    """按日期截面 Pearson 相关，返回按日均值。全向量化（无 apply）。"""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 100:
        return float("nan")
    g = df.groupby(level=0)
    n = g["a"].size().astype(float)
    sa, sb = g["a"].sum(), g["b"].sum()
    sab = (df["a"] * df["b"]).groupby(level=0).sum()
    sa2 = (df["a"] ** 2).groupby(level=0).sum()
    sb2 = (df["b"] ** 2).groupby(level=0).sum()
    cov = sab - sa * sb / n
    va, vb = sa2 - sa ** 2 / n, sb2 - sb ** 2 / n
    corr = (cov / np.sqrt(va * vb)).replace([np.inf, -np.inf], np.nan)
    return float(corr.dropna().mean())


def _residualize(y: pd.Series, x: pd.Series) -> pd.Series:
    """按日期截面做单变量 OLS 取残差（闭式解，全向量化）。"""
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1)
    mask = df["y"].notna() & df["x"].notna()
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if not mask.any():
        return out
    sub = df[mask]
    d = sub.index.get_level_values(0)
    g = sub.groupby(d)
    n = g["y"].size().astype(float)
    sy, sx = g["y"].sum(), g["x"].sum()
    sxy = (sub["y"] * sub["x"]).groupby(d).sum()
    sxx = (sub["x"] ** 2).groupby(d).sum()
    beta = ((sxy - sy * sx / n) / (sxx - sx ** 2 / n)).replace([np.inf, -np.inf], np.nan)
    alpha = sy / n - beta * (sx / n)
    codes = sub.index.get_level_values(0)
    b = beta.reindex(codes).to_numpy()
    a_ = alpha.reindex(codes).to_numpy()
    out.loc[sub.index] = sub["y"].to_numpy() - a_ - b * sub["x"].to_numpy()
    return out


def _cs_zscore(s: pd.Series) -> pd.Series:
    """按日期截面 zscore。"""
    d = s.index.get_level_values(0)
    g = s.groupby(d)
    mu = g.transform("mean")
    sd = g.transform("std")
    return ((s - mu) / sd.replace(0, np.nan)).astype("float32")


def _load_candidates(args) -> list[dict]:
    if args.expr_file:
        return json.loads(Path(args.expr_file).read_text(encoding="utf-8"))
    reg = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json"
    data = json.loads(reg.read_text(encoding="utf-8"))
    out = []
    for name, rec in data.items():
        expr = rec.get("expr") or rec.get("expression")
        if not expr:
            continue
        m = rec.get("metrics") or {}
        out.append({"name": name, "expr": expr,
                    "stored_ic": m.get("ic"), "stored_icir": m.get("icir")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr-file", default=None, help="待审因子；缺省=候选池自身（此时对每个因子用 LOO 组合）")
    ap.add_argument("--combo-pool", default=None,
                    help="用于合成 combo_score 的因子集；缺省=候选池。审计新因子时必须保持候选池，"
                         "否则 combo 会包含被审因子自己")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", default="2020-01-01", help="train 段起点（盲测段 2025+ 不参与）")
    ap.add_argument("--end", default="2024-12-31", help="train 段终点（默认到 2024 末，避开盲测）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names

    targets_all = _load_candidates(args)
    if args.limit:
        targets_all = targets_all[: args.limit]
    # combo 池：默认候选池（审计候选池自己时即 targets，用 LOO 区分）
    if args.combo_pool:
        pool = json.loads(Path(args.combo_pool).read_text(encoding="utf-8"))
    elif args.expr_file:
        pool = _load_candidates(argparse.Namespace(expr_file=None))
    else:
        pool = targets_all

    # 跳过引用 funda_*/fsi_ 的因子：本机没有 fundamentals 缓存，include_fundamentals=True
    # 会触发整份 8GB 重建（实测 463s + 数 GB 内存）。
    def _split(items):
        keep, skip = [], []
        for c in items:
            try:
                refs = set(referenced_column_names(c["expr"]))
            except Exception:  # noqa: BLE001
                skip.append((c["name"], "expr_parse_fail"))
                continue
            if any(str(r).startswith(("funda_", "fsi_")) for r in refs):
                skip.append((c["name"], "needs_fundamentals"))
                continue
            keep.append(c)
        return keep, skip

    cands, skipped = _split(targets_all)
    pool_ok, pool_skip = _split(pool)
    pool_names = {c["name"] for c in pool_ok}
    n_funda = sum(1 for _, r in skipped if r == "needs_fundamentals")
    print(f"[audit] 待审 {len(cands)} 个（跳过 {n_funda} 个需基本面列）；"
          f"combo 池 {len(pool_ok)} 个", flush=True)

    # 求值集合 = 待审 ∪ combo 池（去重保序）
    eval_list: list[dict] = []
    _seen: set[str] = set()
    for c in list(pool_ok) + list(cands):
        if c["name"] in _seen:
            continue
        _seen.add(c["name"])
        eval_list.append(c)

    need: set[str] = {LABEL}
    for c in eval_list:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start=args.start, end=args.end,
                                include_fundamentals=False,
                                keep_columns=tuple(sorted(need)))
    print(f"[audit] panel {panel.shape} load={time.perf_counter()-t0:.0f}s "
          f"窗口 {args.start}~{args.end}", flush=True)
    if LABEL not in panel.columns:
        print(f"[audit] 面板缺 {LABEL}")
        return 1
    label = panel[LABEL].astype("float64")

    # 1) 求值 → 标准化
    print("[audit] 求值因子值矩阵...", flush=True)
    zs: dict[str, pd.Series] = {}
    for i, c in enumerate(eval_list, 1):
        try:
            v = eval_multi_line_factor(c["expr"], panel)
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
            s = pd.Series(v).astype("float64").reindex(panel.index)
            zs[c["name"]] = _cs_zscore(s)
            tag = "pool" if c["name"] in pool_names else "new "
            print(f"  [{i:>3}/{len(eval_list)}] {tag} {c['name'][:32]:32s} "
                  f"notna={s.notna().mean()*100:5.1f}%", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:>3}/{len(eval_list)}] ERR {c['name'][:32]:32s} "
                  f"{type(exc).__name__}: {str(exc)[:70]}", flush=True)

    pool_z = {n: z for n, z in zs.items() if n in pool_names}
    if not pool_z:
        print("[audit] combo 池为空，无法合成")
        return 1
    Zp = pd.DataFrame(pool_z)
    p_sum = Zp.sum(axis=1, skipna=True)
    p_cnt = Zp.notna().sum(axis=1)
    print(f"[audit] combo 合成: {Zp.shape[1]} 个因子", flush=True)

    # 2) 逐因子：combo（库内做 LOO）→ 残差化 → 指标
    rows = []
    for i, c in enumerate(cands, 1):
        name = c["name"]
        if name not in zs:
            rows.append({**c, "raw_ic": None, "resid_ic": None, "increment": None,
                         "combo_corr": None, "error": "eval_failed"})
            continue
        t = time.perf_counter()
        z = zs[name]
        if name in pool_z and Zp.shape[1] > 1:
            # 库内因子：从 combo 里剔除它自己（LOO），否则低估自身增量
            s_combo = (p_sum - z.fillna(0)) / (p_cnt - z.notna().astype(int)).replace(0, np.nan)
            loo = True
        else:
            s_combo = p_sum / p_cnt.replace(0, np.nan)
            loo = False
        raw_ic = _xs_corr(z, label)
        if s_combo.notna().sum() < 1000:
            rows.append({**c, "raw_ic": raw_ic, "resid_ic": None, "increment": None,
                         "combo_corr": None, "error": "no_combo", "loo": loo})
            continue
        combo_corr = _xs_corr(z, s_combo)
        resid = _residualize(z, s_combo)
        resid_ic = _xs_corr(resid, label)
        inc = (abs(resid_ic) / abs(raw_ic)) if (raw_ic and not np.isnan(raw_ic)
                                               and not np.isnan(resid_ic) and raw_ic != 0) else None
        rows.append({**c, "raw_ic": raw_ic, "resid_ic": resid_ic, "increment": inc,
                     "combo_corr": combo_corr, "loo": loo, "secs": time.perf_counter() - t})
        print(f"  [{i:>3}/{len(cands)}] {name[:32]:32s} raw={raw_ic:+.4f} "
              f"resid={resid_ic:+.4f} inc={inc if inc is None else round(inc, 3)} "
              f"ccorr={combo_corr:+.3f}{' LOO' if loo else ''} "
              f"{time.perf_counter()-t:.1f}s", flush=True)

    ok = [r for r in rows if r.get("increment") is not None]
    print(f"\n[audit] 有效 {len(ok)}/{len(rows)}")
    if ok:
        ok.sort(key=lambda r: -r["increment"])
        print(f"\n{'因子':36s} {'raw_ic':>8s} {'resid_ic':>9s} {'增量比':>8s} {'与组合相关':>10s}")
        for r in ok:
            print(f"{r['name'][:36]:36s} {r['raw_ic']:+8.4f} {r['resid_ic']:+9.4f} "
                  f"{r['increment']:8.3f} {r['combo_corr']:+10.3f}")
        print("\n解读：增量比 >0.5 = 组合解释不掉的新信息；<0.2 = 与库内高度重叠的重复料")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"\n[audit] 明细写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

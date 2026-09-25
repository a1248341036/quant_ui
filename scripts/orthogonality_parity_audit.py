#!/usr/bin/env python3
"""正交判定口径一致性 + 盲测隔离影响 审计（Phase 1，只读）。

回答两个问题（见 docs/specs/alphaagent_mine_precheck_spec.md §3.1 / §7-Q1、Q3）：

- **Q1 口径差异**：三套正交口径对同一批因子对的读数差异
  - A：`_orthogonality_check` —— 5 锚点 × 20 连续交易日，**Spearman**（offline hook 判决口径）
  - B：`SimilarityMatrix` —— zoo `sample_row_ids`（10 万行），**Pearson**（stage_one 判决口径）
  - C：`_candidate_registry_similarity` —— 全量逐日截面，**Pearson 均值**（仅报告）
- **Q3 隔离影响**：相似度数据域从「全区间」收窄到「train ∪ val（可见区间）」后，
  多少存量准入结论翻转（stage_one 正式库阈值 0.4 / offline hook 阈值 0.7）

同时校验 registry 存量 `similarity` 字段是否可信。

只读；产物落 ``<data-root>/artifacts/alphaagent/reports/orthogonality_parity_<ts>.json``。

用法::

    .venv\\Scripts\\python.exe scripts/orthogonality_parity_audit.py --data-root D:\\Quant\\quant_ui
    .venv\\Scripts\\python.exe scripts/orthogonality_parity_audit.py --data-root . --no-full-c
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VISIBLE_END = "2024-12-31"     # = val_end；盲测段自 2025-01-01 起
STAGE_ONE_MAX_CORR = 0.4       # ProductionCriteria.max_abs_corr
CANDIDATE_MAX_CORR = 0.5       # CandidateCriteria.max_abs_corr
ORTHO_MAX_CORR = 0.7           # _ORTHO_MAX_CORR
ORTHO_N_DATES = 5
ORTHO_BLOCK_DAYS = 20
MIN_PAIRS = 30


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── 数据加载 ────────────────────────────────────────────────────────────


def load_panel(data_root: Path, start: str, end: str, include_fundamentals: bool) -> pd.DataFrame:
    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    t0 = time.perf_counter()
    panel = load_panel_from_cne(
        start=start, end=end, universe_mask=False,
        include_fundamentals=include_fundamentals, asset_type="stock",
    )
    print(f"[panel] shape={panel.shape} in {time.perf_counter() - t0:.1f}s", flush=True)
    return panel


def materialize_candidates(registry: dict, panel: pd.DataFrame) -> tuple[dict, list]:
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel

    out: dict[str, np.ndarray] = {}
    failures: list[dict] = []
    t0 = time.perf_counter()
    items = sorted(registry.items())
    for i, (fid, entry) in enumerate(items, 1):
        expr = str((entry or {}).get("expr") or "").strip()
        if not expr:
            failures.append({"factor_id": fid, "reason": "empty_expr"})
            continue
        try:
            raw = eval_factor(expr, panel)
            out[fid] = np.asarray(align_series_to_panel(raw, panel), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            failures.append({"factor_id": fid, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
        print(f"[materialize] {i}/{len(items)} {fid} ok={len(out)} fail={len(failures)} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return out, failures


def build_sample_matrix(cand_values: dict, prod_ids: list, zoo_prod) -> tuple[np.ndarray, pd.DatetimeIndex, list]:
    """口径 B 的输入矩阵：行 = 候选 + 正式库，列 = zoo.sample_row_ids。

    候选值在 panel 行序上，`align_values_to_rows` 负责对齐到采样行（panel 未覆盖的
    更早年份自然成为 NaN，由 min_pairs 掩码剔除）；正式库因子直接用库内采样摘要。
    """
    from alphaagent.factor.ingest import align_values_to_rows

    rows = zoo_prod.index.rows
    sample_rows = rows.iloc[zoo_prod.index.sample_row_ids]
    sample_dates = pd.DatetimeIndex(pd.to_datetime(sample_rows["datetime"]))

    fids = sorted(cand_values)
    blocks = []
    for fid in fids:
        s = pd.Series(np.asarray(cand_values[fid], dtype=np.float64), index=_PANEL_INDEX)
        blocks.append(np.asarray(align_values_to_rows(s, sample_rows), dtype=np.float64))

    summaries, order = zoo_prod.read_sample_summaries()
    pos = {f: i for i, f in enumerate(order)}
    for p in prod_ids:
        if p in pos:
            blocks.append(np.asarray(summaries[pos[p]], dtype=np.float64))

    return np.vstack(blocks), sample_dates, fids


_PANEL_INDEX: pd.MultiIndex | None = None


# ── 相关（NaN 感知）────────────────────────────────────────────────────


def pairwise_corr(matrix: np.ndarray, min_pairs: int = MIN_PAIRS) -> np.ndarray:
    F = matrix.shape[0]
    X = np.asarray(matrix, dtype=np.float64)
    out = np.full((F, F), np.nan)
    for i in range(F):
        xi = X[i]
        mi = np.isfinite(xi)
        for j in range(i, F):
            if i == j:
                out[i, j] = 1.0
                continue
            xj = X[j]
            m = mi & np.isfinite(xj)
            if int(m.sum()) < min_pairs:
                continue
            a = xi[m] - xi[m].mean()
            b = xj[m] - xj[m].mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            out[i, j] = out[j, i] = (float((a * b).sum() / den) if den > 0 else np.nan)
    return out


def _avg_rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def pairwise_spearman(matrix: np.ndarray, min_pairs: int = MIN_PAIRS) -> np.ndarray:
    if matrix.shape[1] < min_pairs:
        return np.full((matrix.shape[0],) * 2, np.nan)
    ranked = np.vstack([_avg_rank(matrix[i]) for i in range(matrix.shape[0])])
    return pairwise_corr(ranked, min_pairs=min_pairs)


def full_day_corr(matrix: np.ndarray, index: pd.MultiIndex) -> np.ndarray:
    """口径 C：全量逐日截面 Pearson 均值（逐日成对剔除 NaN）。

    注意：不能用"当日全部因子非 NaN 才计算"的实现——真实因子的 coverage 不全，
    该口径会把**每一天**都跳过（实测 days used=0）。这里用 pandas 的成对完整观测
    相关（min_periods 兜底），逐日累计。
    """
    from alphaagent.factor.metrics import _day_slices

    F = matrix.shape[0]
    acc = np.zeros((F, F))
    cnt = np.zeros((F, F))
    Xd = np.asarray(matrix, dtype=np.float64)
    slices = _day_slices(index)
    if slices is None:
        print("[full_c] _day_slices unavailable -> skip", flush=True)
        return np.full((F, F), np.nan)
    bounds = np.asarray(slices[0])
    n_days = 0
    for st, en in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        block = Xd[:, st:en]
        if block.shape[1] < MIN_PAIRS:
            continue
        valid_rows = np.isfinite(block).sum(axis=1) >= MIN_PAIRS
        if int(valid_rows.sum()) < 2:
            continue
        sub = block[valid_rows]
        c = pd.DataFrame(sub.T).corr(min_periods=MIN_PAIRS).to_numpy()
        ok = np.isfinite(c)
        if not ok.any():
            continue
        idx_ok = np.flatnonzero(valid_rows)
        acc[np.ix_(idx_ok, idx_ok)] += np.where(ok, c, 0.0)
        cnt[np.ix_(idx_ok, idx_ok)] += ok.astype(np.float64)
        n_days += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    np.fill_diagonal(out, 1.0)
    print(f"[full_c] days used={n_days}", flush=True)
    return out


# ── 判决翻转 ────────────────────────────────────────────────────────────


def flip(full: float, visible: float, threshold: float) -> str:
    f_pass, v_pass = full < threshold, visible < threshold
    if f_pass and not v_pass:
        return "pass_to_fail"
    if (not f_pass) and v_pass:
        return "fail_to_pass"
    return "unchanged"


def max_vs(corr: np.ndarray, row: int, cols: list[int]) -> float:
    if not cols:
        return 0.0
    vals = np.abs(corr[row, cols])
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else 0.0


def max_offdiag(corr: np.ndarray, row: int, n_cand: int) -> float:
    vals = np.abs(corr[row, :n_cand])
    vals = np.delete(vals, row)
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else 0.0


def dist_stats(deltas: list[float]) -> dict:
    if not deltas:
        return {}
    a = np.asarray(deltas, dtype=np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 6),
            "median": round(float(np.median(a)), 6),
            "p10": round(float(np.percentile(a, 10)), 6),
            "p90": round(float(np.percentile(a, 90)), 6),
            "max_abs": round(float(np.max(np.abs(a))), 6)}


def counts(rows: list[dict]) -> dict:
    c: dict[str, int] = {}
    for r in rows:
        c[r["verdict"]] = c.get(r["verdict"], 0) + 1
    return c


# ── 主流程 ──────────────────────────────────────────────────────────────


def main() -> int:
    global _PANEL_INDEX
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--no-full-c", action="store_true", help="跳过全量逐日口径 C（最贵）")
    ap.add_argument("--include-fundamentals", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--panel-end", default="2026-09-17",
                    help="panel 右端（默认对齐现有全列缓存以命中，避免重建 7.7M 行面板）")
    ap.add_argument("--panel-parquet", default="",
                    help="直接读现有 panel parquet（省去 100s+ 重建；平表或 MultiIndex 均可）")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    t0 = time.perf_counter()

    from alphaagent.factor.zoo import FactorZoo

    reg_path = data_root / "artifacts/alphaagent/factorzoo/candidate_main/mining_candidate_registry.json"
    registry_all = json.loads(reg_path.read_text(encoding="utf-8"))
    registry = dict(sorted(registry_all.items())[: args.limit]) if args.limit else registry_all
    print(f"[input] candidates={len(registry)}/{len(registry_all)}", flush=True)

    if args.panel_parquet:
        t_p = time.perf_counter()
        raw = pd.read_parquet(args.panel_parquet)
        if not isinstance(raw.index, pd.MultiIndex):
            raw = raw.set_index(["datetime", "instrument"]).sort_index()
        panel = raw
        print(f"[panel] from parquet shape={panel.shape} in {time.perf_counter() - t_p:.1f}s", flush=True)
    else:
        panel = load_panel(data_root, "2020-01-01", args.panel_end, args.include_fundamentals)
    _PANEL_INDEX = panel.index
    idx = panel.index
    dt = pd.DatetimeIndex(idx.get_level_values("datetime"))
    vis_mask = np.asarray(dt <= pd.Timestamp(VISIBLE_END))
    print(f"[mask] rows={len(dt)} visible={int(vis_mask.sum())} dropped={int((~vis_mask).sum())}", flush=True)

    cand_values, failures = materialize_candidates(registry, panel)
    if not cand_values:
        print("[abort] no candidate materialized", flush=True)
        return 2

    zoo_prod = FactorZoo.open(data_root / "artifacts/alphaagent/factorzoo/production_main")
    prod_ids = zoo_prod.catalog.list_factor_ids()
    print(f"[prod] {prod_ids}", flush=True)

    res: dict = {"meta": {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "n_candidates_total": len(registry_all), "n_candidates_evaluated": len(cand_values),
        "n_production": len(prod_ids),
        "panel_shape": list(panel.shape),
        "include_fundamentals": bool(args.include_fundamentals),
        "visible_end": VISIBLE_END,
        "rows_full": int(len(dt)), "rows_visible": int(vis_mask.sum()),
        "rows_dropped": int((~vis_mask).sum()),
        "thresholds": {"stage_one_production": STAGE_ONE_MAX_CORR,
                       "candidate": CANDIDATE_MAX_CORR, "ortho_hook": ORTHO_MAX_CORR},
    }, "materialize_failures": failures}

    # ── 口径 B（stage_one 判决口径）：采样行 Pearson ──
    matB, sample_dates, b_fids = build_sample_matrix(cand_values, prod_ids, zoo_prod)
    nB = len(b_fids)
    b_vis = np.asarray(sample_dates <= pd.Timestamp(VISIBLE_END))
    corr_B_full = pairwise_corr(matB)
    corr_B_vis = pairwise_corr(matB[:, b_vis]) if int(b_vis.sum()) >= MIN_PAIRS else np.full((nB, nB), np.nan)
    prod_cols = list(range(nB, matB.shape[0]))
    print(f"[B] matrix={matB.shape} visible_cols={int(b_vis.sum())} dropped={int((~b_vis).sum())}", flush=True)

    rows_stage = []
    for k, fid in enumerate(b_fids):
        vf = max_vs(corr_B_full, k, prod_cols)
        vv = max_vs(corr_B_vis, k, prod_cols)
        rows_stage.append({"factor_id": fid, "full": round(vf, 6), "visible": round(vv, 6),
                           "delta": round(vv - vf, 6), "threshold": STAGE_ONE_MAX_CORR,
                           "verdict": flip(vf, vv, STAGE_ONE_MAX_CORR)})
    res["stage_one_production_B"] = {"counts": counts(rows_stage), "pairs": rows_stage}

    # ── 口径 A（offline hook 判决口径）：5×20 日 Spearman ──
    dates = pd.DatetimeIndex(dt.unique()).sort_values()
    required = ORTHO_N_DATES * ORTHO_BLOCK_DAYS
    if len(dates) >= required:
        rng = np.random.default_rng(42)
        anchors = rng.choice(np.arange(ORTHO_BLOCK_DAYS - 1, len(dates)), size=ORTHO_N_DATES, replace=False)
        sel: set = set()
        for a in anchors:
            sel.update(dates[max(0, int(a) - ORTHO_BLOCK_DAYS + 1): int(a) + 1])
        a_mask = np.asarray(dt.isin(sel))
    else:
        a_mask = np.ones(len(dt), dtype=bool)
    a_vis = a_mask & vis_mask

    matrix = np.vstack([cand_values[f] for f in sorted(cand_values)])
    fids = sorted(cand_values)
    n_cand = len(fids)
    corr_A_full = pairwise_spearman(matrix[:, a_mask])
    corr_A_vis = pairwise_spearman(matrix[:, a_vis])
    print(f"[A] block_rows={int(a_mask.sum())} visible_block_rows={int(a_vis.sum())}", flush=True)

    rows_hook = []
    for k, fid in enumerate(fids):
        hf = max(max_vs(corr_B_full, k, prod_cols), max_offdiag(corr_A_full, k, n_cand))
        hv = max(max_vs(corr_B_vis, k, prod_cols), max_offdiag(corr_A_vis, k, n_cand))
        rows_hook.append({"factor_id": fid, "full": round(hf, 6), "visible": round(hv, 6),
                          "delta": round(hv - hf, 6), "threshold": ORTHO_MAX_CORR,
                          "verdict": flip(hf, hv, ORTHO_MAX_CORR)})
    res["offline_hook_A"] = {"counts": counts(rows_hook), "pairs": rows_hook}

    # ── 口径 C：全量逐日（仅候选池内部，报告口径）──
    if args.no_full_c:
        corr_C_full = corr_C_vis = np.full((n_cand, n_cand), np.nan)
    else:
        corr_C_full = full_day_corr(matrix, idx)
        corr_C_vis = full_day_corr(matrix[:, vis_mask], idx[vis_mask])

    # ── 口径差异：A vs C、B vs C（候选池内部两两）──
    d_ac, d_bc = [], []
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            a, c = corr_A_full[i, j], corr_C_full[i, j]
            b = corr_B_full[i, j]
            if np.isfinite(a) and np.isfinite(c):
                d_ac.append(float(c - a))
            if np.isfinite(b) and np.isfinite(c):
                d_bc.append(float(c - b))
    res["parity"] = {"A_vs_C_delta": dist_stats(d_ac), "B_vs_C_delta": dist_stats(d_bc)}

    # ── registry 存量 similarity 陈旧性 ──
    stored = [{"factor_id": fid, "stored_max_abs_corr": ((e or {}).get("similarity") or {}).get("max_abs_corr"),
               "stored_basis": ((e or {}).get("similarity") or {}).get("basis")}
              for fid, e in registry.items()]
    res["stored_similarity_audit"] = {
        "n": len(stored),
        "all_zero": all((s["stored_max_abs_corr"] or 0) == 0 for s in stored),
        "bases": sorted({str(s["stored_basis"]) for s in stored}),
        "note": "候选入库时正式库为空 -> 存量 similarity 恒 0，不能作为重筛判决依据；"
                "rescreen 用它做 corr 判定 = 该判据对存量失效",
    }
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)

    out_dir = data_root / "artifacts/alphaagent/reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"orthogonality_parity_{_ts()}.json"
    jp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== stage_one (B 口径, 候选 vs 正式库, 阈值 0.4) ===", flush=True)
    print(json.dumps(res["stage_one_production_B"]["counts"], ensure_ascii=False), flush=True)
    print("=== offline hook (A 口径, 阈值 0.7) ===", flush=True)
    print(json.dumps(res["offline_hook_A"]["counts"], ensure_ascii=False), flush=True)
    print("=== 口径差 A vs C ===", flush=True)
    print(json.dumps(res["parity"]["A_vs_C_delta"], ensure_ascii=False), flush=True)
    print("=== 口径差 B vs C ===", flush=True)
    print(json.dumps(res["parity"]["B_vs_C_delta"], ensure_ascii=False), flush=True)
    print(f"\n[done] {jp}  elapsed={res['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

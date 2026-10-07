#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢档门槛标定测量：同一批因子在 label_1d / label_5d / label_20d 上做**配对** train 评估，
输出 |IC| / |ICIR| 的配对中位比值，供 `core/research_modes.py` 慢档门槛等比放大使用。

为什么需要：`min_abs_ic` / `min_icir` / `min_val_abs_ic` 按 label_1d 尺度标定，换到
label_5d / label_20d 后同一因子的 |IC|、|ICIR| 会系统性变大；沿用 1d 线会让慢档门近乎失效。
按配对比值等比放大可让各档在同一因子池上的过线率相当。

用法（只读：不落库、不改 registry，只创建会话并调 eval_train）：
    .venv\\Scripts\\python.exe scripts\\calibrate_slow_tier_thresholds.py            # 全量 42 因子
    .venv\\Scripts\\python.exe scripts\\calibrate_slow_tier_thresholds.py --limit 10 # 快速冒烟
    .venv\\Scripts\\python.exe scripts\\calibrate_slow_tier_thresholds.py --out rows.jsonl

2026-10-07 实测（42 因子 × 3 label，n=41 有效配对，train 2020-2022，约 35 分钟）：
    label_5d/1d  = |IC| 1.776 / |ICIR| 1.800
    label_20d/1d = |IC| 2.865 / |ICIR| 3.032
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.data.adapters.cnequity import CNE_SOURCE  # noqa: E402
from alphaagent.factor.mining.eval.schemas import (  # noqa: E402
    EvalTrainRequest,
    SessionCreateRequest,
)
from alphaagent.factor.mining.registry_io import load_mining_registry  # noqa: E402
from alphaagent.factor.mining.service import StockEvalService  # noqa: E402
from core import factor_categories  # noqa: E402

LABEL_MAP = {
    "1d": "label_1d_open_to_open",
    "5d": "label_5d_close_to_close",
    "20d": "label_20d_close_to_close",
}


def pick_exprs(limit: int) -> list[tuple[str, str]]:
    """因子池：候选池现有表达式（真实、跨族；含基本面因子故会话需带 funda 列）。"""
    reg = load_mining_registry(factor_categories.candidate_registry_path("technical"))
    out = [(fid, str(e.get("expr") or "").strip())
           for fid, e in reg.items() if isinstance(e, dict) and str(e.get("expr") or "").strip()]
    out.sort(key=lambda t: t[0])
    return out[:limit]


def extract(res: dict) -> dict:
    """eval_train 返回 → (ic, icir, coverage)。指标在 summary 里。"""
    s = res.get("summary") if isinstance(res, dict) else None
    s = s if isinstance(s, dict) else {}
    return {"ic": s.get("ic"), "icir": s.get("icir"), "coverage": s.get("factor_coverage")}


def main() -> int:
    ap = argparse.ArgumentParser(description="慢档门槛标定测量（配对设计）")
    ap.add_argument("--limit", type=int, default=42, help="因子数上限")
    ap.add_argument("--labels", default="1d,5d,20d", help="参与测量的 label 档")
    ap.add_argument("--out", default="", help="逐行结果写出路径（JSONL，可选）")
    args = ap.parse_args()

    labels = [LABEL_MAP[x.strip()] for x in args.labels.split(",") if x.strip() in LABEL_MAP]
    exprs = pick_exprs(args.limit)
    print(f"[calib] 因子池 {len(exprs)} 个; labels={labels}", flush=True)

    svc = StockEvalService(max_parallel_eval=1)
    rows: list[dict] = []
    for label in labels:
        t0 = time.perf_counter()
        resp = svc.create_session(SessionCreateRequest(
            panel_path=str(CNE_SOURCE), label_col=label, include_fundamentals=True,
        ))
        sid = resp.session_id
        print(f"[calib] session={sid} label={label} rows={resp.panel_rows}", flush=True)
        try:
            for i, (fid, expr) in enumerate(exprs, 1):
                t = time.perf_counter()
                try:
                    got = extract(svc.eval_train(EvalTrainRequest(
                        session_id=sid, multi_line_expr=expr, factor_name=fid,
                        include_detail_tables=False, label_quantile_n=10,
                    )))
                    err = None
                except Exception as exc:  # noqa: BLE001 — 单因子失败不中断标定
                    got, err = {"ic": None, "icir": None, "coverage": None}, f"{type(exc).__name__}: {exc}"
                rows.append({"label": label, "factor": fid,
                             "ms": round((time.perf_counter() - t) * 1000), "err": err, **got})
                print(f"  [{label}] {i:3d}/{len(exprs)} {fid:44s} ic={got['ic']} icir={got['icir']} "
                      f"{rows[-1]['ms']}ms {err or ''}", flush=True)
        finally:
            svc.release_session(sid)
        print(f"[calib] label={label} 用时 {time.perf_counter() - t0:.0f}s", flush=True)

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"[calib] 明细已写 {args.out}", flush=True)

    # ── 汇总：各档分位 + 配对比值（标定依据）──
    print("\n=== 各 label 的 |IC| / |ICIR| 分布（配对样本）===")
    for lab in labels:
        rs = [r for r in rows if r["label"] == lab and r.get("ic") is not None]
        ic = sorted(abs(float(r["ic"])) for r in rs)
        ir = sorted(abs(float(r["icir"])) for r in rs if r.get("icir") is not None)

        def q(v, p):
            return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")

        print(f"  {lab:24s} n={len(rs):3d} |IC| p50={q(ic, .5):.4f} p90={q(ic, .9):.4f} | "
              f"|ICIR| p50={q(ir, .5):.4f} p90={q(ir, .9):.4f}")

    base = {r["factor"]: r for r in rows if r["label"] == labels[0]}
    print(f"\n=== 配对比值 vs {labels[0]}（中位数 = 标定放大系数）===")
    for lab in labels[1:]:
        r_ic, r_ir = [], []
        for r in rows:
            if r["label"] != lab:
                continue
            b = base.get(r["factor"])
            if not b or r.get("ic") is None or b.get("ic") in (None, 0):
                continue
            if abs(float(b["ic"])) > 1e-6:
                r_ic.append(abs(float(r["ic"])) / abs(float(b["ic"])))
            if r.get("icir") is not None and b.get("icir") not in (None, 0) and abs(float(b["icir"])) > 1e-6:
                r_ir.append(abs(float(r["icir"])) / abs(float(b["icir"])))
        print(f"  {lab:24s} n_ic={len(r_ic):3d} median|IC|ratio={statistics.median(r_ic) if r_ic else float('nan'):.3f} | "
              f"n_icir={len(r_ir):3d} median|ICIR|ratio={statistics.median(r_ir) if r_ir else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
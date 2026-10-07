#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness 自驱动挖掘 · 真实交付（把自研因子走完整交付链路入库）。

与 `agentscope_run` 构造 `FactorSubmitService` 的参数完全一致（research_mode /
max_cs_corr / delivery_policy / similar_top_k / overwrite），因此判定口径与真实 run 相同：
stage_one（统计门槛）→ 盲测（test 段）→ stage_two（精筛 + 库内正交）→ engine_gate（净值回测）。
stage_one 通过即 `candidate_stored=true`（候选写入 `artifacts/.../candidate_main/`）。

用法：
    .venv\\Scripts\\python.exe scripts\\harness_submit.py --mode technical_monthly --factors winners.json

factors JSON 格式：
    [{"name": "...", "expr": "多行 DSL", "comment": "机制与经济直觉（必填）",
      "evidence": {"train": {...}, "val": {...}}}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.data.adapters.cnequity import CNE_SOURCE  # noqa: E402
from alphaagent.factor.mining.delivery.submit import FactorSubmitService  # noqa: E402
from alphaagent.factor.mining.eval.schemas import SessionCreateRequest  # noqa: E402
from alphaagent.factor.mining.research_spec import effective_research_spec  # noqa: E402
from alphaagent.factor.mining.service import StockEvalService  # noqa: E402

MODE_LABEL = {
    "technical": "label_1d_open_to_open",
    "technical_weekly": "label_5d_close_to_close",
    "technical_monthly": "label_20d_close_to_close",
    "fundamental": "label_20d_close_to_close",
}
MODE_FREQ = {
    "technical": "daily",
    "technical_weekly": "weekly",
    "technical_monthly": "monthly",
    "fundamental": "monthly",
}
_KEEP = ("ok", "stored", "candidate_stored", "candidate_storage", "skipped_reason", "error",
         "error_type", "promotion_status", "review_status", "factor_id", "rebalance_freq",
         "delivery_check", "test_holdout", "candidate_similarity", "engine_backtest")


def main() -> int:
    ap = argparse.ArgumentParser(description="Harness 挖掘：真实交付入库")
    ap.add_argument("--mode", default="technical_monthly", choices=sorted(MODE_LABEL))
    ap.add_argument("--factors", required=True, help="JSON: [{name, expr, comment, evidence?}, ...]")
    ap.add_argument("--max-cs-corr", type=float, default=0.8, help="离线正交门（与 MiningConfig 默认一致）")
    ap.add_argument("--similar-top-k", type=int, default=3)
    ap.add_argument("--out", default="", help="提交结果写出路径（JSON）")
    ap.add_argument("--fundamentals", action="store_true", help="会话载入 funda_* 列")
    args = ap.parse_args()

    label = MODE_LABEL[args.mode]
    freq = MODE_FREQ[args.mode]
    cands = json.loads(Path(args.factors).read_text(encoding="utf-8"))
    spec = effective_research_spec(args.mode)
    svc = StockEvalService(max_parallel_eval=1)
    t0 = time.perf_counter()
    resp = svc.create_session(SessionCreateRequest(
        panel_path=str(CNE_SOURCE), label_col=label, include_fundamentals=args.fundamentals,
    ))
    sid = resp.session_id
    print(f"[submit] mode={args.mode} label={label} freq={freq} session={sid} "
          f"rows={resp.panel_rows} load={time.perf_counter() - t0:.0f}s", flush=True)
    submitter = FactorSubmitService(
        svc, repo_root=ROOT, research_mode=args.mode, max_cs_corr=args.max_cs_corr,
        delivery_policy=spec.get("delivery_policy") or {}, similar_top_k=args.similar_top_k,
        overwrite=False,
    )
    results = []
    try:
        for c in cands:
            t = time.perf_counter()
            print(f"\n[submit] === {c['name']} ===", flush=True)
            try:
                res = submitter.submit(
                    sid, multi_line_expr=c["expr"], factor_name=c["name"],
                    comment=c.get("comment", ""), evaluation_evidence=c.get("evidence"),
                    rebalance_freq=freq,
                )
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            brief = {k: res.get(k) for k in _KEEP if k in res}
            print(f"[submit] 结果({time.perf_counter() - t:.0f}s): "
                  f"{json.dumps(brief, ensure_ascii=False, default=str)[:900]}", flush=True)
            results.append({"name": c["name"], "result": res})
    finally:
        svc.release_session(sid)
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                  encoding="utf-8")
    stored = sum(1 for r in results if r["result"].get("candidate_stored"))
    print(f"\n[submit] 完成：{len(results)} 个提交，候选入库 {stored} 个，"
          f"用时 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
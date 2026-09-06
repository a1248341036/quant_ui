#!/usr/bin/env python3
"""把研究记忆里的 promising 因子重新走完整交付链（stage_one→盲测→正交→stage_two→engine_gate）。

背景：挖掘 run 的 LLM 偶发"训练过线却不调 submit"（run_end 的
unsubmitted_promising 审计抓到 24 个）。这些因子不在候选池 registry 里，
promote_candidates.py（重放候选池）覆盖不到——本脚本从 research_memory.db
直接取 ``verdict='promising'`` 的条目补走 submit。已被提交拒绝的条目
verdict 已变成 rejected，天然被排除。

用法：
  .venv\\Scripts\\python.exe scripts\\resubmit_promising_factors.py --run-id ab657f24e796 --dry-run
  .venv\\Scripts\\python.exe scripts\\resubmit_promising_factors.py --run-id ab657f24e796
  .venv\\Scripts\\python.exe scripts\\resubmit_promising_factors.py --run-id ab657f24e796 --mode fundamental

说明：
- 与 promote_candidates 同口径：不调 LLM reviewer（review_hook=None），
  stage_one/盲测/正交/stage_two/engine_gate 全链照跑，最终裁决是统计门槛+回测。
- 消融基线腿（*_ablation_*）是 dispatch 的诊断产物，默认跳过
  （--include-ablation 才纳入）。
- 候选池/正式库已有同名或同表达式的因子自动跳过（幂等）。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.factor.mining.config import MiningConfig  # noqa: E402
from alphaagent.factor.mining.context import StockEvalContext  # noqa: E402
from alphaagent.factor.mining.registry_io import load_mining_registry  # noqa: E402
from alphaagent.factor.mining.research_spec import effective_research_spec  # noqa: E402
from alphaagent.factor.mining.schemas import SessionCreateRequest  # noqa: E402
from alphaagent.factor.mining.service import StockEvalService  # noqa: E402
from alphaagent.factor.mining.submit import FactorSubmitService  # noqa: E402
from alphaagent.data.adapters.cnequity import CNE_SOURCE  # noqa: E402
from core import factor_categories  # noqa: E402

MEMORY_DB = ROOT / "artifacts" / "alphaagent" / "research_memory.db"


def _load_promising(run_id: str) -> list[dict]:
    """取指定 run 的 promising 条目（verdict 在 submit 被拒后会更新，天然排除已试过的）。"""
    db = sqlite3.connect(f"file:{MEMORY_DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT factor_name, expression, verdict, metrics_json FROM memory_entries "
        "WHERE last_run_id = ? AND verdict = 'promising' ORDER BY factor_name",
        (run_id,),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def _known_factors(mode: str) -> tuple[set[str], set[str]]:
    """候选池 + 正式库已注册的 (factor_name 集合, 规范化表达式集合)——幂等跳过用。"""
    names: set[str] = set()
    exprs: set[str] = set()
    for path in (factor_categories.candidate_registry_path(mode),
                 factor_categories.production_registry_path(mode)):
        registry = load_mining_registry(Path(path))
        for fid, entry in (registry or {}).items():
            if not isinstance(entry, dict):
                continue
            names.add(str(entry.get("name") or fid).strip().lower())
            exprs.add(" ".join(str(entry.get("expr") or "").split()))
    return names, exprs


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay promising memory factors through delivery chain")
    ap.add_argument("--run-id", required=True, help="研究记忆的 last_run_id")
    ap.add_argument("--mode", choices=list(factor_categories.all_categories()), default="fundamental")
    ap.add_argument("--dry-run", action="store_true", help="只列出将提交的因子，不执行")
    ap.add_argument("--include-ablation", action="store_true",
                    help="纳入消融基线腿（*_ablation_*，默认跳过）")
    args = ap.parse_args()

    spec = effective_research_spec(args.mode)
    label_col = str(spec.get("recommended_label_col") or "label_1d_open_to_open")
    entries = _load_promising(args.run_id)
    known_names, known_exprs = _known_factors(args.mode)

    pending: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for r in entries:
        name = str(r["factor_name"]).strip()
        expr = str(r["expression"]).strip()
        norm = " ".join(expr.split())
        if not args.include_ablation and "_ablation_" in name:
            skipped.append((name, "ablation 基线腿（诊断产物）"))
            continue
        if name.lower() in known_names:
            skipped.append((name, "候选池/正式库已有同名因子"))
            continue
        if norm in known_exprs:
            skipped.append((name, "候选池/正式库已有同表达式"))
            continue
        pending.append(r)

    print(f"run={args.run_id} mode={args.mode} label={label_col}")
    print(f"promising {len(entries)} 条 → 待提交 {len(pending)}，跳过 {len(skipped)}")
    for name, why in skipped:
        print(f"  [跳过] {name}: {why}")
    for r in pending:
        m = json.loads(r["metrics_json"] or "{}")
        print(f"  [待提交] {r['factor_name']}  train_ic={m.get('ic')}")

    if args.dry_run or not pending:
        return 0

    ctx = StockEvalContext(
        panel_path=CNE_SOURCE,
        train_start="2020-01-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2024-12-31",
        label_col=label_col,
        include_fundamentals=(args.mode == "fundamental"),
    )
    cfg = MiningConfig(eval=ctx, research_spec=spec, max_tokens=4096)
    service = StockEvalService(max_parallel_eval=1)
    submit_service = FactorSubmitService(
        service,
        factorlib_path=factor_categories.production_dir(args.mode),
        registry_path=factor_categories.production_registry_path(args.mode),
        expr_dir=factor_categories.production_expr_dir(args.mode),
        repo_root=ROOT,
        research_mode=args.mode,
        delivery_policy=(cfg.research_spec or {}).get("delivery_policy"),
    )

    session_resp = service.create_session(SessionCreateRequest(
        panel_path=str(ctx.panel_path),
        train_start=ctx.train_start,
        train_end=ctx.train_end,
        val_start=ctx.val_start,
        val_end=ctx.val_end,
        label_col=ctx.label_col,
        include_fundamentals=ctx.include_fundamentals,
    ))

    results: list[dict] = []
    try:
        for r in pending:
            name = str(r["factor_name"]).strip()
            expr = str(r["expression"]).strip()
            m = json.loads(r["metrics_json"] or "{}")
            print(f"\n=== submit {name} ===")
            try:
                result = submit_service.submit(
                    session_resp.session_id,
                    multi_line_expr=expr,
                    factor_name=name,
                    comment=(
                        f"重放提交：run {args.run_id} 训练过线（train IC {m.get('ic')}）"
                        "但 LLM 未调用 submit，由 resubmit 脚本补走交付链。"
                    ),
                    evaluation_evidence={
                        "source": "memory_replay",
                        "run_id": args.run_id,
                        "train_ic": m.get("ic"),
                        "train_icir": m.get("icir"),
                    },
                )
            except Exception as exc:  # noqa: BLE001 — 单因子失败不阻断批次
                print(f"  ! 异常 {type(exc).__name__}: {str(exc)[:160]}")
                results.append({"factor": name, "ok": False, "error": str(exc)[:200]})
                continue
            stored = bool(result.get("stored"))
            cand = bool(result.get("candidate_stored"))
            skip = result.get("skipped_reason")
            bt = result.get("delivery_check", {}).get("blind_test") or {}
            print(f"  ok={result.get('ok')} stored={stored} candidate={cand} "
                  f"status={result.get('promotion_status')} skip={skip}")
            if bt:
                print(f"  blind: test_ic={bt.get('test_ic')} retention={bt.get('ic_retention')}")
            if result.get("error"):
                print(f"  error={str(result.get('error'))[:160]}")
            results.append({
                "factor": name,
                "ok": bool(result.get("ok")),
                "stored": stored,
                "candidate_stored": cand,
                "status": result.get("promotion_status"),
                "skipped_reason": skip,
            })
    finally:
        service.release_session(session_resp.session_id)

    stored_n = sum(1 for r in results if r.get("stored"))
    cand_n = sum(1 for r in results if r.get("candidate_stored"))
    print(f"\n==== 完成：{len(results)} 个 | 入正式库 {stored_n} | 入候选池 {cand_n} ====")
    for r in results:
        print(f"  {r['factor']}: stored={r.get('stored')} candidate={r.get('candidate_stored')} "
              f"status={r.get('status')} skip={r.get('skipped_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

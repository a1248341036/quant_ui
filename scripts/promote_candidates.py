#!/usr/bin/env python3
"""Replay candidate-pool factors through the (fixed) two-stage promotion pipeline.

背景：submit.py 历史上把 FactorReviewer 的 `revise` 判定当作硬门槛阻断 stage_two，
同时 stage_two 的相似度检查混入候选池内部冗余（max_abs_corr 取 zoo+candidate 合并
最大值），导致统计达标的候选堆积候选池、正式库长期为空。本脚本把这些已入库的
候选重新走一遍修复后的 submit 链路（revise 不再阻断、stage_two 只看正式库），
统计达标者晋升进正式库。

用法：
  .venv\\Scripts\\python.exe scripts\\promote_candidates.py
  .venv\\Scripts\\python.exe scripts\\promote_candidates.py --mode technical
  .venv\\Scripts\\python.exe scripts\\promote_candidates.py --dry-run

说明：
- 批量重放**不调用 LLM reviewer**（submit 的 review_hook 传 None），
  stage_two 统计门槛 + engine_gate 完整回测才是正式库准入的最终裁决，省成本且无副作用。
- 幂等：仅处理 promotion_status 不是 promoted 的候选；重复运行不会重复晋升已入库因子。
- 候选的 ingest_config（panel/train/val/label）决定会话构建口径，与挖掘时一致。
"""

from __future__ import annotations

import argparse
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
from alphaagent.factor.types import (  # noqa: E402
    DEFAULT_LABEL_COL,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)
from core import factor_categories  # noqa: E402

PROMOTED = {"promoted"}
# 正式库已存在（跳过）或本轮明确不该重放的状态
SKIP_STATES: set[str] = set()


def _mode_config(mode: str) -> MiningConfig:
    spec = effective_research_spec(mode)
    label_col = str(spec.get("recommended_label_col") or DEFAULT_LABEL_COL)
    ctx = StockEvalContext(
        panel_path=CNE_SOURCE,
        train_start=DEFAULT_TRAIN_START,
        train_end=DEFAULT_TRAIN_END,
        val_start=DEFAULT_VAL_START,
        val_end=DEFAULT_VAL_END,
        label_col=label_col,
        include_fundamentals=(mode == "fundamental"),
    )
    return MiningConfig(eval=ctx, research_spec=spec, max_tokens=4096)


def _candidate_dates(entry: dict) -> dict[str, str]:
    """从候选 ingest_config 推断 train/val 切分（缺省回落挖掘默认）。"""
    ic = entry.get("ingest_config") or {}
    train_start = str(ic.get("train_start") or DEFAULT_TRAIN_START)
    val_start = str(ic.get("val_start") or DEFAULT_VAL_START)
    train_end = str(ic.get("train_end") or DEFAULT_TRAIN_END)
    val_end = str(ic.get("val_end") or DEFAULT_VAL_END)
    label = str(ic.get("label_col") or DEFAULT_LABEL_COL)
    return {"train_start": train_start, "train_end": train_end,
            "val_start": val_start, "val_end": val_end, "label_col": label}


def _promote_one(
    service: StockEvalService,
    cfg: MiningConfig,
    mode: str,
    factor_id: str,
    entry: dict,
    *,
    dry_run: bool,
) -> dict:
    dates = _candidate_dates(entry)
    ctx = StockEvalContext(
        panel_path=CNE_SOURCE,
        train_start=dates["train_start"],
        train_end=dates["train_end"],
        val_start=dates["val_start"],
        val_end=dates["val_end"],
        label_col=dates["label_col"],
        include_fundamentals=(mode == "fundamental"),
    )
    cfg.eval = ctx
    root = ROOT
    submit_service = FactorSubmitService(
        service,
        factorlib_path=factor_categories.production_dir(mode),
        registry_path=factor_categories.production_registry_path(mode),
        expr_dir=factor_categories.production_expr_dir(mode),
        repo_root=root,
        research_mode=mode,
        delivery_policy=(cfg.research_spec or {}).get("delivery_policy"),
    )

    if dry_run:
        return {
            "factor_id": factor_id,
            "action": "dry_run",
            "expr_head": str(entry.get("expr") or "")[:120],
            "note": "dry-run：不执行晋升",
        }

    session_resp = service.create_session(
        SessionCreateRequest(
            panel_path=str(ctx.panel_path),
            train_start=ctx.train_start,
            train_end=ctx.train_end,
            val_start=ctx.val_start,
            val_end=ctx.val_end,
            label_col=ctx.label_col,
            include_fundamentals=ctx.include_fundamentals,
        )
    )
    try:
        result = submit_service.submit(
            session_resp.session_id,
            multi_line_expr=str(entry.get("expr") or ""),
            factor_name=str(entry.get("name") or factor_id),
            comment=str(entry.get("comment") or ""),
            evaluation_evidence=entry.get("evaluation_evidence"),
            interaction=entry.get("interaction"),
        )
    finally:
        service.release_session(session_resp.session_id)

    return {
        "factor_id": factor_id,
        "action": "submit",
        "ok": bool(result.get("ok")),
        "stored": bool(result.get("stored")),
        "candidate_stored": bool(result.get("candidate_stored")),
        "promotion_status": result.get("promotion_status"),
        "stage_one": result.get("delivery_check", {}).get("stage_one"),
        "stage_two": result.get("delivery_check", {}).get("stage_two"),
        "skipped_reason": result.get("skipped_reason"),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
        "max_abs_corr": (result.get("similarity") or {}).get("max_abs_corr"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay candidate factors through promotion")
    ap.add_argument("--mode", choices=list(factor_categories.all_categories()), default=None)
    ap.add_argument("--dry-run", action="store_true", help="只列出待晋升候选，不执行")
    ap.add_argument("--all", action="store_true", help="也重放已 stage_two_failed/engine_gate_failed 的候选")
    args = ap.parse_args()

    modes = [args.mode] if args.mode else ["technical", "fundamental"]
    for mode in modes:
        cand_path = factor_categories.candidate_registry_path(mode)
        registry = load_mining_registry(cand_path)
        if not registry:
            print(f"[{mode}] 候选池为空（{cand_path}）")
            continue

        pending = {
            fid: entry for fid, entry in registry.items()
            if isinstance(entry, dict)
            and str(entry.get("promotion_status") or "pending") not in PROMOTED
            and (args.all or str(entry.get("promotion_status") or "pending") != "stage_two_failed")
        }
        if not pending:
            print(f"[{mode}] 无待晋升候选")
            continue

        cfg = _mode_config(mode)
        service = StockEvalService(max_parallel_eval=1)
        print(f"[{mode}] 待晋升候选 {len(pending)} 个")
        for fid in sorted(pending):
            entry = pending[fid]
            prev = str(entry.get("promotion_status") or "pending")
            print(f"  - {fid} (prev={prev})")
            result = _promote_one(service, cfg, mode, fid, entry, dry_run=args.dry_run)
            if args.dry_run:
                print(f"      → dry-run：{result['expr_head'][:80]}")
                continue
            status = result.get("promotion_status")
            stored = result.get("stored")
            st2 = result.get("stage_two") or {}
            print(
                f"      → stored={stored} status={status} "
                f"stage_two_passed={st2.get('passed')} "
                f"reasons={st2.get('fail_reasons')} "
                f"skip={result.get('skipped_reason')}"
            )
            if result.get("error_type"):
                print(f"      ! error={result.get('error_type')}: {result.get('error')}")
        service = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

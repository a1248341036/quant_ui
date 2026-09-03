#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重放 run 3754c22c40c8 回收的过线因子：走修复后的 submit 链路入库。

背景：该 run 的 submit_factor 全部因子 repo_root 解析错误（agent/ 子包化后
parents[3] 少一级）而失败，因子没进候选池。本脚本把轨迹里 |IC|>=0.015 且
|ICIR|>0.25 的表达式重新走 stage_one → (盲测终审) → stage_one相似度 → 入候选池
链路。review_hook 不调 LLM（与 promote_candidates.py 同口径）。

用法：
  .venv\\Scripts\\python.exe scripts\\_replay_3754_harvest.py            # 全部
  .venv\\Scripts\\python.exe scripts\\_replay_3754_harvest.py --dry-run  # 只看链路是否打通
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.factor.mining.config import MiningConfig  # noqa: E402
from alphaagent.factor.mining.context import StockEvalContext  # noqa: E402
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

HARVEST = ROOT / "artifacts" / "alphaagent" / "replay_3754c22c40c8.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="逗号分隔的 factor_name 过滤")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    records = json.loads(HARVEST.read_text(encoding="utf-8"))
    if only:
        records = [r for r in records if r["factor_name"] in only]
    print(f"待重放 {len(records)} 个因子")

    spec = effective_research_spec("technical")
    label_col = str(spec.get("recommended_label_col") or DEFAULT_LABEL_COL)
    service = StockEvalService(max_parallel_eval=1)
    submit_service = FactorSubmitService(
        service,
        factorlib_path=factor_categories.production_dir("technical"),
        registry_path=factor_categories.production_registry_path("technical"),
        expr_dir=factor_categories.production_expr_dir("technical"),
        repo_root=ROOT,
        research_mode="technical",
        delivery_policy=spec.get("delivery_policy"),
    )

    results = []
    for rec in records:
        name = rec["factor_name"]
        ctx = StockEvalContext(
            panel_path=CNE_SOURCE,
            train_start=DEFAULT_TRAIN_START,
            train_end=DEFAULT_TRAIN_END,
            val_start=DEFAULT_VAL_START,
            val_end=DEFAULT_VAL_END,
            label_col=label_col,
            include_fundamentals=False,
        )
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
                multi_line_expr=str(rec["expression"] or ""),
                factor_name=name,
                comment=f"replay from run 3754c22c40c8 (train IC {rec['train_ic']:+.4f})",
                interaction=rec.get("interaction"),
            )
        except Exception as exc:
            results.append({"factor_name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ! {name}: EXC {type(exc).__name__}: {exc}")
            continue
        finally:
            service.release_session(session_resp.session_id)

        row = {
            "factor_name": name,
            "train_ic": rec["train_ic"],
            "ok": bool(result.get("ok")),
            "stored": bool(result.get("stored")),
            "candidate_stored": bool(result.get("candidate_stored")),
            "promotion_status": result.get("promotion_status"),
            "stage_one": (result.get("delivery_check") or {}).get("stage_one"),
            "skipped_reason": result.get("skipped_reason"),
            "error": result.get("error"),
        }
        results.append(row)
        st1 = (result.get("delivery_check") or {}).get("stage_one") or {}
        print(f"  {name}: stored={row['stored']} cand={row['candidate_stored']} "
              f"status={row['promotion_status']} stage_one_passed={st1.get('passed')} "
              f"skip={row['skipped_reason']} err={str(row['error'])[:80]}")

    out = ROOT / "artifacts" / "alphaagent" / "replay_3754c22c40c8_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

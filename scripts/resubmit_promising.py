#!/usr/bin/env python3
"""把研究记忆里的 promising 存量重新提交走完整入库链路。

背景：挖掘循环里"评估→提交"的跟进率不高，记忆库积累了大量 verdict=promising
（训练段过海选线）但从未 submit 的因子。本脚本按 |IC| 降序取出这批存量，
用当前门槛逐个重走 submit 链路（stage_one → 正交 → 候选池 → stage_two →
engine_gate），让"评估过但没人提交"的线索有一次正式转化机会。

用法：
  .venv\\Scripts\\python.exe scripts\\resubmit_promising.py --dry-run
  .venv\\Scripts\\python.exe scripts\\resubmit_promising.py --limit 30
  .venv\\Scripts\\python.exe scripts\\resubmit_promising.py --mode technical --min-ic 0.025

说明：
- 提交链路含全区间复检与 JIT 编译，首次提交可能较慢（与挖掘时一致）；
- submit 自带与候选池/正式库的相似度与正交检查，重复表达自然被拦；
- 会话按模式懒构建：默认 technical（label_1d/严门槛），--mode fundamental 切换。
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
from alphaagent.factor.mining.delivery.submit import FactorSubmitService  # noqa: E402
from alphaagent.factor.mining.research_spec import effective_research_spec  # noqa: E402
from alphaagent.factor.mining.schemas import SessionCreateRequest  # noqa: E402
from alphaagent.factor.mining.service import StockEvalService  # noqa: E402
from alphaagent.factor.types import (  # noqa: E402
    DEFAULT_LABEL_COL,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)
from alphaagent.data.adapters.cnequity import CNE_SOURCE  # noqa: E402
from core import factor_categories  # noqa: E402

MEMORY_DB = ROOT / "artifacts" / "alphaagent" / "research_memory.db"


def load_promising(limit: int, min_ic: float, max_attempts: int = 3) -> list[dict]:
    """按 |IC| 降序取 promising 存量；低质量（attempts 过多）靠后。"""
    conn = sqlite3.connect(MEMORY_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, factor_name, expression, metrics_json, attempts, updated_at
        FROM memory_entries
        WHERE verdict = 'promising' AND expression IS NOT NULL AND expression != ''
        ORDER BY updated_at DESC
        """
    ).fetchall()
    conn.close()
    out: list[dict] = []
    for r in rows:
        m = json.loads(r["metrics_json"] or "{}")
        ic = m.get("ic")
        if ic is None:
            continue
        ic = float(ic)
        if abs(ic) < min_ic:
            continue
        out.append(
            {
                "entry_id": r["id"],
                "name": r["factor_name"] or "expr",
                "expr": r["expression"],
                "ic": ic,
                "icir": m.get("icir"),
                "coverage": m.get("factor_coverage", m.get("coverage")),
                "attempts": int(r["attempts"] or 1),
                "updated_at": r["updated_at"],
            }
        )
    out.sort(key=lambda x: (-abs(x["ic"]), x["attempts"]))
    if max_attempts:
        out = [x for x in out if x["attempts"] <= max_attempts]
    return out[:limit]


def already_delivered(expr: str, mode: str) -> bool:
    """表达式已在候选/正式 registry 中则跳过（submit 也会拦，这里省一次会话构建）。"""
    import hashlib

    expr_norm = "\n".join(line.strip() for line in expr.splitlines() if line.strip())
    expr_hash = hashlib.sha256(expr_norm.encode("utf-8")).hexdigest()[:20]
    for reg_path in (
        factor_categories.candidate_registry_path(mode),
        factor_categories.production_registry_path(mode),
    ):
        if not reg_path.is_file():
            continue
        try:
            registry = json.loads(reg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if expr_hash in registry:
            return True
        for entry in registry.values():
            if isinstance(entry, dict) and str(entry.get("expr") or "").strip() == expr_norm:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Resubmit promising memory entries through delivery")
    ap.add_argument("--mode", choices=["technical", "fundamental"], default="technical")
    ap.add_argument("--dry-run", action="store_true", help="只列出待提交因子，不执行")
    ap.add_argument("--limit", type=int, default=30, help="最多处理条数（按 |IC| 降序）")
    ap.add_argument("--min-ic", type=float, default=0.025, help="|IC| 下限（默认对齐正式库 train 线）")
    ap.add_argument("--max-attempts", type=int, default=3, help="跳过反复尝试仍无进展的条目")
    args = ap.parse_args()

    mode = args.mode
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
    cfg = MiningConfig(eval=ctx, research_spec=spec, max_tokens=4096)

    pool = load_promising(limit=args.limit, min_ic=args.min_ic, max_attempts=args.max_attempts)
    # 已在候选/正式库的表达式跳过
    pool = [x for x in pool if not already_delivered(x["expr"], mode)]
    if not pool:
        print("无可处理的 promising 存量（全部已提交或不达 |IC| 下限）")
        return 0

    print(f"[{mode}] 待重提 promising 存量 {len(pool)} 个（min |IC|={args.min_ic}）")
    for x in pool:
        print(f"  - |IC|={abs(x['ic']):.4f} {x['name']} (attempts={x['attempts']})")
    if args.dry_run:
        print("dry-run：不执行提交")
        return 0

    service = StockEvalService(max_parallel_eval=1)
    submit_service = FactorSubmitService(
        service,
        factorlib_path=factor_categories.production_dir(mode),
        registry_path=factor_categories.production_registry_path(mode),
        expr_dir=factor_categories.production_expr_dir(mode),
        repo_root=ROOT,
        research_mode=mode,
        delivery_policy=(cfg.research_spec or {}).get("delivery_policy"),
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
    stored_count = 0
    try:
        for x in pool:
            print(f"\n>>> {x['name']} |IC|={abs(x['ic']):.4f}")
            try:
                result = submit_service.submit(
                    session_resp.session_id,
                    multi_line_expr=x["expr"],
                    factor_name=x["name"],
                    comment=f"resubmit_promising: 记忆库 promising 存量重提 (IC={x['ic']:+.4f})",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ! 异常: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            stored = bool(result.get("stored"))
            cand = bool(result.get("candidate_stored"))
            stored_count += int(stored or cand)
            st1 = result.get("delivery_check", {}).get("stage_one") or {}
            print(
                f"    stored={stored} candidate={cand} "
                f"stage_one_passed={st1.get('passed')} reasons={(st1.get('fail_reasons') or [])[:3]}"
            )
            skip = result.get("skipped_reason")
            if skip:
                print(f"    skip={str(skip)[:180]}")
            err = result.get("error")
            if err:
                print(f"    error={str(err)[:180]}")
    finally:
        service.release_session(session_resp.session_id)

    print(f"\n完成：{len(pool)} 个重提，{stored_count} 个进入候选池/正式库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

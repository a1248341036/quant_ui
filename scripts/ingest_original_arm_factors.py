#!/usr/bin/env python3
"""把对比实验中原版 AlphaAgent arm 提交的因子走 quant_ui 正规交付链路。

背景：对比 run（docs/alphaagent_对比结果_run1.md）中原版 arm 因空库未初始化，
25 次 submit 全被 factorlib_not_initialized 拦截；其提交表达式已在轨迹中提取。
本脚本让这些因子在 quant_ui 的两阶段链路（stage_one 统计门槛 → 正交查重 →
候选池 → stage_two → engine_gate）下重新参赛，回答"能不能进候选/晋升"。

- 表达式目录：artifacts/alphaagent/comparison/original_run2/expressions（内容去重）
- 会话面板：与挖掘时相同的导出面板（2020~2024），2025+ 盲测段不可见
- 不调用 LLM reviewer（与 promote_candidates 同口径）
- 晋升裁决 = stage_two 统计门槛 + engine_gate 完整回测

用法:
  .venv/Scripts/python.exe scripts/ingest_original_arm_factors.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
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
from core import factor_categories  # noqa: E402

EXPR_DIR = Path(r"D:\Quant\quant_ui\artifacts\alphaagent\comparison\original_run2\expressions")
# 盲测终审门禁需要 test 段（2025+）数据——必须用 CNE 源构建会话面板
# （导出面板只到 2024-12-31，会让盲测评估因缺数据报 blind_test_ic_missing）
PANEL = CNE_SOURCE
MODE = "technical"
TRAIN = ("2020-01-01", "2022-12-31")
VAL = ("2023-01-01", "2024-12-31")
LABEL = "label_1d_close_to_close"


def _load_unique_exprs() -> list[tuple[str, str]]:
    by_hash: dict[str, tuple[str, str]] = {}
    for f in sorted(EXPR_DIR.glob("*.dsl")):
        expr = f.read_text(encoding="utf-8").strip()
        h = hashlib.sha256(expr.encode("utf-8")).hexdigest()
        by_hash.setdefault(h, (f.stem, expr))
    return list(by_hash.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    factors = _load_unique_exprs()
    print(f"unique expressions: {len(factors)} (from {len(list(EXPR_DIR.glob('*.dsl')))} files)")
    if args.dry_run:
        for name, expr in factors:
            print(f"  - {name}: {expr.splitlines()[-1][:80]}")
        return 0

    spec = effective_research_spec(MODE)
    cfg = MiningConfig(
        eval=StockEvalContext(
            panel_path=PANEL,
            train_start=TRAIN[0], train_end=TRAIN[1],
            val_start=VAL[0], val_end=VAL[1],
            label_col=LABEL,
            include_fundamentals=False,
        ),
        research_spec=spec,
    )
    service = StockEvalService(max_parallel_eval=2)
    session_resp = service.create_session(SessionCreateRequest(
        panel_path=str(PANEL),
        train_start=TRAIN[0], train_end=TRAIN[1],
        val_start=VAL[0], val_end=VAL[1],
        label_col=LABEL,
        include_fundamentals=False,
    ))
    sid = session_resp.session_id
    print(f"session {sid[:12]} panel rows={session_resp.panel_rows:,}")

    submit_service = FactorSubmitService(
        service,
        factorlib_path=factor_categories.production_dir(MODE),
        registry_path=factor_categories.production_registry_path(MODE),
        expr_dir=factor_categories.production_expr_dir(MODE),
        repo_root=ROOT,
        research_mode=MODE,
        delivery_policy=(cfg.research_spec or {}).get("delivery_policy"),
    )

    summary = []
    for name, expr in factors:
        comment = (
            "跨系统对比实验（docs/alphaagent_对比结果_run1.md）中原版 AlphaAgent arm "
            "自选提交因子，经 quant_ui 两阶段链路重新参赛；挖掘窗口 2020-2024，"
            "label_1d_close_to_close。"
        )
        result = submit_service.submit(
            sid,
            multi_line_expr=expr,
            factor_name=name,
            comment=comment,
        )
        metrics = result.get("metrics") or {}
        row = {
            "name": name,
            "ok": bool(result.get("ok")),
            "stored": bool(result.get("stored")),
            "candidate_stored": bool(result.get("candidate_stored")),
            "promotion_status": result.get("promotion_status"),
            "stage_one": (result.get("delivery_check") or {}).get("stage_one"),
            "stage_two": (result.get("delivery_check") or {}).get("stage_two"),
            "gate": (result.get("delivery_check") or {}).get("engine_gate"),
            "max_abs_corr": (result.get("similarity") or {}).get("max_abs_corr"),
            "train_ic": metrics.get("ic"),
            "train_icir": metrics.get("icir"),
            "train_cov": metrics.get("factor_coverage"),
            "test_holdout": result.get("test_holdout") or {},
            "error": str(result.get("error") or result.get("skipped_reason") or "")[:100],
        }
        summary.append(row)
        print(
            f"  {name:34s} ok={row['ok']} cand={row['candidate_stored']} "
            f"train_ic={row['train_ic']} train_icir={row['train_icir']} "
            f"corr={row['max_abs_corr']} {row['error']}"
        )
    service.release_session(sid)

    print("\n==== 汇总 ====")
    for row in summary:
        print(
            f"  {row['name']:34s} status={row['promotion_status']} "
            f"stage_one={row['stage_one']} stage_two={row['stage_two']} gate={row['gate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

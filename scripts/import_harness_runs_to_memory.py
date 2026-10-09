#!/usr/bin/env python3
"""把 harness_mine.py 的 JSONL 评估记录导入研究记忆库（research_memory.db）。

背景：harness 直驱挖掘（harness_mine.py CLI）不经过 agentscope 框架，
评估结果只落本地 JSONL，前端"研究总结"页（/api/alphaagent/research-memory）
看不到。本脚本把 JSONL 转成 record_tool_result 需要的格式批量导入，
使 harness 直驱战役也进研究记忆库。

用法：
  .venv\\Scripts\\python.exe scripts\\import_harness_runs_to_memory.py
  .venv\\Scripts\\python.exe scripts\\import_harness_runs_to_memory.py --dir artifacts/alphaagent/harness_runs --pattern "batch*_rows.jsonl" --run-id harness-20261009
  .venv\\Scripts\\python.exe scripts\\import_harness_runs_to_memory.py --dry-run  # 只打印不写库
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.factor.mining.research_memory import ResearchMemoryStore  # noqa: E402


def _mode_to_research_mode(mode: str) -> str:
    """harness_mine 的 mode → research_memory 的 research_mode。"""
    m = str(mode or "").strip()
    # harness_mine mode: technical / technical_weekly / technical_monthly / fundamental / report
    # research_memory research_mode: 同名（RESEARCH_MODES 已覆盖）
    return m or "technical"


def _build_tool_row(entry: dict, split: str) -> dict | None:
    """把 harness_mine JSONL 的一条记录转成 record_tool_result 的 row 格式。

    split: "train" / "val" —— harness_mine 的主评估是 train，--val 追加的是 val。
    """
    name = str(entry.get("name") or "").strip()
    expr = str(entry.get("expr") or "").strip()
    if not name or not expr:
        return None

    mode = _mode_to_research_mode(entry.get("mode"))
    err = entry.get("err")

    # tool name 映射
    if split == "val":
        tool_name = "eval_on_val_set"
        val_data = entry.get("val") if isinstance(entry.get("val"), dict) else {}
        ic = val_data.get("ic")
        icir = val_data.get("icir")
        rank_ic = val_data.get("rank_ic")
        coverage = val_data.get("coverage")
        cs_autocorr = val_data.get("cs_autocorr")
        turnover = val_data.get("turnover_rebalance")
    else:
        tool_name = "eval_on_train_set"
        ic = entry.get("ic")
        icir = entry.get("icir")
        rank_ic = entry.get("rank_ic")
        coverage = entry.get("coverage")
        cs_autocorr = entry.get("cs_autocorr")
        turnover = entry.get("turnover_rebalance")

    # 构造 result dict（_classify 从这里读 metrics / split / error）
    metrics = {
        "ic": ic,
        "icir": icir,
        "rank_ic": rank_ic,
        "factor_coverage": coverage,
        "coverage": coverage,
        "cs_autocorr": cs_autocorr,
        "turnover_rebalance": turnover,
        "research_mode": mode,
    }
    # 去掉 None 值（_safe_float 会处理 None）
    metrics = {k: v for k, v in metrics.items() if v is not None}

    result = {
        "split": split,
        "metrics": metrics,
        "factor_name": name,
        "error": str(err) if err else "",
    }

    # arguments_raw：JSON 字符串，含 multi_line_expr + factor_name
    args = {"multi_line_expr": expr, "factor_name": name}
    arguments_raw = json.dumps(args, ensure_ascii=False)

    return {
        "name": tool_name,
        "arguments_raw": arguments_raw,
        "result": result,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="artifacts/alphaagent/harness_runs",
                    help="JSONL 所在目录")
    ap.add_argument("--pattern", default="batch*_rows.jsonl",
                    help="文件 glob 模式")
    ap.add_argument("--run-id", default="harness-import",
                    help="写入研究记忆的 run_id")
    ap.add_argument("--db", default=None,
                    help="research_memory.db 路径（缺省用 service 默认）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印不写库")
    ap.add_argument("--skip-val", action="store_true",
                    help="跳过 val 段（只导 train）")
    args = ap.parse_args()

    # 定位 db
    if args.db:
        db_path = Path(args.db)
    else:
        from backend.routers import alphaagent as _ag
        db_path = Path(_ag.service.RESEARCH_MEMORY_FILE)
    print(f"[import] db: {db_path}")
    if not db_path.exists():
        print(f"[import] ERROR: db not found: {db_path}")
        return 2

    # 收集 JSONL
    files = sorted(glob.glob(os.path.join(ROOT, args.dir, args.pattern)))
    if not files:
        print(f"[import] no files match {args.dir}/{args.pattern}")
        return 1
    print(f"[import] {len(files)} files")

    store = ResearchMemoryStore(str(db_path))

    total_entries = 0
    total_observations = 0
    skipped = 0
    t0 = time.perf_counter()

    for f in files:
        batch_name = os.path.basename(f)
        batch_count = 0
        try:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # train 段
                row = _build_tool_row(entry, "train")
                if row is None:
                    skipped += 1
                    continue
                if not args.dry_run:
                    res = store.record_tool_result(
                        run_id=args.run_id,
                        row=row,
                        run_freq_context={"research_mode": _mode_to_research_mode(entry.get("mode"))},
                        enable_sspm_write=False,  # harness 直驱不写 SSPM cells
                    )
                    if res:
                        batch_count += 1
                        total_observations += 1
                else:
                    batch_count += 1

                # val 段（如果有）
                if not args.skip_val and isinstance(entry.get("val"), dict):
                    row_val = _build_tool_row(entry, "val")
                    if row_val is not None and not args.dry_run:
                        store.record_tool_result(
                            run_id=args.run_id,
                            row=row_val,
                            run_freq_context={"research_mode": _mode_to_research_mode(entry.get("mode"))},
                            enable_sspm_write=False,
                        )
                        total_observations += 1
                    elif row_val is not None and args.dry_run:
                        total_observations += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[import] WARN: {batch_name} error: {exc}")
        total_entries += batch_count
        if batch_count:
            print(f"  {batch_name}: {batch_count} entries")

    elapsed = time.perf_counter() - t0
    print(f"\n[import] done: {total_entries} entries, {total_observations} observations, "
          f"{skipped} skipped, {elapsed:.1f}s, dry_run={args.dry_run}")
    if not args.dry_run:
        stats = store.statistics()
        print(f"[import] db statistics: entries={stats.get('entries')}, "
              f"observations={stats.get('observations')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

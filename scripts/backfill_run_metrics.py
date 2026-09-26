#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历史 Run 批量回填与指标诊断脚本。

遍历 logs/factor_mining/ui/* 目录，离线重算/生成 scorecard.json (v4)，
分析各指标在历史 run 上的非空率、分布范围及退化情况（恒为0或常数），
并将汇总报告持久化至 artifacts/alphaagent/bench/backfill_report.json。

用法：
    python scripts/backfill_run_metrics.py
    python scripts/backfill_run_metrics.py --limit 30
    python scripts/backfill_run_metrics.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaagent.core.timeutil import utc_now_iso
from core.atomicio import atomic_write_text
from alphaagent.factor.mining.run_metrics import generate_scorecard

UI_ROOT = ROOT / "logs" / "factor_mining" / "ui"
BENCH_DIR = ROOT / "artifacts" / "alphaagent" / "bench"


def _is_valid_run_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    # 至少包含 run_*.jsonl 或 steps.log
    jsonl_exists = any(d.glob("run_*.jsonl"))
    steps_exists = (d / "steps.log").is_file()
    meta_exists = (d / "run_meta.json").is_file()
    return jsonl_exists or steps_exists or meta_exists


def run_backfill(limit: int = 0, dry_run: bool = False) -> dict[str, Any]:
    if not UI_ROOT.is_dir():
        print(f"错误: 目录 {UI_ROOT} 不存在", file=sys.stderr)
        return {}

    all_dirs = sorted([d for d in UI_ROOT.iterdir() if _is_valid_run_dir(d)],
                      key=lambda d: d.stat().st_mtime, reverse=True)
    if limit > 0:
        all_dirs = all_dirs[:limit]

    print(f"找到 {len(all_dirs)} 个有效 run 目录，开始批量回填与解析...")

    records: list[dict[str, Any]] = []
    failed_runs: list[tuple[str, str]] = []

    for i, d in enumerate(all_dirs, start=1):
        rid = d.name
        try:
            # 离线重算生成 v4 scorecard
            if dry_run:
                from alphaagent.factor.mining.bench.extended_metrics import compute_extended_metrics
                from alphaagent.factor.mining.run_metrics import compute_run_metrics
                bm = compute_run_metrics(rid, d)
                ext = compute_extended_metrics(rid, d, base_metrics=bm)
                sc = {
                    "run_id": rid,
                    "schema_version": 4,
                    "summary": bm,
                    "headline": ext.get("headline", {}),
                    "exploration": ext.get("exploration", {}),
                    "dynamics": ext.get("dynamics", {}),
                    "quality": ext.get("quality", {}),
                    "cost": ext.get("cost", {}),
                    "process": ext.get("process", {}),
                    "integrity": ext.get("integrity", {}),
                    "time_meta": ext.get("time_meta", {}),
                }
            else:
                sc = generate_scorecard(rid, d)

            records.append(sc)
            if i % 15 == 0 or i == len(all_dirs):
                print(f"  进度: [{i}/{len(all_dirs)}] {rid}")
        except Exception as exc:
            failed_runs.append((rid, str(exc)))
            print(f"  [WARN] run {rid} 回填失败: {exc}", file=sys.stderr)

    print(f"\n回填完成: 成功 {len(records)} 个, 失败 {len(failed_runs)} 个\n")

    # 展开指标进行统计分布分析
    metric_values: dict[str, list[float]] = defaultdict(list)
    null_counts: dict[str, int] = defaultdict(int)

    def _collect(prefix: str, block: Any):
        if not isinstance(block, dict):
            return
        for k, v in block.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if math.isfinite(v):
                    metric_values[full_key].append(float(v))
                else:
                    null_counts[full_key] += 1
            elif v is None:
                null_counts[full_key] += 1
            elif isinstance(v, dict):
                # 递归嵌套字典（如 near_miss_progress, library_novelty）
                _collect(full_key, v)

    for sc in records:
        for grp in ("headline", "exploration", "dynamics", "quality", "cost", "process", "integrity", "summary", "funnel", "cognition"):
            _collect(grp, sc.get(grp))

    total_n = len(records)
    distribution_report: dict[str, dict[str, Any]] = {}
    degenerate_metrics: list[str] = []

    for k in sorted(metric_values.keys()):
        vals = sorted(metric_values[k])
        n_vals = len(vals)
        non_null_pct = round((n_vals / total_n) * 100.0, 1) if total_n else 0.0

        if not vals:
            distribution_report[k] = {
                "non_null_pct": 0.0,
                "count": 0,
                "status": "ALL_NULL",
            }
            degenerate_metrics.append(k)
            continue

        min_v = vals[0]
        max_v = vals[-1]
        med_v = vals[n_vals // 2]
        p25_v = vals[int(n_vals * 0.25)]
        p75_v = vals[int(n_vals * 0.75)]
        is_const = (min_v == max_v)

        status = "OK"
        if is_const:
            status = "CONSTANT_ZERO" if min_v == 0.0 else f"CONSTANT_{min_v}"
            degenerate_metrics.append(k)

        distribution_report[k] = {
            "non_null_pct": non_null_pct,
            "count": n_vals,
            "min": round(min_v, 4),
            "p25": round(p25_v, 4),
            "median": round(med_v, 4),
            "p75": round(p75_v, 4),
            "max": round(max_v, 4),
            "status": status,
        }

    # 打印终端看板
    print("=" * 88)
    print(f"【AlphaAgent 历史 Run v4 指标分布与退化诊断看板】(样本数 N={total_n})")
    print("=" * 88)
    fmt_row = f"  {'{:<38}':<38} | {'{:<10}':<10} | {'{:<8}':<8} | {'{:<8}':<8} | {'{:<8}':<8} | {'{:<10}'}"
    print(fmt_row.format("指标名称", "非空率 (%)", "Min", "Median", "Max", "状态"))
    print("-" * 92)

    # 重点打印 headline 与主要维度
    for k, d_info in distribution_report.items():
        if d_info.get("status") == "ALL_NULL":
            print(fmt_row.format(k, "0.0%", "N/A", "N/A", "N/A", "ALL_NULL"))
        else:
            print(fmt_row.format(
                k[:38],
                f"{d_info['non_null_pct']:.1f}%",
                f"{d_info['min']:.4g}",
                f"{d_info['median']:.4g}",
                f"{d_info['max']:.4g}",
                d_info['status']
            ))
    print("=" * 88)

    if degenerate_metrics:
        print(f"\n[提示] 发现 {len(degenerate_metrics)} 个退化或常数指标:")
        for dm in degenerate_metrics[:15]:
            print(f"  - {dm}: {distribution_report[dm]['status']}")
        if len(degenerate_metrics) > 15:
            print(f"  ... 另有 {len(degenerate_metrics) - 15} 项")

    report_payload = {
        "generated_at": utc_now_iso(),
        "total_runs": total_n,
        "failed_runs": failed_runs,
        "degenerate_metrics": degenerate_metrics,
        "metrics_distribution": distribution_report,
    }

    if not dry_run:
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        report_file = BENCH_DIR / "backfill_report.json"
        atomic_write_text(report_file, json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n")
        print(f"\n已将详细诊断报告保存至: {report_file}")

    return report_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaAgent 历史 Run 批量回填与指标诊断")
    parser.add_argument("--limit", type=int, default=0, help="限制处理最新 N 个 run（0=全量）")
    parser.add_argument("--dry-run", action="store_true", help="不写回 scorecard.json，仅内存打分统计")
    args = parser.parse_args()

    run_backfill(limit=args.limit, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

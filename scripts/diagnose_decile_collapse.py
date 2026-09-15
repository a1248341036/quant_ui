#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子分箱塌缩诊断与修复脚本 (Diagnose and Remediate Decile Bin Collapse).

诊断候选池和正式库中是否存在等频十分位分箱塌缩（bins < 10 或 collapse_ratio > 0.30），
定位塌缩算子（GATED_SIGNAL / PIECEWISE_STATE / IF_THEN_ELSE 等），
并支持对塌缩因子打标 soft-drop，移出 ML 训练集，防止污染组合模型。

使用示例：
    # 纯诊断打印
    python scripts/diagnose_decile_collapse.py

    # 诊断并执行 soft-drop 标记
    python scripts/diagnose_decile_collapse.py --apply-softdrop
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

_COLLAPSE_OPERATORS = (
    "GATED_SIGNAL",
    "PIECEWISE_STATE",
    "IF_THEN_ELSE",
    "FILLNA",
    "CS_GROUP_RANK",
    "CS_WINSORIZE",
)


def extract_collapse_operators(expr: str) -> list[str]:
    """从因子 DSL 表达式中提取可能引起分箱塌缩的关键算子。"""
    found = []
    for op in _COLLAPSE_OPERATORS:
        if re.search(rf"\b{op}\b", expr):
            found.append(op)
    return found


def diagnose_entry(
    factor_id: str,
    entry: dict[str, Any],
    expr_text: str = "",
    min_bins: int = 8,
    max_collapse_ratio: float = 0.30,
) -> dict[str, Any]:
    """分析单因子的分箱健康度。"""
    metrics = entry.get("metrics") or {}
    dml = metrics.get("decile_mean_label")
    qp = metrics.get("quantile_portfolio") or {}
    eg = metrics.get("engine_gate") or {}

    # bins 数量
    if isinstance(dml, list):
        n_bins = len(dml)
    elif isinstance(dml, dict):
        n_bins = len(dml)
    else:
        n_bins = 0

    # 塌缩率
    collapse_ratio = qp.get("collapse_ratio")
    try:
        collapse_ratio_f = float(collapse_ratio) if collapse_ratio is not None else None
    except (TypeError, ValueError):
        collapse_ratio_f = None

    # 超额收益
    ann_excess = qp.get("annualized_excess_return")
    eg_excess = eg.get("annual_excess_return") or eg.get("annualized_excess")

    # 算子扫描
    expr = str(entry.get("expr") or expr_text or "")
    ops = extract_collapse_operators(expr)

    # 判定是否塌缩
    is_collapsed = False
    reasons = []
    if n_bins < min_bins:
        is_collapsed = True
        reasons.append(f"bins={n_bins}<{min_bins}")
    if collapse_ratio_f is not None and collapse_ratio_f > max_collapse_ratio:
        is_collapsed = True
        reasons.append(f"collapse_ratio={collapse_ratio_f:.2f}>{max_collapse_ratio:.2f}")

    return {
        "factor_id": factor_id,
        "name": str(entry.get("name") or factor_id),
        "n_bins": n_bins,
        "collapse_ratio": collapse_ratio_f,
        "annual_excess": ann_excess,
        "engine_excess": eg_excess,
        "operators": ops,
        "is_collapsed": is_collapsed,
        "reasons": reasons,
        "dropped_from_ml": bool(entry.get("dropped_from_ml")),
        "dropped_reason": entry.get("dropped_reason"),
    }


def run_diagnosis(
    data_root: Path,
    min_bins: int = 8,
    max_collapse_ratio: float = 0.30,
    apply_softdrop: bool = False,
) -> list[dict[str, Any]]:
    cand_reg_path = data_root / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json"
    expr_dir = data_root / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "expressions"

    if not cand_reg_path.exists():
        print(f"[-] 注册表不存在: {cand_reg_path}")
        return []

    registry = json.loads(cand_reg_path.read_text(encoding="utf-8"))
    results = []

    for fid, entry in registry.items():
        expr_path = expr_dir / f"{fid}.dsl"
        expr_text = expr_path.read_text(encoding="utf-8") if expr_path.exists() else ""
        res = diagnose_entry(
            fid,
            entry,
            expr_text=expr_text,
            min_bins=min_bins,
            max_collapse_ratio=max_collapse_ratio,
        )
        results.append(res)

    # 排序：塌缩的排前面，按 bins 升序
    results.sort(key=lambda r: (not r["is_collapsed"], r["n_bins"]))

    # 输出表格
    print("\n" + "=" * 105)
    print(f"{'因子名称':<32} | {'bins':<4} | {'塌缩率':<8} | {'引擎/多空超额':<12} | {'塌缩相关算子':<24} | {'状态'}")
    print("-" * 105)
    for r in results:
        cr_str = f"{r['collapse_ratio']:.2%}" if r['collapse_ratio'] is not None else "N/A"
        ex_str = f"{r['engine_excess'] or r['annual_excess'] or 0.0:+.2%}"
        ops_str = ",".join(r["operators"][:2]) or "无"
        status = "[COLLAPSED]" if r["is_collapsed"] else "[OK]"
        if r["dropped_from_ml"]:
            status += " (soft-dropped)"
        print(f"{r['name'][:32]:<32} | {r['n_bins']:<4} | {cr_str:<8} | {ex_str:<12} | {ops_str:<24} | {status}")
    print("=" * 105)

    n_collapsed = sum(1 for r in results if r["is_collapsed"])
    print(f"\n诊断统计：共检测 {len(results)} 个因子，塌缩因子 {n_collapsed} 个（占比 {n_collapsed/max(1, len(results)):.1%}）。")

    if apply_softdrop and n_collapsed > 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bak_path = cand_reg_path.with_name(f"{cand_reg_path.stem}.bak-decile-collapse-{stamp}.json")
        bak_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[+] 备份原注册表至: {bak_path}")

        now_iso = datetime.now(timezone.utc).isoformat()
        dropped_count = 0
        for r in results:
            if r["is_collapsed"]:
                entry = registry[r["factor_id"]]
                if not entry.get("dropped_from_ml"):
                    entry["dropped_from_ml"] = True
                    entry["dropped_reason"] = f"prescreen_gate:decile_collapse({','.join(r['reasons'])})"
                    entry["dropped_at"] = now_iso
                    dropped_count += 1

        tmp_path = cand_reg_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, cand_reg_path)
        print(f"[+] 成功更新 {dropped_count} 个塌缩因子为 soft-drop (dropped_from_ml=True)，已持久化！\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="因子分箱塌缩诊断与修复")
    parser.add_argument("--data-root", default=str(ROOT), help="数据根目录")
    parser.add_argument("--min-bins", type=int, default=8, help="有效十分位组数下限（默认 8）")
    parser.add_argument("--max-collapse-ratio", type=float, default=0.30, help="塌缩天数比例上限（默认 0.30）")
    parser.add_argument("--apply-softdrop", action="store_true", help="对塌缩因子执行 soft-drop 标记并写回注册表")
    args = parser.parse_args()

    run_diagnosis(
        Path(args.data_root),
        min_bins=args.min_bins,
        max_collapse_ratio=args.max_collapse_ratio,
        apply_softdrop=args.apply_softdrop,
    )


if __name__ == "__main__":
    main()

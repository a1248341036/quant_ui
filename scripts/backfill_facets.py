# -*- coding: utf-8 -*-
"""Backfill facets classification fields into factor zoo registries.

对 candidate_main/mining_candidate_registry.json 与 production_main/mining_delivered_registry.json
的每个条目幂等补齐：facets（缺失时按表达式现算）/ is_fusion / family（classify_family_ex 新口径）/
eval_label。字段已存在且非空时不覆盖（保留历史值）。

用法：
    python scripts/backfill_facets.py            # 执行回填
    python scripts/backfill_facets.py --dry-run  # 只报告不写
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.factor.mining.memory.expressions import classify_family_ex, expr_facets

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "alphaagent" / "factorzoo"
REGISTRIES = [
    ROOT / "candidate_main" / "mining_candidate_registry.json",
    ROOT / "production_main" / "mining_delivered_registry.json",
]


def _entry_expr(entry: dict) -> str:
    expr = str(entry.get("expr") or "")
    if expr:
        return expr
    rel = str(entry.get("expression_file") or "")
    path = Path(__file__).resolve().parents[1] / rel if rel else None
    return path.read_text(encoding="utf-8").strip() if path and path.is_file() else ""


def backfill(registry_path: Path, *, dry_run: bool) -> None:
    if not registry_path.is_file():
        print(f"[skip] {registry_path.name} not found")
        return
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"[skip] {registry_path.name}: unexpected structure")
        return

    changed = 0
    for fid, entry in data.items():
        if not isinstance(entry, dict):
            continue
        expr = _entry_expr(entry)
        name = str(entry.get("name") or fid)

        facets = entry.get("facets") if isinstance(entry.get("facets"), list) and entry.get("facets") else None
        if facets is None:
            facets = sorted(expr_facets(name + " " + expr))
            if facets:
                entry["facets"] = facets
                changed += 1

        is_fusion = entry.get("is_fusion")
        if not isinstance(is_fusion, bool):
            entry["is_fusion"] = len(facets) >= 2
            changed += 1

        family = str(entry.get("family") or "")
        family_value = classify_family_ex(name, expr)
        # classify_family_ex 返回 (family, facets_set) 元组，family 字段只存键名
        new_family = family_value[0] if isinstance(family_value, tuple) else family_value
        if not family:
            entry["family"] = new_family
            changed += 1
        elif family != new_family:
            # 旧细粒度口径与新 classify_family_ex 对齐（融合因子用面对组合键）
            entry["family"] = new_family
            changed += 1

        if not entry.get("eval_label"):
            entry["eval_label"] = None
            changed += 1

        print(
            f"  {name}: facets={entry.get('facets')} is_fusion={entry.get('is_fusion')} "
            f"family={entry.get('family')!r}"
        )

    if dry_run:
        print(f"[dry-run] {registry_path.name}: would update {changed} fields")
        return
    if changed:
        registry_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[done] {registry_path.name}: updated {changed} fields -> {registry_path}")
    else:
        print(f"[ok] {registry_path.name}: already complete")


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    for path in REGISTRIES:
        print(f"===== {path.name} =====")
        backfill(path, dry_run=dry_run)


if __name__ == "__main__":
    main()

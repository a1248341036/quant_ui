#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""题库 `facets`（数据面）回填：用既有 `expr_facets` 口径从题面要素确定性推导。

背景（2026-10-04 整夜巡检发现）：`data/research_reports/knowledge/research_questions.jsonl`
的 **196 道题全部没有 `facets` 字段** → `get_question_for_turn(..., focus_facets=...)` 的
过滤结果恒为空、按既有实现**静默回退全池**：UI/监控传了"价量面"等聚焦面却毫无效果
（`tests/test_question_field_gate.py::test_ungated_matches_legacy_selection_with_focus`
因此长期飘红）。

本脚本不发明新词表：面名与识别键全部来自 `alphaagent.factor.facets.FACET_DEFS`
（前端 chips 与 data 层共用同一张表），推导方式 = 把题目已声明的要素拼成探针文本后调用
`expr_facets()`：
- `primary.spec_requirements.fields`（如 `$dv_ttm` → 基本面）
- `primary.spec_requirements.operators` / `structure_ops`（如 `CHIP_PEAK_LOC` → 筹码面）
- `primary.expr_local` / `expr_raw`（原文/推导算式里的 `$列` 与算子名）
- `reproduction_target`

用法::

    # 干跑（缺省）：只打印分布与样例，不写文件
    .venv\\Scripts\\python.exe scripts/backfill_question_facets.py

    # 落盘（先自动备份 research_questions.jsonl.bak-facets-<ts>）
    .venv\\Scripts\\python.exe scripts/backfill_question_facets.py --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 允许 `python scripts/xxx.py` 直跑

from alphaagent.factor.facets import expr_facets  # noqa: E402

DEFAULT_BANK = Path(__file__).resolve().parents[1] / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"


def probe_text(row: dict) -> str:
    """把题目已声明的要素拼成探针文本（供 expr_facets 做前缀匹配）。"""
    primary = row.get("primary") or {}
    req = primary.get("spec_requirements") or {}
    parts: list[str] = []
    for key in ("fields", "operators", "structure_ops", "chain", "windows"):
        val = req.get(key)
        if isinstance(val, (list, tuple)):
            parts.extend(str(x) for x in val)
        elif val:
            parts.append(str(val))
    for key in ("expr_local", "expr_raw", "spec_text", "name"):
        if primary.get(key):
            parts.append(str(primary[key]))
    if row.get("reproduction_target"):
        parts.append(str(row["reproduction_target"]))
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="题库 facets 回填（确定性推导，缺省干跑）")
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--apply", action="store_true", help="真正写回（先备份）")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.bank.read_text(encoding="utf-8").splitlines() if line.strip()]
    dist: Counter[str] = Counter()
    filled = 0
    samples: list[tuple[str, list[str], str]] = []
    for row in rows:
        had = row.get("facets")
        got = sorted(expr_facets(probe_text(row)))
        if got:
            filled += 1
            for f in got:
                dist[f] += 1
        if len(samples) < 8 and got:
            samples.append((str(row.get("question_id")), got, str(row.get("topic"))[:34]))

    print(f"题库: {args.bank}  n={len(rows)}")
    print(f"可推导出 facets 的题: {filled}/{len(rows)}（已有 facets 的: "
          f"{sum(1 for r in rows if r.get('facets'))}）")
    print("面分布:", dict(dist.most_common()))
    print("样例:")
    for qid, got, topic in samples:
        print(f"  {qid}: {got}  ← {topic}")
    missing = [str(r.get("question_id")) for r in rows if not expr_facets(probe_text(r))]
    if missing:
        print(f"推导不出面的题（{len(missing)}）: {missing[:10]}{' …' if len(missing) > 10 else ''}")

    if not args.apply:
        print("\n[干跑] 未写文件；加 --apply 落盘。")
        return 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = args.bank.with_suffix(args.bank.suffix + f".bak-facets-{ts}")
    shutil.copy2(args.bank, backup)
    out_lines = []
    for row in rows:
        got = sorted(expr_facets(probe_text(row)))
        if got:
            row["facets"] = got
        out_lines.append(json.dumps(row, ensure_ascii=False))
    args.bank.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\n已写回 {args.bank}（备份: {backup}），回填 {filled} 道题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

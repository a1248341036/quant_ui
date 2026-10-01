#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锚校准工具：把某个 run 真实写出的复现表达式回放到结构锚上，看拦截率是否合理。

用法：python scripts/replay_anchor_on_run.py --run a898c626b991
判读：① 若**该 run 中已过线的复现**被拦 → 锚过严；② 若一行式（≤1 算子/1 字段）没被拦 → 锚过松。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(r"D:\Quant\quant_ui")
sys.path.insert(0, str(REPO))
from alphaagent.factor.mining.agent.question_queue import check_reproduce_fidelity  # noqa: E402

K = REPO / "data" / "research_reports" / "knowledge"
UI = REPO / "logs" / "factor_mining" / "ui"
OPRE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\(")
FRE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    D = UI / args.run
    bank = {str(r.get("question_id")): r for r in
            (json.loads(l) for l in (K / "research_questions.jsonl").read_text(
                encoding="utf-8", errors="replace").splitlines() if l.strip())}
    recs = [json.loads(l) for l in (K / "factor_records_pilot.jsonl").read_text(
        encoding="utf-8", errors="replace").splitlines() if l.strip()]

    txt = (D / "steps.log").read_text(encoding="utf-8", errors="replace")
    parent, passed_live = {}, set()
    for line in txt.splitlines():
        m = re.search(r"memory\.record \| (\S+) verdict=\w+[^\n]*parent=reproduce_of:(RQ_\w+)", line)
        if m:
            parent.setdefault(m.group(1), m.group(2))
        p = re.search(r"report_reproduce_judge \| qid=(RQ_\w+) factor=(\S+) PASS", line)
        if p:
            passed_live.add(p.group(2))
    exprs: dict[str, str] = {}
    for fp in D.glob("run_*.jsonl"):
        for line in fp.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"arguments_raw"' not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            for item in ev.get("results") or []:
                try:
                    a = json.loads(item.get("arguments_raw") or "{}")
                except Exception:  # noqa: BLE001
                    continue
                if a.get("multi_line_expr"):
                    exprs.setdefault(str(a.get("factor_name") or ""), str(a["multi_line_expr"]))

    print(f"run {args.run}：表达式 {len(exprs)} 条；live 过线的复现 {len(passed_live)} 条\n")
    print(f"{'因子':<36}{'算子':>4}{'字段':>5}  {'锚':<8}{'live过线':<9}原因")
    print("-" * 120)
    blocked = 0
    kill_live = []
    for fac, expr in list(exprs.items())[:40]:
        q = bank.get(parent.get(fac, "")) or {}
        if not q:
            continue
        out = check_reproduce_fidelity(q, expr, records=recs, spec={})
        n_ops, n_fld = len(set(OPRE.findall(expr))), len(set(FRE.findall(expr)))
        ok = bool(out.get("passed"))
        blocked += 0 if ok else 1
        if not ok and fac in passed_live:
            kill_live.append(fac)
        print(f"{fac[:34]:<36}{n_ops:>4}{n_fld:>5}  {'通过' if ok else '拦截':<8}"
              f"{'是' if fac in passed_live else '':<9}{str(out.get('reason'))[:50]}")
    print(f"\n拦截率 {blocked}/{len(exprs)}；**误杀 live 已过线案例 {len(kill_live)} 条** "
          f"{kill_live[:5]}")
    print("判读：误杀 >0 → 锚过严；一行式未被拦 → 锚过松。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

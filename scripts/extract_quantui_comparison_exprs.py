#!/usr/bin/env python3
"""从 quant_ui 对比 run 的 JSONL 轨迹提取 LLM 提出的表达式（含 train IC），
按 |rank_ic| 排序写出 .dsl 文件，供盲测裁决使用。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
jsonl = ROOT / "artifacts/alphaagent/comparison/quantui_run1/logs/run_20260906_124942.jsonl"
steps = ROOT / "artifacts/alphaagent/comparison/quantui_run1/logs/steps.log"
out_dir = ROOT / "artifacts/alphaagent/comparison/quantui_run1/expressions"
TOP_K = 8

# steps.log: evaluate | evaluate_factor <name> | split=train | ic=.. | icir=.. | rank_ic=..
ic_pat = re.compile(
    r"evaluate(?:_factor)? (?P<name>\S+) \| split=train \| ic=(?P<ic>-?[\d.]+) \| icir=(?P<icir>-?[\d.]+) \| rank_ic=(?P<rank>-?[\d.]+)"
)
ics: dict[str, dict] = {}
for m in ic_pat.finditer(steps.read_text(encoding="utf-8")):
    g = m.groupdict()
    ics[g["name"]] = {"ic": float(g["ic"]), "icir": float(g["icir"]), "rank_ic": float(g["rank"])}

exprs: dict[str, str] = {}
for line in jsonl.read_text(encoding="utf-8").splitlines():
    e = json.loads(line)
    if e.get("event") != "assistant_tool_call":
        continue
    if e.get("name") not in ("evaluate_factor", "eval_on_train_set"):
        continue
    try:
        a = json.loads(e.get("arguments_raw") or "{}")
    except json.JSONDecodeError:
        continue
    name, expr = a.get("factor_name"), a.get("multi_line_expr")
    if name and expr:
        exprs[name] = expr.strip()

out_dir.mkdir(parents=True, exist_ok=True)
paired = [(n, exprs[n], ics[n]) for n in exprs if n in ics]
paired.sort(key=lambda x: -abs(x[2]["rank_ic"]))
print(f"expressions={len(exprs)} ics={len(ics)} paired={len(paired)}")
safe = re.compile(r"[^A-Za-z0-9_]+")
for n, expr, ic in paired[:TOP_K]:
    fname = safe.sub("_", n)[:60]
    (out_dir / f"{fname}.dsl").write_text(expr + "\n", encoding="utf-8")
    print(f"  {n:40s} ic={ic['ic']:+.4f} rank_ic={ic['rank_ic']:+.4f} icir={ic['icir']:+.3f}")
print(f"top {min(TOP_K, len(paired))} -> {out_dir}")

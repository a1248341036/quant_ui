# -*- coding: utf-8 -*-
"""用昨晚真实 vwap+平滑表达式验证 AST 指纹聚类效果。"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.factor.mining.tools._prefilter import (
    _ast_signal_fingerprint,
    _signal_fingerprint,
    _has_smoothing,
    _homogenization_block,
)

arms = list(Path("artifacts/ablation_prompt_20260918").glob("*_rep1"))
all_vwap = []
for arm in arms:
    jsonl_files = list(arm.glob("run_*.jsonl"))
    if not jsonl_files:
        continue
    for line in jsonl_files[0].read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if (obj.get("event") or obj.get("type")) != "assistant_tool_call":
            continue
        d = obj.get("data") or obj
        if not isinstance(d, dict):
            continue
        args = d.get("arguments") or d.get("arguments_raw") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        e = args.get("multi_line_expr") or args.get("expression") or ""
        if not e:
            continue
        if "vwap" in e.lower() and _has_smoothing(e):
            all_vwap.append(e)

print(f"全部 vwap+平滑: {len(all_vwap)}")

# AST 指纹分布
ast_counter = Counter()
for e in all_vwap:
    fp = _ast_signal_fingerprint(e)
    ast_counter[fp] += 1
print(f"\nAST 指纹不同簇数: {len(ast_counter)}")
print("Top 5 AST 指纹簇:")
for fp, n in ast_counter.most_common(5):
    print(f"  {n:3d}x  {fp[:120]}")

# 对比：正则指纹分布
regex_counter = Counter()
for e in all_vwap:
    sig = _signal_fingerprint(e)
    regex_counter[sig] += 1
print(f"\n正则指纹不同簇数: {len(regex_counter)}")
print("Top 5 正则指纹簇:")
for sig, n in regex_counter.most_common(5):
    ops_str = ",".join(sorted(sig[0])) or "(无)"
    fields_str = ",".join(sorted(sig[1]))
    print(f"  {n:3d}x  ops={ops_str} fields={fields_str}")

# 模拟连续提交最大 AST 簇
top_fp = ast_counter.most_common(1)[0][0]
top_count = ast_counter.most_common(1)[0][1]
print(f"\n最大 AST 簇: {top_count} 个表达式")
print(f"  指纹: {top_fp[:150]}")

# 找一个属于最大簇的表达式
sample = None
for e in all_vwap:
    if _ast_signal_fingerprint(e) == top_fp:
        sample = e
        break

# 模拟连续 3 次同族
recent = [
    {"fingerprint": f"fp{i}", "turnover": 0.3, "has_smoothing": True,
     "signal_fingerprint": _signal_fingerprint(sample),
     "signal_fingerprint_ast": top_fp}
    for i in range(3)
]
block = _homogenization_block(sample, recent, max_consecutive=3, enabled=True)
print(f"第 4 次同族拦截: {block is not None}")
if block:
    print(f"  消息: {block['error'][:200]}")

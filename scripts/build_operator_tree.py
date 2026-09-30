#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算子组成树生成器：从 operators.py 静态分析出「算子由什么组成」。

产出：
- 家族分类（TS_/CS_/CHIP_/CROWD_/PRICE_GAP_/结构单例/基础）
- 复合算子（内部调用了其它算子）→ children；并展开成链（如 SOFT_GATE ← CS_ZSCORE ← RANK）
- 叶子算子（内部不调用其它算子，直接落在 numpy/numba 原语上）
- 本次研报因子里的实际使用频次（有则标注）

用法::
    python scripts/build_operator_tree.py \
        --factors data/research_reports/knowledge/factor_records_pilot.jsonl \
        -o data/research_reports/knowledge/operator_tree
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OPS_FILE = ROOT / "alphaagent" / "dsl" / "core" / "operators.py"


def operator_names() -> set[str]:
    from alphaagent.dsl.core import operators as ops

    return {n for n in dir(ops) if n.isupper() and not n.startswith("_")}


def family_of(name: str) -> str:
    pre = name.split("_")[0]
    if pre in ("TS", "CS", "CHIP", "CROWD", "PRICE", "VOLUME", "WICK", "KLINE", "MUTUAL"):
        return pre
    return "结构/单例" if "_" in name else "基础运算"


def analyse(ops: set[str]) -> dict:
    tree = ast.parse(OPS_FILE.read_text(encoding="utf-8"))
    nodes: dict[str, dict] = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if not name or name not in ops:
            continue
        calls: Counter = Counter()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                fname = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if fname:
                    calls[fname] += 1
            elif isinstance(sub, ast.Name):
                calls[sub.id] += 0  # 记录出现但不确定是调用
        # 只保留**真实调用**（count>0），排除仅被引用的常量名
        children = sorted({c for c, v in calls.items() if v > 0 and c in ops and c != name})
        primitives = sorted({c for c, v in calls.items() if v > 0 and c not in ops and not c.startswith(("np", "numba", "math"))})
        nodes[name] = {"family": family_of(name),
                       "children": children,
                       "primitives": primitives[:8],
                       "composite": bool(children),
                       "lineno": node.lineno}
    # 展开成链（深度优先，最多 3 层）
    def chain(n: str, depth: int = 0, seen: tuple = ()) -> str:
        if n in seen or depth >= 3:
            return n
        ch = nodes.get(n, {}).get("children") or []
        if not ch:
            return n
        return f"{n} ← " + " ; ".join(chain(c, depth + 1, seen + (n,)) for c in ch[:3])

    for n, d in nodes.items():
        d["chain"] = chain(n)
    return nodes


def main() -> int:
    ap = argparse.ArgumentParser(description="算子组成树生成器")
    ap.add_argument("--factors", default="", help="因子清单 jsonl（用于标注使用频次）")
    ap.add_argument("-o", "--out", default=str(ROOT / "data" / "research_reports" / "knowledge" / "operator_tree"))
    args = ap.parse_args()

    ops = operator_names()
    nodes = analyse(ops)
    print(f"算子 {len(ops)} 个 | 源码中解析到实现 {len(nodes)} 个")

    usage = Counter()
    if args.factors and Path(args.factors).exists():
        rows = [json.loads(l) for l in Path(args.factors).read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if r.get("expr_local"):
                usage.update(re.findall(r"([A-Z][A-Z0-9_]*)\s*\(", r["expr_local"]))
        for n, d in nodes.items():
            d["used_in_pilot"] = usage.get(n, 0)

    by_fam: dict[str, list[str]] = defaultdict(list)
    for n, d in nodes.items():
        by_fam[d["family"]].append(n)

    base = Path(args.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(nodes, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 算子组成树（由 operators.py 静态解析生成）", "",
             f"算子总数 **{len(ops)}**；解析到实现 {len(nodes)}；其中**复合算子** "
             f"{sum(1 for d in nodes.values() if d['composite'])} 个、叶子算子 "
             f"{sum(1 for d in nodes.values() if not d['composite'])} 个。", ""]
    for fam in sorted(by_fam, key=lambda f: -len(by_fam[f])):
        names = sorted(by_fam[fam], key=lambda n: -nodes[n].get("used_in_pilot", 0))
        lines.append(f"## {fam}（{len(names)}）")
        lines.append("")
        for n in names:
            d = nodes[n]
            u = f" · 本轮用 {d['used_in_pilot']} 次" if d.get("used_in_pilot") else ""
            if d["composite"]:
                lines.append(f"- **{n}** ← {', '.join(d['children'])}{u}")
            else:
                pv = f"（原语: {', '.join(d['primitives'][:5])}）" if d["primitives"] else ""
                lines.append(f"- {n} ·叶子{pv}{u}")
        lines.append("")
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n复合算子 {sum(1 for d in nodes.values() if d['composite'])} 个：")
    for n, d in sorted(nodes.items(), key=lambda x: -len(x[1]["children"]))[:14]:
        if d["composite"]:
            print(f"  {d['chain'][:110]}")
    print(f"\n本轮用到但源码里没解析到实现的算子: {sorted(set(usage) - set(nodes))}")
    print(f"写出: {base.with_suffix('.md')} / {base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness 直驱挖掘 · 同族/重复预检（评估前的族内去重建议，stage 0）。

背景（2026-10-09）：整夜研报重挖 86 批 488 因子 → 7 候选全部同源
（illiqgrp5×prem/wma 族）。harness 直驱的评估/预检环节没有 agentscope 的记忆拦截
（AST 指纹去重 + 连续同族提醒），本脚本补上最轻的一环：设计批进评估前，用现成的
`_structure_fingerprint`（AST 拓扑归一化，变量/数字均归一化，memory/expressions.py
同源）做三层检查。纯文本分析：零评估、零面板、零网络依赖，毫秒级。

三层判定：
1. EXACT —— 批内互重 / 与候选 registry 同结构指纹。AST 指纹把变量与数字都归一化，
   所以"同构换窗长/换变量"（TS_MEAN(c,5) vs TS_MEAN(c,20)）也归为 EXACT：正是要拦的
   重复劳动。
2. FAMILY —— 批内强同族组：算子集合完全相同但指纹不同（如嵌套深度不同），典型
   "同族变体"。每组建议只保留 1 个代表进 dry_run（评估后取组内最优为最终代表）。
3. REGISTRY —— 与候选池同算子集条目（信息性提示：或已评估过，或晋升时撞正交门）。

用法：
    .venv\\Scripts\\python.exe scripts\\harness_precheck_family.py --factors batch87.json
    .venv\\Scripts\\python.exe scripts\\harness_precheck_family.py --factors batch87.json --json
    .venv\\Scripts\\python.exe scripts\\harness_precheck_family.py --factors batch87.json --strict
      # --strict：存在批内 EXACT / 撞 registry EXACT / 批内强同族组之一 → exit 1（作门禁）

factors JSON 格式与 harness_mine.py 一致：[{"name": ..., "expr": "多行 DSL（末行输出）", "why": ...}, ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.factor.mining.memory.expressions import expression_features  # noqa: E402 公共入口（算子/变量/窗口/指纹）

DEFAULT_REGISTRY = (
    ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json"
)


def load_factors(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("factors 必须是列表 [{name, expr, ...}, ...]")
    rows: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 个因子项必须是 dict {{name, expr, ...}}，实际为 {type(item).__name__}")
        name = str(item.get("name") or f"f{i}")
        expr = str(item.get("expr") or "")
        if not expr.strip():
            raise ValueError(f"[{name}] expr 为空")
        rows.append({"name": name, "expr": expr, "why": item.get("why") or ""})
    return rows


def load_registry(path: Path, *, expect_file: bool = False) -> list[tuple[str, str]]:
    """候选注册表 → [(name, expr), ...]（缺 expr 的条目跳过）。

    - 文件不存在：默认路径（worktree/未同步 artifacts）→ 返回 []（降级，由调用方告警）；
      显式传入（expect_file=True）→ FileNotFoundError（路径拼错必须暴露，防 fail-open）。
    - 文件存在但 JSON 损坏/顶层非 dict → ValueError（防"registry EXACT 0 处"假阴性）。
    """
    p = Path(path)
    if not p.exists():
        if expect_file:
            raise FileNotFoundError(f"registry 不存在: {path}")
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"registry 解析失败（{path}）: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"registry 顶层必须是 dict（keyed by 因子名）: {path}")
    out: list[tuple[str, str]] = []
    for name, entry in data.items():
        if isinstance(entry, dict) and entry.get("expr"):
            out.append((str(name), str(entry["expr"])))
    return out


def _struct(expr: str) -> dict:
    """表达式结构特征（公共入口 expression_features；fingerprint 与 AST 指纹同源）。"""
    return expression_features(expr or "")


def precheck(factors: list[dict], registry: list[tuple[str, str]]) -> dict:
    """三层检查；返回可 JSON 序列化的报告。"""
    entries: list[dict] = []
    for row in factors:
        st = _struct(row["expr"])
        entries.append({
            "name": row["name"],
            "fp": str(st.get("fingerprint") or ""),
            "ops": sorted(set(st.get("operators") or [])),  # 集合化：嵌套重复算子不算不同族
        })

    # 1a. 批内 EXACT（同结构指纹）→ 按指纹分组（同组 = 可互相替代的重复）
    exact_batch: list[list[str]] = []
    fp_groups: dict[str, list[str]] = {}
    for e in entries:
        if e["fp"]:
            fp_groups.setdefault(e["fp"], []).append(e["name"])
    for names in fp_groups.values():
        if len(names) > 1:
            exact_batch.append(names)

    # 1b. 与 candidate registry 的 EXACT 碰撞
    reg_by_fp: dict[str, list[str]] = {}
    for rname, rexpr in registry:
        st = _struct(rexpr)
        fp = str(st.get("fingerprint") or "")
        if fp:
            reg_by_fp.setdefault(fp, []).append(rname)
    exact_registry: list[dict] = []
    for e in entries:
        if e["fp"] in reg_by_fp:
            exact_registry.append({"factor": e["name"], "registry": reg_by_fp[e["fp"]]})

    # 2. 批内 FAMILY：算子集相同但结构指纹不同的因子（同族变体）；同指纹的归 EXACT
    #    （第 1 层，is_dup 直接从族分组剔除——重复只留该指纹代表）。
    fp_repr: dict[str, str] = {}
    for e in entries:
        if e["fp"]:
            fp_repr.setdefault(e["fp"], e["name"])
    ops_to_names: dict[tuple[str, ...], set[str]] = {}
    for e in entries:
        key = tuple(e["ops"])
        if key and e["name"] == fp_repr.get(e["fp"]):
            ops_to_names.setdefault(key, set()).add(e["name"])
    family_groups: list[list[str]] = []
    for names in ops_to_names.values():
        if len(names) > 1:
            family_groups.append(sorted(names))  # 代表 = sorted 首项（评估后取组内最优）
    family_groups.sort(key=lambda g: g[0])

    # 3. REGISTRY 关联（同算子集的候选池条目，信息性提示）
    reg_by_ops: dict[tuple[str, ...], list[str]] = {}
    for rname, rexpr in registry:
        ops = tuple(sorted(_struct(rexpr).get("operators") or []))
        if ops:
            reg_by_ops.setdefault(ops, []).append(rname)
    registry_links: list[dict] = []
    for e in entries:
        hits = reg_by_ops.get(tuple(e["ops"]), []) if e["ops"] else []
        if hits:
            registry_links.append({"factor": e["name"], "registry": hits[:3]})

    return {
        "exact_batch": exact_batch,
        "exact_registry": exact_registry,
        "family_groups": family_groups,
        "registry_links": registry_links,
    }


def format_report(rep: dict, n: int) -> str:
    lines = [
        f"[precheck] {n} 因子：批内 EXACT {len(rep['exact_batch'])} 处 / "
        f"registry EXACT {len(rep['exact_registry'])} 处 / 强同族组 {len(rep['family_groups'])} 组 / "
        f"registry 关联 {len(rep['registry_links'])} 条"
    ]
    if rep["exact_batch"]:
        lines.append("")
        lines.append("[!] EXACT_DUPLICATE（批内同指纹——同构换窗长/变量也被 AST 指纹归一化捕获）：")
        for grp in rep["exact_batch"]:
            lines.append(f"  {' == '.join(grp)}（组内同构重复，保留一个即可）")
    if rep["exact_registry"]:
        lines.append("")
        lines.append("[!] EXACT_DUPLICATE（撞候选池 registry——已评估过，纯重复劳动）：")
        for it in rep["exact_registry"]:
            lines.append(f"  {it['factor']} == registry:{', '.join(it['registry'])}")
    if rep["family_groups"]:
        lines.append("")
        lines.append("[|] 批内强同族组（算子集相同=同族变体；每组只保留代表进 dry_run，评估后取组内最优）：")
        for grp in rep["family_groups"]:
            lines.append(f"  group: {' , '.join(grp)}  → 候选代表: {grp[0]}")
    if rep["registry_links"]:
        lines.append("")
        lines.append("[~] registry 同算子集关联（提示：或已评估过，或晋升时撞正交门）：")
        for it in rep["registry_links"]:
            lines.append(f"  {it['factor']} ≈ registry:{', '.join(it['registry'])}")
    if not any([rep["exact_batch"], rep["exact_registry"], rep["family_groups"], rep["registry_links"]]):
        lines.append("  无冲突：本批与 registry/批内均无同指纹或同算子集，可放心评估。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Harness 挖掘：同族/重复预检（评估前族内去重建议）")
    ap.add_argument("--factors", required=True, help="JSON: [{name, expr, why?}, ...]（与 harness_mine.py 同格式）")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="候选注册表路径（缺省默认 registry）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 报告")
    ap.add_argument("--strict", action="store_true", help="存在批内/registry EXACT 或强同族组 → exit 1")
    args = ap.parse_args()

    # Windows 控制台可能是 GBK 代码页：stdout 强制 UTF-8 防 UnicodeEncodeError（不损失信息）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        factors = load_factors(Path(args.factors))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[precheck] ERROR {exc}", file=sys.stderr)
        return 2

    # registry 加载：显式传入但缺失/损坏 → 硬错（防"EXACT 0 处"假阴性 fail-open）；
    # 默认路径缺失（worktree/未同步 artifacts）→ 降级告警，registry 检查跳过。
    reg_path = Path(args.registry)
    explicit = os.path.normcase(str(reg_path.resolve())) != os.path.normcase(str(DEFAULT_REGISTRY.resolve()))
    try:
        registry = load_registry(reg_path, expect_file=explicit)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[precheck] ERROR {exc}", file=sys.stderr)
        return 2
    if not reg_path.exists() and not explicit:
        print("[precheck][warn] 默认候选池 registry 不存在（worktree/未同步 artifacts？）"
              "——registry 撞库与同算子集检查已跳过；如需检查用 --registry 指向主仓库路径。",
              file=sys.stderr)
    rep = precheck(factors, registry)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(format_report(rep, len(factors)))
    if args.strict and (rep["exact_batch"] or rep["exact_registry"] or rep["family_groups"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
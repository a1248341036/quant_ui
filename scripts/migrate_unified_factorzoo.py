#!/usr/bin/env python3
"""一次性迁移：按研究模式分库 → 统一大库（candidate_main / production_main）。

2026-09-03 大库重构（docs/融合因子与大库重构总计划.md Phase 2）。

做法：
1. 备份：旧目录重命名为 <name>.bak-unified-<ts>（rename，零拷贝，可回滚）。
2. candidate_main：以新 index（e3cd5d96…，三库共享）结构为基准；
   合并 candidate_technical + candidate_fundamental 两份 registry（去重），
   每条按 Phase 1 新口径补 facets/is_fusion/family/source_mode；
   candidate_technical 的 dense 值基于旧 index（10862d8a…，少 5547 行），
   不搬运——DSL 与评估证据保留在 registry，dense 需要时可按表达式重算。
3. production_main：两库均为空，直接复制 production_fundamental 的空结构。
4. 幂等：目标目录已存在时拒绝执行（先手工处理备份）。

用法：
  .venv\\Scripts\\python.exe scripts\\migrate_unified_factorzoo.py            # 执行
  .venv\\Scripts\\python.exe scripts\\migrate_unified_factorzoo.py --dry-run  # 只打印计划
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ZOO_ROOT = ROOT / "artifacts" / "alphaagent" / "factorzoo"
OLD_CAND = ["candidate_technical", "candidate_fundamental"]
OLD_PROD = ["production_technical", "production_fundamental"]
FRESH_INDEX_SOURCE = "production_technical"  # 空 zoo、新 index（e3cd…），结构完整

DRY_RUN = "--dry-run" in sys.argv


def _log(msg: str) -> None:
    print(msg, flush=True)


def _reg_path(lib: Path, which: str) -> Path:
    return lib / f"mining_{'candidate' if which == 'candidate' else 'delivered'}_registry.json"


def _load_registry(lib: Path, which: str) -> dict:
    p = _reg_path(lib, which)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _enrich(entry: dict, source_mode: str) -> dict:
    """按 Phase 1 新口径补数据面元数据（幂等：已带 facets 的条目只补 source_mode）。"""
    from alphaagent.factor.mining.memory.expressions import classify_family_ex, expr_facets

    out = dict(entry)
    out["source_mode"] = source_mode
    if "facets" not in out:
        expr = str(out.get("expr") or "")
        if not expr:
            rel = str(out.get("expression_file") or "")
            p = ROOT / rel if rel else None
            expr = p.read_text(encoding="utf-8") if p and p.is_file() else ""
        facets = sorted(expr_facets(str(out.get("name") or "") + " " + expr))
        out["facets"] = facets
        out["is_fusion"] = len(facets) >= 2
        out["family"] = classify_family_ex(str(out.get("name") or ""), expr)[0]
    return out


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cand_main = ZOO_ROOT / "candidate_main"
    prod_main = ZOO_ROOT / "production_main"
    if cand_main.exists() or prod_main.exists():
        _log("拒绝执行：candidate_main/production_main 已存在（幂等保护）。先手工处理备份目录。")
        return 1

    # ── 计划 ──
    merged_cand: dict[str, dict] = {}
    for lib_name, mode in [("candidate_technical", "technical"), ("candidate_fundamental", "fundamental")]:
        reg = _load_registry(ZOO_ROOT / lib_name, "candidate")
        for fid, entry in reg.items():
            if isinstance(entry, dict):
                merged_cand[fid] = _enrich(entry, mode)
    fusion_n = sum(1 for e in merged_cand.values() if e.get("is_fusion"))
    _log(f"计划：合并候选 registry {len(merged_cand)} 条（融合 {fusion_n} 条）")
    for fid, e in sorted(merged_cand.items()):
        _log(f"  - {fid} [{e.get('source_mode')}] family={e.get('family')} promotion={e.get('promotion_status')}")

    if DRY_RUN:
        _log("dry-run：未执行任何变更")
        return 0

    # ── 1. 备份（rename，零拷贝）──
    for name in [*OLD_CAND, *OLD_PROD]:
        src = ZOO_ROOT / name
        if src.exists():
            dst = ZOO_ROOT / f"{name}.bak-unified-{stamp}"
            src.rename(dst)
            _log(f"备份：{name} → {dst.name}")

    # ── 2. production_main：空结构直接从备份副本复制（rename 后源已改名，从 .bak 拷）──
    src_prod = ZOO_ROOT / f"{FRESH_INDEX_SOURCE}.bak-unified-{stamp}"
    shutil.copytree(src_prod, prod_main)
    # 空 delivered registry
    (prod_main / "mining_delivered_registry.json").write_text("{}", encoding="utf-8")
    _log(f"建库：production_main（空正式库，index_hash 见 manifest.json）")

    # ── 3. candidate_main：同样以新 index 空结构为基准 ──
    shutil.copytree(src_prod, cand_main)
    (cand_main / "mining_delivered_registry.json").unlink(missing_ok=True)

    # 合并 registry + expressions
    (cand_main / "mining_candidate_registry.json").write_text(
        json.dumps(merged_cand, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    expr_dir = cand_main / "expressions"
    expr_dir.mkdir(exist_ok=True)
    copied = 0
    for lib_name, _mode in [("candidate_technical", "technical"), ("candidate_fundamental", "fundamental")]:
        src_expr = ZOO_ROOT / f"{lib_name}.bak-unified-{stamp}" / "expressions"
        if src_expr.is_dir():
            for dsl in src_expr.glob("*.dsl"):
                shutil.copy2(dsl, expr_dir / dsl.name)
                copied += 1
    _log(f"建库：candidate_main（registry {len(merged_cand)} 条，DSL {copied} 份）")
    _log("注意：candidate_technical 的 2 个 dense 因子（旧 index 10862d8a）未搬运 dense 值，"
         "DSL/证据已保留，需要时按表达式在新 panel 上重算。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

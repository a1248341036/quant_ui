# -*- coding: utf-8 -*-
"""把扩库抽取结果合并进在用题库（2026-10-03）。

- 按 `source` 去重（已存在的跳过）
- 新题补 `question_id = RQ_ + md5(source)[:6]`（与在用题库同规则）
- 保留抽取产出的全部字段；`perf_label` 交给 scripts/label_question_quality.py 统一打标
- 写回前自动备份；默认 dry-run

用法：python scripts/merge_question_expansion.py --src question_v2_expand1.jsonl [--apply]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

def _shared_root() -> Path:
    """定位共享工作区根（worktree 里的 data/logs 可能是独立占位 → 必须命中真实产物路径）。"""
    root = Path(__file__).resolve().parents[1]
    for p in [root, *root.parents]:
        if (p / "logs" / "factor_mining" / "ui").is_dir() and (p / "artifacts").is_dir():
            return p
    return root


REPO = _shared_root()
K = REPO / "data" / "research_reports" / "knowledge"
BANK = K / "research_questions.jsonl"


def qid_of(source: str) -> str:
    return "RQ_" + hashlib.md5(str(source).encode("utf-8")).hexdigest()[:6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="question_v2_expand1.jsonl")
    ap.add_argument("--apply", action="store_true", help="缺省仅预览（dry-run）")
    args = ap.parse_args()

    src = K / args.src
    if not src.is_file():
        print(f"!! 找不到 {src}")
        return 1
    exp = [json.loads(l) for l in src.read_text(encoding="utf-8", errors="replace").splitlines()
           if l.strip()]
    bank = [json.loads(l) for l in BANK.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    have = {str(r.get("source")) for r in bank}
    new = [r for r in exp if str(r.get("source")) not in have]

    order0 = max([int(r.get("_order") or -1) for r in bank] + [-1])
    for i, r in enumerate(new, 1):
        r["question_id"] = qid_of(str(r.get("source")))
        r["_order"] = order0 + i
        r.pop("perf_label", None)      # 由 label 脚本统一生成
        r.pop("perf_cause", None)
        r.pop("perf_evidence", None)

    srv = [r for r in new if r.get("has_reproducible_structure") is True
           and r.get("method_type") not in ("ml_model", "graph_deep")]
    print(f"  {src.name}: {len(exp)} 行 | 与在用题库重叠 {len(exp) - len(new)} | **净新增 {len(new)}**")
    print(f"  其中可直接出题（结构准入 & 非 ml/graph）: **{len(srv)}**")
    if not args.apply:
        print("  （dry-run：未写回；加 --apply 执行）")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(BANK, BANK.with_suffix(f".jsonl.bak-merge-{stamp}"))
    merged = bank + new
    BANK.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
                    encoding="utf-8")
    print(f"  ✓ 在用题库 {len(bank)} → **{len(merged)}** 题（备份 .bak-merge-{stamp}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

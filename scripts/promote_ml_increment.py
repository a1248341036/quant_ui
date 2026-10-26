# -*- coding: utf-8 -*-
"""把 ML 筛选（permutation importance）通过的新因子入库到 **ML 增量候选库**。

**这个库是什么**（2026-10-20 新增，与 candidate_main 同级但用途不同）：
- 位置：`artifacts/alphaagent/factorzoo/ml_increment_main/mining_ml_increment_registry.json`
- 入库标准：**对 lgbm OOS IC 的边际贡献**（permutation importance，见
  `scripts/ml_driven_screen.py`），**不走** stage_one/盲测交付链——它的用途就是喂 ML。
- 与 candidate_main 的关系：互不覆盖。入库时与候选池重复的表达式会被
  `collect_factor_entries` 的 seen_exprs 去重（训练里只出现一次）。

**数据来源**（两个文件，都是 ml_driven_screen / harness_mine 的产出）：
1. `ml_screen.json`：`keep` 名单 + 每个 因子的 `ml_contribution`；
2. `batch*_rows.jsonl`：因子的 `expr` / `why` / raw 指标 / 增量指标。

用法：
    python scripts/promote_ml_increment.py \
        --screen artifacts/alphaagent/harness_runs/2026-10-20_increment/ml_screen.json \
        --rows   artifacts/alphaagent/harness_runs/2026-10-20_increment/batch1_rows.jsonl \
                 artifacts/alphaagent/harness_runs/2026-10-20_increment/batch3_rows.jsonl \
                 artifacts/alphaagent/harness_runs/2026-10-20_increment/batch4_rows.jsonl
缺省 dry-run；`--apply` 真正写入。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "ml_increment_main"
REGISTRY = LIB_DIR / "mining_ml_increment_registry.json"
try:
    from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC
    LABEL = str((DEFAULT_RESEARCH_SPEC.get("ml_increment_policy") or {}).get("label")
                or "label_20d_close_to_close")
except Exception:  # 脚本独立运行 / 包不可导入时回落字面量（2026-10-26 收口）
    LABEL = "label_20d_close_to_close"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="ml_driven_screen 产出的 json（keep + contributions）")
    ap.add_argument("--rows", nargs="+", required=True, help="harness_mine 的 rows.jsonl（取 expr 与指标）")
    ap.add_argument("--min-contribution", type=float, default=0.0005,
                    help="入库的 ML 贡献下限（与 ml_driven_screen 的 keep 判定一致）")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    screen = json.loads(Path(args.screen).read_text(encoding="utf-8"))
    keep = set(screen.get("keep") or [])
    contrib = {r["name"]: r["ml_contribution"] for r in screen.get("contributions") or []}

    rows: dict[str, dict] = {}
    for p in args.rows:
        path = Path(p)
        if not path.is_file():
            print(f"[promote] 跳过（不存在）: {p}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("name") and r.get("expr"):
                rows[r["name"]] = r

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.is_file() else {}

    added, updated, missing = [], [], []
    for name in sorted(keep):
        c = contrib.get(name)
        if c is None or c < args.min_contribution:
            missing.append(name)
            continue
        r = rows.get(name)
        if r is None:
            missing.append(name)
            continue
        entry = {
            "schema_version": 1,
            "name": name,
            "expr": r["expr"],
            "comment": r.get("why") or "",
            "ingested_at": now,
            "library": "ml_increment_main",
            "source": "ml_increment_screen",
            "eval_label": LABEL,
            "review_status": "ml_increment",   # 标记：ML 增量库，不走盲测交付链
            "promotion_status": "ml_sample",
            "facets": [],
            "metrics": {
                "label_col": LABEL,
                "ml_contribution": c,
                "raw_ic_audit": r.get("raw_ic_audit"),
                "resid_ic": r.get("resid_ic"),
                "increment": r.get("increment"),
                "combo_corr": r.get("combo_corr"),
                "ic": r.get("ic"), "icir": r.get("icir"),
                "coverage": r.get("coverage"),
                "turnover_rebalance": r.get("turnover_rebalance"),
                "cs_autocorr": r.get("cs_autocorr"),
            },
        }
        if name in registry:
            updated.append(name)
        else:
            added.append(name)
        registry[name] = entry

    print(f"[promote] keep={len(keep)} 可入库={len(added)+len(updated)} "
          f"（新增 {len(added)} / 更新 {len(updated)}；缺 expr 或低于阈值 {len(missing)}）")
    for n in added:
        print(f"  + {n}  contribution={contrib.get(n):+.5f}")
    for n in updated:
        print(f"  ~ {n}  contribution={contrib.get(n):+.5f}")
    for n in missing:
        print(f"  - {n}（跳过）")

    if not args.apply:
        print("（dry-run；加 --apply 写入）")
        return 0
    if not added and not updated:
        print("没有可写入的条目")
        return 0
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[promote] 已写入 {REGISTRY}（库内共 {len(registry)} 个因子）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

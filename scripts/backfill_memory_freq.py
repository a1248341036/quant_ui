# -*- coding: utf-8 -*-
"""研究记忆调仓频率存量回填：按 last_run_id 关联 run 目录 research_spec.json。

- 先备份 research_memory.db（.bak-freq-<时间戳>）
- 幂等：metrics 里已有 rebalance_freq 的条目跳过
- 用法：.venv\\Scripts\\python.exe scripts\\backfill_memory_freq.py [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.factor.mining.research_memory import ResearchMemoryStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    parser.add_argument("--root", default=str(ROOT), help="仓库根目录（默认脚本所在仓库）")
    args = parser.parse_args()

    repo = Path(args.root).expanduser().resolve()
    db_path = repo / "artifacts" / "alphaagent" / "research_memory.db"
    log_root = repo / "logs" / "factor_mining" / "ui"
    if not db_path.is_file():
        print(f"memory db not found: {db_path}")
        return 1

    if not args.dry_run:
        backup = db_path.with_name(f"{db_path.name}.bak-freq-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, backup)
        print(f"backup: {backup}")

    store = ResearchMemoryStore(db_path)
    if args.dry_run:
        # dry-run：临时挂到内存库上跑一遍真库的只读统计
        import json
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, last_run_id, metrics_json FROM memory_entries").fetchall()
        conn.close()
        scanned = skipped = updated = unresolvable = 0
        freq_dist: dict[str, int] = {}
        for row in rows:
            scanned += 1
            metrics = json.loads(row["metrics_json"] or "{}")
            if metrics.get("rebalance_freq"):
                skipped += 1
                continue
            rid = str(row["last_run_id"] or "")
            spec_path = log_root / rid / "research_spec.json" if rid else None
            freq = None
            if spec_path and spec_path.is_file():
                try:
                    spec = json.loads(spec_path.read_text(encoding="utf-8"))
                    gate = ((spec.get("delivery_policy") or {}).get("production") or {}).get("engine_gate") or {}
                    freq = str(gate.get("freq") or "") or None
                except Exception:
                    freq = None
            if freq:
                freq_dist[freq] = freq_dist.get(freq, 0) + 1
                updated += 1
            else:
                unresolvable += 1
        print(f"[dry-run] scanned={scanned} would_update={updated} skip={skipped} unresolvable={unresolvable}")
        print(f"[dry-run] freq dist: {freq_dist}")
        return 0

    summary = store.backfill_freq_from_run_specs(log_root)
    print("backfill summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

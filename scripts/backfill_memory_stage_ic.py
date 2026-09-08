# -*- coding: utf-8 -*-
"""研究记忆分窗口 IC 存量回填：按 last_run_id 关联 run 目录 JSONL 轨迹。

把 submit/eval 工具结果里的 train_ic / val_ic / test_ic 等分窗口指标补进
memory_entries.metrics_json（signature 精确匹配表达式，setdefault 只补缺，
幂等可重跑）。新代码入库时已携带，本脚本只服务存量条目。

- 先备份 research_memory.db（.bak-stageic-<时间戳>）
- 用法：.venv\\Scripts\\python.exe scripts\\backfill_memory_stage_ic.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
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
        backup = db_path.with_name(f"{db_path.name}.bak-stageic-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(db_path, backup)
        print(f"backup: {backup}")

    if args.dry_run:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, last_run_id, metrics_json FROM memory_entries").fetchall()
        conn.close()
        scanned = resolvable = would_update = 0
        key_dist = {"train_ic": 0, "val_ic": 0, "test_ic": 0}
        for row in rows:
            scanned += 1
            metrics = json.loads(row["metrics_json"] or "{}")
            missing = [k for k in key_dist if metrics.get(k) is None]
            if not missing:
                continue
            rid = str(row["last_run_id"] or "")
            if rid and (log_root / rid).is_dir():
                resolvable += 1
                for k in missing:
                    key_dist[k] += 1
        print(f"[dry-run] scanned={scanned} 有 run 目录可回填={resolvable}")
        print(f"[dry-run] 缺失键分布: {key_dist}")
        return 0

    store = ResearchMemoryStore(db_path)
    summary = store.backfill_stage_ics_from_logs(log_root)
    print("backfill summary:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

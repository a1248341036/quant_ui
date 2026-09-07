# -*- coding: utf-8 -*-
"""因子中台 ID 一次性回填 + factor_index.db 全量重建。

步骤（幂等，可重复执行）：
1. 备份研究记忆库（sqlite backup API，一致性快照）
2. registry JSON 条目盖 factor_uid（候选 + 正式，含历史所有注册路径文件）
3. 打开研究记忆库触发 v5 迁移（memory_factors 维表 + entries.factor_uid 回填）
4. 从三源（候选 registry / 正式 registry / 记忆库因子名）重建 factor_index.db

事实源仍是 registry JSON / FactorZoo / 研究记忆库，本脚本只派生。
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from core import factor_categories  # noqa: E402
from alphaagent.factor.identity import factor_uid  # noqa: E402
from alphaagent.factor.index import FactorIndex  # noqa: E402
from alphaagent.core.paths import FACTOR_INDEX_PATH  # noqa: E402

MEMORY_DB = ROOT / "artifacts" / "alphaagent" / "research_memory.db"


def _registry_paths() -> list[tuple[str, Path, str]]:
    """[(kind, path, kind_label)]：候选 + 正式 registry（按 category 去重）。"""
    out: dict[Path, tuple[str, str]] = {}
    for cat in ("technical", "fundamental"):
        cand = Path(factor_categories.candidate_registry_path(cat))
        prod = Path(factor_categories.production_registry_path(cat))
        out.setdefault(cand, ("candidate", str(cand)))
        out.setdefault(prod, ("production", str(prod)))
    return [(kind, path, label) for path, (kind, label) in sorted(out.items())]


def _stamp_registry(path: Path) -> tuple[int, int]:
    """给 registry 条目盖 factor_uid；返回 (总数, 新盖数)。"""
    if not path.is_file():
        return 0, 0
    reg = json.loads(path.read_text(encoding="utf-8"))
    stamped = 0
    for fid, entry in reg.items():
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or fid)
        uid = factor_uid(name)
        if entry.get("factor_uid") != uid:
            entry["factor_uid"] = uid
            stamped += 1
    if stamped:
        path.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(reg), stamped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backup", action="store_true", help="跳过记忆库备份")
    args = parser.parse_args()

    # 1. 备份记忆库
    if not args.skip_backup and MEMORY_DB.is_file():
        backup = MEMORY_DB.with_suffix(
            f".bak-uid5-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.db"
        )
        src = sqlite3.connect(str(MEMORY_DB))
        dst = sqlite3.connect(str(backup))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
        print(f"[1/4] 记忆库已备份 -> {backup.name}")
    else:
        print("[1/4] 跳过备份")

    # 2. registry 盖 uid
    total_entries = 0
    total_stamped = 0
    for kind, path, _label in _registry_paths():
        n, stamped = _stamp_registry(path)
        total_entries += n
        total_stamped += stamped
        print(f"[2/4] {kind:<10} {path.name}: {n} 条，新盖 {stamped}")
    print(f"[2/4] 合计 {total_entries} 条，新盖 {total_stamped}")

    # 3. 打开记忆库触发 v5 迁移
    from alphaagent.factor.mining.research_memory import ResearchMemoryStore

    store = ResearchMemoryStore(MEMORY_DB)
    with store._open() as conn:
        n_factors = conn.execute("SELECT COUNT(*) FROM memory_factors").fetchone()[0]
        n_missing = conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE factor_uid IS NULL"
        ).fetchone()[0]
        version = conn.execute("SELECT v FROM store_meta WHERE k='data_version'").fetchone()
    print(f"[3/4] 记忆库 v5：memory_factors={n_factors}，未回填条目={n_missing}，data_version={version[0] if version else '?'}")

    # 4. 重建索引（三源：候选 registry / 正式 registry / 记忆库因子名）
    rows: list[dict] = []
    seen_uids: set[str] = set()
    memory_dbs: set[str] = set()
    for kind, path, _label in _registry_paths():
        if not path.is_file():
            continue
        reg = json.loads(path.read_text(encoding="utf-8"))
        for fid, entry in reg.items():
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or fid)
            uid = str(entry.get("factor_uid") or factor_uid(name))
            seen_uids.add(uid)
            rel_expr = str(entry.get("expression_file") or "")
            row = {
                "uid": uid,
                "name": name,
                "expr": str(entry.get("expr") or ""),
                "family": entry.get("family"),
                "facets": entry.get("facets") if isinstance(entry.get("facets"), list) else None,
                "status": kind,
                "promotion_status": entry.get("promotion_status"),
                "dsl_path": str(ROOT / rel_expr) if rel_expr else None,
            }
            if kind == "candidate":
                row["candidate_registry"] = str(path)
            else:
                row["production_registry"] = str(path)
            rows.append(row)

    if MEMORY_DB.is_file():
        with store._open() as conn:
            for uid, name in conn.execute("SELECT uid, factor_name FROM memory_factors").fetchall():
                memory_dbs.add(str(MEMORY_DB))
                if uid in seen_uids:
                    continue
                seen_uids.add(uid)
                rows.append({"uid": uid, "name": name, "status": "memory_only", "memory_db": str(MEMORY_DB)})

    index = FactorIndex(FACTOR_INDEX_PATH)
    n = index.rebuild(rows)
    print(f"[4/4] factor_index.db 重建完成：{n} 行（{index.path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

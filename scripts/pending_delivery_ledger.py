# -*- coding: utf-8 -*-
"""B-05：达标未交付（pending delivery）台账生成器。

问题（docs/specs/alphaagent_known_bugs_spec.md B-05）：挖掘循环评估达标
（verdict ∈ {train_passed, validated}）后无强制交付闭环，存量因子悬置在
记忆库中从未 submit（2026-10-20 发现 158 个达标未入库、104 全门通过）。

本脚本**只读**扫描 research_memory.db，把达标且无失败记录的条目落成
JSONL 台账 + summary JSON（走 B-04 统一 writer），供 run 收尾提示与
用户决策。不改变任何库状态、不做自动 submit（rescreen/晋升属要问项）。

用法：
  python scripts/pending_delivery_ledger.py \
      --db artifacts/alphaagent/research_memory.db --out artifacts/alphaagent/ledgers
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from alphaagent.factor.mining.infra.artifact_io import write_artifact_json  # noqa: E402

DEFAULT_DB = REPO / "artifacts" / "alphaagent" / "research_memory.db"
DEFAULT_OUT = REPO / "artifacts" / "alphaagent" / "ledgers"
PENDING_VERDICTS = ("train_passed", "validated")

_QUERY = """
SELECT factor_name, expression, verdict, attempts, created_at,
       metrics_json, family, last_run_id
FROM memory_entries
WHERE verdict IN ('train_passed', 'validated')
  AND (error IS NULL OR error = '')
  AND failure_code IS NULL
ORDER BY created_at
"""


def fetch_pending(db_path: Path) -> list[dict]:
    """只读查询达标未交付条目（error/failure_code 必须为空）。"""
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(_QUERY).fetchall()
    finally:
        con.close()
    out: list[dict] = []
    for name, expr, verdict, attempts, created, metrics_raw, family, run_id in rows:
        metrics: dict = {}
        try:
            m = json.loads(metrics_raw) if metrics_raw else {}
            if isinstance(m, dict):
                for key in ("ic", "icir", "rank_ic", "turnover"):
                    if key in m:
                        metrics[key] = m[key]
        except Exception:  # noqa: BLE001 — metrics 解析失败不阻塞台账
            metrics = {}
        out.append(
            {
                "factor_name": name,
                "expression": expr,
                "verdict": verdict,
                "attempts": attempts,
                "created_at": created,
                "family": family,
                "last_run_id": run_id,
                "metrics": metrics,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="B-05 达标未交付台账生成器（只读）")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    rows = fetch_pending(args.db)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    ledger = args.out / f"pending_delivery_{stamp}.jsonl"
    with ledger.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_verdict = Counter(r["verdict"] for r in rows)
    by_family = Counter(r["family"] or "(none)" for r in rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db": str(args.db),
        "ledger": ledger.name,
        "total_pending": len(rows),
        "by_verdict": dict(by_verdict),
        "top_families": dict(by_family.most_common(10)),
        "note": (
            "台账=挖掘循环认定达标（train_passed/validated）且无失败记录、"
            "未进入交付流程的因子条目；只读生成，不做自动 submit"
            "（rescreen/晋升属要问项）。"
        ),
    }
    write_artifact_json(args.out / f"pending_delivery_summary_{stamp}.json", summary)
    print(f"pending={len(rows)} by_verdict={dict(by_verdict)} ledger={ledger.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

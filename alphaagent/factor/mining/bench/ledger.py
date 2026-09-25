"""Bench 评测台账与记录存储。

- 单次记录：artifacts/alphaagent/bench/eval_records/<ts>_<commit>.json
- 汇总台账：artifacts/alphaagent/bench/ledger.csv
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from alphaagent.core.timeutil import utc_now_iso
from core.atomicio import atomic_write_text

ROOT = Path(__file__).resolve().parents[4]
BENCH_DIR = ROOT / "artifacts" / "alphaagent" / "bench"
RECORDS_DIR = BENCH_DIR / "eval_records"
LEDGER_CSV = BENCH_DIR / "ledger.csv"

LEDGER_COLUMNS = [
    "created_at",
    "ended_at",
    "wall_minutes",
    "run_id",
    "branch",
    "commit",
    "dirty",
    "note",
    "config_hash",
    "stage_one_yield_pct",
    "gate_survival_pct",
    "effective_novelty_rate",
    "saturation_turn",
    "late_gain_share",
    "cost_per_candidate_k_tokens",
    "temporal_stability",
    "unsubmitted_passing_rate",
    "verdict",
    "exit_code",
    "total_tokens",
]


def record_eval(
    curr_sc: dict[str, Any],
    diff_res: dict[str, Any] | None = None,
    note: str = "",
    branch: str = "",
    commit: str = "",
    dirty: bool = False,
    exit_code: int = 0,
) -> Path:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    c_meta = curr_sc.get("time_meta") or {}
    created_at = c_meta.get("created_at") or utc_now_iso()
    ended_at = c_meta.get("ended_at") or created_at
    wall_min = c_meta.get("wall_minutes") or 0.0
    run_id = curr_sc.get("run_id", "unknown")
    cfg_hash = c_meta.get("config_hash") or ""

    hl = curr_sc.get("headline") or {}
    sm = curr_sc.get("summary") or {}
    verdict = diff_res.get("overall") if diff_res else "NO_BASELINE"

    # 生成记录文件名
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_commit = commit[:8] if commit else "head"
    rec_filename = f"{stamp}_{short_commit}.json"
    rec_file = RECORDS_DIR / rec_filename

    rec_payload = {
        "recorded_at": utc_now_iso(),
        "run_id": run_id,
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
        "note": note,
        "config_hash": cfg_hash,
        "verdict": verdict,
        "exit_code": exit_code,
        "scorecard": curr_sc,
        "diff": diff_res,
    }
    atomic_write_text(rec_file, json.dumps(rec_payload, ensure_ascii=False, indent=2) + "\n")

    # 追加或更新 ledger.csv
    row = {
        "created_at": created_at,
        "ended_at": ended_at,
        "wall_minutes": wall_min,
        "run_id": run_id,
        "branch": branch,
        "commit": short_commit,
        "dirty": dirty,
        "note": note,
        "config_hash": cfg_hash,
        "stage_one_yield_pct": hl.get("stage_one_yield_pct", 0.0),
        "gate_survival_pct": hl.get("gate_survival_pct", 0.0),
        "effective_novelty_rate": hl.get("effective_novelty_rate", 0.0),
        "saturation_turn": hl.get("saturation_turn"),
        "late_gain_share": hl.get("late_gain_share", 0.0),
        "cost_per_candidate_k_tokens": hl.get("cost_per_candidate_k_tokens"),
        "temporal_stability": hl.get("temporal_stability"),
        "unsubmitted_passing_rate": hl.get("unsubmitted_passing_rate", 0.0),
        "verdict": verdict,
        "exit_code": exit_code,
        "total_tokens": sm.get("total_tokens", 0),
    }

    file_exists = LEDGER_CSV.is_file()
    with open(LEDGER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return rec_file


def load_ledger() -> list[dict[str, Any]]:
    if not LEDGER_CSV.is_file():
        return []
    rows = []
    with open(LEDGER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    return rows

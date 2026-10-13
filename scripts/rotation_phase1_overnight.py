#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overnight loop for rotation Phase 1 minimal validation (spec 5).

Repeatedly runs scripts/rotation_phase1_validation.py on the real panel,
rotating the style-exposure window, to prove infra stability across a full
night and collect a rollup summary by the deadline.

Design choices:
  * subprocess per round -> memory fully returned between rounds (panel mmap);
  * rotate --style-window over the given set -> check state-series robustness
    to window choice without touching canon/near-dedup/broken-reject logic;
  * fixed sleep between rounds to bound machine load (user runs other jobs);
  * deadline default = next 07:00; runner stops and writes summary after it.

Usage:
  .venv\\Scripts\\python.exe scripts\\rotation_phase1_overnight.py \\
      --deadline 2026-10-14T07:00:00 [--interval-sec 900] \\
      [--windows "10 15 20 40 60"] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION = ROOT / "scripts" / "rotation_phase1_validation.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deadline", default="", help="ISO datetime; default next 07:00")
    ap.add_argument("--interval-sec", type=float, default=900.0,
                    help="min seconds between round starts (bound machine load)")
    ap.add_argument("--windows", default="10 15 20 40 60",
                    help="space-separated style windows to rotate")
    ap.add_argument("--limit-tickers", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "artifacts/alphaagent/rotation_validation_overnight"))
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = until deadline")
    return ap.parse_args()


def _default_deadline() -> datetime:
    now = datetime.now()
    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def main() -> int:
    args = _parse_args()
    deadline = (datetime.fromisoformat(args.deadline)
                if args.deadline else _default_deadline())
    windows = [int(w) for w in args.windows.split()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "overnight.log"
    summary_path = out / "summary.jsonl"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"overnight Phase1 validation start: deadline={deadline.isoformat()} "
        f"windows={windows} interval={args.interval_sec}s")
    rows: list[dict] = []
    round_no = 0
    while True:
        round_no += 1
        if args.max_rounds and round_no > args.max_rounds:
            log("max_rounds reached; stop")
            break
        if datetime.now() >= deadline:
            log("deadline reached; stop")

            break
        w = windows[(round_no - 1) % len(windows)]
        round_out = out / f"round-{round_no:03d}-w{w}"
        round_out.mkdir(parents=True, exist_ok=True)
        log(f"round {round_no}: style-window={w} -> {round_out.name}")
        t0 = time.time()
        cmd = [str(PY), str(VALIDATION),
               "--out", str(round_out),
               "--style-window", str(w),
               "--limit-tickers", str(args.limit_tickers)]
        try:
            res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                 timeout=60 * 30, encoding="utf-8", errors="replace")
            rc = res.returncode
        except subprocess.TimeoutExpired:
            rc = -1
        elapsed = round(time.time() - t0, 1)
        entry: dict = {
            "round": round_no, "style_window": w, "rc": rc,
            "elapsed_sec": elapsed, "started": (datetime.now() - timedelta(seconds=elapsed)).isoformat(timespec="seconds"),
        }
        report_path = round_out / "report.json"
        if report_path.is_file():
            try:
                rep = json.loads(report_path.read_text(encoding="utf-8"))
                s = rep.get("stages", {})
                entry["elapsed"] = rep.get("elapsed_seconds")
                entry["strategy_id"] = s.get("canonical", {}).get("strategy_id")
                entry["series_hash"] = s.get("state_sequence", {}).get("series_hash")
                entry["state_counts"] = s.get("state_sequence", {}).get("state_counts")
                entry["near_dup_rejected"] = s.get("registry", {}).get("near_duplicate_rejected")
                entry["repro_same_id"] = s.get("reproducibility", {}).get("same_id")
                entry["repro_same_hash"] = s.get("reproducibility", {}).get("same_series_hash")
                entry["health"] = s.get("carrier_health", {})
            except Exception as e:  # noqa: BLE001
                entry["parse_error"] = str(e)[:200]
        rows.append(entry)
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        log(f"round {round_no} done rc={rc} elapsed={elapsed}s")
        # bounded sleep: respect interval between round starts
        elapsed_here = max(0.0, time.time() - t0)
        sleep_for = max(0.0, args.interval_sec - elapsed_here)
        if sleep_for > 0:
            # wake in small slices so Ctrl/stop is responsive
            end = time.time() + sleep_for
            while time.time() < end:
                time.sleep(min(60.0, end - time.time()))

    summary_md = out / "summary.md"
    lines = [
        "# Rotation Phase 1 overnight validation rollup",
        "",
        f"- deadline: {deadline.isoformat()}",
        f"- rounds: {len(rows)}",
        "",
        "| round | window | rc | elapsed | strategy_id | series_hash | near_dup | repro |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('round')} | {r.get('style_window')} | {r.get('rc')} | "
            f"{r.get('elapsed')} | {r.get('strategy_id', '-')} | "
            f"{r.get('series_hash', '-')} | {r.get('near_dup_rejected', '-')} | "
            f"{r.get('repro_same_id')}/{r.get('repro_same_hash')} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"overnight done: {len(rows)} rounds -> {summary_path} / {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

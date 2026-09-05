# -*- coding: utf-8 -*-
"""Dump recent backfill runs + batch errors from the manifest."""
import json
import sqlite3

conn = sqlite3.connect(r"D:\Quant\quant_ui\CNEquity\data\quant_dataset\_cnequity\meta\manifest.db")
conn.row_factory = sqlite3.Row

print("=== backfill runs (latest 14) ===")
rows = conn.execute(
    """SELECT run_id, status, started_at, finished_at, error_message, metadata_json
       FROM ingestion_runs WHERE job_name='backfill'
       ORDER BY started_at DESC LIMIT 14"""
).fetchall()
for r in rows:
    md = json.loads(r["metadata_json"] or "{}")
    ds = md.get("dataset") or ",".join(md.get("datasets", []) or []) or "?"
    err = (r["error_message"] or "")[:160]
    print(f"{r['started_at'][:16]}  {ds:<26} {r['status']:<8} {err}")

print()
print("=== failed batches of recent backfill runs (latest 20) ===")
rows = conn.execute(
    """SELECT b.run_id, b.dataset, b.status, b.window_start, b.window_end,
              b.error_message, r.started_at
       FROM ingestion_batches b JOIN ingestion_runs r ON r.run_id = b.run_id
       WHERE r.job_name='backfill' AND b.status IN ('failed','stale','running')
       ORDER BY r.started_at DESC, b.dataset LIMIT 20"""
).fetchall()
for r in rows:
    err = (r["error_message"] or "")[:150].replace("\n", " | ")
    print(f"{r['started_at'][:16]}  {r['dataset']:<24} {r['status']:<8} "
          f"{r['window_start']}..{r['window_end']}  {err}")
conn.close()

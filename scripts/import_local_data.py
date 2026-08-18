#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Server-side receive for local machine -> server data import.

Local machine pushes files into a staging directory (default ~/quant_ui_import),
then runs this script (or the local sync wrapper triggers it via SSH).  It
backs up the current server files, atomically replaces them, and refreshes
DuckDB views so running services see the new data immediately.

Example (run from the local machine):
    rsync -a ./etf.csv ./etf_panel.parquet ... ubuntu@SERVER:~/quant_ui_import/
    ssh ubuntu@SERVER "python /home/ubuntu/quant/quant_ui/scripts/import_local_data.py"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.store import DATA_DIR  # noqa: E402

# Files owned by the local machine (relative to data dir).
LOCAL_FILES = [
    "etf.csv",
    "etf_panel.parquet",
    "fund.csv",
    "fund_nav.parquet",
    "fund_panel.parquet",
    "tech.csv",
    "index.csv",
    "universe.csv",
    "pg_parquet/report_rc.parquet",
]


def _backup(dest: Path, backup_dir: Path) -> None:
    """Copy the current server file into the timestamped backup dir."""
    rel = dest.relative_to(DATA_DIR)
    target = backup_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, target)
    print(f"  backup: {rel} -> {target}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path,
                    default=Path.home() / "quant_ui_import",
                    help="staging dir that local pushes files into")
    ap.add_argument("--backup-dir", type=Path,
                    default=Path.home() / "quant_sync_backup",
                    help="where previous server files are backed up")
    ap.add_argument("--extra", action="append", default=[],
                    help="additional relative path(s) to import (repeatable)")
    args = ap.parse_args()

    src_root: Path = args.src.expanduser()
    if not src_root.exists():
        print(f"staging dir not found: {src_root}", file=sys.stderr)
        return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = args.backup_dir.expanduser() / ts
    files = LOCAL_FILES + [e.lstrip("/") for e in args.extra]
    imported: list[str] = []
    missing: list[str] = []

    for rel in files:
        src = src_root / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dest = DATA_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            _backup(dest, backup_root)
        # Atomic replace: write tmp in the destination dir first.
        tmp = dest.with_name(f".{dest.name}.import.tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        imported.append(rel)
        print(f"  import: {rel}", flush=True)

    if not imported:
        print("nothing imported", flush=True)
    else:
        try:
            from core import db
            db.refresh_views()
            print("duckdb views refreshed", flush=True)
        except Exception as exc:
            print(f"duckdb view refresh failed: {exc}", file=sys.stderr, flush=True)

    print(f"imported={len(imported)} missing={len(missing)}", flush=True)
    if missing:
        print("missing in staging:", ", ".join(missing), flush=True)
    return 0 if imported else 1


if __name__ == "__main__":
    raise SystemExit(main())

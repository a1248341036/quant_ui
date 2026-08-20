#!/usr/bin/env python3
"""Check the remote data-status service and pull newer data before local startup.

The remote service is read-only from this script's perspective. It is queried
through the existing SSH host alias, and the local sync script is responsible
for pulling files and creating local backups.

Default behavior is fail-open: an unavailable remote service or a failed sync
does not prevent the local application from starting. Use ``--strict`` when a
caller needs a non-zero exit code for those failures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "scripts" / "sync_manifest.json"
SYNC_SCRIPT = ROOT / "scripts" / "sync_server.py"
LOCK_FILE = ROOT / "data" / ".startup_remote_sync.lock"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
ACTION_RE = re.compile(r"^\s+\[(download|upload|merge)\s+\]\s+(\S+)")


def log(message: str) -> None:
    print(f"[startup-sync] {message}", flush=True)


def read_config() -> tuple[str, str]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    host = os.getenv("QUANT_REMOTE_HOST", data["server"])
    url = os.getenv(
        "QUANT_REMOTE_STATUS_URL",
        "http://127.0.0.1:8001/api/status",
    )
    return host, url


def acquire_lock() -> bool:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
        except OSError:
            age = 0
        if age > 2 * 60 * 60:
            try:
                LOCK_FILE.unlink()
            except OSError:
                pass
            return acquire_lock()
        log("another startup sync is already running; skip")
        return False
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(f"pid={os.getpid()}\n")
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def remote_status(host: str, url: str, timeout: int) -> dict:
    ssh = shutil.which("ssh") or shutil.which("ssh.exe")
    if not ssh:
        raise RuntimeError("ssh executable not found")
    remote_cmd = f"curl -fsS --max-time {max(5, timeout)} {url!r}"
    proc = subprocess.run(
        [ssh, *SSH_OPTS, host, remote_cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout + 5,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(detail or f"ssh/curl exit {proc.returncode}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid status JSON: {exc}") from exc


def status_is_healthy(status: dict) -> bool:
    if status.get("overall") != "ok":
        return False
    sources = status.get("groups", {}).get("sources", {})
    return sources.get("status") == "ok"


def run_sync(dry_run: bool, timeout: int) -> tuple[int, list[tuple[str, str]], str]:
    # Startup is deliberately pull-only. The server's own refresh pipeline is
    # outside this script, and local_to_server items must not be touched here.
    command = [sys.executable, str(SYNC_SCRIPT), "--group", "server_to_local"]
    if not dry_run:
        command.append("--apply")
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=dry_run,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        timeout=timeout,
    )
    output = (proc.stdout or "") + (proc.stderr or "") if dry_run else ""
    actions = []
    if dry_run:
        for line in output.splitlines():
            match = ACTION_RE.match(line)
            if match:
                actions.append((match.group(1), match.group(2)))
        if output:
            print(output, end="")
    return proc.returncode, actions, output


def failure(message: str, strict: bool) -> int:
    log(f"warning: {message}")
    return 2 if strict else 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Check remote data_status and pull newer server data before local startup"
    )
    parser.add_argument("--dry-run", action="store_true", help="report local differences without applying sync")
    parser.add_argument("--strict", action="store_true", help="return non-zero when the check or sync fails")
    parser.add_argument("--timeout", type=int, default=1800, help="local sync timeout in seconds")
    args = parser.parse_args()

    host, url = read_config()
    try:
        status = remote_status(host, url, min(args.timeout, 30))
    except Exception as exc:
        return failure(f"remote data_status unavailable: {exc}", args.strict)

    sources = status.get("groups", {}).get("sources", {})
    log(
        f"remote={host} overall={status.get('overall')} "
        f"stock_daily={sources.get('stock_daily_max')} "
        f"lag_days={sources.get('lag_days')}"
    )
    if not status_is_healthy(status):
        return failure("remote data status is not healthy; local sync skipped", args.strict)

    if not acquire_lock():
        return 0
    try:
        code, actions, _ = run_sync(dry_run=True, timeout=args.timeout)
        if code:
            return failure(f"local sync dry-run failed with exit code {code}", args.strict)
        if not actions:
            log("local data is already aligned; no sync needed")
            return 0

        log(f"local data differs: {len(actions)} action(s)")
        if args.dry_run:
            return 0

        code, _, _ = run_sync(dry_run=False, timeout=args.timeout)
        if code:
            return failure(f"local sync failed with exit code {code}", args.strict)
        log("local sync completed")
        return 0
    except subprocess.TimeoutExpired:
        return failure("local sync timed out", args.strict)
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())

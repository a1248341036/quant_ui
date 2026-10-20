#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factorzoo 周维护：候选去重干跑审计 + 旧 run 归档 + 监控日志归档 + 通知。

三项都偏只读/可逆，删除类操作保持显式：
  1. dedup 审计：dedup_candidate_factors.py --dry-run（只列出冗余对，不删；
     确认后用 post_mining_pipeline.py --apply-dedup 真删）
  2. 旧 run 归档：终态且结束时间早于 --archive-days（缺省 14 天）的 run，
     逐个 POST /api/alphaagent/runs/{id}/archive；后端不可达则跳过（不算失败）
  3. 日志归档：--log-dir（缺省 Desktop\quant\logs）下 mtime 早于 --log-days
     （缺省 30 天）的 *.log 移入 <log-dir>\_old\（移动而非删除，可随时找回）

用法（建议挂周计划任务）：
  python scripts/factorzoo_weekly.py
  python scripts/factorzoo_weekly.py --archive-days 7 --no-notify
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_LOG_DIR = Path(r"C:\Users\zhoubw\Desktop\quant\logs")

TERMINAL_STATUSES = {"completed", "failed", "stopped", "interrupted"}

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _ts(value) -> float | None:
    """宽松解析时间戳：epoch 秒或 ISO 字符串；解析失败返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(value)).timestamp()
    except (ValueError, TypeError):
        return None


def pick_archivable(runs: list[dict], cutoff_ts: float,
                    statuses: set[str] = frozenset(TERMINAL_STATUSES)) -> list[str]:
    """选出可归档 run：终态且时间戳（宽松取 ended/updated/finished/created）早于 cutoff。"""
    out = []
    for r in runs:
        if r.get("status") not in statuses:
            continue
        ts = next((t for t in (_ts(r.get(k)) for k in
                               ("ended_at", "updated_at", "finished_at", "created_at"))
                   if t is not None), None)
        if ts is not None and ts < cutoff_ts:
            out.append(str(r.get("run_id")))
    return out


def archive_runs(backend: str, run_ids: list[str]) -> tuple[int, list[str]]:
    """逐个归档；返回 (成功数, 失败 run_id)。网络错误按失败计，不抛出。"""
    ok, failed = 0, []
    for rid in run_ids:
        req = urllib.request.Request(
            f"{backend.rstrip('/')}/api/alphaagent/runs/{rid}/archive",
            data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 200 <= resp.status < 300:
                    ok += 1
                else:
                    failed.append(rid)
        except (urllib.error.URLError, TimeoutError, OSError):
            failed.append(rid)
    return ok, failed


def rotate_logs(log_dir: Path, days: int, archive_name: str = "_old") -> int:
    """把 mtime 早于 days 的 *.log 移入 <log_dir>/<archive_name>/；返回移动数。"""
    log_dir, dst = Path(log_dir), Path(log_dir) / archive_name
    if not log_dir.is_dir():
        return 0
    cutoff = time.time() - days * 86400
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in log_dir.glob("*.log"):
        if f.stat().st_mtime < cutoff:
            shutil.move(str(f), str(dst / f.name))
            moved += 1
    return moved


def run_dedup_dry() -> int:
    """候选去重干跑审计（子进程透传输出，退出码即结果）。"""
    cmd = [sys.executable, str(SCRIPTS / "dedup_candidate_factors.py"), "--dry-run"]
    print(f"$ {' '.join(cmd[1:])}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="factorzoo 周维护（详见模块头注释）")
    ap.add_argument("--archive-days", type=int, default=14,
                    help="终态 run 归档阈值（天，按 run 结束时间计，缺省 14）")
    ap.add_argument("--log-days", type=int, default=30,
                    help="监控日志归档阈值（天，按 mtime 计，缺省 30）")
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument("--backend", default="http://127.0.0.1:17891")
    ap.add_argument("--no-notify", dest="notify", action="store_false", default=True)
    args = ap.parse_args()

    results: dict[str, str] = {}

    # 1. dedup 干跑审计（失败不阻断后两项，但计入通知）
    rc = run_dedup_dry()
    results["dedup_dry"] = "ok" if rc == 0 else f"rc={rc}"

    # 2. 旧 run 归档（后端不可达 → 跳过，不算失败）
    try:
        with urllib.request.urlopen(f"{args.backend.rstrip('/')}/api/alphaagent/runs",
                                    timeout=15) as r:
            data = json.loads(r.read().decode("utf-8") or "[]")
        runs = data if isinstance(data, list) else (data.get("runs") or [])
        cutoff = time.time() - args.archive_days * 86400
        ids = pick_archivable(runs, cutoff)
        if ids:
            ok, failed = archive_runs(args.backend, ids)
            results["archive"] = f"{ok}/{len(ids)} 归档" + (f"，失败 {failed}" if failed else "")
        else:
            results["archive"] = "无可归档 run"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        results["archive"] = f"后端不可达，跳过（{type(exc).__name__}）"

    # 3. 日志归档
    moved = rotate_logs(args.log_dir, args.log_days)
    results["log_rotate"] = f"移动 {moved} 个过期日志 → _old/"

    for k, v in results.items():
        print(f"[{k}] {v}")
    if args.notify:
        try:
            import ops_notify

            ops_notify.send("factorzoo 周维护完成",
                            "\n".join(f"{k}: {v}" for k, v in results.items()),
                            level="error" if rc != 0 else "info")
        except Exception as exc:  # noqa: BLE001
            print(f"通知失败（忽略）: {exc}")
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())

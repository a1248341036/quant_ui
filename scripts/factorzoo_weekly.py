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
    """选出可归档 run：终态且**创建时间**早于 cutoff。

    口径说明（OCR 2026-10-20 核实）：后端 /api/alphaagent/runs 快照只含
    created_at，没有 ended_at/updated_at 字段——归档阈值按创建时间计。
    已知取舍：久创建、晚结束的 run 会被提前归档；归档可找回，不删数据。
    """
    out = []
    for r in runs:
        if r.get("status") not in statuses:
            continue
        ts = _ts(r.get("created_at"))
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
            dst_file = dst / f.name
            if dst_file.exists():  # 同名保留双方：旧归档不被覆盖，新文件追加时间戳
                dst_file = dst / f"{f.stem}_{time.strftime('%Y%m%d%H%M%S')}{f.suffix}"
            shutil.move(str(f), str(dst_file))
            moved += 1
    return moved


def run_dedup_dry() -> int:
    """候选去重干跑审计（子进程透传输出，退出码即结果）。"""
    cmd = [sys.executable, str(SCRIPTS / "dedup_candidate_factors.py"), "--dry-run"]
    print(f"$ {' '.join(cmd[1:])}", flush=True)
    try:
        return subprocess.run(cmd).returncode
    except OSError as exc:  # 脚本被删/解释器不可用：不阻断后两项
        print(f"dedup 干跑启动失败：{exc}", flush=True)
        return 127


def main() -> int:
    ap = argparse.ArgumentParser(description="factorzoo 周维护（详见模块头注释）")
    ap.add_argument("--archive-days", type=int, default=14,
                    help="终态 run 归档阈值（天，**按创建时间计**——后端快照只含 created_at；缺省 14）")
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

    # 2. 旧 run 归档（后端不可达 → 跳过，不算失败；per-run 失败 → 计入退出码）
    archive_failed = False
    try:
        with urllib.request.urlopen(f"{args.backend.rstrip('/')}/api/alphaagent/runs",
                                    timeout=15) as r:
            data = json.loads(r.read().decode("utf-8") or "[]")
        runs = data if isinstance(data, list) else (data.get("runs") or [])
        cutoff = time.time() - args.archive_days * 86400
        ids = pick_archivable(runs, cutoff)
        if ids:
            ok, failed = archive_runs(args.backend, ids)
            archive_failed = bool(failed)
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
        except ImportError:
            print("通知跳过：ops_notify 不在当前分支（随 feat/overnight-finalize 合并后生效）")
        except Exception as exc:  # noqa: BLE001
            print(f"通知失败（忽略）: {exc}")
    # 退出码语义：dedup 失败优先；其次归档 per-run 失败（无人值守任务需要感知）
    return rc if rc != 0 else (1 if archive_failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())

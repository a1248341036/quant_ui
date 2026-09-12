"""整夜挖掘监控服务：以后端子进程托管 overnight_mining_monitor.py。

与 stacking_service 同模式：subprocess.Popen + threading 管理，
提供启动/停止/状态查询能力，前端悬浮窗通过 REST API 交互。

脚本路径：C:\\Users\\zhoubw\\Desktop\\quant\\overnight_mining_monitor.py
（独立于仓库的桌面脚本，后端只做进程托管 + 日志尾读）
"""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = ROOT / ".venv" / "Scripts" / "python.exe"
MONITOR_SCRIPT = Path(r"C:\Users\zhoubw\Desktop\quant\overnight_mining_monitor.py")
LOG_DIR = Path(r"C:\Users\zhoubw\Desktop\quant\logs")
# 状态快照（供前端轮询）：monitor 子进程状态 + 日志尾 + 最近 run 概要
STATE_FILE = LOG_DIR / "overnight_monitor_state.json"

_lock = threading.Lock()
_current: dict[str, Any] | None = None  # {proc, started_at, params, log_file}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_state(status: str, **extra: Any) -> None:
    """写状态快照（供前端轮询，避免每次都读日志文件）。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        snap = {"status": status, "updated_at": _now_iso(), **extra}
        STATE_FILE.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _read_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {"status": "idle"}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "idle"}


def _proc_status() -> tuple[str, dict[str, Any] | None]:
    global _current
    with _lock:
        cur = _current
        if cur is None:
            return "idle", None
        code = cur["proc"].poll()
        if code is None:
            return "running", cur
        status = "completed" if code == 0 else "failed"
        cur["exit_code"] = code
        # 进程已结束：写最终状态后清空 _current
        _write_state(status, exit_code=code,
                     params=cur.get("params"), finished_at=_now_iso())
        _current = None
        return status, cur


def _tail_log(log_path: Path, lines: int = 80) -> list[str]:
    if not log_path.is_file():
        return []
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return all_lines[-lines:]
    except OSError:
        return []


def start_monitor(params: dict[str, Any]) -> dict[str, Any]:
    """启动整夜挖掘监控子进程。

    参数映射到 overnight_mining_monitor.py 的 CLI 参数：
      deadline:   --deadline（HH:MM，缺省 07:00）
      max_runs:   --max-runs（0=不限）
      focus_facets: --focus-facets（逗号分隔，空=自由探索）
      message:    --message（覆盖初始用户消息）
      max_turns:  --max-turns（0=后端默认）
      restart_backend: --restart-backend / --no-restart-backend
      keep_awake: --keep-awake / --no-keep-awake
      poll:       --poll（轮询秒数）
    """
    global _current
    with _lock:
        if _current is not None and _current["proc"].poll() is None:
            return {"error": "monitor_already_running"}

        if not MONITOR_SCRIPT.is_file():
            return {"error": f"script_not_found: {MONITOR_SCRIPT}"}

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"overnight_monitor_{datetime.now().strftime('%Y%m%d')}.log"

        command = [
            str(PYTHON_EXECUTABLE), str(MONITOR_SCRIPT),
            "--repo", str(ROOT),
            "--deadline", str(params.get("deadline") or "07:00"),
            "--poll", str(int(params.get("poll") or 60)),
        ]
        max_runs = int(params.get("max_runs") or 0)
        if max_runs > 0:
            command += ["--max-runs", str(max_runs)]
        focus = params.get("focus_facets")
        if focus:
            if isinstance(focus, list):
                focus = ",".join(str(f) for f in focus)
            if focus.strip():
                command += ["--focus-facets", str(focus)]
        message = params.get("message")
        if message:
            command += ["--message", str(message)]
        max_turns = int(params.get("max_turns") or 0)
        if max_turns > 0:
            command += ["--max-turns", str(max_turns)]
        if params.get("restart_backend") is False:
            command += ["--no-restart-backend"]
        if params.get("keep_awake") is False:
            command += ["--no-keep-awake"]

        # 日志追加写入（脚本自己也写同一文件，但 stdout 可能有些行不同步）
        log_handle = log_file.open("a", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()

        _current = {
            "proc": proc,
            "started_at": _now_iso(),
            "params": params,
            "log_file": str(log_file),
        }
        _write_state("running", started_at=_current["started_at"],
                     params=params, log_file=str(log_file))
        return {"ok": True, "status": "running", "log_file": str(log_file)}


def stop_monitor() -> dict[str, Any]:
    global _current
    with _lock:
        cur = _current
        if cur is None or cur["proc"].poll() is not None:
            _current = None
            _write_state("idle")
            return {"ok": True, "status": "idle", "note": "monitor_not_running"}
        cur["proc"].kill()
        _write_state("stopped", params=cur.get("params"), stopped_at=_now_iso())
        _current = None
        return {"ok": True, "status": "stopped"}


def _active_runs(tail: int = 40) -> list[dict[str, Any]]:
    """当前活跃挖掘 run 摘要（starting/running/stopping），供悬浮窗停止单个 run。

    懒惰导入 alphaagent_service：避免模块级循环依赖（routers 同时加载两者）。
    并发数本身受 alphaagent_service 准入控制约束，这里只做只读查询。
    """
    try:
        from backend import alphaagent_service as svc

        runs = svc.list_runs(include_archived=True)
    except Exception:  # noqa: BLE001 — 查询失败不影响监控状态返回
        return []
    active = [
        r for r in runs
        if r.get("status") in {"starting", "running", "stopping"}
        and not r.get("archived")
    ]
    out: list[dict[str, Any]] = []
    for r in active:
        evs = r.get("events") or []
        out.append({
            "run_id": r.get("run_id"),
            "status": r.get("status"),
            "title": (r.get("title") or "")[:60],
            "started_at": r.get("created_at"),
            "event_count": r.get("event_count") or len(evs),
            "last_event": ((evs[-1] or {}).get("event") if evs else None),
            "outcome": r.get("outcome"),
        })
    return out


def get_status(tail_lines: int = 80) -> dict[str, Any]:
    """当前监控状态 + 日志尾 + 脚本日志尾（合并视图）。"""
    status, cur = _proc_status()
    result: dict[str, Any] = {"status": status}

    # 当前仍在运行的挖掘 run（供悬浮窗「停止当前 run」逐条操作）
    result["active_runs"] = _active_runs()

    # 从状态文件补充上次更新时间/参数（即使 _current 已清空也能看到历史）
    saved = _read_state()
    if saved.get("updated_at"):
        result["updated_at"] = saved["updated_at"]
    if saved.get("started_at"):
        result["started_at"] = saved["started_at"]
    if saved.get("params"):
        result["params"] = saved["params"]
    if saved.get("exit_code") is not None:
        result["exit_code"] = saved["exit_code"]
    if saved.get("finished_at"):
        result["finished_at"] = saved["finished_at"]

    # 日志尾
    log_path = None
    if cur and cur.get("log_file"):
        log_path = Path(cur["log_file"])
    elif saved.get("log_file"):
        log_path = Path(saved["log_file"])
    if log_path and log_path.is_file():
        result["log_tail"] = _tail_log(log_path, tail_lines)
        try:
            result["log_size"] = log_path.stat().st_size
        except OSError:
            pass
    else:
        result["log_tail"] = []

    return result

"""ML 组合（stacking）训练运行管理。

与挖掘 run 同模式：子进程执行 scripts/train_ml_composite.py，stdout 逐行写入
进度日志（progress.log），完成后 report.json 落盘到确定性输出目录。
同一时间只允许一个训练（panel 全量物化内存重，并发只会互相拖慢）。
"""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STACKING_ROOT = ROOT / "artifacts" / "alphaagent" / "stacking"
PYTHON_EXECUTABLE = ROOT / ".venv" / "Scripts" / "python.exe"

_lock = threading.Lock()
_current: dict[str, Any] | None = None  # {train_id, proc, out_dir, log_path, params}


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _progress_log(train_id: str) -> Path:
    d = ROOT / "artifacts" / "alphaagent" / "stacking_ui" / train_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "progress.log"


def start_training(params: dict[str, Any]) -> dict[str, Any]:
    """启动一次 ML 组合训练。返回 {train_id, status} 或 {error}。"""
    global _current
    with _lock:
        if _current is not None and _current["proc"].poll() is None:
            return {"error": "training_already_running",
                    "train_id": _current["train_id"]}

        train_id = _now_id()
        modes = params.get("modes") or ["technical", "fundamental"]
        if isinstance(modes, str):
            modes = [modes]
        modes = [str(m) for m in modes]
        out_dir = STACKING_ROOT / train_id
        log_path = _progress_log(train_id)

        command = [
            str(PYTHON_EXECUTABLE), str(ROOT / "scripts" / "train_ml_composite.py"),
            "--modes", *modes,
            "--model", str(params.get("model") or "both"),
            "--label-days", str(int(params.get("label_days") or 5)),
            "--train-months", str(int(params.get("train_months") or 18)),
            "--step-months", str(int(params.get("step_months") or 6)),
            "--purge-days", str(int(params.get("purge_days") or 5)),
            "--warmup-days", str(int(params.get("warmup_days") or 250)),
            "--max-corr", str(float(params.get("max_corr") or 0.6)),
            "--out-dir", str(out_dir),
        ]
        if params.get("mining_end"):
            command += ["--mining-end", str(params["mining_end"])]
        if params.get("no_candidate"):
            command += ["--no-candidate"]
        if params.get("no_gate"):
            command += ["--no-gate"]
        command += ["--isolation", str(params.get("isolation") or "holdout")]
        if params.get("size_neutral") is False:
            command += ["--no-size-neutral"]

        log_handle = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()  # Popen 已继承句柄

        _current = {
            "train_id": train_id,
            "proc": proc,
            "out_dir": out_dir,
            "log_path": log_path,
            "params": params,
        }
        return {"train_id": train_id, "status": "running", "out_dir": str(out_dir)}


def _proc_status() -> tuple[str, dict[str, Any] | None]:
    """返回 (status, _current)。running 时等待子进程结束则转为完成/失败。"""
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
        return status, cur


def list_trainings(limit: int = 30) -> list[dict[str, Any]]:
    """列出历史训练：磁盘上所有 stacking 输出目录 + 当前运行状态。"""
    status, cur = _proc_status()
    running_id = cur["train_id"] if cur and status == "running" else None
    out: list[dict[str, Any]] = []
    if STACKING_ROOT.is_dir():
        for d in sorted(STACKING_ROOT.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            report_path = d / "report.json"
            item: dict[str, Any] = {
                "train_id": d.name,
                "status": "running" if d.name == running_id else
                          ("completed" if report_path.is_file() else "unknown"),
                "out_dir": str(d),
            }
            if d.name == running_id:
                item["status"] = "running"
            elif cur is not None and d.name == cur["train_id"] and status == "failed":
                item["status"] = "failed"
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    blended = report.get("oos_ic_blended") or {}
                    gate = report.get("gate") or {}
                    item["oos_ic_mean"] = blended.get("ic_mean")
                    item["oos_ic_ir"] = blended.get("ic_ir")
                    item["n_folds"] = report.get("folds")
                    item["n_features"] = len(report.get("feature_names") or [])
                    item["gate_passed"] = gate.get("passed")
                    item["time_isolation"] = report.get("time_isolation")
                except (json.JSONDecodeError, OSError):
                    pass
            out.append(item)
        # running 状态但目录尚未创建（panel 加载阶段）也补一条
        if running_id and not any(x["train_id"] == running_id for x in out):
            out.insert(0, {"train_id": running_id, "status": "running"})
    return out[: max(1, limit)]


def get_training(train_id: str, *, tail_lines: int = 40) -> dict[str, Any]:
    """单个训练详情：状态 + 进度日志尾部 + 完成时的 report。"""
    status, cur = _proc_status()
    running_id = cur["train_id"] if cur else None
    log_path = _progress_log(train_id)
    tail: list[str] = []
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-tail_lines:]

    out: dict[str, Any] = {"train_id": train_id, "status": "unknown", "progress_tail": tail}
    if train_id == running_id:
        out["status"] = "running"
    out_dir = STACKING_ROOT / train_id
    report_path = out_dir / "report.json"
    if train_id != running_id:
        out["status"] = "completed" if report_path.is_file() else "unknown"
    if report_path.is_file():
        try:
            out["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            out["report_error"] = str(exc)
    elif train_id == running_id:
        pass
    else:
        out["progress_tail"] = tail or [f"未找到训练 {train_id} 的输出"]
    return out


def stop_training(train_id: str) -> dict[str, Any]:
    global _current
    with _lock:
        cur = _current
        if cur is None or cur["train_id"] != train_id:
            return {"ok": False, "error": "training_not_running"}
        if cur["proc"].poll() is None:
            cur["proc"].kill()
        return {"ok": True, "train_id": train_id}

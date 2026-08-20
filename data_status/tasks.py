"""Serial script-task runner.

Scripts (healthcheck / sync / export / refresh / rebuild-panel) are launched
as subprocesses one at a time to protect the 3.6G machine from concurrent
heavy jobs. Task state is persisted to tasks.json so restarts keep history.
"""
from __future__ import annotations

import queue
import shlex
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import LOGS_DIR, PYTHON, QUANT_UI_ROOT, SCRIPTS_DIR, TASK_CATALOG
from .state import load_tasks, save_tasks


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_PARAM_TYPES = {"str": str, "int": int, "float": float, "bool": bool}


def _task_defs() -> dict:
    """Convert the JSON task catalog to the runner's validated command specs."""
    defs = {}
    for item in TASK_CATALOG:
        spec = dict(item)
        params = {}
        for name, type_name in spec.get("params", {}).items():
            try:
                params[name] = _PARAM_TYPES[type_name]
            except KeyError as exc:
                raise ValueError(f"unknown task parameter type: {type_name}") from exc
        spec["params"] = params
        defs[spec["id"]] = spec
    return defs


TASK_DEFS = _task_defs()


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_command(task_type: str, params: dict) -> list[str]:
    spec = TASK_DEFS.get(task_type)
    if not spec:
        raise ValueError(f"unknown task type: {task_type}")
    argv = [PYTHON, str(SCRIPTS_DIR / spec["script"])]
    argv.extend(spec.get("base", []))
    for key in spec.get("required", []):
        if key not in params:
            raise ValueError(f"missing required param: {key}")
    for key, typ in spec.get("params", {}).items():
        if key not in params:
            continue
        value = params[key]
        if typ is bool:
            if value is True:
                argv.append(_flag(key))
            continue
        if not isinstance(value, typ):
            raise ValueError(f"param {key} must be {typ.__name__}")
        if typ is str:
            if any(c in value for c in (";", "|", "&", "$", "`", "\n", "\r")):
                raise ValueError(f"unsafe value for param {key}")
        argv.extend([_flag(key), str(value)])
    return argv


class TaskRunner:
    def __init__(self) -> None:
        self._queue: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._current_id: str | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="task-runner")
        self._thread.start()

    def _tasks(self) -> list[dict]:
        return load_tasks()

    def _save(self, tasks: list[dict]) -> None:
        save_tasks(tasks)

    def list(self) -> list[dict]:
        with self._lock:
            return list(reversed(load_tasks()))

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            for t in load_tasks():
                if t["id"] == task_id:
                    return t
        return None

    def submit(self, task_type: str, params: dict | None = None) -> dict:
        params = params or {}
        argv = build_command(task_type, params)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        task = {
            "id": uuid.uuid4().hex[:12],
            "type": task_type,
            "params": params,
            "argv": argv,
            "status": "queued",
            "created_at": _utcnow(),
            "log": str(LOGS_DIR / f"{uuid.uuid4().hex[:8]}.log"),
        }
        with self._lock:
            tasks = self._tasks()
            tasks.append(task)
            self._save(tasks)
        self._queue.put(task)
        return task

    def cancel(self, task_id: str) -> dict:
        with self._lock:
            if self._current_id == task_id and self._proc and self._proc.poll() is None:
                self._proc.terminate()
                return {"ok": True, "detail": "terminating"}
        tasks = self._tasks()
        for t in tasks:
            if t["id"] == task_id and t["status"] == "queued":
                t["status"] = "cancelled"
                self._save(tasks)
                return {"ok": True, "detail": "cancelled from queue"}
        return {"ok": False, "detail": "task not running or queued"}

    def _update(self, task_id: str, **fields) -> None:
        with self._lock:
            tasks = self._tasks()
            for t in tasks:
                if t["id"] == task_id:
                    t.update(fields)
                    break
            self._save(tasks)

    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                self._run(task)
            except Exception as exc:
                self._update(task["id"], status="failed", error=str(exc), finished_at=_utcnow())
            finally:
                self._current_id = None
                self._proc = None

    def _run(self, task: dict) -> None:
        self._current_id = task["id"]
        self._update(task["id"], status="running", started_at=_utcnow())
        log_path = Path(task["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(shlex.quote(a) for a in task['argv'])}\n\n")
            logf.flush()
            try:
                proc = subprocess.Popen(
                    task["argv"],
                    cwd=str(QUANT_UI_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception as exc:
                self._update(task["id"], status="failed", error=str(exc), finished_at=_utcnow())
                return
            self._proc = proc
            rc = proc.wait()
            status = "success" if rc == 0 else "failed"
            self._update(
                task["id"],
                status=status,
                exit_code=rc,
                finished_at=_utcnow(),
            )


_runner: TaskRunner | None = None


def get_runner() -> TaskRunner:
    global _runner
    if _runner is None:
        _runner = TaskRunner()
    return _runner

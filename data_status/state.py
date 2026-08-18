"""Atomic JSON state helpers used by the collector and task runner."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from .config import STATE_FILE, TASKS_FILE


def _atomic_write(path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_read(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return default


def load_state() -> dict:
    return _atomic_read(STATE_FILE, {})


def save_state(state: dict) -> None:
    _atomic_write(STATE_FILE, state)


def load_tasks() -> list[dict]:
    return _atomic_read(TASKS_FILE, [])


def save_tasks(tasks: list[dict]) -> None:
    _atomic_write(TASKS_FILE, tasks)

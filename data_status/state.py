"""Atomic JSON state helpers used by the collector and task runner."""
from __future__ import annotations

import json
from typing import Any

from core.atomicio import atomic_write_text

from .config import STATE_FILE, TASKS_FILE


def _atomic_write_json(path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, default=str))


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
    _atomic_write_json(STATE_FILE, state)


def load_tasks() -> list[dict]:
    return _atomic_read(TASKS_FILE, [])


def save_tasks(tasks: list[dict]) -> None:
    _atomic_write_json(TASKS_FILE, tasks)

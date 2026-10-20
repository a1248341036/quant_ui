# -*- coding: utf-8 -*-
"""factorzoo_weekly 单测：归档挑选、日志轮转；dedup 子进程不在单测内真跑。"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_fw():
    spec = importlib.util.spec_from_file_location("factorzoo_weekly", SCRIPTS / "factorzoo_weekly.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_archivable_terminal_and_old():
    """归档按 created_at 口径（后端快照只含该字段，OCR 2026-10-20 核实）。"""
    fw = _load_fw()
    now = time.time()
    runs = [
        {"run_id": "old-done", "status": "completed", "created_at": now - 20 * 86400},
        {"run_id": "recent-done", "status": "completed", "created_at": now - 2 * 86400},
        {"run_id": "old-running", "status": "running", "created_at": now - 20 * 86400},
        {"run_id": "old-no-ts", "status": "failed"},               # 无时间戳 → 不敢归档
        {"run_id": "old-iso", "status": "stopped",
         "created_at": "2026-09-01T00:00:00"},                     # ISO 时间戳可解析
    ]
    picked = fw.pick_archivable(runs, cutoff_ts=now - 14 * 86400)
    assert picked == ["old-done", "old-iso"]


def test_rotate_logs_moves_old_only(tmp_path: Path):
    fw = _load_fw()
    old, new = tmp_path / "monitor_20260901.log", tmp_path / "monitor_today.log"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    past = time.time() - 40 * 86400
    os.utime(old, (past, past))
    moved = fw.rotate_logs(tmp_path, days=30)
    assert moved == 1
    assert (tmp_path / "_old" / old.name).is_file()
    assert new.is_file() and not (tmp_path / "_old" / new.name).exists()
    # 幂等：再跑一遍不移动任何东西
    assert fw.rotate_logs(tmp_path, days=30) == 0
    # 同名再移动：旧归档不被覆盖，新文件带时间戳落在 _old/
    again = tmp_path / "monitor_20260901.log"
    again.write_text("again", encoding="utf-8")
    past2 = time.time() - 35 * 86400
    os.utime(again, (past2, past2))
    assert fw.rotate_logs(tmp_path, days=30) == 1
    archived = list((tmp_path / "_old").glob("monitor_20260901*.log"))
    assert len(archived) == 2

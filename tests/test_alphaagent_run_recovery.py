# -*- coding: utf-8 -*-
"""后端重启后的 run 状态恢复：孤儿进程 pid 判定与「继续」可用性。

背景：后台强制中断（重启/断电）会杀死挖矿子进程但不写终态事件，旧实现只靠
“JSONL 15 分钟无新事件”降级 running→interrupted——期间前端一直绿点，用户点
「继续」走的是追加消息队列而不是原地续跑，消息无人消费，功能看起来坏了。
修复：spawn 时把 pid 持久化到 run_meta，恢复路径探测 pid；已死立即 interrupted。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend import alphaagent_service as svc


def _make_run_dir(tmp_path: Path, run_id: str, pid: int | None) -> Path:
    d = tmp_path / run_id
    d.mkdir()
    (d / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-09-09T00:00:00+00:00",
                "params": {},
                "pid": pid,
            }
        ),
        encoding="utf-8",
    )
    with (d / "run_20260909_000000.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "user_message",
                    "content": "start",
                }
            )
            + "\n"
        )
    return d


def test_pid_alive_probe() -> None:
    assert svc._pid_is_alive(None) is None
    assert svc._pid_is_alive(0) is None
    assert svc._pid_is_alive(-3) is None
    assert svc._pid_is_alive(999_999_999) is False
    import os

    assert svc._pid_is_alive(os.getpid()) is True


def test_dead_pid_restore_is_interrupted(tmp_path) -> None:
    dead = subprocess.Popen([sys.executable, "-c", "pass"]).pid
    # 让子进程确实退出
    import time

    time.sleep(0.4)
    d = _make_run_dir(tmp_path, "deadrun", dead)
    run = svc._load_run_from_disk(d)
    assert run is not None
    assert run.pid == dead
    # JSONL 时间戳很新鲜，但 pid 已死 → 不允许再挂 running
    assert run.status == "interrupted"
    run.refresh()
    assert run.status == "interrupted"


def test_alive_pid_restore_stays_running(tmp_path) -> None:
    import os

    d = _make_run_dir(tmp_path, "liverun", os.getpid())
    run = svc._load_run_from_disk(d)
    assert run is not None
    assert run.status == "running"


def test_save_meta_persists_pid_and_status(tmp_path) -> None:
    import time

    run = svc.AgentRun(run_id="meta", command=[], log_dir=tmp_path)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    run.process = proc
    run.pid = proc.pid
    run.save_meta()
    meta = json.loads(run.meta_file.read_text(encoding="utf-8"))
    assert meta["pid"] == proc.pid
    assert "status" in meta
    proc.wait()
    time.sleep(0.2)

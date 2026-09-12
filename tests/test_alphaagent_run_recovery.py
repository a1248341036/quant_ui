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


# ── 活跃 run 计数：退出进程句柄不得占用并发额度（2026-09-12 泄漏修复） ──
# 背景：spawn 的 run 进程退出后 process 引用仍留在 _RUNS，旧 _active_run_count
# 只判 process is not None → failed/completed run 永久占坑，新 run 全被 429 拒绝。
# 修复：有句柄看 poll()（None=存活），无句柄按轨迹状态（running/starting/stopping）。


def _reset_service_state(monkeypatch, tmp_path: Path) -> None:
    """隔离全局 _RUNS/LOG_ROOT，让测试不触碰真实运行日志。"""
    monkeypatch.setattr(svc, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(svc, "_RUNS", {})
    monkeypatch.setattr(svc, "_LOCK", __import__("threading").Lock())


def test_active_count_excludes_exited_process(tmp_path, monkeypatch) -> None:
    import time

    _reset_service_state(monkeypatch, tmp_path)
    d = _make_run_dir(tmp_path, "exitedrun", None)
    run = svc.AgentRun(run_id="exitedrun", command=[], log_dir=d)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    run.process = proc
    run.pid = proc.pid
    proc.wait()
    time.sleep(0.2)  # 确保进程退出可见
    run.status = "failed"
    svc._RUNS[run.run_id] = run
    # 进程已退出：即便句柄仍在内存，也不应占用并发额度
    assert svc._active_run_count() == 0


def test_active_count_includes_live_process(tmp_path, monkeypatch) -> None:
    _reset_service_state(monkeypatch, tmp_path)
    d = _make_run_dir(tmp_path, "liverun2", None)
    run = svc.AgentRun(run_id="liverun2", command=[], log_dir=d)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run.process = proc
        run.pid = proc.pid
        run.status = "running"
        svc._RUNS[run.run_id] = run
        assert svc._active_run_count() == 1
    finally:
        proc.kill()
        proc.wait()


def test_active_count_uses_event_status_for_handleless_runs(tmp_path, monkeypatch) -> None:
    _reset_service_state(monkeypatch, tmp_path)
    # 无句柄 + 终态事件（session_end）：refresh 后不算活跃
    d = _make_run_dir(tmp_path, "done-run", None)
    with (d / "run_20260909_000001.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                             "event": "session_end"}) + "\n")
    run = svc._load_run_from_disk(d)
    assert run is not None
    assert run.status == "completed"
    svc._RUNS[run.run_id] = run
    assert svc._active_run_count() == 0

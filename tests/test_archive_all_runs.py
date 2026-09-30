# -*- coding: utf-8 -*-
"""批量归档（全部归档）+ 取消归档 + 运行中跳过口径。

背景（2026-09-30）：最近任务只能一条条归档，加「全部归档」按钮。批量动作必须与
「一键删除」同口径地跳过仍在推进的 run——归档后它会从最近任务列表消失，正在跑的
任务失联比"多留一行"麻烦得多。同时把单条归档改成可逆（后端早已支持 archived=false，
前端此前不传 body，永远只能归档）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from backend import alphaagent_service as svc


@pytest.fixture()
def isolated_runs(tmp_path, monkeypatch):
    """隔离全局 _RUNS / LOG_ROOT：只用本用例构造的 run 目录，不碰真实运行日志。"""
    monkeypatch.setattr(svc, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(svc, "_RUNS", {})
    return tmp_path


def _make_run(
    root: Path,
    run_id: str,
    *,
    archived: bool = False,
    active: bool = False,
) -> svc.AgentRun:
    """建一个 run 目录并登记到 _RUNS。active=True 表示轨迹"刚写过"（视为运行中）。"""
    d = root / run_id
    d.mkdir()
    jsonl = d / "run_20260101_000000.jsonl"
    jsonl.write_text(json.dumps({"event": "user_message", "content": "x"}) + "\n", encoding="utf-8")
    if not active:
        old = time.time() - 3600  # 静默 1 小时 → run_is_active() 为 False
        os.utime(jsonl, (old, old))
    run = svc.AgentRun(run_id=run_id, command=[], log_dir=d, params={"user_message": "x"})
    run.archived = archived
    svc._RUNS[run_id] = run
    return run


def _meta(root: Path, run_id: str) -> dict:
    return json.loads((root / run_id / "run_meta.json").read_text(encoding="utf-8"))


def test_run_is_active_needs_process_or_fresh_trace(isolated_runs) -> None:
    assert svc.run_is_active(_make_run(isolated_runs, "live", active=True)) is True
    assert svc.run_is_active(_make_run(isolated_runs, "idle", active=False)) is False


def test_archive_all_archives_idle_and_skips_active(isolated_runs) -> None:
    a = _make_run(isolated_runs, "old1")
    b = _make_run(isolated_runs, "old2")
    live = _make_run(isolated_runs, "live", active=True)
    done = _make_run(isolated_runs, "done", archived=True)

    result = svc.archive_all_runs()

    assert sorted(result["archived"]) == ["old1", "old2"]
    assert result["skipped"] == ["live"]
    assert result["count"] == 2
    assert a.archived is True and b.archived is True
    assert live.archived is False  # 运行中不隐藏
    assert done.archived is True
    # 落盘：后端重启后仍保持归档态；被跳过的 run 不写 meta（没动它）
    assert _meta(isolated_runs, "old1")["archived"] is True
    assert not (isolated_runs / "live" / "run_meta.json").exists()


def test_archive_all_noop_when_all_archived(isolated_runs) -> None:
    _make_run(isolated_runs, "done1", archived=True)
    _make_run(isolated_runs, "done2", archived=True)
    result = svc.archive_all_runs()
    assert result == {"ok": True, "archived": [], "skipped": [], "count": 0}


def test_archive_run_is_reversible(isolated_runs) -> None:
    run = _make_run(isolated_runs, "back", archived=True)
    svc.archive_run("back", archived=False)
    assert run.archived is False
    assert _meta(isolated_runs, "back")["archived"] is False

    svc.archive_run("back", archived=True)
    assert run.archived is True
    assert _meta(isolated_runs, "back")["archived"] is True


def test_delete_run_still_skips_active(isolated_runs) -> None:
    """delete_run 改用 run_is_active 后口径不变：活跃拒绝删、静默可删。"""
    live = _make_run(isolated_runs, "live", active=True)
    assert svc.delete_run("live") == {"run_id": "live", "deleted": False, "reason": "run_still_running"}
    assert live.log_dir.is_dir()

    _make_run(isolated_runs, "idle")
    assert svc.delete_run("idle")["deleted"] is True
    assert not (isolated_runs / "idle").exists()


def test_archive_all_endpoint_delegates(monkeypatch) -> None:
    from backend.routers import alphaagent as router_mod

    calls = {"n": 0}

    def _fake():
        calls["n"] += 1
        return {"ok": True, "archived": ["a"], "skipped": [], "count": 1}

    monkeypatch.setattr(router_mod.service, "archive_all_runs", _fake)
    assert router_mod.archive_all_runs()["count"] == 1
    assert calls["n"] == 1


def test_archived_routes_registered_before_run_id_route() -> None:
    """/runs/archived 必须在 /runs/{run_id} 之前注册，否则 DELETE 会被路径参数吞掉。"""
    from backend.routers import alphaagent as router_mod

    rows = [
        (getattr(r, "path", ""), set(getattr(r, "methods", ()) or ()))
        for r in router_mod.router.routes
    ]

    def _index(path: str, method: str) -> int:
        for i, (p, methods) in enumerate(rows):
            if p == path and method in methods:
                return i
        raise AssertionError(f"{method} {path} 未注册")

    idx_archive_delete = _index("/api/alphaagent/runs/archived", "DELETE")
    idx_archive_post = _index("/api/alphaagent/runs/archived", "POST")
    idx_run_id_delete = _index("/api/alphaagent/runs/{run_id}", "DELETE")
    assert idx_archive_delete < idx_run_id_delete
    assert idx_archive_post < idx_run_id_delete

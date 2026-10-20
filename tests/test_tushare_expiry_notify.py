# -*- coding: utf-8 -*-
"""tushare_expiry_notify 单测：提醒窗口边界、消息要素、主流程 dry-run/去重/缺配置。"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import date, timedelta
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_ttn():
    spec = importlib.util.spec_from_file_location(
        "tushare_expiry_notify", SCRIPTS / "tushare_expiry_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pick_phase_windows():
    ttn = _load_ttn()
    assert ttn.pick_phase(8) is None            # 太远不打扰
    assert ttn.pick_phase(7) == (7, "d7")
    assert ttn.pick_phase(4) == (7, "d7")
    assert ttn.pick_phase(3) == (3, "d3")
    assert ttn.pick_phase(2) == (3, "d3")
    assert ttn.pick_phase(1) == (1, "d1")
    assert ttn.pick_phase(0) == (0, "d0")
    assert ttn.pick_phase(-2) == (-3, "overdue3")
    assert ttn.pick_phase(-4) is None           # 过期太久不再提醒


def test_build_message_key_facts():
    ttn = _load_ttn()
    msg = ttn.build_message(3, "Tushare token 还有 3 天到期")
    assert "剩余 3 天" in msg and "TUSHARE_EXPIRE_DATE" in msg
    over = ttn.build_message(-2, "已过期")
    assert "已过期 2 天" in over and "续费" in over


def _setup_env(tmp_path: Path, monkeypatch, days_left: int):
    ttn = _load_ttn()
    expire = (date.today() + timedelta(days=days_left)).isoformat()
    (tmp_path / ".env").write_text(f"TUSHARE_EXPIRE_DATE={expire}\n", encoding="utf-8")
    monkeypatch.setattr(ttn, "QUANT_ROOT", tmp_path)
    monkeypatch.setattr(ttn, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.delenv("TUSHARE_EXPIRE_DATE", raising=False)  # 防真实环境干扰，teardown 还原
    return ttn


def test_main_dry_run_no_push(tmp_path, monkeypatch, capsys):
    ttn = _setup_env(tmp_path, monkeypatch, 7)
    monkeypatch.setattr(sys, "argv", ["tushare_expiry_notify.py", "--dry-run"])
    assert ttn.main() == 0
    out = capsys.readouterr().out
    assert "剩余 7 天" in out and "[dry-run]" in out
    assert not (tmp_path / "state.json").exists()  # dry-run 不记去重状态


def test_main_push_then_dedup(tmp_path, monkeypatch, capsys):
    ttn = _setup_env(tmp_path, monkeypatch, 7)
    calls: list = []
    stub = types.SimpleNamespace(
        send=lambda title, body="", level="info": calls.append((title, level)) or {"qqbot": "ok"})
    monkeypatch.setitem(sys.modules, "ops_notify", stub)
    monkeypatch.setattr(sys, "argv", ["tushare_expiry_notify.py"])
    assert ttn.main() == 0
    assert calls and calls[0][1] == "warn" and "【Tushare】" in calls[0][0]
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["d7"] == date.today().isoformat()
    # 同一天再跑：去重跳过，不再推送
    capsys.readouterr()
    assert ttn.main() == 0
    assert len(calls) == 1
    assert "已提醒过" in capsys.readouterr().out


def test_main_missing_config(tmp_path, monkeypatch):
    ttn = _load_ttn()
    monkeypatch.setattr(ttn, "QUANT_ROOT", tmp_path)  # 无 .env
    monkeypatch.delenv("TUSHARE_EXPIRE_DATE", raising=False)
    monkeypatch.setattr(sys, "argv", ["tushare_expiry_notify.py"])
    assert ttn.main() == 1

# -*- coding: utf-8 -*-
"""ops_notify 单测：无配置退化、单通道失败不抛异常、落盘日志。"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_ops_notify():
    spec = importlib.util.spec_from_file_location("ops_notify", SCRIPTS / "ops_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_mod(tmp_path: Path):
    mod = _load_ops_notify()
    mod.LOG_PATH = tmp_path / "ops_notify.log"
    return mod


def test_send_without_config_degrades_to_console_file(tmp_path: Path):
    mod = _make_mod(tmp_path)
    status = mod.send("标题", "内容", config_path=tmp_path / "不存在的配置.json")
    assert status["console"] == "ok"
    assert status["file"] == "ok"
    assert "未配置推送通道" in status["push"]
    log = mod.LOG_PATH.read_text(encoding="utf-8")
    assert "标题" in log and "[info]" in log


def test_bad_config_json_same_as_missing(tmp_path: Path):
    mod = _make_mod(tmp_path)
    bad = tmp_path / "notify.json"
    bad.write_text("{broken json", encoding="utf-8")
    status = mod.send("t", config_path=bad)
    assert "未配置推送通道" in status["push"]


def test_non_dict_config_same_as_missing(tmp_path: Path):
    """合法 JSON 但非 dict（如数组）→ 不得 AttributeError 击穿 send()。"""
    mod = _make_mod(tmp_path)
    bad = tmp_path / "notify.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    status = mod.send("t", config_path=bad)
    assert "未配置推送通道" in status["push"]


def test_channel_failure_swallowed_not_raised(tmp_path: Path):
    mod = _make_mod(tmp_path)
    cfg = tmp_path / "notify.json"
    cfg.write_text(json.dumps({
        # 端口 1 必然拒绝连接：单通道失败必须被吞成 error 状态，不能抛出
        "qqmail": {"user": "a@qq.com", "auth_code": "x", "host": "127.0.0.1", "port": 1,
                   "to": ["a@qq.com"]},
        "onebot": {"http_url": "http://127.0.0.1:1", "msg_type": "private", "target_id": 1},
    }, ensure_ascii=False), encoding="utf-8")
    status = mod.send("t", "b", level="warn", config_path=cfg)
    assert status["qqmail"].startswith("error:")
    assert status["onebot"].startswith("error:")
    # 通道失败不影响 console / 落盘
    assert status["console"] == "ok" and status["file"] == "ok"


def test_unknown_level_falls_back_to_info(tmp_path: Path):
    mod = _make_mod(tmp_path)
    status = mod.send("t", config_path=tmp_path / "none.json", level="bogus")
    assert status["file"] == "ok"
    assert "[info]" in mod.LOG_PATH.read_text(encoding="utf-8")


def test_qqbot_adapter_reuses_healthcheck(tmp_path: Path, monkeypatch):
    """qqbot 适配器必须复用 healthcheck 的 QQ 官方机器人推送（凭据 ~/qqbot/.env）。"""
    mod = _make_mod(tmp_path)
    calls: list[str] = []
    stub = types.SimpleNamespace(send_qq_alert=lambda text: calls.append(text))
    monkeypatch.setitem(sys.modules, "healthcheck", stub)
    cfg = tmp_path / "notify.json"
    cfg.write_text(json.dumps({"qqbot": True}), encoding="utf-8")
    status = mod.send("标题t", "正文b", config_path=cfg)
    assert status["qqbot"].startswith("ok")  # fire-and-forget：已调用即 ok
    assert calls and "标题t" in calls[0]

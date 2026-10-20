# -*- coding: utf-8 -*-
"""data_preflight 单测：复用 healthcheck.check_data 的门禁语义（OK→0 / FAIL→1）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_preflight():
    spec = importlib.util.spec_from_file_location("data_preflight", SCRIPTS / "data_preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preflight_pass_and_fail(monkeypatch):
    pf = _load_preflight()  # 模块加载时把 scripts/ 插进 sys.path，healthcheck 才可导入
    import healthcheck

    monkeypatch.setattr(healthcheck, "check_data", lambda: (True, "CNE 日线档案 OK，最新数据 2026-10-19"))
    assert pf.main() == 0
    monkeypatch.setattr(healthcheck, "check_data", lambda: (False, "CNE stock_daily 最新 2026-10-01，距今 19 天"))
    assert pf.main() == 1

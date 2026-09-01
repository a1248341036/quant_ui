# -*- coding: utf-8 -*-
"""挖掘步骤日志：steps.log 落盘 / stdout 镜像 / 字段格式 / 异常安全 / 重配置。"""
from __future__ import annotations

import pytest

from alphaagent.factor.mining import runlog


@pytest.fixture(autouse=True)
def _reset_runlog():
    yield
    runlog.setup_run_logger(None)
    runlog.set_turn(None)


def test_log_step_writes_file_and_formats(tmp_path, capsys):
    runlog.setup_run_logger(tmp_path)
    runlog.set_turn(3)
    runlog.log_step("evaluate", "vwap_dev_20", ic=0.0321, cov=0.998, verdict="promising", skip=None)
    runlog.set_turn(None)
    runlog.log_step("run_end", "outcome=candidate_only", tool_calls=41)

    text = (tmp_path / "steps.log").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "turn=3" in lines[0] and "evaluate" in lines[0]
    assert "ic=0.0321" in lines[0] and "verdict=promising" in lines[0]
    assert "skip=" not in lines[0]  # None 字段不输出
    assert "turn=-" in lines[1] and "tool_calls=41" in lines[1]
    # stdout 镜像（backend drain 进 console.log 的那份）
    out = capsys.readouterr().out
    assert "evaluate" in out and "run_end" in out


def test_log_step_never_raises(tmp_path):
    runlog.setup_run_logger(tmp_path)
    # 不可格式化对象 / 超长文本：吞异常或截断，绝不影响挖掘调用方
    runlog.log_step("x", "y", weird=object(), long="a" * 5000)


def test_setup_run_logger_reconfigures(tmp_path):
    runlog.setup_run_logger(tmp_path / "a")
    runlog.log_step("first", "")
    runlog.setup_run_logger(tmp_path / "b")
    runlog.log_step("second", "")
    assert (tmp_path / "a" / "steps.log").exists()
    assert "second" in (tmp_path / "b" / "steps.log").read_text(encoding="utf-8")

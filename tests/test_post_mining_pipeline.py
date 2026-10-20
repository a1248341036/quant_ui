# -*- coding: utf-8 -*-
"""post_mining_pipeline 单测：阶段命令构造、确认门、失败即停、断点续跑。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_pmp():
    spec = importlib.util.spec_from_file_location(
        "post_mining_pipeline", SCRIPTS / "post_mining_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kw) -> argparse.Namespace:
    base = dict(mode=None, yes=False, apply_dedup=False, blind_clean=False,
                blind_args="", notify=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_stage_cmd_defaults_and_gates():
    pmp = _load_pmp()
    dry = pmp.build_stage_cmd("promote_dry", _args(mode="technical"))
    assert dry[1].endswith("promote_candidates.py") and "--dry-run" in dry \
        and "--mode" in dry and "technical" in dry
    apply_cmd = pmp.build_stage_cmd("promote_apply", _args())
    assert "--dry-run" not in apply_cmd
    # dedup 缺省 dry-run（该脚本没有 --apply 参数，缺省即真删——OCR 2026-10-20 核实）
    assert "--dry-run" in pmp.build_stage_cmd("dedup", _args())
    assert "--dry-run" not in pmp.build_stage_cmd("dedup", _args(apply_dedup=True))
    # blind_test 缺省只出报告
    blind = pmp.build_stage_cmd("blind_test", _args(blind_clean=True, blind_args="--libs production_main"))
    assert "--clean" in blind and "--libs" in blind
    with pytest.raises(ValueError):
        pmp.build_stage_cmd("nope", _args())


def test_confirm_gate():
    pmp = _load_pmp()
    assert pmp.confirm_promote(_args(yes=True)) is True
    # 非交互（无 tty）且无 --yes → 拒绝
    fake_stdin = SimpleNamespace(isatty=lambda: False)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(sys, "stdin", fake_stdin)
    try:
        assert pmp.confirm_promote(_args(yes=False)) is False
    finally:
        monkey.undo()


def _stub_ops_notify(monkeypatch: pytest.MonkeyPatch, calls: list):
    class _Stub:
        @staticmethod
        def send(title, body="", level="info"):
            calls.append((title, level))
    monkeypatch.setitem(sys.modules, "ops_notify", _Stub)


def test_run_chain_runs_in_order_and_stops_on_failure(monkeypatch: pytest.MonkeyPatch):
    pmp = _load_pmp()
    executed: list[str] = []

    def runner(cmd):
        executed.append(Path(cmd[1]).name)
        return 1 if cmd[1].endswith("dedup_candidate_factors.py") else 0

    calls: list = []
    _stub_ops_notify(monkeypatch, calls)
    rc = pmp.run_chain(list(pmp.STAGE_ORDER), _args(yes=True), runner=runner)
    assert rc == 1
    assert executed == ["promote_candidates.py", "promote_candidates.py",
                        "dedup_candidate_factors.py"]  # blind_test 未执行
    assert calls and calls[0][1] == "error"  # 中断告警


def test_run_chain_confirm_aborts_before_apply(monkeypatch: pytest.MonkeyPatch):
    pmp = _load_pmp()
    executed: list[str] = []

    def runner(cmd):
        executed.append(Path(cmd[1]).name)
        return 0

    fake_stdin = SimpleNamespace(isatty=lambda: False)
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    rc = pmp.run_chain(list(pmp.STAGE_ORDER), _args(yes=False), runner=runner)
    assert rc == 130
    assert executed == ["promote_candidates.py"]  # 确认门拦下，apply 未执行


def test_run_chain_direct_apply_skips_confirm(monkeypatch: pytest.MonkeyPatch):
    pmp = _load_pmp()
    executed: list[str] = []

    def runner(cmd):
        executed.append(cmd[1])
        return 0

    fake_stdin = SimpleNamespace(isatty=lambda: False)
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    rc = pmp.run_chain(["promote_apply", "dedup"], _args(yes=False), runner=runner)
    assert rc == 0
    assert len(executed) == 2  # 直跳 apply 视为已确认


def test_latest_blind_report_newest_dir(tmp_path: Path):
    pmp = _load_pmp()
    for d, n in (("20260101_000000", 3), ("20260202_000000", 7)):
        (tmp_path / d).mkdir()
        (tmp_path / d / "report.json").write_text(
            json.dumps({"factors": [{"factor_id": f"f{i}"} for i in range(n)]}),
            encoding="utf-8")
    rep = pmp._latest_blind_report(tmp_path)
    assert rep["dir"] == "20260202_000000" and rep["n_factors"] == 7

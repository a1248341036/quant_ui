# -*- coding: utf-8 -*-
"""方案 B（档位自动推断）+ 调仓频率选择回归。

- infer_research_mode：勾基本面/股东面 → fundamental；纯价量族/空 → technical
- StartRequest：rebalance_freq 覆盖 spec.engine_gate.freq；显式 research_mode 优先
- delivery_criteria.to_prompt_text：渲染 rebalance_freq 必传指令
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core.research_modes import infer_research_mode


# ── 自动定档 ──

@pytest.mark.parametrize("facets,expected", [
    ([], "technical"),
    (None, "technical"),
    (["价量面"], "technical"),
    (["价量面", "量能面", "筹码面"], "technical"),
    (["基本面"], "fundamental"),
    (["股东面"], "fundamental"),
    (["价量面", "基本面"], "fundamental"),
    (["量能面", "股东面"], "fundamental"),
])
def test_infer_research_mode(facets, expected):
    assert infer_research_mode(facets) == expected


def test_infer_research_mode_strips_and_ignores_blanks():
    assert infer_research_mode([" 基本面 "]) == "fundamental"
    assert infer_research_mode(["", "  "]) == "technical"


# ── StartRequest 频率覆盖 + 模式推断（FastAPI TestClient 级别）──

def _build_request(payload: dict):
    """绕过 HTTP 层直接走 router.start 的校验/转换逻辑。"""
    from backend.routers.alphaagent import StartRequest

    return StartRequest(**payload)


def test_start_request_auto_mode_by_facets():
    from backend.routers.alphaagent import StartRequest

    req = StartRequest(focus_facets=["基本面"])
    assert req.research_mode is None  # 未显式传 → 由 start() 自动推断


def test_start_request_rebalance_freq_pattern():
    from pydantic import ValidationError
    from backend.routers.alphaagent import StartRequest

    assert StartRequest(rebalance_freq="weekly").rebalance_freq == "weekly"
    assert StartRequest().rebalance_freq is None
    with pytest.raises(ValidationError):
        StartRequest(rebalance_freq="hourly")


def test_start_endpoint_wires_freq_and_mode(monkeypatch):
    """start() 端点逻辑：自动推断 mode + rebalance_freq 写入 spec（不真正启动 run）。"""
    from backend.routers import alphaagent as router_mod

    captured = {}

    class _FakeRun:
        def snapshot(self):
            return {"run_id": "fake", "status": "running"}

    def _fake_start_run(payload):
        captured.update(payload)
        return _FakeRun()

    monkeypatch.setattr(router_mod.service, "start_run", _fake_start_run)

    req = router_mod.StartRequest(focus_facets=["基本面"], rebalance_freq="weekly")
    result = router_mod.start(req)

    assert result["run_id"] == "fake"
    assert captured["research_mode"] == "fundamental"  # 自动推断
    eg = captured["research_spec"]["delivery_policy"]["production"]["engine_gate"]
    assert eg["freq"] == "weekly"  # 用户频率覆盖 monthly 默认
    assert "weekly" in eg["allowed_freqs"]  # 白名单不受影响


def test_start_endpoint_default_freq_follows_mode(monkeypatch):
    from backend.routers import alphaagent as router_mod

    captured = {}

    class _FakeRun:
        def snapshot(self):
            return {"run_id": "fake2", "status": "running"}

    def _fake_start_run(payload):
        captured.update(payload)
        return _FakeRun()

    monkeypatch.setattr(router_mod.service, "start_run", _fake_start_run)

    # 勾基本面 → fundamental 档 → engine_gate.freq 保持 monthly 默认
    req = router_mod.StartRequest(focus_facets=["基本面"])
    router_mod.start(req)
    eg = captured["research_spec"]["delivery_policy"]["production"]["engine_gate"]
    assert eg["freq"] == "monthly"

    # 纯价量 → technical 档 → weekly 默认
    captured.clear()
    req2 = router_mod.StartRequest(focus_facets=["价量面"])
    router_mod.start(req2)
    assert captured["research_mode"] == "technical"
    eg2 = captured["research_spec"]["delivery_policy"]["production"]["engine_gate"]
    assert eg2["freq"] == "weekly"


# ── prompt 渲染：频率指令 ──

def test_to_prompt_text_contains_rebalance_freq_directive():
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria

    spec = {
        "delivery_policy": {
            "candidate": {"min_abs_ic": 0.02},
            "production": {
                "min_train_abs_ic": 0.025,
                "engine_gate": {"enabled": True, "freq": "monthly"},
            },
        },
    }
    criteria = DeliveryCriteria.from_spec(spec)
    text = criteria.to_prompt_text()
    assert 'rebalance_freq 必须传 "monthly"' in text
    assert "daily" in text and "weekly" in text  # 可选范围列出

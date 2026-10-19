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

# ── 自动定档（数据面 × 调仓频率两个正交维度，2026-10-08 正交化）──

@pytest.mark.parametrize("facets,freq,expected", [
    # 未指定频率 → 数据面惯性默认：基本面 → fundamental（monthly），价量 → technical
    ([], None, "technical"),
    (None, None, "technical"),
    (["价量面"], None, "technical"),
    (["价量面", "量能面", "筹码面"], None, "technical"),
    (["基本面"], None, "fundamental"),
    (["股东面"], None, "fundamental"),
    (["价量面", "基本面"], None, "fundamental"),
    (["量能面", "股东面"], None, "fundamental"),
    # 显式频率 → 数据面正交：基本面数据面也按频率落档（门槛/label 跟随频率）
    (["基本面"], "daily", "fundamental_daily"),
    (["基本面"], "weekly", "fundamental_weekly"),
    (["基本面"], "monthly", "fundamental"),
    (["价量面", "基本面"], "weekly", "fundamental_weekly"),
    (["股东面"], "daily", "fundamental_daily"),
    (["价量面"], "daily", "technical_daily"),
    (["价量面"], "weekly", "technical_weekly"),
    (["价量面"], "monthly", "technical_monthly"),
])
def test_infer_research_mode(facets, freq, expected):
    assert infer_research_mode(facets, freq) == expected


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


def test_start_request_reasoning_effort_pattern():
    from pydantic import ValidationError
    from backend.routers.alphaagent import StartRequest

    # 默认 None → 走 MiningConfig 默认（medium）；显式档位白名单校验
    assert StartRequest().reasoning_effort is None
    assert StartRequest(reasoning_effort="medium").reasoning_effort == "medium"
    assert StartRequest(reasoning_effort="none").reasoning_effort == "none"
    with pytest.raises(ValidationError):
        StartRequest(reasoning_effort="ultra")


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

    req = router_mod.StartRequest(focus_facets=["价量面"], rebalance_freq="weekly")
    result = router_mod.start(req)

    assert result["run_id"] == "fake"
    # 2026-10-04：用户给频率 → 自动落到对齐子档（label 5d + weekly），不再走解耦主档
    assert captured["research_mode"] == "technical_weekly"
    eg = captured["research_spec"]["delivery_policy"]["production"]["engine_gate"]
    assert eg["freq"] == "weekly"
    assert eg["allowed_freqs"] == ["weekly"]


def test_start_endpoint_fundamental_orthogonal_freq(monkeypatch):
    """2026-10-08 数据面正交：勾基本面 + weekly 不再被锁在 fundamental（monthly），
    而是落到 fundamental_weekly（label_5d + weekly + 5d 门槛）——数据面只决定
    信号族/载列，调仓频率决定 label 与门槛。"""
    from backend.routers import alphaagent as router_mod

    captured = {}

    class _FakeRun:
        def snapshot(self):
            return {"run_id": "fake_orth", "status": "running"}

    def _fake_start_run(payload):
        captured.update(payload)
        return _FakeRun()

    monkeypatch.setattr(router_mod.service, "start_run", _fake_start_run)

    req = router_mod.StartRequest(focus_facets=["基本面"], rebalance_freq="weekly")
    result = router_mod.start(req)

    assert result["run_id"] == "fake_orth"
    assert captured["research_mode"] == "fundamental_weekly"
    eg = captured["research_spec"]["delivery_policy"]["production"]["engine_gate"]
    assert eg["freq"] == "weekly"
    assert eg["allowed_freqs"] == ["weekly"]
    # 门槛跟随频率（5d 档），与数据面无关
    crit = captured["research_spec"]["delivery_policy"]["candidate"]
    assert crit["min_abs_ic"] == 0.030
    # 数据面仍决定载列/信号族（基本面组 = fundamental_* 四族 + 机构/股东集中两族；
    # 2026-10-19 补后两族：FACET_GROUPS 已把机构面/股东集中面归入基本面组，
    # 此前档位族清单缺它们，导致勾这两个面时无可挂之族）
    assert captured["research_spec"]["search_policy"]["allowed_signal_families"] == [
        "fundamental_quality", "fundamental_growth",
        "fundamental_value", "fundamental_revision",
        "institutional", "shareholder_concentration",
    ]


def test_start_endpoint_rejects_label_freq_mismatch(monkeypatch):
    """2026-10-04 强制一致仍生效：显式钉死的模式与频率持有期不符必须被拒，
    而不是静默放行。正交化后自动推断不再产生冲突（数据面×频率自由组合），
    冲突只会来自用户同时显式指定不匹配的 research_mode + rebalance_freq。"""
    from fastapi import HTTPException

    from backend.routers import alphaagent as router_mod

    monkeypatch.setattr(router_mod.service, "start_run", lambda payload: None)
    # 显式钉 technical（label_1d + daily）却要求 weekly 交付 → 422
    req = router_mod.StartRequest(research_mode="technical", rebalance_freq="weekly")
    with pytest.raises(HTTPException) as exc:
        router_mod.start(req)
    assert exc.value.status_code == 422
    assert "label_freq_consistency.mismatch" in str(exc.value.detail)


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

    # 纯价量 → technical 档 → daily（label_1d 对齐后的门禁频率，2026-10-04 起）
    captured.clear()
    req2 = router_mod.StartRequest(focus_facets=["价量面"])
    router_mod.start(req2)
    assert captured["research_mode"] == "technical"
    eg2 = captured["research_spec"]["delivery_policy"]["production"]["engine_gate"]
    assert eg2["freq"] == "daily"
    assert eg2["allowed_freqs"] == ["daily"]


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


# ── 研报档（report）：注册表 / 启动请求 / 页头 chip 数据源 ──
# 2026-09-30：report 档 09-29 只进了注册表，UI 入口缺失（前端页头「研报」按钮 + chip 后补）。
# 这组用例把"注册表可达 → 启动请求不被数据面推断改写 → 快照向 UI 暴露档位"钉住。

def test_report_mode_registered_and_exposed_to_frontend():
    """report 必须是顶层档位且出现在 /research-modes（页头按钮的数据源）。"""
    from core.research_modes import RESEARCH_MODES, get_research_mode, ui_options

    assert "report" in RESEARCH_MODES
    assert get_research_mode("report").needs_fundamentals is True
    values = [opt["value"] for opt in ui_options()]
    assert "report" in values
    # internal 标记语义（2026-10-08 数据面正交）：top-level 档位（technical/fundamental/
    # report）internal=False 且永远在列；内部子档（technical_*/fundamental_*）internal=True，
    # 供前端下拉过滤、switchResearchMode 白名单与门槛文件读写仍全量可达。
    top = [o for o in ui_options() if not o["internal"]]
    assert [o["value"] for o in top] == ["technical", "fundamental", "report"]
    internals = [o["value"] for o in ui_options() if o["internal"]]
    assert sorted(internals) == sorted([
        "technical_daily", "technical_weekly", "technical_monthly",
        "technical_event_monthly",
        "fundamental_daily", "fundamental_weekly",
    ])
    # 子档元数据齐全（前端 switchResearchMode 需要 hint/default_user_message）
    for o in ui_options():
        if o["internal"]:
            assert o["hint"] and o["default_user_message"]


def test_start_endpoint_report_mode_passthrough(monkeypatch):
    """显式 report 档必须胜出数据面推断，且 spec 自带 report_policy（复现门禁/阶段化注入）。"""
    from backend.routers import alphaagent as router_mod

    captured = {}

    class _FakeRun:
        def snapshot(self):
            return {"run_id": "reportrun", "status": "running"}

    def _fake_start_run(payload):
        captured.update(payload)
        return _FakeRun()

    monkeypatch.setattr(router_mod.service, "start_run", _fake_start_run)

    # 即便勾了慢因子面（自动推断会落 fundamental），显式 report 也不能被改写
    req = router_mod.StartRequest(research_mode="report", focus_facets=["基本面"])
    router_mod.start(req)

    assert captured["research_mode"] == "report"
    spec = captured["research_spec"]
    assert spec["research_mode"] == "report"
    policy = spec["report_policy"]
    assert policy["reproduce_first"] is True
    assert policy["reproduce_of_required"] is True
    assert policy["knowledge_mode_by_phase"] == {
        "reproduce": "mechanism_cards",
        "diverge": "report_rag",
    }


def test_snapshot_exposes_research_mode(tmp_path):
    """页头档位 chip 的数据源：内存态从 spec 取，后端重启恢复态从 run_meta.params 取。"""
    from backend import alphaagent_service as svc

    in_memory = svc.AgentRun(
        run_id="mem", command=[], log_dir=tmp_path,
        params={"research_spec": {"research_mode": "report"}},
    )
    assert in_memory.snapshot(tail=0)["research_mode"] == "report"

    # 子进程覆盖写的 run_meta 只有 research_mode（没有整份 spec）
    hydrated = svc.AgentRun(
        run_id="hyd", command=[], log_dir=tmp_path,
        params={"user_message": "x", "research_mode": "technical"},
    )
    assert hydrated.snapshot(tail=0)["research_mode"] == "technical"

    unknown = svc.AgentRun(run_id="unk", command=[], log_dir=tmp_path, params={})
    assert unknown.snapshot(tail=0)["research_mode"] is None

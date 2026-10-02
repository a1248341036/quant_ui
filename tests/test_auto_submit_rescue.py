# -*- coding: utf-8 -*-
"""兜底补交（P1）确定性单测：rescue_unsubmitted_promising()。

背景：2026-10-02 端到端验证不可靠——`max_turns=1` 的 run 连"过线"都没发生（0 个
train_passed），自然没有"过线未提交"可供补交（`unsubmitted_promising=0`）。故把逻辑抽成
纯函数（依赖注入 submit_service）后做确定性验证：排名、上限、异常隔离、开关关闭。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.agentscope_run import rescue_unsubmitted_promising


class _FakeSubmit:
    """记录调用的假 submit 服务。"""

    def __init__(self, *, candidate_names: tuple[str, ...] = (), stored_names: tuple[str, ...] = (),
                 raise_on: tuple[str, ...] = ()) -> None:
        self.calls: list[dict] = []
        self._cand = set(candidate_names)
        self._stored = set(stored_names)
        self._raise = set(raise_on)

    def submit(self, session_id: str, *, multi_line_expr: str, factor_name: str, comment: str) -> dict:
        if factor_name in self._raise:
            raise RuntimeError("boom")
        self.calls.append({"session_id": session_id, "expr": multi_line_expr,
                           "name": factor_name, "comment": comment})
        return {"stored": factor_name in self._stored,
                "candidate": factor_name in self._cand,
                "skipped": "stage_one_failed:ic,icir"}


def _row(name: str, icir: float, expr: str = "CS_ZSCORE($close)") -> dict:
    return {"factor_name": name, "expression": expr, "metrics": {"icir": icir}}


def test_ranks_by_abs_icir_and_caps_top():
    svc = _FakeSubmit()
    rows = [_row("a", 0.10), _row("b", -0.50), _row("c", 0.30), _row("d", 0.90)]
    stat = rescue_unsubmitted_promising(svc, "sid", rows, 2)
    assert [c["name"] for c in svc.calls] == ["d__autorescue", "b__autorescue"]  # |ICIR| 降序（含负值按绝对值）
    assert stat["attempted"] == 2
    assert all(c["session_id"] == "sid" for c in svc.calls)
    assert all("兜底补交" in c["comment"] for c in svc.calls)


def test_reports_stored_and_candidate_counts():
    svc = _FakeSubmit(stored_names=("a__autorescue",), candidate_names=("b__autorescue",))
    stat = rescue_unsubmitted_promising(svc, "sid", [_row("a", 0.5), _row("b", 0.4)], 3)
    assert stat == {"attempted": 2, "stored": 1, "candidate": 1, "failed": 0}


def test_single_failure_does_not_stop_others():
    svc = _FakeSubmit(raise_on=("a__autorescue",))
    stat = rescue_unsubmitted_promising(svc, "sid", [_row("a", 0.9), _row("b", 0.8)], 3)
    assert stat["attempted"] == 2 and stat["failed"] == 1
    assert [c["name"] for c in svc.calls] == ["b__autorescue"]      # b 仍被提交


def test_skips_rows_without_expression():
    svc = _FakeSubmit()
    stat = rescue_unsubmitted_promising(svc, "sid", [_row("a", 0.9, expr=""), _row("b", 0.5)], 3)
    assert stat["attempted"] == 1
    assert [c["name"] for c in svc.calls] == ["b__autorescue"]


def test_noop_when_off_or_missing():
    svc = _FakeSubmit()
    rows = [_row("a", 0.9)]
    assert rescue_unsubmitted_promising(svc, "sid", rows, 0)["attempted"] == 0      # 开关关闭
    assert rescue_unsubmitted_promising(svc, None, rows, 3)["attempted"] == 0       # 无 session
    assert rescue_unsubmitted_promising(None, "sid", rows, 3)["attempted"] == 0     # 无服务
    assert rescue_unsubmitted_promising(svc, "sid", [], 3)["attempted"] == 0        # 无对象
    assert svc.calls == []


def test_log_fn_receives_events():
    svc = _FakeSubmit(candidate_names=("a__autorescue",))
    events: list[tuple] = []
    rescue_unsubmitted_promising(svc, "sid", [_row("a", 0.9)], 1,
                                 lambda *a, **k: events.append((a, k)))
    assert events and events[0][0][0] == "auto_submit_rescue"

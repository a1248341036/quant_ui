# -*- coding: utf-8 -*-
"""血统门禁覆盖回归测试（2026-10-02）。

背景：原实现把门禁内联在 `evaluate_factor` / `submit_factor`，而批量评估主路径
`eval_on_train_set`（实测 108 次/run）没装 → 34/154（22%）因子绕过门禁挂在其它课题血统上。
本测试锁定共享助手 `_report_lineage_block` 的行为（4 个入口都调用它）。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.agentscope_tools import (
    _report_lineage_block,
    _report_lineage_fill,
)


class _T:
    def __init__(self, gate: dict | None = None) -> None:
        self.report_reproduce_gate = gate or {}


def test_diverge_gate_blocks_off_lineage():
    t = _T({"diverge_parent": True, "phase": "diverge", "qid": "RQ_X",
            "parent_name": "rq_x_repro_v1"})
    blocked = _report_lineage_block(t, "mix_ovlead_wma5_kg90")
    assert blocked and blocked.startswith("⛔ 发散门禁")
    # 复现版及其衍生命名放行
    assert _report_lineage_block(t, "rq_x_repro_v1_v2") is None


def test_reproduce_gate_requires_qid():
    t = _T({"required": True, "phase": "reproduce", "qid": "RQ_X"})
    assert _report_lineage_block(t, "foo") is not None
    assert _report_lineage_block(t, "reproduce_of:RQ_X") is None


def test_no_gate_passes_everything():
    t = _T({})
    assert _report_lineage_block(t, None) is None
    assert _report_lineage_block(t, "anything") is None


def test_all_eval_entrypoints_call_the_helper():
    """覆盖检查（OCR 2026-10-02 后收紧）：**4 个**入口都必须调用助手，且不得再有内联副本。
    - eval_on_train_set / eval_on_val_set / evaluate_factor / submit_factor
    - 原先 submit_factor 是内联副本（口径漂移风险），已改为调用助手。"""
    import inspect

    import alphaagent.factor.mining.agent.agentscope_tools as m
    src = inspect.getsource(m)
    assert src.count("_report_lineage_block(tools, parent_factor)") >= 4, \
        "_report_lineage_block 调用点少于 4 处（可能又有入口漏装内联副本）"
    # 内联副本的判据：⛔ 文案只应出现在助手内一处
    assert src.count("⛔ 发散门禁") == 1, "发散门禁文案出现多次 → 又有内联副本"
    assert src.count("⛔ 复现门禁") == 1, "复现门禁文案出现多次 → 又有内联副本"


# ── 血统自动补全（2026-10-03）────────────────────────────────────────

def test_autofill_reproduce_phase():
    """复现轮漏传 → 补 `reproduce_of:<qid>`，且补后能过门禁。"""
    t = _T({"required": True, "phase": "reproduce", "qid": "RQ_X", "parent_autofill": True})
    assert _report_lineage_fill(t, None) == "reproduce_of:RQ_X"
    assert _report_lineage_fill(t, "  ") == "reproduce_of:RQ_X"
    assert _report_lineage_block(t, _report_lineage_fill(t, None)) is None


def test_autofill_diverge_phase_uses_parent_name():
    t = _T({"diverge_parent": True, "phase": "diverge", "qid": "RQ_X",
            "parent_name": "rq_x_repro_v1", "parent_autofill": True})
    assert _report_lineage_fill(t, None) == "rq_x_repro_v1"
    assert _report_lineage_block(t, _report_lineage_fill(t, None)) is None


def test_autofill_keeps_explicit_value():
    """传了值就原样保留（包括传错的非空血统 → 仍由门禁硬拦）。"""
    t = _T({"diverge_parent": True, "phase": "diverge", "qid": "RQ_X",
            "parent_name": "rq_x_repro_v1", "parent_autofill": True})
    assert _report_lineage_fill(t, "mix_ovlead") == "mix_ovlead"
    assert _report_lineage_block(t, _report_lineage_fill(t, "mix_ovlead")) is not None


def test_autofill_switch_off_and_empty_gate():
    """开关关掉 → 不补；空网关/非研报阶段 → 不补。"""
    off = _T({"required": True, "phase": "reproduce", "qid": "RQ_X", "parent_autofill": False})
    assert _report_lineage_fill(off, None) is None
    assert _report_lineage_fill(_T({}), None) is None
    assert _report_lineage_fill(_T({"phase": "diverge", "diverge_parent": True,
                                    "parent_name": "", "parent_autofill": True}), None) is None


def test_fill_helper_used_by_all_entrypoints():
    """补全助手必须与门禁助手同点挂载（否则又会出现"某入口绕过补全"）。"""
    import inspect

    import alphaagent.factor.mining.agent.agentscope_tools as m
    src = inspect.getsource(m)
    assert src.count("_report_lineage_fill(tools, parent_factor)") == \
        src.count("_report_lineage_block(tools, parent_factor)") >= 4

# -*- coding: utf-8 -*-
"""血统门禁覆盖回归测试（2026-10-02）。

背景：原实现把门禁内联在 `evaluate_factor` / `submit_factor`，而批量评估主路径
`eval_on_train_set`（实测 108 次/run）没装 → 34/154（22%）因子绕过门禁挂在其它课题血统上。
本测试锁定共享助手 `_report_lineage_block` 的行为（4 个入口都调用它）。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.agentscope_tools import _report_lineage_block


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

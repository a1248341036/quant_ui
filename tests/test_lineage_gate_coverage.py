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


def test_reproduce_gate_accepts_same_question_lowercase_name():
    """2026-10-03 修：因子名是 `rq<qid-小写>_*`，课题号是大写 `RQ_*` → 必须大小写不敏感。

    实测（4 run / 38 条拦截）里 14 条属于这种"同课题被误杀"（如 RQ_17eebb 传
    `rq17eebb_rev_timed_turn_slope`）；跨课题血统仍要硬拦。
    """
    t = _T({"required": True, "phase": "reproduce", "qid": "RQ_17eebb"})
    assert _report_lineage_block(t, "rq17eebb_rev_timed_turn_slope") is None
    assert _report_lineage_block(t, "reproduce_of:rq17eebb") is None
    assert _report_lineage_block(t, "RQ17EEBB_x") is None
    # 另一种命名写法（带下划线）同样放行
    t2 = _T({"required": True, "phase": "reproduce", "qid": "RQ_addf97"})
    assert _report_lineage_block(t2, "rq_addf97_growth_resid_softgate") is None
    # 跨课题血统（另一个课题的复现版）仍然拦截
    assert _report_lineage_block(t, "rqffd0ba_style_rotation_anch") is not None


def test_diverge_gate_case_insensitive_parent_name():
    t = _T({"diverge_parent": True, "phase": "diverge", "qid": "RQ_X",
            "parent_name": "rq_x_repro_v1"})
    assert _report_lineage_block(t, "RQ_X_REPRO_V1_mut") is None
    assert _report_lineage_block(t, "mix_ovlead") is not None


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


def test_same_lineage_and_recommendation_filter():
    """跨课题记忆推荐在研报两阶段必被门禁拦 → 推荐阶段就要过滤掉（2026-10-03）。"""
    from alphaagent.factor.mining.report_channels import (
        filter_lineage_recs,
        same_lineage,
    )

    assert same_lineage("rq_addf97_growth_resid", "RQ_addf97")
    assert same_lineage("reproduce_of:rq17eebb", "RQ_17eebb")
    assert same_lineage("rq17eebb_rev_timed_turn_slope", "RQ_17eebb")
    assert not same_lineage("rqffd0ba_style_rotation_anch", "RQ_addf97")
    assert not same_lineage(None, "RQ_addf97")
    assert not same_lineage("rq_addf97_x", "")

    recs = [
        {"parent_factor": "rq_62f045_ssa_timing_mom"},
        {"parent_factor": "amtcap120_x_prem20_x_prem10q4_rank"},
        {"parent_factor": "peakclear_mom10_pw"},
    ]
    kept = filter_lineage_recs(recs, "RQ_62f045")
    assert [r["parent_factor"] for r in kept] == ["rq_62f045_ssa_timing_mom"]
    # 非研报模式（无课题号）→ 原样返回
    assert filter_lineage_recs(recs, "") == recs


def test_recommendation_filter_wired_in_run_loop():
    """防止修复被静默摘掉：run 主循环必须调用过滤器，并带 steps.log 锚点。"""
    import inspect

    import alphaagent.factor.mining.agent.agentscope_run as m
    src = inspect.getsource(m)
    assert "filter_lineage_recs" in src, "记忆推荐未接血统过滤"
    assert "memory_suggest_lineage_filter" in src, "过滤缺少可观测锚点"
    assert '_report_phase_box["qid"]' in src, "阶段盒未携带课题号 → 过滤拿不到 qid"


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

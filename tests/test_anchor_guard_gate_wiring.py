# -*- coding: utf-8 -*-
"""护栏「接线」回归测试（2026-10-03）。

教训：run `e10835374b7a` 里同一题出现 **16 次 `off_reference`** 却 **0 条 `report_anchor_skip`**
——护栏逻辑没错，但阈值**没随 run 网关传递**：`_effective_report_policy` 只合并
`_GATE_CARRIED_POLICY_KEYS` 里的键，未入列 → `rp.get("anchor_block_skip_threshold")` 恒 None
→ 静默关闭。纯函数单测测不出这类"接线"缺失，故这里显式锁死三条接线。
"""
from __future__ import annotations

import inspect

from alphaagent.factor.mining.agent import agentscope_run as ar
from alphaagent.factor.mining.tools._dispatch import (
    _GATE_CARRIED_POLICY_KEYS,
    _effective_report_policy,
)


def test_threshold_is_gate_carried():
    """① 阈值必须列入"网关携带"白名单，否则合并结果里读不到。"""
    assert "anchor_block_skip_threshold" in _GATE_CARRIED_POLICY_KEYS


def test_effective_policy_carries_threshold_from_gate():
    """② 功能验证：网关里的阈值经 _effective_report_policy 必须可见（工具侧为 None 的情形）。"""
    merged = _effective_report_policy(None, {"anchor_block_skip_threshold": 5})
    assert merged.get("anchor_block_skip_threshold") == 5
    # 工具侧有值、网关无该键 → 保持工具侧值（不回归）
    assert _effective_report_policy({"anchor_block_skip_threshold": 3}, {}) == {
        "anchor_block_skip_threshold": 3
    }
    # 网关显式给 0（关闭护栏）必须覆盖工具侧（0 是合法值，不能被 None 语义吞掉）
    merged_off = _effective_report_policy({"anchor_block_skip_threshold": 8},
                                          {"anchor_block_skip_threshold": 0})
    assert merged_off.get("anchor_block_skip_threshold") == 0


def test_run_gate_builder_passes_threshold():
    """③ 源码级锁定：run 收尾/轮次网关构建处必须把阈值写进 gate（防重构时漏掉）。"""
    src = inspect.getsource(ar)
    assert '"anchor_block_skip_threshold"' in src, "run 网关构建处没有传护栏阈值"
    assert 'report_policy.get("anchor_block_skip_threshold"' in src


def test_dispatch_warns_when_threshold_missing():
    """④ 不留静默 fail-open：阈值缺失时必须留下可见告警。"""
    src = inspect.getsource(
        __import__("alphaagent.factor.mining.tools._dispatch", fromlist=["_x"])
    )
    assert "护栏阈值未随网关传入" in src


def test_guard_block_uses_self_not_bare_tools():
    """⑤ 作用域锁定：护栏块在 `_DispatchMixin` 方法内，工具对象只能叫 `self`。

    实测 run 38b61ab42be6：护栏 5 次触发全是
    `NameError: name 'tools' is not defined`（该作用域没有 tools）→ 止损仍不生效。
    """
    import re

    src = inspect.getsource(
        __import__("alphaagent.factor.mining.tools._dispatch", fromlist=["_x"])
    )
    # 锚点必须精确到**护栏块自身的注释**（"（2026-10-03）：逐题累计"），
    # 否则会从 _GATE_CARRIED_POLICY_KEYS 的注释处开始切，块区过大误报。
    i = src.find("连续锚失败护栏（2026-10-03）")
    j = src.find('if not _fid.get("passed")')
    block = src[i:j] if i > 0 and j > i else ""
    assert block, "定位不到护栏块（可能被重构/删除）"
    code = "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"(?<![_\w.])tools\.", code), \
        "护栏块里出现裸 `tools` —— 该作用域没有此变量（mixin 方法用 self）"
    assert "self._anchor_block" in code, "护栏状态没有挂在 self（工具实例）上"


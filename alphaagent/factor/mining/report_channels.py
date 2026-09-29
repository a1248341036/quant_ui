# -*- coding: utf-8 -*-
"""研报知识注入通道选择（三选一 + 关闭）——唯一真源。

``research_spec.report_policy.knowledge_mode``：

- ``report_rag``（默认）R3/R4：原始文献 RAG（prompt 模块 report_rag）+ 逐轮课题对齐注入
- ``static_manual``    R1：静态机制知识手册（prompt 模块 report_mechanisms）
- ``mechanism_cards``  R2：机制卡检索注入（prompt 模块 report_prior）
- ``off``              三条通道全关（课题队列仍由 ``enable_question_queue`` 单独控制）

旧布尔键（``enable_report_rag`` / ``enable_report_mechanisms`` / ``enable_report_prior``）
在未设置 ``knowledge_mode`` 时继续生效（向后兼容）；一旦显式设置 ``knowledge_mode``，
三通道互斥由它决定，避免同一条研报知识被两条通道重复注入。
"""

from __future__ import annotations

from typing import Any

# 允许的通道取值（顺序即前端/文档展示顺序）
KNOWLEDGE_MODES: tuple[str, ...] = ("report_rag", "static_manual", "mechanism_cards", "off")
DEFAULT_KNOWLEDGE_MODE = "report_rag"


def knowledge_mode(spec: dict[str, Any] | None) -> str:
    """读取显式 knowledge_mode；未设置或非法返回空串（回落旧布尔键）。"""
    policy = (spec or {}).get("report_policy") or {}
    raw = str(policy.get("knowledge_mode") or "").strip().lower()
    return raw if raw in KNOWLEDGE_MODES else ""


def resolve_report_channels(spec: dict[str, Any] | None, phase: str | None = None) -> dict[str, bool]:
    """三通道启用表（互斥，最多一个为真）。

    ``phase`` 非空且 ``report_policy.knowledge_mode_by_phase`` 命中该阶段时，**按阶段取值**：
    研报模式下复现阶段只给机制卡、发散阶段才给研报 RAG（2026-09-29）。
    未配置/未命中 → 回落 ``knowledge_mode`` → 再回落旧布尔键（全部老行为不变）。
    """
    policy = (spec or {}).get("report_policy") or {}
    mode = ""
    if phase:
        by_phase = policy.get("knowledge_mode_by_phase") or {}
        if isinstance(by_phase, dict):
            raw = str(by_phase.get(phase) or "").strip().lower()
            mode = raw if raw in KNOWLEDGE_MODES else ""
    if not mode:
        mode = knowledge_mode(spec)
    if mode:
        return {
            "report_rag": mode == "report_rag",
            "static_manual": mode == "static_manual",
            "mechanism_cards": mode == "mechanism_cards",
        }
    return {
        "report_rag": bool(policy.get("enable_report_rag", True)),
        "static_manual": bool(policy.get("enable_report_mechanisms", False)),
        "mechanism_cards": bool(policy.get("enable_report_prior", False)),
    }


def report_rag_enabled(spec: dict[str, Any] | None, phase: str | None = None) -> bool:
    """逐轮/系统提示词两处 RAG 注入共用判定（支持按阶段）。"""
    return bool(resolve_report_channels(spec, phase=phase)["report_rag"])


# ── 研报模式流程开关（仅 report 模式设置；其他模式这些键不存在 → 老行为） ──

def report_flow_enabled(spec: dict[str, Any] | None) -> bool:
    """是否启用「复现 → 判定 → 发散」流程（研报模式）。"""
    return bool(((spec or {}).get("report_policy") or {}).get("reproduce_first"))


def reproduce_of_required(spec: dict[str, Any] | None) -> bool:
    return bool(((spec or {}).get("report_policy") or {}).get("reproduce_of_required"))


def reproduce_lock_rounds(spec: dict[str, Any] | None) -> int:
    """复现通过后锁定该课题的发散轮数（默认 3）。"""
    raw = ((spec or {}).get("report_policy") or {}).get("reproduce_lock_rounds", 3)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 3


def resolve_report_phase(spec: dict[str, Any] | None, turn: int, *, locked: bool = False) -> str | None:
    """当前轮属于哪个知识阶段：``reproduce`` / ``diverge``；非研报模式返回 None。

    Phase 0 的确定性规则（Phase 2 的课题状态机落地后由状态机接管）：
    研报模式下第 0 轮 = 复现，其后 = 发散；``locked=True`` 时一律发散。
    """
    if not report_flow_enabled(spec):
        return None
    if locked:
        return "diverge"
    return "reproduce" if int(turn) <= 0 else "diverge"

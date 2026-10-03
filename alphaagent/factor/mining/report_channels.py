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


def diverge_parent_required(spec: dict[str, Any] | None) -> bool:
    """发散轮是否强制 `parent_factor` 指向复现版（默认关，2026-09-30）。

    关：只靠题面约束，模型可脱稿自由探索（实测产出 train_passed 但脱离研报血统）；
    开：「围绕研报发散」成为硬约束，代价是候选产出可能下降。
    """
    return bool(((spec or {}).get("report_policy") or {}).get("diverge_parent_required"))


def lineage_parent_autofill(spec: dict[str, Any] | None) -> bool:
    """血统门禁是否对**漏传**的 `parent_factor` 自动补全（默认开，2026-10-03）。

    开：漏传（传空）时按当前阶段补 `reproduce_of:<qid>` / 复现版因子名；**传了非空的
    错血统仍硬拦** —— 治的是"模型忘填字段白烧一轮"（实测 13+/run）；
    关：回到"血统必须由模型显式声明"的严格口径（诊断用）。
    """
    return bool(((spec or {}).get("report_policy") or {}).get("parent_autofill", True))


# ── 血统比对键（2026-10-03）──────────────────────────────────────────
# 同一课题号在因子名里有 `rq<qid>`（`rq17eebb_*`）与 `rq_<qid>`（`rq_addf97_*`）两种写法，
# 逐字比较会把"同课题"误判成"跨课题"（实测 4 run / 38 条门禁拦截里 17 条属此类）。

def lineage_key(text: Any) -> str:
    """血统比对键：忽略大小写与下划线。"""
    return str(text or "").lower().replace("_", "")


def same_lineage(parent_factor: Any, qid_or_name: Any) -> bool:
    """`parent_factor` 是否落在该课题号/复现版名的血统里。"""
    key = lineage_key(qid_or_name)
    return bool(key) and key in lineage_key(parent_factor)


def filter_lineage_recs(recs: list[dict[str, Any]], qid: Any,
                        parent_name: Any = None) -> list[dict[str, Any]]:
    """研报模式：只保留指向**本课题血统**的记忆推荐。

    跨课题推荐在复现/发散两处门禁下**必被拦**（复现要求含本课题号、发散要求指向复现版），
    留着只会白烧当轮评估名额（实测跨课题推荐 17 条/晚）。

    ``parent_name`` = 本课题复现版因子名（发散阶段合法父本）：**它常常不含课题号**
    （如 `amtcap120_x_prem20_x_prem10q4_rank_med`），只按 qid 过滤会把它误杀。
    """
    if not str(qid or ""):
        return list(recs or [])
    return [
        r for r in (recs or [])
        if same_lineage((r or {}).get("parent_factor"), qid)
        or same_lineage((r or {}).get("parent_factor"), parent_name)
    ]


# ── 进程内"当前研报网关"（2026-09-30）────────────────────────────────
# 背景：gate 原先挂在 tools 实例属性上，但判定在 dispatch() 里跑，
# 实测两侧对象身份不一致 → 判定静默 early-return。每次 run 是独立进程、
# 同一时刻只有一个 turn，故用模块级全局最稳妥。
_CURRENT_GATE: dict[str, Any] = {}


def set_run_gate(gate: dict[str, Any] | None) -> None:
    _CURRENT_GATE.clear()
    if gate:
        _CURRENT_GATE.update(gate)


def get_run_gate() -> dict[str, Any]:
    """返回进程内网关的**副本**（调用方不得依赖其身份）。

    2026-09-30 review 修：原实现 `return _CURRENT_GATE` 返回全局字典本体，任何调用方
    原地改（例如判定通过后 `gate["phase"] = "diverge"`）都会**永久污染**全局网关 →
    同一进程后续新题的复现判定 phase 永远是 diverge，被静默跳过。
    """
    return dict(_CURRENT_GATE)

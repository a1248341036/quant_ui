# -*- coding: utf-8 -*-
"""模块 05e · report_rag：研报原始文献 RAG 检索注入（R3 方案）。

在运行期根据当前 focus_facets 与研究模式，通过 OpenViking / 本地高速检索引擎，
检索最具参考价值的券商金工原始文献段落作为灵感注入。
"""
from __future__ import annotations

import logging
from typing import Any

from alphaagent.factor.mining.memory.ov_store import OVStore

logger = logging.getLogger(__name__)

NAME = "report_rag"
TITLE = "研报原始文献 RAG 检索"
ORDER = 58
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"
PHASES = frozenset({"explore", "deepen", "full"})


def enabled(ctx: Any) -> bool:
    if getattr(ctx, "asset_type", "stock") != "stock":
        return False
    from alphaagent.factor.mining.report_channels import report_rag_enabled

    return report_rag_enabled(getattr(ctx, "research_spec", None))


def render(ctx: Any) -> str:
    spec = getattr(ctx, "research_spec", None) or {}
    policy = spec.get("report_policy") or {}
    budget = int(policy.get("report_rag_max_chars", 1600))
    top_k = int(policy.get("report_rag_top_k", 3))

    endpoint = policy.get("ov_endpoint", "http://127.0.0.1:1933")
    store = OVStore(endpoint=endpoint, inject_max_chars=budget)

    # 动态课题对齐：如果启用了 R4 问题队列，提取当前关联课题的关键词作为核心检索 Query
    query_text = None
    if bool(policy.get("enable_question_queue", False)):
        try:
            from alphaagent.factor.mining.agent.question_queue import get_question_for_turn
            # 系统提示词装配时，按当前聚焦面与种子匹配首选课题
            q = get_question_for_turn(0, spec=spec, focus_facets=getattr(ctx, "focus_facets", ()))
            if q:
                # 以当前课题的 主题 + 机制假设 作为精准召回检索词
                query_text = f"{q.get('topic', '')} {q.get('hypothesis', '')}"
        except Exception:
            pass

    return store.retrieve_report_knowledge(
        focus_facets=getattr(ctx, "focus_facets", ()),
        research_mode=getattr(ctx, "research_mode", "technical"),
        query_text=query_text,
        limit=top_k,
        max_chars=budget,
    )

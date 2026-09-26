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
    spec = getattr(ctx, "research_spec", None) or {}
    policy = spec.get("report_policy") or {}
    return bool(policy.get("enable_report_rag", True))


def render(ctx: Any) -> str:
    spec = getattr(ctx, "research_spec", None) or {}
    policy = spec.get("report_policy") or {}
    budget = int(policy.get("report_rag_max_chars", 1600))
    top_k = int(policy.get("report_rag_top_k", 3))

    endpoint = policy.get("ov_endpoint", "http://127.0.0.1:1933")
    store = OVStore(endpoint=endpoint, inject_max_chars=budget)

    return store.retrieve_report_knowledge(
        focus_facets=getattr(ctx, "focus_facets", ()),
        research_mode=getattr(ctx, "research_mode", "technical"),
        limit=top_k,
        max_chars=budget,
    )

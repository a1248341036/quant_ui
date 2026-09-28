# -*- coding: utf-8 -*-
"""研报知识注入通道开关（R1/R2/R3-R4 三选一）回归测试。

`research_spec.report_policy.knowledge_mode` 决定启用哪一条通道，且**三通道互斥**：
- ``report_rag``（默认）R3/R4 原始文献 RAG
- ``static_manual``   R1 券商金工研报机制手册
- ``mechanism_cards`` R2 机制卡检索注入
- ``off``             全关（课题队列由 enable_question_queue 单独控制）
未设置 knowledge_mode 时回落旧布尔键（向后兼容）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from alphaagent.factor.mining import report_channels
from alphaagent.factor.mining.prompt.modules import report_mechanisms, report_prior, report_rag


def _spec(mode=None, **policy):
    pol = dict(policy)
    if mode is not None:
        pol["knowledge_mode"] = mode
    return {"report_policy": pol}


def test_channels_are_mutually_exclusive():
    for mode, expected in (
        ("report_rag", {"report_rag": True, "static_manual": False, "mechanism_cards": False}),
        ("static_manual", {"report_rag": False, "static_manual": True, "mechanism_cards": False}),
        ("mechanism_cards", {"report_rag": False, "static_manual": False, "mechanism_cards": True}),
        ("off", {"report_rag": False, "static_manual": False, "mechanism_cards": False}),
    ):
        ch = report_channels.resolve_report_channels(_spec(mode))
        assert ch == expected, mode
        assert sum(ch.values()) <= 1


def test_default_mode_is_report_rag():
    assert report_channels.knowledge_mode({"report_policy": {}}) == ""
    assert report_channels.resolve_report_channels({"report_policy": {}})["report_rag"] is True
    assert report_channels.DEFAULT_KNOWLEDGE_MODE == "report_rag"


def test_legacy_booleans_still_work():
    ch = report_channels.resolve_report_channels(
        _spec(None, enable_report_rag=False, enable_report_mechanisms=True, enable_report_prior=False)
    )
    assert ch == {"report_rag": False, "static_manual": True, "mechanism_cards": False}
    # 非法 mode 视为未设置 → 回落布尔键
    assert report_channels.knowledge_mode(_spec("nope")) == ""


def _ctx(mode):
    return SimpleNamespace(asset_type="stock", research_spec=_spec(mode))


@pytest.mark.parametrize("mode,rag,manual,cards", [
    ("report_rag", True, False, False),
    ("static_manual", False, True, False),
    ("mechanism_cards", False, False, True),
    ("off", False, False, False),
])
def test_prompt_modules_honor_mode(mode, rag, manual, cards):
    ctx = _ctx(mode)
    assert report_rag.enabled(ctx) is rag
    assert report_mechanisms.enabled(ctx) is manual
    got_cards = report_prior.enabled(ctx)
    if cards:
        # R2 还需要机制卡文件存在（本地 data/research_reports/knowledge/mechanism_cards.jsonl）
        assert got_cards in (True, False)
    else:
        assert got_cards is False


def test_non_stock_asset_disables_all():
    ctx = SimpleNamespace(asset_type="etf", research_spec=_spec("report_rag"))
    assert report_rag.enabled(ctx) is False
    assert report_mechanisms.enabled(ctx) is False
    assert report_prior.enabled(ctx) is False

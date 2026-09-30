"""研报 RAG 摘录质量回归（2026-10-01 整夜优化）。

背景：RQ_030 实测 OV/本地摘录注入的是**图表目次**（`- 图 1: ... - 表 2: ...`），
对"公式明确"零价值；且 `retrieve_report_knowledge` 把摘录硬截到 280 字符，
1600 的预算永远用不满。
"""
from __future__ import annotations

import inspect

from alphaagent.factor.mining.memory.ov_store import OVStore, _best_paragraph


def test_toc_with_dot_leaders_is_skipped():
    toc = "- 图 1 动量因子累计多空收益（2005.01-2024.02） ................ 5"
    mech = "本节我们构建估值异常 EPA 因子：剔除 Beta、成长与价值风格后，月度 RankIC 均值 6.13%。"
    got = _best_paragraph(toc + "\n\n" + mech, ["因子"])
    assert "EPA" in got
    assert "...." not in got


def test_toc_label_list_is_skipped():
    toc = "- 图 1 净值走势 - 图 2 分组收益 - 图 3 累计收益 - 表 1 参数"
    mech = "因子构建方法：以 20 日换手率残差作为低注意力代理，并对流通市值中性化后取截面排序。"
    got = _best_paragraph(toc + "\n\n" + mech, ["因子"])
    assert "因子构建方法" in got
    assert "图 2" not in got


def test_toc_only_content_returns_nothing():
    """整篇只有图表目次时返回空串，而不是把目次当摘录注入。"""
    toc = "- 图 1 净值走势 - 图 2 分组收益 - 图 3 累计收益 - 表 1 参数"
    assert _best_paragraph(toc, ["因子"]) == ""


def test_formula_paragraph_preferred_over_numeric_table_talk():
    numeric = "回测区间 2005-2024，样本 3000 只，分组 5 档，多空组合年化收益 18.29%，信息比率 3.76。"
    formula = "动量 beta 因子计算方法：先对 ret20 排序，再取 TS_RANK(volume, 20) 作为门控。"
    got = _best_paragraph(numeric + "\n\n" + formula, ["动量"])
    assert "计算方法" in got


def test_max_chars_respected():
    long_para = "因子构建方式说明。" + "细节描述" * 400
    got = _best_paragraph(long_para, ["因子构建"], max_chars=200)
    assert len(got) <= 201


def test_snippet_cap_default_raised():
    """摘录上限必须是可配的、且默认远大于旧值 280，否则 1600 预算用不满。"""
    sig = inspect.signature(OVStore.retrieve_report_knowledge)
    assert "snippet_chars" in sig.parameters
    assert sig.parameters["snippet_chars"].default >= 600

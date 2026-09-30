"""研报 RAG 摘录质量回归（2026-10-01 整夜优化）。

背景：RQ_030 实测 OV/本地摘录注入的是**图表目次**（`- 图 1: ... - 表 2: ...`），
对"公式明确"零价值；且 `retrieve_report_knowledge` 把摘录硬截到 280 字符，
1600 的预算永远用不满。
"""
from __future__ import annotations

import inspect

from alphaagent.factor.mining.memory.ov_store import OVStore, _best_paragraph, _is_recap_doc


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


def test_recap_docs_detected():
    """OV 实测命中的三条复盘噪声都要被识别（好让位给本地因子语料）。"""
    assert _is_recap_doc("viking://resources/research_reports/渤海证券/20240924_5773740_公募基金周报.md")
    assert _is_recap_doc("权益市场主要指数震荡修复房地产领涨.md")
    assert _is_recap_doc("上周市场回顾._1/上周市场回顾._1_1.md")


def test_factor_mechanism_docs_not_flagged_as_recap():
    assert not _is_recap_doc("东吴证券 金融工程 “基本面选股因子”系列：从布林带到估值异常因子")
    assert not _is_recap_doc("海通证券 金融工程 选股因子系列（九十六）：动量beta的择时、优选与alpha因子构建")
    assert not _is_recap_doc("5771450 多因子选股周报：估值因子表现出色，中证500指增组合年内超额12.98%")


def test_local_corpus_prefers_mineru_then_falls_back(tmp_path, monkeypatch):
    """未显式指定语料时优先 MinerU 重抽版（含公式层），缺失才回落 v1 parsed。"""
    import alphaagent.factor.mining.memory.ov_store as ovs

    (tmp_path / "parsed").mkdir()
    monkeypatch.setattr(ovs, "_research_reports_roots", lambda: [tmp_path])
    monkeypatch.delenv("ALPHA_REPORT_CORPUS", raising=False)
    assert ovs._report_local_corpus() == tmp_path / "parsed"          # 只有 v1
    (tmp_path / "parsed_mineru").mkdir()
    assert ovs._report_local_corpus() == tmp_path / "parsed_mineru"   # 有 MinerU 则优先


def test_local_corpus_env_override_wins(tmp_path, monkeypatch):
    import alphaagent.factor.mining.memory.ov_store as ovs

    (tmp_path / "parsed").mkdir()
    (tmp_path / "parsed_mineru").mkdir()
    monkeypatch.setattr(ovs, "_research_reports_roots", lambda: [tmp_path])
    monkeypatch.setenv("ALPHA_REPORT_CORPUS", "parsed")
    assert ovs._report_local_corpus() == tmp_path / "parsed"

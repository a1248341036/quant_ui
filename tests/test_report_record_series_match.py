"""因子清单注入匹配的「同系列姊妹篇防误配」回归（2026-10-01 整夜实测缺陷）。

实测：课题 `之十三 alpha预测` 被对到报告 `之十八 在alpha衰退之前`，19 个命中里约 8 个
是假阳性 —— 复现题面会注入**另一篇报告**的公式。根因：同系列标题共享大量 boilerplate，
字符 bigram Jaccard 越过 0.35 阈值。
"""
from __future__ import annotations

from alphaagent.factor.mining.agent.question_queue import (
    _paren_no,
    _series_no,
    match_report_records,
)

REC_18 = {"source": r"nn73\01\2016-12-05_东方证券_金融工程_《因子选股系列研究之十八》：在alpha衰退之前.md"}
REC_27 = {"source": r"nn73\01\2017-07-09_东方证券_金融工程_《因子选股系列研究之二十七》：预期外的盈利能力.md"}
REC_ZS = {"source": r"nn73\01\2017-03-27_浙商证券_金融工程_hs300指数增强：基于有效因子的多因子增强.md"}
REC_SB3 = {"source": r"nn73\01\2018-01-18_因子投资与smartbeta研究(三)_市场环境与因子组合表现_海通证券.md"}
RECORDS = [REC_18, REC_27, REC_ZS, REC_SB3]


def test_series_no_parsing():
    assert _series_no("《因子选股系列研究之十八》：在alpha衰退之前") == 18
    assert _series_no("研究之二十七") == 27
    assert _series_no("之三十二") == 32
    assert _series_no("之九十六") == 96
    assert _series_no("之十") == 10
    assert _series_no("之100") == 100
    assert _series_no("hs300指数增强基于有效因子") is None


def test_sibling_series_not_matched():
    q13 = {"source": "东方证券《金融工程_因子选股系列研究之十三alpha预测》"}
    assert match_report_records(q13, None, RECORDS) == ([], "")
    q29 = {"source": "东方证券《金融工程_因子选股系列研究之二十九质优股量化投资》"}
    assert match_report_records(q29, None, RECORDS) == ([], "")
    q43 = {"source": "东方证券《金融工程_因子选股系列研究之四十三盈利预测与市价隐含预期收益》"}
    assert match_report_records(q43, None, RECORDS) == ([], "")


def test_same_series_matched():
    q18 = {"source": "东方证券《金融工程_因子选股系列研究之十八在alpha衰退之前》"}
    recs, src = match_report_records(q18, None, RECORDS)
    assert recs and "十八" in src


def test_no_series_still_matches_by_jaccard():
    qz = {"source": "浙商证券《金融工程_hs300指数增强基于有效因子的多因子增强》"}
    recs, src = match_report_records(qz, None, RECORDS)
    assert recs and "浙商" in src


def test_card_path_exact_match_wins():
    card = {"source": {"path": REC_27["source"]}}
    recs, src = match_report_records({"source": "无关课题"}, card, RECORDS)
    assert recs and src == REC_27["source"]


def test_paren_series_guard():
    """括号序号守卫：smartbeta研究（四）不得对到（三）。"""
    assert _paren_no("smartbeta研究（四）单因子多组合还是多因子单组合") == 4
    assert _paren_no("smartbeta研究(三)_市场环境与因子组合表现") == 3
    q4 = {"source": "海通证券《金融工程_因子投资与smartbeta研究（四）单因子多组合还是多因子单组合》"}
    assert match_report_records(q4, None, RECORDS) == ([], "")
    q3 = {"source": "因子投资与smartbeta研究(三)《市场环境与因子组合表现_海通证券》"}
    recs, src = match_report_records(q3, None, RECORDS)
    assert recs and "smartbeta" in src

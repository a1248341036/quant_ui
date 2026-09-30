"""因子清单抽取"门"的回归测试（P0：放宽表头 + 补正文公式段）。

背景：旧门 `因子(名称|序号)…(计算方法|公式|定义)` 全语料 1900 篇只命中 26 篇，
真公式表约 186 篇、正文公式段约 233 篇 —— 复现题面因此拿不到公式。
"""
from __future__ import annotations

import pytest

from scripts.extract_factor_records import (
    build_all_blocks,
    build_blocks,
    build_formula_blocks,
    build_prose_blocks,
    is_formula_header,
)

# 实测语料里的真实表头（旧门全漏）
REAL_HEADERS = [
    "| 因子名称 | 因子含义 | 计算方式 |",
    "| 因子简称 | 计算方法 |",
    "|  | 因子说明 | 计算公式 |",
    "| 因子 | 定义 |",
    "| 因子名称 | 因子定义 | 排序 | 行业中性风格中性 |  |",
]

# 业绩周报表头（无公式可抽，必须排除）
PERF_HEADERS = [
    "| 因子名称 | 因子方向 最近一周 | 最近一月 | 今年以来 | 近1年年化 |  | 历史年化 近一年趋势近十年趋势 |  |",
    "| 因子名称 | 因子方向 | 月收益率 |",
    "| 因子名称 | 沪深300中的多头超额 | 中证500中的多头超额 | 因子释义 |",
]


@pytest.mark.parametrize("row", REAL_HEADERS)
def test_real_formula_headers_accepted(row):
    assert is_formula_header(row) is True


@pytest.mark.parametrize("row", PERF_HEADERS)
def test_performance_headers_rejected(row):
    assert is_formula_header(row) is False


def _table(header: str) -> list[str]:
    return [
        header,
        "| --- | --- | --- |",
        "| EP | 盈利收益率 | 1/PE |",
        "| ROE | 净资产收益率 | 净利润/净资产 |",
    ]


def test_build_blocks_picks_widened_table():
    blocks = build_blocks(_table("| 因子名称 | 因子含义 | 计算方式 |"))
    assert len(blocks) == 1
    assert blocks[0]["rows"][0].startswith("| 因子名称")
    assert len(blocks[0]["rows"]) >= 3


def test_build_blocks_skips_performance_table():
    assert build_blocks(_table("| 因子名称 | 因子方向 | 月收益率 |")) == []
    assert build_blocks(_table("| 因子名称 | 沪深300中的多头超额 | 中证500中的多头超额 | 因子释义 |")) == []


def test_prose_blocks_finds_formula_section():
    lines = [
        "# 某报告",
        "## 因子构建方法",
        "EP 因子定义为盈利收益率，公式如下：",
        "TS_RANK(EP, 20) 为时序排名",
        "## 下一节",
        "无关内容",
    ]
    blocks = build_prose_blocks(lines)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "prose"
    assert any("EP" in r for r in blocks[0]["rows"])


def test_build_prose_blocks_skips_section_without_formula():
    lines = ["## 因子构建方法", "本报告不披露具体构造。", "## 下一节", "x"]
    assert build_prose_blocks(lines) == []


def test_build_all_blocks_merges_with_unique_index():
    lines = _table("| 因子名称 | 因子含义 | 计算方式 |") + [
        "",
        "## 因子构建方法",
        r"R O E _ { t } = \frac { 净利润 _ { t } } { 净资产 _ { t } }",
        "TS_RANK(EP, 20) 为 EP 的时序排名",
    ]
    blocks = build_all_blocks(lines)
    assert len(blocks) >= 2
    assert [b["index"] for b in blocks] == list(range(len(blocks)))
    assert "table" in {b.get("kind", "table") for b in blocks}


def test_formula_blocks_catch_cluster_without_heading():
    """公式成片出现在小节标题之外（MinerU LaTeX 段）时必须被捕获。"""
    lines = [
        "# 报告",
        "以下为因子定义。",
        r"O C V P _ { t } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left( \frac { V O L } { V O L } \right)",
        r"T W O C V P _ { t } = \sum _ { i = 1 } ^ { \infty } w _ { t - i } * C V P _ { t - i }",
        r"B C V P _ { t } = \frac { V O L _ { b i d } } { V O L _ { t o t a l } }",
        "资料来源：研究所",
    ]
    blocks = build_formula_blocks(lines)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "formula"
    assert blocks[0]["n_rows"] >= 2
    assert any("OCVP" in normalize_row(r) for r in blocks[0]["rows"])


def normalize_row(s: str) -> str:
    return s.replace(" ", "")


def test_formula_blocks_capped_and_empty_safe():
    assert build_formula_blocks(["# 无公式", "只有文字"]) == []
    lines = [f"EP{i} = ROE{i} + PB{i}" for i in range(30)]  # 连续 30 行公式 → 单簇，不超上限
    assert len(build_formula_blocks(lines, max_blocks=2)) == 1
    # 单行公式不成块（噪声太大，交给 prose 通道）
    assert build_formula_blocks(["EP = ROE + PB"]) == []

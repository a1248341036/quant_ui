"""机制卡抽取的 JSON 容错回归测试（extract_report_cards.extract_json）。

回归背景（2026-09-30）：模型在 ``formula.text`` / ``dsl_hint`` 里输出 LaTeX 裸反斜杠
（``\\;``、``\\sigma``、``\\frac``）时，那段文本不是合法 JSON 转义序列，
``json.loads`` 直接抛错 → 整篇报告被判 FAIL。实测把 ``--max-tokens`` 从 1500 提到 3000
复现完全相同的失败，确认根因不是回复被截断而是转义非法。

本测试只覆盖纯函数，不联网、不读语料。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_report_cards import extract_json, repair_json_escapes  # noqa: E402


def test_plain_json_parses() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_parses() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_embedded_in_prose_parses() -> None:
    assert extract_json('好的，结果如下：\n{"a": {"b": 2}}\n希望有帮助。') == {"a": {"b": 2}}


def test_latex_bare_backslash_is_repaired() -> None:
    # 真实失败样本形态：\; 与 \sigma 都不是合法 JSON 转义
    raw = '{"formula": {"text": "stochastic = \\; straition_{f_1} + \\sigma^2"}}'
    obj = extract_json(raw)
    assert obj is not None
    assert "straition_{f_1}" in obj["formula"]["text"]
    assert "\\sigma^2" in obj["formula"]["text"]


def test_malformed_unicode_escape_is_repaired() -> None:
    # \u 后面不足 4 位十六进制 → 非法
    obj = extract_json('{"x": "\\u12"}')
    assert obj == {"x": "\\u12"}


def test_valid_escapes_are_preserved() -> None:
    obj = extract_json('{"x": "line1\\nline2", "y": "C:\\\\path"}')
    assert obj == {"x": "line1\nline2", "y": "C:\\path"}


def test_repair_is_identity_on_valid_json() -> None:
    s = '{"a": "b", "c": [1, 2], "d": "e\\nf"}'
    assert repair_json_escapes(s) == s


def test_repair_only_touches_bad_backslashes() -> None:
    # `\f` `\b` `\t` 本身是合法 JSON 转义 → 修复不动它们（\frac 保持原样，
    # 其"换页符+rac"的语义问题由公式回证环节兜底：回证不上就不写入 formula_text）
    assert repair_json_escapes(r"\; \sigma \frac{1}{2}") == r"\\; \\sigma \frac{1}{2}"


def test_valid_double_backslash_pair_is_not_re_escaped() -> None:
    """回归：早先的负向环视实现会把合法的 ``\\\\cdots`` 二次转义成 ``\\\\\\cdots``。

    真实失败样本（2024-04-24 季报解析）：
    ``"text": "stochastic = \\; straition_{f_1} + \\\\cdots + ..."``
    第一个 ``\\;`` 非法、第二个 ``\\\\cdots`` 合法，必须只修前者。
    """
    raw = r'{"t": "a \\cdots b \; c"}'
    obj = extract_json(raw)
    assert obj == {"t": "a \\cdots b \\; c"}


def test_non_dict_returns_none() -> None:
    assert extract_json("[1, 2, 3]") is None
    assert extract_json("这里没有 JSON") is None
    assert extract_json("") is None

"""字段别名 + 研报课题基本面联动的回归测试（2026-09-28）。

背景（1200 道研报课题的字段对账）：
- 课题/记忆推荐常写 `$turnover`（语义=换手率），面板真实列名是 `turnover_rate`
  → 旧行为直接报"表达式引用了不可用字段"白丢一次评估（63 次字段命中）。
- 自由探索（未聚焦）时后端会给子进程加 `--no-fundamentals`，但课题库 45% 的题
  点名 eps/roe/net_profit 等 funda_* 字段 → 课题必然落空。现按"题库是否含基本面
  组课题"自动判定是否载入基本面列。
"""
from __future__ import annotations

import pytest

from alphaagent.dsl import eval as dsl_eval


def test_turnover_alias_rewrites_to_real_column():
    code = dsl_eval.compile_multi_line_factor(
        "turn = $turnover\nCS_ZSCORE(turn)", columns=["turnover_rate", "close"]
    )
    assert "$turnover_rate" in code
    assert "$turnover\n" not in code


def test_alias_does_not_touch_real_columns():
    code = dsl_eval.compile_multi_line_factor(
        "a = $turnover_rate\nb = $turnover_rate_f\nCS_ZSCORE(SUBTRACT(a, b))",
        columns=["turnover_rate", "turnover_rate_f"],
    )
    assert "$turnover_rate\n" in code and "$turnover_rate_f" in code


def test_unknown_field_still_rejected():
    with pytest.raises(Exception) as exc:
        dsl_eval.compile_multi_line_factor("z = $not_a_real_field", columns=["close"])
    assert "不可用字段" in str(exc.value)


def test_question_bank_fundamentals_detection(monkeypatch):
    from alphaagent.factor.mining.agent import question_queue as qq
    import backend.alphaagent_service as svc

    monkeypatch.setattr(
        qq, "load_question_queue",
        lambda spec=None: [{"facets": ["价量面"]}, {"facets": ["基本面", "价量面"]}],
    )
    assert svc._question_bank_needs_fundamentals() is True

    monkeypatch.setattr(qq, "load_question_queue", lambda spec=None: [{"facets": ["价量面"]}])
    assert svc._question_bank_needs_fundamentals() is False

    def _boom(spec=None):
        raise RuntimeError("bank unreadable")

    monkeypatch.setattr(qq, "load_question_queue", _boom)
    # 判定失败必须退回旧行为（不因题库读不到就把 run 打挂）
    assert svc._question_bank_needs_fundamentals() is False

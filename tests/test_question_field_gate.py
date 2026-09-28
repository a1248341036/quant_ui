# -*- coding: utf-8 -*-
"""课题（research question）字段门控单测（2026-09-28）。

规格：docs/specs/alphaagent_question_field_gate_spec.md §3.3 / §4.1。

判定要点：
- 纯缺列题（题面可判定字段全部落在未载入族上）跳过并顺延；
- 部分缺列题保留，但在任务块里注入「字段可用性」提示行；
- 题面只有无法判定的散文词 → 放行；
- 未传 ``available_fields`` → 与旧行为逐字节一致（零回归）。
"""
from __future__ import annotations

import hashlib
import json

import pytest

from alphaagent.factor.mining.agent.question_queue import (
    DEFAULT_QUESTION_FIELD_GATE,
    classify_question_fields,
    get_question_for_turn,
    get_task_for_turn,
    load_question_queue,
    question_missing_fields,
    render_question_task,
    resolve_question_field_gate,
)

# 一次自由探索 run 的可用列（2026-09-27 整夜实测：行情族 + 资金/事件/股东/业绩/机构，
# 无 funda_/industry_/div_/mgn_）
_TECHNICAL_COLUMNS = [
    "adj_close", "adj_open", "adj_high", "adj_low", "close", "open", "high", "low",
    "volume", "amount", "float_cap", "turnover_rate",
    "ff_super_net", "dt_net_buy_90d", "holder_count_chg_pct", "inst_count",
    "pred_surprise", "exp_net_profit",
]

_FIXTURE_QUESTIONS: list[dict] = [
    {   # 0：纯基本面题 → 必须跳过
        "question_id": "RQ_T0",
        "topic": "现金流质量",
        "hypothesis": "OCF/总资产越高越好",
        "suggested_fields": ["funda_ocf", "funda_total_assets", "roe"],
        "suggested_operators": ["CS_ZSCORE"],
        "expected_shape": "单调正向",
        "expected_sign": "positive",
        "falsifier": "无",
        "facets": ["基本面"],
    },
    {   # 1：部分缺列题（eps 缺）→ 保留 + 提示
        "question_id": "RQ_T1",
        "topic": "换手率波动",
        "hypothesis": "换手率波动高的股票后续收益高",
        "suggested_fields": ["amount", "close", "eps", "high", "low"],
        "suggested_operators": ["TS_STD"],
        "expected_shape": "单调正向",
        "expected_sign": "positive",
        "falsifier": "无",
        "facets": ["价量面", "量能面"],
    },
    {   # 2：全部可用
        "question_id": "RQ_T2",
        "topic": "大单净流入持续性",
        "hypothesis": "超大单净买入对未来收益有正预测",
        "suggested_fields": ["ff_super_net", "amount"],
        "suggested_operators": ["CS_ZSCORE"],
        "expected_shape": "单调正向",
        "expected_sign": "positive",
        "falsifier": "无",
        "facets": ["资金面"],
    },
    {   # 3：只有散文词 → 放行
        "question_id": "RQ_T3",
        "topic": "拥挤交易衰减",
        "hypothesis": "拥挤度高的标的后续走弱",
        "suggested_fields": ["crowding", "unmapped_prose_word"],
        "suggested_operators": ["SOFT_GATE"],
        "expected_shape": "单调负向",
        "expected_sign": "negative",
        "falsifier": "无",
        "facets": ["拥挤面"],
    },
]


@pytest.fixture()
def queue_file(tmp_path):
    path = tmp_path / "research_questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in _FIXTURE_QUESTIONS),
        encoding="utf-8",
    )
    return path


def _spec(queue_file) -> dict:
    return {"report_policy": {"question_queue_file": str(queue_file)}}


def _ungated_pick(turn: int, queue: list[dict], session_id: str | None = None) -> str | None:
    """复刻旧实现 ``(offset + turn) % len(pool)``。"""
    offset = 0
    if session_id:
        offset = int(hashlib.md5(session_id.encode("utf-8")).hexdigest(), 16)
    return queue[(offset + turn) % len(queue)]["question_id"]


# ── 零回归 ────────────────────────────────────────────────────────────


def test_ungated_matches_legacy_selection():
    """未传 available_fields → 逐轮选题与旧实现完全一致（真实 1200 题题库）。"""
    queue = load_question_queue()
    assert len(queue) > 100
    for turn in range(10):
        picked = get_question_for_turn(turn)
        assert picked is not None
        assert picked["question_id"] == _ungated_pick(turn, queue)


def test_ungated_matches_legacy_selection_with_focus():
    """聚焦面过滤 + 未传 available_fields → 仍与旧实现一致。"""
    queue = load_question_queue()
    focus = ("价量面",)
    pool = [q for q in queue if set(q.get("facets", [])) & set(focus)]
    assert pool
    for turn in range(5):
        picked = get_question_for_turn(turn, focus_facets=focus)
        assert picked is not None
        assert picked["question_id"] == _ungated_pick(turn, pool)


def test_gate_disabled_restores_legacy(queue_file):
    """question_field_gate=false → 即使传了可用列也走旧行为。"""
    spec = _spec(queue_file)
    spec["report_policy"]["question_field_gate"] = False
    q = get_question_for_turn(0, spec=spec, available_fields=_TECHNICAL_COLUMNS)
    assert q is not None and q["question_id"] == "RQ_T0"  # 旧行为：不跳过纯缺列题


# ── 门控行为 ──────────────────────────────────────────────────────────


def test_pure_missing_family_question_is_skipped(queue_file):
    """纯基本面题（字段全缺）跳过 → 顺延到 RQ_T1。"""
    stats: dict = {}
    q = get_question_for_turn(
        0, spec=_spec(queue_file), available_fields=_TECHNICAL_COLUMNS, stats=stats
    )
    assert q is not None and q["question_id"] == "RQ_T1"
    assert stats["status"] == "partial"
    assert stats["scanned"] == 2
    assert stats["missing"] == ["eps"]


def test_partial_question_keeps_and_warns(queue_file):
    """部分缺列题保留，并在任务块注入「字段可用性」提示行。"""
    spec = _spec(queue_file)
    task = get_task_for_turn(1, spec=spec, available_fields=_TECHNICAL_COLUMNS)
    assert "### 本轮研报定向攻关课题" in task
    assert "RQ_T1" in task
    assert "字段可用性" in task and "eps" in task


def test_all_available_question_status_ok(queue_file):
    stats: dict = {}
    q = get_question_for_turn(
        2, spec=_spec(queue_file), available_fields=_TECHNICAL_COLUMNS, stats=stats
    )
    assert q is not None and q["question_id"] == "RQ_T2"
    assert stats["status"] == "ok" and stats["missing"] == []


def test_prose_only_question_passes(queue_file):
    """题面只有散文词 → 判据不足 → 放行（不因词表不足误杀）。"""
    q = get_question_for_turn(
        3, spec=_spec(queue_file), available_fields=_TECHNICAL_COLUMNS
    )
    assert q is not None and q["question_id"] == "RQ_T3"


def test_all_questions_ineligible_returns_none(queue_file):
    """连续 scan_limit 道题全不合格 → 本轮不派课题（None），不阻塞 run。"""
    spec = _spec(queue_file)
    spec["report_policy"]["question_field_gate_scan_limit"] = 1
    stats: dict = {}
    q = get_question_for_turn(
        0, spec=spec, available_fields=["close"], stats=stats
    )
    assert q is None
    assert stats["status"] == "skipped"


def test_min_ratio_threshold_respected(queue_file):
    """min_ratio 提高后，部分缺列题也会被跳过。"""
    spec = _spec(queue_file)
    spec["report_policy"]["question_field_gate_min_ratio"] = 0.95
    # RQ_T0（0/3 可用）与 RQ_T1（4/5 = 0.8 < 0.95）都不合格 → 落到 RQ_T2
    q = get_question_for_turn(0, spec=spec, available_fields=_TECHNICAL_COLUMNS)
    assert q is not None and q["question_id"] == "RQ_T2"


def test_warn_can_be_disabled(queue_file):
    spec = _spec(queue_file)
    spec["report_policy"]["question_field_warn"] = False
    task = get_task_for_turn(1, spec=spec, available_fields=_TECHNICAL_COLUMNS)
    assert "RQ_T1" in task
    assert "字段可用性" not in task


# ── 分类与配置（纯函数）──────────────────────────────────────────────


def test_classify_question_fields_buckets():
    avail = set(_TECHNICAL_COLUMNS)
    hits, missing, unknown = classify_question_fields(
        {"suggested_fields": ["amount", "close", "eps", "funda_ocf", "weird_prose"]}, avail
    )
    assert set(hits) == {"amount", "close"}
    assert set(missing) == {"eps", "funda_ocf"}
    assert unknown == ["weird_prose"]


def test_missing_fields_uses_dsl_alias():
    """``turnover`` 走 DSL 别名 → 面板有 turnover_rate 时不算缺列。"""
    assert question_missing_fields({"suggested_fields": ["turnover"]}, ["turnover_rate"]) == ()
    assert question_missing_fields({"suggested_fields": ["turnover"]}, ["close"]) == ("turnover",)


def test_resolve_gate_defaults_and_override():
    assert resolve_question_field_gate(None) == DEFAULT_QUESTION_FIELD_GATE
    cfg = resolve_question_field_gate(
        {"report_policy": {"question_field_gate_min_ratio": 0.8, "question_field_gate_scan_limit": 7}}
    )
    assert cfg["min_ratio"] == 0.8 and cfg["scan_limit"] == 7
    # 运行期 override 优先
    cfg2 = resolve_question_field_gate(
        {"report_policy": {"question_field_gate_min_ratio": 0.8}}, override={"min_ratio": 0.2}
    )
    assert cfg2["min_ratio"] == 0.2


def test_render_question_task_without_warning():
    text = render_question_task(_FIXTURE_QUESTIONS[2], missing_fields=["eps"], spec=None)
    assert "RQ_T2" in text and "字段可用性" in text


def test_question_gate_falls_back_to_builtin_when_file_empty(tmp_path):
    """空题库文件 → 回退内置 5 题（load_question_queue 既有语义），门控照常生效。"""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    stats: dict = {}
    q = get_question_for_turn(0, spec=_spec(empty), available_fields=_TECHNICAL_COLUMNS, stats=stats)
    assert q is not None and q["question_id"].startswith("RQ_")
    assert stats["status"] in {"ok", "partial", "ungated"}

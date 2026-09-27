# -*- coding: utf-8 -*-
"""测试 R4 研报研究问题队列加载与轮次任务派发。"""
from alphaagent.factor.mining.agent.question_queue import (
    load_question_queue,
    get_task_for_turn,
)


def test_load_default_question_queue():
    questions = load_question_queue()
    assert len(questions) >= 5
    for q in questions:
        assert q["question_id"].startswith("RQ_")
        assert "hypothesis" in q
        assert "suggested_fields" in q
        assert len(q["facets"]) >= 1


def test_get_task_for_turn():
    task_0 = get_task_for_turn(0)
    assert "### 本轮研报定向攻关课题【RDAgent 研究问题驱动】" in task_0
    assert "RQ_" in task_0
    assert "推荐正交补充面" in task_0
    assert "SOFT_GATE" in task_0

    task_1 = get_task_for_turn(1)
    assert "RQ_" in task_1

    # 数据面聚焦匹配
    task_focus = get_task_for_turn(0, focus_facets=("基本面",))
    assert "基本面" in task_focus or "RQ_" in task_focus

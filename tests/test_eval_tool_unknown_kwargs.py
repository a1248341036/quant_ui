# -*- coding: utf-8 -*-
"""评估工具未知关键字参数防御测试。

LLM 常把 submit_factor 的 comment / eval_on_val_set 的 expected_sign
误传给 evaluate_factor / eval_on_train_set —— 过去直接 TypeError 白烧整轮。
修复：三个 eval 工具函数收 `**_legacy_kwargs`，吸收未声明参数并回传
ignored_arguments（与 submit_factor 的既有模式一致）。
"""

from __future__ import annotations

import asyncio
import inspect

from alphaagent.factor.mining.agent.agentscope_tools import build_factor_eval_toolkit
from alphaagent.factor.mining.tools import FactorEvalTools


class _ProfileService:
    def _eval(self, r, profile_id):
        self.request = r
        return {
            "ok": True,
            "split": "val",
            "profile": {"profile_id": profile_id},
            "profile_hash": "profilehash",
            "candidate": {"candidate_id": "cand_1", "factor_name": r.factor_name, "expression": r.multi_line_expr},
            "metrics": {"cross_sectional_core": {"ic": 0.02, "icir": 0.3, "rank_ic": 0.02, "factor_coverage": 1.0}},
            "rule_results": [],
        }

    def eval_profile(self, request):
        return self._eval(request, request.profile_id)

    def eval_train(self, request):
        return self._eval(
            type("R", (), {
                "session_id": request.session_id,
                "profile_id": "train_screen",
                "multi_line_expr": request.multi_line_expr,
                "factor_name": request.factor_name,
            })(),
            "train_screen",
        )

    def eval_val(self, request):
        return self._eval(
            type("R", (), {
                "session_id": request.session_id,
                "profile_id": "validation",
                "multi_line_expr": request.multi_line_expr,
                "factor_name": request.factor_name,
            })(),
            "validation",
        )


def _tool_func(name: str):
    service = _ProfileService()
    tools = FactorEvalTools(service, "session")
    toolkit = build_factor_eval_toolkit(tools, max_workers=1)

    async def _get():
        return await toolkit.get_tool(name)

    tool = asyncio.run(_get())
    return tool._func


def _chunk_text(chunk) -> str:
    return "".join(b.text for b in (getattr(chunk, "content", None) or []) if hasattr(b, "text"))


def _run(fn, **kwargs):
    return asyncio.run(fn(**kwargs))


def test_tool_signatures_accept_unknown_kwargs():
    """三个 eval 工具签名含 **kwargs（不再 TypeError）。"""
    for name in ("evaluate_factor", "eval_on_train_set", "eval_on_val_set"):
        sig = inspect.signature(_tool_func(name))
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD in kinds, f"{name} 缺少 **kwargs 防御"


def test_evaluate_factor_ignores_comment_and_expected_sign():
    """evaluate_factor 收到多余 comment/expected_sign 不再 TypeError，并回传提醒。"""
    out = _run(
        _tool_func("evaluate_factor"),
        multi_line_expr="$adj_close",
        profile_id="train_screen",
        prediction={
            "expected_shape": "monotonic_increasing",
            "expected_strong_side": "high_factor",
            "expected_sign": 1,
        },
        comment="should be ignored",
        expected_sign=1,
    )
    text = _chunk_text(out)
    assert text
    assert "ignored_arguments" in text
    assert "comment" in text and "expected_sign" in text


def test_eval_on_train_set_ignores_comment():
    """eval_on_train_set 收到多余 comment 不再 TypeError。"""
    out = _run(
        _tool_func("eval_on_train_set"),
        multi_line_expr="$adj_close",
        prediction={
            "expected_shape": "monotonic_increasing",
            "expected_strong_side": "high_factor",
            "expected_sign": 1,
        },
        comment="should be ignored",
    )
    text = _chunk_text(out)
    assert text
    assert "ignored_arguments" in text


def test_eval_on_val_set_ignores_comment():
    """eval_on_val_set 收到多余 comment 不再 TypeError。"""
    out = _run(
        _tool_func("eval_on_val_set"),
        multi_line_expr="$adj_close",
        expected_sign=1,
        comment="should be ignored",
    )
    text = _chunk_text(out)
    assert text
    assert "ignored_arguments" in text
# -*- coding: utf-8 -*-
"""数据面聚焦的提示词投影：让 LLM 看不到未选数据面的可复制代码片段。

与工具层硬拦截（tools/_dispatch._facet_lock_block）配套——拦截保证"越界表达式
不会被执行"，本模块保证"越界示例根本不出现在上下文里"，从源头消除 LLM
试探-被拦-重写的 token 浪费。

规则：聚焦面非空时，任何**含 `$列` 引用**的代码片段（行内 `` `...` `` 或
围栏 ``` 块），若其触及的数据面不全落在聚焦面内，就整段替换为一行说明。
纯算子名/无列引用的片段（如 ``TS_MEAN``）不受影响——通用算子对所有面都可用。
"""
from __future__ import annotations

import re

from alphaagent.factor.mining.memory.expressions import expr_facets, facet_allowed_scope

_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_RE = re.compile(r"`([^`\n]+)`")


def _note(outside: list[str]) -> str:
    return "（本 run 未选「" + "、".join(outside) + "」数据面，示例略）"


def _violating(expr: str, allowed: set[str], required: set[str]) -> list[str] | None:
    """返回越界面列表；片段无列引用或完全合规 → None。

    - 触及勾选面及其隐含输入面之外的面 → 越界（整段裁掉）；
    - 算子调用片段（含括号）只用了输入列、没触及任何勾选面 → 视为越界示例
      （与工具层"必须触及勾选面"同口径，防止 LLM 照抄纯价量片段）；
    - 裸列名提及（字段表行）不套用"必须触及"规则——CHIP_* 的输入列需要在表里可见。
    """
    facets = expr_facets(expr)
    if not facets:
        return None
    outside = sorted(facets - allowed)
    if outside:
        return outside
    if required and "(" in expr and not (facets & required):
        return sorted(facets)
    return None


def scrub_out_of_scope(text: str, focus_facets) -> str:
    """把触及未选数据面的代码片段替换为说明行（聚焦为空时原样返回）。"""
    focus = {str(f) for f in (focus_facets or ()) if f}
    if not focus or not text:
        return text
    allowed = facet_allowed_scope(focus)

    def _fence(match: re.Match[str]) -> str:
        outside = _violating(match.group(0), allowed, focus)
        return _note(outside) if outside else match.group(0)

    def _inline(match: re.Match[str]) -> str:
        outside = _violating(match.group(1), allowed, focus)
        return _note(outside) if outside else match.group(0)

    return _INLINE_RE.sub(_inline, _FENCE_RE.sub(_fence, text))

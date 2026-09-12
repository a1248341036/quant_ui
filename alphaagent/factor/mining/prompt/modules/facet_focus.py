# -*- coding: utf-8 -*-
"""模块 13.5 · facet_focus：数据面聚焦指令（用户多选，非空时挂载）。

用户在 run 表单勾选数据面（与 expressions.FACET_DEFS 对齐）后：
- ≥2 面：跨面融合模式（融合算子 + _x_ 命名 + 单面因子须说明失败原因）；
- 单面：单面聚焦模式（表达式应触及该面，不要求融合）。
与 CLI 用户消息块、每轮记忆提醒同口径——三层分工：system prompt 板块=约束契约、
user message=任务指令、每轮提醒=持续引导。
"""

NAME = "facet_focus"
TITLE = "数据面聚焦指令（用户指定）"
ORDER = 135
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"

# 不属于任何数据面的通用列：聚焦时始终可用（市值/可交易标记/行业分组）
_NEUTRAL_COLUMNS = ("float_cap", "tot_cap", "is_trade", "not_st", "industry_sw_l1", "industry_zx_l1")


def enabled(ctx) -> bool:  # noqa: ANN001
    return bool(getattr(ctx, "focus_facets", None))


def _whitelist_lines(ctx, facets: list[str]) -> list[str]:  # noqa: ANN001
    """按 panel 实际列生成"本 run 唯一可用列"白名单（LLM 只看到勾选面的数据）。

    勾选面的专属算子若隐含消费其他面的列（如 CHIP_* 需要 close/low/high/volume），
    这些输入列一并列出但**分组呈现**：与聚焦列混排时输入列按字母序排在最前
    （$adj_* 优先入目），实测会诱导 LLM 直接拼纯价量因子（2026-09-10 run
    51e02d47a3f3 turn0 被拦截的根因）。
    """
    from alphaagent.factor.mining.memory.expressions import (
        expr_facets,
        facet_allowed_scope,
        facet_column_hint,
    )

    scope = facet_allowed_scope(facets)
    input_faces = scope - set(facets)
    hints = "；".join(f"{f} 可用: {facet_column_hint(f) or '—'}" for f in facets)
    cols = getattr(ctx, "panel_columns", None)
    if not cols:
        return [
            f"- 本 run 可用列族（白名单）：{hints}。",
            "- 其他数据面的列/算子一律不可用——越界表达式会被工具层直接拦截。",
        ]
    focus_set = set(facets)
    focus_cols: list[str] = []
    input_cols: list[str] = []
    for name in [str(c) for c in cols]:
        if name in _NEUTRAL_COLUMNS:
            continue
        hit = expr_facets("$" + name)
        if not hit or not hit <= scope:
            continue
        (focus_cols if hit & focus_set else input_cols).append("$" + name)
    focus_cols.sort()
    input_cols.sort()
    neutral = [f"${c}" for c in _NEUTRAL_COLUMNS if c in set(cols)]
    out: list[str] = []
    if focus_cols:
        out.append(
            "- 本 run 唯一可用列（白名单，表达式只能引用这些）——**聚焦面列**"
            "（表达式必须至少触及其中之一）：" + "、".join(focus_cols) + "。"
        )
    else:
        out.append(f"- 本 run 可用列族（白名单）：{hints}（该面以算子派生为主）。")
    if input_cols:
        out.append(
            "- **仅作输入列**（不属于聚焦面，不能单独构成因子）：" + "、".join(input_cols)
            + "。只能与聚焦面列组合使用（比值/相关/门控等），或作为聚焦面专属算子的输入。"
        )
    if neutral:
        out.append("- 通用中性列（不属于任何数据面，随时可用）：" + "、".join(neutral) + "。")
    if input_faces:
        from alphaagent.factor.mining.prompt.modules.operator_catalog import _focused_prefixes

        prefixes = _focused_prefixes(facets)
        anchor = (
            "、".join(prefixes) + " 等聚焦面专属算子需要它们作输入"
            if prefixes
            else "比值/相关等组合构造需要它们"
        )
        out.append(
            "- 「" + "、".join(sorted(input_faces)) + "」仅作为聚焦面算子的输入面"
            "（" + anchor + "）；单独用它们构造因子仍会被拦截——表达式必须触及勾选的聚焦面。"
        )
    out.append(
        "- 白名单之外的列（其他数据面）本 run 不可用；系统提示其他章节若出现相关字段或"
        "示例，一律视为不适用（越界表达式会被工具层直接拦截，不必尝试）。"
    )
    if cols is not None:
        present = set(str(c) for c in cols)
        not_present = [c for c in _NEUTRAL_COLUMNS if c not in present]
        if not_present:
            out.append(
                "- 上方未列出的字段名（如 "
                + "、".join("`$" + c + "`" for c in not_present)
                + " 等行业/规模列）**本 run 面板并未加载**，引用即报「不可用字段」并被拦截"
                "——只使用上方明确列出的列，不要凭熟悉度猜测列名（聚焦 run 不会加载未选"
                "数据面的辅助列族）。"
            )
    return out


def render(ctx) -> str:  # noqa: ANN001
    facets = list(getattr(ctx, "focus_facets", None) or [])
    if not facets:
        return ""
    lines: list[str]
    if len(facets) >= 2:
        lines = [
            "## 数据面聚焦（用户指定，优先级最高，硬锁定）",
            f"本轮挖掘跨面融合、锁定以下数据面：{'、'.join(facets)}。",
            "- 优先构造同时触及 ≥2 个所选面的融合因子，融合模式（按历史命中率优先）：",
            "  ① 分组条件 CS_GROUP_RANK(面A信号, CS_BUCKET(面B门控,5))；② 分歧表达 DIVERGENCE_RANK(面A, 面B)；",
            "  ③ 正交残差 CS_RESIDUALIZE(主信号, CS_BUCKET(面B控制变量,10))；④ 条件门控 GATED_SIGNAL(主信号, 面B门控, 阈值)；",
            "  ⑤ 比值 DIVIDE(面A, 面B 规模)；⑥ 链式组合（分歧→门控/残差→平滑）——"
            "只传末位结构算子的 interaction 契约。禁止 MULTIPLY：默认 spec 直接拦截。",
            "- 同一交互算子不要连续使用超过 2 次——同模板边际递减，结构轮换优先。",
            "- 单面因子只有在融合尝试失败后才能提交，且 eval 调用须在 edit_note 里说明失败原因。",
            "- 因子名用 _x_ 连接面名（如 funda_mom_x_price），便于辨识融合因子。",
        ]
    else:
        lines = [
            "## 数据面聚焦（用户指定，优先级最高，硬锁定）",
            f"本轮挖掘聚焦【{facets[0]}】：表达式只能使用该面的数据列/算子，不要漂移到其他数据面。",
            "- 单面聚焦不要求跨面融合，正常按该面思路构造并提交因子即可。",
        ]
    lines.append(
        "- 硬性锁定（越界即拦截）：只允许引用上述聚焦面的列族/算子；触达未选面列/算子"
        "或完全未触及聚焦面 = 越界，evaluate/submit 会直接返回 facet_lock_violation 且不执行——"
        "请按报错改用锁定列族重写（$float_cap 等非面中性列不受限）。"
    )
    lines.extend(_whitelist_lines(ctx, facets))
    # 豁免声明：ResearchSpec 契约（extra_instructions 板块）渲染了
    # "允许的信号族"白名单，与融合指令直接矛盾——聚焦生效时必须显式解禁，
    # 否则 LLM 会回避触及指定面的表达式。
    lines.append(
        "- 信号族白名单对本指令豁免：触及上述聚焦面的因子（含融合因子）"
        "不受 ResearchSpec 中允许/禁止信号族白名单限制；禁止项仅保留"
        "「纯市值暴露套壳」等无信息结构，Reviewer 仍按原标准审查。"
    )
    return "\n".join(lines)

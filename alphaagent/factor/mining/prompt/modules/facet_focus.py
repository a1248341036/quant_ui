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


def enabled(ctx) -> bool:  # noqa: ANN001
    return bool(getattr(ctx, "focus_facets", None))


def render(ctx) -> str:  # noqa: ANN001
    facets = list(getattr(ctx, "focus_facets", None) or [])
    if not facets:
        return ""
    lines: list[str]
    if len(facets) >= 2:
        lines = [
            "## 数据面聚焦（用户指定，优先级最高）",
            f"本轮挖掘聚焦以下数据面的因子与跨面融合：{'、'.join(facets)}。",
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
            "## 数据面聚焦（用户指定，优先级最高）",
            f"本轮挖掘聚焦【{facets[0]}】：表达式应触及该面的算子或数据列，不要漂移到其他数据面。",
            "- 单面聚焦不要求跨面融合，正常按该面思路构造并提交因子即可。",
        ]
    # 豁免声明：ResearchSpec 契约（extra_instructions 板块）渲染了
    # "允许的信号族"白名单，与融合指令直接矛盾——聚焦生效时必须显式解禁，
    # 否则 LLM 会回避触及指定面的表达式。
    lines.append(
        "- 信号族白名单对本指令豁免：触及上述聚焦面的因子（含融合因子）"
        "不受 ResearchSpec 中允许/禁止信号族白名单限制；禁止项仅保留"
        "「纯市值暴露套壳」等无信息结构，Reviewer 仍按原标准审查。"
    )
    return "\n".join(lines)

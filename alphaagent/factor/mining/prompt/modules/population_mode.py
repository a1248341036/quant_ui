# -*- coding: utf-8 -*-
"""模块 13 · population_mode：种群批量模式提示（propose_population 启用时挂载）。"""

NAME = "population_mode"
TITLE = "种群批量模式"
ORDER = 130
REQUIRED = False
SEP_BEFORE = "\n\n"


def enabled(ctx) -> bool:  # noqa: ANN001
    return bool(ctx.population_max and ctx.population_max > 0)


def render(ctx) -> str:  # noqa: ANN001
    return (
        f"**种群批量模式已启用（`propose_population`，单轮候选上限 {ctx.population_max}）**："
        "需要做参数敏感性扫描或机制邻域探索时，优先用该工具一次性覆盖整个参数网格，"
        "再对 top 候选用 `evaluate_factor(train_screen)` 复核并提交；"
        "避免逐条手工试参。骨架模板用 `{param}` 占位符，网格总量不要铺满上限。"
    )

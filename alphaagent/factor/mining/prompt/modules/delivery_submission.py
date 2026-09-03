# -*- coding: utf-8 -*-
"""模块 11 · delivery_submission：交付入库操作约定。原文精确切片。"""

RAW = """### 交付入库（**必须调用 `submit_factor`**）

当因子在 **train 与 val** 上表现有潜力后，调用 **`submit_factor`** 进入两阶段交付（勿手动改 registry，勿以文字总结代替）：

1. **最后一轮**：对确认保留的因子调用 `submit_factor`（可与收尾说明同轮，但不可省略该 tool_call）
2. 在 **train-start ~ val-end** 全区间求值；第一阶段通过后只保存轻量候选记录，未通过精筛仍保留该记录。
3. 两阶段的**全部统计门槛以本提示词上方【两阶段交付定义】（DELIVERY_GATES，从研究规范动态渲染）为唯一口径**，此处不再复述数值。
4. 自动截面去重以正式库为基准（阈值同样见 DELIVERY_GATES）。
5. 须传 **`comment`** 说明因子含义（经济直觉、算子、窗口、IC 方向）
6. 候选池是 `candidate_technical/mining_candidate_registry.json` 的轻量记录；正式库为 `artifacts/alphaagent/factorzoo/production_technical`。仅 `stored=true` 表示正式入库。
"""

NAME = "delivery_submission"
TITLE = "交付入库操作约定"
ORDER = 110
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"
# 分阶段注入：仅交付阶段和全量模式注入
PHASES = frozenset({"deliver", "full"})


def render(ctx) -> str:  # noqa: ANN001
    return RAW

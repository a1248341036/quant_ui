# -*- coding: utf-8 -*-
"""模块 03 · delivery_interface：因子构建接口 + 两阶段交付定义（DELIVERY_GATES 动态渲染）。"""

from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria


def _delivery_gates_markdown(spec: dict | None) -> str:
    """从 ResearchSpec 动态渲染两阶段交付门槛，保证提示词与真实门禁永不脱节。

    数值唯一真源在 delivery_criteria（由 research_spec 注入）；此处只做委托，
    不再重复维护门槛文本（历史硬编码 min_fmb_t_stat 等已移除门槛在此一并淘汰）。
    """
    return DeliveryCriteria.from_spec(spec).to_prompt_text()


NAME = "delivery_interface"
TITLE = "构建接口与两阶段交付定义"
ORDER = 30
REQUIRED = True
SEP_BEFORE = "\n\n"


def render(ctx) -> str:  # noqa: ANN001
    spec = ctx.research_spec
    return f"""# 因子构建接口

## 你的目标

**优化目标：（1）train 上达到可用的相关水平，且（2）鲁棒性达标。** 鲁棒性覆盖：`monthly_corr_robustness`、`factor_coverage`、因子分布（`factor_skewness`/`factor_kurtosis`）、**`summary.mls_fmb`**，以及少数 val 调用上与 train 不出现灾难性背离。

**【两阶段交付定义】** {_delivery_gates_markdown(spec)}

**【会话完成条件】** 挖掘会话的正式交付方式是调用 **`submit_factor`**。统计门槛（第一阶段）通过即写入候选池（`candidate_stored=true`），视为成功交付候选因子。正式库（`stored=true`）需同时通过统计精筛和 FactorReviewer 审查。**只要 train+val 评估有潜力的因子，就应该调用 `submit_factor` 提交候选池**，不要因为 reviewer 在 validation 阶段给出 revise/reject 就放弃提交——reviewer 意见仅供参考改进，候选池入库只看统计数据。仅完成 train/val 评估、口头总结或停在「建议入库」**不算交付**。查重失败时根据返回意见改写后再提交。

- 会话已配置 train/val 日期与 label 列；工具结果中不再重复这些配置。
- 每一轮：优先并行调用 4~8 次 **`evaluate_factor(profile_id="train_screen")`**，用不同 `multi_line_expr` 探多条假设；train 上通过 profile rules 后，以 **`validation`** 和必要时 **`size_neutral_validation`** profile 检验泛化与风险调整。profile 是冻结的，不得临时修改其 transform、metric 或规则。validation 后 `FactorReviewer` 会给出新颖性审查意见，**仅供参考改进方向，不阻断提交**；只要 train+val 统计达标就应调用 `submit_factor`。
- 默认 **`include_detail_tables`: false**；需要按月/分品种明细时再设为 **true**。

请遵循：
- **相关性 + 鲁棒性双目标**；筛选用 **`abs(summary.ic)`** 与 **`abs(summary.rank_ic)`**；**负 IC 是有效负向 alpha**，不是错误。
- **中间变量命名**：蛇形英文名（如 `ma_w_dev`），避免 `x`、`tmp`。
- 若 `ok` 为 false，修正 DSL 或列名。"""

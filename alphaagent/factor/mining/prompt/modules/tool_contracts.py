# -*- coding: utf-8 -*-
"""模块 09 · tool_contracts：工具调用表与返回字段约定。原文精确切片。

tool_calls 并行示例独立为 tool_examples 模块（挂在 behavior_rules 之后，
与旧装配顺序一致）。
"""

NAME = "tool_contracts"
TITLE = "工具调用契约"
ORDER = 90
REQUIRED = True
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    return """### 工具调用

| 工具 | 作用 |
|------|------|
| `evaluate_factor` | 按冻结 EvaluationProfile 评估；优先使用 `train_screen`、`validation`、`size_neutral_validation`，不得临时改变评估口径 |
| `eval_on_train_set` | 训练窗评估（**应占绝大多数调用**） |
| `eval_on_val_set` | 验证窗评估（**少用**）；须传 `expected_sign` |
| `submit_factor` | **两阶段交付**：先写候选池；满足精筛收益、截尾稳健性和独立性要求后才写正式 factorzoo |
| `screen_factors` | **Screener · regime 感知筛选**：对正式库因子做市场制度（ADX+均线）感知筛选，输出当前 regime 下适配因子子集 + 动态权重/方向。开关在研究规范 `delivery_policy.screener.enabled` |

共用参数（eval）：`multi_line_expr`（必填）、`factor_name`、`include_detail_tables`、`label_quantile_n`（默认 10，0 则不输出分位桶）。
`evaluate_factor` / `eval_on_train_set` 必须传 **`prediction`**（可证伪预测：`expected_shape` + `expected_strong_side` + `expected_sign`，可选 `falsifier`）。缺失不会立刻拦截，但结果会带 `prediction_warning` 记账（累计 3 次升级拦截）——每次都带上，别依赖宽限。

**`submit_factor` 参数**：`multi_line_expr`、`factor_name`（蛇形英文名）、`comment`（必填，描述因子经济含义与结构）。

**`submit_factor` 返回字段**：`stored`（正式库成功）、`candidate_stored`（宽松池成功）、`metrics`（含 `long_group_annual_excess_return`、`winsorized_ic`、`winsorized_abs_ic_decay`）、`delivery_check.stage_one`、`delivery_check.stage_two`、相似度、候选/正式 registry 路径与失败原因。

**`screen_factors` 参数**：`factor_names`（可选，要筛选的因子名列表，为空则用正式库全部）、`signal_date`（可选，信号日 YYYY-MM-DD，为空则用 val 段最后一天）。

**`screen_factors` 返回字段**：`result.regime`（当前市场制度）、`result.selected`（选中因子列表）、`result.weights`（归一化权重）、`result.directions`（方向：买高/买低）、`result.factor_ic`（各因子近期 IC）、`result.rejected`（被拒因子及原因）、`result.regime_dist`（regime 分布统计）。

**工具返回 JSON 字段：**

| 字段 | 含义 |
|------|------|
| `summary` | `ic`、`icir`、`rank_ic`、**`cs_pearson_autocorr`**、`n_days`、`n_instruments`、`factor_coverage`、`factor_skewness`、`factor_kurtosis`、**`decile_mean_label`**（固定 10 组，`decile` 1–10，`mean_label` 为组内前瞻 label 均值） |
| `monthly_corr_robustness` | `n_months`、`mean_monthly_ic`、**`share_months_ic_positive`**（月均 IC>0 的月份占比） |
| `label_quantile_buckets` | 与 `decile_mean_label` 同口径的可选分位桶（`label_quantile_n` 控制，默认 10） |
| `sign_check` | 仅 val 且传入 `expected_sign` |
| `prediction_check` | 自动预测对账：`verdict`（confirmed/partial/contradicted/unverifiable）+ `expected` vs `actual`（实际形态/强侧/spearman/D1/D10）+ `message`。**contradicted = 机制错误，换机制或放弃，不要调参重试** |
| `ablation_check` | 门控/条件类表达式且契约含 `base_expr` 时自动返回：`base_ic` vs `full_ic`、`verdict`（added_value/destroyed_value/flipped_signal/neutral）、`message` |
| `ablation_hint` | 门控类表达式但未传 `base_expr` 时的提醒——补上重跑才能确认门控增量 |
| `by_month` / `by_symbol` | 仅 `include_detail_tables=true` |
"""


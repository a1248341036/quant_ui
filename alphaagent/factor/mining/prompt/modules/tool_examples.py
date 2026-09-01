# -*- coding: utf-8 -*-
"""模块 09b · tool_examples：tool_calls 并行示例 JSON（含基本面变体）。原文精确切片。"""

import json


def _tool_call_examples_section(*, include_fundamentals: bool = True) -> str:
    examples = [
        {
            "name": "eval_on_train_set",
            "arguments": {
                "multi_line_expr": "ma20 = TS_MEAN($adj_close, 20)\nSUBTRACT($adj_close, ma20)",
                "factor_name": "ma20_dev",
            },
        },
        {
            "name": "eval_on_train_set",
            "arguments": {
                "multi_line_expr": "ma_w = TS_MEAN($adj_close@1w, 4)\nSUBTRACT($adj_close, ma_w)",
                "factor_name": "ma_w_dev",
            },
        },
    ]
    if include_fundamentals:
        examples.append(
            {
                "name": "eval_on_train_set",
                "arguments": {
                    "multi_line_expr": "roe_pure = CS_RESIDUALIZE(CS_ZSCORE(CS_WINSORIZE($funda_roe, 0.01, 0.99)), LOG($float_cap))\ngro_rank = RANK(CS_ZSCORE(CS_WINSORIZE($funda_netprofit_yoy, 0.01, 0.99)))\nADD(roe_pure, gro_rank)",
                    "factor_name": "funda_roe_growth_neutral",
                    "interaction": {
                        "interaction_type": "residual_signal",
                        "base_signal": "盈利能力质量",
                        "condition_signal": "市值暴露",
                        "economic_mechanism": "剥离市值暴露后保留不可由规模解释的盈利质量",
                        "expected_subgroup_pattern": {"purpose": "size-neutral quality"},
                        "ablation_required": True
                    }
                },
            }
        )
    examples.append(
        {
            "name": "eval_on_train_set",
            "arguments": {
                "multi_line_expr": "base = NEG(TS_PCTCHANGE($adj_close, 5))\nstate = RANK(DIVIDE(TS_MEAN($amount, 20), LOG($float_cap)))\nGATED_SIGNAL(base, state, 0.8, true, 0)",
                "factor_name": "reversal_high_liquidity_gate",
                "interaction": {
                    "interaction_type": "gated_signal",
                    "base_signal": "短期过度反应后的修复压力",
                    "condition_signal": "高流动性状态",
                    "economic_mechanism": "高流动性股票的过度反应更容易被套利资金修正",
                    "expected_subgroup_pattern": {"high_liquidity": "信号启用", "other": "中性"},
                    "ablation_required": True
                }
            },
        }
    )
    examples.append(
        {
            "name": "submit_factor",
            "arguments": {
                "multi_line_expr": "ma20 = TS_MEAN($adj_close, 20)\nSUBTRACT($adj_close, ma20)",
                "factor_name": "ma20_dev",
                "comment": "20日均价偏离：价格相对短期均线的回归/动量；负IC表示均值回归。",
            },
        }
    )
    submit_note = (
        "\n\n**交付示例**：train/val 均达标后，须调用 `submit_factor`（上表第 4 条）；"
        "查重失败则读 `similarity.top_neighbors[].expr` 改写后重试。"
    )
    body = json.dumps(examples, ensure_ascii=False, indent=2)
    dims = "动量、周线偏离、基本面残差、门控反转" if include_fundamentals else "动量、周线偏离、门控反转"
    note = (
        f"上表为同轮并行 `eval_on_train_set` 示例（{dims}）。"
        "建议每轮 3～5 条并行；仅当 train 有满意候选时，偶尔对少数 factor 做 val 抽检。"
        + submit_note
    )
    return (
        "---\n\n## ``tool_calls`` 示例（**并行 train + 最终 submit**）\n\n"
        + note
        + "\n\n```json\n"
        + body
        + "\n```\n"
    )


NAME = "tool_examples"
TITLE = "tool_calls 并行示例"
ORDER = 125
REQUIRED = True
SEP_BEFORE = "\n\n"


def render(ctx) -> str:  # noqa: ANN001
    funda_effective = ctx.include_fundamentals and (
        ctx.panel_columns is None
        or any(c.startswith("funda_") for c in ctx.panel_columns)
    )
    return _tool_call_examples_section(include_fundamentals=funda_effective)

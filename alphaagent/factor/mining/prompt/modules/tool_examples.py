# -*- coding: utf-8 -*-
"""模块 09b · tool_examples：tool_calls 并行示例 JSON（含基本面变体）。原文精确切片。

数据面聚焦生效时按 facets 裁剪：触及未选面的示例整体移除——示例是最容易被
照抄的上下文，聚焦 run 里出现价量示例就等于把 LLM 往越界表达式上引。
"""

import json

from alphaagent.factor.mining.memory.expressions import expr_facets


def _example_in_scope(example: dict, scope: set[str]) -> bool:
    if not scope:
        return True
    expr = str((example.get("arguments") or {}).get("multi_line_expr") or "")
    facets = expr_facets(expr)
    return bool(facets) and facets <= scope


# 聚焦 run 的兜底示例：默认示例库整体越界被裁空后，聚焦 run 没有任何合规骨架可抄，
# LLM 会按先验拼经典价量结构（2026-09-10 run 51e02d47a3f3 turn0 纯价量因子被拦的
# 根因之一）。按勾选面的代表列合成示例——只用聚焦面列，绝不引用仅作输入的面。
_FACE_REPR_COLUMN: dict[str, str] = {
    "价量面": "adj_close",
    "量能面": "amount",
    "业绩面": "pred_surprise",
    "基本面": "funda_roe",
    "股东面": "holder_count_chg_pct",
    "资金面": "ff_super_net",
    "两融面": "mgn_buy",
    "事件面": "dt_net_buy_90d",
    "机构面": "inst_ratio",
    "股东集中面": "th_top10_pct",
    "披露面": "ds_days_since_actual",
    "分红面": "div_cash_div",
}
_FACE_ASCII: dict[str, str] = {
    "价量面": "pv",
    "量能面": "vol",
    "业绩面": "earn",
    "基本面": "funda",
    "股东面": "holder",
    "资金面": "ff",
    "两融面": "margin",
    "事件面": "event",
    "机构面": "inst",
    "股东集中面": "topholder",
    "披露面": "disc",
    "分红面": "div",
}


def _scoped_fallback_examples(focus: list[str], panel_columns) -> list[dict]:
    """按聚焦面代表列合成示例（面无代表列或列不在 panel 时跳过该面）。"""
    available = None if panel_columns is None else {str(c) for c in panel_columns}
    faces = [
        f
        for f in (focus or [])
        if _FACE_REPR_COLUMN.get(f)
        and (available is None or _FACE_REPR_COLUMN[f] in available)
    ]
    if not faces:
        return []
    if len(faces) == 1:
        col = "$" + _FACE_REPR_COLUMN[faces[0]]
        return [
            {
                "name": "eval_on_train_set",
                "arguments": {
                    "multi_line_expr": (
                        f"chg = TS_DELTA({col}, 20)\n"
                        f"CS_ZSCORE(CS_WINSORIZE(chg, 0.01, 0.99))"
                    ),
                    "factor_name": f"{_FACE_ASCII.get(faces[0], 'face')}_delta_z",
                },
            }
        ]
    face_a, face_b = faces[0], faces[1]
    col_a = "$" + _FACE_REPR_COLUMN[face_a]
    col_b = "$" + _FACE_REPR_COLUMN[face_b]
    name_a = _FACE_ASCII.get(face_a, "faceA")
    name_b = _FACE_ASCII.get(face_b, "faceB")
    return [
        {
            "name": "eval_on_train_set",
            "arguments": {
                "multi_line_expr": (
                    f"base = CS_ZSCORE(CS_WINSORIZE({col_a}, 0.01, 0.99))\n"
                    f"state = RANK(TS_MEAN({col_b}, 20))\n"
                    f"GATED_SIGNAL(base, state, 0.8, true, 0)"
                ),
                "factor_name": f"{name_a}_x_{name_b}_gate",
                "interaction": {
                    "interaction_type": "gated_signal",
                    "base_signal": f"{face_a}核心信号",
                    "condition_signal": f"{face_b}活跃度状态",
                    "economic_mechanism": (
                        f"{face_a}信号在{face_b}活跃放大的状态下更可靠，"
                        "门控集中高置信组"
                    ),
                    "expected_subgroup_pattern": {"high_state": "信号启用", "other": "中性"},
                    "ablation_required": True,
                },
            },
        }
    ]


def _tool_call_examples_section(
    *, include_fundamentals: bool = True, focus_facets=None, panel_columns=None
) -> str:
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
    scope = [str(f) for f in (focus_facets or ()) if f]
    synthesized = False
    if scope:
        examples = [e for e in examples if _example_in_scope(e, set(scope))]
        if not examples:
            examples = _scoped_fallback_examples(scope, panel_columns)
            synthesized = bool(examples)
        if not examples:
            return ""
    submit_note = (
        "\n\n**交付示例**：train/val 均达标后，须调用 `submit_factor`（上表第 4 条）；"
        "查重失败则读 `similarity.top_neighbors[].expr` 改写后重试。"
    )
    body = json.dumps(examples, ensure_ascii=False, indent=2)
    if scope:
        dims = (
            "本 run 聚焦数据面合成示例（默认示例越界已裁剪，按勾选面代表列生成的骨架，仅供参考）"
            if synthesized
            else "本 run 聚焦数据面示例（越界示例已按数据面裁剪）"
        )
    else:
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
# required=False：数据面聚焦时若所有示例都越界，本模块会整体不注入
# （调用格式由常驻 tool_contracts 覆盖），此时不应记为"核心板块缺失"。
REQUIRED = False
SEP_BEFORE = "\n\n"
# 2026-09-06 曾把 explore 从 PHASES 移除（2026-09-12 回退）：实测 run
# 42254d3990f0 中 explore 阶段因无调用示例 + 算子签名被藏，工具失败率
# 13.1%→18.1%（模型靠报错自愈盲试，console.log 满屏 "Signature unknown.
# errors are cheap"）；tool_examples 仅 ~450-630 token/轮（含基本面），
# 以极小成本换调用格式稳定。multi_period/delivery_submission 仍保持裁剪。
PHASES = frozenset({"explore", "deepen", "deliver", "full"})


def render(ctx) -> str:  # noqa: ANN001
    funda_effective = ctx.include_fundamentals and (
        ctx.panel_columns is None
        or any(c.startswith("funda_") for c in ctx.panel_columns)
    )
    return _tool_call_examples_section(
        include_fundamentals=funda_effective,
        focus_facets=getattr(ctx, "focus_facets", ()),
        panel_columns=ctx.panel_columns,
    )

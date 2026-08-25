# 回归：契约缺 interaction_type 时从表达式触发算子自动推断 / 忽略多余契约
from alphaagent.factor.mining.interactions import lint_expression_interaction

MECH = "放量急跌是流动性冲击导致的过度反应，套利资金易进入并快速修正定价"


def test_interaction_type_inference_and_tolerance():
    # 1) JSON 字符串 + 缺 interaction_type + 表达式含 GATED_SIGNAL → 推断为 gated_signal
    spec_str = (
        '{"ablation_required": true, "base_signal": "5日收盘急跌幅度", '
        '"condition_signal": "成交额放大倍数", "economic_mechanism": "' + MECH + '"}'
    )
    expr = (
        "rev5 = NEG(TS_PCTCHANGE($adj_close, 5))\n"
        "sig = GATED_SIGNAL(rev5, amt_surp, 0.8, true, 0)"
    )
    spec, warning, error = lint_expression_interaction(expr, spec_str, policy={})
    assert error is None and spec is not None
    assert spec["interaction_type"] == "gated_signal"

    # 2) dict 形式同样推断（PIECEWISE_STATE → piecewise_state）
    spec2 = {
        "base_signal": "归一化超跌",
        "condition_signal": "ER 效率比",
        "economic_mechanism": "低效率比震荡市中急跌多为噪声冲击，均值回归主导",
    }
    expr2 = "raw = PIECEWISE_STATE(rev_norm, er_state, 0.2, 0.8, 1.0, -1.0, 0.0)"
    spec2n, _, err2 = lint_expression_interaction(expr2, spec2, policy={})
    assert err2 is None and spec2n["interaction_type"] == "piecewise_state"

    # 3) TS_RANKCORR → rolling_relation
    spec3 = {
        "base_signal": "量价秩相关",
        "condition_signal": "-",
        "economic_mechanism": "量价滚动秩相关捕捉趋势的量能确认度与背离结构",
    }
    expr3 = "NEG(CS_WINSORIZE(TS_RANKCORR($volume, $adj_close, 20), 0.01, 0.99))"
    spec3n, _, err3 = lint_expression_interaction(expr3, spec3, policy={})
    assert err3 is None and spec3n["interaction_type"] == "rolling_relation"

    # 4) MULTIPLY 默认禁用：即使推断出类型也应被 allowed 拦截
    spec4 = '{"base_signal": "a", "condition_signal": "b", "economic_mechanism": "' + MECH + '"}'
    err4 = lint_expression_interaction(
        "MULTIPLY(x, y)", spec4,
        policy={"allowed_interaction_types": ["gated_signal"], "block_undeclared_multiply": True},
    )[2]
    assert err4 is not None, "multiplication 应被默认禁用拦截"

    # 5) 表达式无触发算子 + 契约缺类型 → 多余契约被忽略，不报错
    spec5 = '{"base_signal": "a", "condition_signal": "b", "economic_mechanism": "' + MECH + '"}'
    clean_expr = (
        "rev_rank = RANK(NEG(TS_PCTCHANGE($adj_close, 5)))\n"
        "amt_rank = RANK(LOG(DIVIDE($volume, TS_MEAN($volume, 20))))\n"
        "combo = ADD(rev_rank, amt_rank)"
    )
    s5, w5, e5 = lint_expression_interaction(clean_expr, spec5, policy={
        "allowed_interaction_types": ["gated_signal"],
    })
    assert e5 is None and s5 is None and w5 is None

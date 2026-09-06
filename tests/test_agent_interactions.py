from alphaagent.factor.mining.interactions import lint_expression_interaction


def _spec(kind: str = "gated_signal") -> dict:
    return {
        "interaction_type": kind,
        "base_signal": "short-term reversal",
        "condition_signal": "liquidity state",
        "economic_mechanism": "overreaction is corrected faster when arbitrage capital can trade",
    }


def test_undeclared_multiply_is_blocked() -> None:
    result = lint_expression_interaction(
        "MULTIPLY(CS_ZSCORE(a), CS_ZSCORE(b))",
        None,
        policy={"block_undeclared_multiply": True},
    )
    assert result[2] is not None
    assert result[2]["error_type"] == "UndeclaredInteractionError"


def test_typed_operator_without_contract_is_blocked() -> None:
    result = lint_expression_interaction(
        "GATED_SIGNAL(a, b, 0.8, true, 0)",
        None,
        policy={"require_contract_for_typed_interactions": True},
    )
    assert result[2] is not None
    assert result[2]["error_type"] == "UndeclaredInteractionError"


def test_gated_contract_passes_and_requires_ablation() -> None:
    spec, warning, error = lint_expression_interaction(
        "GATED_SIGNAL(a, b, 0.8, true, 0)",
        _spec(),
        policy={"block_undeclared_multiply": True},
    )
    assert error is None
    assert spec is not None
    assert spec["ablation_required"] is True
    assert warning is None


def test_multiply_with_wrong_contract_is_blocked() -> None:
    _, _, error = lint_expression_interaction(
        "MULTIPLY(a, b)",
        _spec("gated_signal"),
        policy={},
    )
    assert error is not None
    assert error["error_type"] == "InteractionMismatchError"


def test_missing_contract_autofilled_with_warning() -> None:
    """容错 4：表达式带结构算子但没传契约 → 自动补全 + warning（2026-09-06）。

    实测单面 run 一轮 13 评仅 1 条带契约，其余全部被拦——格式瑕疵不应
    烧掉整轮评估；机制纪律由 Reviewer 把守，warning 教 LLM 显式传。
    """
    spec, warning, error = lint_expression_interaction(
        "conc = RANK(NEG($holder_count_chg_pct)) ; mom60 = RANK(TS_PCTCHANGE($adj_close, 60)) ; "
        "CS_RESIDUALIZE(conc, mom60)",
        None,
        policy={"require_contract_for_typed_interactions": True},
    )
    assert error is None
    assert spec is not None
    assert spec["interaction_type"] == "residual_signal"
    # 信号从表达式列自动识别
    assert "$holder_count_chg_pct" in spec["base_signal"]
    # 机制为占位标记（明确标注非 LLM 声明）
    assert "自动补全" in spec["economic_mechanism"]
    assert warning is not None and "显式传 interaction" in warning


def test_autofill_type_is_last_structural_op() -> None:
    """链式表达式的补全类型按末位结构算子（末位算子定义输出结构）。"""
    spec, warning, error = lint_expression_interaction(
        "div = DIVERGENCE_RANK(RANK($a), RANK($b)) ; out = GATED_SIGNAL(div, RANK($c), 0.8, true, 0)",
        None,
        policy={},
    )
    assert error is None and spec is not None
    assert spec["interaction_type"] == "gated_signal"
    assert warning is not None


def test_autofill_disabled_blocks() -> None:
    """policy.auto_fill_missing_contract=False → 保持硬拦截。"""
    _, _, error = lint_expression_interaction(
        "CS_RESIDUALIZE(RANK($holder_count), LOG($float_cap))",
        None,
        policy={"require_contract_for_typed_interactions": True,
                "auto_fill_missing_contract": False},
    )
    assert error is not None
    assert error["error_type"] == "UndeclaredInteractionError"


def test_autofill_without_columns_falls_back_to_block() -> None:
    """表达式无 $列引用（合成测试用例）时无法补全信号 → 仍拦截。"""
    _, _, error = lint_expression_interaction(
        "GATED_SIGNAL(a, b, 0.8, true, 0)",
        None,
        policy={"require_contract_for_typed_interactions": True},
    )
    assert error is not None
    assert error["error_type"] == "UndeclaredInteractionError"


def test_multiply_still_blocked_despite_autofill() -> None:
    """MULTIPLY 无契约不走自动补全（乘法默认禁用，语义错误而非格式瑕疵）。"""
    _, _, error = lint_expression_interaction(
        "MULTIPLY(RANK($a), RANK($b))",
        None,
        policy={"block_undeclared_multiply": True},
    )
    assert error is not None
    assert error["error_type"] == "UndeclaredInteractionError"

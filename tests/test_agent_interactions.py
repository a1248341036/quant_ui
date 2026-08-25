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

import numpy as np

from core.selection import PortfolioBuilder


def test_portfolio_builder_fixed_and_dynamic_selection():
    builder = PortfolioBuilder(["a", "b", "c", "d"])
    candidates = np.array([0, 1, 2, 3])
    scores = np.array([0.2, 0.9, 0.3, 0.1])
    assert builder.rank_select(candidates, scores, False, 2) == [1, 2]
    assert builder.rank_select(
        candidates, scores, False, 2, selection_mode="top_pct", selection_pct=0.5,
    ) == [1, 2]


def test_portfolio_builder_industry_cap():
    builder = PortfolioBuilder(
        ["a", "b", "c"], {"a": "x", "b": "x", "c": "y"}, 1,
    )
    selected = builder.rank_select(
        np.array([0, 1, 2]), np.array([0.9, 0.8, 0.7]), False, 2,
    )
    assert selected == [0, 2]
    assert builder.equal_weights(selected) == {0: 0.5, 2: 0.5}

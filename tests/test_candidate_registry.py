from pathlib import Path

from alphaagent.factor.mining.registry_io import (
    load_mining_registry,
    set_candidate_promotion,
    set_candidate_review,
    write_candidate_registry,
)
from alphaagent.factor.types import IngestPolicy


def test_candidate_registry_is_value_free(tmp_path: Path) -> None:
    root = tmp_path / "candidate_1d"
    policy = IngestPolicy(train_start="2020-01-01", val_end="2024-12-31")

    _, dsl = write_candidate_registry(
        root / "mining_candidate_registry.json",
        factor_id="demo",
        name="demo",
        expr="CS_ZSCORE($close)",
        expr_dir=root / "expressions",
        repo_root=tmp_path,
        policy=policy,
        metrics={"ic": 0.01, "coverage": 0.99},
        similarity={"max_abs_corr": 0.1},
        data_fingerprint={"index_hash": "abc", "n_rows": 10},
    )

    entry = load_mining_registry(root / "mining_candidate_registry.json")["demo"]
    assert Path(dsl).read_text(encoding="utf-8").strip() == "CS_ZSCORE($close)"
    assert entry["review_status"] == "pending_review"
    assert entry["data_fingerprint"]["index_hash"] == "abc"
    assert not list(root.rglob("*.memmap"))

    set_candidate_review(
        root / "mining_candidate_registry.json",
        factor_id="demo",
        review={"verdict": "approve", "novelty": "medium"},
        promotion_status="stage_two_failed",
    )
    set_candidate_promotion(
        root / "mining_candidate_registry.json",
        factor_id="demo",
        promotion_status="promoted",
    )
    updated = load_mining_registry(root / "mining_candidate_registry.json")["demo"]
    assert updated["review_status"] == "approve"
    assert updated["promotion_status"] == "promoted"

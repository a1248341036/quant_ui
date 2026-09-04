"""调仓频率/档位溯源测试：registry_io 落字段 + label_col 推导兜底。"""

from __future__ import annotations

from pathlib import Path

from alphaagent.factor.mining.registry_io import (
    derive_freq_from_label_col,
    load_mining_registry,
    upsert_mining_registry,
    write_candidate_registry,
)
from alphaagent.factor.types import IngestPolicy


def test_derive_freq_from_label_col() -> None:
    assert derive_freq_from_label_col("label_1d_open_to_open") == ("technical", "weekly")
    assert derive_freq_from_label_col("label_20d_close_to_close") == ("fundamental", "monthly")
    assert derive_freq_from_label_col("label_10d_close_to_close") == ("fundamental", "monthly")
    assert derive_freq_from_label_col("") == (None, None)
    assert derive_freq_from_label_col(None) == (None, None)
    assert derive_freq_from_label_col("not_a_label") == (None, None)


def test_candidate_registry_records_freq_meta(tmp_path: Path) -> None:
    root = tmp_path / "candidate_main"
    policy = IngestPolicy(train_start="2020-01-01", val_end="2024-12-31")

    write_candidate_registry(
        root / "mining_candidate_registry.json",
        factor_id="demo",
        name="demo",
        expr="CS_ZSCORE($close)",
        expr_dir=root / "expressions",
        repo_root=tmp_path,
        policy=policy,
        metrics={"ic": 0.01, "coverage": 0.99},
        similarity={"max_abs_corr": 0.1},
        rebalance_freq="weekly",
        research_mode="technical",
    )
    entry = load_mining_registry(root / "mining_candidate_registry.json")["demo"]
    assert entry["rebalance_freq"] == "weekly"
    assert entry["research_mode"] == "technical"

    # 重写时不传频率参数：沿用旧值（重提交场景不丢溯源）
    write_candidate_registry(
        root / "mining_candidate_registry.json",
        factor_id="demo",
        name="demo",
        expr="CS_ZSCORE($close)",
        expr_dir=root / "expressions",
        repo_root=tmp_path,
        policy=policy,
        metrics={"ic": 0.02, "coverage": 0.99},
        similarity={"max_abs_corr": 0.1},
    )
    entry = load_mining_registry(root / "mining_candidate_registry.json")["demo"]
    assert entry["rebalance_freq"] == "weekly"
    assert entry["research_mode"] == "technical"


def test_upsert_registry_records_freq_meta(tmp_path: Path) -> None:
    root = tmp_path / "production_main"
    policy = IngestPolicy(train_start="2020-01-01", val_end="2024-12-31")

    upsert_mining_registry(
        root / "mining_delivered_registry.json",
        factor_id="demo",
        name="demo",
        expr="CS_ZSCORE($close)",
        expr_dir=root / "expressions",
        repo_root=tmp_path,
        policy=policy,
        metrics={"ic": 0.03},
        similarity={"max_abs_corr": 0.1},
        ingest_status="production",
        rebalance_freq="monthly",
        research_mode="fundamental",
    )
    entry = load_mining_registry(root / "mining_delivered_registry.json")["demo"]
    assert entry["rebalance_freq"] == "monthly"
    assert entry["research_mode"] == "fundamental"
    # ingest_config 里的 label_col 是推导兜底的依据
    assert "label_col" in entry["ingest_config"]

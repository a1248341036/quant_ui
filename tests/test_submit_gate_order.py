from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor.mining import submit as submit_module
from alphaagent.factor.mining.submit import FactorSubmitService
from alphaagent.factor.mining.agentscope_tools import _sample_orthogonality_panel


@pytest.fixture
def submit_env(monkeypatch, tmp_path):
    calls = {"candidate_registry": 0, "orthogonality": 0}
    metrics = {
        "ic": 0.05,
        "icir": 0.6,
        "coverage": 0.99,
        "mls_fmb": {"nw_t_ls": 3.0},
        "long_group_annual_excess_return": 0.05,
        "winsorized_abs_ic_decay": 0.01,
    }
    assessment = SimpleNamespace(metrics=metrics, similarity={"max_abs_corr": 0.1})
    production = SimpleNamespace(stored=True, metrics=metrics, similarity={"max_abs_corr": 0.1}, skipped_reason=None)

    class FakeManifest:
        n_rows = 2
        index_hash = "hash"
        panel_path = "cne://"

    class FakeZoo:
        manifest = FakeManifest()

    class FakeService:
        sessions = SimpleNamespace(get=lambda session_id: SimpleNamespace(ctx=SimpleNamespace(
            train_start="2018-01-01",
            val_end="2025-12-31",
            panel_path="cne://",
            label_col="label_1d_open_to_open",
        )))

    monkeypatch.setattr(submit_module, "FactorZoo", SimpleNamespace(open=lambda root: FakeZoo()))
    monkeypatch.setattr(submit_module, "load_panel_for_zoo", lambda zoo, panel_path: list(range(2)))
    monkeypatch.setattr(
        submit_module,
        "prepare_stored_values",
        lambda *args, **kwargs: (np.zeros(2), "expr", [], {}),
    )
    monkeypatch.setattr(
        submit_module,
        "ingest_factor",
        lambda *args, **kwargs: assessment if kwargs.get("dry_run") else production,
    )
    monkeypatch.setattr(
        submit_module,
        "write_candidate_registry",
        lambda *args, **kwargs: (
            calls.__setitem__("candidate_registry", calls["candidate_registry"] + 1),
            str(tmp_path / "reg"),
            str(tmp_path / "reg.dsl"),
        )[1:],
    )
    monkeypatch.setattr(submit_module, "set_candidate_review", lambda *args, **kwargs: None)
    monkeypatch.setattr(submit_module, "set_candidate_promotion", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        submit_module,
        "upsert_mining_registry",
        lambda *args, **kwargs: (str(tmp_path / "prod-reg"), str(tmp_path / "prod.dsl")),
    )

    service = FactorSubmitService(
        FakeService(),
        factorlib_path=tmp_path / "production_technical",
        registry_path=tmp_path / "registry.json",
        expr_dir=tmp_path / "expressions",
        repo_root=tmp_path,
    )
    return service, calls


def test_review_rejection_runs_before_orthogonality_and_candidate_storage(submit_env):
    service, calls = submit_env
    result = service.submit(
        "session",
        multi_line_expr="CLOSE",
        factor_name="factor_a",
        comment="economic logic",
        review_hook=lambda candidate: {"verdict": "reject"},
        orthogonality_hook=lambda: calls.__setitem__("orthogonality", calls["orthogonality"] + 1) or {"passed": True},
    )
    assert result["error_type"] == "FactorReviewRejected"
    assert calls["orthogonality"] == 0
    assert calls["candidate_registry"] == 0


def test_review_revise_stores_candidate_but_blocks_production(submit_env):
    """统计达标即入候选池；Reviewer revise 只阻断转正式，不再吞掉候选。"""
    service, calls = submit_env
    result = service.submit(
        "session",
        multi_line_expr="CLOSE",
        factor_name="factor_c",
        comment="economic logic",
        review_hook=lambda candidate: {"verdict": "revise",
                                       "required_changes": ["补消融"]},
        orthogonality_hook=lambda: (_ for _ in ()).throw(AssertionError("不应进入正交检查")),
    )
    assert result["candidate_stored"] is True
    assert result["ok"] is False
    assert result["promotion_status"] == "review_blocked"
    assert "revise" in result["skipped_reason"]
    assert calls["candidate_registry"] == 1
    assert calls["orthogonality"] == 0


def test_offline_orthogonality_runs_after_approval_candidate_already_stored(submit_env):
    """approve 后先入候选池再做正交/正式入库；正交失败时候选保留。"""
    service, calls = submit_env
    result = service.submit(
        "session",
        multi_line_expr="CLOSE",
        factor_name="factor_b",
        comment="economic logic",
        review_hook=lambda candidate: {"verdict": "approve"},
        orthogonality_hook=lambda: calls.__setitem__("orthogonality", calls["orthogonality"] + 1) or {
            "passed": False,
            "max_abs_corr": 0.9,
        },
    )
    assert result["error_type"] == "OfflineOrthogonalityError"
    assert calls["orthogonality"] == 1
    assert calls["candidate_registry"] == 1


def test_orthogonality_sampler_preserves_contiguous_lookback():
    dates = pd.bdate_range("2024-01-02", periods=120)
    index = pd.MultiIndex.from_product(
        [dates, pd.Index(["A"], name="instrument")],
        names=["datetime", "instrument"],
    )
    panel = pd.DataFrame({"close": range(len(index))}, index=index)

    sampled = _sample_orthogonality_panel(panel)
    sampled_dates = sampled.index.get_level_values("datetime").unique().sort_values()

    assert len(sampled_dates) <= 5 * 20
    assert len(sampled_dates) > 20

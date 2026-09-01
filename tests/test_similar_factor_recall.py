# -*- coding: utf-8 -*-
"""相似因子召回：_orthogonality_check top-K 清单 + 评估结果挂 similar_existing。"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.factor.mining.agent.agentscope_tools import (  # noqa: E402
    _orthogonality_check,
    _sample_orthogonality_panel,
)


def _make_panel(n_days: int = 40, n_inst: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    insts = [f"I{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    return pd.DataFrame({"close": close, "ret": rng.normal(0, 0.02, len(idx))}, index=idx)


def _make_tools(panel: pd.DataFrame) -> SimpleNamespace:
    session = SimpleNamespace(panel=panel)
    return SimpleNamespace(service=SimpleNamespace(sessions={"s1": session}), session_id="s1")


def _zoo_with_factor(root: Path, panel: pd.DataFrame, factor_id: str, values: np.ndarray):
    from alphaagent.factor.zoo import FactorZoo
    from alphaagent.factor.zoo.index import init_library

    init_library(root, panel=panel, n_sample_rows=1000, max_factors=8)
    zoo = FactorZoo.open(root)
    zoo.append_factor(factor_id=factor_id, name=factor_id, expr="RANK($close)", values=values)
    return zoo


def test_ortho_reports_top_similar_factors(tmp_path, monkeypatch):
    """达标因子评估后召回库内最相似因子：top-1 是高相关的既有因子。"""
    panel = _make_panel()
    tools = _make_tools(panel)

    # 库内因子 = panel.close 的秩（与新因子"RANK($close)"完全相关 → 拦截 + top1）
    close_vals = panel["close"].to_numpy(dtype=np.float64)
    fp = tmp_path / "cand"
    _zoo_with_factor(fp, panel, "rank_close_v1", close_vals)
    # registry 同样放一个高相关 DSL
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json_dumps({"reg_high": {"expr": "RANK($close)"}}), encoding="utf-8")

    monkeypatch.setattr(
        "alphaagent.core.paths.FACTORZOO_DIR", fp, raising=False)
    import core.factor_categories as fc
    monkeypatch.setattr(fc, "candidate_dir", lambda mode: fp, raising=False)
    monkeypatch.setattr(fc, "candidate_registry_path",
                        lambda mode: reg_path, raising=False)

    res = _orthogonality_check(tools, "RANK($close)")
    assert res["compared_factors"] >= 1
    assert res["max_abs_corr"] > 0.9
    assert res["passed"] is False
    assert res["similar_factors"], "应有 top-K 相似清单"
    assert res["similar_factors"][0]["corr"] > 0.9


def test_ortho_low_corr_passes(tmp_path, monkeypatch):
    panel = _make_panel(seed=11)
    tools = _make_tools(panel)
    fp = tmp_path / "cand"
    # 库内因子与 $close 无关的噪声 → 低相关通过
    rng = np.random.default_rng(3)
    _zoo_with_factor(fp, panel, "noise_v1", rng.normal(0, 1, len(panel)))
    monkeypatch.setattr("alphaagent.core.paths.FACTORZOO_DIR", fp, raising=False)
    import core.factor_categories as fc
    monkeypatch.setattr(fc, "candidate_dir", lambda mode: fp, raising=False)
    monkeypatch.setattr(fc, "candidate_registry_path",
                        lambda mode: tmp_path / "missing.json", raising=False)

    res = _orthogonality_check(tools, "RANK($close)")
    assert res["passed"] is True
    assert res["similar_factors"] and res["similar_factors"][0]["corr"] < 0.7


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)

# -*- coding: utf-8 -*-
"""B-02/B-03 修复测试：cs_ic_summary 三口径并列 + 口径标注。

覆盖 docs/specs/alphaagent_known_bugs_spec.md B-02/B-03：
- 主口径 icir（去重叠）保持原语义不变；
- 新增 icir_overlap（名义重叠 ICIR，诊断）、ic_t_nw（Newey-West t）、
  ic_n_eff（有效样本数）、icir_caliber（口径标注字符串）；
- hold=1 时 icir_overlap == icir 且 caliber=daily（零行为变化）；
- hold>1 时三口径与手算逐位一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaagent.factor.metrics import cs_ic_summary, newey_west_mean_tstat


def _known_series(n: int = 50) -> pd.Series:
    # 确定性序列：均值正、std 非 0（手算可复核）
    vals = np.sin(np.arange(n) * 0.7) * 0.03 + 0.01
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(vals, index=idx, name="ic")


def test_hold1_caliber_matches_daily_and_overlap_equals_icir():
    s = _known_series()
    out = cs_ic_summary(s, holding_days=1)
    assert out["icir_caliber"] == "daily"
    assert np.isclose(out["icir_overlap"], out["icir"])
    assert out["n_days"] == len(s)


def test_hold5_three_calibers_match_hand_computation():
    s = _known_series(50)
    hold = 5
    out = cs_ic_summary(s, holding_days=hold)

    # 主口径：去重叠（每 hold 日取一点）
    de = s.iloc[::hold]
    assert out["n_days"] == len(de)
    assert np.isclose(out["ic"], float(de.mean()))
    assert np.isclose(out["icir"], float(de.mean() / de.std(ddof=1)))
    assert out["icir_caliber"] == "deoverlap_by_holding_days"

    # 名义重叠口径：全序列手算
    assert np.isclose(out["icir_overlap"], float(s.mean() / s.std(ddof=1)))

    # Newey-West t（lags=hold-1）与有效样本数
    nw = newey_west_mean_tstat(s.to_numpy(dtype=float), lags=hold - 1)
    assert np.isclose(out["ic_t_nw"], float(nw["t_nw"]))
    se_nominal = float(s.std(ddof=1) / np.sqrt(len(s)))
    assert np.isclose(out["ic_n_eff"], (se_nominal / float(nw["se_nw"])) ** 2)


def test_empty_series_returns_all_keys():
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    out = cs_ic_summary(empty, holding_days=20)
    for key in ("ic", "icir", "rank_ic", "n_days", "icir_overlap", "ic_t_nw", "ic_n_eff", "icir_caliber"):
        assert key in out
    assert out["n_days"] == 0
    assert out["icir_caliber"] == "deoverlap_by_holding_days"
    # NaN 键可 JSON 序列化判断：均为 float
    for key in ("icir", "icir_overlap", "ic_t_nw", "ic_n_eff"):
        assert isinstance(out[key], float)

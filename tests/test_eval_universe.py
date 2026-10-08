# -*- coding: utf-8 -*-
"""股票池（universe）评估维度回归测试（2026-10-08）。

背景：实测同一批研报因子在"全市场"与"市值 Top300"下表现可以反号（PE 因子 IC
+0.0092 → −0.0132），故把股票池做成**评估口径**（会话级参数），而不是因子属性：
非成分股票整行剔除，IC/覆盖率/换手/回测全在池内计算，结果按池记账。
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor.mining.eval.context import StockEvalContext
from alphaagent.factor.mining.eval.schemas import SessionCreateRequest
from alphaagent.factor.mining.eval.universe import (
    apply_universe,
    available_universes,
    describe_universe,
    parse_universe,
)


def _panel() -> pd.DataFrame:
    """2 个交易日 × 5 只股票；tot_cap 每日固定 500/400/300/200/100（降序对应 s1..s5）。"""
    dates = [dt.date(2024, 1, 2), dt.date(2024, 1, 3)]
    syms = ["s1", "s2", "s3", "s4", "s5"]
    idx = pd.MultiIndex.from_product([dates, syms], names=["datetime", "instrument"])
    caps = [500.0, 400.0, 300.0, 200.0, 100.0] * len(dates)
    return pd.DataFrame(
        {
            "tot_cap": caps,
            "idx_szcomp": [0, 0, 1, 1, 0] * len(dates),
            "ret": np.linspace(-0.01, 0.01, len(idx)),
        },
        index=idx,
    )


def test_parse_universe_valid_and_invalid():
    assert parse_universe("all")["kind"] == "all"
    assert parse_universe("TOP300CAP")["n"] == 300            # 大小写不敏感
    assert parse_universe("top1000cap")["n"] == 1000
    assert parse_universe("mid301_800cap") == {"kind": "cap_mid", "name": "mid301_800cap",
                                              "a": 301, "b": 800}
    assert parse_universe("szcomp")["column"] == "idx_szcomp"
    assert parse_universe("chinext")["column"] == "idx_chinext"

    assert parse_universe("")["kind"] == "all"                 # 空值回落全市场（默认）

    for bad in ("top0cap", "mid500_500cap", "hs300", "top300"):
        with pytest.raises(ValueError) as ei:
            parse_universe(bad)
        # 报错必须点名"股票池"并说明原因，便于调用方自助纠正
        assert "股票池" in str(ei.value) or "可用" in str(ei.value)


def test_describe_universe_is_readable():
    assert "全市场" in describe_universe("all")
    assert "前 300" in describe_universe("top300cap")
    assert "301" in describe_universe("mid301_800cap")
    assert "szcomp" not in available_universes() or "szcomp" in available_universes()


def test_apply_universe_all_is_noop():
    panel = _panel()
    out, meta = apply_universe(panel, "all")
    assert out is panel
    assert meta["rows_before"] == meta["rows_after"] == len(panel)
    assert meta["dropped_rows"] == 0


def test_apply_universe_top_n_cap_keeps_ranked_names():
    panel = _panel()
    out, meta = apply_universe(panel, "top2cap")
    assert set(out.index.get_level_values("instrument")) == {"s1", "s2"}
    assert len(out) == 4                     # 2 天 × 2 只
    assert meta["rows_after"] == 4 and meta["dropped_rows"] == len(panel) - 4
    assert meta["stocks_per_day"] == pytest.approx(2.0)
    # 池内仍保留全部列（下游指标/回测口径不变）
    assert {"tot_cap", "idx_szcomp", "ret"} <= set(out.columns)


def test_apply_universe_mid_range_closed_interval():
    panel = _panel()
    out, _ = apply_universe(panel, "mid2_4cap")   # 闭区间：排名第 2~4 名 → s2、s3、s4
    assert set(out.index.get_level_values("instrument")) == {"s2", "s3", "s4"}
    assert len(out) == 6


def test_apply_universe_flag_pool_and_missing_column():
    panel = _panel()
    out, meta = apply_universe(panel, "szcomp")
    assert set(out.index.get_level_values("instrument")) == {"s3", "s4"}
    assert meta["universe_desc"].startswith("指数成分")

    # 面板缺指数列时必须显式报错（而不是静默当空池）
    with pytest.raises(ValueError) as ei:
        apply_universe(panel.drop(columns=["idx_szcomp"]), "szcomp")
    assert "idx_szcomp" in str(ei.value)


def test_apply_universe_empty_result_raises():
    panel = _panel()
    panel = panel.assign(idx_chinext=0.0)
    with pytest.raises(ValueError) as ei:
        apply_universe(panel, "chinext")
    assert "无数据" in str(ei.value)


def test_universe_defaults_are_all():
    ctx = StockEvalContext(panel_path="cne://panel")            # type: ignore[arg-type]
    assert ctx.universe == "all"
    assert SessionCreateRequest(panel_path="cne://panel").universe == "all"


def test_load_panel_accepts_cache_column_form(tmp_path):
    """面板缓存以 reset_index() 落盘（datetime/instrument 是列）时也要能加载。

    2026-10-08 实测：缓存文件直接当 ``panel_path`` 会在 slice_panel 处
    ``KeyError: 'Requested level (datetime)'``；本次补上按列重建索引。
    """
    from alphaagent.data.panel import load_panel

    df = pd.DataFrame({
        "datetime": [dt.date(2024, 1, 2)] * 2 + [dt.date(2024, 1, 3)] * 2,
        "instrument": ["s1", "s2"] * 2,
        "tot_cap": [500.0, 400.0, 500.0, 400.0],
        "ret": [0.01, -0.01, 0.02, -0.02],
    })
    p = tmp_path / "panel_cache.parquet"
    df.to_parquet(p)

    out = load_panel(p)
    assert isinstance(out.index, pd.MultiIndex)
    assert list(out.index.names) == ["datetime", "instrument"]
    assert pd.api.types.is_datetime64_any_dtype(out.index.get_level_values("datetime"))
    assert len(out) == 4

    # 兼容旧的 date/code 命名
    df2 = df.rename(columns={"datetime": "date", "instrument": "code"})
    p2 = tmp_path / "panel_cache_alt.parquet"
    df2.to_parquet(p2)
    out2 = load_panel(p2)
    assert list(out2.index.names) == ["datetime", "instrument"]
    assert len(out2) == 4

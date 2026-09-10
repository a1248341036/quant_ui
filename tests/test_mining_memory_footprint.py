# -*- coding: utf-8 -*-
"""挖掘内存占用四件套回归：零拷贝 arrow / 视图切片 / 数据面列裁剪 / run 准入控制。

背景（2026-09-10 实测）：8.2M×124 列全量面板下单挖掘进程私有内存 11.6GiB
（panel 4.9GiB 私有拷贝 + train/val 切片 2.8GiB），4~8GB 服务器完全跑不动。
本组测试锁定四个修复的行为契约：

1. ``panel_mmap``：数值列以 **无 null** 写入（NaN 作为真实浮点值），读取时
   ``to_numpy(zero_copy_only=True)`` 必须成功 → 只读 numpy 视图（writeable=False）；
2. ``slice_panel``：日期区间切片走 ``iloc`` 连续区间视图（与父面板共享内存），
   且与旧布尔掩码路径**逐值等价**；
3. ``cnequity``：聚焦面裁剪列族（辅助插件按需加载、核心/标签/交付依赖列永远保留），
   缓存按数据面签名区分且能命中；
4. ``alphaagent_service``：并发/内存准入控制（超限抛 RunAdmissionError）。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── 1. arrow 零拷贝 ──


def _flat_panel(rows: int = 1000) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=rows // 10), [f"{i:06d}" for i in range(10)]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(0)
    data = {
        "close": rng.random(len(idx)).astype("float32"),
        "volume": rng.random(len(idx)).astype("float32"),
    }
    # 含 NaN 的列：旧实现（Table.from_pandas）会写成 Arrow null → 零拷贝失效
    with_nan = data["close"].copy()
    with_nan[::3] = np.nan
    data["with_nan"] = with_nan
    data["int_flag"] = np.ones(len(idx), dtype="int8")
    df = pd.DataFrame(data, index=idx)
    return df.reset_index()


class TestArrowZeroCopy:
    def test_numeric_columns_have_no_nulls(self, tmp_path):
        from alphaagent.data.adapters.panel_mmap import write_panel_arrow

        flat = _flat_panel()
        path = tmp_path / "p.arrow"
        assert write_panel_arrow(path, flat)

        import pyarrow as pa
        import pyarrow.ipc as ipc

        src = pa.memory_map(str(path), "r")
        table = ipc.open_file(src).read_all()
        # 数值列 null 数必须为 0（NaN 存为真实浮点值）
        assert table.column("with_nan").null_count == 0
        assert table.column("close").null_count == 0
        assert table.column("int_flag").null_count == 0

    def test_read_back_is_zero_copy_readonly(self, tmp_path):
        from alphaagent.data.adapters.panel_mmap import (
            read_panel_arrow_mmap,
            write_panel_arrow,
        )

        flat = _flat_panel()
        path = tmp_path / "p.arrow"
        assert write_panel_arrow(path, flat)
        panel = read_panel_arrow_mmap(path)
        assert panel is not None
        arr = panel["with_nan"].to_numpy()
        # 零拷贝的直接证据：pyarrow 从不可变 arrow 数组给出的 numpy 视图是只读的
        assert arr.flags.writeable is False
        assert np.isnan(arr).sum() == int(flat["with_nan"].isna().sum())

    def test_column_projection_reads_subset(self, tmp_path):
        from alphaagent.data.adapters.panel_mmap import (
            read_panel_arrow_mmap,
            write_panel_arrow,
        )

        flat = _flat_panel()
        path = tmp_path / "p.arrow"
        assert write_panel_arrow(path, flat)
        panel = read_panel_arrow_mmap(path, columns=["close"])
        assert panel is not None
        assert list(panel.columns) == ["close"]


# ── 2. 视图切片 ──


def _panel(n_dates: int = 30, n_inst: int = 5) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2022-01-03", periods=n_dates, freq="B"), [f"{i:06d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(1)
    return pd.DataFrame({"close": rng.random(len(idx)).astype("float32")}, index=idx).sort_index()


class TestSlicePanelView:
    def test_slice_shares_memory_with_parent(self):
        from alphaagent.data.panel import slice_panel

        panel = _panel()
        out = slice_panel(panel, start="2022-01-10", end="2022-01-20")
        assert np.shares_memory(panel["close"].to_numpy(), out["close"].to_numpy())

    def test_slice_matches_mask_semantics(self):
        from alphaagent.data.panel import slice_panel

        panel = _panel()
        for start, end in (("2022-01-10", "2022-01-20"), (None, "2022-01-12"),
                           ("2022-02-01", None), ("2022-01-01", "2022-01-05"),
                           ("2030-01-01", "2030-02-01")):
            out = slice_panel(panel, start=start, end=end)
            dt = panel.index.get_level_values("datetime")
            mask = pd.Series(True, index=panel.index)
            if start is not None:
                mask &= dt >= pd.Timestamp(start)
            if end is not None:
                mask &= dt <= pd.Timestamp(end)
            expected = panel.loc[mask]
            assert out.equals(expected), f"slice mismatch for {start}~{end}"
            assert list(out.index) == list(expected.index)

    def test_full_range_returns_parent(self):
        from alphaagent.data.panel import slice_panel

        panel = _panel()
        assert slice_panel(panel, start="2020-01-01", end="2030-01-01") is panel

    def test_unsorted_index_falls_back_to_mask(self):
        from alphaagent.data.panel import slice_panel

        panel = _panel().sort_index(ascending=False)
        out = slice_panel(panel, start="2022-01-10", end="2022-01-14")
        dt = panel.index.get_level_values("datetime")
        expected = panel.loc[(dt >= pd.Timestamp("2022-01-10")) & (dt <= pd.Timestamp("2022-01-14"))]
        assert out.equals(expected)


# ── 3. 数据面列裁剪 ──


class _FakePlugin:
    def __init__(self, name, cols, priority):
        self.name = name
        self.dataset = f"cne:{name}"
        self._cols = list(cols)
        self.priority = priority

    def panel_columns(self):
        return list(self._cols)

    def is_core(self):
        return self.priority == 0


class TestFacetPruning:
    def test_needed_aux_columns_follows_allowed_scope(self):
        from alphaagent.data.adapters.cnequity import _facet_allowed, _needed_aux_columns

        plugins = [
            _FakePlugin("stock_daily_wide", ["close", "volume"], 0),
            _FakePlugin("forecast", ["pred_surprise", "pred_days_since"], 31),
            _FakePlugin("fundamental", ["funda_roe", "funda_net_profit"], 30),
            _FakePlugin("margin", ["mgn_balance"], 51),
        ]
        focus = ["业绩面"]
        allowed = _facet_allowed(focus)
        assert allowed == {"业绩面"}  # 业绩面无隐含输入面
        wanted = _needed_aux_columns(plugins, allowed)
        assert wanted == {"pred_surprise", "pred_days_since"}
        # 量能面 → 隐含价量输入（辅助插件里没有价量列，故仍为空）
        allowed2 = _facet_allowed(["量能面"])
        assert allowed2 == {"量能面", "价量面"}

    def test_prune_keeps_internal_dependencies(self):
        from alphaagent.data.adapters.cnequity import _facet_allowed, _prune_panel_columns

        panel = pd.DataFrame(
            {
                "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                "amount": [1.0], "volume": [1.0], "turnover_rate": [1.0],
                "adj_close": [1.0], "adjfactor": [1.0], "float_cap": [1.0],
                "is_trade": [1], "not_st": [1], "label_1d_open_to_open": [0.1],
                "vwap": [1.0], "adj_vwap": [1.0], "ret": [0.0],
                "pred_surprise": [0.5], "pred_days_since": [3.0],
                "exp_net_profit": [1e8],
                "funda_roe": [0.1], "mgn_balance": [1e6], "dt_net_buy_90d": [1.0],
            }
        )
        out = _prune_panel_columns(panel, allowed=_facet_allowed(["业绩面"]))
        cols = set(out.columns)
        # 交付/评估内部依赖列必须保留（engine_gate 长表契约 + IC 衰减 + 市值中性）
        assert {"open", "high", "low", "close", "amount", "volume", "turnover_rate",
                "adj_close", "float_cap", "is_trade", "not_st"} <= cols
        assert "label_1d_open_to_open" in cols
        # 聚焦面列保留
        assert {"pred_surprise", "exp_net_profit"} <= cols
        # 未选面辅助列裁掉
        assert not ({"funda_roe", "mgn_balance", "dt_net_buy_90d"} & cols)

    def test_prune_none_when_no_focus(self):
        from alphaagent.data.adapters.cnequity import _facet_allowed

        assert _facet_allowed(None) is None
        assert _facet_allowed([]) is None

    def test_expected_sentinels_scoped_to_focus(self):
        from alphaagent.data.adapters.cnequity import _expected_sentinel_columns

        full = _expected_sentinel_columns(include_fundamentals=True, focus=None)
        assert "mgn_balance" in full and "exp_net_profit" in full
        scoped = _expected_sentinel_columns(include_fundamentals=True, focus=["业绩面", "量能面"])
        assert scoped == frozenset({"exp_net_profit"})
        # 无基本面开关时不做哨兵校验
        assert _expected_sentinel_columns(include_fundamentals=False, focus=None) == frozenset()

    def test_missing_sentinels_with_focus_does_not_require_funda_cols(self):
        """聚焦面板没有 funda_ 列也不能被判为"未命中"（否则缓存永远打不中）。"""
        from alphaagent.data.adapters.cnequity import _missing_funda_sentinels

        panel = pd.DataFrame({"pred_surprise": [1.0], "exp_net_profit": [1.0]})
        assert _missing_funda_sentinels(panel, frozenset({"exp_net_profit"})) == frozenset()
        assert _missing_funda_sentinels(panel, frozenset({"funda_roe"})) == frozenset({"funda_roe"})
        # 旧语义（expected=None）：无 funda_ 列 → 空
        assert _missing_funda_sentinels(panel) == frozenset()

    def test_registry_skips_unneeded_aux_plugins(self, monkeypatch):
        from alphaagent.data.adapters import registry as reg_mod

        calls: list[str] = []

        class _Registry(reg_mod.PluginRegistry):
            def _call_loader(self, plugin, **kwargs):  # noqa: ANN001
                calls.append(plugin.name)
                raise RuntimeError("loader not stubbed")

        registry = _Registry()
        registry._plugins = {
            "stock_daily_wide": _FakePlugin("stock_daily_wide", ["close"], 0),
            "forecast": _FakePlugin("forecast", ["pred_surprise"], 31),
            "margin": _FakePlugin("margin", ["mgn_balance"], 51),
        }
        registry._loaders = {k: (lambda *a, **k2: None) for k in registry._plugins}
        with pytest.raises(RuntimeError):  # 核心 loader 未桩 → 构建失败，但调用记录有效
            registry.build_panel(include_columns={"pred_surprise"})
        assert "margin" not in calls
        assert "forecast" in calls or "stock_daily_wide" in calls

    def test_facet_signature_stable_and_distinct(self):
        from alphaagent.data.adapters.cnequity import _facet_signature

        assert _facet_signature(None) == "all"
        assert _facet_signature([]) == "all"
        a = _facet_signature(["业绩面", "量能面"])
        b = _facet_signature(["量能面", "业绩面"])  # 顺序无关
        c = _facet_signature(["业绩面"])
        assert a == b and a != c and a.startswith("f")


# ── 4. run 准入控制 ──


class TestRunAdmission:
    def test_admission_blocks_over_limit(self, monkeypatch):
        from backend import alphaagent_service as svc

        monkeypatch.setattr(svc, "_MAX_ACTIVE_RUNS", 1)
        monkeypatch.setattr(svc, "_MIN_FREE_GB", 0.0)
        monkeypatch.setattr(svc, "_active_run_count", lambda: 1)
        with pytest.raises(svc.RunAdmissionError) as exc:
            svc._admission_check()
        assert "上限" in str(exc.value)

    def test_admission_blocks_low_memory(self, monkeypatch):
        import psutil

        from backend import alphaagent_service as svc

        monkeypatch.setattr(svc, "_MAX_ACTIVE_RUNS", 0)
        monkeypatch.setattr(svc, "_MIN_FREE_GB", 999.0)
        with pytest.raises(svc.RunAdmissionError) as exc:
            svc._admission_check()
        assert "可用内存" in str(exc.value)
        assert psutil is not None

    def test_admission_passes_when_disabled(self, monkeypatch):
        from backend import alphaagent_service as svc

        monkeypatch.setattr(svc, "_MAX_ACTIVE_RUNS", 0)
        monkeypatch.setattr(svc, "_MIN_FREE_GB", 0.0)
        svc._admission_check()  # 不抛

    def test_admission_status_shape(self, monkeypatch):
        from backend import alphaagent_service as svc

        monkeypatch.setattr(svc, "_MAX_ACTIVE_RUNS", 2)
        monkeypatch.setattr(svc, "_MIN_FREE_GB", 0.0)
        monkeypatch.setattr(svc, "_active_run_count", lambda: 1)
        status = svc.admission_status()
        assert status["max_active_runs"] == 2
        assert status["active_runs"] == 1
        assert status["accepting"] is True

    def test_start_run_runs_admission_first(self, monkeypatch):
        """准入失败必须发生在 Popen 之前（不能起了进程才报错）。"""
        from backend import alphaagent_service as svc

        monkeypatch.setattr(
            svc, "_admission_check", lambda: (_ for _ in ()).throw(svc.RunAdmissionError("blocked"))
        )
        called = {"popen": False}
        monkeypatch.setattr(svc.subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True))
        with pytest.raises(svc.RunAdmissionError):
            svc.start_run({"user_message": "x"})
        assert called["popen"] is False

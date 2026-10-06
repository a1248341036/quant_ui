# -*- coding: utf-8 -*-
"""S1 分段物化的回归门禁（2026-10-07 OCR review M-1 / M-2 落地）。

评审文档：``docs/review/ocr_review_2026-10-07_submit_split_materialize.md``

钉住四件事：
1. 截面类表达式：分段求值 == 全量物化（逐位）；
2. TS 类表达式：差异**恰好**落在 val/test 段首 ``window-1`` 天，且**不是 NaN**
   （这是分段口径的既定代价，不是 bug —— 把它变成显式契约，防止后续误读）；
3. 覆盖不全 / 求值异常：一律回退全量物化（返回 ``None``）；
4. 两条路径都落 ``submit.materialize`` 步骤日志（``mode=split`` / ``mode=full`` + reason），
   否则「优化静默失效」在生产不可观测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alphaagent.factor.ingest import materialize_factor
from alphaagent.factor.mining.delivery import submit as submit_module

SPLITS = {
    "train": ("2024-01-01", "2024-02-09"),
    "val": ("2024-02-12", "2024-03-15"),
    "test": ("2024-03-18", "2024-04-30"),
}


def _panel(days: int = 60, insts: int = 20, seed: int = 0) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=days), [f"S{i:03d}" for i in range(insts)]],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "close": rng.normal(10, 1, len(idx)).astype(np.float32),
            "adj_close": rng.normal(10, 1, len(idx)).astype(np.float32),
            "open": rng.normal(10, 1, len(idx)).astype(np.float32),
            "high": rng.normal(11, 1, len(idx)).astype(np.float32),
            "low": rng.normal(9, 1, len(idx)).astype(np.float32),
            "volume": rng.random(len(idx)).astype(np.float32),
            "amount": rng.random(len(idx)).astype(np.float32),
            "label_1d_open_to_open": rng.normal(0, 0.02, len(idx)).astype(np.float32),
        },
        index=idx,
    )


class _FakeSession:
    """最小会话桩：按日期三段切片 + 空 aux_cache（不碰磁盘/共享缓存）。"""

    def __init__(self, panel: pd.DataFrame, splits: dict | None = None) -> None:
        self.panel = panel
        self.splits = splits or SPLITS

    def get_split_panel(self, split: str):
        start, end = self.splits[split]
        dt = self.panel.index.get_level_values("datetime")
        sub = self.panel.loc[(dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))]
        return sub, start, end

    def get_aux_cache(self, split: str) -> dict:
        return {}


class _BrokenSession:
    """求值期抛错的会话桩（测异常回退）。"""

    def get_split_panel(self, split: str):
        raise RuntimeError("split_panel_boom")

    def get_aux_cache(self, split: str) -> dict:
        return {}


def _capture_logs(monkeypatch) -> list[dict]:
    logged: list[dict] = []
    monkeypatch.setattr(
        submit_module,
        "log_step",
        lambda step, message="", **kw: logged.append({"step": step, "message": message, **kw}),
    )
    return logged


def _differing_days(a, b, panel) -> set:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    dt = panel.index.get_level_values("datetime")
    return {d.date() for d in dt[~same]}


def test_cross_sectional_expr_matches_full_materialize():
    """截面类表达式必须与全量物化逐位一致（分段不改变截面算子结果）。"""
    panel = _panel()
    expr = "CS_ZSCORE($close)"
    split = submit_module._materialize_split_aware(expr, _FakeSession(panel), panel)
    full = materialize_factor(expr, panel)

    assert split is not None
    assert split.expr == full.expr
    np.testing.assert_allclose(split.values, full.values, rtol=0, atol=0, equal_nan=True)


def test_ts_expr_differs_only_at_segment_heads_and_is_not_nan():
    """TS 类表达式：差异恰好是 val/test 段首 window-1 天，且为数值变化（非 NaN）。"""
    panel = _panel()
    window = 5
    expr = f"TS_MEAN($close, {window})"
    split = submit_module._materialize_split_aware(expr, _FakeSession(panel), panel)
    full = materialize_factor(expr, panel)
    assert split is not None

    dt = panel.index.get_level_values("datetime")
    expected_days: set = set()
    for seg in ("val", "test"):
        start, end = SPLITS[seg]
        seg_days = sorted({d for d in dt if pd.Timestamp(start) <= d <= pd.Timestamp(end)})
        expected_days |= {d.date() for d in seg_days[: window - 1]}

    differing = _differing_days(split.values, full.values, panel)
    assert differing == expected_days
    a = np.asarray(split.values, dtype=np.float64)
    b = np.asarray(full.values, dtype=np.float64)
    differ = ~((a == b) | (np.isnan(a) & np.isnan(b)))
    n_inst = panel.index.get_level_values("instrument").nunique()
    assert int(differ.sum()) == len(differing) * n_inst
    # 段首是"窗口被截断后的数值"，不是 NaN（只有 min_periods == window 的算子才出 NaN）
    assert np.isfinite(a[differ]).all()
    assert np.isfinite(b[differ]).all()


def test_incomplete_coverage_falls_back_and_logs_full(monkeypatch):
    """三段未完全覆盖全量（自定义窗口有 gap）→ 回退全量，并留 mode=full 日志。"""
    panel = _panel()
    logged = _capture_logs(monkeypatch)
    # train 前移 + val/test 不动 → 覆盖不全
    splits = {
        "train": ("2024-02-01", "2024-02-29"),
        "val": ("2024-03-01", "2024-03-15"),
        "test": ("2024-03-18", "2024-04-30"),
    }
    out = submit_module._materialize_split_aware(
        "TS_MEAN($close, 5)", _FakeSession(panel, splits), panel, name="f_gap"
    )

    assert out is None
    assert logged and logged[0]["step"] == "submit.materialize"
    assert logged[0]["message"] == "f_gap"
    assert logged[0]["mode"] == "full"
    assert "未完全覆盖" in logged[0]["reason"]


def test_eval_error_falls_back_and_logs_reason(monkeypatch):
    """求值期抛错 → 回退全量（不把新失败点带给 submit），日志带异常类型。"""
    panel = _panel()
    logged = _capture_logs(monkeypatch)
    out = submit_module._materialize_split_aware(
        "CS_ZSCORE($close)", _BrokenSession(), panel, name="f_err"
    )

    assert out is None
    assert logged and logged[0]["mode"] == "full"
    assert "RuntimeError" in logged[0]["reason"]


def test_split_path_logs_mode_split(monkeypatch):
    """命中分段路径时也必须留痕（否则无法确认优化是否生效）。"""
    panel = _panel()
    logged = _capture_logs(monkeypatch)
    out = submit_module._materialize_split_aware(
        "CS_ZSCORE($close)", _FakeSession(panel), panel, name="f_ok"
    )

    assert out is not None
    assert [row["mode"] for row in logged] == ["split"]
    assert logged[0]["step"] == "submit.materialize"
    assert isinstance(logged[0]["ms"], int)

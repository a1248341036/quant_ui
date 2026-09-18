"""ST（风险警示板）横截面剔除掩码：正确性 / 缓存 / 开关 / fail-open / 三段一致。"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor.evaluation.engine import EvaluationEngine
from alphaagent.factor.evaluation.profile import default_evaluation_profiles
from alphaagent.factor.ingest import compute_ingest_metrics
from alphaagent.factor.metrics import st_mask
from alphaagent.factor.mining.context import StockEvalContext
from alphaagent.factor.mining.session import StockEvalSession
from alphaagent.factor.types import IngestPolicy

ST_SYMBOLS = ("000001.SZ", "000002.SZ", "000003.SZ")
ALL_SYMBOLS = tuple(f"{i:06d}.SZ" for i in range(1, 41)) + ("510300.SH",)
# 每只 ST 票的 ST 交易日序号区间 [start, stop)：跨 train（0~11）/ val（12~23）窗口
ST_SPANS = {"000001.SZ": (0, 5), "000002.SZ": (10, 15), "000003.SZ": (18, 23)}
LABEL_COL = "label_1d_open_to_open"
TRAIN_DAYS = 12


def _dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=24)


def _panel(dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """41 只标的（含 1 只 ETF 代码）× 24 个交易日；ST 票的价位刻意偏离。"""
    dates = _dates() if dates is None else dates
    instruments = list(ALL_SYMBOLS)
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    cross = np.tile(np.linspace(-1, 1, len(instruments)), len(dates))
    drift = np.repeat(np.linspace(0, 0.05, len(dates)), len(instruments))
    is_st_inst = np.tile(np.isin(instruments, ST_SYMBOLS), len(dates))
    close = 10 + cross + drift + 5.0 * is_st_inst
    return pd.DataFrame(
        {
            "adj_close": close,
            "open": close * (1 + 0.001 * cross),
            "amount": np.tile(np.linspace(2e7, 8e7, len(instruments)), len(dates)),
            "turnover_rate": np.tile(np.linspace(0.5, 3.0, len(instruments)), len(dates)),
            "float_cap": np.tile(np.linspace(1e8, 5e10, len(instruments)), len(dates)),
            "volume": np.tile(np.linspace(1e5, 1e6, len(instruments)), len(dates)),
            LABEL_COL: 0.002 * cross + 0.0001 * drift,
        },
        index=index,
    )


def _write_st_dataset(root: Path, pairs: list[tuple[str, str]]) -> Path:
    """按 (symbol, trade_date) 写一份 month 分区的 stock_st 数据集（含周末行）。"""
    df = pd.DataFrame(pairs, columns=["symbol", "trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["name"] = "ST*"
    df["type"] = "ST"
    for ym, grp in df.groupby(df["trade_date"].dt.strftime("%Y-%m")):
        part = root / f"trade_date={ym}"
        part.mkdir(parents=True, exist_ok=True)
        grp.to_parquet(part / "part-merged.parquet", index=False)
    return root


@pytest.fixture(autouse=True)
def _clean_mask_cache():
    st_mask.clear_cache()
    yield
    st_mask.clear_cache()


@pytest.fixture
def st_dataset(tmp_path, monkeypatch) -> Path:
    """合成 ST 数据集：3 只票各覆盖若干交易日（跨 train/val 窗口）+ 一个周末行。"""
    dates = _dates()
    pairs: list[tuple[str, str]] = []
    for sym, (lo, hi) in ST_SPANS.items():
        for d in dates[lo:hi]:
            pairs.append((sym, str(pd.Timestamp(d).date())))
    pairs.append(("000004.SZ", "2024-01-06"))  # 周六（源端会出周末行；对面板无害）
    root = _write_st_dataset(tmp_path / "stock_st", pairs)
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(root))
    monkeypatch.delenv("ALPHA_ST_MASK", raising=False)
    return root


def _expected_st_rows(panel: pd.DataFrame) -> np.ndarray:
    """独立于被测实现的期望：按数据集定义逐行判定。"""
    pairs = _expected_st_pairs()
    inst = panel.index.get_level_values("instrument")
    dt = pd.Index(panel.index.get_level_values("datetime"))
    return np.array(
        [(str(i), str(pd.Timestamp(d).date())) in pairs for i, d in zip(inst, dt)],
        dtype=bool,
    )


def _expected_st_pairs() -> set[tuple[str, str]]:
    """数据集里真实写入的 (symbol, trade_date) 集合（按 ST_SPANS 展开）。"""
    dates = _dates()
    pairs: set[tuple[str, str]] = set()
    for sym, (lo, hi) in ST_SPANS.items():
        for d in dates[lo:hi]:
            pairs.add((sym, str(pd.Timestamp(d).date())))
    return pairs


def _expected_finite_share(sub: pd.DataFrame) -> float:
    """切片内非 ST 行占比（= 因子有限值占比的自证期望）。"""
    return 1.0 - float(_expected_st_rows(sub).mean())


# ── 掩码正确性 ────────────────────────────────────────────────────────


def test_row_mask_marks_only_st_rows(st_dataset) -> None:
    panel = _panel()
    mask = st_mask.st_row_mask(panel)
    expected = _expected_st_rows(panel)
    assert mask is not None
    assert mask.dtype == bool and len(mask) == len(panel)
    assert np.array_equal(mask, expected)
    info = st_mask.st_mask_info(panel)
    assert info["available"] is True
    assert info["n_st_rows"] == int(expected.sum())
    assert info["st_row_share"] == pytest.approx(float(expected.mean()), abs=1e-6)
    assert info["membership"]["n_symbols"] == 4


def test_mask_values_sets_nan_only_on_st_rows(st_dataset) -> None:
    panel = _panel()
    factor = pd.Series(
        panel["adj_close"].to_numpy(dtype=np.float32), index=panel.index, name="f"
    )
    out = st_mask.mask_values(factor, panel)
    expected = _expected_st_rows(panel)
    assert isinstance(out, pd.Series) and out.name == "f"
    assert out.index.equals(factor.index)
    assert np.array_equal(np.isnan(out.to_numpy()), expected)
    assert np.array_equal(
        out.to_numpy()[~expected], factor.to_numpy()[~expected]
    )
    # 输入不被改动
    assert not np.isnan(factor.to_numpy()).any()
    # ETF（510300.SH）不受影响
    assert out.loc[(slice(None), "510300.SH")].notna().all()


def test_mask_values_accepts_ndarray_and_ignores_length_mismatch(st_dataset) -> None:
    panel = _panel()
    arr = panel["adj_close"].to_numpy(dtype=np.float32)
    out = st_mask.mask_values(arr, panel)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.array_equal(np.isnan(out), _expected_st_rows(panel))
    short = np.ones(3, dtype=np.float32)
    assert st_mask.mask_values(short, panel) is short


def test_non_multiindex_panel_is_noop(st_dataset) -> None:
    flat = pd.DataFrame({"adj_close": [1.0, 2.0]})
    values = np.array([1.0, 2.0], dtype=np.float32)
    assert st_mask.st_row_mask(flat) is None
    assert st_mask.mask_values(values, flat) is values


# ── 缓存与开关 ────────────────────────────────────────────────────────


def test_row_mask_cached_by_index_identity(st_dataset, monkeypatch) -> None:
    panel = _panel()
    calls = {"n": 0}
    original = st_mask._build_mask

    def _counting(index, membership):
        calls["n"] += 1
        return original(index, membership)

    monkeypatch.setattr(st_mask, "_build_mask", _counting)
    first = st_mask.st_row_mask(panel)
    second = st_mask.st_row_mask(panel)
    assert calls["n"] == 1
    assert first is second  # 命中缓存返回同一数组
    # 不同索引身份的切片各自构建（口径按各自横截面）
    other = panel.iloc[:100]
    assert st_mask.st_row_mask(other) is not None
    assert calls["n"] == 2


def test_env_switch_disables_mask(st_dataset, monkeypatch) -> None:
    panel = _panel()
    monkeypatch.setenv("ALPHA_ST_MASK", "0")
    assert st_mask.st_row_mask(panel) is None
    values = np.ones(len(panel), dtype=np.float32)
    assert st_mask.mask_values(values, panel) is values
    info = st_mask.st_mask_info(panel)
    assert info["available"] is False and info["reason"] == "disabled_by_env"
    monkeypatch.setenv("ALPHA_ST_MASK", "1")
    assert st_mask.st_row_mask(panel) is not None


def test_asset_type_guard_skips_etf(st_dataset) -> None:
    panel = _panel()
    assert st_mask.st_row_mask(panel, asset_type="etf") is None
    assert st_mask.st_row_mask(panel, asset_type="stock") is not None


# ── fail-open ────────────────────────────────────────────────────────


def test_missing_dataset_fails_open(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "nope"))
    panel = _panel()
    values = np.ones(len(panel), dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        assert st_mask.st_row_mask(panel) is None
        assert st_mask.mask_values(values, panel) is values
    assert any("ST" in rec.message or "ST" in rec.getMessage() for rec in caplog.records)


def test_empty_dataset_dir_fails_open(tmp_path, monkeypatch, caplog) -> None:
    root = tmp_path / "stock_st"
    root.mkdir()
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(root))
    with caplog.at_level(logging.WARNING):
        assert st_mask.st_row_mask(_panel()) is None
    assert st_mask.st_mask_info(_panel())["available"] is False


def test_corrupt_partition_fails_open(tmp_path, monkeypatch, caplog) -> None:
    root = tmp_path / "stock_st" / "trade_date=2024-01"
    root.mkdir(parents=True)
    (root / "part-merged.parquet").write_bytes(b"not-a-parquet")
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(root.parent))
    with caplog.at_level(logging.WARNING):
        assert st_mask.st_row_mask(_panel()) is None


def test_dataset_unavailable_does_not_raise_in_info(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "nope"))
    info = st_mask.st_mask_info(_panel())
    assert info["available"] is False and info["st_row_share"] == 0.0


# ── 三段（train / val / 盲测）口径一致 ────────────────────────────────


def _session(panel: pd.DataFrame | None = None) -> StockEvalSession:
    dates = _dates()
    return StockEvalSession(
        session_id="st-test",
        ctx=StockEvalContext(
            panel_path=Path("panel.parquet"),
            train_start=str(dates[0].date()),
            train_end=str(dates[TRAIN_DAYS - 1].date()),
            val_start=str(dates[TRAIN_DAYS].date()),
            val_end=str(dates[-1].date()),
        ),
        panel=_panel() if panel is None else panel,
    )


def _profile_panel(session: StockEvalSession, profile_id: str) -> pd.DataFrame:
    """引擎同一套 panel 选择逻辑（train/val 切片 / full 全窗）。"""
    from alphaagent.data.panel import slice_panel

    profile = default_evaluation_profiles()[profile_id]
    if profile.split == "full":
        start, end = session.ctx.coverage_range()
        return slice_panel(session.panel, start=start, end=end)
    return session.get_split_panel(profile.split)[0]


def test_engine_excludes_st_in_all_profiles(st_dataset) -> None:
    """引擎三段（train_screen / validation / production_delivery=full）都剔除 ST。"""
    session = _session()
    engine = EvaluationEngine(default_evaluation_profiles())

    for profile_id in ("train_screen", "validation", "production_delivery"):
        result = engine.evaluate(
            session, profile_id=profile_id, multi_line_expr="$adj_close",
            include_charts=False,
        )
        assert result["ok"], result.get("error")
        core = result["metrics"]["cross_sectional_core"]
        expected = _expected_finite_share(_profile_panel(session, profile_id))
        assert 0.0 < expected < 1.0
        assert core["factor_coverage"] == pytest.approx(expected, abs=1e-9)
        assert isinstance(core["decile_mean_label"], list) and core["decile_mean_label"]


def test_engine_switch_off_keeps_full_cross_section(st_dataset, monkeypatch) -> None:
    session = _session()
    engine = EvaluationEngine(default_evaluation_profiles())
    on = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr="$adj_close", include_charts=False
    )["metrics"]["cross_sectional_core"]
    monkeypatch.setenv("ALPHA_ST_MASK", "0")
    off = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr="$adj_close", include_charts=False
    )["metrics"]["cross_sectional_core"]
    assert off["factor_coverage"] == pytest.approx(1.0, abs=1e-12)
    assert on["factor_coverage"] < off["factor_coverage"]
    # 剔除真实改变了横截面（不是只改了 coverage 字段）
    assert not np.isclose(on["ic"], off["ic"], atol=1e-9)


def test_ingest_metrics_use_same_mask_as_engine(st_dataset) -> None:
    """submit 三段门槛消费的 compute_ingest_metrics 与引擎同口径。"""
    panel = _panel().sort_index()
    dates = _dates()
    values = panel["adj_close"].to_numpy(dtype=np.float32)
    policy = IngestPolicy(
        train_start=str(dates[0].date()),
        val_end=str(dates[-1].date()),
        label_col=LABEL_COL,
    )
    metrics = compute_ingest_metrics(values, panel, policy)
    # 门槛字段名 coverage（DeliveryChecker.stage_one_stats 优先读它）
    assert metrics["coverage"] == pytest.approx(_expected_finite_share(panel), abs=1e-9)
    # finite_ratio 描述落库值（不做 ST 剔除）
    assert metrics["finite_ratio"] == pytest.approx(1.0, abs=1e-12)

    engine = EvaluationEngine(default_evaluation_profiles())
    session = _session(panel)
    prof = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr="$adj_close", include_charts=False
    )["metrics"]["cross_sectional_core"]
    # 同一份 panel 全窗：引擎 train 切片 vs ingest 全窗口径各自按自身横截面剔除，
    # 两者都等于"该切片非 ST 行占比"
    assert prof["factor_coverage"] == pytest.approx(
        _expected_finite_share(_profile_panel(session, "train_screen")), abs=1e-9
    )


def test_ingest_metrics_switch_off(st_dataset, monkeypatch) -> None:
    panel = _panel().sort_index()
    dates = _dates()
    values = panel["adj_close"].to_numpy(dtype=np.float32)
    policy = IngestPolicy(
        train_start=str(dates[0].date()),
        val_end=str(dates[-1].date()),
        label_col=LABEL_COL,
    )
    monkeypatch.setenv("ALPHA_ST_MASK", "0")
    assert compute_ingest_metrics(values, panel, policy)["coverage"] == pytest.approx(
        1.0, abs=1e-12
    )


def test_ingest_metrics_fail_open_on_missing_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHA_ST_MASK_PATH", str(tmp_path / "missing"))
    panel = _panel().sort_index()
    dates = _dates()
    values = panel["adj_close"].to_numpy(dtype=np.float32)
    policy = IngestPolicy(
        train_start=str(dates[0].date()),
        val_end=str(dates[-1].date()),
        label_col=LABEL_COL,
    )
    assert compute_ingest_metrics(values, panel, policy)["coverage"] == pytest.approx(
        1.0, abs=1e-12
    )

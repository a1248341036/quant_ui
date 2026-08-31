"""统一时间窗口配置中心。

Alpha 全链路（因子挖掘 train/val/test 窗口、ML 组合训练时间边界）的时间
边界在此收口，任何组件不应再散落硬编码日期：

- train（2020-01-01 ~ 2022-12-31）：统计门槛（IC/ICIR/coverage/autocorr）
- val（2023-01-01 ~ 2024-12-31）：样本外保留率 + engine_gate 可交易性
- test（2025-01-01 ~ 数据最新交易日）：留出测试段终审（只报告不拦截）

窗口重映射记录（2026-08-29）：
- 旧窗口 train 2018-2022 / val 2023-2025 / test 2026-01 起（约 8 个月，walk-forward
  每折仅 ~2 个月，OOS IC 波动过大、终审可信度低）。
- 现窗口 train 2020-2022 / val 2023-2024 / test 2025-01 起（约 20 个月）：
  测试段样本更足、终审更可信；train 只取近 3 年（2020-2022，含牛熊震荡），
  避免过远期数据因因子衰减污染当前信号；代价是既有 2026 起盲测锁定与相关
  报告口径作废。
- 数据源覆盖自 2009-01-05，train 起点前移/缩短在数据上完全可行。

TEST_END 不硬编码：数据湖每日增量更新，固定日期会随数据演进过期
（历史教训：硬编码 2026-08-29 超过了实际数据截至 2026-08-28）。
动态解析链（以数据源为准，逐级回落）：

1. ``stock_daily_wide`` —— external Tushare 宽表（panel 实际数据源）最新交易日；
2. ``daily_bars`` —— CNE curated 全市场日线（canonical 兜底）；
3. CNE 数据湖 watermark（``meta/state/daily_bars.json`` 的 last_success_trade_date）；
4. 最后兜底：今天（防御性，正常不会走到）。

解析结果进程级缓存（``lru_cache``），数据每日更新由新进程感知；
长时间驻留进程（如 backend）一天内取值稳定，符合数据演进节奏。
"""

from __future__ import annotations

import functools
import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 静态窗口边界（挖掘循环专用，训练/验证窗口锁定不改）────────────────
DEFAULT_TRAIN_START = "2020-01-01"
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VAL_START = "2023-01-01"
DEFAULT_VAL_END = "2024-12-31"
DEFAULT_TEST_START = "2025-01-01"
# 测试段右端：None = 动态解析数据源最新交易日（见 resolve_test_end）。
# 不用固定日期作默认值——固定值会随数据演进失效并误导（暗示数据覆盖到）。
DEFAULT_TEST_END: str | None = None

# ── 回测/对比页默认起点（前端表单初始化）──────────────────────────────
# 回测与因子对比默认从 train 起点开始，终点动态取数据源最新交易日；
# 让前端 Backtest/Composite/Code 等页面不再散落硬编码日期。
BT_DEFAULT_START: str = DEFAULT_TRAIN_START

# 数据源优先级：panel 实际数据源优先，canonical 日线兜底。
_DATASET_PRIORITY = ("stock_daily_wide", "daily_bars")
# ETF 数据源：本地 etf_panel.parquet 桥接的 external 数据集（无 curated 兜底）。
_ETF_DATASET_PRIORITY = ("etf_bars",)
# external（非 curated）数据集：由 CNE external adapter 提供 coverage。
_EXTERNAL_DATASETS = frozenset({"stock_daily_wide", "etf_bars"})

# CNE 仓库根（window_config.py ← alphaagent/factor → 项目根）
_CNE_ROOT = Path(__file__).resolve().parents[2] / "CNEquity"
_CNE_CONFIG = _CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"


# ── 动态解析：数据源最新交易日 ────────────────────────────────────────


def _load_cne_config():
    """加载 CNE 配置（惰性）。失败返回 None（调用方走兜底链）。"""
    try:
        from cnequity.config import load_config

        if not _CNE_CONFIG.is_file():
            return None
        return load_config(_CNE_CONFIG)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE 配置加载失败（TEST_END 走兜底链）: %s", exc)
        return None


def _external_coverage_end(cfg, dataset: str) -> str | None:
    """external 数据集（stock_daily_wide 等）最新交易日。"""
    try:
        from cnequity.external.registry import external_coverage_bounds

        _lo, hi = external_coverage_bounds(cfg, dataset)
        return _to_iso(hi) if hi is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("external_coverage_bounds(%s) 失败: %s", dataset, exc)
        return None


def _curated_coverage_end(cfg, dataset: str) -> str | None:
    """curated 数据集（daily_bars 等）最新交易日。"""
    try:
        from cnequity.query.universe import coverage_end_date

        hi = coverage_end_date(cfg, dataset)
        return _to_iso(hi) if hi is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("coverage_end_date(%s) 失败: %s", dataset, exc)
        return None


def _state_watermark(cfg) -> str | None:
    """CNE 数据湖 watermark（data/state/daily_bars.json 的 last_success_trade_date）。"""
    try:
        state_file = cfg.data_root / "meta" / "state" / "daily_bars.json"
        if state_file.is_file():
            state = json.loads(state_file.read_text(encoding="utf-8"))
            wm = state.get("last_success_trade_date")
            if wm:
                return str(wm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE watermark 读取失败: %s", exc)
    return None


def _to_iso(value) -> str:
    """date / datetime / Timestamp / str → 'YYYY-MM-DD'。"""
    if isinstance(value, str):
        return value[:10]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@functools.lru_cache(maxsize=4)
def resolve_test_end(asset_type: str = "stock") -> str:
    """解析测试段右端：数据源最新交易日（进程内缓存）。

    Parameters
    ----------
    asset_type : str
        "stock"（默认）或 "etf"。ETF 的数据源为本地 etf_bars，
        无 curated 兜底；股票维持既有 stock_daily_wide → daily_bars 链。

    Returns
    -------
    str
        ``YYYY-MM-DD``，必然 >= DEFAULT_TEST_START（否则返回 TEST_START）。
    """
    cfg = _load_cne_config()
    if cfg is not None:
        priority = _ETF_DATASET_PRIORITY if asset_type == "etf" else _DATASET_PRIORITY
        for ds in priority:
            val = (
                _external_coverage_end(cfg, ds)
                if ds in _EXTERNAL_DATASETS
                else _curated_coverage_end(cfg, ds)
            )
            if val:
                logger.info("TEST_END 数据源 %s → %s", ds, val)
                end = val
                break
        else:
            wm = _state_watermark(cfg)
            if wm:
                logger.info("TEST_END 数据源 CNE watermark → %s", wm)
                end = wm
            else:
                end = date.today().isoformat()
                logger.warning("TEST_END 无可用数据源，兜底今天 %s（首次调用）", end)
    else:
        end = date.today().isoformat()
        logger.warning("TEST_END 配置加载失败，兜底今天 %s", end)

    # 防御：解析结果不得早于测试段起点（异常数据回退起点）
    if end < DEFAULT_TEST_START:
        logger.warning("TEST_END=%s 早于 TEST_START=%s，回退起点", end, DEFAULT_TEST_START)
        return DEFAULT_TEST_START
    return end


def test_window(asset_type: str = "stock") -> tuple[str, str]:
    """(test_start, test_end) 测试段完整窗口。"""
    return DEFAULT_TEST_START, resolve_test_end(asset_type)


def mining_window() -> tuple[str, str, str, str]:
    """(train_start, train_end, val_start, val_end) 挖掘窗口。"""
    return DEFAULT_TRAIN_START, DEFAULT_TRAIN_END, DEFAULT_VAL_START, DEFAULT_VAL_END


def coverage_window(asset_type: str = "stock") -> tuple[str, str]:
    """panel 加载范围（train∪val∪test）。"""
    test_start, test_end = test_window(asset_type)
    return (
        min(DEFAULT_TRAIN_START, DEFAULT_VAL_START, test_start),
        max(DEFAULT_TRAIN_END, DEFAULT_VAL_END, test_end),
    )


def window_defaults(asset_type: str = "stock") -> dict[str, str]:
    """完整窗口字段 dict（train/val/test + 回测默认），供 CLI/API 默认值构造。"""
    test_start, test_end = test_window(asset_type)
    return {
        "train_start": DEFAULT_TRAIN_START,
        "train_end": DEFAULT_TRAIN_END,
        "val_start": DEFAULT_VAL_START,
        "val_end": DEFAULT_VAL_END,
        "test_start": test_start,
        "test_end": test_end,
        "bt_start": BT_DEFAULT_START,
        "bt_end": test_end,
    }
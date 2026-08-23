"""CNEquity 数据湖读取接入层。

quant_ui 全面迁移 CNE 数据源的阶段 1：把 `cnequity.query.reader.load()`
封装成量化框架可直接消费的 pandas 口径（code/date 宽表 + turn20/am20）。

设计约定
--------
* 懒加载：首次调用才 import cnequity（避免后端启动时强依赖 CNE 已安装）。
* 配置解析与 CNE serve 一致：CNEquity 配置里的相对路径（如
  ``../data/quant_dataset``）是相对 ``CNEquity/`` 目录解析的，
  因此本模块先 ``os.chdir`` 到 CNEquity 目录再 ``load_config``，
  与 ``cne serve --config`` 行为一致。
* 双路径：``QUANT_USE_CNE=1`` 时启用 CNE 读取；未启用/失败时抛
  ``CneUnavailable``，由上层回退到旧路径（panel.parquet / 年度 parquet）。
* 代码口径：框架内部统一 6 位 code（``600519``），CNE 用 symbol
  （``600519.SH``）。code↔symbol 映射来自 CNE ``instruments`` 数据集，
  进程内缓存，避免每次请求都全量扫描。
"""

from __future__ import annotations

import os
import json
import threading
from pathlib import Path

import pandas as pd

# 是否启用 CNE 读取（环境变量开关，默认关闭，逐项灰度开启）
USE_CNE = os.getenv("QUANT_USE_CNE", "").strip().lower() in ("1", "true", "yes", "on")

# CNEquity 仓库根目录
CNE_ROOT = Path(__file__).resolve().parent.parent / "CNEquity"
CNE_CONFIG = CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"


class CneUnavailable(RuntimeError):
    """CNE 未启用/未安装/读取失败，调用方应回退旧路径。"""


# ── 懒加载状态 ────────────────────────────────────────────────────────
_lazy_lock = threading.Lock()
_lazy: dict = {"cfg": None, "load": None, "failed": None}
_instruments_cache: dict[str, str] | None = None
_instruments_lock = threading.Lock()


def _lazy_load() -> tuple[object, object]:
    """懒加载 cnequity 配置与 load 函数；失败抛 CneUnavailable。"""
    if not USE_CNE:
        raise CneUnavailable("QUANT_USE_CNE 未开启，CNE 读取未启用")
    if _lazy["failed"] is not None:
        raise CneUnavailable(_lazy["failed"])
    if _lazy["load"] is not None:
        return _lazy["cfg"], _lazy["load"]
    with _lazy_lock:
        if _lazy["load"] is not None:
            return _lazy["cfg"], _lazy["load"]
        try:
            if not CNE_CONFIG.is_file():
                raise FileNotFoundError(f"CNE 配置文件不存在: {CNE_CONFIG}")
            from cnequity.config import load_config
            from cnequity.query.reader import load as cne_load

            # CNE 配置里的相对路径按 CNEquity/ 目录解析，先 chdir 保持一致
            old = Path.cwd()
            try:
                os.chdir(CNE_ROOT)
                cfg = load_config(CNE_CONFIG)
            finally:
                os.chdir(old)
            _lazy["cfg"] = cfg
            _lazy["load"] = cne_load
            return cfg, cne_load
        except Exception as exc:  # noqa: BLE001
            _lazy["failed"] = f"CNE 初始化失败: {exc}"
            raise CneUnavailable(_lazy["failed"]) from exc


def cne_enabled() -> bool:
    """CNE 读取是否可用（环境开关 + 初始化成功）。"""
    try:
        _lazy_load()
        return True
    except CneUnavailable:
        return False


def source_status() -> dict:
    """Return the last CNE daily-bars source status for UI diagnostics."""
    try:
        cfg, _ = _lazy_load()
        path = cfg.meta_root / "state" / "daily_bars.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        status = payload.get("source_status")
        if not isinstance(status, dict):
            status = {}
        return {
            "status": "ok" if payload.get("tushare_available", False) else "warn",
            "provider": status.get("provider"),
            "tushare_available": bool(payload.get("tushare_available", False)),
            "fallback_used": bool(payload.get("fallback_used", False)),
            "tushare_error": status.get("tushare_error"),
            "updated_at": payload.get("updated_at"),
            "path": str(path),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "unknown", "error": str(exc)}


def _load_instruments() -> dict[str, str]:
    """code(6位) -> symbol(带后缀) 映射，进程内缓存。"""
    global _instruments_cache
    if _instruments_cache is not None:
        return _instruments_cache
    with _instruments_lock:
        if _instruments_cache is not None:
            return _instruments_cache
        cfg, cne_load = _lazy_load()
        try:
            df = cne_load("instruments", config=cfg)
            mapping: dict[str, str] = {}
            for row in df.select(["symbol"]).iter_rows():
                sym = str(row[0]).strip()
                if "." in sym:
                    mapping[sym.split(".")[0].zfill(6)] = sym
            _instruments_cache = mapping
            return mapping
        except Exception as exc:  # noqa: BLE001
            raise CneUnavailable(f"读取 instruments 失败: {exc}") from exc


def code_to_symbol(code: str) -> str | None:
    """6 位 code -> CNE symbol（如 600519 -> 600519.SH）；查不到返回 None。"""
    code = str(code).zfill(6)
    try:
        return _load_instruments().get(code)
    except CneUnavailable:
        return None


def symbol_to_code(symbol: str) -> str:
    """CNE symbol -> 6 位 code（600519.SH -> 600519）。"""
    return str(symbol).split(".")[0].zfill(6)


def _codes_to_symbols(codes: list[str] | None) -> list[str] | None:
    if not codes:
        return None
    six = {str(c).zfill(6) for c in codes}
    mapping = _load_instruments()
    missing = six - set(mapping.keys())
    if missing:
        raise CneUnavailable(f"以下代码在 CNE instruments 中不存在: {sorted(missing)[:5]}")
    return [mapping[c] for c in six]


def load_stock_daily(
    codes: list[str] | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    adjust: str = "qfq",
    factor_buffer_days: int = 40,
) -> pd.DataFrame:
    """从 CNE 原生 ``daily_bars`` 读取日线并整理成框架面板口径。

    返回列：date, open, high, low, close, turnover, amount, code,
            turn20, am20, volume（与 ``core.data._finalize_stock_df`` 一致）。
    失败抛 CneUnavailable，由上层回退。

    factor_buffer_days: 与旧路径 ``_load_panel_pg_parquet`` 一致，查询起点
    前移的自然日缓冲（保证 turn20/am20 滚动窗口冷启动口径一致）。
    """
    cfg, cne_load = _lazy_load()
    symbols = _codes_to_symbols(codes)
    # 与旧路径口径对齐：start 前移缓冲，保证头部滚动因子完整
    calc_start = start
    if start and factor_buffer_days > 0:
        calc_start = (
            pd.Timestamp(start) - pd.Timedelta(days=factor_buffer_days)
        ).date().isoformat()
    try:
        df = cne_load(
            "daily_bars",
            start=calc_start,
            end=end,
            symbols=symbols,
            config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        raise CneUnavailable(f"读取 CNE daily_bars 失败: {exc}") from exc
    if df is None or df.is_empty():
        raise CneUnavailable("CNE daily_bars 无数据（区间/股票池过滤后为空）")
    # 转 pandas 并统一列口径
    pdf = df.to_pandas()
    pdf = pdf.rename(columns={"trade_date": "date"})
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["ts_code"] = pdf["symbol"].astype(str)
    pdf["code"] = pdf["ts_code"].str[:6]
    pdf["turnover"] = 0.0

    # daily_bars is the canonical OHLCV source. Enrich only the legacy-derived
    # fields (turnover and adj_factor) when the wide bridge has them; this keeps
    # the UI contract stable without making the wide archive authoritative.
    try:
        wide = cne_load(
            "stock_daily_wide",
            start=calc_start,
            end=end,
            symbols=symbols,
            config=cfg,
        ).to_pandas()
        wide = wide.rename(columns={"trade_date": "date", "turnover_rate": "turnover"})
        wide["date"] = pd.to_datetime(wide["date"])
        wide["ts_code"] = wide["ts_code"].astype(str)
        fields = [c for c in ("ts_code", "date", "turnover", "adj_factor") if c in wide.columns]
        if len(fields) > 2:
            pdf = pdf.drop(columns=[c for c in ("turnover", "adj_factor") if c in pdf.columns])
            pdf = pdf.merge(wide[fields], on=["ts_code", "date"], how="left")
    except Exception:
        pdf["adj_factor"] = 1.0

    pdf["turnover"] = pd.to_numeric(pdf.get("turnover", 0.0), errors="coerce").fillna(0.0)
    pdf["adj_factor"] = pd.to_numeric(pdf.get("adj_factor", 1.0), errors="coerce").fillna(1.0)
    from .data import _finalize_stock_df
    return _finalize_stock_df(pdf, adj=adjust)


def load_stock_daily_single(code: str, days: int = 250, adjust: str = "qfq") -> pd.DataFrame:
    """单只股票最近 days 个交易日行情（走 CNE）。"""
    code = str(code).zfill(6)
    df = load_stock_daily(codes=[code], adjust=adjust)
    if df.empty:
        raise CneUnavailable(f"CNE 无 {code} 行情")
    return df.sort_values("date").tail(days).reset_index(drop=True)


def load_index(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """从 CNE ``index_bars_external`` 读取指数日线。

    返回列与框架 ``core.data.load_index`` 一致：date, code, name, open, close。
    失败抛 CneUnavailable，由上层回退到 csv 路径。
    """
    cfg, cne_load = _lazy_load()
    try:
        df = cne_load("index_bars_external", start=start, end=end, config=cfg)
    except Exception as exc:  # noqa: BLE001
        raise CneUnavailable(f"读取 index_bars_external 失败: {exc}") from exc
    if df is None or df.is_empty():
        raise CneUnavailable("index_bars_external 无数据")
    pdf = df.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["code"] = pdf["code"].astype(str)
    pdf["name"] = pdf["name"].astype(str)
    cols = [c for c in ("date", "code", "name", "open", "close") if c in pdf.columns]
    return pdf[cols].sort_values(["code", "date"]).reset_index(drop=True)

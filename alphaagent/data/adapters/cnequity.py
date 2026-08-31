"""CNEquity 数据源 adapter —— 从 CNE 数据湖实时构建 AlphaAgent panel。

**插件化架构**：本模块是入口，实际数据加载由 ``plugins/`` 下的插件完成。
每个插件对应一个 CNE 数据集，声明列映射与加载逻辑。

加新数据源只需在 ``plugins/`` 下新建一个 .py 文件，无需修改本模块。
注册中心会自动发现并合并所有插件的数据。

用法
----
    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    panel = load_panel_from_cne(start="2020-01-01", end="2024-12-31")

在 StockEvalContext 中使用特殊 panel_path ``cne://`` 触发此 adapter，
SessionStore.create() 会检测该标识并走 adapter 而非读 parquet。

插件列表
--------
- ``stock_daily_wide`` (priority=0)：核心行情，OHLCV + adjfactor + 估值 + ST 标记
- 新增插件在 ``plugins/`` 目录下创建 .py 文件即可

架构
----
    load_panel_from_cne()
        → PluginRegistry.build_panel()
            → plugins/stock_daily_wide.load()    (核心, priority=0)
            → plugins/<other>.load()             (辅助, priority>0)
            → build_panel_from_hq()              (衍生列: adj_*, ret, vwap, label)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.data.adapters.registry import (
    DataSourcePlugin,
    PluginRegistry,
    get_registry,
)

logger = logging.getLogger(__name__)

# 标识符：panel_path == CNE_SOURCE 时走 adapter
CNE_SOURCE = "cne://"

# 磁盘面板缓存：构建好的 panel 落盘 parquet，避免每次评估都重算 30-70s。
# key = (start, end, include_fundamentals, schema_version)，不含数据湖 watermark：
# 历史固定区间永远命中（不需要因 watermark 每天变化而重建累积文件）。
# 命中条件 = 请求区间被缓存面板实际覆盖范围（文件内 datetime 列）包含。
# 目录文件数上限 _CACHE_MAX_FILES，写入时淘汰最旧文件，锁死长期空间。
_CACHE_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "panel" / "cache"
_CNE_STATE_FILE = (
    Path(__file__).resolve().parents[4]
    / "CNEquity" / "data" / "quant_dataset" / "_cnequity" / "meta" / "state" / "daily_bars.json"
)
# 缓存格式/构建逻辑版本：代码变更影响 panel 内容时 +1 强制全部重建
_CACHE_SCHEMA_VERSION = 3  # v3: +forecast/shareholder_counts/event_faces 列族
# 缓存文件数上限（每个 ~0.9GB，6 个 ≈ 5.4GB；按需调大）
_CACHE_MAX_FILES = 6
_CACHE_INDEX_COLS = ["datetime", "instrument"]


def _cne_watermark() -> str:
    """数据湖最后成功交易日（仅用于日志/可观测，不参与缓存 key）。"""
    try:
        if _CNE_STATE_FILE.is_file():
            state = json.loads(_CNE_STATE_FILE.read_text(encoding="utf-8"))
            wm = state.get("last_success_trade_date")
            if wm:
                return str(wm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE watermark 读取失败: %s", exc)
    return "unknown"


def _cache_path(start: str | None, end: str | None, include_fundamentals: bool) -> Path:
    """由参数 + schema 版本生成写入路径（同一个参数组合稳定复用同一文件）。"""
    mk = "-".join(str(x or "all") for x in (start, end))
    key = hashlib.sha256(
        f"{mk}|{include_fundamentals}|v{_CACHE_SCHEMA_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    # 文件名带 schema 版本前缀：_find_cached_panel 只认当前版本的缓存，
    # 版本升版后旧缓存自然失效（文件数由 _CACHE_MAX_FILES 淘汰）
    return _CACHE_ROOT / f"panel_v{_CACHE_SCHEMA_VERSION}_{key}.parquet"


def _find_cached_panel(start: str | None, end: str | None, include_fundamentals: bool) -> pd.DataFrame | None:
    """遍历缓存目录，返回第一个覆盖请求区间且基本面开关匹配的缓存面板。"""
    try:
        if not _CACHE_ROOT.is_dir():
            return None
        req_start = pd.Timestamp(start) if start else None
        req_end = pd.Timestamp(end) if end else None
        for path in sorted(_CACHE_ROOT.glob(f"panel_v{_CACHE_SCHEMA_VERSION}_*.parquet")):
            try:
                df = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CNE panel 缓存读取失败，跳过 %s: %s", path.name, exc)
                continue
            if not isinstance(df.index, pd.MultiIndex):
                if set(_CACHE_INDEX_COLS).issubset(df.columns):
                    df = df.set_index(_CACHE_INDEX_COLS)
                else:
                    continue
            idx_min, idx_max = _panel_coverage(df)
            if idx_min is None or idx_max is None:
                continue
            # 起点兼容判断：缓存覆盖从「首个交易日」（如 2018-01-02）开始，
            # 而请求常用自然日期起点（如 2018-01-01）。若两者同属一个自然月，
            # 说明请求起点只是缓存首日前面的非交易日，视作命中而非重建。
            if req_start is not None and req_start < idx_min:
                if not (req_start.to_period("M") == pd.Timestamp(idx_min).to_period("M")):
                    continue  # 请求起点早于缓存覆盖起点（且跨月，非同日历月容差）
            if req_end is not None and req_end > idx_max:
                continue  # 请求终点晚于缓存覆盖终点
            has_funda = any(str(c).startswith("funda_") for c in df.columns)
            if has_funda != include_fundamentals:
                continue  # 基本面开关不匹配，跳过
            logger.info("CNE adapter: panel 命中磁盘缓存 %s (%d 行)", path.name, len(df))
            return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存扫描异常（忽略，将重建）: %s", exc)
    return None


def _panel_coverage(panel: pd.DataFrame) -> tuple[Any, Any]:
    """面板实际覆盖的 [min_date, max_date]（datetime 索引层）。"""
    dts = panel.index.get_level_values("datetime")
    if len(dts) == 0:
        return None, None
    return dts.min(), dts.max()


def _purge_old_cache(keep: Path) -> None:
    """缓存目录超过上限时删除最旧文件（保留刚写入的 keep）。"""
    try:
        if not _CACHE_ROOT.is_dir():
            return
        files = sorted(
            (p for p in _CACHE_ROOT.glob("panel_*.parquet") if p != keep),
            key=lambda p: p.stat().st_mtime,
        )
        while len(files) + 1 > _CACHE_MAX_FILES:  # +1 为刚写入的文件
            victim = files.pop(0)
            try:
                victim.unlink()
                logger.info("CNE panel 缓存淘汰旧文件: %s (%.0fMB)",
                            victim.name, victim.stat().st_size / 1e6)
            except OSError as exc:
                logger.warning("CNE panel 缓存淘汰失败: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存淘汰异常（忽略）: %s", exc)


def _save_cached_panel(path: Path, panel: pd.DataFrame) -> None:
    """原子写缓存（MultiIndex 先落平表列），写入后按上限淘汰旧文件。"""
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        flat = panel.reset_index()
        fd, tmp = tempfile.mkstemp(dir=str(_CACHE_ROOT), prefix=".panel.", suffix=".parquet.tmp")
        os.close(fd)
        try:
            flat.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        except BaseException:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # 写入成功后清理超出上限的旧文件
        _purge_old_cache(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存写入失败（不影响使用）: %s", exc)


def load_panel_from_cne(
    *,
    start: str | None = None,
    end: str | None = None,
    universe_mask: bool = False,
    include_fundamentals: bool = True,
    asset_type: str = "stock",
) -> pd.DataFrame:
    """从 CNE 数据湖实时构建 AlphaAgent panel（插件化）。

    Parameters
    ----------
    start, end : str | None
        日期范围（闭区间），传给各插件做日期过滤。
    universe_mask : bool
        是否启用 ST/停牌过滤。默认 False。
    include_fundamentals : bool
        是否加载 fundamental（PIT 基本面）插件。False 时跳过，
        避免最重的 join_asof 展开耗时（默认 True 保持向后兼容）。
    asset_type : str
        资产类型：'stock'（默认）/ 'etf'。
        - stock：加载全部插件（stock_daily_wide 为核心行情 + 辅助插件）。
        - etf：只加载 etf_bars 插件（跳过股票行情与基本面辅助插件）。
        ETF 无基本面/市值/估值列，评估 profile 会跳过市值类指标。

    Returns
    -------
    pd.DataFrame
        AlphaAgent panel（float32, MultiIndex(datetime, instrument)）。
        列数取决于已注册插件提供的列。

    首次构建后落盘缓存：key 按 (start, end, include_fundamentals, schema 版本)
    生成，命中条件为请求区间落在缓存面板实际覆盖范围内。因此历史固定区间
    不受数据湖 watermark 每日更新影响，长期稳定秒级命中；目录文件数由
    _CACHE_MAX_FILES 上限约束，不会无限累积。
    """
    cache_path = _cache_path(start, end, include_fundamentals)

    # 归一化日期参数为 YYYY-MM-DD 字符串：调用方（如 stacking 脚本）可能传
    # pd.Timestamp，核心行情插件能容忍，但 fundamental（date.fromisoformat）
    # 与 fund_flow（CNE reader 的日期比较）会静默失败 → 辅助插件整列丢失
    start = pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else None
    end = pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None

    if not universe_mask and asset_type == "stock":
        cached = _find_cached_panel(start, end, include_fundamentals)
        if cached is not None and not cached.empty:
            return cached

    registry = get_registry()
    logger.info("CNE adapter: building panel from %d plugins (start=%s end=%s include_fundamentals=%s asset_type=%s)",
                len(registry.list_plugins()), start, end, include_fundamentals, asset_type)

    # 确保 core 插件加载后补充 is_trade / not_st 标记列
    panel = registry.build_panel(
        start=start,
        end=end,
        universe_mask=universe_mask,
        include_fundamentals=include_fundamentals,
        asset_type=asset_type,
    )

    # 补充 core 插件特有的衍生标记列（is_trade, not_st）
    _enrich_trade_flags(panel)

    # 落盘缓存（universe_mask=True 的过滤结果不与通用缓存混用）
    if not universe_mask and asset_type == "stock":
        _save_cached_panel(cache_path, panel)

    logger.info("CNE adapter: panel shape=%s, columns=%d", panel.shape, panel.shape[1])
    return panel


def _enrich_trade_flags(panel: pd.DataFrame) -> None:
    """就地补充 is_trade / not_st 列（如果缺失）。

    这些列由 core 插件的列映射提供 is_st，但需要额外计算：
    - is_trade = volume > 0
    - not_st = 1 - is_st
    """
    if "is_trade" not in panel.columns and "volume" in panel.columns:
        panel["is_trade"] = (panel["volume"].fillna(0) > 0).astype("int8")

    if "not_st" not in panel.columns and "is_st" in panel.columns:
        # CNE is_st: 1=ST, 0=正常 → not_st = 1 - is_st
        panel["not_st"] = (1 - panel["is_st"].fillna(0).astype(int)).astype("int8")
    elif "not_st" not in panel.columns:
        panel["not_st"] = pd.Series(1, index=panel.index, dtype="int8")


def is_cne_source(panel_path: str | Path | None) -> bool:
    """判断 panel_path 是否指向 CNE 数据源。"""
    return panel_path is not None and str(panel_path) == CNE_SOURCE


def list_available_plugins() -> list[DataSourcePlugin]:
    """列出所有已注册的数据源插件（按优先级排序）。"""
    return get_registry().list_plugins()


def list_available_columns() -> list[str]:
    """列出所有插件提供的 Panel 列名。"""
    return get_registry().all_panel_columns()


def add_plugin(plugin: DataSourcePlugin, loader: Any) -> None:
    """运行时动态注册插件（高级用法，一般推荐通过 plugins/ 目录自动发现）。"""
    get_registry().register(plugin, loader)

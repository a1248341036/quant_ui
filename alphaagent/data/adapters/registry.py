"""数据源插件协议与注册中心。

设计要点
--------
1. 每个插件声明 ``name``、``dataset``（CNE 数据集名）、``join_keys``（索引列）
   和 ``column_map``（原始列 → Panel 列名映射）。
2. ``load()`` 返回原始 DataFrame（pandas），注册中心负责统一构建 Panel。
3. 插件自动发现：扫描 ``alphaagent/data/adapters/plugins/`` 下所有非下划线开头的
   ``.py`` 模块，导入后检查是否暴露 ``PLUGIN`` 实例。
4. Panel 构建时按插件优先级合并列：core 插件（行情）先加载，其他插件左 join。

加新数据源只需在 ``plugins/`` 下新增一个 ``.py`` 文件：

    from alphaagent.data.adapters.registry import DataSourcePlugin

    PLUGIN = DataSourcePlugin(
        name="my_new_data",
        dataset="my_cne_dataset",
        join_keys=("trade_date", "ts_code"),
        column_map={"raw_col": "panel_col", ...},
        priority=50,
    )

    def load(dataset, *, start=None, end=None, config=None):
        # 从 CNE 加载并返回 pandas DataFrame
        ...
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ─── 插件协议 ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DataSourcePlugin:
    """一个数据源插件的声明。

    Parameters
    ----------
    name : str
        插件唯一标识，用于日志与去重。
    dataset : str
        对应的 CNE 数据集名称（传给 ``cnequity.query.reader.load``）。
    join_keys : tuple[str, ...]
        与主表 join 的键列（如 ``("trade_date", "ts_code")``）。
        必须包含 datetime 键和 instrument 键。
    datetime_key : str
        join_keys 中哪个是日期列（原始名）。
    instrument_key : str
        join_keys 中哪个是标的列（原始名）。
    column_map : dict[str, str]
        原始列名 → Panel 列名映射。未映射的列将被丢弃。
    priority : int
        加载优先级（小先加载）。核心行情数据应为 0，补充数据 50+。
    raw_filter : str | None
        可选的 polars filter 表达式，在加载时应用。
    """

    name: str
    dataset: str
    join_keys: tuple[str, ...]
    datetime_key: str
    instrument_key: str
    column_map: dict[str, str] = field(default_factory=dict)
    priority: int = 50
    raw_filter: str | None = None

    def panel_columns(self) -> list[str]:
        """此插件映射后提供的 Panel 列名。"""
        return list(self.column_map.values())

    def is_core(self) -> bool:
        """是否为核心插件（提供行情数据，决定 Panel 的行索引）。"""
        return self.priority == 0


# ─── 加载函数签名 ──────────────────────────────────────────────────────

# 每个插件模块应暴露一个 ``load`` 函数：
# def load(dataset: str, *, start=None, end=None, config=None) -> pd.DataFrame
LoadFunc = Callable[..., pd.DataFrame]


# ─── 注册中心 ──────────────────────────────────────────────────────────


class PluginRegistry:
    """数据源插件注册中心。

    生命周期：
        1. 启动时 ``discover()`` 扫描 plugins/ 目录
        2. 按优先级排序
        3. ``build_panel()`` 时逐个调用 ``load()``，以 core 插件为基准左 join
    """

    def __init__(self) -> None:
        self._plugins: dict[str, DataSourcePlugin] = {}
        self._loaders: dict[str, LoadFunc] = {}

    # ── 注册 ──

    def register(
        self,
        plugin: DataSourcePlugin,
        loader: LoadFunc,
    ) -> None:
        if plugin.name in self._plugins:
            logger.warning("插件 %r 已注册，覆盖旧定义", plugin.name)
        self._plugins[plugin.name] = plugin
        self._loaders[plugin.name] = loader
        logger.info("注册数据源插件: %s (dataset=%s, %d 列, priority=%d)",
                    plugin.name, plugin.dataset, len(plugin.column_map), plugin.priority)

    # ── 发现 ──

    def discover(self, package: str = "alphaagent.data.adapters.plugins") -> int:
        """扫描包目录下所有非下划线 .py 模块，导入并收集 ``PLUGIN`` + ``load``。"""
        try:
            mod = importlib.import_module(package)
        except ImportError:
            return 0

        pkg_path = getattr(mod, "__path__", None)
        if not pkg_path:
            return 0
        root = Path(pkg_path[0])
        if not root.is_dir():
            return 0

        n = 0
        for path in sorted(root.glob("*.py")):
            stem = path.stem
            if stem == "__init__" or stem.startswith("_"):
                continue
            full_name = f"{package}.{stem}"
            try:
                m = importlib.import_module(full_name)
            except Exception as exc:
                logger.warning("跳过插件模块 %s: %s", full_name, exc)
                continue

            plugin = getattr(m, "PLUGIN", None)
            loader = getattr(m, "load", None)
            if plugin is None or loader is None:
                logger.debug("模块 %s 缺少 PLUGIN 或 load，跳过", full_name)
                continue
            if not isinstance(plugin, DataSourcePlugin):
                logger.warning("模块 %s 的 PLUGIN 不是 DataSourcePlugin 实例", full_name)
                continue
            if not callable(loader):
                logger.warning("模块 %s 的 load 不可调用", full_name)
                continue

            self.register(plugin, loader)
            n += 1

        return n

    # ── 查询 ──

    def list_plugins(self) -> list[DataSourcePlugin]:
        """按优先级排序返回所有已注册插件。"""
        return sorted(self._plugins.values(), key=lambda p: p.priority)

    def get(self, name: str) -> DataSourcePlugin | None:
        return self._plugins.get(name)

    def all_panel_columns(self) -> list[str]:
        """所有插件提供的 Panel 列名（去重、保持优先级顺序）。"""
        seen: set[str] = set()
        out: list[str] = []
        for plugin in self.list_plugins():
            for col in plugin.panel_columns():
                if col not in seen:
                    seen.add(col)
                    out.append(col)
        return out

    # ── Panel 构建 ──

    def build_panel(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        cne_root: Path | str | None = None,
        cne_config: Path | str | None = None,
        universe_mask: bool = False,
        include_fundamentals: bool = True,
        asset_type: str = "stock",
    ) -> pd.DataFrame:
        """从所有已注册插件加载数据，合并为统一 Panel。

        1. 加载 core 插件（priority=0），构建 (datetime, instrument) MultiIndex 基表
        2. 按优先级逐个加载其他插件，左 join 到基表
        3. 对核心列做衍生计算（adj_*, ret, vwap, label 等）

        include_fundamentals=False 时跳过 fundamental 插件（PIT 展开是全链路
        最重的 join_asof 广播），避免无谓耗时。

        asset_type='etf' 时只加载 etf_bars 插件（跳过股票行情与全部辅助插件）：
        ETF 域没有基本面/市值/行业/资金流等辅助列，评估 profile 会跳过这些指标。
        """
        from alphaagent.data.panel import build_panel_from_hq

        plugins = self.list_plugins()
        if not plugins:
            raise RuntimeError("无已注册的数据源插件")

        if asset_type == "etf":
            etf_plugins = [p for p in plugins if p.name == "etf_bars"]
            if not etf_plugins:
                raise RuntimeError("asset_type='etf' 但未发现 etf_bars 数据源插件")
            plugins = etf_plugins
            logger.info("asset_type=etf：仅加载插件 %s", [p.name for p in plugins])
        else:
            # stock 模式：排除 etf_bars（同为 priority=0 核心插件，列冲突）
            plugins = [p for p in plugins if p.name != "etf_bars"]
            logger.info("asset_type=%s：排除 etf_bars，加载插件 %s", asset_type, [p.name for p in plugins])

        core_plugins = [p for p in plugins if p.is_core()]
        aux_plugins = [p for p in plugins if not p.is_core()]
        if not include_fundamentals:
            skipped = [p.name for p in aux_plugins if p.name == "fundamental"]
            if skipped:
                logger.info("include_fundamentals=False，跳过辅助插件: %s", skipped)
            aux_plugins = [p for p in aux_plugins if p.name != "fundamental"]

        if not core_plugins:
            raise RuntimeError("无核心插件（priority=0），无法确定 Panel 行索引")

        # 1. 加载核心插件，构建 hq 基表
        hq: pd.DataFrame | None = None
        for plugin in core_plugins:
            logger.info("加载核心插件 %s (dataset=%s)", plugin.name, plugin.dataset)
            raw = self._call_loader(plugin, start=start, end=end,
                                    cne_root=cne_root, cne_config=cne_config)
            if raw is None or raw.empty:
                logger.warning("核心插件 %s 返回空数据", plugin.name)
                continue
            mapped = self._apply_column_map(raw, plugin)
            hq = self._merge_core(hq, mapped, plugin)
            if hq is None:
                continue

        if hq is None or hq.empty:
            raise RuntimeError("所有核心插件均返回空数据，无法构建 Panel")

        # 2. 加载辅助插件，左 join
        for plugin in aux_plugins:
            logger.info("加载辅助插件 %s (dataset=%s)", plugin.name, plugin.dataset)
            try:
                raw = self._call_loader(plugin, start=start, end=end,
                                        cne_root=cne_root, cne_config=cne_config)
            except Exception as exc:
                logger.warning("辅助插件 %s 加载失败: %s", plugin.name, exc)
                continue
            if raw is None or raw.empty:
                logger.debug("辅助插件 %s 无数据，跳过", plugin.name)
                continue
            mapped = self._apply_column_map(raw, plugin)
            hq = self._left_join(hq, mapped, plugin)

        # 3. 构建衍生列 + Panel
        logger.info("所有插件加载完成，hq shape=%s, 列数=%d", hq.shape, hq.shape[1])
        panel = build_panel_from_hq(hq, universe_mask=universe_mask)
        logger.info("Panel 构建完成: shape=%s", panel.shape)
        return panel

    # ── 内部方法 ──

    def _call_loader(
        self,
        plugin: DataSourcePlugin,
        *,
        start: str | None = None,
        end: str | None = None,
        cne_root: Path | str | None = None,
        cne_config: Path | str | None = None,
    ) -> pd.DataFrame | None:
        """调用插件的 load 函数，注入 CNE 配置。"""
        loader = self._loaders.get(plugin.name)
        if loader is None:
            return None

        # 统一注入 cne_root / cne_config（通过环境变量或闭包）
        kwargs: dict[str, Any] = {}
        if cne_root is not None:
            kwargs["cne_root"] = str(cne_root)
        if cne_config is not None:
            kwargs["cne_config"] = str(cne_config)

        try:
            return loader(plugin.dataset, start=start, end=end, **kwargs)
        except TypeError:
            # 插件 loader 可能不接受 cne_root/cne_config 参数
            return loader(plugin.dataset, start=start, end=end)

    @staticmethod
    def _apply_column_map(raw: pd.DataFrame, plugin: DataSourcePlugin) -> pd.DataFrame:
        """应用列名映射，保留 column_map 中的列 + join_keys。

        column_map 的 value（映射后的列名）也会保留，
        例如 is_st 映射为 is_st（identity mapping）也会保留。
        """
        keep = set(plugin.column_map.keys()) | set(plugin.join_keys)
        available = [c for c in keep if c in raw.columns]
        subset = raw[available].copy()
        # 重命名：原始列名 → Panel 列名
        rename = {k: v for k, v in plugin.column_map.items() if k in subset.columns}
        subset = subset.rename(columns=rename)
        return subset

    @staticmethod
    def _merge_core(
        base: pd.DataFrame | None,
        new: pd.DataFrame,
        plugin: DataSourcePlugin,
    ) -> pd.DataFrame:
        """合并核心插件数据：构建 (datetime, instrument) MultiIndex 基表。

        第一个核心插件建立基表，后续核心插件补充列（同索引左 join）。
        """
        dt_col = plugin.datetime_key
        inst_col = plugin.instrument_key

        # join_keys 在 _apply_column_map 中保留为原始列名（不在 column_map 中）
        if dt_col not in new.columns:
            raise ValueError(f"核心插件 {plugin.name} 缺少 datetime 列: {dt_col}")
        if inst_col not in new.columns:
            raise ValueError(f"核心插件 {plugin.name} 缺少 instrument 列: {inst_col}")

        # 构建 MultiIndex
        df = new.copy()
        dt = pd.to_datetime(df[dt_col])
        inst = df[inst_col].astype(str)

        # 数据列 = 除 join_keys 外的所有列
        data_cols = [c for c in df.columns if c not in (dt_col, inst_col)]
        result = df[data_cols].copy()
        result.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

        if base is None:
            return result

        # 后续核心插件：左 join 补充列（重名列加后缀防冲突）
        overlapping = set(base.columns) & set(result.columns)
        if overlapping:
            result = result.rename(columns={c: f"{c}__{plugin.name}" for c in overlapping})
        return base.join(result, how="left")

    @staticmethod
    def _left_join(
        base: pd.DataFrame,
        new: pd.DataFrame,
        plugin: DataSourcePlugin,
    ) -> pd.DataFrame:
        """辅助插件数据左 join 到基表。"""
        dt_col = plugin.datetime_key
        inst_col = plugin.instrument_key

        if dt_col not in new.columns or inst_col not in new.columns:
            logger.debug("辅助插件 %s 缺少 join 键列，跳过", plugin.name)
            return base

        df = new.copy()
        dt = pd.to_datetime(df[dt_col])
        inst = df[inst_col].astype(str)

        # 数据列 = 除 join_keys 外的所有列
        data_cols = [c for c in df.columns if c not in (dt_col, inst_col)]
        if not data_cols:
            return base

        right = df[data_cols].copy()
        right.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])
        right = right[~right.index.duplicated(keep="first")]

        # 左 join（重名列加后缀防冲突）
        overlapping = set(base.columns) & set(right.columns)
        if overlapping:
            right = right.rename(columns={c: f"{c}__{plugin.name}" for c in overlapping})

        return base.join(right, how="left")


# ─── 全局单例 ──────────────────────────────────────────────────────────

_global_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """获取全局插件注册中心（首次调用时自动发现插件）。"""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
        n = _global_registry.discover()
        logger.info("插件发现完成，共注册 %d 个数据源插件", n)
    return _global_registry


def reset_registry() -> None:
    """重置全局注册中心（测试用）。"""
    global _global_registry
    _global_registry = None

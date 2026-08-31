# -*- coding: utf-8 -*-
"""数据插件协议与注册中心(进程级缓存 + 点时切片)。

数据根目录:
- QDATA       data/quant_dataset              CNE 管理的 tushare wide 存档(日线)
- PG          data/pg_parquet                 CNE 注册外部集(财务/基本信息)
- CNE_CURATED CNEquity/.../_cnequity/curated  CNE 原生 curated 表

扩展方法: 在 plugins/ 下新增一个非下划线开头的 .py 文件:

    PLUGIN = JQDataPlugin(name=..., description=..., date_column=...,
                          entity_keys=(...))
    def load(...) -> pd.DataFrame: ...

即可通过 datalake.load(name)/asof(name, date) 使用; 未注册的数据集抛出
带扩展指引的 NotImplementedError, API 层据此给出清晰的"缺数据"报错。
"""
from __future__ import annotations

import dataclasses
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[4]
QDATA = ROOT / "data" / "quant_dataset"
PG = ROOT / "data" / "pg_parquet"
CNE_CURATED = ROOT / "CNEquity" / "data" / "quant_dataset" / "_cnequity" / "curated"

_PLUGINS: dict[str, JQDataPlugin] = {}
_CACHE: dict[tuple, pd.DataFrame] = {}


# ─── 插件协议 ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class JQDataPlugin:
    """一个数据集插件。

    name                唯一标识(如 "income" / "index_bars")
    description         一句话说明(进 status 诊断表)
    date_column         点时列名(trade_date/ann_date/as_of_date...);
                        None 表示无日期概念(不可 asof)
    entity_keys         asof 去重键: 每个实体取 date<=d 的最新一行
                        (如 ("code","system")); 空=只做日期过滤
    asof_fallback_first 请求日早于最早数据时, 回退取最早一期快照
                        (行业分类月度快照需要)
    load_fn             加载函数(无参或关键字参数); 缺省取模块级 load()
    """

    name: str
    description: str = ""
    date_column: str | None = None
    entity_keys: tuple[str, ...] = ()
    asof_fallback_first: bool = False
    load_fn: Callable[..., pd.DataFrame] | None = None


def register(plugin: JQDataPlugin) -> None:
    """注册/覆盖一个插件(外部扩展入口)。"""
    _PLUGINS[plugin.name] = plugin
    _CACHE.pop((plugin.name, None), None)


def discover() -> list[str]:
    """扫描 plugins/ 目录自动发现插件模块(模块级 PLUGIN 声明 + load 函数)。"""
    pkg_dir = Path(__file__).resolve().parent / "plugins"
    names: list[str] = []
    for f in sorted(pkg_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__package__}.plugins.{f.stem}")
        plugin = getattr(mod, "PLUGIN", None)
        if plugin is None:
            logger.warning("数据插件 %s 未声明 PLUGIN, 跳过", f.name)
            continue
        if plugin.load_fn is None:
            mod_load = getattr(mod, "load", None)
            if mod_load is None:
                logger.warning("数据插件 %s 未实现 load(), 跳过", f.name)
                continue
            plugin = dataclasses.replace(plugin, load_fn=mod_load)
        _PLUGINS[plugin.name] = plugin
        names.append(plugin.name)
    return names


# ─── 访问 API ───────────────────────────────────────────────────────────
def get(name: str, fresh: bool = False, **kwargs: Any) -> pd.DataFrame:
    """加载插件数据(进程级缓存; kwargs 参与缓存键)。"""
    plugin = _PLUGINS.get(name)
    if plugin is None:
        raise NotImplementedError(
            f"数据插件未注册: {name}。已注册: {sorted(_PLUGINS)}。"
            f"扩展方法: 在 core/event_engine/jq/datalake/plugins/ 下新增"
            f"插件文件(见本模块 docstring)")
    if plugin.load_fn is None:
        raise NotImplementedError(f"数据插件 {name} 未实现 load()")
    key = (name, tuple(sorted(kwargs.items())) or None)
    if fresh or key not in _CACHE:
        _CACHE[key] = plugin.load_fn(**kwargs)
    return _CACHE[key]


def asof(name: str, date) -> pd.DataFrame:
    """点时切片: date_column <= date, 按 entity_keys 每实体取最新一行。"""
    plugin = _PLUGINS[name]
    if plugin.date_column is None:
        raise NotImplementedError(f"数据插件 {name} 无日期列, 不支持 asof")
    df = get(name)
    if not len(df):
        return df
    d = pd.Timestamp(date)
    dates = pd.to_datetime(df[plugin.date_column])
    sub = df[dates <= d]
    if sub.empty:
        if not plugin.asof_fallback_first:
            return sub
        first = dates.min()
        return df[dates == first]
    if plugin.entity_keys:
        # 原表已按日期稳定排序 -> keep="last" 即每实体最新一行
        sub = sub.drop_duplicates(list(plugin.entity_keys), keep="last")
    return sub.reset_index(drop=True)


def available() -> list[str]:
    """已注册插件名单。"""
    return sorted(_PLUGINS)


def status() -> pd.DataFrame:
    """诊断表: 各插件行数/日期范围/内存占用(错误定位用)。"""
    rows = []
    for name in sorted(_PLUGINS):
        plugin = _PLUGINS[name]
        try:
            df = get(name)
            date_col = plugin.date_column
            if len(df) and date_col and date_col in df.columns:
                dts = pd.to_datetime(df[date_col])
                rng = f"{dts.min().date()}~{dts.max().date()}"
            else:
                rng = "-"
            rows.append({"plugin": name, "rows": len(df), "coverage": rng,
                         "MB": round(df.memory_usage(deep=True).sum() / 1e6, 1),
                         "description": plugin.description})
        except TypeError as exc:
            # load() 必传参数(如 stock_daily 的 start/end)未给 -> 按参数加载
            rows.append({"plugin": name, "rows": 0, "coverage": "按参数加载",
                         "MB": 0.0, "description": str(exc)[:80]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"plugin": name, "rows": -1, "coverage": "ERROR",
                         "MB": 0.0, "description": str(exc)[:80]})
    return pd.DataFrame(rows)


def clear_cache(name: str | None = None) -> None:
    """清缓存(测试/刷新用); name=None 清全部。"""
    if name is None:
        _CACHE.clear()
        return
    for key in [k for k in _CACHE if k[0] == name]:
        _CACHE.pop(key, None)


# 语义别名: load 与 get 同一入口(插件加载的主读接口)
load = get

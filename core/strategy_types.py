"""统一策略定义模型：三种来源（注册表 / 配置池 / 回测归档）收敛到同一形状。

历史上有三套并行的"策略定义"表达：
- strategies/registry.py：dict {factor, ascending, group, desc, types, 附加参数}
- core/strategy_pool.py 配置池：PG 行 {factor, ascending, params, group, desc, source}
- backtest_runs 归档：由历史回测参数反向提取的 params_summary

本模块定义 StrategyDefinition 作为唯一模型，并提供 fingerprint 用于
"同名策略是否为同一策略"的判定（策略池/归档去重、账户策略语义版本化）。

策略的执行形态不统一：因子轮动走 build_scores，事件策略走 on_bar，
这里只统一"定义 / 注册 / 解析 / 身份"，不统一执行器。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

StrategyKind = Literal["factor", "composite", "dsl", "event", "code"]


@dataclass(frozen=True)
class StrategyDefinition:
    """一份策略定义（与来源无关的规范形态）。

    kind:
      - factor：注册表/配置池的因子轮动策略（factor + ascending）
      - composite：多因子自由组合（factor_weights/factor_directions）
      - dsl：AlphaAgent 因子库 DSL（dsl_expr）
      - event：事件驱动策略类（module/class_name 交由加载器解析）
      - code：代码实验室自定义 build_factor_frames（builder 契约）
    params：策略自带的附加引擎参数（industry_cap/adx_filter/min_score/
            long_short/short_n/short_cost_rate/industry_neutral 等）。
    """

    id: str
    kind: StrategyKind
    display_name: str
    factor: str | None = None
    factor_weights: dict[str, float] | None = None
    factor_directions: dict[str, bool] | None = None
    dsl_expr: str | None = None
    ascending: bool = False
    params: dict = field(default_factory=dict)
    types: tuple[str, ...] = ("stock", "etf", "fund")
    group: str = "其他"
    desc: str = ""
    source: str = "registry"  # registry / pool / archive / dsl / code / event
    version: int = 1

    def to_dict(self) -> dict:
        """兼容旧 resolve_strategy 的 dict 形状（老调用方 strat["factor"] 等）。"""
        return {
            "factor": self.factor,
            "ascending": self.ascending,
            "group": self.group,
            "desc": self.desc,
            "types": list(self.types),
            "kind": self.kind,
            "source": self.source,
            "version": self.version,
            **self.params,
        }

    def fingerprint(self) -> str:
        """规约化参数签名：同指纹 = 同一策略（用于去重/冲突检测）。"""
        canonical = {
            "kind": self.kind,
            "factor": self.factor or "",
            "ascending": bool(self.ascending),
            "factor_weights": _sorted_dict(self.factor_weights or {}),
            "factor_directions": _sorted_dict(self.factor_directions or {}),
            "dsl_expr": self.dsl_expr or "",
            "params": _sorted_dict(self.params),
        }
        payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _sorted_dict(d: dict) -> dict:
    return {str(k): _jsonable(v) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}


def _jsonable(v):
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, dict):
        return _sorted_dict(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


# 旧 dict 里的元数据键（排除后剩余键都视为附加引擎参数，保持旧 resolve_strategy
# "扁平展开所有参数"的语义，避免白名单漏参数）。
_META_KEYS = {
    "name", "factor", "ascending", "group", "desc", "types",
    "factor_weights", "factor_directions",
}


def from_legacy_dict(
    name: str,
    d: dict,
    *,
    source: str = "registry",
    default_group: str = "其他",
) -> StrategyDefinition:
    """从旧 dict 形状（registry 项 / pool_def / archive 行）构造定义。

    default_group：不同来源的旧默认分组不同（pool="配置池"、
    archive="回测历史"、registry 项自带 group），调用方按来源传入。
    """
    params = {
        k: v for k, v in d.items()
        if k not in _META_KEYS and v is not None
    }
    return StrategyDefinition(
        id=name,
        kind="composite" if d.get("factor_weights") else "factor",
        display_name=str(d.get("name") or name),
        factor=d.get("factor"),
        factor_weights=d.get("factor_weights"),
        factor_directions=d.get("factor_directions"),
        ascending=bool(d.get("ascending", False)),
        params=params,
        types=tuple(d.get("types") or ("stock", "etf", "fund")),
        group=str(d.get("group") or default_group),
        desc=str(d.get("desc") or ""),
        source=source,
    )


def from_dsl_factor(
    factor_id: str,
    *,
    name: str,
    dsl_expr: str,
    ascending: bool = False,
    params: dict | None = None,
    source: str = "dsl",
) -> StrategyDefinition:
    """把 AlphaAgent 因子库因子动态构造成策略定义（不回填策略注册表）。

    DSL 因子与注册表策略是两套体系：这里只在需要"把 DSL 因子当策略跑
    回测/信号"时做一次性解析，不产生持久化。
    """
    return StrategyDefinition(
        id=factor_id,
        kind="dsl",
        display_name=str(name or factor_id),
        dsl_expr=str(dsl_expr),
        ascending=bool(ascending),
        params=dict(params or {}),
        types=("stock",),
        group="AlphaAgent 因子",
        desc=f"ALPHA DSL 因子 {factor_id}",
        source=source,
    )
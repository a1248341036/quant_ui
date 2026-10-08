# -*- coding: utf-8 -*-
"""股票池（universe）维度：把因子评估限定在指定股票池内。

**池子是评估口径，不是因子属性**（2026-10-08 定调）。同一因子在不同股票池表现可以
完全不同（实测：同一批研报因子在"全市场"与"市值 Top300"下 PE 因子 IC 从 +0.0092
变 −0.0132）。因此：

1. 因子表达式**保持池子无关**（池子不进 DSL，避免污染正交性/血缘比较）；
2. 会话/请求级参数 ``universe`` 指定池子，**评估结果与候选元数据记录
   ``evaluated_universe``** —— 同一因子跨池评估即得"池子敏感性"结论。

语法
----
| 值 | 含义 | 数据依赖 |
|---|---|---|
| ``all`` | 全市场（默认，行为与历史完全一致） | 无 |
| ``top<N>cap`` | 每日总市值前 N（``top300cap`` / ``top500cap`` / ``top1000cap``） | 面板 ``tot_cap`` |
| ``mid<A>_<B>cap`` | 每日总市值第 A~B 名（``mid301_800cap``，与"HS300=1~300、中证500=301~800"惯例一致） | 面板 ``tot_cap`` |
| ``szcomp`` | 深证成指成分 | 面板 ``idx_szcomp``（2021-12 起） |
| ``chinext`` | 创业板指成分 | 面板 ``idx_chinext``（2021-12 起） |

语义：**非成分股票整行剔除**，因此 IC / 覆盖率 / 换手 / 分组 / 引擎回测全部在池内计算。
（不采用"池外置 NaN"的写法：那样覆盖率与换手仍按全市场统计，口径会自相矛盾。）

注意：规则池用 ``tot_cap`` 每日排名近似"大盘/中盘/小盘"，**不等于指数成分**（指数有
自由流通市值加权、分级靠档与半年调样）。真实指数成分历史见 CNE ``index_constituents``
（现仅深证成指/创业板指），HS300/中证500 成分历史需另行接入。
当日 ``tot_cap`` 为 NaN 的股票无法参与排名，与指数池（非成员填 0）一致**按非成员剔除**。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

UNIVERSE_ALL = "all"

# 规则池：top<N>cap / mid<A>_<B>cap
_CAP_RE = re.compile(r"^(?P<kind>top|mid)(?P<a>\d+)(?:_(?P<b>\d+))?cap$")
# 指数池：面板 1/0 标志列
_FLAG_POOLS: dict[str, str] = {
    "szcomp": "idx_szcomp",
    "chinext": "idx_chinext",
}


def available_universes() -> list[str]:
    """内置池清单（错误提示与 ``--help`` 用）。"""
    return [UNIVERSE_ALL, "top300cap", "top500cap", "top1000cap", "mid301_800cap",
            *_FLAG_POOLS.keys()]


def parse_universe(name: str) -> dict[str, Any]:
    """解析池子名 → 规格字典；非法名抛 ``ValueError``（附可用清单）。"""
    raw = (name or UNIVERSE_ALL).strip().lower()
    if raw == UNIVERSE_ALL:
        return {"kind": "all", "name": UNIVERSE_ALL}

    if raw in _FLAG_POOLS:
        return {"kind": "flag", "name": raw, "column": _FLAG_POOLS[raw]}

    m = _CAP_RE.match(raw)
    if m:
        kind, a = m.group("kind"), int(m.group("a"))
        b = int(m.group("b")) if m.group("b") else None
        if a <= 0:
            raise ValueError(f"非法股票池 {name!r}: 排名须为正整数")
        if kind == "top":
            if b is not None:
                raise ValueError(f"非法股票池 {name!r}: top<N>cap 不应带下界")
            return {"kind": "cap_top", "name": raw, "n": a}
        if b is None or b <= a:
            raise ValueError(f"非法股票池 {name!r}: mid<A>_<B>cap 需满足 B > A")
        return {"kind": "cap_mid", "name": raw, "a": a, "b": b}

    raise ValueError(
        f"未知股票池 {name!r}；可用：{', '.join(available_universes())}"
        "（规则池 top<N>cap / mid<A>_<B>cap 可自填数字）"
    )


def describe_universe(name: str) -> str:
    """人类可读描述（日志与元数据用）。"""
    spec = parse_universe(name)
    if spec["kind"] == "all":
        return "全市场"
    if spec["kind"] == "flag":
        return f"指数成分（{spec['column']}）"
    if spec["kind"] == "cap_top":
        return f"每日总市值前 {spec['n']}"
    return f"每日总市值第 {spec['a']}~{spec['b']} 名"


def apply_universe(panel: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按池子掩码面板；返回 (掩码后面板, 元数据)。

    非成分股票**整行剔除**。``all`` 原样返回（零开销、行为与历史一致）。
    """
    spec = parse_universe(name)
    before = int(len(panel))
    meta: dict[str, Any] = {
        "universe": spec["name"],
        "universe_desc": describe_universe(spec["name"]),
        "rows_before": before,
    }
    if spec["kind"] == "all":
        meta.update({"rows_after": before, "stocks_per_day": None, "dropped_rows": 0})
        return panel, meta

    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("apply_universe 需要 (datetime, instrument) MultiIndex 面板")

    if spec["kind"] == "flag":
        col = spec["column"]
        if col not in panel.columns:
            raise ValueError(
                f"股票池 {spec['name']!r} 需要面板列 {col!r}，当前面板没有该列"
                "（该列由 index_membership 插件提供，需重建面板或改用规则池）"
            )
        keep = pd.to_numeric(panel[col], errors="coerce").fillna(0).to_numpy() > 0
    else:
        if "tot_cap" not in panel.columns:
            raise ValueError(f"股票池 {spec['name']!r} 需要面板列 'tot_cap'，当前面板没有该列")
        rank = panel.groupby(level=0, sort=False)["tot_cap"].rank(ascending=False, method="first")
        if spec["kind"] == "cap_top":
            keep = (rank <= spec["n"]).to_numpy()
        else:
            keep = ((rank >= spec["a"]) & (rank <= spec["b"])).to_numpy()

    out = panel.loc[keep]
    if out.empty:
        raise ValueError(f"股票池 {spec['name']!r} 掩码后无数据（检查 tot_cap/成分列覆盖与窗口）")
    per_day = out.groupby(level=0, sort=False).size()
    meta.update({
        "rows_after": int(len(out)),
        "dropped_rows": before - int(len(out)),
        "stocks_per_day": float(per_day.median()),
    })
    return out, meta

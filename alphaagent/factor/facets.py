"""数据面（facet）识别与聚焦范围计算（纯函数，无 mining 依赖）。

data 层（cnequity 等）与 mining 层（expressions/scope_filter 等）共用，
避免 data → factor.mining 的模块边界倒置。mining 侧在
``alphaagent.factor.mining.memory.expressions`` re-export 保持兼容。
"""

from __future__ import annotations

# ── 数据面识别（跨面融合引导用）────────────────────────────────
# 面板实际接入的数据列族（与 data/adapters/plugins 一致）。每个可独立聚焦的
# 数据源/列族对应一个面，前端 chips 与 run 表单直接读取本表：
# - 价量/量能/筹码/拥挤（stock_daily_wide + CHIP_*/CROWD_* 派生）
# - 基本面(funda_*) / 股东面(holder_* 股东户数) / 机构面(inst_* 机构持仓) /
#   股东集中面(th_* 十大流通股东)
# - 资金面(ff_* 主力资金流) / 两融面(mgn_* 融资融券)
# - 事件面(dt_*/bt_* 龙虎榜/大宗) / 业绩面(pred_*/exp_* 业绩预告/快报) /
#   披露面(ds_* 披露日历) / 分红面(div_* 现金分红)
FACET_DEFS: list[tuple[str, tuple[str, ...]]] = [
    ("价量面", ("$adj_", "$close", "$open", "$high", "$low", "$ret", "$vwap")),
    ("量能面", ("$volume", "$amount", "$turnover")),
    ("筹码面", ("chip_",)),
    ("拥挤面", ("crowd_",)),
    ("基本面", ("funda_",)),
    ("股东面", ("holder_",)),
    ("机构面", ("$inst_",)),
    ("股东集中面", ("$th_",)),
    ("资金面", ("$ff_", "$inflow", "$outflow")),
    ("两融面", ("$mgn_",)),
    ("事件面", ("$dt_", "$bt_")),
    ("业绩面", ("$pred", "$exp")),
    ("披露面", ("$ds_",)),
    ("分红面", ("$div_",)),
]

# 数据源分组：融合（family 用面对组合键）只在跨组时成立。
# 同组多面（如 价量面+量能面 都来自 stock_daily_wide 行情面板）是普通单因子
# 的常见形态，判成"融合"会把族分类从细粒度退化为粗粒度对键，污染记忆桶。
# $float_cap 是行情面板列，已从股东面识别键移出，不参与面判定。
FACET_GROUPS: dict[str, tuple[str, ...]] = {
    "行情组": ("价量面", "量能面", "筹码面", "拥挤面"),
    "基本面组": ("基本面", "股东面", "机构面", "股东集中面"),
    "事件资金组": ("资金面", "两融面", "事件面", "业绩面", "披露面", "分红面"),
}

# 面 → 该面专属算子在实现上必然消费的输入列族。
# 依据算子签名（2026-09 核对）：CHIP_*(close, low, high, volume, …) 需要行情组的
# 价量+量能列；VOLUME_CLOCK_VPIN(price, volume)/MUTUAL_INFO_LAG(close, volume) 需要
# 价格序列；CROWD_* 的 dimension/attribute 是任意输入序列，常见为量价。
# 这些输入列在聚焦时一并放行，但表达式仍必须触及勾选的聚焦面本身
# （见 facet_scope_violation 的"必须触及"规则）——否则纯价量因子会借着
# "输入列放行"混进筹码/拥挤 run。
FACET_IMPLIED_INPUTS: dict[str, tuple[str, ...]] = {
    "筹码面": ("价量面", "量能面"),
    "拥挤面": ("价量面", "量能面"),
    "量能面": ("价量面",),
}

_FACET_ORDER: dict[str, int] = {name: i for i, (name, _) in enumerate(FACET_DEFS)}
_FACET_TO_GROUP: dict[str, str] = {
    facet: group for group, facets in FACET_GROUPS.items() for facet in facets
}


def expr_facets(expression: str) -> set[str]:
    """识别一个表达式触及的数据面（按列/算子前缀匹配）。"""
    low = str(expression or "").lower()
    out: set[str] = set()
    for name, keys in FACET_DEFS:
        if any(k.lower() in low for k in keys):
            out.add(name)
    return out


def facet_column_hint(facet_name: str) -> str:
    """给 LLM/人看的列族速记：价量面 → '$close/$open/…'，两融面 → '$mgn_*'。"""
    for name, keys in FACET_DEFS:
        if name == facet_name:
            parts = [(k + "*" if k.endswith("_") else k) for k in keys]
            return "、".join(parts) if parts else ""
    return ""


def facet_allowed_scope(focus_facets) -> set[str]:
    """聚焦 run 的可用面范围 = 勾选面 ∪ 这些面的算子隐含输入面。"""
    focus = {str(f) for f in (focus_facets or ()) if f}
    extra: set[str] = set()
    for facet in focus:
        extra |= set(FACET_IMPLIED_INPUTS.get(facet, ()))
    return focus | extra


def facet_groups(facets: set[str]) -> set[str]:
    """一个面集合覆盖的数据源组。"""
    return {_FACET_TO_GROUP[f] for f in facets if f in _FACET_TO_GROUP}


def is_cross_group_fusion(facets: set[str]) -> bool:
    """≥2 个面且跨数据源组才算融合（同组多面不算，防线见 FACET_GROUPS 注释）。"""
    return len(facets) >= 2 and len(facet_groups(facets)) >= 2


def fusion_family_key(facets: set[str]) -> str:
    """融合因子的族键：面名按 FACET_DEFS 稳定排序 × 连接（如 基本面×价量面）。"""
    ordered = sorted(
        (f for f in facets if f in _FACET_ORDER),
        key=lambda f: _FACET_ORDER[f],
    )
    return "×".join(ordered)
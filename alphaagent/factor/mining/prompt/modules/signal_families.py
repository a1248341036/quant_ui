# -*- coding: utf-8 -*-
"""模块 06.5 · signal_families：信号族白名单（ResearchSpec → 有效约束）。

**为什么需要这个模块（2026-10-19）**
``research_spec.search_policy.allowed_signal_families`` 此前只在**旧 CLI 通路**
（``scripts/alphaagent_factor_mining.py`` 用 ``extra_instructions=research_policy_prompt(...)``）
被渲染成文本；Web/agentscope 通路的 system prompt 模块表里**没有任何模块消费它**
——即该配置在主力链路上是死配置。后果：往注册表里新加族名对运行时零影响，
"放开某个数据面"只能靠改文案，无法通过 spec 生效。

本模块把它接进模块装配（ORDER 介于 strategy_tracks 与 data_fields 之间），
并解决族名的第二处歧义：族名是英文短名（``capital_flow``），而提示词正文的字段
说明是中文区块（``$ff_*`` 资金流）——两者此前**没有任何映射表**，LLM 只能猜。
这里给每个族配上「中文名 + 字段族提示」，并按 panel 实际列标注可用性。

**兼容性（黄金基线）**：仅当白名单里出现「经典族之外」的族时才注入。
经典族 = ``_LEGACY_FAMILIES``（technical/fundamental/report 各档沿用的那批），
它们的历史行为由其它板块承担，本模块对老档**逐字节无影响**。
"""
from __future__ import annotations

NAME = "signal_families"
TITLE = "信号族白名单（ResearchSpec 允许/禁止族）"
ORDER = 26
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"

# 老档沿用的族：出现它们**不触发**本模块（保持黄金基线逐字节不变）。
# 新增族必须同时在此表之外的 FAMILY_FIELD_HINTS 里登记，否则渲染时会被忽略。
_LEGACY_FAMILIES: frozenset[str] = frozenset({
    "volume_price", "volatility", "chip", "momentum_reversal",
    "fundamental_quality", "fundamental_growth",
    "fundamental_value", "fundamental_revision",
})

# 族名 → (中文名, 字段族提示, 可用性探测前缀)
# 探测前缀用于判断"本次面板是否真的载入了该族"——教了不可用的字段会白烧评估预算
# （2026-10-19 实测：$ac_* 一致预期覆盖率仅 0.85%，但提示词一直在教）。
FAMILY_FIELD_HINTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "volume_price": ("价量", "$adj_*/$close/$open/$high/$low/$vwap/$volume/$amount", ("adj_close", "close")),
    "volatility": ("波动", "$adj_high/$adj_low/$ret 的 TS_STD/TS_AMP*/振幅类算子", ("adj_high", "ret")),
    "chip": ("筹码", "CHIP_* 算子族（由 $close/$low/$high/$volume 派生的成本分布）", ("close",)),
    "momentum_reversal": ("动量反转", "$ret/$adj_close 的 DELTA/TS_PCTCHANGE/TS_RANK/TS_TREND_RANK", ("ret", "adj_close")),
    "fundamental_quality": ("基本面·质量", "$funda_roe/$funda_roic/$funda_gross_margin/$funda_ocf/$funda_total_assets", ("funda_roe", "fsi_total_assets")),
    "fundamental_growth": ("基本面·成长", "$funda_netprofit_yoy/$funda_or_yoy/$funda_ocf_yoy 与 $delta 派生", ("funda_netprofit_yoy",)),
    "fundamental_value": ("基本面·估值", "$pe_ttm/$pb/$ps_ttm/$dv_ttm", ("pe_ttm", "pb")),
    "fundamental_revision": ("基本面·预期修正", "$pred_surprise/$exp_netprofit_yoy/$rc_eps_y（预期变化类）", ("pred_surprise", "rc_eps_y")),
    # ── 2026-10-19 新增族（数据早已在面板，此前无族可挂）──
    "capital_flow": ("资金面", "$ff_main_net/$ff_super_net/$ff_large_net/$ff_medium_net/$ff_small_net（主力-散户资金分歧）", ("ff_super_net", "ff_main_net")),
    "margin": ("两融", "$mgn_balance/$mgn_buy/$mgn_short_balance/$mgn_short_sell_vol（杠杆拥挤度）", ("mgn_balance", "mgn_buy")),
    "institutional": ("机构持仓", "$inst_*（合计 + 基金/QFII/社保/保险/券商/信托分型 count/ratio/mv）", ("inst_ratio", "inst_fund_ratio")),
    "shareholder_concentration": ("股东集中", "$holder_count_chg_pct/$holder_avg_value/$th_top10_pct/$th_inst_pct（筹码集中度）", ("holder_count_chg_pct", "th_top10_pct")),
    "corporate_event": ("事件", "$dt_*/$bt_*（龙虎榜/大宗）+ $unlock_*（限售解禁供给冲击）+ $ann_*（增减持/回购/质押/问询）", ("dt_cnt_90d", "unlock_days_since", "ann_cnt_20d")),
    "expectation": ("业绩与卖方预期", "$pred_*/$exp_*（预告/快报）+ $rc_*/$ac_*（研报预测，ac_* 覆盖率 <1% 慎用）", ("pred_surprise", "rc_cnt_90d")),
    "disclosure_dividend": ("披露与分红", "$ds_*（披露延迟节奏）+ $div_*（已实施分红/除息日）", ("ds_days_since_actual", "div_cash_div")),
}

# 禁止族 → 原因（提示词里要给出替代方向，否则 LLM 只是换个壳继续做）
_FORBIDDEN_REASONS: dict[str, str] = {
    "pure_size": "纯市值/规模暴露（RANK($float_cap) 之类）不是 alpha，且与已有因子高相关",
    "pure_price_momentum": "裸动量/裸反转已被价量族占满，单独成因子必然撞正交门",
}


def _panel_has(cols, prefixes: tuple[str, ...]) -> bool:
    """面板是否载入该族（cols 为 None 时视为未知 → 一律显示，不误伤）。"""
    if cols is None:
        return True
    present = {str(c) for c in cols}
    return any(p in present for p in prefixes)


def _spec(ctx):  # noqa: ANN001
    return getattr(ctx, "research_spec", None) or {}


def enabled(ctx) -> bool:  # noqa: ANN001
    # 聚焦 run 由 facet_focus 硬锁定面并显式豁免白名单（见该模块末尾"豁免声明"），
    # 此处再渲染一次白名单会与聚焦指令冲突。
    if getattr(ctx, "focus_facets", None):
        return False
    search = _spec(ctx).get("search_policy") or {}
    allowed = [str(f) for f in (search.get("allowed_signal_families") or []) if str(f)]
    # 只有出现"经典族之外"的族才注入 → 老档 prompts 逐字节不变（黄金基线）。
    return any(f not in _LEGACY_FAMILIES for f in allowed)


def render(ctx) -> str:  # noqa: ANN001
    search = _spec(ctx).get("search_policy") or {}
    allowed = [str(f) for f in (search.get("allowed_signal_families") or []) if str(f)]
    forbidden = [str(f) for f in (search.get("forbidden_signal_families") or []) if str(f)]
    cols = getattr(ctx, "panel_columns", None)

    avail: list[str] = []
    missing: list[str] = []
    unknown: list[str] = []
    for fam in allowed:
        hint = FAMILY_FIELD_HINTS.get(fam)
        if hint is None:
            # 未登记的族：只报名字，不编字段（宁可少说，不可教错列名）
            unknown.append(fam)
            continue
        zh, fields, prefixes = hint
        line = f"- `{fam}`（{zh}）：{fields}"
        (avail if _panel_has(cols, prefixes) else missing).append(line)

    lines = [
        "## 信号族白名单（本次运行 ResearchSpec 约束，硬性）",
        "本轮候选的**信号主体**必须落在下列允许族之内；族是按「数据面」划分的信息来源，"
        "不是算子分类——同一族内可自由换算子/窗口，跨族组合按结构化交互规则处理。",
    ]
    if avail:
        lines.append("**允许的信号族（本次面板已载入）**：")
        lines.extend(avail)
    if missing:
        lines.append("**允许但本次面板未载入的族**（列不存在，引用会报「不可用字段」，不要尝试）：")
        lines.extend(missing)
    if unknown:
        lines.append("- 未登记字段提示的族：" + "、".join(f"`{f}`" for f in unknown) + "。")
    if forbidden:
        reasons = "；".join(
            f"`{f}`（{_FORBIDDEN_REASONS.get(f, '无独立信息，禁止作为信号主体')}）" for f in forbidden
        )
        lines.append(f"**禁止作为独立候选的信号族**：{reasons}。")
    lines.append(
        "跨族融合不受此表限制，但必须触及至少一个**允许族**作为信号主体："
        "把禁止族/未载入族当主信号、允许族只做门控壳，视为违规。"
    )
    return "\n".join(lines)

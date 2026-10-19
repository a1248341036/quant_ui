# -*- coding: utf-8 -*-
"""事件/资金数据面解锁（2026-10-19）回归。

背景（体检结论）：面板 161 列里资金面 99.9%、机构面 96.2%、股东集中面 96.3%、
股东户数 99.6%、业绩预告 90.4%、披露 96.6% 覆盖的字段早已载入，但 11 个研究档位的
signal_families 只覆盖「价量/波动/筹码 + 基本面」，没有族能挂这些字段 —— 52 条库内
因子里 12 个事件/资金族字段合计仅被引用 15 次，6 个数据面的在库因子数为 0。

本组测试锁定四件事：
1. ``facets`` 面识别修正：``$float_cap``/``$tot_cap`` 不再属基本面面（曾误判 36/52 条
   纯价量因子）；``unlock_*`` 归事件面（否则聚焦 run 整族被裁掉）。
2. 新档 ``technical_event_monthly``：同 label_20d ⇒ 同门槛，仅族与冷门赛道不同；
   **其余档位 cold_track 必须仍为 False**（逐字节不变）。
3. ``infer_research_mode``：勾选事件/资金面不再静默落到 technical_monthly。
4. ``signal_families`` 提示词模块：把 allowed_signal_families 从死配置变成活约束，
   且**老档 prompt 逐字节不变**（只有出现新族才注入）。
"""
from __future__ import annotations

import re
from pathlib import Path

from alphaagent.factor.facets import expr_facets
from alphaagent.factor.mining.prompt.modules import signal_families as sf_mod
from alphaagent.factor.mining.prompts import build_system_prompt
from alphaagent.factor.mining.research_spec import (
    build_run_research_spec,
    default_research_spec,
)
from core.research_modes import infer_research_mode, mode_ids

# 面板列子集：价量 + 事件/资金/机构/股东/预期各族哨兵列
_BASE_COLS = [
    "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close",
    "volume", "amount", "float_cap", "tot_cap", "vwap", "adj_vwap", "ret",
    "is_trade", "not_st", "industry_sw_l1", "label_20d_close_to_close",
]
_EVENT_COLS = [
    "ff_main_net", "ff_super_net", "ff_large_net", "ff_medium_net", "ff_small_net",
    "mgn_balance", "mgn_buy", "mgn_short_balance", "mgn_short_sell_vol",
    "inst_count", "inst_ratio", "inst_mv", "inst_fund_count", "inst_fund_ratio",
    "inst_fund_mv", "inst_qfii_ratio", "inst_social_security_ratio",
    "holder_count", "holder_count_chg_pct", "holder_avg_value",
    "th_top1_pct", "th_top10_pct", "th_inst_pct",
    "dt_cnt_90d", "dt_net_buy_90d", "bt_amt_90d", "bt_premium_last",
    "unlock_days_since", "unlock_ratio_last", "unlock_ratio_past180",
    "unlock_days_to_next", "unlock_ratio_next",
    "ann_cnt_20d", "ann_flag_reduce", "pred_surprise", "exp_netprofit_yoy",
    "ds_days_since_actual", "div_cash_div",
]


def _prompt(*, spec=None, cols=None, focus=None, include_fundamentals=False) -> str:
    return build_system_prompt(
        include_operator_catalog=False,
        label_col="label_20d_close_to_close",
        include_fundamentals=include_fundamentals,
        panel_columns=list(_BASE_COLS if cols is None else cols),
        population_max=0,
        research_spec=spec,
        asset_type="stock",
        focus_facets=focus,
        prompt_phase="full",
    )


# ── 1. 面识别修正 ─────────────────────────────────────────────────────

class TestFacetRecognition:
    def test_float_cap_is_not_fundamental(self):
        """曾把 36/52 条纯价量因子误判成"触及基本面面"的根因。"""
        assert expr_facets("$float_cap") == set()
        assert expr_facets("LOG($tot_cap)") == set()
        # 典型 VWAP 价量因子：过去会被打上"基本面"标签
        facets = expr_facets("CS_NEUTRALIZE(NEG($ret), CS_BUCKET(LOG($float_cap), 10))")
        assert "基本面" not in facets
        assert "价量面" in facets

    def test_real_fundamental_still_recognized(self):
        assert "基本面" in expr_facets("DIVIDE($funda_ocf, $funda_total_assets)")
        assert "基本面" in expr_facets("$pe_ttm")

    def test_unlock_belongs_to_event_facet(self):
        """unlock_* 此前不命中任何面 → 聚焦 run 的列过滤会整族丢掉。"""
        for col in ("unlock_days_to_next", "unlock_ratio_past180", "unlock_ratio_next"):
            assert "事件面" in expr_facets(f"${col}")


# ── 2. 新档与冷门赛道 ────────────────────────────────────────────────

class TestEventResearchMode:
    def test_mode_registered(self):
        assert "technical_event_monthly" in mode_ids()

    def test_event_mode_spec(self):
        spec = default_research_spec("technical_event_monthly")
        assert spec["research_mode"] == "technical_event_monthly"
        assert spec["recommended_label_col"] == "label_20d_close_to_close"
        # 同 label_20d ⇒ 同门槛（2026-10-08 定调：门槛只由 label 决定）
        monthly = default_research_spec("technical_monthly")
        for key in ("min_train_abs_ic", "min_train_icir", "min_val_abs_ic"):
            assert (
                spec["delivery_policy"]["production"][key]
                == monthly["delivery_policy"]["production"][key]
            ), key
        assert spec["delivery_policy"]["production"]["engine_gate"]["freq"] == "monthly"
        # 冷门赛道：仅本档打开
        assert spec["cold_track"]["enabled"] is True

    def test_event_mode_signal_families_include_new_facets(self):
        families = set(default_research_spec("technical_event_monthly")["search_policy"]["allowed_signal_families"])
        for fam in (
            "capital_flow", "margin", "institutional",
            "shareholder_concentration", "corporate_event",
            "expectation", "disclosure_dividend",
        ):
            assert fam in families, fam
        # 行情族仍保留（作为门控/交互腿）
        assert "volume_price" in families

    def test_other_modes_keep_cold_track_off(self):
        """回归保护：任何既有档位不得被本改动打开冷门赛道。"""
        for mode in mode_ids():
            if mode == "technical_event_monthly":
                continue
            assert default_research_spec(mode)["cold_track"]["enabled"] is False, mode

    def test_build_run_spec_accepts_mode(self):
        built = build_run_research_spec({"research_mode": "technical_event_monthly"})
        assert built["cold_track"]["enabled"] is True
        assert "capital_flow" in built["search_policy"]["allowed_signal_families"]


# ── 3. 档位自动推断 ──────────────────────────────────────────────────

class TestInferResearchMode:
    def test_event_facets_route_to_event_mode(self):
        assert infer_research_mode(["资金面"], "monthly") == "technical_event_monthly"
        assert infer_research_mode(["事件面"], None) == "technical_event_monthly"
        assert infer_research_mode(["两融面"], "monthly") == "technical_event_monthly"
        assert infer_research_mode(["披露面"], None) == "technical_event_monthly"

    def test_daily_weekly_fall_back_to_technical_buckets(self):
        """新档只有 20d 门槛标定；daily/weekly 不得硬套（否则同 label 不同门槛）。"""
        assert infer_research_mode(["资金面"], "daily") == "technical_daily"
        assert infer_research_mode(["事件面"], "weekly") == "technical_weekly"

    def test_existing_routing_unchanged(self):
        assert infer_research_mode(["价量面"], "monthly") == "technical_monthly"
        assert infer_research_mode(["价量面"], None) == "technical"
        assert infer_research_mode(["基本面"], "monthly") == "fundamental"
        assert infer_research_mode(["机构面"], "weekly") == "fundamental_weekly"
        assert infer_research_mode(None, None) == "technical"


# ── 4. signal_families 提示词模块 ────────────────────────────────────

class TestSignalFamiliesPromptModule:
    def test_disabled_without_new_families(self):
        """纯经典族的档位不得注入 → 黄金基线逐字节不变。"""
        for mode in ("technical", "technical_monthly", "report"):
            spec = default_research_spec(mode)
            assert sf_mod.enabled(_ctx(spec)) is False, mode
            assert "信号族白名单" not in _prompt(spec=spec, cols=_BASE_COLS + _EVENT_COLS), mode

    def test_fundamental_mode_injects_institutional_families(self):
        """fundamental 档含 institutional/shareholder_concentration（基本面组补齐）
        → 会注入，且必须显式列出这两族（否则"勾机构面却不知道有机构族"仍可复现）。"""
        spec = default_research_spec("fundamental")
        assert sf_mod.enabled(_ctx(spec)) is True
        text = _prompt(spec=spec, cols=_BASE_COLS + _EVENT_COLS, include_fundamentals=True)
        assert "信号族白名单" in text
        assert "`institutional`" in text
        assert "`shareholder_concentration`" in text

    def test_injected_for_event_mode(self):
        spec = default_research_spec("technical_event_monthly")
        text = _prompt(spec=spec, cols=_BASE_COLS + _EVENT_COLS)
        assert "信号族白名单" in text
        assert "`capital_flow`" in text
        assert "资金面" in text
        # 禁止族要带原因与替代方向
        assert "pure_size" in text
        assert "纯市值" in text

    def test_missing_columns_flagged(self):
        """面板未载入的族必须显式标注，避免教了不存在的列白烧评估预算。"""
        spec = default_research_spec("technical_event_monthly")
        text = _prompt(spec=spec, cols=_BASE_COLS)  # 无任何事件/资金列
        assert "允许但本次面板未载入的族" in text
        assert "`capital_flow`" in text

    def test_focus_facets_exempt(self):
        """聚焦 run 由 facet_focus 硬锁定并显式豁免白名单，本模块必须让位。"""
        spec = default_research_spec("technical_event_monthly")
        assert sf_mod.enabled(_ctx(spec, focus=("资金面",))) is False

    def test_unregistered_family_reported_without_field_hint(self):
        """未登记的族只报名字，不编字段（宁可少说，不可教错列名）。"""
        spec = default_research_spec("technical_event_monthly")
        spec["search_policy"]["allowed_signal_families"] = list(
            spec["search_policy"]["allowed_signal_families"]
        ) + ["mystery_family"]
        text = _prompt(spec=spec, cols=_BASE_COLS + _EVENT_COLS)
        assert "未登记字段提示的族" in text
        assert "mystery_family" in text


def _ctx(spec, focus=()):
    """构造 signal_families.enabled() 需要的最小上下文。"""

    class _Ctx:
        research_spec = spec
        focus_facets = tuple(focus)
        panel_columns = None

    return _Ctx()


# ── 5. data_fields 新块按面板列启停 ───────────────────────────────────

class TestDataFieldsInjection:
    def test_unlock_block_follows_panel_columns(self):
        spec = default_research_spec("technical_event_monthly")
        with_unlock = _prompt(spec=spec, cols=_BASE_COLS + _EVENT_COLS)
        # 用小节标题定位 data_fields 块（signal_families 模块里也会提到 $unlock_*）
        assert "### 限售解禁" in with_unlock
        assert "unlock_ratio_past180" in with_unlock
        # PIT 分级警示：next 两列有前视风险，必须出现
        assert "PIT 分级" in with_unlock
        without_unlock = _prompt(spec=spec, cols=_BASE_COLS + ["ff_main_net", "mgn_balance"])
        assert "### 限售解禁" not in without_unlock

    def test_institutional_breakdown_present(self):
        spec = default_research_spec("technical_event_monthly")
        text = _prompt(spec=spec, cols=_BASE_COLS + _EVENT_COLS)
        assert "分机构类型口径" in text
        for col in ("inst_fund_ratio", "inst_qfii_ratio", "inst_social_security_ratio"):
            assert col in text, col

    def test_ac_coverage_warning_present(self):
        """$ac_* 实测覆盖率 0.85%，提示词必须给出警示与替代方向。"""
        spec = default_research_spec("technical_event_monthly")
        text = _prompt(
            spec=spec,
            cols=_BASE_COLS + _EVENT_COLS + ["ac_eps_fy", "ac_pe_fy", "rc_cnt_90d"],
        )
        assert "覆盖率实测" in text
        assert "0.85%" in text


# ── 6. 前后端口径一致（防漂移）──────────────────────────────────────
# 前端 static/src/store/alphaagent.js 的 inferredMode 是后端 infer_research_mode 的
# **复制实现**；startAgent 会显式提交 effectiveMode，后端 start() 仅在 research_mode
# 为 None 时才推断 —— 两者一旦漂移，后端改动在 UI 路径上完全不生效
# （2026-10-19 OCR 实锤：本改动的第一版就踩了这个坑，新档在 UI 上永远选不中）。

_STORE_JS = (
    Path(__file__).resolve().parents[1] / "static" / "src" / "store" / "alphaagent.js"
)


def _js_facet_list(var_name: str) -> set[str]:
    src = _STORE_JS.read_text(encoding="utf-8")
    m = re.search(rf"const\s+{var_name}\s*=\s*\[([^\]]*)\]", src)
    assert m, f"前端 {var_name} 面清单未找到（写法变更需同步本测试）"
    return set(re.findall(r"'([^']+)'", m.group(1)))


class TestFrontendBackendParity:
    def test_event_facets_match(self):
        from core.research_modes import _EVENT_FACETS

        assert _js_facet_list("event") == set(_EVENT_FACETS)

    def test_slow_facets_match(self):
        from core.research_modes import _SLOW_FACETS

        assert _js_facet_list("slow") == set(_SLOW_FACETS)

    def test_event_mode_wired_in_frontend(self):
        src = _STORE_JS.read_text(encoding="utf-8")
        assert "technical_event_monthly" in src
        # daily/weekly 不得落 event 档（无对应 label 标定）
        assert "freq !== 'daily' && freq !== 'weekly'" in src

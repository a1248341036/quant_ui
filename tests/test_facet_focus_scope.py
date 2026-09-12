# -*- coding: utf-8 -*-
"""数据面聚焦的"上下文投影"：让 LLM 只看到勾选面的数据。

配合工具层硬拦截（tests/test_facet_focus_lock.py）：拦截保证越界表达式不执行，
本组测试保证越界示例/字段/历史证据根本不进入上下文（省掉试探-被拦-重写的 token）。

覆盖：
- scope_filter.scrub_out_of_scope：行内与围栏代码片段的越界替换；
- build_system_prompt 聚焦裁剪：未选面字段/算子/示例消失，白名单在；
- 记忆检索按面过滤：证据块 / 跨族保底正向父本 / 经验块。
"""
from __future__ import annotations

import json

from alphaagent.factor.mining.agent.agentscope_tools import _dispatch_sync  # noqa: F401  (导入冒烟)
from alphaagent.factor.mining.memory.expressions import (
    facet_allowed_scope,
)
from alphaagent.factor.mining.memory.retrieval import (
    _entry_in_scope,
    _facets_in_scope,
)
from alphaagent.factor.mining.prompt.scope_filter import scrub_out_of_scope
from alphaagent.factor.mining.prompts import build_system_prompt, last_assembly_report
from alphaagent.factor.mining.research_memory import ResearchMemoryStore

_COLS = [
    "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close",
    "volume", "amount", "float_cap", "tot_cap", "vwap", "adj_vwap", "ret",
    "is_trade", "not_st", "industry_sw_l1", "label_1d_open_to_close",
    "funda_roe", "funda_netprofit_yoy",
    "pred_direction", "pred_change_mid", "pred_net_profit_mid", "pred_surprise", "pred_days_since",
    "exp_revenue", "exp_net_profit", "exp_roe", "exp_netprofit_yoy",
    "mgn_balance", "mgn_buy", "dt_cnt_90d", "dt_net_buy_90d",
    "chip_conc_top10", "crowd_overlap",
]


def _prompt(focus=None, *, catalog=True, phase="full", panel_columns=None) -> str:
    return build_system_prompt(
        include_operator_catalog=catalog,
        label_col="label_1d_open_to_close",
        include_fundamentals=True,
        panel_columns=list(panel_columns) if panel_columns is not None else _COLS,
        population_max=0,
        research_spec=None,
        asset_type="stock",
        focus_facets=focus,
        prompt_phase=phase,
    )


# ── 1. 片段裁剪 ──

class TestScrubOutOfScope:
    def test_no_focus_keeps_everything(self):
        text = "示例 `TS_MEAN($adj_close, 20)` 与 ```\nRANK($volume)\n```"
        assert scrub_out_of_scope(text, None) == text
        assert scrub_out_of_scope(text, ()) == text

    def test_inline_in_scope_kept(self):
        text = "用 `TS_RANK($mgn_balance, 20)` 构造"
        assert scrub_out_of_scope(text, ["两融面"]) == text

    def test_inline_out_of_scope_replaced(self):
        out = scrub_out_of_scope("用 `TS_MEAN($adj_close, 20)` 构造", ["两融面"])
        assert "$adj_close" not in out
        assert "未选「价量面」" in out

    def test_fenced_block_replaced(self):
        text = "```\nbase = TS_PCTCHANGE($adj_close, 5)\nRANK(base)\n```"
        out = scrub_out_of_scope(text, ["事件面"])
        assert "$adj_close" not in out
        assert "示例略" in out

    def test_pure_operator_snippet_kept(self):
        """不含列引用的纯算子片段对所有面都可用，不裁剪。"""
        text = "分组条件 `CS_GROUP_RANK(sig, CS_BUCKET(cond,5))`"
        assert scrub_out_of_scope(text, ["事件面"]) == text

    def test_multi_focus_partial_coverage_scrubbed(self):
        out = scrub_out_of_scope("`DIVIDE($mgn_balance, $close)`", ["两融面", "事件面"])
        assert "$close" not in out and "$mgn_balance" not in out


# ── 2. system prompt 投影 ──

class TestPromptScopeProjection:
    def test_earnings_face_hides_price_and_other_faces(self):
        """业绩面没有隐含输入面：价量/财务/两融/事件列全部不可见。"""
        text = _prompt(["业绩面"])
        assert "$adj_close" not in text
        assert "$close" not in text
        assert "funda_roe" not in text
        assert "$mgn_" not in text
        assert "$dt_net_buy_90d" not in text
        assert "$pred_surprise" in text
        assert "$exp_net_profit" in text

    def test_operator_face_keeps_input_columns_with_note(self):
        """筹码面：CHIP_* 需要 close/low/high/volume，这些输入列保留并标注用途。"""
        text = _prompt(["筹码面"])
        assert "$close" in text and "$volume" in text
        assert "仅作为聚焦面算子的输入" in text
        assert "funda_roe" not in text
        assert "$mgn_" not in text

    def test_volume_face_keeps_price_input(self):
        """量能面：VOLUME_CLOCK_VPIN/MUTUAL_INFO_LAG 需要价格序列。"""
        text = _prompt(["量能面"])
        assert "$adj_close" in text
        assert "$volume" in text
        assert "funda_roe" not in text

    def test_whitelist_emitted(self):
        text = _prompt(["业绩面", "量能面"])
        assert "本 run 唯一可用列（白名单，表达式只能引用这些）" in text
        assert "$pred_surprise" in text
        assert "通用中性列" in text and "$float_cap" in text
        assert "仅作为聚焦面算子的输入" in text  # 量能面 → 价量面输入

    def test_price_focus_keeps_price_fields(self):
        text = _prompt(["价量面", "量能面"])
        assert "$adj_close" in text
        assert "$volume" in text
        assert "funda_roe" not in text

    def test_operator_catalog_hides_unselected_family(self):
        from alphaagent.dsl.catalog import operator_catalog_markdown
        from alphaagent.factor.mining.prompt.modules.operator_catalog import _excluded_prefixes

        excluded = _excluded_prefixes(["业绩面"])
        assert "CHIP_" in excluded and "PRICE_" in excluded and "VOLUME_" in excluded
        catalog = operator_catalog_markdown(excluded_prefixes=excluded)
        assert "CHIP_" not in catalog
        assert "TS_MEAN" in catalog  # 通用算子不受影响
        # 选中筹码面后该族不再被排除
        assert "CHIP_" not in _excluded_prefixes(["业绩面", "筹码面"])

    def test_tool_examples_synthesized_for_focus(self):
        """默认示例全越界被裁空时，按聚焦面代表列合成合规骨架——
        没有骨架时 LLM 会按先验拼纯价量结构（run 51e02d47a3f3 教训）。"""
        _prompt(["业绩面", "量能面"])
        row = next(r for r in last_assembly_report if r["module"] == "tool_examples")
        assert row["chars"] > 0
        assert row["required_empty"] is False

    def test_tool_examples_empty_without_repr_faces(self):
        _prompt(["筹码面"])
        row = next(r for r in last_assembly_report if r["module"] == "tool_examples")
        assert row["chars"] == 0
        assert row["required_empty"] is False

    def test_no_focus_prompt_unchanged_shape(self):
        text = _prompt(None)
        assert "$adj_close" in text and "funda_roe" in text
        assert "本 run 唯一可用列" not in text

    def test_missing_neutral_columns_warned_against_guessing(self):
        """聚焦 run 面板缺行业列时，提示词必须点破"别猜列名"（反幻觉）。

        2026-09-12 实证：LLM 反复引用 $industry_sw_l1 等未加载列（34 次
        MultiLineFactorEvalError），因为这些是"熟悉但不存在于本 run 面板"的列名。
        """
        import copy

        cols = copy.deepcopy(_COLS)
        cols = [c for c in cols if c not in ("industry_sw_l1", "industry_zx_l1")]
        text = _prompt(["业绩面"], panel_columns=cols)
        assert "$industry_sw_l1" in text  # 明确点名，警告不可引用
        assert "本 run 面板并未加载" in text
        assert "不要凭熟悉度猜测列名" in text

    def test_present_neutral_columns_still_usable(self):
        """行业列在面板里时仍是通用中性列，不出反幻觉警告。"""
        # _COLS 含 industry_sw_l1；把 zx 也补进面板 → 两个中性列都 present，无警告
        text = _prompt(["业绩面"], panel_columns=_COLS + ["industry_zx_l1"])
        assert "本 run 面板并未加载" not in text


class TestInputColumnGrouping:
    """聚焦列与仅作输入列分组呈现（白名单 + 变量表双处标注）。"""

    def test_whitelist_splits_focus_and_input_columns(self):
        text = _prompt(["业绩面", "量能面"])
        lines = text.splitlines()
        focus_line = next(l for l in lines if "聚焦面列" in l and "唯一可用列" in l)
        input_line = next(l for l in lines if "仅作输入列" in l)
        assert "$pred_surprise" in focus_line and "$amount" in focus_line
        assert "$adj_close" not in focus_line
        assert "$adj_close" in input_line and "$pred_surprise" not in input_line

    def test_vars_table_marks_input_only_scope(self):
        text = _prompt(["业绩面", "量能面"])
        assert "输入列提示" in text
        assert "仅作输入" in text

    def test_no_input_note_without_implied_faces(self):
        text = _prompt(["业绩面"])
        assert "输入列提示" not in text
        assert "仅作输入列" not in text

    def test_chip_focus_marks_price_volume_as_input(self):
        text = _prompt(["筹码面"])
        assert "输入列提示" in text
        assert "价量面" in text and "量能面" in text


# ── 3. 记忆检索按面过滤 ──

def _eval_row(name, expr, factor_name, ic):
    return {
        "name": name,
        "arguments_raw": json.dumps({"multi_line_expr": expr, "factor_name": factor_name}),
        "result": {
            "ok": True,
            "split": "train",
            "metrics": {
                "cross_sectional_core": {"ic": ic, "icir": 0.4, "factor_coverage": 0.9},
            },
        },
    }


class TestMemoryScopeFilter:
    def test_entry_in_scope_rules(self):
        assert _entry_in_scope({"facets": ["两融面"]}, {"两融面", "事件面"})
        assert not _entry_in_scope({"facets": ["价量面"]}, {"两融面"})
        # facets 缺失 → 按表达式现算
        assert _entry_in_scope({"expression": "RANK($mgn_balance)"}, {"两融面"})
        assert not _entry_in_scope({"expression": "RANK($adj_close)"}, {"两融面"})
        assert _facets_in_scope("RANK($mgn_balance)", {"两融面"})

    def test_required_intersection_blocks_borrowed_inputs(self):
        """隐含输入面（价量=筹码算子输入）放行，但纯价量因子不能借道混入。"""
        scope = facet_allowed_scope(["筹码面"])
        assert scope == {"筹码面", "价量面", "量能面"}
        assert _entry_in_scope({"expression": "CHIP_ENTROPY($close,$low,$high,$volume,30,$float_cap)"},
                               scope, {"筹码面"})
        assert not _entry_in_scope({"expression": "TS_MEAN($adj_close, 20)"}, scope, {"筹码面"})

    def _store(self, tmp_path):
        store = ResearchMemoryStore(tmp_path / "m.db")
        store.record_tool_result(
            run_id="r1",
            row=_eval_row("evaluate_factor", "TS_MEAN($adj_close, 20)", "price_mom_20", ic=0.031),
        )
        store.record_tool_result(
            run_id="r1",
            row=_eval_row("evaluate_factor", "TS_RANK($mgn_balance, 20)", "mgn_bal_rank", ic=0.028),
        )
        return store

    def test_evidence_block_scope(self, tmp_path):
        store = self._store(tmp_path)
        block = store._evidence_block(
            "RANK($mgn_balance)", limit=8, include_rejected=True, include_expression=True,
            facet_scope={"两融面"},
        )
        assert "mgn_bal_rank" in block
        assert "price_mom_20" not in block

    def test_guaranteed_positives_scope(self, tmp_path):
        store = self._store(tmp_path)
        all_pos = store._guaranteed_positives(5)
        names = {e.get("factor_name") for e in all_pos}
        assert "price_mom_20" in names
        scoped = store._guaranteed_positives(5, facet_scope={"两融面"})
        scoped_names = {e.get("factor_name") for e in scoped}
        assert "mgn_bal_rank" in scoped_names
        assert "price_mom_20" not in scoped_names

    def test_context_for_scope(self, tmp_path):
        store = self._store(tmp_path)
        block = store.context_for(
            "RANK($mgn_balance)", limit=6, enable_factor_retrieval=True,
            facet_scope={"两融面"},
        )
        assert "mgn_bal_rank" in block
        assert "price_mom_20" not in block

    def test_recommend_edits_accepts_scope(self, tmp_path):
        store = ResearchMemoryStore(tmp_path / "m.db")
        assert store.recommend_edits(2, facet_scope={"两融面"}) == []

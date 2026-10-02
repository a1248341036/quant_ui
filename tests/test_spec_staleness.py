# -*- coding: utf-8 -*-
"""spec 新鲜度自检回归（2026-10-02）。"""
from __future__ import annotations

from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC, default_missing_keys


def test_complete_spec_has_no_missing():
    assert default_missing_keys(DEFAULT_RESEARCH_SPEC) == []


def test_stale_spec_reports_missing_keys():
    """模拟"后端冻结的旧 spec"：report_policy 少两个键。"""
    stale = {"report_policy": {k: v for k, v in DEFAULT_RESEARCH_SPEC["report_policy"].items()
                               if k not in ("require_report_structure", "diverge_parent_required")}}
    missing = default_missing_keys(stale)
    assert "report_policy.require_report_structure" in missing
    assert "report_policy.diverge_parent_required" in missing


def test_empty_spec_reports_everything():
    assert len(default_missing_keys({})) > 5

# -*- coding: utf-8 -*-
"""Persistent, evidence-backed memory for AlphaAgent factor research.

This module is the stable import surface for the research memory. The v3-lite
implementation is split by responsibility under
:mod:`alphaagent.factor.mining.memory`.
"""

from __future__ import annotations

from .memory.calibration import _apv_gate, _eq7_confidence, _parent_bucket
from .memory.constants import (
    APV_TAU_C_DEFAULT,
    APV_TAU_V_DEFAULT,
    BASELINE_HALF_LIFE_DAYS,
    DATA_VERSION,
    EDIT_PRIOR_HARD_CONF_DEFAULT,
    EDIT_PRIOR_RECOMMEND_CONF_DEFAULT,
    EDIT_PRIOR_VETO_CONF_DEFAULT,
    EQ7_KAPPA_DEFAULT,
    INVALID_WEIGHT,
    NEGATIVE_VERDICTS,
    PARENT_ORIGIN_WEIGHT,
    POSITIVE_VERDICTS,
    VERDICT_ORDER,
    VERDICT_WEIGHT,
)
from .memory.diagnostics import (
    _FORBIDDEN_SIGNATURES,
    _SUCCESS_SIGNATURES,
    _extract_fail_detail,
    _failure_code,
    _match_signature,
    _now,
    _parse_args,
    _rebuild_conclusion,
    _safe_float,
)
from .memory.expressions import (
    MOTIFS,
    _cjk_tokens,
    _structure_fingerprint,
    _tokens,
    classify_family,
    expression_features,
    expression_ops,
    expression_windows,
    extract_edit_motif,
    motif_from_note,
    template_from_expression,
)
from .memory.store import ResearchMemoryStore

__all__ = [
    "ResearchMemoryStore",
    "MOTIFS",
    "POSITIVE_VERDICTS",
    "NEGATIVE_VERDICTS",
    "VERDICT_ORDER",
    "VERDICT_WEIGHT",
    "EQ7_KAPPA_DEFAULT",
    "APV_TAU_C_DEFAULT",
    "APV_TAU_V_DEFAULT",
    "EDIT_PRIOR_HARD_CONF_DEFAULT",
    "EDIT_PRIOR_RECOMMEND_CONF_DEFAULT",
    "EDIT_PRIOR_VETO_CONF_DEFAULT",
    "BASELINE_HALF_LIFE_DAYS",
    "PARENT_ORIGIN_WEIGHT",
    "INVALID_WEIGHT",
    "DATA_VERSION",
    "template_from_expression",
    "extract_edit_motif",
    "motif_from_note",
    "classify_family",
    "expression_features",
    "expression_ops",
    "expression_windows",
]

"""Frozen-profile DSL evaluation with plugin transforms, metrics and rules."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.dsl import eval_factor
from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.factor.align import align_series_to_panel
from alphaagent.factor.evaluation.context import EvaluationContext
from alphaagent.factor.evaluation.plugins import get_metric, get_transform
from alphaagent.factor.evaluation.profile import EvaluationProfile
from alphaagent.factor.evaluation.rules import evaluate_rules
from alphaagent.data.panel import slice_panel


class EvaluationEngine:
    def __init__(self, profiles: dict[str, EvaluationProfile]) -> None:
        self.profiles = dict(profiles)

    def profile(self, profile_id: str) -> EvaluationProfile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown_evaluation_profile:{profile_id}") from exc

    def evaluate(
        self,
        session: Any,
        *,
        profile_id: str,
        multi_line_expr: str,
        factor_name: str = "expr",
    ) -> dict[str, Any]:
        profile = self.profile(profile_id)
        panel, date_range = self._panel_for_profile(session, profile)
        if panel.empty:
            return {"ok": False, "error": "evaluation_profile_empty_panel", "error_type": "EmptyData", "profile_id": profile_id}
        label_col = session.ctx.label_col
        if label_col not in panel.columns:
            return {"ok": False, "error": f"panel_missing_label:{label_col}", "error_type": "MissingLabelColumn", "profile_id": profile_id}
        timing: dict[str, float] = {}
        started = time.perf_counter()
        try:
            point = time.perf_counter()
            raw = eval_factor(multi_line_expr, panel)
            timing["dsl_eval_ms"] = (time.perf_counter() - point) * 1000
            if not isinstance(raw, pd.Series):
                raise TypeError(f"factor_output_must_be_series:{type(raw)!r}")
            point = time.perf_counter()
            values = align_series_to_panel(raw, panel)
            factor = pd.Series(values, index=panel.index, name=factor_name, dtype=np.float32)
            context = EvaluationContext(panel=panel, factor=factor, label=panel[label_col], profile=profile, factor_name=factor_name)
            for item in profile.transforms:
                get_transform(str(item["plugin"]))(context, dict(item.get("params") or {}))
            timing["transforms_ms"] = (time.perf_counter() - point) * 1000
            point = time.perf_counter()
            metrics: dict[str, Any] = {}
            for item in profile.metrics:
                name = str(item["plugin"])
                metrics[name] = get_metric(name)(context, dict(item.get("params") or {}))
            timing["metrics_ms"] = (time.perf_counter() - point) * 1000
        except MultiLineFactorEvalError as exc:
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "profile_id": profile_id}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__, "profile_id": profile_id}
        timing["total_ms"] = (time.perf_counter() - started) * 1000
        rule_results = evaluate_rules(metrics, profile.rules)
        dataset_id = hashlib.sha256(f"{session.ctx.panel_path}|{label_col}|{date_range['start']}|{date_range['end']}".encode("utf-8")).hexdigest()[:20]
        return {
            "ok": True,
            "candidate": {"factor_name": factor_name, "expression": multi_line_expr},
            "profile": profile.as_dict(),
            "profile_hash": profile.fingerprint,
            "dataset_fingerprint": dataset_id,
            "split": profile.split,
            "date_range": date_range,
            "label_col": label_col,
            "metrics": metrics,
            "rule_results": rule_results,
            "passed": all(row["passed"] for row in rule_results) if rule_results else True,
            "transforms_applied": context.transforms_applied,
            "timing_ms": timing,
        }

    @staticmethod
    def _panel_for_profile(session: Any, profile: EvaluationProfile) -> tuple[pd.DataFrame, dict[str, str]]:
        if profile.split == "full":
            start, end = session.ctx.coverage_range()
            return slice_panel(session.panel, start=start, end=end), {"start": start, "end": end}
        panel, start, end = session.get_split_panel(profile.split)
        return panel, {"start": start, "end": end}

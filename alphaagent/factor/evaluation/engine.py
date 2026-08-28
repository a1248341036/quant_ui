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
from alphaagent.factor.metrics import (
    daily_long_short_series,
    monthly_ic_robustness,
)
from alphaagent.data.panel import slice_panel


def _series_to_points(s: pd.Series) -> list[dict[str, Any]]:
    """pd.Series → [{date, value}, ...]，跳过 NaN。"""
    out: list[dict[str, Any]] = []
    for idx, val in s.items():
        v = float(val) if np.isfinite(val) else None
        ts = idx
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%Y-%m-%d")
        else:
            ts = str(ts)[:10]
        out.append({"date": ts, "value": v})
    return out


def _monthly_breakdown(daily_series: pd.Series) -> list[dict[str, Any]]:
    """逐日序列 → 月度聚合 [{month, mean, n_days, positive_ratio}]。"""
    if daily_series.empty:
        return []
    s = daily_series.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    finite = s[np.isfinite(s.to_numpy(dtype=float, copy=False))]
    if finite.empty:
        return []
    out: list[dict[str, Any]] = []
    for period, grp in finite.groupby(finite.index.to_period("M"), sort=True):
        vals = grp.to_numpy(dtype=float, copy=False)
        mean_v = float(np.mean(vals)) if len(vals) else float("nan")
        pos_ratio = float(np.mean(vals > 0)) if len(vals) else 0.0
        out.append({
            "month": str(period),
            "mean": round(mean_v, 6) if np.isfinite(mean_v) else None,
            "n_days": int(len(vals)),
            "positive_ratio": round(pos_ratio, 4),
        })
    return out


def _cumulative_returns(daily_ls: pd.Series) -> list[dict[str, Any]]:
    """逐日多空收益 → 累计收益曲线。"""
    if daily_ls.empty:
        return []
    s = daily_ls.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    valid = s[np.isfinite(s.to_numpy(dtype=float, copy=False))]
    if valid.empty:
        return []
    cum = (1.0 + valid).cumprod() - 1.0
    return _series_to_points(cum)


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
        label_quantile_n: int = 0,
        include_detail_tables: bool = False,
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

        # ── 生成详细可视化数据 ──
        daily_ic_series = context.daily_ic()
        daily_rank_ic_series = context.daily_rank_ic()
        daily_ls = daily_long_short_series(
            context.factor, context.label, n_deciles=10, min_stocks=30,
        )

        chart_data: dict[str, Any] = {
            "daily_ic": _series_to_points(daily_ic_series),
            "daily_rank_ic": _series_to_points(daily_rank_ic_series),
            "daily_long_short": _series_to_points(daily_ls),
            "cumulative_long_short": _cumulative_returns(daily_ls),
            "monthly_ic": _monthly_breakdown(daily_ic_series),
            "monthly_long_short": _monthly_breakdown(daily_ls),
        }

        out: dict[str, Any] = {
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
            "chart_data": chart_data,
        }

        # 兼容旧 split 评估契约：label_quantile_buckets / by_month / by_symbol
        if label_quantile_n and label_quantile_n >= 2:
            from alphaagent.factor.metrics import label_quantile_buckets
            out["label_quantile_buckets"] = label_quantile_buckets(
                context.factor.to_numpy(dtype=np.float64, copy=False),
                context.label.to_numpy(dtype=np.float64, copy=False),
                n_quantiles=int(label_quantile_n),
            )
            out["label_quantile_n"] = int(label_quantile_n)
        if include_detail_tables:
            from alphaagent.factor.metrics import by_symbol_ts_ic, monthly_detail_rows
            out["by_month"] = monthly_detail_rows(daily_ic_series, daily_rank_ic_series)
            out["by_symbol"] = by_symbol_ts_ic(context.factor, context.label)

        return out

    @staticmethod
    def _panel_for_profile(session: Any, profile: EvaluationProfile) -> tuple[pd.DataFrame, dict[str, str]]:
        if profile.split == "full":
            start, end = session.ctx.coverage_range()
            return slice_panel(session.panel, start=start, end=end), {"start": start, "end": end}
        panel, start, end = session.get_split_panel(profile.split)
        return panel, {"start": start, "end": end}

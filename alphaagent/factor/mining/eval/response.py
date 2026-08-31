"""将 split 评估原始结果格式化为 LLM tool 兼容 JSON。"""

from __future__ import annotations

from typing import Any

import numpy as np


def _round_float4(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
    except TypeError:
        pass
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return float(round(x, 4)) if np.isfinite(x) else None


def _format_decile_mean_label(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        ml = row.get("mean_label")
        out.append(
            {
                "decile": int(row["decile"]),
                "mean_label": _round_float4(ml) if ml is not None else None,
            }
        )
    return out


def _format_mls_fmb(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    float_keys = (
        "mean_rho",
        "mean_ls",
        "ir_ls",
        "ir_ls_annual",
        "mls",
        "nw_t_rho",
        "nw_t_ls",
        "nw_se_rho",
        "nw_se_ls",
        "annualization_factor",
    )
    int_keys = ("n_days_rho", "n_days_ls", "nw_lags", "n_deciles", "n_deciles_requested", "min_stocks", "min_stocks_requested")
    out: dict[str, Any] = {}
    for k in float_keys:
        if k in raw:
            out[k] = _round_float4(raw.get(k))
    for k in int_keys:
        if k in raw:
            v = raw.get(k)
            out[k] = int(v) if v is not None else None
    if raw.get("note"):
        out["note"] = raw["note"]
    return out


def _format_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "ic": _round_float4(summary.get("ic")),
        "icir": _round_float4(summary.get("icir")),
        "rank_ic": _round_float4(summary.get("rank_ic")),
        "n_days": int(summary["n_days"]) if summary.get("n_days") is not None else None,
        "n_instruments": int(summary["n_instruments"]) if summary.get("n_instruments") is not None else None,
        "factor_coverage": _round_float4(summary.get("factor_coverage")),
        "factor_skewness": _round_float4(summary.get("factor_skewness")),
        "factor_kurtosis": _round_float4(summary.get("factor_kurtosis")),
        "cs_pearson_autocorr": _round_float4(summary.get("cs_pearson_autocorr")),
        "decile_mean_label": _format_decile_mean_label(summary.get("decile_mean_label")),
        "mls_fmb": _format_mls_fmb(summary.get("mls_fmb")),
    }


def monthly_corr_robustness_json(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k == "note":
            out[k] = v
        elif isinstance(v, (float, np.floating)) and np.isfinite(v):
            out[k] = _round_float4(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def format_eval_response(
    raw: dict[str, Any],
    *,
    expected_sign: int | None = None,
) -> dict[str, Any]:
    if not raw.get("ok", False):
        return {
            "ok": False,
            "error": raw.get("error", "unknown"),
            "error_type": raw.get("error_type", "Error"),
            "split": raw.get("split"),
            "date_range": raw.get("date_range"),
        }

    summ = _format_summary(raw.get("summary") or {})
    rob_raw = raw.get("monthly_corr_robustness") or {}

    out: dict[str, Any] = {
        "ok": True,
        "split": raw["split"],
        "date_range": raw["date_range"],
        "eval_wall_seconds": _round_float4(raw.get("eval_wall_seconds")),
        "summary": summ,
        "label_col": raw.get("label_col"),
        "bar_interval": raw.get("bar_interval", "1d"),
        "label_quantile_buckets": [
            {k: v for k, v in row.items() if k != "n"} for row in raw.get("label_quantile_buckets", [])
        ],
        "label_quantile_n": raw.get("label_quantile_n"),
        "monthly_corr_robustness": monthly_corr_robustness_json(rob_raw),
        "timing_ms": {k: round(v, 2) for k, v in raw.get("timing_ms", {}).items()},
    }

    n_q = int(raw.get("label_quantile_n") or 0)
    if n_q >= 2:
        out["label_quantile_note"] = "D1/Q1=因子最低组； D10/Q10=因子最高组。summary.decile_mean_label 为固定十分组。"

    if raw.get("include_detail_tables") or raw.get("by_month") is not None:
        if "by_month" in raw:
            out["by_month"] = raw["by_month"]
        if "by_symbol" in raw:
            out["by_symbol"] = raw["by_symbol"]

    if raw.get("split") == "val" and expected_sign is not None:
        ic_val = summ.get("ic")
        matches = None
        if ic_val is not None and np.isfinite(ic_val):
            matches = (ic_val > 0 and expected_sign == 1) or (ic_val < 0 and expected_sign == -1)
        out["expected_sign"] = int(expected_sign)
        out["sign_check"] = {
            "expected_sign": int(expected_sign),
            "val_ic": ic_val,
            "matches_expected_sign": matches,
        }
    return out

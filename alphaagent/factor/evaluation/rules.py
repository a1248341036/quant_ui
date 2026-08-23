"""Generic metric rule evaluator used by frozen evaluation profiles."""

from __future__ import annotations

import math
from typing import Any


def metric_at(metrics: dict[str, Any], path: str) -> Any:
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def evaluate_rules(metrics: dict[str, Any], rules: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule in rules:
        path = str(rule.get("metric") or "")
        op = str(rule.get("op") or "")
        expected = rule.get("value")
        actual = metric_at(metrics, path)
        try:
            actual_num = float(actual)
            expected_num = float(expected)
            valid = math.isfinite(actual_num) and math.isfinite(expected_num)
        except (TypeError, ValueError):
            actual_num = expected_num = float("nan")
            valid = False
        if op == "gte":
            passed = valid and actual_num >= expected_num
        elif op == "lte":
            passed = valid and actual_num <= expected_num
        elif op == "abs_gte":
            passed = valid and abs(actual_num) >= expected_num
        elif op == "abs_lte":
            passed = valid and abs(actual_num) <= expected_num
        else:
            passed = False
        out.append({"metric": path, "op": op, "expected": expected, "actual": actual, "passed": passed})
    return out

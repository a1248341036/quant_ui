"""事件链路 NaN 防护回归。

挖掘轨迹/工具结果中若混入 NaN，Python json.dumps 默认会写出非法 JSON 字面量，
前端 SSE `JSON.parse` 直接失败（"Unexpected token 'N'"），Starlette
allow_nan=False 的 REST 响应也会 500。所有写出点必须经 json_safe 清洗。
"""

from __future__ import annotations

import json
import math

import pytest

from alphaagent.factor.mining.eval.response import monthly_corr_robustness_json
from alphaagent.factor.mining.infra.jsonutil import json_safe


def _strict_loads(line: str) -> object:
    """parse_constant 只会在 NaN/Infinity/-Infinity 字面量上被调用。"""

    def _reject(constant: str) -> object:
        raise AssertionError(f"非法 JSON 常量: {constant}")

    return json.loads(line, parse_constant=_reject)


def test_json_safe_replaces_nonfinite_floats() -> None:
    payload = {
        "a": float("nan"),
        "b": [1.5, float("inf"), {"c": float("-inf")}],
        "d": None,
        "e": "x",
        "f": 3,
        "g": True,
    }
    clean = json_safe(payload)
    assert clean["a"] is None
    assert clean["b"] == [1.5, None, {"c": None}]
    assert clean["d"] is None
    assert clean["e"] == "x"
    assert clean["f"] == 3
    assert clean["g"] is True
    # 清洗后必须是严格可往返的合法 JSON
    assert _strict_loads(json.dumps(clean)) == clean


def test_json_safe_keeps_finite_values_untouched() -> None:
    value = {"ic": 0.0234, "n": 500, "flag": False, "nested": {"ok": 1.0}}
    assert json_safe(value) == value


def test_json_safe_handles_numpy_scalars() -> None:
    np = pytest.importorskip("numpy")
    clean = json_safe({"x": np.float64("nan"), "y": np.float32(1.25), "z": np.int64(7)})
    assert clean["x"] is None
    assert math.isclose(float(clean["y"]), 1.25)
    assert clean["z"] == 7


def test_monthly_robustness_nan_becomes_null() -> None:
    out = monthly_corr_robustness_json({"2024-01": float("nan"), "2024-02": 0.123456789})
    assert out["2024-01"] is None
    assert out["2024-02"] == 0.1235
    assert _strict_loads(json.dumps(out)) == out


def test_emit_writes_strict_json(tmp_path) -> None:
    """模拟轨迹 _emit 写出：含 NaN 的事件行必须是合法 JSON（历史故障路径）。"""
    record = {
        "ts": "2026-01-01T00:00:00+00:00",
        "event": "tool_results",
        "results": [{"name": "evaluate_factor", "mean_rho": float("nan"), "mean_ls": -0.5}],
    }
    path = tmp_path / "run_x.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False, default=str) + "\n")
    row = _strict_loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["results"][0]["mean_rho"] is None
    assert row["results"][0]["mean_ls"] == -0.5

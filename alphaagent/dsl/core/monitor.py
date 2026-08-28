"""DSL 算子级计时监控：在求值命名空间层包计时代理，零侵入采集各算子耗时。

设计
----
- **零侵入**：不改任何算子函数。在 ``eval_multi_line_factor`` 构造命名空间后，
  用 ``wrap_operator_namespace`` 把大写算子（DSL 命名约定）包一层计时代理。
- **thread-local**：挖掘循环 4 个并行 worker 各自独立收集，互不污染。
  每次评估（``OperatorTimingContext``）一个收集上下文；``collect_report``
  返回本次聚合，``reset`` 清空。
- **参数摘要**：只记录标量/短字符串参数（如窗口大小、分位数），以及第一个
  参数的类型与行数，避免持有大数据对象引用。
- **持久化**：``append_record`` 把单次评估的算子耗时追加到
  ``artifacts/dsl_operator_profiling.jsonl``（累计历史，供长期趋势观察）。

用法（在 eval 层）::

    from alphaagent.dsl.core import monitor
    ctx = monitor.begin()
    try:
        result = evaluate(...)   # 命名空间已 wrap
    finally:
        report = monitor.end(ctx)   # -> dict，可写入评估结果 / JSONL
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PROFILING_LOG = Path(__file__).resolve().parents[3] / "artifacts" / "dsl_operator_profiling.jsonl"
# 环境变量开关：默认开启；关闭时零开销（wrapper 直通）
_ENABLED = os.environ.get("ALPHA_DSL_OPERATOR_MONITOR", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

_thread_local = threading.local()


def _summary_param(value: Any) -> Any:
    """参数摘要：标量/短字符串原样，DataFrame/Series 记类型+形状，其他记类型。"""
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (int, float, bool, str, np.integer, np.floating)):
        if isinstance(value, str) and len(value) > 40:
            return value[:40] + "..."
        return value
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return f"{type(value).__name__}({value.shape[0]}x{getattr(value, 'shape', ())[1] if isinstance(value, pd.DataFrame) else 1})"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}"
    return type(value).__name__


class OperatorTimingContext:
    """单次评估的算子计时收集上下文（thread-local）。"""

    __slots__ = ("_stats", "_call_stack", "_started")

    def __init__(self) -> None:
        self._stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "total_s": 0.0, "max_s": 0.0, "samples": []}
        )
        self._call_stack: list[str] = []
        self._started = time.perf_counter()

    def enter(self, name: str) -> None:
        self._call_stack.append((name, time.perf_counter()))

    def leave(self, name: str, sample: dict[str, Any]) -> None:
        if not self._call_stack:
            return
        n, t0 = self._call_stack.pop()
        if n != name:
            # 防御：栈错位时尽量用最近帧
            n, t0 = self._call_stack[-1] if self._call_stack else (name, t0)
        dt = time.perf_counter() - t0
        st = self._stats[name]
        st["calls"] += 1
        st["total_s"] += dt
        if dt > st["max_s"]:
            st["max_s"] = dt
        if len(st["samples"]) < 3:
            st["samples"].append({**sample, "s": round(dt, 6)})

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, st in sorted(self._stats.items(), key=lambda kv: -kv[1]["total_s"]):
            calls = st["calls"]
            out[name] = {
                "calls": calls,
                "total_s": round(st["total_s"], 6),
                "avg_s": round(st["total_s"] / calls, 6) if calls else 0.0,
                "max_s": round(st["max_s"], 6),
                "pct": round(st["total_s"] / max(self.elapsed_s(), 1e-12) * 100, 2),
                "samples": st["samples"],
            }
        return out

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started


def _current() -> OperatorTimingContext | None:
    return getattr(_thread_local, "op_timing", None)


def is_active() -> bool:
    """当前线程是否有活动的计时上下文（用于决定是否包计时代理）。"""
    return _ENABLED and _current() is not None


def begin() -> OperatorTimingContext | None:
    """开始一次评估的算子计时；返回上下文供 end 使用。"""
    if not _ENABLED:
        return None
    ctx = OperatorTimingContext()
    _thread_local.op_timing = ctx
    return ctx


def end(ctx: OperatorTimingContext | None) -> dict[str, Any] | None:
    """结束计时，返回聚合报告（无上下文/未开启时 None）。"""
    if ctx is None:
        return None
    _thread_local.op_timing = None
    return ctx.report()


def _make_wrapper(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        ctx = _current()
        if ctx is None:
            return fn(*args, **kwargs)
        ctx.enter(name)
        try:
            result = fn(*args, **kwargs)
        finally:
            sample: dict[str, Any] = {}
            if args:
                sample["arg0"] = _summary_param(args[0])
                if len(args) > 1:
                    sample["arg1"] = _summary_param(args[1])
                if len(args) > 2:
                    sample["arg2"] = _summary_param(args[2])
            if kwargs:
                sample["kwargs"] = {k: _summary_param(v) for k, v in list(kwargs.items())[:3]}
            ctx.leave(name, sample)
        return result

    wrapped.__name__ = name
    wrapped.__qualname__ = f"monitored::{name}"
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    return wrapped


def wrap_operator_namespace(ns: dict[str, Any]) -> dict[str, Any]:
    """返回命名空间副本：大写算子（DSL 约定）包计时代理，其余原样。"""
    if not _ENABLED:
        return ns
    out = dict(ns)
    for name, obj in ns.items():
        if name.startswith("_") or not name.isupper():
            continue
        if callable(obj):
            out[name] = _make_wrapper(name, obj)
    return out


def append_record(report: dict[str, Any] | None, *, extra: dict[str, Any] | None = None) -> None:
    """把一次评估的算子耗时追加到 JSONL（累计历史）。"""
    if not report:
        return
    try:
        _PROFILING_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "operators": report}
        if extra:
            rec.update(extra)
        with open(_PROFILING_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("算子监控 JSONL 写入失败（不影响使用）: %s", exc)


def read_accumulated(
    path: Path | None = None,
    *,
    top_k: int = 20,
    since_ts: float | None = None,
) -> dict[str, Any]:
    """读累计 JSONL，跨评估聚合 top 慢算子（供报表/前端）。"""
    p = Path(path) if path is not None else _PROFILING_LOG
    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "total_s": 0.0, "records": 0})
    try:
        if not p.is_file():
            return {"total_records": 0, "operators": {}}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts is not None and float(rec.get("ts", 0)) < since_ts:
                continue
            ops = rec.get("operators") or {}
            for name, st in ops.items():
                g = agg[name]
                g["calls"] += int(st.get("calls", 0))
                g["total_s"] += float(st.get("total_s", 0))
                g["records"] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("算子监控累计读取失败: %s", exc)
        return {"total_records": 0, "operators": {}}
    top = sorted(agg.items(), key=lambda kv: -kv[1]["total_s"])[:top_k]
    return {
        "total_records": sum(1 for _ in p.read_text(encoding="utf-8").splitlines()) if p.is_file() else 0,
        "operators": {
            name: {
                "calls": st["calls"],
                "total_s": round(st["total_s"], 3),
                "avg_s": round(st["total_s"] / st["calls"], 4) if st["calls"] else 0.0,
                "records": st["records"],
            }
            for name, st in top
        },
    }

# -*- coding: utf-8 -*-
"""研究记忆库聚合：按「数据面 / 算子」统计挖掘成功率（整体统计页图表数据源）。

口径（与注入 LLM 的 ``_structure_stats_block`` 保持一致，避免两处漂移）：
- 尝试 n = 记忆库条目数（一条 = 一次 evaluate/submit 落账）
- 有效 n_valid = n − eval_error（"没算出来"= 面板缺列/超时/参数错，不计入分母，
  否则一次数据故障会把所有面/算子的成功率一起拉低）
- 过线 = verdict ∈ POSITIVE_VERDICTS（promising / validated / candidate_approved /
  production_approved）；入库 = candidate_approved / production_approved
- 平均 |IC| 取 ``metrics_json.ic`` 的绝对值均值（缺 IC 的条目不计入该项）
- 数据面来自 ``facets_json``（老行为空时按表达式现算 ``expr_facets``）：多面条目在
  每个触及的面下各计一次，另给「跨面融合(≥2 面)」聚合桶
- 算子来自 ``operator_list_json``（同名算子一条只计一次）；缺列时按表达式正则兜底

纯只读聚合：DB 不存在/读失败时返回空结构，绝不抛给调用方（统计不阻断页面）。
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .constants import POSITIVE_VERDICTS
from .expressions import expr_facets

# 入库（真正躺进因子库）的 verdict
STORED_VERDICTS = frozenset({"candidate_approved", "production_approved"})
EVAL_ERROR_VERDICT = "eval_error"
FUSION_BUCKET = "跨面融合(≥2 面)"

# 表达式兜底扫算子：大写标识符 + 左括号（与 DSL 算子命名风格一致）
_OP_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\s*\(")
# 兜底扫描会误命中的非算子记号（契约参数取值等）
_OP_STOPWORDS = frozenset({"TRUE", "FALSE", "NONE", "NULL", "AND", "OR", "NOT"})


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """只读打开（WAL 库也不写盘）；失败退回普通连接。"""
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_rows(
    db_path: Path,
    run_ids: Sequence[str] | None,
) -> list[sqlite3.Row]:
    conn = _connect_ro(db_path)
    try:
        sql = (
            "SELECT expression, verdict, facets_json, operator_list_json, "
            "metrics_json, last_run_id, created_at FROM memory_entries "
            "WHERE expression IS NOT NULL AND expression != ''"
        )
        params: list[Any] = []
        if run_ids:
            ids = [str(r) for r in run_ids if r]
            if ids:
                sql += f" AND last_run_id IN ({','.join('?' * len(ids))})"
                params = ids
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _facets_of(facets_json: Any, expression: str) -> set[str]:
    facets: set[str] = set()
    if facets_json:
        try:
            loaded = json.loads(facets_json)
            if isinstance(loaded, list):
                facets = {str(x) for x in loaded if x}
        except (TypeError, ValueError, json.JSONDecodeError):
            facets = set()
    if not facets:
        try:
            facets = set(expr_facets(expression))
        except Exception:  # noqa: BLE001 - 识别失败按无面处理
            facets = set()
    return facets


def _ops_of(ops_json: Any, expression: str) -> set[str]:
    ops: set[str] = set()
    if ops_json:
        try:
            loaded = json.loads(ops_json)
            if isinstance(loaded, list):
                ops = {str(x).upper() for x in loaded if x}
        except (TypeError, ValueError, json.JSONDecodeError):
            ops = set()
    if not ops:
        ops = {
            m.group(1) for m in _OP_RE.finditer(str(expression or ""))
            if m.group(1) not in _OP_STOPWORDS
        }
    return ops


def _new_bucket() -> dict[str, Any]:
    return {
        "n": 0, "n_eval_error": 0, "positive": 0, "stored": 0,
        "ic_sum": 0.0, "ic_n": 0,
    }


def _accumulate(bucket: dict[str, Any], verdict: str, ic: float | None) -> None:
    bucket["n"] += 1
    if verdict == EVAL_ERROR_VERDICT:
        bucket["n_eval_error"] += 1
    if verdict in POSITIVE_VERDICTS:
        bucket["positive"] += 1
    if verdict in STORED_VERDICTS:
        bucket["stored"] += 1
    if ic is not None:
        bucket["ic_sum"] += abs(ic)
        bucket["ic_n"] += 1


def _finalize(name: str, bucket: dict[str, Any]) -> dict[str, Any]:
    n = bucket["n"]
    n_valid = n - bucket["n_eval_error"]
    ic_n = bucket["ic_n"]
    return {
        "name": name,
        "n": n,
        "n_valid": n_valid,
        "n_eval_error": bucket["n_eval_error"],
        "positive": bucket["positive"],
        "stored": bucket["stored"],
        # 成功率分母用"有效尝试"（剔除 eval_error），样本为 0 时 None 而非 0
        "rate": round(bucket["positive"] / n_valid, 4) if n_valid else None,
        "stored_rate": round(bucket["stored"] / n_valid, 4) if n_valid else None,
        "mean_abs_ic": round(bucket["ic_sum"] / ic_n, 5) if ic_n else None,
    }


def _entry_ic(metrics_json: Any) -> float | None:
    if not metrics_json:
        return None
    try:
        metrics = json.loads(metrics_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("ic")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # 过滤 NaN


def facet_operator_breakdown(
    db_path: Path | str,
    *,
    run_ids: Iterable[str] | None = None,
    min_attempts: int = 8,
    top_k: int = 12,
) -> dict[str, Any]:
    """按数据面 / 算子聚合成功率。

    Parameters
    ----------
    db_path : research_memory.db 路径。
    run_ids : 只统计这些 run 的条目（None/空 = 全库）。
    min_attempts : 桶的最小有效尝试数，低于此值的桶不返回（样本太小的比率不可信）。
    top_k : 每个维度最多返回的桶数（按尝试数降序）。

    Returns
    -------
    dict: {"scope", "baseline", "facets", "operators", "min_attempts"}；
    DB 缺失/读取失败时返回 ``{"error": ...}`` 形状的空结果。
    """
    path = Path(db_path)
    if not path.is_file():
        return {"error": "memory_db_missing", "scope": {}, "baseline": None,
                "facets": [], "operators": [], "min_attempts": min_attempts}
    run_id_list = [str(r) for r in (run_ids or []) if r]
    try:
        rows = _load_rows(path, run_id_list or None)
    except Exception as exc:  # noqa: BLE001 - 统计失败不阻断页面
        return {"error": f"{type(exc).__name__}: {exc}", "scope": {},
                "baseline": None, "facets": [], "operators": [],
                "min_attempts": min_attempts}

    baseline = _new_bucket()
    facet_buckets: dict[str, dict[str, Any]] = {}
    op_buckets: dict[str, dict[str, Any]] = {}
    fusion = _new_bucket()
    created_min: str | None = None
    created_max: str | None = None

    for row in rows:
        verdict = str(row["verdict"] or "")
        expression = str(row["expression"] or "")
        ic = _entry_ic(row["metrics_json"])
        _accumulate(baseline, verdict, ic)

        created = row["created_at"]
        if created:
            created_min = created if created_min is None else min(created_min, created)
            created_max = created if created_max is None else max(created_max, created)

        facets = _facets_of(row["facets_json"], expression)
        if len(facets) >= 2:
            _accumulate(fusion, verdict, ic)
        for facet in facets:
            _accumulate(facet_buckets.setdefault(facet, _new_bucket()), verdict, ic)
        for op in _ops_of(row["operator_list_json"], expression):
            _accumulate(op_buckets.setdefault(op, _new_bucket()), verdict, ic)

    def _rank(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        finalized = [_finalize(name, b) for name, b in buckets.items()]
        usable = [b for b in finalized if b["n_valid"] >= min_attempts]
        usable.sort(key=lambda b: (-b["n"], -(b["rate"] or 0)))
        return usable[:top_k]

    fusion_final = _finalize(FUSION_BUCKET, fusion)
    facets_out = _rank(facet_buckets)
    if fusion_final["n_valid"] >= min_attempts:
        facets_out = [fusion_final] + facets_out

    baseline_final = _finalize("全部尝试", baseline)
    return {
        "scope": {
            "n_entries": baseline_final["n"],
            "n_valid": baseline_final["n_valid"],
            "n_eval_error": baseline_final["n_eval_error"],
            "n_buckets_facets": len(facet_buckets),
            "n_buckets_operators": len(op_buckets),
            "created_min": created_min,
            "created_max": created_max,
            "filtered_runs": len(run_id_list),
        },
        "baseline": baseline_final,
        "facets": facets_out,
        "operators": _rank(op_buckets),
        "min_attempts": min_attempts,
    }

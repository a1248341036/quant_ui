# -*- coding: utf-8 -*-
"""数据面 / 算子成功率聚合测试（研究记忆库 → 整体统计页图表）。

覆盖：过线率口径（分母剔除 eval_error）、多面条目在各面下各计一次 + 融合桶、
算子去重计数、metrics_json.ic 平均绝对值、min_attempts 过滤、run_ids 窗口过滤、
DB 缺失时返回 error 形状而不是抛异常。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alphaagent.factor.mining.memory.analytics import (
    FUSION_BUCKET,
    facet_operator_breakdown,
)

_COLUMNS = (
    "id", "factor_name", "expression", "verdict", "metrics_json",
    "facets_json", "operator_list_json", "last_run_id", "created_at",
)


def _make_db(path: Path, rows: list[dict]) -> Path:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE memory_entries (
                id TEXT PRIMARY KEY,
                factor_name TEXT,
                expression TEXT,
                verdict TEXT,
                metrics_json TEXT,
                facets_json TEXT,
                operator_list_json TEXT,
                last_run_id TEXT,
                created_at TEXT
            )
            """
        )
        for i, r in enumerate(rows):
            conn.execute(
                f"INSERT INTO memory_entries ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(_COLUMNS))})",
                (
                    f"e{i}", r.get("name", f"f{i}"), r["expression"], r["verdict"],
                    json.dumps(r.get("metrics", {})),
                    json.dumps(r["facets"]) if r.get("facets") is not None else None,
                    json.dumps(r.get("ops")) if r.get("ops") is not None else None,
                    r.get("run_id", "run1"), "2026-09-01T00:00:00+00:00",
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def test_facet_operator_breakdown_rates_and_buckets(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "mem.db", [
        # 价量面 + 股东面（跨面融合），过线，算子显式声明
        {"expression": "A", "verdict": "promising", "facets": ["价量面", "股东面"],
         "ops": ["ts_mean", "cs_zscore"], "metrics": {"ic": 0.03}},
        # 价量面，未过线
        {"expression": "B", "verdict": "weak", "facets": ["价量面"],
         "ops": ["ts_mean"], "metrics": {"ic": -0.01}},
        # eval_error：不计入分母，但计入 n
        {"expression": "C", "verdict": "eval_error", "facets": ["价量面"],
         "ops": ["ts_mean"], "metrics": {}},
        # facets_json 缺失 → 按表达式现算（$holder_count_chg_pct → 股东面）
        {"expression": "RANK($holder_count_chg_pct)", "verdict": "weak",
         "facets": None, "ops": None, "metrics": {"ic": 0.02}},
    ])
    out = facet_operator_breakdown(db, min_attempts=1)

    scope = out["scope"]
    assert scope["n_entries"] == 4
    assert scope["n_valid"] == 3          # 1 条 eval_error 出分母
    assert scope["n_eval_error"] == 1

    # 基线：3 次有效尝试里 1 次过线
    assert out["baseline"]["rate"] == 0.3333
    assert out["baseline"]["mean_abs_ic"] == round((0.03 + 0.01 + 0.02) / 3, 5)

    facets = {r["name"]: r for r in out["facets"]}
    # 价量面命中 2 次有效尝试（A/B）其中 A 过线 → 50%
    assert facets["价量面"]["n_valid"] == 2
    assert facets["价量面"]["rate"] == 0.5
    # 缺失 facets_json 的行按表达式现算，落进股东面；股东面还含跨面因子 A
    assert facets["股东面"]["n_valid"] == 2
    assert facets["股东面"]["rate"] == 0.5
    # 融合桶只含 A
    assert FUSION_BUCKET in facets
    assert facets[FUSION_BUCKET]["n"] == 1
    assert facets[FUSION_BUCKET]["rate"] == 1.0

    ops = {r["name"]: r for r in out["operators"]}
    # TS_MEAN 命中 A/B/C 三条（C 计入 n 但不计入分母）
    assert ops["TS_MEAN"]["n"] == 3
    assert ops["TS_MEAN"]["n_valid"] == 2
    assert ops["CS_ZSCORE"]["n_valid"] == 1
    # 兜底正则从表达式扫出 RANK（operator_list_json 缺失时）
    assert "RANK" in ops


def test_min_attempts_filters_and_run_filter(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "mem.db", [
        {"expression": "A", "verdict": "promising", "facets": ["价量面"],
         "ops": ["ts_mean"], "run_id": "runA"},
        {"expression": "B", "verdict": "weak", "facets": ["基本面"],
         "ops": ["divide"], "run_id": "runB"},
    ])
    # min_attempts=2：两个面各只有 1 次尝试 → 都不返回，融合桶也不返回
    out = facet_operator_breakdown(db, min_attempts=2)
    assert out["facets"] == []
    assert out["operators"] == []
    assert out["scope"]["n_buckets_facets"] == 2  # 桶存在但被样本门槛滤掉

    # run 窗口过滤：只看 runA
    out_a = facet_operator_breakdown(db, run_ids=["runA"], min_attempts=1)
    assert out_a["scope"]["n_entries"] == 1
    assert out_a["scope"]["filtered_runs"] == 1
    assert [r["name"] for r in out_a["facets"]] == ["价量面"]


def test_missing_db_returns_error_payload(tmp_path: Path) -> None:
    out = facet_operator_breakdown(tmp_path / "nope.db")
    assert out["error"] == "memory_db_missing"
    assert out["facets"] == [] and out["operators"] == []

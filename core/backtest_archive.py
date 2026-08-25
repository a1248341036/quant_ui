"""回测结果归档：PG `backtest_runs` 表读写（参数/净值/指标可追溯）。

PG 不可用时静默跳过归档，不影响回测主流程。
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from . import sqldb as pg


def _data_snapshot_hash(params: dict | None) -> str | None:
    from .store import ETF_PANEL_FILE, FUND_NAV_FILE, PANEL_FILE
    universe = str((params or {}).get("universe", ""))
    path = ETF_PANEL_FILE if universe.upper() == "ETF" else (
        FUND_NAV_FILE if "基金" in universe else PANEL_FILE)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(obj: Any) -> Any:
    """把 numpy 标量/NaN/日期转成可 JSON 序列化的类型。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(_clean(obj), ensure_ascii=False)


def save_run(kind: str = "backtest",
             params: dict | None = None,
             metrics: dict | None = None,
             bench_metrics: dict | None = None,
             nav: list | None = None,
             bench: list | None = None,
             drawdown: list | None = None,
             holdings: list | None = None,
             trades: list | None = None,
             summary: dict | None = None,
             data_version: str | None = None,
             data_snapshot_hash: str | None = None,
             error: str | None = None) -> int | None:
    """写入一次回测运行记录，返回 run_id；PG 不可用时返回 None。"""
    if not pg.configured():
        return None
    sql = """
        INSERT INTO backtest_runs
            (kind, params, metrics, bench_metrics, nav, bench, drawdown,
            holdings, trades, summary, data_version, data_snapshot_hash, error)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING run_id
    """
    try:
        with pg.get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (
                kind,
                _dumps(params) or "{}",
                _dumps(metrics),
                _dumps(bench_metrics),
                _dumps(nav),
                _dumps(bench),
                _dumps(drawdown),
                _dumps(holdings),
                _dumps(trades),
                _dumps(summary),
                data_version,
                data_snapshot_hash or _data_snapshot_hash(params),
                error,
            ))
            run_id = int(cur.fetchone()[0])
        return run_id
    except Exception as exc:  # 归档失败不阻断回测
        print(f"[backtest_archive] 归档失败: {exc}", flush=True)
        return None


def list_runs(kind: str | None = None, limit: int = 50) -> pd.DataFrame:
    """最近运行列表（不含净值/交易大字段）。"""
    if not pg.configured():
        return pd.DataFrame(columns=["run_id", "kind", "created_at", "params",
                                     "summary", "data_version", "error"])
    sql = """
        SELECT run_id, kind, created_at, params, summary, data_version,
               data_snapshot_hash, error
        FROM backtest_runs
    """
    if kind:
        sql += " WHERE kind = %s ORDER BY created_at DESC LIMIT %s"
        df = pg.query_df(sql, (kind, int(limit)))
    else:
        sql += " ORDER BY created_at DESC LIMIT %s"
        df = pg.query_df(sql, (int(limit),))
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", errors="coerce")
    for col in ("params", "summary"):
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def backfill_excess_metrics() -> int:
    """Derive missing active metrics from archived strategy and benchmark NAVs."""
    if not pg.configured():
        return 0
    from .metrics import compute_excess_metrics

    df = pg.query_df("SELECT run_id, metrics, nav, bench, summary FROM backtest_runs")
    updated = 0
    with pg.get_conn() as conn, conn.cursor() as cur:
        for row in df.itertuples(index=False):
            try:
                metrics = json.loads(row.metrics) if isinstance(row.metrics, str) else (row.metrics or {})
                if "超额年化" in metrics and "超额夏普" in metrics:
                    continue
                nav = json.loads(row.nav) if isinstance(row.nav, str) else row.nav
                bench = json.loads(row.bench) if isinstance(row.bench, str) else row.bench
                if not nav or not bench:
                    continue
                nav_s = pd.Series({pd.Timestamp(p["date"]): p["value"] for p in nav})
                bench_s = pd.Series({pd.Timestamp(p["date"]): p["value"] for p in bench})
                metrics.update(compute_excess_metrics(nav_s, bench_s))
                summary = json.loads(row.summary) if isinstance(row.summary, str) else (row.summary or {})
                summary["excess_annual"] = metrics["超额年化"]
                summary["excess_sharpe"] = metrics["超额夏普"]
                cur.execute("UPDATE backtest_runs SET metrics = %s, summary = %s WHERE run_id = %s",
                            (_dumps(metrics), _dumps(summary), int(row.run_id)))
                updated += 1
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
    return updated


def get_run(run_id: int) -> dict | None:
    """取一条完整记录。"""
    if not pg.configured():
        return None
    df = pg.query_df(
        "SELECT * FROM backtest_runs WHERE run_id = %s", (int(run_id),))
    if df.empty:
        return None
    row = df.iloc[0]
    out: dict = {}
    for col in df.columns:
        v = row[col]
        if isinstance(v, str) and col not in ("created_at", "data_version", "error", "kind"):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        out[col] = v
    out["created_at"] = str(out.get("created_at"))
    out["run_id"] = int(out["run_id"]) if out.get("run_id") is not None else None
    # 老记录/CLI 记录可能缺大字段：统一补默认空结构，前端无需再判 None。
    for col in ("nav", "bench", "drawdown", "holdings", "trades"):
        if out.get(col) is None:
            out[col] = []
    for col in ("metrics", "bench_metrics", "params", "summary"):
        if out.get(col) is None:
            out[col] = {}
    return out


def delete_run(run_id: int) -> bool:
    if not pg.configured():
        return False
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM backtest_runs WHERE run_id = %s", (int(run_id),))
        # DuckDB 不报告 rowcount（返回 -1），执行成功即视为删除完成
        return True

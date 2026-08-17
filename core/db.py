"""DuckDB 查询层：直接在 data/ 下的 parquet/csv 上提供 SQL 视图。

不复制数据、不依赖外部服务；迁移 = 整个项目目录拷走。
视图基于文件路径，每次查询读取文件最新内容。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

from .store import DATA_DIR, INDEX_FILE, PANEL_FILE, TECH_FILE, UNIVERSE_FILE


_conn = None
_lock = threading.Lock()
_registered = False

PG_PARQUET_DIR = DATA_DIR / "pg_parquet"
PG_TABLE_NAMES = [
    "stock_daily",
    "stock_basic",
    "share_float",
    "fina_indicator",
    "income",
    "balancesheet",
    "cashflow",
    "dividend",
    "stk_surv",
    "forecast",
    "express",
    "namechange",
    "trade_cal",
    "report_rc",
    "index_weight",
]


def _view_sql(name: str, file: Path, columns: dict[str, str] | None = None) -> str | None:
    if not file.exists():
        return None
    path = str(file).replace("'", "''")
    if name == "panel":
        return f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{path}')"
    cols = ", ".join(f"'{k}': '{v}'" for k, v in (columns or {}).items())
    return (
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv('{path}', header=true, columns={{ {cols} }})"
    )


def _pg_parquet_view_sql(name: str) -> str | None:
    path = PG_PARQUET_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    p = str(path).replace("'", "''")
    return f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{p}')"


def _register(conn) -> None:
    global _registered
    stmts = [
        _view_sql("panel", PANEL_FILE),
        _view_sql("universe", UNIVERSE_FILE, {"code": "VARCHAR", "name": "VARCHAR"}),
        _view_sql("tech", TECH_FILE, {"code": "VARCHAR", "name": "VARCHAR", "industry": "VARCHAR"}),
        _view_sql(
            "index",
            INDEX_FILE,
            {"date": "DATE", "code": "VARCHAR", "name": "VARCHAR", "open": "DOUBLE", "close": "DOUBLE"},
        ),
    ]
    stmts.extend(_pg_parquet_view_sql(name) for name in PG_TABLE_NAMES)
    for stmt in stmts:
        if stmt:
            conn.execute(stmt)
    _registered = True


def _ensure_conn_unlocked():
    """假设调用方已持有 _lock。"""
    global _conn
    if duckdb is None:
        raise RuntimeError("duckdb 未安装，请先 pip install duckdb")
    if _conn is None:
        _conn = duckdb.connect(str(DATA_DIR / "duck.db"))
        _register(_conn)
    return _conn


def get_conn():
    """进程内 DuckDB 单例（文件视图模式，线程安全由全局锁保证）。"""
    with _lock:
        return _ensure_conn_unlocked()


def refresh_views() -> None:
    """文件被刷新后重新注册视图（例如每日增量更新后）。"""
    global _registered
    with _lock:
        _registered = False
        _register(_ensure_conn_unlocked())


def query(sql: str, params: Sequence | None = None) -> pd.DataFrame:
    """执行 SQL，返回 pandas DataFrame。"""
    conn = get_conn()
    with _lock:
        return conn.execute(sql, params or []).df()


def tables() -> list[str]:
    conn = get_conn()
    with _lock:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1"
        ).fetchall()
    return [r[0] for r in rows]

"""DuckDB 查询层：直接在 data/ 下的 parquet/csv 上提供 SQL 视图。

不复制数据、不依赖外部服务；迁移 = 整个项目目录拷走。
视图基于文件路径，每次查询读取文件最新内容。

并发说明（3.6G 小机）：读进程每次打开只读连接、用完即关；写进程
（每日刷新/本地导入）把新视图写进临时 duck.db 后原子替换正式文件，
避免读进程常驻连接把 duck.db 锁死导致“DuckDB 视图刷新失败”。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

from .store import DATA_DIR, INDEX_FILE, PANEL_FILE, TECH_FILE, UNIVERSE_FILE


PG_PARQUET_DIR = DATA_DIR / "pg_parquet"
DUCKDB_FILE = DATA_DIR / "duck.db"
DUCKDB_WAL = DATA_DIR / "duck.db.wal"
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


def _build_db_file() -> None:
    """把当前 parquet/csv 视图写入临时 duck.db，再原子替换正式文件。"""
    if duckdb is None:
        raise RuntimeError("duckdb 未安装，请先 pip install duckdb")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DUCKDB_FILE.with_name(f".{DUCKDB_FILE.name}.{os.getpid()}.tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        conn = duckdb.connect(str(tmp))
        try:
            _register(conn)
        finally:
            conn.close()
        os.replace(tmp, DUCKDB_FILE)
        # 旧文件残留的 WAL 不属于新文件；读进程只读打开不会写盘
        if DUCKDB_WAL.exists():
            try:
                DUCKDB_WAL.unlink()
            except OSError:
                pass
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _open(read_only: bool = True):
    if duckdb is None:
        raise RuntimeError("duckdb 未安装，请先 pip install duckdb")
    return duckdb.connect(str(DUCKDB_FILE), read_only=read_only)


def get_conn():
    """兼容接口：返回一个只读连接，调用方负责 close。"""
    if not DUCKDB_FILE.exists():
        _build_db_file()
    return _open(read_only=True)


def refresh_views() -> None:
    """文件被刷新后重建视图文件（原子替换 duck.db，不依赖已有连接）。"""
    _build_db_file()


def query(sql: str, params: Sequence | None = None) -> pd.DataFrame:
    """执行 SQL，返回 pandas DataFrame（每次新建只读连接，用完即关）。"""
    if not DUCKDB_FILE.exists():
        _build_db_file()
    last_exc = None
    for attempt in range(3):
        try:
            conn = _open(read_only=True)
            try:
                return conn.execute(sql, params or []).df()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - 文件可能正被原子替换
            last_exc = exc
            time.sleep(0.1 * (attempt + 1))
    raise last_exc


def tables() -> list[str]:
    if not DUCKDB_FILE.exists():
        _build_db_file()
    conn = _open(read_only=True)
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()

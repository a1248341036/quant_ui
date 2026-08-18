# -*- coding: utf-8 -*-
"""DuckDB 本地存储层（替代 PostgreSQL/TimescaleDB 写库）。

- 单文件数据库：data/quant.duckdb（可用 QUANT_DUCKDB_PATH 覆盖）
- API 与旧 core/pg.py 对齐：configured/get_conn/exec_sql/create_schema/df_to_pg/query_df
- 短连接模式：每次操作临时打开文件，避免常驻连接锁住文件导致
  其他进程（paper_trade.py 等定时脚本）无法打开
- 写库表（backtest_runs/strategy_pool/ledger/paper_*）由 db/schema_duckdb.sql 创建
- 行情数据不落 DuckDB：继续读 data/ 下 parquet（QUANT_DATA_SOURCE=pg_parquet/panel）
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = Path(os.getenv("QUANT_DUCKDB_PATH",
                         str(PROJECT_ROOT / "data" / "quant.duckdb"))).expanduser()
SCHEMA_FILE = PROJECT_ROOT / "db" / "schema_duckdb.sql"


def configured() -> bool:
    """DuckDB 本地文件始终可用。"""
    return True


def _connect(retries: int = 8, delay: float = 0.25) -> duckdb.DuckDBPyConnection:
    """打开 DuckDB 文件；文件被其他进程短暂占用时重试。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for i in range(retries):
        try:
            return duckdb.connect(str(DB_PATH))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(delay * (i + 1))
    raise RuntimeError(f"无法打开 DuckDB 文件 {DB_PATH}: {last}")


def _adapt(sql: str) -> str:
    """psycopg 的 %s 参数占位符 -> DuckDB 的 ?。"""
    return sql.replace("%s", "?")


class _Cursor:
    """把 DuckDB 连接包装成 psycopg 风格 cursor（execute/fetchone/fetchall/description/rowcount）。"""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._result = None
        self.description = None
        self.rowcount = -1

    def execute(self, sql: str, params=None):
        if params is not None:
            params = tuple(params)
        self._result = self._conn.execute(_adapt(sql), params)
        try:
            self.description = self._result.description
        except Exception:  # noqa: BLE001
            self.description = None
        try:
            self.rowcount = self._result.rowcount
        except Exception:  # noqa: BLE001
            self.rowcount = -1
        return self

    def fetchone(self):
        if self._result is None:
            return None
        return self._result.fetchone()

    def fetchall(self):
        if self._result is None:
            return []
        return self._result.fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ConnWrapper:
    """调用方可见的连接对象；with 退出时关闭文件句柄。"""

    def __init__(self):
        self._conn = _connect()

    def cursor(self) -> _Cursor:
        return _Cursor(self._conn)

    def execute(self, sql: str, params=None):
        _Cursor(self._conn).execute(sql, params)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        return False


def get_conn() -> _ConnWrapper:
    """DuckDB 连接包装（可作 context manager，可 cursor()）。"""
    return _ConnWrapper()


def exec_sql(sql: str) -> None:
    with get_conn() as conn:
        conn.execute(_adapt(sql))


def _split_statements(sql: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            out.append("\n".join(buf).strip().rstrip(";"))
            buf = []
    if buf:
        out.append("\n".join(buf).strip().rstrip(";"))
    return [s for s in out if s]


def create_schema() -> None:
    """执行 db/schema_duckdb.sql（幂等）。"""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"schema 文件不存在: {SCHEMA_FILE}")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with get_conn() as conn:
        for stmt in _split_statements(sql):
            try:
                conn.execute(stmt)
            except Exception as exc:  # noqa: BLE001
                print(f"[duckdb] 建表语句失败（忽略）: {stmt[:60]} ... {exc}", flush=True)


def df_to_pg(df: pd.DataFrame, table: str, if_exists: str = "append") -> None:
    """DataFrame 写入 DuckDB 表（if_exists: append/replace）。"""
    if df is None or df.empty:
        return
    with get_conn() as conn:
        con = conn._conn
        if if_exists == "replace":
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
            return
        con.register("_df_tmp", df)
        try:
            cols = ", ".join(f'"{c}"' for c in df.columns)
            con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _df_tmp")
        finally:
            con.unregister("_df_tmp")


def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    if params is not None:
        params = tuple(params)
    with get_conn() as conn:
        return conn._conn.execute(_adapt(sql), params).df()
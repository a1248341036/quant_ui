# -*- coding: utf-8 -*-
"""SQLite 本地存储层（业务数据统一落 data/quant.db，替代 PostgreSQL）。

- 单文件数据库：data/quant.db（可用 QUANT_SQLITE_PATH 覆盖）
- API 与旧 core/duckdb.py 对齐：configured/get_conn/exec_sql/create_schema/df_to_pg/query_df
- WAL 模式 + busy_timeout：多进程可并发读，写者排队等待而非锁死
- 写库表（backtest_runs/strategy_pool/ledger/paper_*）由 db/schema_sqlite.sql 创建
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = Path(os.getenv("QUANT_SQLITE_PATH",
                         str(PROJECT_ROOT / "data" / "quant.db"))).expanduser()
SCHEMA_FILE = PROJECT_ROOT / "db" / "schema_sqlite.sql"


def configured() -> bool:
    """SQLite 本地文件始终可用。"""
    return True


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # autocommit，与旧层行为一致
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass
    return conn


def _adapt(sql: str) -> str:
    """psycopg 的 %s 参数占位符 -> SQLite 的 ?。"""
    return sql.replace("%s", "?")


class _Cursor:
    """把 sqlite3 cursor 包装成 psycopg 风格（execute/fetchone/fetchall/description/rowcount）。"""

    def __init__(self, conn: sqlite3.Connection):
        self._cur = conn.cursor()
        self.description = None
        self.rowcount = -1

    def execute(self, sql: str, params=None):
        if params is not None:
            params = tuple(params)
        if params is None:
            self._cur.execute(_adapt(sql))
        else:
            self._cur.execute(_adapt(sql), params)
        try:
            self.description = self._cur.description
        except Exception:  # noqa: BLE001
            self.description = None
        try:
            self.rowcount = self._cur.rowcount
        except Exception:  # noqa: BLE001
            self.rowcount = -1
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ConnWrapper:
    """最小连接对象，支持 with 与 cursor()；autocommit，退出时关闭。"""

    def __init__(self):
        self._conn = _connect()

    def cursor(self) -> _Cursor:
        return _Cursor(self._conn)

    def execute(self, sql: str, params=None):
        _Cursor(self._conn).execute(sql, params)
        return self

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def get_conn() -> _ConnWrapper:
    return _ConnWrapper()


def exec_sql(sql: str) -> None:
    con = _connect()
    try:
        con.execute(_adapt(sql))
    finally:
        con.close()


def _split_statements(sql: str) -> list[str]:
    """按分号切分建表语句（忽略 -- 注释行）。"""
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
    """执行 db/schema_sqlite.sql（幂等）。"""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"schema 文件不存在: {SCHEMA_FILE}")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    con = _connect()
    try:
        for stmt in _split_statements(sql):
            try:
                con.execute(stmt)
            except Exception as exc:  # noqa: BLE001
                print(f"[sqldb] 建表语句失败（忽略）: {stmt[:60]} ... {exc}", flush=True)
        try:
            con.execute("ALTER TABLE backtest_runs ADD COLUMN data_snapshot_hash TEXT")
        except Exception:
            pass
    finally:
        con.close()


def df_to_pg(df: pd.DataFrame, table: str, if_exists: str = "append") -> None:
    """DataFrame 写入 SQLite 表（if_exists: append/replace）。"""
    if df is None or df.empty:
        return
    con = _connect()
    try:
        if if_exists == "replace":
            con.execute(f'DROP TABLE IF EXISTS "{table}"')
        df.to_sql(table, con, index=False, if_exists="append")
    finally:
        con.close()


def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    con = _connect()
    try:
        if params is not None:
            params = tuple(params)
        if params is None:
            return pd.read_sql_query(_adapt(sql), con)
        return pd.read_sql_query(_adapt(sql), con, params=params)
    finally:
        con.close()

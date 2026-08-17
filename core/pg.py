"""PostgreSQL/TimescaleDB 客户端（连接串从 .env 的 PG_DSN 读取）。"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pandas as pd

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover
    create_engine = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

PG_DSN = os.getenv("PG_DSN", "").strip()
SCHEMA_FILE = PROJECT_ROOT / "db" / "schema.sql"

_engine = None
_lock = threading.Lock()


def configured() -> bool:
    return bool(PG_DSN)


def get_engine():
    global _engine
    if not configured():
        raise RuntimeError("未配置 PG_DSN（quant_ui/.env）")
    if create_engine is None:
        raise RuntimeError("sqlalchemy 未安装")
    with _lock:
        if _engine is None:
            dsn = PG_DSN
            if dsn.startswith("postgresql://"):
                dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
            _engine = create_engine(dsn, pool_pre_ping=True)
    return _engine


def get_conn():
    """psycopg3 原生连接（autocommit）。"""
    if psycopg is None:
        raise RuntimeError("psycopg 未安装")
    conn = psycopg.connect(PG_DSN, autocommit=True)
    return conn


def exec_sql(sql: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def create_schema() -> None:
    """执行 db/schema.sql（幂等）。"""
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(f"schema 文件不存在: {SCHEMA_FILE}")
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def df_to_pg(df: pd.DataFrame, table: str, if_exists: str = "append") -> None:
    """DataFrame 写入 PG 表（sqlalchemy 批量写）。"""
    if df is None or df.empty:
        return
    df.to_sql(table, get_engine(), if_exists=if_exists, index=False, method="multi", chunksize=5000)


def query_df(sql: str, params: tuple | None = None) -> pd.DataFrame:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)

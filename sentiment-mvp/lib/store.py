# -*- coding: utf-8 -*-
"""SQLite 增量存储：按 (code, publish_time, title) 去重，支持增量合并与导出。"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
  key          TEXT PRIMARY KEY,
  code         TEXT NOT NULL,
  publish_time TEXT,
  media        TEXT,
  title        TEXT,
  content      TEXT,
  url          TEXT,
  source       TEXT,
  label        TEXT,
  score        REAL,
  pos_hits     REAL,
  neg_hits     REAL,
  fetched_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_code ON articles(code);
CREATE INDEX IF NOT EXISTS idx_articles_pub ON articles(publish_time);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
"""

COLUMNS = ["key", "code", "publish_time", "media", "title", "content", "url",
           "source", "label", "score", "pos_hits", "neg_hits", "fetched_at"]


def key_of(article):
    raw = f"{article.get('code')}|{article.get('publish_time')}|{article.get('title')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def existing_keys(db_path):
    if not Path(db_path).exists():
        return set()
    conn = connect(db_path)
    try:
        return {r[0] for r in conn.execute("SELECT key FROM articles")}
    finally:
        conn.close()


def upsert_rows(db_path, rows):
    """rows: [{code,publish_time,media,title,content,url,source,label,score,pos_hits,neg_hits}]。
    返回 (插入条数, 已存在条数)。"""
    conn = connect(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    inserted = 0
    existing = 0
    try:
        with conn:
            for a in rows:
                key = key_of(a)
                cur = conn.execute("SELECT 1 FROM articles WHERE key=?", (key,))
                if cur.fetchone():
                    existing += 1
                    continue
                conn.execute(
                    f"INSERT INTO articles ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                    (
                        key, a.get("code", ""), a.get("publish_time", ""),
                        a.get("media", ""), a.get("title", ""), a.get("content", ""),
                        a.get("url", ""), a.get("source", "em"), a.get("label", "neutral"),
                        float(a.get("score", 0.0)), float(a.get("pos_hits", 0) or 0),
                        float(a.get("neg_hits", 0) or 0), now,
                    ),
                )
                inserted += 1
    finally:
        conn.close()
    return inserted, existing


def load_articles(db_path, codes=None, source=None) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = connect(db_path)
    try:
        sql = "SELECT code,publish_time,media,title,content,url,source,label,score,pos_hits,neg_hits,fetched_at FROM articles"
        params = []
        conds = []
        if codes:
            conds.append("code IN (%s)" % ",".join("?" * len(codes)))
            params.extend(codes)
        if source:
            conds.append("source=?")
            params.append(source)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


def stats(db_path):
    if not Path(db_path).exists():
        return {"total": 0, "by_source": {}, "by_label": {}, "min_time": None, "max_time": None}
    conn = connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        by_source = dict(conn.execute("SELECT source, COUNT(*) FROM articles GROUP BY source").fetchall())
        by_label = dict(conn.execute("SELECT label, COUNT(*) FROM articles GROUP BY label").fetchall())
        row = conn.execute("SELECT MIN(publish_time), MAX(publish_time) FROM articles WHERE publish_time != ''").fetchone()
        return {
            "total": total,
            "by_source": by_source,
            "by_label": by_label,
            "min_time": row[0] if row else None,
            "max_time": row[1] if row else None,
        }
    finally:
        conn.close()

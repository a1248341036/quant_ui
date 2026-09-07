# -*- coding: utf-8 -*-
"""factor_index.db —— 因子中台索引。

单一 factors 表：uid（见 identity.factor_uid）为主键，汇总因子在候选库/正式库/
研究记忆各存储中的位置与生命周期状态。索引是**派生物**（事实源 = registry JSON /
FactorZoo / 研究记忆库）：写点双写失败不阻断主流程，rebuild() 全量重建自愈。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from alphaagent.factor.identity import expr_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS factors (
    uid TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    expr TEXT,
    expr_hash TEXT,
    family TEXT,
    facets_json TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    promotion_status TEXT,
    candidate_registry TEXT,
    production_registry TEXT,
    dsl_path TEXT,
    zoo_path TEXT,
    memory_db TEXT,
    first_seen_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_factors_name ON factors(name);
CREATE INDEX IF NOT EXISTS idx_factors_status ON factors(status);
"""

_COLS = (
    "uid", "name", "expr", "expr_hash", "family", "facets_json", "status",
    "promotion_status", "candidate_registry", "production_registry",
    "dsl_path", "zoo_path", "memory_db", "first_seen_at", "updated_at",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FactorIndex:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _open(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(_SCHEMA)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_factor(
        self,
        *,
        uid: str,
        name: str,
        expr: str | None = None,
        family: str | None = None,
        facets: list[str] | None = None,
        status: str | None = None,
        promotion_status: str | None = None,
        candidate_registry: str | None = None,
        production_registry: str | None = None,
        dsl_path: str | None = None,
        zoo_path: str | None = None,
        memory_db: str | None = None,
    ) -> None:
        now = _now()
        with self._open() as conn:
            row = conn.execute("SELECT * FROM factors WHERE uid = ?", (uid,)).fetchone()
            if row is None:
                rec: dict[str, Any] = {
                    "uid": uid, "name": name, "status": status or "candidate",
                    "first_seen_at": now, "updated_at": now,
                }
            else:
                rec = dict(row)
                rec["updated_at"] = now
            for key, val in (
                ("name", name), ("expr", expr), ("family", family), ("status", status),
                ("promotion_status", promotion_status), ("candidate_registry", candidate_registry),
                ("production_registry", production_registry), ("dsl_path", dsl_path),
                ("zoo_path", zoo_path), ("memory_db", memory_db),
            ):
                if val is not None:
                    rec[key] = val
            if facets:
                rec["facets_json"] = json.dumps(facets, ensure_ascii=False)
            if expr:
                rec["expr_hash"] = expr_hash(expr)
            if row is None:
                conn.execute(
                    f"INSERT INTO factors ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})",
                    tuple(rec.get(c) for c in _COLS),
                )
            else:
                sets = ", ".join(f"{c} = ?" for c in _COLS if c not in ("uid", "first_seen_at"))
                args = tuple(rec.get(c) for c in _COLS if c not in ("uid", "first_seen_at"))
                conn.execute(f"UPDATE factors SET {sets} WHERE uid = ?", (*args, uid))

    def delete_factor(self, uid: str) -> bool:
        with self._open() as conn:
            cursor = conn.execute("DELETE FROM factors WHERE uid = ?", (uid,))
            return cursor.rowcount > 0

    def get(self, uid: str) -> dict[str, Any] | None:
        with self._open() as conn:
            row = conn.execute("SELECT * FROM factors WHERE uid = ?", (uid,)).fetchone()
            return dict(row) if row else None

    def by_name(self, name: str) -> list[dict[str, Any]]:
        with self._open() as conn:
            rows = conn.execute("SELECT * FROM factors WHERE name = ? ORDER BY status", (name,)).fetchall()
            return [dict(r) for r in rows]

    def list_factors(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._open() as conn:
            if status:
                rows = conn.execute("SELECT * FROM factors WHERE status = ? ORDER BY name", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM factors ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        with self._open() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM factors").fetchone()[0])

    def rebuild(self, rows: Iterable[dict[str, Any]]) -> int:
        """全量重建（索引是派生物）：清空后逐条 upsert，返回行数。"""
        n = 0
        with self._open() as conn:
            conn.execute("DELETE FROM factors")
        for row in rows:
            self.upsert_factor(**row)
            n += 1
        return n


_DEFAULT: FactorIndex | None = None


def get_factor_index() -> FactorIndex:
    """进程级默认索引（artifacts/alphaagent/factor_index.db）。"""
    global _DEFAULT
    if _DEFAULT is None:
        from alphaagent.core.paths import FACTOR_INDEX_PATH

        _DEFAULT = FactorIndex(FACTOR_INDEX_PATH)
    return _DEFAULT

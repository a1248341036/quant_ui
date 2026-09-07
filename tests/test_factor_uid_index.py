# -*- coding: utf-8 -*-
"""因子中台 ID + factor_index 测试：uid 派生、v5 迁移回填、purge_by_uid、索引读写。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alphaagent.factor.identity import expr_hash, factor_uid
from alphaagent.factor.index import FactorIndex
from alphaagent.factor.mining.infra.registry_io import write_candidate_registry
from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.types import IngestPolicy


def test_factor_uid_deterministic():
    assert factor_uid("abc") == factor_uid("abc")
    assert factor_uid("abc") != factor_uid("abd")
    assert factor_uid("abc").startswith("f")
    assert len(factor_uid("abc")) == 17  # 'f' + 16 hex


def test_expr_hash_ignores_whitespace():
    assert expr_hash("a = $close\nb = RANK(a)") == expr_hash("a=$close b=RANK(a)")


def test_write_candidate_registry_stamps_uid_and_index(tmp_path: Path):
    from alphaagent.factor.index import get_factor_index

    registry_path = tmp_path / "candidate" / "mining_candidate_registry.json"
    expr_dir = tmp_path / "candidate" / "expressions"
    rp, dp = write_candidate_registry(
        registry_path,
        factor_id="demo_factor",
        name="demo_factor",
        expr="RANK($close)",
        expr_dir=expr_dir,
        repo_root=tmp_path,
        policy=IngestPolicy(),
        metrics={"ic": 0.03},
        similarity=None,
    )
    import json

    reg = json.loads(Path(rp).read_text(encoding="utf-8"))
    assert reg["demo_factor"]["factor_uid"] == factor_uid("demo_factor")

    # 双写索引（默认路径 get_factor_index()——测试里直接校验默认索引文件被写入）
    row = get_factor_index().get(factor_uid("demo_factor"))
    assert row is not None
    assert row["name"] == "demo_factor"
    assert row["status"] == "candidate"
    assert row["expr_hash"] == expr_hash("RANK($close)")
    # 清理默认索引里的测试行，避免污染真实索引
    get_factor_index().delete_factor(factor_uid("demo_factor"))


def test_memory_v5_migration_and_purge_by_uid(tmp_path: Path):
    db = tmp_path / "memory.db"
    # 用 store 自身 DDL 建全量 schema，再手动降级为 v4 形态（删 uid 列/维表、版本号改 4）
    seed = ResearchMemoryStore(db)
    with seed._open() as conn:
        conn.execute("DROP INDEX IF EXISTS idx_memory_entries_factor_uid")
        conn.execute("ALTER TABLE memory_entries DROP COLUMN factor_uid")
        conn.execute("DROP TABLE IF EXISTS memory_factors")
        conn.execute("UPDATE store_meta SET v='4' WHERE k='data_version'")
    raw = sqlite3.connect(str(db))
    raw.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, created_at, updated_at) "
        "VALUES ('e1', 'fac_x', 'RANK($close)', 'promising', '2026-01-01', '2026-01-02')"
    )
    raw.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, created_at, updated_at) "
        "VALUES ('e2', 'fac_x', 'RANK($close)', 'candidate_approved', '2026-01-03', '2026-01-04')"
    )
    raw.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, created_at, updated_at) "
        "VALUES ('e3', 'fac_y', 'RANK($open)', 'rejected', '2026-01-05', '2026-01-06')"
    )
    raw.execute(
        "INSERT INTO memory_observations (entry_id, run_id, observed_at, stage, verdict) "
        "VALUES ('e1', 'run1', '2026-01-02', 'train', 'promising')"
    )
    raw.commit()
    raw.close()

    store = ResearchMemoryStore(db)
    from alphaagent.factor.mining.memory.constants import DATA_VERSION

    with store._open() as conn:
        # 迁移触发：data_version 升到当前版，uid 回填完成
        version = conn.execute("SELECT v FROM store_meta WHERE k='data_version'").fetchone()[0]
        assert version == DATA_VERSION
        n_dim = conn.execute("SELECT COUNT(*) FROM memory_factors").fetchone()[0]
        assert n_dim == 2  # fac_x / fac_y
        n_missing = conn.execute(
            "SELECT COUNT(*) FROM memory_entries WHERE factor_uid IS NULL"
        ).fetchone()[0]
        assert n_missing == 0
        # 同名条目共享同一 uid
        uids = {
            r[0]
            for r in conn.execute(
                "SELECT factor_uid FROM memory_entries WHERE factor_name='fac_x'"
            ).fetchall()
        }
        assert len(uids) == 1 and uids.pop() == factor_uid("fac_x")

    # purge by uid：fac_x 两条删除，fac_y 保留；observations 级联；维表孤儿清理
    deleted = store.purge_factor(factor_uids=[factor_uid("fac_x")])
    assert deleted == 2
    with store._open() as conn:
        remaining = {r[0] for r in conn.execute("SELECT factor_name FROM memory_entries").fetchall()}
        assert remaining == {"fac_y"}
        obs = conn.execute("SELECT COUNT(*) FROM memory_observations").fetchone()[0]
        assert obs == 0
        dim = {r[0] for r in conn.execute("SELECT factor_name FROM memory_factors").fetchall()}
        assert dim == {"fac_y"}

    # 幂等：重开 store 不重复回填/不报错
    store2 = ResearchMemoryStore(db)
    with store2._open() as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_factors").fetchone()[0] == 1


def test_factor_index_roundtrip_and_rebuild(tmp_path: Path):
    idx = FactorIndex(tmp_path / "factor_index.db")
    uid = factor_uid("f_a")
    idx.upsert_factor(uid=uid, name="f_a", expr="RANK($close)", status="candidate", facets=["价量面"])
    idx.upsert_factor(
        uid=uid, name="f_a", status="production", production_registry="/prod/reg.json"
    )
    row = idx.get(uid)
    assert row["status"] == "production"  # 状态被更新
    assert row["expr"] == "RANK($close)"
    assert row["first_seen_at"] <= row["updated_at"]

    idx.upsert_factor(uid=factor_uid("f_b"), name="f_b", status="memory_only")
    assert idx.count() == 2
    assert {r["name"] for r in idx.list_factors(status="production")} == {"f_a"}

    assert idx.delete_factor(uid) is True
    assert idx.get(uid) is None
    assert idx.count() == 1

    # rebuild 清空重建
    n = idx.rebuild([{"uid": factor_uid("f_c"), "name": "f_c", "status": "candidate"}])
    assert n == 1 and idx.count() == 1


def test_index_fk_enforced(tmp_path: Path):
    """索引库 foreign_keys=ON：未来子表可 CASCADE；这里验证 PRAGMA 生效。"""
    idx = FactorIndex(tmp_path / "fi.db")
    with idx._open() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

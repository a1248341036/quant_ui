"""Atomic parquet replace — including the Windows WinError-32 retry path."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import polars as pl
import pytest

from cnequity.storage import atomic as atomic_mod
from cnequity.storage.atomic import write_json_atomic, write_parquet_atomic


def _tiny_df() -> pl.DataFrame:
    return pl.DataFrame({"symbol": ["600519.SH"], "trade_date": [date(2024, 6, 28)]})


def test_write_parquet_atomic_roundtrip(tmp_path):
    out = tmp_path / "part.parquet"
    write_parquet_atomic(out, _tiny_df())
    assert out.exists()
    assert pl.read_parquet(out).height == 1


def test_write_parquet_atomic_same_target_is_safe_for_concurrent_refreshes(tmp_path):
    out = tmp_path / "derived.parquet"
    frames = [pl.DataFrame({"value": [index]}) for index in range(8)]

    with ThreadPoolExecutor(max_workers=len(frames)) as pool:
        list(pool.map(lambda frame: write_parquet_atomic(out, frame), frames))

    value = pl.read_parquet(out)["value"][0]
    assert value in range(len(frames))
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_parquet_atomic_retries_permission_error(tmp_path, monkeypatch):
    out = tmp_path / "part.parquet"
    calls = {"n": 0}
    real_replace = atomic_mod.os.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "file in use")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_mod.time, "sleep", lambda _: None)

    write_parquet_atomic(out, _tiny_df())
    assert calls["n"] == 3
    assert out.exists()


def test_write_parquet_atomic_raises_after_exhausted_retries(tmp_path, monkeypatch):
    out = tmp_path / "part.parquet"
    monkeypatch.setattr(
        atomic_mod.os,
        "replace",
        lambda *_a, **_k: (_ for _ in ()).throw(PermissionError(32, "file in use")),
    )
    monkeypatch.setattr(atomic_mod.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        write_parquet_atomic(out, _tiny_df())
    assert not out.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_json_atomic_roundtrip_and_unique_temp_cleanup(tmp_path):
    out = tmp_path / "quality" / "health.json"

    write_json_atomic(out, {"healthy": True, "rows": 3}, indent=2)

    import json

    assert json.loads(out.read_text(encoding="utf-8")) == {"healthy": True, "rows": 3}
    assert not list(out.parent.glob(".*.tmp"))


def test_write_json_atomic_retries_permission_error(tmp_path, monkeypatch):
    out = tmp_path / "quality.json"
    calls = {"n": 0}
    real_replace = atomic_mod.os.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "file in use")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_mod.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_mod.time, "sleep", lambda _: None)

    write_json_atomic(out, {"ok": True})

    assert calls["n"] == 3
    assert out.exists()

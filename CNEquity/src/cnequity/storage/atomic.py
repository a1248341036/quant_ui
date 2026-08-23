"""Atomic writes for curated/derived data and quality metadata."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import polars as pl

# Windows refuses ``os.replace`` while another handle (DuckDB, Explorer preview,
# AV) holds the destination. A short backoff covers the common "query just
# closed" race; persistent locks still surface as PermissionError.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_SEC = 0.05


def _filesystem_path(path: Path) -> str:
    """Return a Windows extended-length path when one is required."""
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    # os.replace reports FileNotFoundError rather than FilenameTooLong once
    # the destination plus temporary name crosses MAX_PATH.
    return "\\\\?\\" + os.path.abspath(text)


def _replace_with_retry(tmp: Path, path: Path) -> None:
    last_exc: BaseException | None = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(_filesystem_path(tmp), _filesystem_path(path))
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt + 1 >= _REPLACE_ATTEMPTS:
                break
            time.sleep(_REPLACE_BACKOFF_SEC * (2**attempt))
    assert last_exc is not None
    raise last_exc


def write_parquet_atomic(path: Path, df: pl.DataFrame, **kwargs) -> Path:
    """Write *df* to *path* via a unique same-directory temp file.

    A unique temporary name matters when two independent workers refresh the
    same cache or derived artifact: a fixed ``target.tmp`` lets them overwrite
    each other's in-progress footer before either ``os.replace`` runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        # Keep the temporary name short: Windows MAX_PATH failures surface as
        # FileNotFoundError inside mkstemp when the target already sits in deep
        # Hive-style snapshot directories.
        prefix=".atomic-",
        suffix=".tmp",
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.write_parquet(tmp, **kwargs)
        # Windows rejects fsync on a read-only file descriptor with EBADF.
        # Reopen read/write so the durability barrier works on every platform.
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
        return path
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def write_json_atomic(path: Path, payload: Any, **kwargs: Any) -> Path:
    """Serialize JSON to a unique temp file, then atomically replace *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
        return path
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

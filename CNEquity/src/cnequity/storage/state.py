"""Per-dataset incremental watermarks under meta/state/."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from contextlib import AbstractContextManager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import IO

from cnequity.file_lock import exclusive_lock


class StateStore:
    """Tracks last-success coverage per dataset (e.g. last trade_date)."""

    def __init__(self, meta_root: Path):
        self.root = meta_root / "state"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, dataset: str) -> Path:
        return self.root / f"{dataset}.json"

    def _lock_path(self, dataset: str) -> Path:
        return self.root / f"{dataset}.lock"

    def _dataset_lock(self, dataset: str) -> AbstractContextManager[IO]:
        return exclusive_lock(self._lock_path(dataset))

    def _read_payload(self, path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_payload(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def set_fields(self, dataset: str, **fields: object) -> None:
        """Persist diagnostics alongside the ordinary dataset watermark."""
        path = self._path(dataset)
        with self._dataset_lock(dataset):
            payload = self._read_payload(path)
            payload.update(fields)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_payload(path, payload)

    def get_date(self, dataset: str, field: str = "last_success_trade_date") -> date | None:
        with self._dataset_lock(dataset):
            value = self._read_payload(self._path(dataset)).get(field)
        if not value:
            return None
        return date.fromisoformat(str(value))

    def set_date(
        self,
        dataset: str,
        value: date,
        *,
        field: str = "last_success_trade_date",
    ) -> None:
        path = self._path(dataset)
        with self._dataset_lock(dataset):
            payload = self._read_payload(path)
            payload[field] = value.isoformat()
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_payload(path, payload)

    def update_max_date(
        self,
        dataset: str,
        candidate: date,
        *,
        field: str = "last_success_trade_date",
    ) -> None:
        path = self._path(dataset)
        with self._dataset_lock(dataset):
            payload = self._read_payload(path)
            current_raw = payload.get(field)
            current = date.fromisoformat(str(current_raw)) if current_raw else None
            if current is None or candidate > current:
                payload[field] = candidate.isoformat()
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._write_payload(path, payload)

    def clear_date(
        self,
        dataset: str,
        *,
        field: str = "last_success_trade_date",
    ) -> None:
        """Remove a date watermark while preserving other dataset state."""
        path = self._path(dataset)
        with self._dataset_lock(dataset):
            payload = self._read_payload(path)
            if field not in payload:
                return
            payload.pop(field, None)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_payload(path, payload)

    def get_string_set(self, dataset: str, field: str) -> set[str]:
        """Read a set-like string field from a dataset state payload."""
        with self._dataset_lock(dataset):
            value = self._read_payload(self._path(dataset)).get(field)
        if value is None:
            return set()
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"state field {dataset}.{field} must be a list of strings")
        return set(value)

    def set_string_set(self, dataset: str, field: str, values: Iterable[str]) -> None:
        """Atomically replace a set-like string field in a dataset state payload."""
        normalized = sorted(set(values))
        path = self._path(dataset)
        with self._dataset_lock(dataset):
            payload = self._read_payload(path)
            if normalized:
                payload[field] = normalized
            else:
                payload.pop(field, None)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_payload(path, payload)

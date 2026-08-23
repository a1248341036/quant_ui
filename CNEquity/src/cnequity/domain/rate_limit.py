from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from cnequity.file_lock import exclusive_lock

DEFAULT_LOCK_TIMEOUT_SECONDS = 15.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitSpec:
    """Pickle-friendly rate limit parameters for worker processes."""

    state_dir: str
    source: str
    min_interval: float
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS


@dataclass
class RateLimiter:
    """Cross-process fixed-spacing limiter using reserved request time slots.

    The file lock protects only the reservation transaction. Waiting for the
    reserved slot happens after the lock is released, so one slow process does
    not make every other process wait behind a sleeping lock holder.
    """

    name: str
    min_interval: float
    state_dir: Path
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS

    def wait(self) -> None:
        if self.min_interval <= 0:
            return

        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_dir / f"{self.name}.lock"
        state_path = self.state_dir / f"{self.name}.json"

        lock_started = time.monotonic()
        with exclusive_lock(lock_path, timeout=self.lock_timeout):
            lock_wait = time.monotonic() - lock_started
            previous_last = 0.0
            next_allowed_at = 0.0
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    previous_last = float(state.get("last", 0.0))
                    next_allowed_at = float(state.get("next_allowed_at", 0.0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    previous_last = 0.0
                    next_allowed_at = 0.0

            if not math.isfinite(previous_last) or previous_last < 0.0:
                previous_last = 0.0
            if not math.isfinite(next_allowed_at) or next_allowed_at < 0.0:
                next_allowed_at = 0.0

            # Migrate the old state format, where `last` was the timestamp of
            # the previous request start. A missing/corrupt state is safe to
            # treat as empty; the first request then gets the current slot.
            if next_allowed_at <= 0.0 and previous_last > 0.0:
                next_allowed_at = previous_last + self.min_interval

            now = time.time()
            slot = max(now, next_allowed_at)
            reserved_next = slot + self.min_interval

            fd, tmp_name = tempfile.mkstemp(
                dir=state_path.parent,
                prefix=f".{state_path.stem}-",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(
                        {"last": slot, "next_allowed_at": reserved_next},
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, state_path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise

        sleep_for = max(0.0, slot - time.time())
        if sleep_for:
            time.sleep(sleep_for)
        logger.debug(
            "rate limit %s: lock_wait=%.3fs sleep=%.3fs",
            self.name,
            lock_wait,
            sleep_for,
        )


def wait_source(
    state_dir: Path | str,
    source: str,
    min_interval: float,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> None:
    RateLimiter(source, min_interval, Path(state_dir), lock_timeout=lock_timeout).wait()


def wait_spec(spec: RateLimitSpec | None) -> None:
    if spec is not None:
        wait_source(
            spec.state_dir,
            spec.source,
            spec.min_interval,
            lock_timeout=spec.lock_timeout,
        )

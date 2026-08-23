import json
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import pytest

from cnequity.domain.rate_limit import RateLimiter, wait_source
from cnequity.file_lock import LockUnavailable

INTERVAL = 0.05

# What the wait must clear. Not `INTERVAL`, because two clocks disagree by a
# little: the limiter computes its sleep from `time.time()` (it has to — the
# deadline is shared across processes through a JSON file, and a monotonic
# clock is not comparable between them), while the assertion measures with
# `perf_counter`. Windows `time.sleep` also returns marginally early.
#
# Measured on CI: 0.03961 against a 0.04 floor — 0.4ms short, on a limiter
# whose job is pacing to ~10 req/s. The margin is generous enough that the
# flake cannot recur and still an order of magnitude tighter than the failure
# it guards against, which is the limiter not sleeping at all.
MIN_OBSERVED = INTERVAL * 0.6


def test_rate_limiter_enforces_minimum_interval(tmp_path):
    state_dir = tmp_path / "rate_limits"
    limiter = RateLimiter("test", INTERVAL, state_dir)
    limiter.wait()
    t0 = time.perf_counter()
    limiter.wait()
    assert time.perf_counter() - t0 >= MIN_OBSERVED


def _worker_wait(state_dir: str) -> float:
    t0 = time.perf_counter()
    wait_source(state_dir, "test", INTERVAL)
    return time.perf_counter() - t0


def test_rate_limiter_serializes_cross_process_requests(tmp_path):
    state_dir = tmp_path / "rate_limits"
    with ProcessPoolExecutor(max_workers=2) as pool:
        durations = list(pool.map(_worker_wait, [str(state_dir), str(state_dir)]))
    # One of the two must have waited: whichever lost the lock race sees the
    # other's timestamp already written.
    assert max(durations) >= MIN_OBSERVED


def test_corrupt_rate_state_is_replaced_atomically(tmp_path):
    state_dir = tmp_path / "rate_limits"
    state_dir.mkdir()
    state_path = state_dir / "test.json"
    state_path.write_text("{truncated", encoding="utf-8")

    RateLimiter("test", 0.01, state_dir).wait()

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["last"] > 0
    assert not list(state_dir.glob(".*.tmp"))


@pytest.mark.parametrize("payload", [{"last": "nan"}, {"next_allowed_at": "inf"}])
def test_non_finite_rate_state_does_not_poison_future_slots(tmp_path, payload):
    state_dir = tmp_path / "rate_limits"
    state_dir.mkdir()
    (state_dir / "test.json").write_text(json.dumps(payload), encoding="utf-8")

    RateLimiter("test", 0.01, state_dir).wait()

    state = json.loads((state_dir / "test.json").read_text(encoding="utf-8"))
    assert state["last"] > 0
    assert state["next_allowed_at"] > state["last"]


def test_rate_limiter_propagates_lock_timeout_instead_of_bypassing(monkeypatch, tmp_path):
    seen = {}

    @contextmanager
    def busy_lock(path, **kwargs):
        seen.update(kwargs)
        raise LockUnavailable("busy")
        yield  # pragma: no cover

    monkeypatch.setattr("cnequity.domain.rate_limit.exclusive_lock", busy_lock)

    with pytest.raises(LockUnavailable, match="busy"):
        RateLimiter("test", INTERVAL, tmp_path / "rate_limits").wait()

    assert seen["timeout"] == 15.0


def test_wait_spec_propagates_custom_lock_timeout(tmp_path, monkeypatch):
    seen = {}

    def fake_wait_source(state_dir, source, min_interval, lock_timeout):
        seen.update(
            state_dir=state_dir,
            source=source,
            min_interval=min_interval,
            lock_timeout=lock_timeout,
        )

    monkeypatch.setattr("cnequity.domain.rate_limit.wait_source", fake_wait_source)
    from cnequity.domain.rate_limit import RateLimitSpec, wait_spec

    wait_spec(RateLimitSpec(str(tmp_path), "tdx_protocol", 0.1, lock_timeout=3.5))

    assert seen["lock_timeout"] == 3.5

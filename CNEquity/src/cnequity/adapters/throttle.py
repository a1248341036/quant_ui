from __future__ import annotations

from dataclasses import dataclass, field

from cnequity.config import Config
from cnequity.domain.rate_limit import RateLimiter


@dataclass
class SourceRateLimiters:
    config: Config
    _limiters: dict[str, RateLimiter] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        state_dir = self.config.meta_root / "rate_limits"
        if self.config.tdx_enabled:
            interval = self.config.tdx_min_interval_ms / 1000.0
            self._limiters["tdx_protocol"] = RateLimiter(
                "tdx_protocol",
                interval,
                state_dir,
                lock_timeout=self.config.tdx_lock_timeout_sec,
            )
        for source, interval in self.config.source_intervals.items():
            self._limiters[source] = RateLimiter(source, interval, state_dir)

    def wait(self, source: str) -> None:
        limiter = self._limiters.get(source)
        if limiter is not None:
            limiter.wait()

"""Registry dispatch: auto-discovered external adapter modules.

Each external adapter module lives under ``cnequity.external`` and exports a
module-level ``ADAPTER`` singleton — an instance of a class implementing the
uniform adapter protocol.

Read-only adapters implement::

    enabled(config, dataset)          -> bool
    has_data(config, dataset)         -> bool
    files(config, dataset, ...)       -> list[Path]
    scan(config, dataset, ...)        -> pl.LazyFrame
    coverage_bounds(config, dataset)  -> tuple[date|None, date|None]

Write-capable adapters (``compactable=True`` datasets) additionally implement::

    compact_target(config, dataset, partition_value) -> Path
    compact_layout()  -> Literal["hive", "yearly_file"]
    compact_pk(dataset) -> list[str]   # PK in the *target file's* column names

The registry discovers adapter modules automatically by scanning the
``cnequity.external`` package.  A module opts in by defining an ``ADAPTER``
attribute (a class instance with the above methods).  ``enabled()`` does the
final gating, so a module that is imported but whose data source is not
configured will simply return ``False`` and let the next adapter try.

Usage::

    from cnequity.external.registry import external_adapter

    adapter = external_adapter(config, "daily_bars")
    if adapter is not None:
        lf = adapter.scan(config, "daily_bars", start=..., end=...)
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from datetime import date
from pathlib import Path
from typing import Protocol

import polars as pl

from cnequity.config import Config

logger = logging.getLogger(__name__)


class ExternalAdapter(Protocol):
    """Uniform protocol every external adapter implements."""

    def enabled(self, config: Config, dataset: str) -> bool: ...
    def has_data(self, config: Config, dataset: str) -> bool: ...
    def files(
        self, config: Config, dataset: str, *, start: date | None = None, end: date | None = None
    ) -> list[Path]: ...
    def scan(
        self,
        config: Config,
        dataset: str,
        *,
        start: date | None = None,
        end: date | None = None,
        symbols: list[str] | None = None,
    ) -> pl.LazyFrame: ...
    def coverage_bounds(self, config: Config, dataset: str) -> tuple[date | None, date | None]: ...


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

# Module-level singletons — built lazily to avoid importing polars/adapter
# modules before the config is loaded.
_adapters: list | None = None


def _discover_adapter_instances() -> list:
    """Import every ``cnequity.external.*`` module that exports an ``ADAPTER``.

    Returns a list of adapter instances.  Each module opts in by defining a
    module-level ``ADAPTER`` attribute (a class instance implementing the
    uniform adapter protocol).
    """
    import cnequity.external as ext_pkg

    found = []
    for info in pkgutil.iter_modules(ext_pkg.__path__):
        name = info.name
        # Skip the registry itself and the fetch helper (not an adapter).
        if name in ("registry", "tushare_fetch"):
            continue
        try:
            mod = importlib.import_module(f"cnequity.external.{name}")
        except Exception as exc:
            logger.debug("adapter module cnequity.external.%s import failed: %s", name, exc)
            continue
        adapter = getattr(mod, "ADAPTER", None)
        if adapter is not None:
            found.append(adapter)
    return found


def _build_adapters() -> list:
    """Build the adapter list.  Order matters: first match wins."""
    return _discover_adapter_instances()


def _get_adapters() -> list:
    global _adapters
    if _adapters is None:
        _adapters = _build_adapters()
    return _adapters


def external_adapter(config: Config, dataset: str):
    """Return the adapter that serves *dataset*, or ``None`` if none is enabled."""
    for adapter in _get_adapters():
        try:
            if adapter.enabled(config, dataset):
                return adapter
        except Exception as exc:
            logger.debug("adapter %s.enabled() raised for %s: %s", adapter, dataset, exc)
    return None


def any_external_enabled(config: Config, dataset: str) -> bool:
    """True if any external adapter is enabled for *dataset*."""
    return external_adapter(config, dataset) is not None


def external_has_data(config: Config, dataset: str) -> bool:
    """True if the adapter for *dataset* has readable data."""
    adapter = external_adapter(config, dataset)
    if adapter is None:
        return False
    return adapter.has_data(config, dataset)


def external_coverage_bounds(config: Config, dataset: str) -> tuple[date | None, date | None]:
    """Coverage bounds from the external adapter for *dataset*, or (None, None)."""
    adapter = external_adapter(config, dataset)
    if adapter is None:
        return None, None
    return adapter.coverage_bounds(config, dataset)


def reset_adapters() -> None:
    """Clear the cached adapter list (for tests that reload config)."""
    global _adapters
    _adapters = None

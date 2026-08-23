"""Numeric parsing helpers shared by source adapters."""

from __future__ import annotations

import math


def finite_int64(
    value: float | int,
    *,
    minimum: int = -(2**63),
    maximum: int = 2**63 - 1,
) -> int:
    """Convert a finite integral number within the signed Int64 range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"expected a finite integer, got {value!r}") from None
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"expected a finite integer, got {value!r}")
    if numeric < minimum or numeric > maximum:
        raise ValueError(f"integer outside Int64 range: {value!r}")
    return int(numeric)

import pytest

from cnequity.adapters.numeric import finite_int64


def test_finite_int64_rejects_non_integral_nonfinite_and_overflow_values():
    assert finite_int64(12.0) == 12
    with pytest.raises(ValueError):
        finite_int64(12.5)
    with pytest.raises(ValueError):
        finite_int64(float("inf"))
    with pytest.raises(ValueError):
        finite_int64(1e300)


def test_finite_int64_supports_domain_lower_bound():
    with pytest.raises(ValueError):
        finite_int64(-1, minimum=0)

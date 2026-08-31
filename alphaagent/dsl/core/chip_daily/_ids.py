"""常量与 ID 解析函数。"""
from __future__ import annotations

# method: 0=uniform, 1=cyq, 2=triangular
_CHIP_METHOD = {
    "uniform": 0,
    "cyq": 1,
    "tri": 2,
    "triangular": 2,
}

# metric op: 0=peak_loc, 1=entropy, 2=com_w_gap, 3=mass_asym, 4=peak_sharpness
CHIP_OP = {
    "peak_loc": 0,
    "entropy": 1,
    "com_w_gap": 2,
    "mass_asym": 3,
    "peak_sharpness": 4,
}


def chip_method_id(method: str) -> int:
    k = str(method).strip().lower()
    if k not in _CHIP_METHOD:
        raise ValueError(
            'method must be "uniform", "cyq", or "tri" (alias: "triangular")'
        )
    return _CHIP_METHOD[k]


def chip_wass_implementation_id(name: str) -> int:
    k = str(name).strip().lower()
    if k == "moment":
        return 0
    if k in ("transport", "w1", "earth"):
        return 1
    raise ValueError(
        'implementation must be "moment" or "transport" (aliases: "w1", "earth")'
    )


def chip_peak_sharpness_impl_id(name: str) -> int:
    k = str(name).strip().lower()
    if k in ("curvature", "curv", "s_curv"):
        return 0
    if k in ("fwhm", "s_fwhm"):
        return 1
    if k in ("combined", "sharp", "s_sharp"):
        return 2
    raise ValueError('implementation must be "curvature", "fwhm", or "combined"')


def chip_bimodal_impl_id(name: str) -> int:
    k = str(name).strip().lower()
    if k in ("simple", "ratio"):
        return 0
    if k in ("dip", "hartigan"):
        return 1
    raise ValueError('implementation must be "simple" or "dip"')

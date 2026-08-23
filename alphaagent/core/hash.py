"""Panel 指纹，用于 factorzoo / bundle 绑定。"""

from __future__ import annotations

import hashlib

import pandas as pd


def panel_index_hash(panel: pd.DataFrame) -> str:
    """根据 (datetime, instrument) 行序计算短 hash。"""
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel 须为 MultiIndex(datetime, instrument)")
    dt = panel.index.get_level_values("datetime").astype(str)
    inst = panel.index.get_level_values("instrument").astype(str)
    payload = "\n".join(f"{d}\t{i}" for d, i in zip(dt, inst, strict=False))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

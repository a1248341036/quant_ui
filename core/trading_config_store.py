# -*- coding: utf-8 -*-
"""代码面板专属参数副本 —— 设置中心只影响代码 tab, 不动全局 trading_config。

优先级: 代码面板副本(data/code_tab_config.json) > core/trading_config.py 默认值。
全局回测/模拟盘/门禁页面的行为完全不受本文件影响。
"""
from __future__ import annotations

import json
from pathlib import Path

from core.store import DATA_DIR

OVERRIDE_FILE = DATA_DIR / "code_tab_config.json"

# 代码面板可调参数白名单
EDITABLE_KEYS = (
    "buy_cost", "sell_cost", "slippage_bps", "max_participation",
    "lot_size", "amount_q", "warmup_days", "limit_flags",
    "min_am20_yuan",
)

_NUM_KEYS = {"buy_cost", "sell_cost", "slippage_bps", "max_participation",
             "amount_q", "min_am20_yuan"}
_INT_KEYS = {"lot_size", "warmup_days"}
_BOOL_KEYS = {"limit_flags"}


def load_overrides() -> dict:
    if not OVERRIDE_FILE.exists():
        return {}
    try:
        data = json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k in EDITABLE_KEYS}
    except (json.JSONDecodeError, OSError):
        return {}


def _cast(key: str, v):
    if key in _BOOL_KEYS:
        return str(v).strip().lower() in ("1", "true", "yes", "on")
    if key in _INT_KEYS:
        return int(float(v))
    if key in _NUM_KEYS:
        return float(v)
    return v


def effective() -> dict:
    """代码面板当前生效参数 = trading_config 默认 + 面板副本覆盖。"""
    from core import trading_config as tc
    base = tc.defaults()
    out = {k: base[k] for k in EDITABLE_KEYS if k in base}
    for k, v in load_overrides().items():
        out[k] = v
    return out


def save_overrides(patch: dict) -> dict:
    """校验并写入面板副本, 返回合并后的生效值。"""
    clean: dict = {}
    for k, v in (patch or {}).items():
        if k not in EDITABLE_KEYS:
            continue
        try:
            clean[k] = _cast(k, v)
        except (TypeError, ValueError):
            continue
    merged = {**load_overrides(), **clean}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OVERRIDE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(OVERRIDE_FILE)
    return effective()


def reset_overrides() -> dict:
    """恢复全局默认(删除面板副本)。"""
    if OVERRIDE_FILE.exists():
        OVERRIDE_FILE.unlink()
    return effective()

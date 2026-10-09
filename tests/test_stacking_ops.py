# -*- coding: utf-8 -*-
"""ML 组合运营总览：纯函数层单测（不依赖数据产物/模拟盘状态）。"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.stacking_ops_service import _is_mainboard, _norm6, _weekly_rhythm  # noqa: E402


def test_is_mainboard_explicit_prefixes():
    # 沪主板 600/601/603/605 + 深主板 000/001/002/003
    for code in ("600519", "601318", "603828", "605111", "000698", "001227", "002211", "003816"):
        assert _is_mainboard(code), code
    # 科创板 688/689 显式排除（用户可能无科创板权限）
    for code in ("688981", "689009"):
        assert not _is_mainboard(code), code
    # 创业板 300/301 排除
    for code in ("300750", "301269"):
        assert not _is_mainboard(code), code


def test_is_mainboard_normalizes_codes():
    # 带交易所后缀 / 前导零缺失等形态归一后判定
    assert _is_mainboard("600519.SH")
    assert _is_mainboard("000698.SZ")
    assert _is_mainboard(688981) is False
    assert _norm6("SZ000698") == "000698"
    assert _norm6("600519") == "600519"


def test_weekly_rhythm_day_mapping():
    # 2026-10-05 是周一：更新=本周五(10-09)，调仓=下周一(10-12)
    import pandas as pd
    from unittest.mock import patch

    monday = pd.Timestamp("2026-10-05")
    with patch("backend.stacking_ops_service.pd") as mock_pd:
        mock_pd.Timestamp.today.return_value = monday
        mock_pd.Timedelta = pd.Timedelta
        r = _weekly_rhythm()
    assert r["next_score_update"] == "2026-10-09"   # 周五
    assert r["next_rebalance"] == "2026-10-05"      # 周一当天=今天（0=今天语义）

    friday = pd.Timestamp("2026-10-09")
    with patch("backend.stacking_ops_service.pd") as mock_pd:
        mock_pd.Timestamp.today.return_value = friday
        mock_pd.Timedelta = pd.Timedelta
        r = _weekly_rhythm()
    assert r["next_score_update"] == "2026-10-09"   # 今天就是周五（0=今天）
    assert r["next_rebalance"] == "2026-10-12"      # 下周一


def test_weekly_rhythm_returns_iso_dates():
    r = _weekly_rhythm()
    for key in ("today", "next_score_update", "next_rebalance"):
        datetime.strptime(r[key], "%Y-%m-%d")
    assert "周五" in r["note"] and "周一" in r["note"]

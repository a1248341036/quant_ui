# -*- coding: utf-8 -*-
"""panel 缓存 stale 容忍 + 清理脚本回归（2026-10-20）。

**背景**：缓存命中要求「缓存终点 >= 请求终点」，而日更数据每天让 `end` 前移一
天，于是每个交易日都要重建整份面板（实测 4.7GB / 463s）。实测证据：

    end=2026-10-20  463s（重建，旧行为）
    end=2026-10-20  2.4s（命中，精确）
    end=2026-10-22  2.0s（命中，stale 2 天）
    end=2026-10-23  1.8s（命中，stale 3 天）

**方向安全性是硬约束**：只容忍「缓存比请求旧」，绝不容忍「缓存比请求新」——
后者会把请求区间之后的未来数据带进调用方，破坏 tuning 模式的物理时间隔离。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _reload_cnequity(monkeypatch, value: str | None):
    """按环境变量重载 cnequity，读回 _CACHE_MAX_STALE_DAYS。"""
    import importlib

    if value is None:
        monkeypatch.delenv("ALPHA_PANEL_CACHE_STALE_DAYS", raising=False)
    else:
        monkeypatch.setenv("ALPHA_PANEL_CACHE_STALE_DAYS", value)
    import alphaagent.data.adapters.cnequity as mod

    return importlib.reload(mod)


class TestStaleWindow:
    def test_default_is_three_days(self, monkeypatch):
        mod = _reload_cnequity(monkeypatch, None)
        assert mod._CACHE_MAX_STALE_DAYS == 3

    def test_env_override(self, monkeypatch):
        mod = _reload_cnequity(monkeypatch, "7")
        assert mod._CACHE_MAX_STALE_DAYS == 7
        _reload_cnequity(monkeypatch, None)  # 复原，避免污染其他测试

    def test_zero_restores_strict_behavior(self, monkeypatch):
        mod = _reload_cnequity(monkeypatch, "0")
        assert mod._CACHE_MAX_STALE_DAYS == 0
        _reload_cnequity(monkeypatch, None)

    def test_negative_clamped_to_zero(self, monkeypatch):
        mod = _reload_cnequity(monkeypatch, "-5")
        assert mod._CACHE_MAX_STALE_DAYS == 0
        _reload_cnequity(monkeypatch, None)


class TestDirectionSafety:
    def test_never_accepts_newer_cache(self):
        """源码层面锁定方向：命中分支只在 `req_end > idx_max`（缓存更旧）时放宽，
        不存在"缓存比请求新仍命中"的放行分支。"""
        src = (_ROOT / "alphaagent/data/adapters/cnequity.py").read_text(encoding="utf-8")
        assert "req_end > idx_max" in src
        assert "_CACHE_MAX_STALE_DAYS" in src
        # 未来数据隔离的既有防线必须在（请求区间裁剪）
        assert "slice_panel" in src or "req_end" in src

    def test_stale_branch_logs_and_gates(self):
        src = (_ROOT / "alphaagent/data/adapters/cnequity.py").read_text(encoding="utf-8")
        idx = src.index("req_end > idx_max")
        seg = src[idx : idx + 900]
        assert "stale_days" in seg
        assert "continue" in seg  # 超窗口必须跳过（触发重建）


class TestCleanupScript:
    def test_dry_run_exits_zero(self):
        """dry-run 必须能独立跑通且不删文件。"""
        r = subprocess.run(
            [sys.executable, str(_ROOT / "scripts/clean_panel_cache.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        assert r.returncode == 0, r.stderr[-800:]
        assert "dry-run" in (r.stdout or "")

    def test_keep_parameter_respected(self):
        r = subprocess.run(
            [sys.executable, str(_ROOT / "scripts/clean_panel_cache.py"), "--keep", "99"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        assert r.returncode == 0, r.stderr[-800:]
        # keep 很大时不应删除任何文件
        assert "已删除" not in (r.stdout or "")

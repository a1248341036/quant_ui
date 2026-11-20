# -*- coding: utf-8 -*-
"""轻量内存打点（零依赖：ctypes 优先，psutil 可选）。

背景
----
2026-11-20 排查「跑挖掘太耗内存」时发现：**全仓没有任何内存打点**——
``metrics_snapshot`` 里没有 RSS，``run_meta.json`` 里也没有，只能每次临时
写探针脚本，每轮重新搭一遍。本模块把这件事收口。

口径（务必分清，2026-11-21 实测教训）
------------------------------------
- ``rss``：WorkingSet（工作集）。任务管理器显示的就是它，**包含 mmap 文件页**；
  只读文件映射页可被 OS 直接回收，所以它**不代表真实压力**。
  典型误导：面板矩阵 v1 读取后 rss 7.1GB，但 private 只有 0.76GB。
- ``private``：Pagefile 使用量（提交内存）。不可共享、不可回收，
  **这才是真实占用**，判断内存压力请用它。
- ``peak``：进程启动至今的工作集峰值。

平台
----
Windows 用 ``K32GetProcessMemoryInfo``（psapi 同名 API 作兜底）；
其它平台返回空 dict —— 只打点、不报错，绝不因为取不到内存而中断挖掘。
"""
from __future__ import annotations

import sys
from typing import Any

__all__ = ["snapshot", "note", "peak", "reset_peak"]

_GB = 1024.0 ** 3
_PEAK: dict[str, float] = {}
_WIN = sys.platform == "win32"


def _read_ctypes() -> dict[str, int] | None:
    """直接读 PROCESS_MEMORY_COUNTERS，无需第三方库。"""
    if not _WIN:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if fn is None:
            import ctypes.wintypes  # noqa: F401

            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            fn = getattr(psapi, "GetProcessMemoryInfo", None)
        if fn is None:
            return None
        fn.restype = wintypes.BOOL
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]

        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not fn(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return {
            "rss": int(c.WorkingSetSize),
            "private": int(c.PagefileUsage),
            "peak": int(c.PeakWorkingSetSize),
        }
    except Exception:  # noqa: BLE001 — 打点永不阻断主流程
        return None


def _read_psutil() -> dict[str, int] | None:
    try:
        import psutil

        m = psutil.Process().memory_info()
        return {
            "rss": int(m.rss),
            "private": int(getattr(m, "private", 0)),
            "peak": int(getattr(m, "peak_wset", 0)),
        }
    except Exception:  # noqa: BLE001
        return None


def snapshot(*, update_peak: bool = True) -> dict[str, float]:
    """当前内存（GB，保留三位小数）。取不到时返回空 dict。"""
    raw = _read_ctypes() or _read_psutil()
    if not raw:
        return {}
    out = {
        "rss_gb": round(raw["rss"] / _GB, 3),
        "private_gb": round(raw["private"] / _GB, 3),
        "peak_gb": round(raw["peak"] / _GB, 3),
    }
    if update_peak:
        for k, v in out.items():
            if v > _PEAK.get(k, 0.0):
                _PEAK[k] = v
    return out


def note(stage: str) -> dict[str, Any]:
    """快照并带上阶段名，便于直接塞进日志/元数据。"""
    snap = snapshot()
    if not snap:
        return {}
    return {"stage": stage, **snap}


def peak() -> dict[str, float]:
    """本次进程生命期内的内存峰值（由 snapshot 累积）。"""
    return dict(_PEAK)


def reset_peak() -> None:
    _PEAK.clear()

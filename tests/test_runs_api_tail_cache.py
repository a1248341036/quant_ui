# -*- coding: utf-8 -*-
"""`GET /runs` 的 event-tail 缓存（2026-10-04）。

背景：`list_runs()` 对**每个** run 调一次 `scan_event_tail_multi` → 原实现每次都把整份
`run_*.jsonl` 读一遍统计行数。143 个 run 时 `GET /runs` 实测 2.8s，冷盘/负载高时超 20s；
monitor 把超时当成"后端挂了"→ 重启后端（当天 3 次，还顺带复活已 stop 的 run）。
缓存后：按 `(path, size, mtime_ns, tail)` 命中，只有正在写的 run 会 miss。
"""
from __future__ import annotations

import json
import time

from backend.alphaagent_service import (
    _TAIL_CACHE,
    scan_event_tail,
    scan_event_tail_multi,
)


def _write(path, rows: int) -> None:
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(rows)), encoding="utf-8")


def test_scan_event_tail_counts_and_tail(tmp_path):
    p = tmp_path / "run_a.jsonl"
    _write(p, 5)
    total, events = scan_event_tail(p, 2)
    assert total == 5
    assert [e["i"] for e in events] == [3, 4]


def test_scan_event_tail_cache_hits_and_invalidates(tmp_path):
    _TAIL_CACHE.clear()
    p = tmp_path / "run_b.jsonl"
    _write(p, 4)
    assert scan_event_tail(p, 10) == (4, [{"i": i} for i in range(4)])
    assert len(_TAIL_CACHE) == 1, "首次调用应写入缓存"
    # 命中缓存（结果一致，且缓存条目数不增长）
    assert scan_event_tail(p, 10)[0] == 4
    assert len(_TAIL_CACHE) == 1
    # 文件变化 → (size, mtime_ns) 变 → 必须重新读，不能返回旧的 4
    time.sleep(0.02)
    _write(p, 6)
    total, events = scan_event_tail(p, 10)
    assert total == 6
    assert len(events) == 6
    # 不同 tail 各自成键（尾部窗口不同）
    assert scan_event_tail(p, 2)[1] == [{"i": 4}, {"i": 5}]
    assert len(_TAIL_CACHE) == 3


def test_scan_event_tail_multi_merges_segments(tmp_path):
    _TAIL_CACHE.clear()
    a = tmp_path / "run_20260101_1.jsonl"
    b = tmp_path / "run_20260102_2.jsonl"
    _write(a, 3)
    b.write_text("".join(json.dumps({"i": 100 + i}) + "\n" for i in range(3)), encoding="utf-8")
    total, events = scan_event_tail_multi([a, b], 4)
    assert total == 6
    # 尾部窗口从最后一段回溯：取 b 的 3 行 + a 的最后 1 行
    assert [e["i"] for e in events] == [2, 100, 101, 102]


def test_scan_event_tail_multi_uses_cache(tmp_path):
    _TAIL_CACHE.clear()
    a = tmp_path / "run_a.jsonl"
    _write(a, 3)
    scan_event_tail_multi([a], 5)
    assert len(_TAIL_CACHE) == 1
    scan_event_tail_multi([a], 5)
    assert len(_TAIL_CACHE) == 1, "第二次必须命中缓存而不是新增条目"


def test_scan_event_tail_handles_missing_and_bad_lines(tmp_path):
    _TAIL_CACHE.clear()
    assert scan_event_tail(tmp_path / "nope.jsonl", 5) == (0, [])
    p = tmp_path / "run_c.jsonl"
    p.write_text('{"i": 0}\nnot-json\n{"i": 2}\n', encoding="utf-8")
    total, events = scan_event_tail(p, 10)
    assert total == 2, "不可解析行不计入行数（与 read_events 口径一致）"
    assert [e["i"] for e in events] == [0, 2]

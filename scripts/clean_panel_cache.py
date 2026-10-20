# -*- coding: utf-8 -*-
"""panel 缓存清理：按 (schema 版本, 签名) 分组，每组只保留最新 N 份。

背景（2026-10-20 实测）
--------------------
面板缓存文件形如 ``panel_v10_<signature>_all.{parquet,arrow,meta.json}``，全量一份
≈ 6.3GB（parquet 1.6 + arrow 4.7）。历史上因为：
  1. schema 版本 bump（v9 → v10 → v11）每次都不删旧版本；
  2. 不同 worktree/分支用不同 schema 各建各的（磁盘上 v9/v10/v11 同时存在）；
  3. 请求 ``end`` 每交易日前进一天 → 命中失败 → 重建一份新的；
磁盘累积到 34 个文件 / 43.7GB。

本脚本解决第 1、2 条（保留最近 N 份），第 3 条由
``ALPHA_PANEL_CACHE_STALE_DAYS``（默认 3 天容忍）避免每个交易日重建。

用法
----
    python scripts/clean_panel_cache.py                 # dry-run，只打印计划
    python scripts/clean_panel_cache.py --apply         # 执行删除
    python scripts/clean_panel_cache.py --keep 2 --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_NAME_RE = re.compile(r"^panel_(v\d+)_([0-9a-f]+)_(.+)$")
_SUFFIXES = (".parquet", ".arrow", ".meta.json")


def cache_root() -> Path:
    from alphaagent.data.adapters.cnequity import _CACHE_ROOT  # noqa: PLC0415

    return _CACHE_ROOT


def _entry_suffix(name: str) -> str | None:
    for s in _SUFFIXES:
        if name.endswith(s):
            return s
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=2, help="每个 (schema, 签名) 组保留最新几份（默认 2）")
    ap.add_argument("--apply", action="store_true", help="真正删除（缺省只打印计划）")
    args = ap.parse_args()

    root = cache_root()
    if not root.is_dir():
        print(f"缓存目录不存在: {root}")
        return 0

    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    sigs: dict[str, set[str]] = defaultdict(set)
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        suffix = _entry_suffix(p.name)
        if suffix is None:
            continue
        stem = p.name[: -len(suffix)]
        m = _NAME_RE.match(stem)
        if not m:
            continue
        groups[m.group(1)].append((p, suffix))
        sigs[m.group(1)].add(m.group(2))

    total_del = 0
    total_gb = 0.0
    plan: list[Path] = []

    # 按 **schema 版本**分组（签名含 end 日期，每交易日一个新签名，按签名分组等于不删）。
    # 每个 schema 保留最新 N 份；旧版本（v9 等）自然只剩 0~1 份被清掉。
    for schema, items in sorted(groups.items()):
        # 一个"份"以 parquet 为基准；没有 parquet 时退回 arrow（v9 等旧版本可能
        # 只剩 arrow+meta，也要能被清掉，否则永远占着磁盘）。
        rows = [p for p, s in items if s == ".parquet"]
        if not rows:
            rows = [p for p, s in items if s == ".arrow"]
        rows.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        stale = rows[args.keep:]
        print(f"\n[{schema}] 共 {len(rows)} 份 / {len(sigs[schema])} 个签名，"
              f"保留 {min(args.keep, len(rows))}，删除 {len(stale)}")
        if not stale:
            continue
        stale_stems = set()
        for p in stale:
            name = p.name
            for s in _SUFFIXES:
                if name.endswith(s):
                    stale_stems.add(name[: -len(s)])
                    break
        for stem in stale_stems:
            for s in _SUFFIXES:
                p = root / (stem + s)
                if p.is_file():
                    total_gb += p.stat().st_size / 1e9
                    plan.append(p)
            print(f"   - {stem}")

    print(f"\n合计可删除 {len(plan)} 个文件，约 {total_gb:.2f} GB")
    if not args.apply:
        print("（dry-run；加 --apply 执行）")
        return 0

    for p in plan:
        p.unlink(missing_ok=True)
        total_del += 1
    print(f"已删除 {total_del} 个文件，释放约 {total_gb:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

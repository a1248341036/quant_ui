#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每日收盘后增量刷新行情数据（systemd timer 调用）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fetcher import update_data


def main() -> int:
    print("start refresh", flush=True)
    try:
        result = update_data(mode="incremental")
        print(f"ok: {result}", flush=True)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整夜挖掘前置数据体检门禁（monitor 拉起后端之前调用）。

复用 scripts/healthcheck.py 的 CNE 日线新鲜度检查（duckdb max(trade_date)，
距今 >5 个自然日 FAIL——5 天是 healthcheck 既有口径，不在此改）。

已知边界：节假日后第一晚会误报（水位=节前最后交易日）。此时用 monitor 的
--no-preflight 放行一次，不要放宽这里的阈值。

退出码：0=通过；1=未通过（monitor 据此不开挖并通知）。

用法：
  python scripts/data_preflight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def main() -> int:
    import healthcheck  # 同目录复用，别复制实现

    ok, msg = healthcheck.check_data()
    print(f"[data-preflight] {'OK' if ok else 'FAIL'} - {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

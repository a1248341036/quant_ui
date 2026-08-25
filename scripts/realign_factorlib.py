#!/usr/bin/env python3
"""因子库对齐维护：CNE 数据湖尾部追加后，把 factorzoo 行索引增量对齐到最新 panel。

用法:
    python scripts/realign_factorlib.py                # 增量对齐（默认库/panel）
    python scripts/realign_factorlib.py --dry-run      # 只检查不落盘
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.core.paths import FACTORZOO_DIR  # noqa: E402
from alphaagent.data.adapters.cnequity import CNE_SOURCE, is_cne_source, load_panel_from_cne  # noqa: E402
from alphaagent.data.panel import load_panel  # noqa: E402
from alphaagent.factor.zoo.realign import incremental_realign_factorlib_to_panel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="factorzoo 增量对齐到最新 CNE panel")
    parser.add_argument("--lib", type=Path, default=FACTORZOO_DIR)
    parser.add_argument("--panel", default=CNE_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if is_cne_source(args.panel):
        panel = load_panel_from_cne(universe_mask=False)
        resolved = Path(CNE_SOURCE)
    else:
        resolved = Path(args.panel).expanduser().resolve()
        panel = load_panel(resolved)

    report = incremental_realign_factorlib_to_panel(
        args.lib,
        panel=panel,
        panel_path=resolved,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"耗时: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""补拉指定交易日的 daily/daily_basic/adj_factor（upsert 幂等）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tushare_client import get_pro  # noqa: E402
from scripts.sync_postgres import _fetch_one_trade_date, _log  # noqa: E402


DATES = [
    "2023-10-12", "2023-10-13", "2023-10-16", "2023-10-17",
    "2025-02-27", "2025-02-28",
]


def main() -> int:
    pro = get_pro()
    failed = []
    for d in DATES:
        try:
            _, n = _fetch_one_trade_date(pro, d)
            _log(f"{d}: {n} 行")
        except Exception as exc:
            failed.append(d)
            _log(f"{d} 失败: {type(exc).__name__}: {exc}")
    _log(f"补拉完成: {len(DATES) - len(failed)}/{len(DATES)}，失败: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

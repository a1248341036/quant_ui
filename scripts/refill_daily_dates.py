"""补拉指定交易日的 daily/daily_basic/adj_factor（幂等，写入 CNE 年度档案）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DATES = [
    "2023-10-12", "2023-10-13", "2023-10-16", "2023-10-17",
    "2025-02-27", "2025-02-28",
]

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "sync_daily_to_cne.py"


def main() -> int:
    failed = []
    for d in DATES:
        try:
            r = subprocess.run(
                [sys.executable, str(SCRIPT), "--since", d, "--end", d],
                capture_output=True, text=True, timeout=1800,
            )
            print(r.stdout.strip(), flush=True)
            if r.returncode != 0:
                failed.append(d)
                print(r.stderr.strip(), file=sys.stderr, flush=True)
            else:
                print(f"{d}: ok", flush=True)
        except Exception as exc:
            failed.append(d)
            print(f"{d} 失败: {type(exc).__name__}: {exc}", flush=True)
    print(f"补拉完成: {len(DATES) - len(failed)}/{len(DATES)}，失败: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

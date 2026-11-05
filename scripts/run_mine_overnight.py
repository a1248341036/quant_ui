# -*- coding: utf-8 -*-
"""标准夜跑挖掘执行器（流程内嵌 OOS，禁止事后补跑）。

窗口纪律（与 alphaagent/factor/window_config.py 统一，禁止覆盖）：
    train = 2020-01-01 ~ 2022-12-31   （统计门槛，默认值，不传）
    val   = 2023-01-01 ~ 2024-12-31   （样本外保留率，由 harness_mine --val 强制开启）
    test  = 2025-01-01 起             （终审盲测段，流程内绝不触碰）

用法：
    .venv\\Scripts\\python.exe scripts\\run_mine_overnight.py <批次JSON1> [批次JSON2 ...]

每个批次执行:
    harness_mine.py --mode fundamental --factors <json> --fundamentals --val
                    --out <同目录 xxx_rows.jsonl>
```
--val 会对「raw 全门通过」和「增量轨捞回」两类幸存者当场做 val 段(OOS)评估
（harness_mine.py 2026-11-01 补丁），val_ok 结果随 rows 落盘；批末汇总表列出
每个因子的 train IC / val IC / val_ok，OOS 表现不再事后补跑。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def fmt(v: object, spec: str = "%.4f") -> str:
    return spec % v if isinstance(v, (int, float)) else "  -   "


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rc_all = 0
    for fp in sys.argv[1:]:
        p = Path(fp)
        if not p.exists():
            print(f"[skip] 批次不存在: {fp}")
            continue
        out = p.with_name(p.stem + "_rows.jsonl")
        batch = json.load(open(p, encoding="utf-8"))
        print(f"=== [{p.stem}] 开始 {datetime.now():%H:%M:%S} | "
              f"{len(batch)} 因子 | 窗口纪律 train默认/val强制/test不碰 ===", flush=True)
        cmd = [str(PY), "scripts/harness_mine.py", "--mode", "fundamental",
               "--factors", str(p), "--out", str(out),
               "--fundamentals", "--val"]
        r = subprocess.run(cmd, cwd=ROOT)
        rc_all |= r.returncode
        # 汇总本批：train 门 + val 段 OOS
        rows = []
        if out.exists():
            for l in open(out, encoding="utf-8"):
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
        print(f"=== [{p.stem}] 批末汇总（val 即 OOS）===", flush=True)
        for r0 in sorted(rows, key=lambda x: -(abs(x.get("ic") or 0) / max(x.get("icir") or 1e-6, 1e-6))):
            v = r0.get("val") or {}
            tag = "PASS" if not r0.get("fails") else ("INC" if r0.get("pass_increment") else "---")
            print(f"  {tag:4s} {r0.get('name','?'):28s} train ic={fmt(r0.get('ic'))} "
                  f"icir={fmt(r0.get('icir'))} | val ic={fmt(v.get('ic'))} "
                  f"icir={fmt(v.get('icir'))} cov={fmt(v.get('coverage'),'%.3f')} "
                  f"ok={str(r0.get('val_ok'))}", flush=True)
        print(f"=== [{p.stem}] 结束 {datetime.now():%H:%M:%S} rc={r.returncode} ===\n", flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
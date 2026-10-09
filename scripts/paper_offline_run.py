# -*- coding: utf-8 -*-
"""模拟盘离线重放入口（后端 17891 不在线时的降级路径）。

复刻 backend /api/paper/run 的 jq 分支（_jq_panel → run_paper_trade），
不依赖后端进程：
- 只处理 active、非 alpha 且 universe == JQ_UNIVERSE("全A主板") 的账户；
- ETF / AlphaAgent 因子 / 其他股票池账户离线时跳过，
  并写入返回结果的 skipped_needing_backend；
- 幂等由 paper_core 内置保证（同一 exec_date 重复执行自动跳过）。

用法：
    python scripts/paper_offline_run.py [exec_date]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    exec_date = sys.argv[1] if len(sys.argv) > 1 else None

    from backend.routers.paper import JQ_UNIVERSE, _jq_panel
    from core.paper import list_accounts, run_paper_trade

    accounts = list_accounts()
    # 与 /api/paper/run 分组一致：JQ_UNIVERSE("全A主板") factor 账户走 _jq_panel；
    # alpha / ETF 账户依赖后端专有通道，离线时跳过。
    jq_ids = [a["id"] for a in accounts
              if a.get("status") == "active"
              and a.get("strategy_type") != "alpha"
              and a.get("universe") == JQ_UNIVERSE]
    skipped = [a["id"] for a in accounts
               if a.get("status") == "active" and a["id"] not in jq_ids]
    if not jq_ids:
        # 无可离线处理账户属正常跳过，与在线路径零账户语义对齐，避免定时任务误报失败。
        print(json.dumps({"ok": True, "run_date": None, "accounts": [],
                          "offline": True,
                          "skipped_needing_backend": skipped}, ensure_ascii=False))
        return 0

    panel = _jq_panel(exec_date)
    codes = {JQ_UNIVERSE: sorted(panel["code"].unique())}
    res = run_paper_trade(
        panel, codes,
        account_ids=jq_ids,
        exec_date=exec_date, dry_run=False,
    )
    res["ok"] = bool(res.get("accounts")) and all(
        a.get("processed") != "error" for a in res.get("accounts", []))
    res["offline"] = True
    res["skipped_needing_backend"] = skipped
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        sys.exit(1)

#!/usr/bin/env python3
"""对因子库存量候选批量复检：新引擎门禁（验证集窗口完整约束回测）。

用法：
  .venv\\Scripts\\python.exe scripts\\recheck_engine_gate.py \
      --factorlib artifacts/alphaagent/factorzoo/candidate_1d \
      --val-start 2023-01-01 --val-end 2025-12-31

- panel 默认 cne://（CNE 数据湖实时构建），全程内存，不落盘中间表。
- 报告写入 artifacts/alphaagent/engine_gate_recheck.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.data.adapters.cnequity import is_cne_source, load_panel_from_cne  # noqa: E402
from alphaagent.dsl import eval_factor  # noqa: E402
from alphaagent.factor.mining.engine_gate import run_engine_gate  # noqa: E402


def _load_panel(panel_path: str, start: str, end: str):
    if is_cne_source(panel_path):
        return load_panel_from_cne(start=start, end=end, universe_mask=False).sort_index()
    from alphaagent.data.panel import load_panel
    return load_panel(panel_path).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description="存量候选引擎门禁复检")
    parser.add_argument("--panel", default="cne://")
    parser.add_argument("--factorlib", default="artifacts/alphaagent/factorzoo/candidate_1d")
    parser.add_argument("--registry-name", default=None, help="默认自动选择 candidate/delivered registry")
    parser.add_argument("--val-start", default="2023-01-01")
    parser.add_argument("--val-end", default="2025-12-31")
    parser.add_argument("--warmup-start", default="2018-01-01", help="panel 加载起点，保证滚动窗口预热")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-excess-annual", type=float, default=0.03)
    parser.add_argument("--max-drawdown", type=float, default=0.35)
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "alphaagent" / "engine_gate_recheck.json"),
    )
    args = parser.parse_args()

    lib = Path(args.factorlib)
    registry = (
        Path(args.registry_name)
        if args.registry_name
        else (lib / "mining_candidate_registry.json")
        if (lib / "mining_candidate_registry.json").is_file()
        else lib / "mining_delivered_registry.json"
    )
    entries = json.loads(registry.read_text(encoding="utf-8"))
    factors = {k: v for k, v in entries.items() if isinstance(v, dict) and v.get("expr")}
    print(f"[recheck] registry={registry} factors={len(factors)}", flush=True)

    t0 = time.perf_counter()
    panel = _load_panel(args.panel, args.warmup_start, args.val_end)
    print(f"[recheck] panel rows={len(panel)} cols={panel.shape[1]} load={time.perf_counter()-t0:.0f}s", flush=True)

    policy = {
        "enabled": True,
        "top_n": args.top_n,
        "min_annual_return": 0.0,
        "min_excess_annual": args.min_excess_annual,
        "max_drawdown": args.max_drawdown,
    }
    report: dict[str, dict] = {}
    for factor_id, meta in factors.items():
        t_factor = time.perf_counter()
        try:
            out = eval_factor(meta["expr"], panel)
            values = out.reindex(panel.index).to_numpy(dtype=np.float64) if hasattr(out, "reindex") else None
        except Exception as exc:  # noqa: BLE001
            report[factor_id] = {"ok": False, "error": f"dsl_eval_failed: {exc}"}
            print(f"[recheck] {factor_id}: dsl_eval_failed ({time.perf_counter()-t_factor:.0f}s)", flush=True)
            continue

        ic_sign = 1 if float((meta.get("metrics") or {}).get("ic") or 0.0) >= 0 else -1
        gate = run_engine_gate(
            panel,
            values,
            val_start=args.val_start,
            val_end=args.val_end,
            direction=ic_sign,
            policy=policy,
        )
        gate["direction"] = ic_sign
        report[factor_id] = {"ok": True, **gate}
        m = gate.get("metrics") or {}
        print(
            f"[recheck] {factor_id}: passed={gate['passed']} "
            f"annual={m.get('annual_return')} excess={m.get('excess_annual')} "
            f"mdd={m.get('max_drawdown')} ({time.perf_counter()-t_factor:.0f}s)",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "window": {"val_start": args.val_start, "val_end": args.val_end},
        "policy": policy,
        "results": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[recheck] report -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""对因子库存量候选批量复检：新引擎门禁（验证集窗口完整约束回测）。

用法：
  .venv\\Scripts\\python.exe scripts\\recheck_engine_gate.py \
      --factorlib artifacts/alphaagent/factorzoo/candidate_main \
      --val-start 2023-01-01 --val-end 2024-12-31

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
from alphaagent.factor.zoo import FactorZoo  # noqa: E402


def _load_panel(panel_path: str, start: str, end: str):
    if is_cne_source(panel_path):
        return load_panel_from_cne(start=start, end=end, universe_mask=False).sort_index()
    from alphaagent.data.panel import load_panel
    return load_panel(panel_path).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description="存量候选引擎门禁复检")
    parser.add_argument("--panel", default="cne://")
    parser.add_argument("--factorlib", default="artifacts/alphaagent/factorzoo/candidate_main")
    parser.add_argument("--registry-name", default=None, help="默认自动选择 candidate/delivered registry")
    parser.add_argument("--val-start", default="2023-01-01")
    parser.add_argument("--val-end", default="2024-12-31")
    parser.add_argument("--warmup-start", default="2020-01-01", help="panel 加载起点，保证滚动窗口预热")
    parser.add_argument("--top-n", type=int, default=None, help="固定选股数（默认用统一门槛 top_pct 百分比选股）")
    parser.add_argument("--min-excess-annual", type=float, default=None, help="覆盖净超额年化下限（默认取统一门槛）")
    parser.add_argument("--max-drawdown", type=float, default=None, help="覆盖最大回撤上限（默认取统一门槛）")
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "alphaagent" / "engine_gate_recheck.json"),
    )
    args = parser.parse_args()

    lib = Path(args.factorlib)
    # 候选库 = registry ∪ catalog（与 blind_test_factors 口径一致，去重、registry 优先）
    factors: dict[str, dict] = {}
    reg_candidates = [Path(args.registry_name)] if args.registry_name else [
        lib / "mining_candidate_registry.json",
        lib / "mining_delivered_registry.json",
    ]
    for reg in reg_candidates:
        if reg.is_file():
            entries = json.loads(reg.read_text(encoding="utf-8"))
            factors.update({k: v for k, v in entries.items() if isinstance(v, dict) and v.get("expr")})
            break
    try:
        zoo = FactorZoo.open(lib)
        for fid in zoo.catalog.list_factor_ids():
            if fid in factors:
                continue
            meta = zoo.catalog.get(fid)
            if meta is None or not getattr(meta, "expr", None):
                continue
            factors[fid] = {"expr": meta.expr, "metrics": (meta.extra or {}).get("metrics") or {}}
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] catalog 读取失败（无 meta/factors.parquet）: {exc}", flush=True)
    print(f"[recheck] registry={reg} factors={len(factors)}", flush=True)

    t0 = time.perf_counter()
    panel = _load_panel(args.panel, args.warmup_start, args.val_end)
    print(f"[recheck] panel rows={len(panel)} cols={panel.shape[1]} load={time.perf_counter()-t0:.0f}s", flush=True)

    policy = {
        "enabled": True,
        "min_annual_return": 0.0,
    }
    if args.min_excess_annual is not None:
        policy["min_excess_annual"] = args.min_excess_annual
    if args.max_drawdown is not None:
        policy["max_drawdown"] = args.max_drawdown
    # 统一门槛唯一真源：research_spec 默认 delivery_policy 的 engine_gate。
    # 脚本不重复维护门槛数值，避免与代码/前端面板漂移；CLI 显式参数仅作临时覆盖。
    from alphaagent.factor.mining.research_spec import default_research_spec

    canonical = default_research_spec("technical")["delivery_policy"]["production"]["engine_gate"]
    for k, v in canonical.items():
        policy.setdefault(k, v)
    if args.top_n is not None:
        policy["top_n"] = args.top_n
        policy["selection_mode"] = "top_n"
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

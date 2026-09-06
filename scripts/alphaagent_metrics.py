"""AlphaAgent 挖掘 run 量化指标汇总：效率（第一层）+ 漏斗质量（第二层）。

计算逻辑在 alphaagent/factor/mining/run_metrics.py（与挖掘循环内实时
metrics_snapshot、后端 /runs/{id}/metrics API 共用同一实现），本脚本只是
CLI 外壳：扫全部 run 出表格 + 汇总。

用法：
  python scripts/alphaagent_metrics.py                 # 全部 run + 汇总
  python scripts/alphaagent_metrics.py --last 5        # 最近 5 个 run
  python scripts/alphaagent_metrics.py --runs id1 id2  # 指定 run
  python scripts/alphaagent_metrics.py --json out.json # 同时落 JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.mining.run_metrics import compute_run_metrics  # noqa: E402,F401

LOG_ROOT = ROOT / "logs" / "factor_mining" / "ui"

_EFF_COLS = [
    ("wall_minutes", "时长min", 8),
    ("llm_calls", "LLM次", 6),
    ("input_k_tokens", "输入K", 7),
    ("output_k_tokens", "输出K", 7),
    ("cache_hit_rate", "缓存率", 6),
    ("thinking_k_chars", "thinkingK字", 11),
    ("llm_gen_minutes_est", "LLM生成min", 10),
    ("tool_minutes", "工具min", 7),
    ("n_eval", "评估", 5),
    ("n_submit", "提交", 5),
    ("stored_candidate", "入库", 5),
    ("stored_production", "晋升", 5),
    ("tool_error_rate", "错误率", 6),
    ("dup_dead_end", "死路提醒", 8),
    ("minutes_per_delivered", "min/产出", 9),
]


def _print_table(rows: list[dict]) -> None:
    header = f"{'run_id':14s}" + "".join(f"{label:>{w}}" for _, label, w in _EFF_COLS)
    print(header)
    print("-" * len(header))
    for m in rows:
        cells = ""
        for key, _, w in _EFF_COLS:
            v = m.get(key)
            if v is None:
                cells += f"{'-':>{w}}"
            elif key == "cache_hit_rate":
                cells += f"{v:>{w}.2f}"
            elif isinstance(v, float):
                cells += f"{v:>{w}.1f}"
            else:
                cells += f"{v:>{w}}"
        print(f"{m['run_id'][:14]:14s}{cells}")
    print()
    print("漏斗（评估→提交→stage1→stage2→gate→晋升 / 盲测失败）与 gate 失败分布：")
    for m in rows:
        f = m["funnel"]
        print(f"  {m['run_id'][:14]:14s} eval={m['n_eval']} submit={m['n_submit']} "
              f"s1={f['stage_one_pass']}/{f['stage_one_pass'] + f['stage_one_fail']} "
              f"s2={f['stage_two_pass']}/{f['stage_two_pass'] + f['stage_two_fail']} "
              f"gate={f['gate_pass']}/{f['gate_pass'] + f['gate_fail']} "
              f"晋升={f['promoted']} 盲测拦={f['blind_fail']} "
              f"gate失败项={m['gate_fail_reasons'] or '-'}")
    print()
    print("工具错误分布（ToolArgumentsError=参数/契约违规；eval 类=表达式求值失败；"
          "Stage/Gate/Blind 类=门槛拒绝，属正常流程）：")
    for m in rows:
        rate = m.get("tool_error_rate")
        rate_s = f"{rate:.1%}" if rate is not None else "-"
        print(f"  {m['run_id'][:14]:14s} 错误率={rate_s} 明细={m['error_breakdown'] or '-'}")
    print()
    print("记忆 advisory 命中（duplicate_known_dead_end=重复死路结构提醒；"
          "edit_veto=意向编辑否决；duplicate_prior_result=正向结构重复）：")
    for m in rows:
        print(f"  {m['run_id'][:14]:14s} {m['advisory_breakdown'] or '-'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", default=None, help="指定 run_id（默认全部）")
    ap.add_argument("--last", type=int, default=None, help="最近 N 个 run")
    ap.add_argument("--json", default=None, help="同时写 JSON 到该路径")
    args = ap.parse_args()

    run_dirs = sorted(p for p in LOG_ROOT.iterdir() if p.is_dir())
    if args.runs:
        run_dirs = [p for p in run_dirs if p.name in set(args.runs)]
    elif args.last:
        run_dirs = run_dirs[-args.last:]
    if not run_dirs:
        print("没有匹配的 run 目录")
        sys.exit(1)

    rows = [compute_run_metrics(p.name, p) for p in run_dirs]
    _print_table(rows)

    delivered = [m for m in rows if (m["stored_production"] + m["stored_candidate"]) > 0]
    summary = {
        "n_runs": len(rows),
        "total_wall_minutes": round(sum(m["wall_minutes"] for m in rows), 1),
        "total_input_k_tokens": round(sum(m["input_k_tokens"] for m in rows), 1),
        "total_output_k_tokens": round(sum(m["output_k_tokens"] for m in rows), 1),
        "total_thinking_k_chars": round(sum(m["thinking_k_chars"] for m in rows), 1),
        "total_stored_candidate": sum(m["stored_candidate"] for m in rows),
        "total_stored_production": sum(m["stored_production"] for m in rows),
        "mean_minutes_per_delivered": (
            round(sum(m["wall_minutes"] for m in delivered) /
                  sum(m["stored_production"] + m["stored_candidate"] for m in delivered), 1)
            if delivered else None),
    }
    print("\n汇总:", json.dumps(summary, ensure_ascii=False))

    # ── Reviewer 校准（跨 run 全局）：approve 的存活率应高于 revise ──
    from alphaagent.factor.mining.run_metrics import reviewer_calibration

    cal = reviewer_calibration(
        ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json",
        ROOT / "artifacts" / "alphaagent" / "factorzoo" / "production_main" / "mining_delivered_registry.json",
    )
    print(f"\nReviewer 校准（{cal['n_total']} 个已入库因子，存活=未死在 stage_two/engine_gate）：")
    for verdict, b in sorted(cal["crosstab"].items()):
        print(f"  {verdict:14s} n={b['n']:3d} 晋升={b['promoted']} 存活={b['candidate_alive']} "
              f"gate死={b['gate_failed']} s2死={b['stage_two_failed']} 存活率={b['alive_rate']}")
    c = cal["calibration"]
    print(f"  → 校准提升 lift = approve存活率 {c['approve_alive_rate']} − revise存活率 {c['revise_alive_rate']} = {c['lift']}"
          "（接近 0 或为负 ⇒ Reviewer 意见对晋升无预测力）")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "runs": rows, "reviewer_calibration": cal},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"JSON 已写入 {args.json}")


if __name__ == "__main__":
    main()

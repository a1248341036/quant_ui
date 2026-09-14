#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaAgent 挖掘质量与效率评分卡（Scorecard）CLI 工具。

用法示例：
    # 1. 查看单个已完成 run 的质量与漏斗评分卡
    python scripts/benchmark_agent_run.py --run <run_id>

    # 2. 横向对比两个 run（改动前 vs 改动后）
    python scripts/benchmark_agent_run.py --compare <run_id_A> <run_id_B>

    # 3. 列出最近 runs 并按过线率/吞吐排序
    python scripts/benchmark_agent_run.py --list 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs" / "factor_mining" / "ui"

from alphaagent.factor.mining.run_metrics import generate_scorecard


def _load_or_gen_scorecard(run_id: str) -> dict[str, Any] | None:
    run_dir = LOG_ROOT / run_id
    if not run_dir.is_dir():
        print(f"错误: 未找到 run 目录 {run_dir}", file=sys.stderr)
        return None
    sc_file = run_dir / "scorecard.json"
    if sc_file.is_file():
        try:
            return json.loads(sc_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return generate_scorecard(run_id, run_dir)


def print_scorecard(sc: dict[str, Any]) -> None:
    run_id = sc.get("run_id", "?")
    created_at = sc.get("created_at", "")[:19]
    sm = sc.get("summary", {})
    fn = sc.get("funnel", {})
    cg = sc.get("cognition", {})
    gf = sc.get("gate_failure_reasons", {})
    sf = sc.get("stage_one_failure_reasons", {})
    oos = sc.get("oos_retention", {})

    print("=" * 68)
    print(f"【AlphaAgent 挖掘 Run 评分卡】 Run ID: {run_id} ({created_at})")
    print("=" * 68)

    print("\n1. 产出效率 (Efficiency)")
    print(f"  - 评估吞吐:          {sm.get('eval_throughput', 0):.2f} eval/min")
    print(f"  - 墙钟耗时:          {sm.get('wall_time_minutes', 0):.1f} min")
    print(f"  - 算力利用率:        {(sm.get('compute_utilization', 0) * 100):.1f}% (工具执行占比)")
    print(f"  - LLM 缓存命中率:    {(sm.get('cache_hit_rate', 0) * 100):.1f}%")
    print(f"  - Token 转化效率:    {sm.get('token_factor_yield', 0):.2f} 候选因子 / 100k Token")
    print(f"  - 语法自愈成功率:    {(sm.get('valid_attempt_ratio', 0) * 100):.1f}%")

    print("\n2. 质量漏斗转化 (Quality Funnel)")
    u_eval = fn.get("unique_train_evaluated", 0)
    cand = fn.get("candidate_stored", 0)
    prod = fn.get("production_stored", 0)
    s1_yield = fn.get("stage_one_yield_pct", 0)
    g_surv = fn.get("gate_survival_pct", 0)
    print(f"  - 独特尝试训练:      {u_eval} 次")
    print(f"  - 入候选池:          {cand} 个 (海选过线率: {s1_yield:.1f}%)")
    print(f"  - 晋升正式库:        {prod} 个 (实盘存活率: {g_surv:.1f}%)")
    print(f"  - 决策断层 (未提交): {fn.get('unsubmitted_promising', 0)} 个训练过线因子被 LLM 遗漏")
    print(f"  - 近海选线因子数:    {fn.get('near_miss_count', 0)} 个 (差临门一脚)")
    print(f"  - 样本外过拟合嫌疑:  {'是 [!] (存在方向反转或衰减>90%)' if fn.get('overfit_suspected') else '否 [OK]'}")
    med_ret = oos.get("median_ic_retention")
    ret_str = f"{(med_ret * 100):.1f}%" if med_ret is not None else "N/A"
    print(f"  - OOS 样本外保留比:  {ret_str} (样本数={oos.get('matched_count', 0)})")

    print("\n3. 认知与对账 (Cognition & Prediction)")
    p_conf = cg.get("prediction_confirmed", 0)
    p_contra = cg.get("prediction_contradicted", 0)
    p_ratio = cg.get("confirmed_ratio_pct")
    ratio_str = f"{p_ratio:.1f}%" if p_ratio is not None else "N/A"
    print(f"  - 预测对账证实率:    {ratio_str} ({p_conf} 证实 / {p_contra} 证伪 / {cg.get('prediction_partial', 0)} 部分吻合)")
    print(f"  - 消融复合算子增益:  +{cg.get('ablation_added_value', 0)} 增益 / -{cg.get('ablation_destroyed_value', 0)} 损毁 / {cg.get('ablation_flipped_signal', 0)} 信号反转")
    print(f"  - 撞已知死路率:      {(cg.get('dup_dead_end_rate', 0) * 100):.1f}%")

    if gf:
        print("\n4. 实盘 Engine Gate 拦截归因 (Why Failed)")
        for reason, count in sorted(gf.items(), key=lambda kv: -kv[1]):
            print(f"  - [{reason:<20}] 拦截 {count} 次")
    if sf:
        print("\n5. 海选 Stage One 拦截归因")
        for reason, count in sorted(sf.items(), key=lambda kv: -kv[1]):
            print(f"  - [{reason:<20}] 拦截 {count} 次")
    print("=" * 68)


def compare_scorecards(sc_a: dict[str, Any], sc_b: dict[str, Any]) -> None:
    id_a = sc_a.get("run_id", "Run-A")[:10]
    id_b = sc_b.get("run_id", "Run-B")[:10]
    sm_a, sm_b = sc_a.get("summary", {}), sc_b.get("summary", {})
    fn_a, fn_b = sc_a.get("funnel", {}), sc_b.get("funnel", {})
    cg_a, cg_b = sc_a.get("cognition", {}), sc_b.get("cognition", {})

    print("=" * 72)
    print(f"【AlphaAgent 性能与质量双 Run 横向对比】 {id_a} vs {id_b}")
    print("=" * 72)
    fmt_row = f"  {'{:<28}':<28} | {'{:<16}':<16} | {'{:<16}':<16} | {'{:<10}'}"

    print(fmt_row.format("度量指标", f"Run-A ({id_a})", f"Run-B ({id_b})", "变动/提升"))
    print("-" * 78)

    metrics_to_comp = [
        ("评估吞吐 (eval/min)", sm_a.get("eval_throughput", 0), sm_b.get("eval_throughput", 0), False, "{:.2f}"),
        ("墙钟总时长 (min)", sm_a.get("wall_time_minutes", 0), sm_b.get("wall_time_minutes", 0), True, "{:.1f}"),
        ("算力利用率 (%)", sm_a.get("compute_utilization", 0) * 100, sm_b.get("compute_utilization", 0) * 100, False, "{:.1f}%"),
        ("Token 因子转化率", sm_a.get("token_factor_yield", 0), sm_b.get("token_factor_yield", 0), False, "{:.2f}"),
        ("语法自愈率 (%)", sm_a.get("valid_attempt_ratio", 0) * 100, sm_b.get("valid_attempt_ratio", 0) * 100, False, "{:.1f}%"),
        ("海选过线率 (%)", fn_a.get("stage_one_yield_pct", 0), fn_b.get("stage_one_yield_pct", 0), False, "{:.1f}%"),
        ("实盘 Gate 存活率 (%)", fn_a.get("gate_survival_pct", 0), fn_b.get("gate_survival_pct", 0), False, "{:.1f}%"),
        ("正式库产出数", fn_a.get("production_stored", 0), fn_b.get("production_stored", 0), False, "{:d}"),
        ("候选池产出数", fn_a.get("candidate_stored", 0), fn_b.get("candidate_stored", 0), False, "{:d}"),
        ("预测证实率 (%)", cg_a.get("confirmed_ratio_pct") or 0, cg_b.get("confirmed_ratio_pct") or 0, False, "{:.1f}%"),
        ("撞死路率 (%)", cg_a.get("dup_dead_end_rate", 0) * 100, cg_b.get("dup_dead_end_rate", 0) * 100, True, "{:.1f}%"),
    ]

    for label, val_a, val_b, lower_is_better, fmt_spec in metrics_to_comp:
        val_a_s = fmt_spec.format(val_a)
        val_b_s = fmt_spec.format(val_b)
        delta = val_b - val_a
        if abs(delta) < 1e-4:
            chg = "="
        else:
            good = (delta < 0) if lower_is_better else (delta > 0)
            sign = "+" if delta > 0 else ""
            delta_s = fmt_spec.format(delta)
            chg = f"{sign}{delta_s} {'[+]' if good else '[-]'}"
        print(fmt_row.format(label, val_a_s, val_b_s, chg))
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaAgent Run Scorecard & Benchmark CLI")
    parser.add_argument("--run", type=str, help="单个 run_id")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), help="横向对比两个 run_id")
    parser.add_argument("--list", type=int, default=0, help="列出最近 N 个 run 并概要评分")
    args = parser.parse_args()

    if args.compare:
        sc_a = _load_or_gen_scorecard(args.compare[0])
        sc_b = _load_or_gen_scorecard(args.compare[1])
        if not sc_a or not sc_b:
            return 1
        compare_scorecards(sc_a, sc_b)
        return 0

    if args.run:
        sc = _load_or_gen_scorecard(args.run)
        if not sc:
            return 1
        print_scorecard(sc)
        return 0

    if args.list > 0:
        if not LOG_ROOT.is_dir():
            print("未找到日志目录", file=sys.stderr)
            return 1
        run_dirs = sorted([d for d in LOG_ROOT.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)[:args.list]
        header = f"{'Run ID':<14} | {'时间':<16} | {'吞吐':<10} | {'尝试':<6} | {'候选':<6} | {'正式':<6} | {'Gate%':<8}"
        print(header)
        print("-" * len(header))
        for d in run_dirs:
            sc = _load_or_gen_scorecard(d.name)
            if not sc:
                continue
            sm = sc.get("summary", {})
            fn = sc.get("funnel", {})
            print(f"{d.name:<14} | {sc.get('created_at','')[:16]:<16} | {sm.get('eval_throughput',0):.2f}/m   | {fn.get('unique_train_evaluated',0):<6} | {fn.get('candidate_stored',0):<6} | {fn.get('production_stored',0):<6} | {fn.get('gate_survival_pct',0):.1f}%")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

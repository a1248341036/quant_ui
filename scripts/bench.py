#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaAgent 挖掘评测 Bench CLI 统一入口。

用法：
    # 1. 执行一次评测（一条命令到底）
    python scripts/bench.py run -m "prompt: 修 S4 换手口径"
    python scripts/bench.py run --detach -m "后台跑评测"

    # 2. 评测指标与对比
    python scripts/bench.py score <run_id>
    python scripts/bench.py diff <run_id> [--baseline <base_run_id>]

    # 3. 基线管理
    python scripts/bench.py baseline --set <run_id> -m "基线说明"
    python scripts/bench.py baseline --show

    # 4. 台账与历史追踪
    python scripts/bench.py ls [--limit 20]
    python scripts/bench.py trend [--metric effective_novelty_rate] [--last 20]

    # 5. 配置管理与批量回填
    python scripts/bench.py config show
    python scripts/bench.py config set max_turns=8
    python scripts/bench.py backfill [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

UI_ROOT = ROOT / "logs" / "factor_mining" / "ui"


def cmd_run(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.bench.runner import run_bench
    code, _, _ = run_bench(note=args.message, detach=args.detach)
    return code


def cmd_score(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.run_metrics import generate_scorecard
    from alphaagent.factor.mining.bench.compare import format_one_page_report
    run_dir = UI_ROOT / args.run_id
    if not run_dir.is_dir():
        print(f"错误: 未找到 run 目录 {run_dir}", file=sys.stderr)
        return 1
    sc = generate_scorecard(args.run_id, run_dir)
    print(format_one_page_report(sc, base_sc=None))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.run_metrics import generate_scorecard
    from alphaagent.factor.mining.bench.baseline import load_baseline
    from alphaagent.factor.mining.bench.compare import compare_scorecards, format_one_page_report
    curr_dir = UI_ROOT / args.run_id
    if not curr_dir.is_dir():
        print(f"错误: 未找到 run 目录 {curr_dir}", file=sys.stderr)
        return 1
    curr_sc = generate_scorecard(args.run_id, curr_dir)

    base_sc = None
    if args.baseline:
        b_dir = UI_ROOT / args.baseline
        if not b_dir.is_dir():
            print(f"错误: 未找到基线目录 {b_dir}", file=sys.stderr)
            return 1
        base_sc = generate_scorecard(args.baseline, b_dir)
    else:
        b_payload = load_baseline()
        if not b_payload:
            print("错误: 尚未设定冻结基线，请先运行 `python scripts/bench.py baseline --set <run_id>` 或指定 `--baseline`", file=sys.stderr)
            return 1
        base_sc = b_payload.get("scorecard")

    diff_res = compare_scorecards(curr_sc, base_sc)
    report = format_one_page_report(curr_sc, base_sc, diff_res)
    print(report)

    if args.report_md:
        md_file = Path(args.report_md)
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(report + "\n", encoding="utf-8")
        print(f"已导出报告至: {md_file}")

    return 1 if diff_res.get("overall") == "REGRESSED" else 0


def cmd_baseline(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.bench.baseline import load_baseline, set_baseline
    if args.set:
        res = set_baseline(args.set, note=args.message or "")
        print(f"已成功将 run {args.set} 设定为当前冻结基线:")
        print(f"  时间: {res.get('baseline_set_at')}")
        print(f"  配置哈希: {res.get('config_hash')}")
        if res.get("note"):
            print(f"  说明: {res.get('note')}")
        return 0

    if args.show:
        b = load_baseline()
        if not b:
            print("暂无已设定的基线。")
            return 0
        print(f"当前冻结基线: {b.get('run_id')} (设定于 {b.get('baseline_set_at')})")
        print(f"配置哈希: {b.get('config_hash')} | Commit: {b.get('commit')}")
        if b.get("note"):
            print(f"说明: {b.get('note')}")
        return 0

    print("请使用 `bench baseline --set <run_id>` 或 `bench baseline --show`")
    return 1


def cmd_ls(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.bench.ledger import load_ledger
    rows = load_ledger()
    if not rows:
        print("评测台账为空（尚未通过 `bench run` 记录）。")
        return 0

    if args.csv:
        import csv
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        return 0

    if args.limit > 0:
        rows = rows[-args.limit:]

    print("=" * 86)
    print(f"【AlphaAgent 挖掘评测台账】(最近 {len(rows)} 条)")
    print("=" * 86)
    fmt = "  {:<16}  {:<26}  {:<8}  {:<16}  {:<10}  {}"
    print(fmt.format("评测时间", "Run ID", "Commit", "海选过线率", "有效新颖率", "总评"))
    print("-" * 86)
    for r in reversed(rows):
        ts = str(r.get("created_at") or "")[:16].replace("T", " ")
        rid = str(r.get("run_id") or "")[:26]
        commit = str(r.get("commit") or "")[:8]
        s1 = f"{float(r.get('stage_one_yield_pct') or 0.0):.1f}%"
        nov = f"{float(r.get('effective_novelty_rate') or 0.0):.2f}"
        verdict = str(r.get("verdict") or "—")
        print(fmt.format(ts, rid, commit, s1, nov, verdict))
    print("=" * 86)
    return 0


def cmd_trend(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.bench.trend import format_trend_table, get_metric_trend
    metric = args.metric or "effective_novelty_rate"
    items = get_metric_trend(metric, last=args.last, config_hash=args.config_hash)
    if not items:
        print(f"未找到指标 {metric} 的趋势数据。")
        return 0

    if args.csv:
        import csv
        if items:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(items[0].keys()))
            writer.writeheader()
            writer.writerows(items)
        return 0

    print(format_trend_table(metric, items))
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from scripts.backfill_run_metrics import run_backfill
    run_backfill(limit=args.limit, dry_run=args.dry_run)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    from alphaagent.factor.mining.bench.config import (
        compute_config_hash,
        load_bench_config,
        set_config_value,
    )
    if args.action == "show":
        cfg = load_bench_config()
        h = compute_config_hash(cfg)
        print(f"=== Bench 评测冻结配置 (config_hash: {h}) ===")
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return 0

    if args.action == "set":
        if not args.key_value or "=" not in args.key_value:
            print("错误: 请按 key=value 格式传参，例如: `python scripts/bench.py config set max_turns=8`", file=sys.stderr)
            return 1
        k, v = args.key_value.split("=", 1)
        try:
            cfg = set_config_value(k.strip(), v.strip())
            h = compute_config_hash(cfg)
            print(f"已更新配置 {k.strip()} = {v.strip()} (新 config_hash: {h})")
            return 0
        except Exception as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaAgent 挖掘评测 Bench")
    sub = parser.add_subparsers(dest="subcommand", help="子命令")

    # 1. run
    p_run = sub.add_parser("run", help="启动一次标准化评测并比对基线")
    p_run.add_argument("-m", "--message", default="", help="评测说明备注（如：prompt 修换手率）")
    p_run.add_argument("--detach", action="store_true", help="后台运行")

    # 2. score
    p_score = sub.add_parser("score", help="只打分（对已有 run 产出 v4 指标报告）")
    p_score.add_argument("run_id", help="目标 run 目录名/ID")

    # 3. diff
    p_diff = sub.add_parser("diff", help="比对 run 与基线差异")
    p_diff.add_argument("run_id", help="目标 run 目录名/ID")
    p_diff.add_argument("--baseline", help="指定作为对比基准的 run_id（缺省取全局冻结基线）")
    p_diff.add_argument("--report-md", help="输出 Markdown 报告至指定文件路径")

    # 4. baseline
    p_base = sub.add_parser("baseline", help="基线快照管理")
    p_base.add_argument("--set", metavar="RUN_ID", help="将指定 run_id 冻结为全局基线")
    p_base.add_argument("--show", action="store_true", help="查看当前生效的全局基线")
    p_base.add_argument("-m", "--message", default="", help="基线备注说明")

    # 5. ls
    p_ls = sub.add_parser("ls", help="查看评测台账")
    p_ls.add_argument("--limit", type=int, default=20, help="显示最近 N 条")
    p_ls.add_argument("--csv", action="store_true", help="以 CSV 格式输出")

    # 6. trend
    p_trend = sub.add_parser("trend", help="按 Run 查看指标历史变化趋势")
    p_trend.add_argument("--metric", default="effective_novelty_rate", help="目标指标字段名")
    p_trend.add_argument("--last", type=int, default=20, help="显示最近 N 个 run")
    p_trend.add_argument("--config-hash", help="过滤指定配置哈希")
    p_trend.add_argument("--csv", action="store_true", help="以 CSV 格式输出")

    # 7. backfill
    p_bf = sub.add_parser("backfill", help="对历史 run 批量回填 v4 指标")
    p_bf.add_argument("--limit", type=int, default=0, help="限制处理数量")
    p_bf.add_argument("--dry-run", action="store_true", help="不写回 scorecard.json")

    # 8. config
    p_cfg = sub.add_parser("config", help="评测配置管理")
    p_cfg.add_argument("action", choices=["show", "set"], help="show 查看配置, set 修改配置")
    p_cfg.add_argument("key_value", nargs="?", help="设置键值对，例如 max_turns=8")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        return 0

    cmds = {
        "run": cmd_run,
        "score": cmd_score,
        "diff": cmd_diff,
        "baseline": cmd_baseline,
        "ls": cmd_ls,
        "trend": cmd_trend,
        "backfill": cmd_backfill,
        "config": cmd_config,
    }
    return cmds[args.subcommand](args)


if __name__ == "__main__":
    raise SystemExit(main())

"""AlphaAgent 挖掘 run 量化指标汇总：效率（第一层）+ 漏斗质量（第二层）。

只读既有产物，不新增埋点：
- logs/factor_mining/ui/<run_id>/run_*.jsonl —— 事件轨迹（usage/thinking/
  tool_results 时间与结果）；
- logs/factor_mining/ui/<run_id>/steps.log —— 分步日志（stage_one/two/
  engine_gate/promoted 判定行）。

用法：
  python scripts/alphaagent_metrics.py                 # 全部 run + 汇总
  python scripts/alphaagent_metrics.py --last 5        # 最近 5 个 run
  python scripts/alphaagent_metrics.py --runs id1 id2  # 指定 run
  python scripts/alphaagent_metrics.py --json out.json # 同时落 JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "logs" / "factor_mining" / "ui"
LLM_TOKENS_PER_SECOND = 55.0  # 生成速度估计，用于把输出 tokens 折算成墙钟分钟


def _iter_run_events(run_dir: Path) -> list[dict]:
    events: list[dict] = []
    for f in sorted(run_dir.glob("run_*.jsonl")):
        if "messages" in f.name or "summary" in f.name:
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    events.sort(key=lambda e: e.get("ts") or "")
    return events


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def compute_run_metrics(run_id: str, run_dir: Path) -> dict:
    ev = _iter_run_events(run_dir)
    m: dict = {"run_id": run_id}
    stamps = [_parse_ts(e.get("ts") or "") for e in ev]
    good = [t for t in stamps if t]
    m["wall_minutes"] = round((good[-1] - good[0]).total_seconds() / 60, 1) if len(good) >= 2 else 0.0

    # LLM 侧
    tot = next((e for e in reversed(ev) if e.get("event") == "usage_total"), None) or {}
    m["llm_calls"] = int(tot.get("calls") or 0)
    m["input_k_tokens"] = round((tot.get("input_tokens") or 0) / 1000, 1)
    m["output_k_tokens"] = round((tot.get("output_tokens") or 0) / 1000, 1)
    cache_in = tot.get("cache_input_tokens") or 0
    m["cache_hit_rate"] = round(cache_in / (tot.get("input_tokens") or 1), 4)
    m["llm_gen_minutes_est"] = round((tot.get("output_tokens") or 0) / LLM_TOKENS_PER_SECOND / 60, 1)
    m["thinking_k_chars"] = round(sum(len(e.get("content") or "") for e in ev
                                      if e.get("event") == "agent_thinking") / 1000, 1)

    # 工具侧 + 漏斗（tool_results）
    tool_seconds = 0.0
    tool_counts: dict[str, int] = {}
    n_eval = n_eval_val = n_submit = stored = candidate_stored = 0
    for e in ev:
        if e.get("event") != "tool_results":
            continue
        for r in e.get("results") or []:
            name = r.get("name") or "?"
            tool_counts[name] = tool_counts.get(name, 0) + 1
            tool_seconds += r.get("elapsed_seconds") or 0
            if name in ("evaluate_factor", "eval_on_train_set"):
                n_eval += 1
            if name == "eval_on_val_set":
                n_eval_val += 1
            if name == "submit_factor":
                n_submit += 1
                res = r.get("result") or {}
                if res.get("stored"):
                    stored += 1
                elif res.get("candidate_stored"):
                    candidate_stored += 1
    m["tool_minutes"] = round(tool_seconds / 60, 1)
    m["n_eval"] = n_eval
    m["n_eval_val"] = n_eval_val
    m["n_submit"] = n_submit
    m["stored_production"] = stored
    m["stored_candidate"] = candidate_stored
    m["tool_calls_top"] = dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])[:5])

    # 漏斗细节（steps.log 判定行）
    gate_fails: dict[str, int] = {}
    stage = {"stage_one_pass": 0, "stage_one_fail": 0, "stage_two_pass": 0,
             "stage_two_fail": 0, "gate_pass": 0, "gate_fail": 0, "promoted": 0,
             "blind_fail": 0}
    steps = run_dir / "steps.log"
    if steps.exists():
        for line in steps.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "submit.stage_one |" in line:
                key = "stage_one_pass" if " passed=True" in line else "stage_one_fail"
                stage[key] += 1
            elif "submit.stage_two |" in line:
                key = "stage_two_pass" if " passed=True" in line else "stage_two_fail"
                stage[key] += 1
            elif "submit.engine_gate |" in line:
                key = "gate_pass" if " passed=True" in line else "gate_fail"
                stage[key] += 1
                if " passed=False" in line:
                    fm = re.search(r"fail=\[([^\]]*)\]", line)
                    for reason in (fm.group(1).split(",") if fm else []):
                        gate_fails[reason.strip()] = gate_fails.get(reason.strip(), 0) + 1
            elif "submit.promoted" in line:
                stage["promoted"] += 1
            elif "blind_test_failed" in line:
                stage["blind_fail"] += 1
    m["funnel"] = stage
    m["gate_fail_reasons"] = gate_fails

    # 成本效率
    delivered = stored + candidate_stored
    m["minutes_per_delivered"] = round(m["wall_minutes"] / delivered, 1) if delivered else None
    m["output_tokens_per_delivered_k"] = round(m["output_k_tokens"] / delivered, 1) if delivered else None
    return m


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
            cells += f"{v if v is not None else '-':>{w}}" if not isinstance(v, float) else f"{v:>{w}.2f}" if key == "cache_hit_rate" else f"{v:>{w}.1f}"
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

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "runs": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"JSON 已写入 {args.json}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""给在用题库打「质量标签」（2026-10-03）：把"效果不好"的题标出来，避免继续烧预算。

标签（字段 `perf_label`，附 `perf_evidence` 证据）：
  ok               复现至少过线过一次（可用，值得发散）
  not_expressible  0 次过线，且**原文锚/结构算子反复不达标**（≥8 次）→ DSL 表达不了
  weak_signal      0 次过线，判定 ≥6 次**从未达到 IC 门槛**(max|IC|<0.010) → 该题做不出信号
  deprecated       状态机已 abandoned（永久出局）
  untried          尚无判定记录（新鲜题）

判定依据全部来自已落盘日志 + 状态机（不发起任何评估、不碰盲测段）。
用法：python scripts/label_question_quality.py [--runs 40] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

def _repo_root() -> Path:
    """定位"真正的工作区根"：worktree 里的 logs/artifacts 可能是空占位 → 必须命中真实产物路径。"""
    root = Path(__file__).resolve().parents[1]
    for p in [root, *root.parents]:
        if (p / "logs" / "factor_mining" / "ui").is_dir() and (p / "artifacts").is_dir():
            return p
    return root


REPO = _repo_root()
K = REPO / "data" / "research_reports" / "knowledge"
BANK = K / "research_questions.jsonl"
STATE = REPO / "artifacts" / "alphaagent" / "research_specs" / "question_state_report.jsonl"
RUNS = REPO / "logs" / "factor_mining" / "ui"

JUDGE = re.compile(r"report_reproduce_judge \| qid=(\S+) factor=\S+ (.*)$", re.MULTILINE)
ICANY = re.compile(r"ic=([+-]?[\d.]+) icir=([+-]?[\d.]+)")
MIN_IC = 0.010
ANCHOR_MIN = 8      # 锚/结构算子反复不达标的下限
WEAK_MIN = 6        # 弱信号判定次数下限


def gather(run_limit: int) -> dict[str, dict]:
    """扫最近 run_limit 个 run 的 steps.log，统计每题的判定/最高|IC|/过线/锚失败。"""
    stats: dict[str, dict] = defaultdict(lambda: {"judged": 0, "max_ic": 0.0, "pass": 0, "anchor_bad": 0})
    dirs = sorted([d for d in RUNS.iterdir() if (d / "steps.log").is_file()],
                  key=lambda p: (p / "steps.log").stat().st_mtime, reverse=True)[:run_limit]
    for d in dirs:
        t = (d / "steps.log").read_text(encoding="utf-8", errors="replace")
        for m in JUDGE.finditer(t):
            q, tail = m.group(1), m.group(2)
            mm = ICANY.search(tail)
            if mm:
                st = stats[q]
                st["judged"] += 1
                st["max_ic"] = max(st["max_ic"], abs(float(mm.group(1))))
                if "PASS" in tail:
                    st["pass"] += 1
        for m in re.finditer(r"report_fidelity_check \| qid=(\S+)[^\n]*passed=False", t):
            stats[m.group(1)]["anchor_bad"] += 1
    return stats, [d.name for d in dirs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=40, help="扫描最近 N 个 run 的日志")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stats, run_names = gather(args.runs)
    abandoned = set()
    for line in STATE.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            o = json.loads(line)
            if str(o.get("state")) == "abandoned":
                abandoned.add(str(o.get("question_id")))

    rows = [json.loads(l) for l in BANK.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    counts = Counter()
    causes = Counter()
    for r in rows:
        q = str(r.get("question_id"))
        st = stats.get(q) or {"judged": 0, "max_ic": 0.0, "pass": 0, "anchor_bad": 0}
        # 原因分类（与状态独立，便于诊断"为什么差"）：
        if st["pass"] > 0:
            cause = ""
        elif st["anchor_bad"] >= ANCHOR_MIN:
            cause = "not_expressible"     # 原文锚/结构算子反复不达标 → DSL 表达不了
        elif st["judged"] >= WEAK_MIN and st["max_ic"] < MIN_IC:
            cause = "weak_signal"         # 判定够多但从未够到 IC 门槛 → 做不出信号
        elif q in abandoned:
            cause = "no_reproduce_pass"   # 状态机已判复现失败
        elif st["judged"] >= WEAK_MIN:
            cause = "no_pass_sample_ok"   # 有判定但没结论（样本不足以判死）
        else:
            cause = ""
        if q in abandoned:
            label = "deprecated"
        elif cause:
            label = "suspect"
        elif st["judged"] == 0:
            label = "untried"
        else:
            label = "ok"
        r["perf_label"] = label
        r["perf_cause"] = cause
        r["perf_evidence"] = {
            "judged": st["judged"], "max_abs_ic": round(st["max_ic"], 4),
            "pass": st["pass"], "anchor_bad": st["anchor_bad"],
            "scanned_runs": len(run_names),
        }
        counts[label] += 1
        causes[cause or "(可用)"] += 1

    print(f"扫描最近 {len(run_names)} 个 run；题库 {len(rows)} 题 → 标签分布：")
    for k, v in counts.most_common():
        print(f"   {k:<18}{v:>4} 题")
    print("\n   原因分布（perf_cause）：")
    for k, v in causes.most_common():
        print(f"   {k:<20}{v:>4} 题")
    struct_ok = [r for r in rows if r.get("has_reproducible_structure") is True
                 and r.get("method_type") not in ("ml_model", "graph_deep")]
    print(f"\n   其中「可出题」（结构准入 & 非 ml/graph）{len(struct_ok)} 题：")
    for k, v in Counter(str(r.get("perf_label")) for r in struct_ok).most_common():
        print(f"     {k:<18}{v:>4} 题")
    print("\n   被标记为「效果不好」（suspect / deprecated）的题示例：")
    for r in rows:
        if r["perf_label"] in ("suspect", "deprecated"):
            e = r["perf_evidence"]
            print(f"     {r['perf_label']:<11}{str(r.get('perf_cause')):<18}{str(r.get('question_id')):<12}"
                  f"judged={e['judged']:<3} max|IC|={e['max_abs_ic']:<7} 锚失败={e['anchor_bad']:<3}"
                  f" {str(r.get('topic'))[:26]}")

    if args.dry_run:
        print("\n（dry-run：未写回）")
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(BANK, BANK.with_suffix(f".jsonl.bak-perf-{stamp}"))
    BANK.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\n✓ 已写回 {BANK.name}（备份 .bak-perf-{stamp}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

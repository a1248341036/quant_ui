#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研报模式巡检（只读）：复现过线率 / 保真度锚 / 血统门禁三分法 / 死路拦截按阶段 / 错误分布。

背景：`alphaagent-run-watch` skill 的"run 结束后标准动作"里点名了本脚本，
但仓库里一直没有它（skill 与实现脱节，2026-10-04 整夜巡检时才把口径固化下来）。
本脚本只读 `logs/factor_mining/ui/<run_id>/` 下的 `run_summary.json` 与 `steps.log`，
不写任何文件、不触发评估、不碰 run。

用法::

    # 巡检今天 00:00 起的所有 run
    .venv\\Scripts\\python.exe scripts/patrol_report_mode.py

    # 指定起始时间（ISO，本地时区）
    .venv\\Scripts\\python.exe scripts/patrol_report_mode.py --since 2026-10-03T23:00

    # 只看指定 run
    .venv\\Scripts\\python.exe scripts/patrol_report_mode.py --runs 6597fe0e0c84,61ded42e3ff8

口径（与 skill 一致，避免"看起来像故障"的假警报）：
- 复现判定数 = `report_reproduce_judge` **带指标**的行；`phase=diverge` 的"跳过"行不算失败；
- 保真度 = `report_fidelity_check` 的 `passed=True/False`；
- 血统门禁拦截三分法：**空传** / **同课题（忽略大小写与下划线）** / **跨课题**；
- 死路拦截（`memory_blocked_duplicate`）按当前阶段归类（复现阶段已豁免硬拦，2026-10-04 起）。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "logs" / "factor_mining" / "ui"
_GATE_RE = re.compile(r"课题 (RQ_[0-9a-fA-F]+)[^\n]{0,400}?当前传入=([^\n]{0,120})")
_EVAL_ERR_RE = re.compile(r"memory\.record[^\n]*?verdict=eval_error[^\n]*")
_PHASE_RE = re.compile(r"report_state_machine \| turn=\d+ qid=(RQ_\w+) phase=(\w+)")


def _norm(text: str) -> str:
    """血统比对键：忽略大小写与下划线（与 report_channels.lineage_key 同口径）。"""
    return str(text or "").lower().replace("_", "")


def _run_rows(root: Path, since: datetime) -> list[Path]:
    rows = []
    for d in root.iterdir() if root.is_dir() else []:
        if not d.is_dir():
            continue
        log = d / "steps.log"
        ref = log if log.exists() else d
        if datetime.fromtimestamp(ref.stat().st_mtime) >= since:
            rows.append(d)
    return sorted(rows, key=lambda p: (p / "steps.log").stat().st_mtime if (p / "steps.log").exists() else p.stat().st_mtime)


def patrol_run(run_dir: Path) -> dict:
    text = ""
    log = run_dir / "steps.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
    summary = {}
    sp = run_dir / "run_summary.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    funnel = summary.get("candidate_funnel") or {}

    judge_lines = [ln for ln in text.splitlines() if "report_reproduce_judge" in ln]
    judged = [ln for ln in judge_lines if "跳过" not in ln]
    gate = _GATE_RE.findall(text)
    empty = [g for g in gate if g[1].split("。")[0].strip() in ("(空)", "", "None")]
    same = [g for g in gate if g not in empty and _norm(g[0]) in _norm(g[1])]
    cross = [g for g in gate if g not in empty and g not in same]

    # 死路拦截按阶段归类
    phase = qid = "?"
    deadend = Counter()
    for line in text.splitlines():
        m = _PHASE_RE.search(line)
        if m:
            qid, phase = m.group(1), m.group(2)
        if "memory_blocked_duplicate" in line:
            deadend[phase] += 1

    errors = Counter()
    for m in _EVAL_ERR_RE.finditer(text):
        raw = m.group(0)
        err = raw.split("err=", 1)[1].split("|")[0].strip() if "err=" in raw else "无err字段"
        errors[re.sub(r"RQ_[0-9a-fA-F]+", "RQ_*", err)[:64]] += 1

    return {
        "run_id": run_dir.name,
        "started": (run_dir / "run_meta.json").stat().st_mtime if (run_dir / "run_meta.json").exists() else None,
        "outcome": summary.get("outcome") or ("(无 run_summary)" if not summary else "-"),
        "reason": summary.get("termination_reason") or "-",
        "train": funnel.get("unique_train_evaluated"),
        "val": funnel.get("unique_val_evaluated"),
        "cand": funnel.get("candidate_stored"),
        "prod": funnel.get("production_stored"),
        "judged": len(judged),
        "pass": sum(1 for ln in judged if "PASS" in ln),
        "fid": len(re.findall(r"report_fidelity_check", text)),
        "fid_ok": len(re.findall(r"report_fidelity_check.*?passed=True", text)),
        "autofill": text.count("report_lineage_autofill"),
        "recfilter": text.count("memory_suggest_lineage_filter"),
        "gate": (len(empty), len(same), len(cross)),
        "deadend": dict(deadend),
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="研报模式巡检（只读）")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--since", default=None, help="起始时间 ISO（缺省=今天 00:00）")
    ap.add_argument("--runs", default="", help="逗号分隔 run_id（给定时忽略 --since）")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [args.root / r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        since = (datetime.fromisoformat(args.since) if args.since
                 else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
        run_dirs = _run_rows(args.root, since)

    if not run_dirs:
        print("没有匹配的 run。")
        return 0

    print(f"{'run':14s} {'起':6s} {'outcome':13s} {'train':5s} {'val':4s} {'judge':5s} {'PASS':4s} "
          f"{'保真':6s} {'补全':4s} {'推荐滤':5s} {'门禁 空+同/跨':12s} {'死路(阶段)'}")
    all_err: Counter[str] = Counter()
    tot = Counter()
    for d in run_dirs:
        if not d.exists():
            print(f"{d.name:14s} 目录不存在")
            continue
        r = patrol_run(d)
        started = datetime.fromtimestamp(r["started"]).strftime("%H:%M") if r["started"] else "?"
        gate = f"{r['gate'][0]}+{r['gate'][1]}/{r['gate'][2]}"
        print(f"{r['run_id']:14s} {started:6s} {str(r['outcome'])[:13]:13s} {str(r['train']):5s} "
              f"{str(r['val']):4s} {r['judged']:5d} {r['pass']:4d} {str(r['fid_ok']) + '/' + str(r['fid']):6s} "
              f"{r['autofill']:4d} {r['recfilter']:5d} {gate:12s} {r['deadend']}")
        all_err.update(r["errors"])
        for key in ("train", "val", "cand", "prod", "judged", "pass", "fid", "fid_ok", "autofill", "recfilter"):
            tot[key] += int(r[key] or 0)

    print(f"\n合计: train={tot['train']} val={tot['val']} 候选={tot['cand']} 正式={tot['prod']} "
          f"判定={tot['judged']} PASS={tot['pass']} 保真={tot['fid_ok']}/{tot['fid']} "
          f"血统补全={tot['autofill']} 推荐过滤={tot['recfilter']}")
    print("\n== eval_error 模板 Top10 ==")
    for name, n in all_err.most_common(10):
        print(f"  {n:3d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

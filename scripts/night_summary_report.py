#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""整夜汇总：指定时间窗内所有 report run 的关键指标 + 与上一夜基线对照。

用法：
  python scripts/night_summary_report.py --since 2026-10-02T01:30 --out night_20261002.md
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(r"D:\Quant\quant_ui")
UI = REPO / "logs" / "factor_mining" / "ui"
API = "http://127.0.0.1:17891/api/alphaagent/runs"

# 上一夜（2026-10-01 report 模式）基线，用于对照
BASELINE = {
    "reproduce_ok": 22, "abandoned": 28, "train_evals": 830, "dual_gate": 33,
    "dual_gate_rate": 0.041, "candidates": 1, "production": 0,
    "reproduce_share_by_shape": 0.62,
    "reproduce_ic_median": 0.0042, "reproduce_icir_median": 0.079,
}


def runs_since(ts: float) -> list[dict]:
    try:
        with urllib.request.urlopen(API, timeout=15) as r:
            data = json.loads(r.read().decode() or "[]")
    except Exception:  # noqa: BLE001
        return []
    runs = data if isinstance(data, list) else data.get("runs", [])
    out = []
    for x in runs:
        rid = x.get("run_id") or ""
        d = UI / rid
        if not (d / "steps.log").is_file():
            continue
        if (d / "steps.log").stat().st_mtime >= ts:
            out.append(x)
    return out


def summarize(rid: str) -> dict:
    f = UI / rid / "steps.log"
    txt = f.read_text(encoding="utf-8", errors="replace")
    qstate = re.findall(r"report_state_machine \| turn=\d+ qid=(\S+) phase=(\w+)", txt)
    ok_qids = set()
    for m in re.finditer(r"reproduce_ok", txt):
        pass
    # 复现判定
    repro = re.findall(
        r"report_reproduce_judge \| qid=(\S+) factor=(\S+) (?:PASS \[([^\]]*)\]|ic=([-\d.]+) icir=([-\d.]+))", txt)
    passes = [r for r in repro if r[2]]
    # 结构锚
    anchor = Counter(re.findall(r"structure_anchor|structure_ops_missing|declared_fields_hit", txt))
    fid = Counter(re.findall(r"report_fidelity_check \| qid=\S+ factor=\S+ passed=(\w+)", txt))
    fid_reason = Counter(re.findall(r"report_fidelity_check \|[^\n]*reason=([\w:,=<]+)", txt))
    # 评估指标
    evals = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(
        r"\| evaluate \| [^|]+ \| split=train \| ic=(-?[\d.]+) \| icir=(-?[\d.]+)", txt)]
    evals = [(abs(a), abs(b)) for a, b in evals]
    dual = sum(1 for i, r in evals if i >= 0.02 and r >= 0.28)
    icirs = sorted(r for _, r in evals)
    ics = sorted(i for i, _ in evals)
    med = lambda a: a[len(a) // 2] if a else 0.0  # noqa: E731
    # 相位：日志里出现结构锚相关拦截原因 → 该 run 已带硬锚（合并后启动）
    reasons_all = " ".join(re.findall(r"report_fidelity_check \|[^\n]*reason=([\w:,=<.]+)", txt))
    hard_anchor = any(k in reasons_all for k in
                      ("structure_ops_missing", "ops_used=", "fields_used="))
    # 拦截分类：结构锚 vs 原有字段锚
    struct_block = len(re.findall(r"structure_ops_missing|ops_used=\d+<floor|fields_used=\d+<floor",
                                  reasons_all))
    field_block = len(re.findall(r"shared_(?:specific_)?fields=\d+<|shared_ops=", reasons_all))
    # 同根度（防同根灌水误读）：达双门槛因子里最大同父本占比
    # 2026-10-02 实测教训：run a898c626b991 的 22 个达双门槛因子**全部**同一记忆建议父本
    # （mix_ovlead_wma5_kg90_totcap_m15），"12.2% 达双门槛"实为 1 个机制的变体。
    _ev = {}
    for m in re.finditer(r"\| evaluate \| [^|]+ (\S+) \| split=train \| ic=(-?[\d.]+) \| icir=(-?[\d.]+)", txt):
        _ev[m.group(1)] = (abs(float(m.group(2))), abs(float(m.group(3))))
    _dual = [f for f, (i, r) in _ev.items() if i >= 0.02 and r >= 0.28]
    _par = {}
    for m in re.finditer(r"memory\.record \| (\S+) verdict=\w+[^\n]*parent=(\S+)", txt):
        _par.setdefault(m.group(1), m.group(2))
    _pc = Counter(_par.get(f, "(根)") for f in _dual)
    _top_share = (max(_pc.values()) / len(_dual)) if _dual else 0.0
    return {
        "n_unique_eval": len(_ev), "dual_factors": len(_dual),
        "distinct_parents": len(_pc), "top_parent_share": round(_top_share, 3),
        "top_parent": (_pc.most_common(1)[0][0] if _pc else ""),
        "phase": "硬锚后" if hard_anchor else "硬锚前",
        "struct_block": struct_block, "field_block": field_block,
        "run_id": rid, "topics": len({q for q, _ in qstate}),
        "phases": dict(Counter(p for _, p in qstate)),
        "repro_judge": len(repro), "repro_pass": len(passes),
        "fidelity": dict(fid), "fidelity_reason": dict(fid_reason),
        "structure_anchor_events": sum(anchor.values()),
        "evals": len(evals), "dual_gate": dual,
        "ic_median": round(med(ics), 4), "icir_median": round(med(icirs), 4),
        "errors": dict(Counter(re.findall(r"error_type=(\w+)", txt))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-02T01:30")
    ap.add_argument("--out", default="night_summary.md")
    args = ap.parse_args()
    ts = datetime.fromisoformat(args.since).timestamp()
    rs = runs_since(ts)
    rows = [summarize(x.get("run_id")) for x in rs]
    tot_eval = sum(r["evals"] for r in rows)
    tot_dual = sum(r["dual_gate"] for r in rows)
    tot_judge = sum(r["repro_judge"] for r in rows)
    tot_pass = sum(r["repro_pass"] for r in rows)
    anch = sum(r["structure_anchor_events"] for r in rows)

    lines = [f"# 整夜汇总（since {args.since}）", "",
             f"- 运行数 **{len(rows)}**；课题数 **{sum(r['topics'] for r in rows)}**",
             f"- 复现判定 **{tot_judge}** 次 / 过线 **{tot_pass}**"
             f"（{tot_pass / max(1, tot_judge) * 100:.0f}%）",
             f"- 结构锚事件 **{anch}** 次",
             f"- 训练评估 **{tot_eval}** 次；达双门槛 **{tot_dual}**"
             f"（{tot_dual / max(1, tot_eval) * 100:.1f}%）", "",
             "## 与上一夜（2026-10-01）对照", "",
             "| 指标 | 上一夜 | 本夜 |", "|---|---|---|",
             f"| 达双门槛率 | {BASELINE['dual_gate_rate'] * 100:.1f}% | "
             f"{tot_dual / max(1, tot_eval) * 100:.1f}% |",
             f"| 复现 |IC| 中位 | {BASELINE['reproduce_ic_median']} | —（见下）|",
             f"| 复现 ICIR 中位 | {BASELINE['reproduce_icir_median']} | — |", "",
             "## 逐 run", "",
             "| run | 相位 | 课题 | 复现判定/过线 | 保真度拦截(结构锚/字段锚) | 评估 | 达双门槛 | 独立父本 | 最大同根占比 | |IC|中位 | ICIR中位 | 错误 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r['run_id']}` | {r['phase']} | {r['topics']} | {r['repro_judge']}/{r['repro_pass']} | "
            f"{r['struct_block']}/{r['field_block']} | {r['evals']} | {r['dual_gate']} | "
            f"{r['distinct_parents']} | {r['top_parent_share']:.0%} | "
            f"{r['ic_median']} | {r['icir_median']} | {r['errors'] or '-'} |")
    # 相位汇总（A/B 读法）
    for ph in ("硬锚前", "硬锚后"):
        rs = [r for r in rows if r["phase"] == ph]
        if not rs:
            continue
        ev = sum(r["evals"] for r in rs)
        dg = sum(r["dual_gate"] for r in rs)
        jd = sum(r["repro_judge"] for r in rs)
        ps = sum(r["repro_pass"] for r in rs)
        sb = sum(r["struct_block"] for r in rs)
        dual_f = sum(r["dual_factors"] for r in rs)
        uniq = sum(r["n_unique_eval"] for r in rs)
        worst = max((r["top_parent_share"] for r in rs), default=0.0)
        lines += ["", f"**{ph}**：run {len(rs)} 个 | 复现 {ps}/{jd} 过线 | "
                      f"结构锚拦截 {sb} | 评估 {uniq} | 达双门槛 {dual_f}"
                      f"（{dual_f / max(1, uniq) * 100:.1f}%）| 最大同根占比 {worst:.0%}"
                      + ("  ⚠ 同根灌水，勿据此判效果" if worst >= 0.5 else "")]
    out = UI.parent.parent / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

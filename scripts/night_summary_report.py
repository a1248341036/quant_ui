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

# 夜报聚合台账：每晚一行，"与上一夜对照"的基线从这里自动滚动取，
# 取代旧硬编码 BASELINE（仅作台账为空时的兜底，见 load_prev_baseline）。
LEDGER_PATH = UI.parent / "night_summary_ledger.jsonl"


def _ledger_rows() -> list[dict]:
    try:
        raw = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in raw:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def append_ledger(entry: dict) -> None:
    """追加本夜聚合；同 --since 重复执行时替换旧行（幂等）。原子写（tmp+replace）。"""
    rows = [r for r in _ledger_rows() if r.get("since") != entry.get("since")]
    rows.append(entry)
    tmp = LEDGER_PATH.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(LEDGER_PATH)


def load_prev_baseline(since: str) -> dict | None:
    """台账中最近一条不同 --since 的聚合（=上一夜基线）；无则 None → 回落硬编码 BASELINE。"""
    for r in reversed(_ledger_rows()):
        if r.get("since") != since:
            return r
    return None


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
    # 相位：**直接读该 run 冻结的 research_spec.json**（比猜日志可靠）。
    # 三态：硬锚前（无新键）/ 部分生效（有结构准入但缺血统门禁与地板，= 后端未重启，见 Round 18）
    #      / 全套生效（结构准入 + 血统门禁 + 复杂度地板）。
    reasons_all = " ".join(re.findall(r"report_fidelity_check \|[^\n]*reason=([\w:,=<.]+)", txt))
    _phase = "硬锚前"
    # 先看 research_mode：**普通模式 run 不进研报 A/B**（2026-10-02 实测坑：监控 --research-mode
    # 缺省空 → 后端自定 technical，整夜监控自起的 run 全是普通模式，却因 spec 里含 report_policy
    # 键而被误标为"部分生效"）。
    _meta_p = UI / rid / "run_meta.json"
    _mode = ""
    if _meta_p.is_file():
        try:
            _mode = str((json.loads(_meta_p.read_text(encoding="utf-8")).get("params") or {})
                        .get("research_mode") or "")
        except Exception:  # noqa: BLE001
            _mode = ""
    _spec_p = UI / rid / "research_spec.json"
    if _mode and _mode != "report":
        _phase = f"非研报模式({_mode})"
    elif _spec_p.is_file():
        try:
            _spec = json.loads(_spec_p.read_text(encoding="utf-8"))
            _flat = json.dumps(_spec, ensure_ascii=False)
            _has_struct = "require_report_structure" in _flat
            _has_lineage = "diverge_parent_required" in _flat or "min_ops_floor" in _flat
            _phase = ("全套生效" if (_has_struct and _has_lineage)
                      else ("部分生效" if _has_struct else "硬锚前"))
        except Exception:  # noqa: BLE001
            _phase = "硬锚前"
    hard_anchor = _phase == "全套生效"
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
    # 同根≠灌水：研报模式下 diverging 只改一个维度 → 同根是**设计预期**；
    # 真正的问题是"同根且**离线**"（2026-10-02 实测 a898c626b991：22/22 挂记忆建议的 mix_*）。
    _top_parent = _pc.most_common(1)[0][0] if _pc else ""
    _top_in_lineage = bool(_top_parent.startswith("rq") or _top_parent.startswith("reproduce_of"))
    return {
        "n_unique_eval": len(_ev), "dual_factors": len(_dual),
        "distinct_parents": len(_pc), "top_parent_share": round(_top_share, 3),
        "top_parent": _top_parent, "top_parent_in_lineage": _top_in_lineage,
        "phase": _phase,
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
    ap.add_argument("--no-ledger", dest="ledger", action="store_false", default=True,
                    help="不把本夜聚合追加进 night_summary_ledger.jsonl")
    args = ap.parse_args()
    ts = datetime.fromisoformat(args.since).timestamp()
    rs = runs_since(ts)
    rows = [summarize(x.get("run_id")) for x in rs]
    tot_eval = sum(r["evals"] for r in rows)
    tot_dual = sum(r["dual_gate"] for r in rows)
    tot_judge = sum(r["repro_judge"] for r in rows)
    tot_pass = sum(r["repro_pass"] for r in rows)
    anch = sum(r["structure_anchor_events"] for r in rows)

    # 本夜聚合 → 台账（同 since 幂等替换）；对照基线优先取台账上一夜，无历史才回落硬编码
    entry = {
        "since": args.since, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_runs": len(rows), "tot_eval": tot_eval, "tot_dual": tot_dual,
        "tot_judge": tot_judge, "tot_pass": tot_pass, "anch": anch,
        "dual_gate_rate": round(tot_dual / max(1, tot_eval), 4),
        "pass_rate": round(tot_pass / max(1, tot_judge), 4) if tot_judge else None,
    }
    if args.ledger:
        append_ledger(entry)
    prev = load_prev_baseline(args.since)
    base_rate = prev["dual_gate_rate"] if prev else BASELINE["dual_gate_rate"]
    base_pass = (f"{prev['tot_pass']}/{prev['tot_judge']}"
                 if prev and prev.get("tot_judge") is not None else "—")
    base_eval = prev["tot_eval"] if prev else "—"
    base_src = (f"night_summary_ledger（since {prev['since']}）" if prev
                else "脚本内硬编码 2026-10-01（台账尚无历史）")

    lines = [f"# 整夜汇总（since {args.since}）", "",
             f"- 运行数 **{len(rows)}**；课题数 **{sum(r['topics'] for r in rows)}**",
             f"- 复现判定 **{tot_judge}** 次 / 过线 **{tot_pass}**"
             f"（{tot_pass / max(1, tot_judge) * 100:.0f}%）",
             f"- 结构锚事件 **{anch}** 次",
             f"- 训练评估 **{tot_eval}** 次；达双门槛 **{tot_dual}**"
             f"（{tot_dual / max(1, tot_eval) * 100:.1f}%）", "",
             f"## 与上一夜对照（基线来源：{base_src}）", "",
             "| 指标 | 上一夜 | 本夜 |", "|---|---|---|",
             f"| 达双门槛率 | {base_rate * 100:.1f}% | {entry['dual_gate_rate'] * 100:.1f}% |",
             f"| 复现过线 | {base_pass} | {tot_pass}/{tot_judge} |",
             f"| 训练评估次数 | {base_eval} | {tot_eval} |", "",
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
    for ph in ("全套生效", "部分生效", "硬锚前", "非研报模式(technical)"):
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
        worst_off = max((r["top_parent_share"] for r in rs if not r["top_parent_in_lineage"]),
                        default=0.0)
        lines += ["", f"**{ph}**：run {len(rs)} 个 | 复现 {ps}/{jd} 过线 | "
                      f"结构锚拦截 {sb} | 评估 {uniq} | 达双门槛 {dual_f}"
                      f"（{dual_f / max(1, uniq) * 100:.1f}%）| 最大同根占比 {worst:.0%}"
                      + (f"（其中离线同根 {worst_off:.0%}）" if worst_off else "（均在研报血统内，设计预期）")
                      + ("  ⚠ 离线同根灌水，勿据此判效果" if worst_off >= 0.5 else "")]
    out = UI.parent.parent / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\n写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

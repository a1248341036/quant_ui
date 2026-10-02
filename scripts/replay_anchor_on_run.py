#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""锚校准工具：把某个 run 真实写出的复现表达式回放到结构锚上，看拦截率与误杀是否合理。

判读：① 若**该 run 中已过线的复现**被拦 → 锚过严；② 若一行式（≤1 算子/1 字段）没被拦 → 锚过松。

2026-10-02 OCR 第二轮回评后重写（原版 7 处缺陷）：
  · 分母改为**实际参与判定的条目数**（原版用 len(exprs)，而循环只处理前 40 条 → 拦截率被系统性压低）；
  · 打印**覆盖率自检**（exprs / parent / passed_live 三者键交集），格式不匹配不再静默；
  · 同一因子名保留**全部**表达式分支（原 setdefault 只留首个，可能与 live PASS 那次不是同一条）；
  · 单条判定异常计数并继续（真实判定路径有 try 包裹，回放不应裸崩）；
  · JSON 解析失败按行计数并汇总；bank/recs 读取带保护；
  · 去掉死代码（未用的 Counter / qid 捕获组）；
  · 仓库根从 __file__ 推导（可用 --repo 覆盖），不再依赖硬编码盘符。

用法：python scripts/replay_anchor_on_run.py --run a898c626b991 [--limit 0]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_DEFAULT = _HERE.parent.parent          # <repo>/scripts/xxx.py → <repo>
sys.path.insert(0, str(_REPO_DEFAULT))

from alphaagent.factor.mining.agent.question_queue import check_reproduce_fidelity  # noqa: E402

OPRE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\(")
FRE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
# parent 既可能是 `reproduce_of:RQ_xxx` 也可能是裸课题号，两种都收
PARENT_RE = re.compile(r"memory\.record \| (\S+) verdict=\w+[^\n]*parent=(\S+)")
PASS_RE = re.compile(r"report_reproduce_judge \| qid=(RQ_\w+) factor=(\S+) PASS")
QID_RE = re.compile(r"(RQ_\w+)")


def _qid_of(parent: str) -> str:
    """从 parent 里取出课题号：parent 可能是 `reproduce_of:RQ_xxx`、`RQ_xxx_v2` 等形态。
    原版直接用整串查题库 → 全部落空且静默跳过（OCR 第二轮回评指出）。"""
    m = QID_RE.search(str(parent or ""))
    return m.group(1) if m else ""


def _load_jsonl(path: Path) -> tuple[list[dict], int]:
    """逐行解析 JSONL，返回 (记录, 坏行数)。坏行不再静默丢弃。"""
    rows: list[dict] = []
    bad = 0
    if not path.is_file():
        return rows, 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            bad += 1
    return rows, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default=str(_REPO_DEFAULT))
    ap.add_argument("--limit", type=int, default=0, help="最多判定多少条（0=全部）")
    ap.add_argument("--scope", choices=("judged", "all"), default="judged",
                    help="judged=只判真实复现判定过的因子（默认，语义正确）；all=run 内全部表达式")
    args = ap.parse_args()

    repo = Path(args.repo)
    K = repo / "data" / "research_reports" / "knowledge"
    D = repo / "logs" / "factor_mining" / "ui" / args.run
    if not (D / "steps.log").is_file():
        print(f"run 目录/日志不存在: {D}")
        return 1

    bank_rows, bank_bad = _load_jsonl(K / "research_questions.jsonl")
    bank = {str(r.get("question_id")): r for r in bank_rows}
    recs, rec_bad = _load_jsonl(K / "factor_records_pilot.jsonl")

    txt = (D / "steps.log").read_text(encoding="utf-8", errors="replace")
    parent: dict[str, str] = {}
    for m in PARENT_RE.finditer(txt):
        parent.setdefault(m.group(1), m.group(2))
    passed_live = {m.group(2) for m in PASS_RE.finditer(txt)}
    # **只判"真实复现判定过的因子"**：日志里 report_reproduce_judge 且**未标跳过**的条目。
    # 否则会把发散阶段的表达式也算进来——它们本来就不该通过保真度，拦截率会虚高且无意义。
    JUDGE_RE = re.compile(r"report_reproduce_judge \| (?:qid=\S+ )?factor=(\S+)")
    repro_judged: set[str] = set()
    for line in txt.splitlines():
        if "report_reproduce_judge" not in line or "跳过" in line:
            continue
        m = JUDGE_RE.search(line)
        if m:
            repro_judged.add(m.group(1))

    # 同一因子名保留**全部**表达式分支（不 setdefault 丢分支）
    exprs: dict[str, list[str]] = {}
    json_bad = 0
    for fp in sorted(D.glob("run_*.jsonl")):
        evs, bad = _load_jsonl(fp)
        json_bad += bad
        for ev in evs:
            if not isinstance(ev, dict):          # 实测存在非 dict 行（原版用 except 静默吞掉）
                json_bad += 1
                continue
            for item in ev.get("results") or []:
                if not isinstance(item, dict):    # 实测 results 里混有字符串条目
                    json_bad += 1
                    continue
                raw = item.get("arguments_raw")
                if not raw:
                    continue
                try:
                    a = json.loads(raw)
                except Exception:  # noqa: BLE001
                    json_bad += 1
                    continue
                name, expr = str(a.get("factor_name") or ""), str(a.get("multi_line_expr") or "")
                if name and expr and expr not in exprs.setdefault(name, []):
                    exprs[name].append(expr)

    pairs_all = [(n, e) for n, es in exprs.items() for e in es]
    pairs = [(n, e) for n, e in pairs_all if n in repro_judged] if args.scope == "judged" else pairs_all
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"run {args.run}（口径 scope={args.scope}）")
    print(f"  因子 {len(exprs)} 个 / 表达式条目 {len(pairs_all)} 条"
          f" → 真实复现判定过的因子 {len(repro_judged)} 个 / 本口径待判 {len(pairs)} 条")
    print(f"  数据自检：bank 坏行 {bank_bad}，records 坏行 {rec_bad}，run jsonl 坏行 {json_bad}")
    cov_p = len(set(exprs) & set(parent))
    cov_live = len(set(exprs) & passed_live)
    cov_j = len(repro_judged & set(exprs))
    warn = "  ⚠ 覆盖率偏低，结论可能失真" if cov_p < len(exprs) * 0.5 else ""
    print(f"  覆盖率自检：exprs∩parent={cov_p}/{len(exprs)}，"
          f"exprs∩passed_live={cov_live}/{len(passed_live)}，"
          f"exprs∩repro_judged={cov_j}/{len(repro_judged)}{warn}")

    print(f"\n{'因子':<34}{'算子':>4}{'字段':>5}  {'锚':<8}{'live过线':<9}原因")
    print("-" * 118)
    judged = blocked = errors = no_q = 0
    kill_live: list[str] = []
    for name, expr in pairs:
        q = bank.get(_qid_of(parent.get(name, "")))
        if not q:
            no_q += 1
            continue
        try:
            out = check_reproduce_fidelity(q, expr, records=recs, spec={})
        except Exception as exc:  # noqa: BLE001 —— 与真实判定路径一致：异常不中断
            errors += 1
            print(f"{name[:32]:<34}  !! 判定异常: {str(exc)[:60]}")
            continue
        judged += 1
        ok = bool(out.get("passed"))
        if not ok:
            blocked += 1
            if name in passed_live:
                kill_live.append(name)
        print(f"{name[:32]:<34}{len(set(OPRE.findall(expr))):>4}{len(set(FRE.findall(expr))):>5}  "
              f"{'通过' if ok else '拦截':<8}{'是' if name in passed_live else '':<9}"
              f"{str(out.get('reason'))[:48]}")

    if judged:
        print(f"\n拦截率 {blocked}/{judged}（未判定 {no_q} 条：父本/课题关联不上；异常 {errors} 条）")
    else:
        print("\n没有可判定的条目")
    print(f"live 曾过线但当前锚拦截 {len(kill_live)} 条 {kill_live[:5]}")
    print("判读：① 拦截率高且**一条式**（≤1 算子/字段）被拦 → 合理；")
    print("      ② 命中 `structure_ops_missing` → 模型换了结构（锚的本职，非过严）；")
    print("      ③ 只有当**忠实复现**（命中声明结构）仍被拦，才说明锚过严。")
    print("      注意：live 过线发生在锚的某个历史版本下，与本轮判定存在**校准时差**，需逐条看原因。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

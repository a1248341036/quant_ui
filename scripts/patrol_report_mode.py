# -*- coding: utf-8 -*-
"""夜间巡检 v2：跨 run 汇总（复现过线率 / 结构锚触发 / 母本强度 / 错误）。"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(r"D:\Quant\quant_ui")
UI = REPO / "logs" / "factor_mining" / "ui"

try:
    with urllib.request.urlopen("http://127.0.0.1:17891/api/alphaagent/runs", timeout=10) as r:
        data = json.loads(r.read().decode() or "[]")
    runs = data if isinstance(data, list) else data.get("runs", [])
except Exception as exc:  # noqa: BLE001
    print("API 失败:", exc)
    runs = []
act = [x for x in runs if x.get("status") in ("starting", "running", "stopping")]
print(f"活动 run = {len(act)}: {[(x.get('run_id'), x.get('status'), x.get('event_count')) for x in act]}")

# 只看今天凌晨起的 run
dirs = [d for d in sorted(UI.iterdir()) if d.is_dir() and (d / "steps.log").is_file()]
mine = []
for d in dirs:
    t = (d / "steps.log").stat().st_mtime
    if t > 1790875000:            # 约 2026-10-02 01:30 之后
        mine.append(d)
print(f"\n本夜巡检 run 数 = {len(mine)}")

tot = Counter()
OPRE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\(")
FRE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
STRUCT = {"SOFT_GATE", "CS_GROUP_RANK", "DIVERGENCE_RANK", "CS_RESIDUALIZE",
          "CS_NEUTRALIZE", "IF_THEN_ELSE", "CS_BUCKET"}
for d in mine:
    txt = (d / "steps.log").read_text(encoding="utf-8", errors="replace")
    qids = re.findall(r"report_state_machine \| turn=\d+ qid=(\S+) phase=(\w+)", txt)
    repro = re.findall(r"report_reproduce_judge \| qid=(\S+) factor=(\S+) (?:PASS \[([^\]]*)\]|ic=([-\d.]+) icir=([-\d.]+))", txt)
    fid = re.findall(r"report_fidelity_check \| qid=\S+ factor=\S+ passed=(\w+)", txt)
    anch = re.findall(r"structure_anchor", txt)
    errs = Counter(re.findall(r"error_type=(\w+)", txt))
    passes = [r for r in repro if r[2]]
    print(f"\n▸ {d.name}  题={len(set(q[0] for q in qids))}  阶段={Counter(q[1] for q in qids)}")
    print(f"   复现判定 {len(repro)} 次，过线 {len(passes)}；保真度 {len(fid)} 次 "
          f"({Counter(fid)})；结构锚出现 {len(anch)} 次；错误 {dict(errs) or '无'}")
    for r in passes[:3]:
        print(f"      PASS {r[1][:36]:<38} {r[2][:70]}")
    tot["judge"] += len(repro); tot["pass"] += len(passes); tot["fid"] += len(fid)

# 母本强度：发散阶段因子的 IC/ICIR
icirs = []
for d in mine:
    txt = (d / "steps.log").read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"\| evaluate \| \S+ (\S+) \| split=train \| ic=(-?[\d.]+) \| icir=(-?[\d.]+)", txt):
        icirs.append((abs(float(m.group(2))), abs(float(m.group(3)))))
if icirs:
    dual = sum(1 for i, r in icirs if i >= 0.02 and r >= 0.28)
    print(f"\n评估 {len(icirs)} 次：均值|IC|={sum(i for i,_ in icirs)/len(icirs):.4f} "
          f"均值|ICIR|={sum(r for _,r in icirs)/len(icirs):.3f} 达双门槛={dual} "
          f"({dual/len(icirs)*100:.1f}%)")
print(f"\n合计：复现判定 {tot['judge']} 次 / 过线 {tot['pass']}；保真度校验 {tot['fid']} 次")

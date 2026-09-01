# -*- coding: utf-8 -*-
"""临时诊断 4: 注入块全文 + tool_results 结构 + params + 经验层。用完即删。"""
import json
import sqlite3
from pathlib import Path

LOGS = Path(r"D:/Quant/quant_ui/logs/factor_mining/ui")

meta = json.loads((LOGS / "8b0a496a7c21" / "run_meta.json").read_text(encoding="utf-8"))
print("=== params ===")
print(json.dumps(meta.get("params"), ensure_ascii=False, indent=1)[:800])

f = LOGS / "8b0a496a7c21" / "run_20260901_143332.jsonl"
events = [json.loads(l) for l in f.read_text(encoding="utf-8", errors="replace").splitlines()
          if l.strip()]

retr = [e for e in events if e.get("event") == "research_memory_retrieved"]
print()
print("=== research_memory_retrieved: %d 条 ===" % len(retr))
if retr:
    c = retr[0].get("content") or retr[0].get("context") or ""
    if not c:
        print("keys:", sorted(retr[0].keys()))
        c = json.dumps(retr[0], ensure_ascii=False)
    print(str(c)[:2200])

tres = [e for e in events if e.get("event") == "tool_results"]
print()
print("=== 第一条 tool_results 结构 ===")
if tres:
    s = json.dumps(tres[0], ensure_ascii=False)
    print("keys:", sorted(tres[0].keys()), "len:", len(s))
    print(s[:1500])

last = events[-3:]
print()
print("=== 最后 3 条事件 ===")
for e in last:
    print(json.dumps({k: str(v)[:150] for k, v in e.items()}, ensure_ascii=False))

con = sqlite3.connect(r"file:D:/Quant/quant_ui/artifacts/alphaagent/research_memory.db?mode=ro", uri=True)
cur = con.cursor()
n = cur.execute("select count(*) from memory_experience").fetchone()[0]
print()
print("memory_experience rows:", n)
for r in cur.execute("select kind, name, occurrence_count, updated_at from memory_experience order by updated_at desc limit 6"):
    print(" ", r)
con.close()

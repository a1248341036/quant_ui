# -*- coding: utf-8 -*-
"""清理 memory_experience 里的历史残留 insight（2026-09-01 form_memory 逐因子分支的死数据）。

规则：
- 名字是因子名（非 global_alpha_thin）的 insight 全部删除；
- 其中内容含 "submit_factor 被否定" 的 gate 失败条目：先转写为规范的
  gate_rejected:{family} forbidden 行（occurrence_count 累加继承），再删原条目；
- global_alpha_thin 全局预警保留。
幂等：可重复运行。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DB = Path(r"D:\Quant\quant_ui\artifacts\alphaagent\research_memory.db")
NOW = "2026-09-03T00:00:00+00:00"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT id, name, content, occurrence_count, template, evidence_json, created_at, updated_at "
    "FROM memory_experience WHERE kind='insight'"
).fetchall()
print(f"insight rows before: {len(rows)}")

deleted = 0
converted = 0
kept = 0
for r in rows:
    name = str(r["name"] or "")
    content = str(r["content"] or "")
    if name.startswith("global_alpha_thin"):
        kept += 1
        print(f"  [keep] {name} (全局预警)")
        continue
    if "submit_factor 被否定" in content:
        mrow = conn.execute(
            "SELECT family, expression FROM memory_entries WHERE factor_name LIKE ? ORDER BY rowid DESC LIMIT 1",
            (f"%{name}%",),
        ).fetchone()
        family = str(mrow["family"] or "other") if mrow else "other"
        expr = str(mrow["expression"] or "") if mrow else ""
        fail_reasons = "submit 被否定"
        for marker in ("stage_two_failed:", "blind_test_failed:", "engine_gate_failed:"):
            idx = content.find(marker)
            if idx >= 0:
                fail_reasons = content[idx:idx + 120]
                break
        gate_content = (
            f"DO NOT 直接把 {family} 族的高 IC 结构当实盘可用：{name} 统计达标但被交付链路拒绝"
            f"（{fail_reasons}）。统计强 ≠ 实盘可用，新构造优先压低换手、"
            f"提升可交易性，再谈 IC。"
        )
        gate_id = hashlib.sha256(f"forbidden|gate_rejected:{family}".encode("utf-8")).hexdigest()[:20]
        existing = conn.execute(
            "SELECT occurrence_count FROM memory_experience WHERE id=?", (gate_id,)
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE memory_experience SET occurrence_count = occurrence_count + ?, updated_at=? WHERE id=?",
                (r["occurrence_count"], NOW, gate_id),
            )
            print(f"  [merge] {name} -> gate_rejected:{family} (occ += {r['occurrence_count']})")
        else:
            conn.execute(
                """INSERT INTO memory_experience
                   (id, kind, name, content, template, evidence_json, example_factors_json,
                    correlated_json, occurrence_count, run_id, created_at, updated_at)
                   VALUES (?, 'forbidden', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    gate_id,
                    f"gate_rejected:{family}",
                    gate_content,
                    r["template"],
                    json.dumps({
                        "factor_names": [name],
                        "fail_reasons": fail_reasons,
                        "converted_from": r["id"],
                        "n_failures": r["occurrence_count"],
                    }, ensure_ascii=False),
                    json.dumps([expr] if expr else [], ensure_ascii=False),
                    json.dumps([name], ensure_ascii=False),
                    int(r["occurrence_count"]),
                    "cleanup_20260903",
                    NOW,
                    NOW,
                ),
            )
            print(f"  [convert] {name} -> gate_rejected:{family} (新 forbidden 行, occ={r['occurrence_count']})")
        converted += 1
    else:
        print(f"  [delete] {name[:44]} :: {content[:56]}")
    conn.execute("DELETE FROM memory_experience WHERE id=?", (r["id"],))
    deleted += 1

conn.commit()

after = conn.execute(
    "SELECT kind, COUNT(*) AS c FROM memory_experience GROUP BY kind"
).fetchall()
print("\nafter cleanup:", {r["kind"]: r["c"] for r in after})
print(f"deleted={deleted} converted_to_forbidden={converted} kept_global={kept}")
conn.close()

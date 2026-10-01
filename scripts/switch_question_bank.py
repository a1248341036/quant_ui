#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等 v2 题库抽完 → （可选）等 run 空闲 → 备份旧题库 → 切换默认题库（可自动补 question_id）。

用法：
  python scripts/switch_question_bank.py --src question_v2_prod2.jsonl --min 100 \
         --wait --add-qid --wait-idle --max-wait 2400
  python scripts/switch_question_bank.py --restore
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

K = Path(r"D:\Quant\quant_ui\data\research_reports\knowledge")
DEFAULT = K / "research_questions.jsonl"
API = "http://127.0.0.1:17891/api/alphaagent/runs"
DROP = {"_order", "elapsed", "body_chars", "ok", "unknown_ops"}


def count(p: Path) -> int:
    return sum(1 for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()) \
        if p.is_file() else 0


def active_runs() -> int:
    try:
        with urllib.request.urlopen(API, timeout=10) as r:
            data = json.loads(r.read().decode() or "[]")
        runs = data if isinstance(data, list) else data.get("runs", [])
        return len([x for x in runs if x.get("status") in ("starting", "running", "stopping")])
    except Exception:  # noqa: BLE001
        return -1


def qid_for(src: str) -> str:
    return "RQ_" + hashlib.md5(str(src or "").encode("utf-8")).hexdigest()[:6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="question_v2_prod.jsonl")
    ap.add_argument("--min", type=int, default=60)
    ap.add_argument("--max-wait", type=int, default=2400)
    ap.add_argument("--wait", action="store_true", help="等抽够 --min 篇")
    ap.add_argument("--wait-idle", action="store_true", help="等没有活动 run 再切（避免中途换题库）")
    ap.add_argument("--add-qid", action="store_true", help="补 question_id 并清理抽取元字段")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.restore:
        baks = sorted(K.glob("research_questions.jsonl.bak-*"))
        if not baks:
            print("!! 没有备份可回滚")
            return 1
        shutil.copy2(baks[-1], DEFAULT)
        print(f"已回滚到 {baks[-1].name}（{count(DEFAULT)} 行）")
        return 0

    src = K / args.src
    t0 = time.time()
    while args.wait:
        n = count(src)
        print(f"  [{time.strftime('%H:%M:%S')}] {args.src} = {n} 篇", flush=True)
        if n >= args.min or time.time() - t0 > args.max_wait:
            break
        time.sleep(30)

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    if args.add_qid:
        seen, out = set(), []
        for r in rows:
            r = {k: v for k, v in r.items() if k not in DROP}
            qid = str(r.get("question_id") or "") or qid_for(r.get("source"))
            while qid in seen:
                qid += "x"
            seen.add(qid)
            r["question_id"] = qid
            if not str(r.get("topic") or "").strip():
                r["topic"] = Path(str(r.get("source") or "")).stem[:60]
            out.append(r)
        rows = out

    struct = [r for r in rows if r.get("has_reproducible_structure") is True]
    if len(struct) < 20:
        print(f"!! 结构化课题只有 {len(struct)} 道，太少，不切换")
        return 1

    if args.wait_idle:
        for _ in range(120):
            a = active_runs()
            if a == 0:
                print("  当前无活动 run → 执行切换", flush=True)
                break
            print(f"  [{time.strftime('%H:%M:%S')}] 活动 run = {a}，等待空闲…", flush=True)
            time.sleep(30)
        else:
            print("!! 等空闲超时，未切换（避免中途换题库）")
            return 1

    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = K / f"research_questions.jsonl.bak-{stamp}"
    shutil.copy2(DEFAULT, bak)
    DEFAULT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                       encoding="utf-8")
    print(f"\n✅ 已切换题库：{bak.name}（旧 {count(bak)} 行）→ research_questions.jsonl"
          f"（{len(rows)} 行，其中有结构 {len(struct)} 道，source={args.src}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

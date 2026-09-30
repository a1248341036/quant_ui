#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调仓频率判定的「第二意见」复核（独立提示词、反向质询）。

Spec §5.3 闸门 4：落定记录必须经独立复核；复核不通过 → 一律改 null（宁可空，不要错）。
本脚本读 `report_freq_tables.jsonl`（或任意 report_freq_*.jsonl）中 `align_guard=ok` 的记录，
重新定位原文片段，用**不同角度的提示词**独立复核 pass-1 的判断，输出：

    report_freq_verified.jsonl:
      {source, title, freq_pass1, basis_pass1, evidence_pass1,
       agree, correct_freq, reason, freq_final, label_final, guard_final}

用法::
    python scripts/verify_report_freq.py \
      -i data/research_reports/knowledge/report_freq_tables.jsonl \
      -o data/research_reports/knowledge/report_freq_verified.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "research_reports" / "parsed_mineru"
BASE_URL = os.environ.get("FACTOR_REC_BASE_URL", "http://127.0.0.1:8317/v1")
API_KEY = os.environ.get("FACTOR_REC_API_KEY", "123456")
MODEL = os.environ.get("FACTOR_REC_MODEL", "Qwen3.8-27B")

FREQ_TO_LABEL = {"daily": "label_1d_open_to_open", "weekly": "label_5d_close_to_close",
                 "monthly": "label_20d_close_to_close"}
RE_ANCHOR = re.compile(r"调仓|换仓|调仓频率|换手|持有期|持有\d|月度|每月|周度|每周|日频|每日")

PROMPT = """请对一份券商研报的「调仓频率」判断做**独立复核**，不要盲从下面的既有判断。

既有判断：调仓频率 = {freq}
其依据引文：「{quote}」

要求：
1. 先判断该引文是否**真的支持**该频率（引文若只是在做对比、或讲的是数据频率而非调仓，都不算直接支持）；
2. 再从片段中查找是否有**更明确的采用/推荐表述**（如"我们以…调仓""本报告采用…""结论建议…调仓""优于"）；
3. 输出严格 JSON：
{{"agree": true/false,
  "correct_freq": "daily|weekly|monthly|quarterly|semi_annual|bi-weekly|unknown",
  "reason": "一句话说明"}}
4. 只依据片段，不得脑补；引文不足以支撑时 `agree=false` 且 `correct_freq` 填你从片段得到的频率（实在看不出填 unknown）。

片段：
---
{chunk}
---
"""


def normalize(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", str(s)) if not c.isspace())


def build_chunk(text: str, window: int = 130, limit: int = 5000) -> str:
    out, seen = [], set()
    for m in RE_ANCHOR.finditer(text):
        a, b = max(0, m.start() - window), min(len(text), m.end() + window)
        seg = text[a:b].replace("\n", " ")
        if seg[:60] in seen:
            continue
        seen.add(seg[:60])
        out.append(seg)
        if sum(map(len, out)) > limit:
            break
    return "\n- ".join(out)


def call_llm(chunk: str, freq: str, quote: str, timeout: int = 180) -> dict:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 800,
            "messages": [{"role": "user", "content": PROMPT.format(chunk=chunk, freq=freq, quote=quote)}]}
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data["choices"][0]["message"]["content"] or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def verify(row: dict) -> dict:
    out = {**row, "agree": None, "correct_freq": None, "reason": None,
           "freq_final": None, "label_final": None, "guard_final": None}
    if row.get("align_guard") != "ok" or not row.get("rebalance_freq"):
        out["guard_final"] = row.get("align_guard") or "not_landed"
        return out
    path = CORPUS / row["source"]
    if not path.exists():
        out["guard_final"] = "source_missing"
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunk = build_chunk(text)
    try:
        obj = call_llm(chunk, row["rebalance_freq"], row.get("evidence_quote") or "")
    except Exception as e:  # noqa: BLE001
        out["guard_final"] = f"verify_error:{type(e).__name__}"
        return out
    agree = str(obj.get("agree")).lower() in ("true", "1", "yes")
    correct = str(obj.get("correct_freq") or "unknown").strip().lower()
    out.update({"agree": agree, "correct_freq": correct, "reason": str(obj.get("reason") or "")[:200]})
    if agree:
        out["freq_final"] = row["rebalance_freq"]
        out["label_final"] = FREQ_TO_LABEL.get(row["rebalance_freq"])
        out["guard_final"] = "ok"
    elif correct in FREQ_TO_LABEL and correct != row["rebalance_freq"]:
        # 复核给出另一档：不自动采信，交给人工（宁空勿错）
        out["guard_final"] = "verifier_disagrees"
    else:
        out["guard_final"] = "verifier_disagrees"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="调仓频率判定第二意见复核")
    ap.add_argument("-i", "--inp", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    landed = [r for r in rows if r.get("align_guard") == "ok" and r.get("rebalance_freq")]
    print(f"输入 {len(rows)} 篇 | 待复核（已落定）{len(landed)} 篇", flush=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify, r): r for r in landed}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            mark = "✅" if r["guard_final"] == "ok" else "⚠"
            print(f"  [{i}/{len(landed)}] {mark} {r['title'][:40]:42s} {r['rebalance_freq']:7s} "
                  f"agree={r['agree']} correct={r['correct_freq']}", flush=True)
    fallback = [{**r, "freq_final": None, "label_final": None, "guard_final": r.get("align_guard") or "not_landed"}
                for r in rows if r.get("align_guard") != "ok" or not r.get("rebalance_freq")]
    allrows = results + fallback
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in allrows), encoding="utf-8")
    ok = sum(1 for r in allrows if r["guard_final"] == "ok")
    dis = sum(1 for r in allrows if r["guard_final"] == "verifier_disagrees")
    print(f"\n复核后最终落定 {ok} 篇 | 复核不通过（改 null 交人工）{dis} 篇 | 其余本来未落定")
    print(f"写出: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

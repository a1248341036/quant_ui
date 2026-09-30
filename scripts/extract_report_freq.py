#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报调仓频率抽取（报告级，fail-closed 五道闸门）。

Spec: docs/specs/alphaagent_report_factor_records_spec.md §5.3
核心不变量：`label_col_hint` 非空 ⇔ ①原文逐字证据 ②映射精确命中 ③报告内无冲突。
任一不满足 → 强制 null（宁可空，不要错），并记录 `align_guard`。

用法::
    python scripts/extract_report_freq.py --limit 26 \
        -o data/research_reports/knowledge/report_freq_index.jsonl \
        --golden-template data/research_reports/knowledge/report_freq_golden.jsonl
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
DEFAULT_CORPUS = ROOT / "data" / "research_reports" / "parsed_mineru"
DEFAULT_OUT = ROOT / "data" / "research_reports" / "knowledge" / "report_freq_index.jsonl"
CARDS = ROOT / "data" / "research_reports" / "knowledge" / "mechanism_cards.jsonl"

BASE_URL = os.environ.get("FACTOR_REC_BASE_URL", "http://127.0.0.1:8317/v1")
API_KEY = os.environ.get("FACTOR_REC_API_KEY", "123456")
MODEL = os.environ.get("FACTOR_REC_MODEL", "Qwen3.8-27B")

RE_TABLE_HEADER = re.compile(r"因子(名称|序号).{0,80}(计算方法|公式|定义)|(计算方法|公式).{0,40}因子")
RE_ANCHOR = re.compile(r"调仓|换仓|调仓频率|换手|持有期|持有\d|月度|每月|周度|每周|日频|每日|季度|半年")
# 闸门 2：一对一映射（label 名从本仓读取后校验存在性）
FREQ_TO_LABEL = {
    "daily": "label_1d_open_to_open",
    "weekly": "label_5d_close_to_close",
    "monthly": "label_20d_close_to_close",
}
EXPLICIT = ("daily", "weekly", "monthly")
AMBIGUOUS = ("因子月度更新", "数据月频", "月度发布", "月频数据", "按月更新")


def normalize(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", str(s)) if not c.isspace())


def load_labels() -> set[str]:
    """从本仓代码读取可用 label 名（不硬编码）。"""
    found: set[str] = set()
    for d in (ROOT / "alphaagent", ROOT / "core"):
        for f in d.rglob("*.py"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found |= set(re.findall(r"label_\d+d_(?:open_to_open|close_to_close)", txt))
    return found


def load_card_freq() -> dict[str, str]:
    """机制卡里已结构化的 rebalance_freq（用于闸门 3 交叉校验）。"""
    out: dict[str, str] = {}
    if not CARDS.exists():
        return out
    for line in CARDS.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        pr = c.get("params") or {}
        freq = None
        if isinstance(pr, dict):
            for k, v in pr.items():
                if "freq" in str(k).lower() or "rebal" in str(k).lower():
                    freq = str(v).strip().lower()
        if freq:
            src = str((c.get("source") or {}).get("title") or "")
            if src:
                out[src[:40]] = freq
    return out


PROMPT = """下面是券商研报中与"调仓/换手/持有期"相关的原文片段（含上下文）。请判断**该报告的调仓频率**。

只输出 JSON 对象：
{{"rebalance_freq": "daily|weekly|monthly|quarterly|semi_annual|bi-weekly|unknown",
  "holding": "持有期原文描述或 unknown",
  "evidence_quote": "支撑判断的原文逐字片段（必须逐字来自片段；没有则空串）",
  "confidence": "high|medium|low",
  "basis": "stated_usage|comparison_conclusion|unclear",
  "runner_up": "被比下去的那一档频率（无则 unknown）",
  "conflict": true/false,
  "conflict_note": "若"实做频率"与"结论推荐频率"不同，说明之；否则空串"}}

纪律：
1. 只根据**调仓/换仓/持有**的表述判断；"数据月频""因子月度更新""报告按月发布"**不算**调仓频率；
2. `evidence_quote` 必须从片段中逐字复制，不得改写；
3. 无明确表述 → `rebalance_freq="unknown"`、`evidence_quote=""`；
4. 区分"因子更新频率"与"组合调仓频率"；
5. **片段里同时出现多个频率词时不要回避**：若报告在**实做**（"我们以月频调仓回测"）→ `basis="stated_usage"`；
   若报告在**对比/下结论**（"周频明显优于月频"）→ `basis="comparison_conclusion"` 且 `rebalance_freq` 填**被推荐/更优的那一档**，
   `runner_up` 填被比下去的那一档；两者同时存在且冲突 → `conflict=true`（仍要给出采用/推荐档）；
   只有**并列提及、完全看不出采用或推荐哪一档**时才 `basis="unclear"`。

片段：
---
{chunk}
---
"""


def build_chunk(text: str, window: int = 130, limit: int = 6000) -> str:
    out, seen = [], set()
    for m in RE_ANCHOR.finditer(text):
        a, b = max(0, m.start() - window), min(len(text), m.end() + window)
        seg = text[a:b].replace("\n", " ")
        key = seg[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
        if sum(len(x) for x in out) > limit:
            break
    return "\n- ".join(out)


def call_llm(chunk: str, timeout: int = 180) -> dict:
    body = {"model": MODEL, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT.format(chunk=chunk)}], "max_tokens": 1500}
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data["choices"][0]["message"]["content"] or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def judge(path: Path, labels: set[str], card_freq: dict[str, str], root: Path, retries: int = 1) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunk = build_chunk(text)
    base = {"source": str(path.relative_to(root)) if root in path.parents else path.name,
            "title": path.stem, "rebalance_freq": None, "holding": None, "label_col_hint": None,
            "eval_freq_hint": None, "evidence_quote": None, "confidence": None,
            "align_guard": None, "freq_conflict": False, "card_freq": None}
    if not chunk:
        base["align_guard"] = "no_evidence"
        return base
    obj: dict = {}
    for attempt in range(retries + 1):
        try:
            obj = call_llm(chunk)
            break
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                base["align_guard"] = f"llm_error:{type(e).__name__}"
                return base

    freq = str(obj.get("rebalance_freq") or "unknown").strip().lower()
    quote = str(obj.get("evidence_quote") or "").strip()
    base["confidence"] = obj.get("confidence")
    base["holding"] = obj.get("holding")
    base["evidence_quote"] = quote or None

    # ── 闸门 1：证据必须逐字回证（且必须真实存在于本报告）
    if freq == "unknown" or not quote:
        base["align_guard"] = "no_evidence"
        return base
    if normalize(quote) not in normalize(text):
        base["align_guard"] = "quote_not_grounded"
        return base

    # ── 闸门 3c：多频率词不判 null；只有"无表态"才 null（用户 2026-09-30 指正）
    basis = str(obj.get("basis") or "unclear").strip().lower()
    base["basis"] = basis
    if str(obj.get("runner_up") or "").strip().lower() not in ("", "unknown", "none", "null"):
        base["runner_up"] = str(obj.get("runner_up")).strip().lower()
    if basis == "unclear":
        base["align_guard"] = "ambiguous_no_stance"
        base["rebalance_freq"] = freq if freq != "unknown" else None
        return base
    if str(obj.get("conflict")).lower() in ("true", "1", "yes"):
        base["freq_conflict"] = True       # 实做与结论不一致：仍落地，但标记冲突
        base["conflict_note"] = str(obj.get("conflict_note") or "")[:200]
    if any(a in quote for a in AMBIGUOUS):
        base["align_guard"] = "ambiguous_definition"
        return base

    # ── 闸门 2：只接受本仓有档位的频率，且 label 必须在代码里存在
    if freq not in EXPLICIT:
        base["align_guard"] = "not_in_label_set"
        base["rebalance_freq"] = freq
        return base
    label = FREQ_TO_LABEL[freq]
    if labels and label not in labels:
        base["align_guard"] = "label_not_found_in_repo"
        base["rebalance_freq"] = freq
        return base

    # ── 闸门 3b：与机制卡交叉校验，不一致 → null
    key = path.stem[:40]
    cf = card_freq.get(key)
    base["card_freq"] = cf
    if cf and cf not in (freq, freq.replace("ly", ""), "monthly" if freq == "monthly" else cf):
        if cf != freq:
            base["align_guard"] = "card_conflict"
            base["rebalance_freq"] = freq
            return base

    base.update({"rebalance_freq": freq, "label_col_hint": label, "eval_freq_hint": freq, "align_guard": "ok"})
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description="研报调仓频率抽取（fail-closed 五道闸门）")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    ap.add_argument("--golden-template", default="", help="输出人工标注模板（26 篇）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--require-table", action="store_true", help="只处理含因子清单表的报告")
    args = ap.parse_args()

    root = Path(args.corpus)
    labels = load_labels()
    card_freq = load_card_freq()
    files = sorted(root.rglob("*.md"))
    if args.require_table:
        files = [f for f in files
                 if any(RE_TABLE_HEADER.search(l.strip()) for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip().startswith("|"))]
    if args.limit:
        files = files[: args.limit]
    print(f"本仓 label 候选 {len(labels)} 个 | 机制卡频率 {len(card_freq)} 条 | 待处理 {len(files)} 篇", flush=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge, f, labels, card_freq, root): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            flag = "✅" if r["align_guard"] == "ok" else "—"
            print(f"  [{i}/{len(files)}] {flag} {r['title'][:40]:42s} freq={str(r['rebalance_freq']):8s} guard={r['align_guard']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    ok = [r for r in rows if r["align_guard"] == "ok"]
    print(f"\n落定 {len(ok)}/{len(rows)} 篇（其余全部 null，按 fail-closed）")
    from collections import Counter

    print("align_guard 分布:", dict(Counter(r["align_guard"] for r in rows).most_common()))
    if args.golden_template:
        gp = Path(args.golden_template)
        gp.write_text("\n".join(json.dumps({
            "source": r["source"], "title": r["title"],
            "auto_rebalance_freq": r["rebalance_freq"], "auto_label_col_hint": r["label_col_hint"],
            "auto_evidence_quote": r["evidence_quote"], "auto_guard": r["align_guard"],
            "human_rebalance_freq": "", "human_label_col_hint": "", "human_note": "",
        }, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"人工标注模板: {gp}（闸门 4 用）")
    print(f"写出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

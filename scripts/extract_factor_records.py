#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报因子清单抽取（P1-a）：表块切分 + 三档公式 + 本仓算子/字段硬约束。

Spec: docs/specs/alphaagent_report_factor_records_spec.md
- Fix 1：按"因子清单表"切块，一表一次 LLM 调用（不再整篇截断）
- Fix 3：formula_kind = verbatim（原文逐字，回证）/ derived（文字定义推导，附依据假设）/ null（给原因）
- 硬约束：expr_local 只能含**仓库已有算子**（alphaagent.dsl.core.operators 的大写名）
  与**本仓字段**（data_fields 目录里的 $xxx），且括号平衡；否则判 null 并记录原因。

用法::
    python scripts/extract_factor_records.py --require-table --limit 10 --workers 2 \
        -o data/research_reports/knowledge/factor_records_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CORPUS = ROOT / "data" / "research_reports" / "parsed_mineru"
DEFAULT_OUT = ROOT / "data" / "research_reports" / "knowledge" / "factor_records.jsonl"

BASE_URL = os.environ.get("FACTOR_REC_BASE_URL", "http://127.0.0.1:8317/v1")
API_KEY = os.environ.get("FACTOR_REC_API_KEY", "123456")
MODEL = os.environ.get("FACTOR_REC_MODEL", "Qwen3.8-27B")

RE_TABLE_HEADER = re.compile(r"因子(名称|序号).{0,80}(计算方法|公式|定义)|(计算方法|公式).{0,40}因子")
RE_FUNC = re.compile(r"([A-Z][A-Z0-9_]{1,})\s*\(")
RE_FIELD = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
RE_ROW_NUM = re.compile(r"^\s*\|\s*\d+\s*\|")

# 报告级频率索引（authoritative，由 extract_report_freq/verify_report_freq 产出）
FREQ_INDEX: dict[str, dict] = {}


# ── 本仓算子与字段（运行时加载，禁止硬编码）─────────────────────────────
INDEX_CACHE = ROOT / "data" / "research_reports" / "knowledge" / "operator_index.json"
_SRC_FILES = [ROOT / "alphaagent" / "dsl" / "core" / "operators.py",
              ROOT / "alphaagent" / "factor" / "mining" / "prompt" / "modules" / "data_fields.py"]


def _src_stamp() -> dict:
    return {str(f): f.stat().st_mtime for f in _SRC_FILES if f.exists()}


def load_operators() -> set[str]:
    from alphaagent.dsl.core import operators as ops

    return {n for n in dir(ops) if n.isupper() and not n.startswith("_")}


def load_fields() -> set[str]:
    from alphaagent.factor.mining.prompt.modules import data_fields as df

    text = "\n".join(str(getattr(df, n)) for n in dir(df) if isinstance(getattr(df, n), str))
    fields = set(RE_FIELD.findall(text))
    fields |= {str(c).lstrip("$") for c in getattr(df, "FF_PANEL_COLUMNS", ())}
    return fields


from scripts._factor_expr_tools import apply_alias, parse_op_tree, try_construct

def load_index(force: bool = False) -> tuple[set[str], set[str]]:
    """算子/字段集合：优先读落盘缓存（源文件 mtime 未变即复用），否则重建并写回。"""
    stamp = _src_stamp()
    if not force and INDEX_CACHE.exists():
        try:
            blob = json.loads(INDEX_CACHE.read_text(encoding="utf-8"))
            if blob.get("stamp") == stamp:
                return set(blob["operators"]), set(blob["fields"])
        except Exception:  # noqa: BLE001
            pass
    ops, flds = load_operators(), load_fields()
    INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_CACHE.write_text(json.dumps({"stamp": stamp, "operators": sorted(ops),
                                       "fields": sorted(flds)}, ensure_ascii=False), encoding="utf-8")
    return ops, flds


OPERATORS, FIELDS = load_index()


def validate_expr(expr: str) -> tuple[bool, str]:
    """校验 expr_local：仅用仓库已有算子 + 本仓字段 + 括号平衡（DSL dry-run 尽力）。"""
    if not expr or not str(expr).strip():
        return False, "empty"
    e = str(expr)
    try:
        from alphaagent.dsl.core.parser import check_parentheses_balance

        check_parentheses_balance(e)
    except Exception as exc:  # noqa: BLE001
        return False, f"parens:{type(exc).__name__}"
    bad_ops = {f for f in RE_FUNC.findall(e) if f not in OPERATORS}
    if bad_ops:
        return False, "unknown_operator:" + ",".join(sorted(bad_ops))
    bad_fields = {f for f in RE_FIELD.findall(e) if f not in FIELDS}
    if bad_fields:
        return False, "unknown_field:" + ",".join(sorted(bad_fields))
    try:
        from alphaagent.dsl.core.parser import expr as grammar

        grammar.parseString(e, parseAll=True)
    except Exception:  # noqa: BLE001
        pass  # 语法器仅尽力；结构性校验已通过
    return True, "ok"


def normalize(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", str(s)) if not c.isspace())


PROMPT = """你在把券商研报的"因子清单表"结构化成可执行记录。下面是研报表块的 Markdown（含表标题）。

对**表中每个因子**输出一条记录，返回严格 JSON 数组（无解释、无代码围栏）：

{{
 "name": "因子名称（原文）",
 "category": "因子类别（原文，可 null）",
 "direction": "从小到大|从大到小|null",
 "formula_kind": "verbatim|derived|null",
 "expr_raw": "verbatim: 原文逐字公式；derived: null",
 "expr_local": "本仓 DSL 表达式（见硬约束）",
 "fields_local": ["本仓字段名，不含 $"],
 "derivation": "derived 时：如何从文字定义推出（一句话）",
 "assumptions": "derived 时：假设（如用日线近似、季频差分 4 期）",
 "confidence": "high|medium|low",
 "null_reason": "null 时必填：missing_field:<名>|missing_operator|data_unavailable|ambiguous_definition",
 "report_rebalance_freq": "monthly|weekly|daily|null（只按研报自述，不猜）",
 "report_holding": "研报持有期描述或 null",
 "label_col_hint": "label_1d_open_to_open|label_5d_close_to_close|label_20d_close_to_close|null",
 "eval_freq_hint": "daily|weekly|monthly|null"
}}

硬约束（违反则该条作废）：
1. **expr_local 只能使用本仓已有算子**，可用集合（必须原样使用，禁止自造/改名）：
{operators}
2. **字段只能使用本仓面板字段**（写 `$字段名`），可用集合：
{fields}
3. `expr_raw` 仅在 `formula_kind=verbatim` 时填写，且必须**逐字来自下面片段**；
4. 原文只有文字描述时：`formula_kind=derived`，必须给出 `derivation`（依据行业标准构造）+ `assumptions`，
   `expr_local` 必须能落地；确实推不出（缺字段/缺算子/数据不可得）→ `formula_kind=null` + `null_reason`；
5. 调仓频率/标签：**只按研报自述**（如"月度调仓"→monthly+label_20d_close_to_close），未说明填 null；
6. 最多 120 条；表中没有因子就返回 []。

表标题：{title}
片段：
---
{chunk}
---
"""


def build_blocks(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("|") and RE_TABLE_HEADER.search(s):
            title = ""
            for j in range(i - 1, max(-1, i - 4), -1):
                if lines[j].strip():
                    title = lines[j].strip()
                    break
            rows, j = [], i
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(lines[j].rstrip())
                j += 1
            if len(rows) >= 3:
                blocks.append({"index": len(blocks), "title": title, "rows": rows, "n_rows": sum(1 for r in rows if RE_ROW_NUM.match(r))})
            i = j
        else:
            i += 1
    return blocks


TABLE_CACHE: dict[str, list[dict]] = {}
NO_CACHE = False


def _cache_key(source: str, blk: dict) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(("\n".join(blk["rows"]) + str(blk["title"])).encode("utf-8"))
    return h.hexdigest()


def load_table_cache(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out[rec.get("key", "")] = rec.get("records") or []
    return out


def save_table_cache(path: Path, cache: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"key": k, "records": v}, ensure_ascii=False)
                              for k, v in cache.items()), encoding="utf-8")


def call_llm(chunk: str, title: str, timeout: int = 240) -> list[dict]:
    prompt = PROMPT.format(
        operators=", ".join(sorted(OPERATORS))[:6000],
        fields=", ".join(sorted(FIELDS))[:6000],
        title=title or "(无标题)",
        chunk=chunk[:14000],
    )
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 8000}
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
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return []
    return [x for x in arr if isinstance(x, dict)]


def meta_of(path: Path, root: Path) -> dict:
    stem = path.stem
    m = re.search(r"_(.+?证券)", stem)
    return {"source": str(path.relative_to(root)), "date": stem[:10].replace("_", "-"),
            "org": (m.group(1) if m else ""), "title": stem}


def parse_one(path: Path, root: Path, retries: int = 2) -> tuple[list[dict], list[dict]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    full_norm = normalize("\n".join(lines))
    blocks = build_blocks(lines)
    recs: list[dict] = []
    stats: list[dict] = []
    for blk in blocks:
        arr: list[dict] = []
        _ck = _cache_key(str(path.relative_to(root)), blk)
        if _ck in TABLE_CACHE and not NO_CACHE:
            arr = TABLE_CACHE[_ck]
        else:
            for attempt in range(retries + 1):
                try:
                    arr = call_llm("\n".join(blk["rows"]), blk["title"])
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt >= retries:
                        print(f"  [ERR] {path.name[:40]} 表{blk['index']}: {type(e).__name__}", flush=True)
        n_src = max(1, blk["n_rows"])
        stats.append({"source": str(path.relative_to(root)), "table_index": blk["index"],
                      "n_src_rows": blk["n_rows"], "n_extracted": len(arr)})
        if not NO_CACHE:
            TABLE_CACHE[_ck] = arr
        for it in arr:
            kind = str(it.get("formula_kind") or "").strip().lower()
            raw = it.get("expr_raw")
            grounded = bool(raw) and normalize(raw) in full_norm
            expr_local = (it.get("expr_local") or "").strip() or None
            ok, why = validate_expr(expr_local) if expr_local else (False, "no_expr_local")
            # (2)(3) 修复链：别名改写 → 构造模板兜底；任一成功即视为可执行
            repair_note = None
            _TREE = None
            if not ok and expr_local:
                fixed, notes = apply_alias(expr_local)
                if notes:
                    ok2, _ = validate_expr(fixed)
                    if ok2:
                        expr_local, ok, why, repair_note = fixed, True, "ok", "alias:" + ",".join(notes)
            if not ok:
                tmpl, kw = try_construct(str(it.get("name") or ""),
                                         str(it.get("expr_raw") or "") + " " + str(it.get("derivation") or ""))
                if tmpl:
                    ok3, _ = validate_expr(tmpl)
                    if ok3:
                        expr_local, ok, why, repair_note = tmpl, True, "ok", "construct:" + str(kw)
            if kind == "verbatim" and not grounded:
                kind, raw = "derived", None  # 未回证 → 降级为推导，不冒充原文
            if kind == "null" or not expr_local or not ok:
                kind = "null" if not ok else kind
            rec = {
                **meta_of(path, root),
                "table_index": blk["index"], "table_title": blk["title"][:120],
                "name": (it.get("name") or "")[:120], "category": it.get("category"),
                "direction": it.get("direction"),
                "formula_kind": kind,
                "expr_raw": raw if (kind == "verbatim" and grounded) else None,
                "expr_local": expr_local if ok else None,
                "fields_local": it.get("fields_local") or [],
                "derivation": it.get("derivation"), "assumptions": it.get("assumptions"),
                "confidence": it.get("confidence"),
                "executable": bool(ok),
                "validate": why,
                "repair": repair_note,
                "ops_used": list(_TREE[1]) if _TREE else [],
                "op_tree": _TREE[0] if _TREE else None,
                "null_reason": (it.get("null_reason") or (None if ok else why)) if not ok else None,
                "grounded": grounded,
                "report_rebalance_freq": it.get("report_rebalance_freq"),
                "report_holding": it.get("report_holding"),
                "label_col_hint": it.get("label_col_hint"),
                "eval_freq_hint": it.get("eval_freq_hint"),
            }
            # ── 频率/标签：以报告级索引为准（fail-closed，缺则 null）──
            _fr = FREQ_INDEX.get(rec["source"]) or {}
            if _fr.get("label_final"):
                rec["report_rebalance_freq"] = _fr.get("freq_final")
                rec["label_col_hint"] = _fr.get("label_final")
                rec["eval_freq_hint"] = _fr.get("freq_final")
                rec["freq_source"] = "report_index"
                rec["align_guard"] = "ok"
            else:
                rec["report_rebalance_freq"] = None
                rec["label_col_hint"] = None
                rec["eval_freq_hint"] = None
                rec["freq_source"] = "none"
                rec["align_guard"] = _fr.get("guard_final") or "no_report_freq"
            recs.append(rec)
    return recs, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="研报因子清单抽取 P1-a（表块 + 三档公式 + 本仓算子/字段硬约束）")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    ap.add_argument("--stats", default="", help="表级统计输出 jsonl（覆盖率用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--require-table", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="忽略表级抽取缓存，强制重新调用 LLM")
    ap.add_argument("--freq-index", default=str(ROOT / "data" / "research_reports" / "knowledge" / "report_freq_verified.jsonl"),
                    help="报告级频率索引 jsonl（fail-closed：缺失即 label=null）")
    args = ap.parse_args()

    global FREQ_INDEX, TABLE_CACHE, NO_CACHE
    NO_CACHE = bool(args.no_cache)
    _tc = ROOT / "data" / "research_reports" / "knowledge" / "factor_records_table_cache.jsonl"
    TABLE_CACHE = {} if NO_CACHE else load_table_cache(_tc)
    print(f"表级缓存: {len(TABLE_CACHE)} 个表块{'（已禁用）' if NO_CACHE else ''}", flush=True)
    _fp = Path(args.freq_index)
    if _fp.exists():
        FREQ_INDEX = {}
        for _l in _fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not _l.strip():
                continue
            try:
                _r = json.loads(_l)
            except Exception:  # noqa: BLE001
                continue
            FREQ_INDEX[_r.get("source", "")] = _r
        print(f"频率索引: {len(FREQ_INDEX)} 篇（落定 {sum(1 for v in FREQ_INDEX.values() if v.get('label_final'))} 篇）", flush=True)
    else:
        print(f"频率索引不存在（{_fp}）→ 全部 label_col_hint=null", flush=True)

    root, out = Path(args.corpus), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.md"))
    if args.only:
        files = [f for f in files if args.only in str(f)]
    if args.require_table:
        files = [f for f in files if build_blocks(f.read_text(encoding="utf-8", errors="ignore").splitlines())]
    if args.limit:
        files = files[: args.limit]
    print(f"算子 {len(OPERATORS)} 个 / 字段 {len(FIELDS)} 个 | 待处理 {len(files)} 篇 | model={MODEL}", flush=True)

    all_stats: list[dict] = []
    with out.open("w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_one, f, root): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            recs, stats = fut.result()
            all_stats += stats
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            exe = sum(1 for r in recs if r["executable"])
            print(f"  [{i}/{len(files)}] {futs[fut].name[:44]} → {len(recs)} 条（可执行 {exe}）", flush=True)

    if args.stats:
        Path(args.stats).write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in all_stats), encoding="utf-8")

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    n, exe = len(rows), sum(1 for r in rows if r["executable"])
    grounded = sum(1 for r in rows if r["grounded"])
    src_rows = sum(s["n_src_rows"] for s in all_stats)
    kinds = Counter(r["formula_kind"] for r in rows)
    print(f"\n记录 {n} 条 | 可执行 {exe} ({exe/max(1,n):.0%}) | 回证 {grounded} ({grounded/max(1,n):.0%})")
    print(f"表行覆盖: 抽出 {n} / 源表行 {src_rows} = {n/max(1,src_rows):.0%}（门槛 ≥80%）")
    print(f"formula_kind: {dict(kinds)}")
    print("null_reason 分布:", dict(Counter((r.get('null_reason') or '').split(':')[0] for r in rows if r.get('null_reason')).most_common(6)))
    if not NO_CACHE:
        save_table_cache(_tc, TABLE_CACHE)
        print(f"表级缓存已更新: {_tc}（{len(TABLE_CACHE)} 个表块）", flush=True)
    print(f"写出: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

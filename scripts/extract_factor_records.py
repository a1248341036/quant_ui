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

# 因子清单表识别：由"窄表头"放宽为"因子列 + 公式列"双条件（2026-10-01）。
# 旧式 `因子(名称|序号)…(计算方法|公式|定义)` 全语料只命中 26/1900 篇；实测真公式表 186 篇，
# 另有 233 篇公式只出现在正文 —— 漏检原因是 `计算方式`/`因子简称`/`因子说明`/`因子|定义` 等真实写法。
RE_TABLE_HEADER = re.compile(
    r"(因子|指标|变量|符号|释义)[^\n|]{0,24}[|｜][^\n]*?"
    r"(计算公式|计算方式|计算方法|构建方式|构建方法|公式|定义|含义|说明|解释|释义|参数)"
    r"|(计算公式|计算方式|计算方法|构建方式|构建方法|公式)[^\n|]{0,40}"
    r"(因子|指标|变量|符号)"
)
# 强公式列：命中即判为公式表，无需再排除业绩表
RE_STRONG_FORMULA = re.compile(r"计算公式|计算方式|计算方法|构建方式|构建方法|公式")
# 业绩列：只有弱公式列（定义/含义/说明/解释/释义/参数）时，≥2 个业绩标记判为业绩周报表（无公式可抽）
RE_PERF_COL = re.compile(r"方向|最近一周|最近一月|今年以来|年化|趋势|收益率|超额|排名|胜率|多空|回测|涨跌幅")
# 正文公式段锚点：小节标题含"因子构建/指标计算…"，或直引"公式："
RE_PROSE_SECTION = re.compile(r"(因子|指标|变量)[^\n]{0,12}(构建|计算|定义|构造|度量)")
RE_PROSE_HEADING = re.compile(r"^#{1,6}\s*\S+")
# 公式行强特征：LaTeX 命令 / 上下标 / 希腊·数学符号 / 本仓算子式
RE_FORMULA_LINE = re.compile(
    r"\\[a-zA-Z]{2,}|_\{|\^\{|[\u0370-\u03ff\u2200-\u22ff]"
    r"|(?:TS|CS|DIVERGENCE|GATED|RANK|STD|MEAN|CORR)_[A-Z_]*\s*\("
)
# 无 LaTeX 时的弱特征：等号 + ≥2 个变量/算子 token（用于 `EP = 1 / PE` 这类纯文本公式）
RE_EQ = re.compile(r"[=＝]")
RE_MATH_TOKEN = re.compile(r"[A-Za-z_]{2,}|\d+\.\d+")
RE_NOISE_LINE = re.compile(r"\.{4,}|图\s*\d+|图表\s*\d+|资料来源|目\s*录|^\s*[-*]\s*图")


def _is_formula_line(ln: str) -> bool:
    """公式行判定：强特征直接算；否则要求 等号 + ≥2 个变量 token，并排除图注/目录行。"""
    s = ln.strip()
    if len(s) < 8:
        return False
    if RE_FORMULA_LINE.search(s):
        return True
    if not RE_EQ.search(s) or RE_NOISE_LINE.search(s):
        return False
    return len(RE_MATH_TOKEN.findall(s)) >= 2
RE_FUNC = re.compile(r"([A-Z][A-Z0-9_]{1,})\s*\(")
RE_FIELD = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
RE_ROW_NUM = re.compile(r"^\s*\|\s*\d+\s*\|")


def is_formula_header(row: str) -> bool:
    """因子清单表表头判定：因子列 + 公式列，且排除只有业绩列的因子周报表。"""
    if not RE_TABLE_HEADER.search(row):
        return False
    if RE_STRONG_FORMULA.search(row):
        return True
    # 弱公式列（定义/含义/说明/解释/释义/参数）：同行出现 ≥2 个业绩标记即判为业绩周报表
    # （按出现次数而非"种类"计数：`多头超额` 在多列重复正是业绩表特征）
    return len(RE_PERF_COL.findall(row)) < 2

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


from scripts._factor_expr_tools import (
    apply_alias,
    infer_focus_facets,
    op_families,
    parse_op_tree,
    try_construct,
)

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


# 块类型 → 喂给 LLM 的片段说明（prompt 用）
_KIND_DESC = {
    "table": "因子清单表，含表标题",
    "prose": "正文公式段落（非表格，含因子定义/计算方法）",
    "formula": "正文公式片段（LaTeX/算子表达式；可能缺因子名，需结合上下文命名）",
}


PROMPT = """你在把券商研报的"因子清单"结构化成可执行记录。下面是研报片段（{kind_desc}）。

对**片段中每个因子**输出一条记录，返回严格 JSON 数组（无解释、无代码围栏）：

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
6. 最多 120 条；片段中没有因子就返回 []。

片段标题：{title}
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
        if s.startswith("|") and is_formula_header(s):
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


def build_prose_blocks(lines: list[str], max_blocks: int = 2) -> list[dict]:
    """正文公式段兜底：小节标题含"因子构建/指标计算…"或直引"公式："时，取其下含公式特征的行成块。

    覆盖公式只出现在正文、没有因子清单表的报告（全语料实测约 233 篇）。
    每篇最多 ``max_blocks`` 段，避免单篇几十节"计算方法"把 LLM 预算打满。
    """
    blocks: list[dict] = []
    i = 0
    while i < len(lines) and len(blocks) < max_blocks:
        s = lines[i].strip()
        is_anchor = bool(RE_PROSE_SECTION.search(s)) and (s.startswith("#") or len(s) <= 40)
        is_direct = bool(re.match(r"^公式[：:]", s))
        if not (is_anchor or is_direct):
            i += 1
            continue
        j = i + 1
        body: list[str] = []
        while j < len(lines) and len(body) < 24:
            t = lines[j].strip()
            if RE_PROSE_HEADING.match(t) or (t.startswith("|") and is_formula_header(t)):
                break
            if t:
                body.append(t)
            j += 1
        hits = [t for t in body if _is_formula_line(t)]
        if hits:
            blocks.append({"index": 0, "kind": "prose", "title": s[:120],
                           "rows": body, "n_rows": len(hits)})
        i = max(j, i + 1)
    return blocks


def build_formula_blocks(lines: list[str], max_blocks: int = 1, gap: int = 2,
                         min_cluster: int = 3) -> list[dict]:
    """全篇公式行聚类兜底：把散落在正文（不限于"因子构建"小节）的公式行成块。

    实测：公式常成片出现在小节标题**之外**（MinerU 输出的 LaTeX 段，如
    `OCVP_{t} = \\frac{1}{d}\\sum ...`），只按标题锚点会漏掉它们。
    按"公式行间隔 ≤ gap"聚类，只取成片（≥min_cluster 行）的簇，默认每篇最多 1 段
    （单篇多段会把 LLM 预算打满，实测全语料块数会从 ~600 涨到 ~2000）。
    """
    idxs = [i for i, ln in enumerate(lines) if _is_formula_line(ln)]
    if not idxs:
        return []
    groups: list[list[int]] = []
    cur = [idxs[0]]
    for i in idxs[1:]:
        if i - cur[-1] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    # 只要成片公式（≥min_cluster 行）：单行/双行弱公式噪声太大，交给 prose 通道即可
    groups = [g for g in groups if len(g) >= min_cluster]
    groups.sort(key=len, reverse=True)

    blocks: list[dict] = []
    for g in groups[:max_blocks]:
        lo, hi = max(0, g[0] - 2), min(len(lines), g[-1] + 3)
        body = [lines[k].strip() for k in range(lo, hi) if lines[k].strip()]
        title = ""
        for k in range(g[0] - 1, max(-1, g[0] - 30), -1):
            t = lines[k].strip()
            if RE_PROSE_HEADING.match(t):
                title = t
                break
        blocks.append({"index": 0, "kind": "formula", "title": title[:120],
                       "rows": body, "n_rows": len(g)})
    return blocks


def build_all_blocks(lines: list[str]) -> list[dict]:
    """表块 + 正文公式段 + 公式行聚类，统一编号、按内容去重。

    这是 parse_one / --require-table 的唯一入口。
    """
    blocks = build_blocks(lines)
    seen = {normalize("".join(b["rows"])) for b in blocks}
    # 每篇最多补 1 段散文 + 1 段公式簇；与已选块互为子串即视为重复（取先到者）
    for b in build_prose_blocks(lines, max_blocks=1) + build_formula_blocks(lines):
        key = normalize("".join(b["rows"]))
        if not key or any(key in s or s in key for s in seen):
            continue
        seen.add(key)
        blocks.append({**b, "index": len(blocks)})
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


def call_llm(chunk: str, title: str, kind_desc: str = "因子清单表，含表标题", timeout: int = 240) -> list[dict]:
    prompt = PROMPT.format(
        operators=", ".join(sorted(OPERATORS))[:6000],
        fields=", ".join(sorted(FIELDS))[:6000],
        title=title or "(无标题)",
        chunk=chunk[:14000],
        kind_desc=kind_desc,
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
    blocks = build_all_blocks(lines)
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
                    arr = call_llm("\n".join(blk["rows"]), blk["title"],
                                   kind_desc=_KIND_DESC.get(blk.get("kind", "table"),
                                                            _KIND_DESC["table"]))
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
            # 算子树：在**修复链之后**用最终 expr_local 解析（7b18037 引入 _TREE 占位后
            # 漏了这一步，导致新抽记录的 ops_used/op_tree 恒为空）。
            if ok and expr_local:
                _TREE = parse_op_tree(expr_local)
            if kind == "verbatim" and not grounded:
                kind, raw = "derived", None  # 未回证 → 降级为推导，不冒充原文
            if kind == "null" or not expr_local or not ok:
                kind = "null" if not ok else kind
            rec = {
                **meta_of(path, root),
                "table_index": blk["index"], "table_title": blk["title"][:120],
                "block_kind": blk.get("kind", "table"),
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
                "ops_used": list(dict.fromkeys(_TREE[1])) if _TREE else [],
                "op_tree": _TREE[0] if _TREE else None,
                "op_families": op_families(_TREE[1]) if _TREE else [],
                "focus_facets": infer_focus_facets(expr_local, it.get("fields_local") or []),
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


def enrich_records(rows: list[dict]) -> list[dict]:
    """给**已有**记录补齐两个派生字段（纯函数，不调 LLM、不改其它字段）。

    - ``op_families``：由 ``ops_used`` 归族（族名取自 operator_tree.json）；
    - ``focus_facets``：由 ``expr_local``/``fields_local`` 推断数据面（复用 FACET_DEFS 命名）。
    顺带按最终 ``expr_local`` 重算 ``ops_used``/``op_tree``（修 7b18037 引入的空树回归）。
    """
    out: list[dict] = []
    for rec in rows:
        rec = dict(rec)
        expr = rec.get("expr_local")
        if rec.get("executable") and expr and validate_expr(str(expr))[0]:
            tree, ops = parse_op_tree(str(expr))
            rec["op_tree"] = tree
            rec["ops_used"] = list(dict.fromkeys(ops))
        rec["op_families"] = op_families(rec.get("ops_used") or [])
        rec["focus_facets"] = infer_focus_facets(expr, rec.get("fields_local") or [])
        out.append(rec)
    return out


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
    ap.add_argument("--enrich", default="",
                    help="只给已有 jsonl 原地补齐 focus_facets/op_families（不调 LLM，保内容不变）")
    ap.add_argument("--freq-index", default=str(ROOT / "data" / "research_reports" / "knowledge" / "report_freq_verified.jsonl"),
                    help="报告级频率索引 jsonl（fail-closed：缺失即 label=null）")
    args = ap.parse_args()

    if args.enrich:
        src = Path(args.enrich)
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = enrich_records(rows)
        src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        exe = [r for r in rows if r.get("executable")]
        print(f"补齐 {len(rows)} 条（可执行 {len(exe)}）| focus_facets 非空 "
              f"{sum(1 for r in rows if r.get('focus_facets'))} | op_families 非空 "
              f"{sum(1 for r in rows if r.get('op_families'))} → {src}", flush=True)
        return 0

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
        files = [f for f in files if build_all_blocks(f.read_text(encoding="utf-8", errors="ignore").splitlines())]
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

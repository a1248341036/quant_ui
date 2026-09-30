#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报机制卡批量抽取（MinerU 语料 → mechanism_cards.jsonl 追加）。

用法::

    .venv\\Scripts\\python.exe scripts\\extract_report_cards.py --limit 400 --workers 2
    .venv\\Scripts\\python.exe scripts\\extract_report_cards.py --only "01_多因子与选股体系" --limit 50
    .venv\\Scripts\\python.exe scripts\\extract_report_cards.py --selftest

红线（不可违背，见脚本内 _ground_* 实现）：
  * ``formula_text`` 只有在该片段能逐字回证于语料原文时才写入，否则 ``None``；
  * ``evidence`` 的 ic/icir/sample/decile_shape 必须以原文引文回证，否则留空，绝不臆造数值；
  * ``dsl_hint`` 由 LLM 生成 → ``extraction.generated_hint=True``，并据此下调 confidence。

工程约束：
  * 纯线程 + requests，不启动任何子进程 → Windows 下不会弹黑框；
  * 断点续跑：输出 JSONL 里已出现的 ``source.path`` 直接跳过；
  * **跳过/失败台账** ``<out>.skipped.jsonl``（如 ``mechanism_cards.skipped.jsonl``）：
    LLM 判定"未描述可复现机制"的 skip 会落盘并参与断点续跑（否则每次重跑都要把几百篇
    重新喂一遍 LLM）；fail 只记账、不参与跳过，保持可重试。想重评跳过项用
    ``--rescan-skipped``；
  * JSON 解析容忍非法转义：模型输出的 LaTeX 裸反斜杠（``\\;`` ``\\sigma``）不是合法
    JSON 转义，``repair_json_escapes`` 会补成合法转义后二次解析。**注意**：``\\f``
    ``\\b`` ``\\t`` 本身是合法 JSON 转义，修复不会动它们，所以 ``\\frac`` 仍会被解析成
    换页符 + "rac" —— 这类公式片段会因回证不上而被丢弃（不臆造数值），但不会丢整张卡；
  * 只写 ``--out`` 指定的 JSONL（默认现有 mechanism_cards.jsonl，追加写）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaagent.dsl.core import operators as _ops  # noqa: E402
from alphaagent.dsl.core.parser import parse_expression  # noqa: E402
from alphaagent.factor.facets import FACET_DEFS  # noqa: E402

VALID_FACETS = tuple(name for name, _ in FACET_DEFS)
VALID_FACET_SET = frozenset(VALID_FACETS)
OPERATOR_NAMES = frozenset(n for n in dir(_ops) if n.isupper() and not n.startswith("_"))

DEFAULT_CORPUS = ROOT / "data" / "research_reports" / "parsed_mineru"
DEFAULT_OUT = ROOT / "data" / "research_reports" / "knowledge" / "mechanism_cards.jsonl"

PAGE_RE = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+(\d+)\s*-->")

# 面板真实列名白名单（前缀族 + 具体列）。LLM 给出的 fields 不在其中一律丢弃。
_BASE_FIELDS = {
    "close", "open", "high", "low", "volume", "amount", "turnover", "turnover_rate",
    "vwap", "adj_close", "adj_open", "adj_high", "adj_low", "adj_close", "ret",
    "float_cap", "total_cap", "free_float_cap", "market_cap", "pe", "pb", "ps", "pcf",
    "funda_ocfps", "funda_roe", "funda_eps", "funda_bps", "funda_revenue", "funda_netprofit",
    "funda_gross_margin", "funda_net_margin", "funda_debt_ratio", "funda_total_assets",
    "ff_super_net", "ff_large_net", "ff_main_net", "ff_small_net", "inflow", "outflow",
    "mgn_buy", "mgn_balance", "mgn_repay", "mgn_net", "holder_num", "holder_avg",
    "inst_hold_ratio", "th_hold_ratio", "dt_net_buy", "dt_buy", "dt_sell", "bt_amount",
    "pred_yoy", "pred_np", "exp_eps", "exp_np", "ds_flag", "div_yield", "div_ratio",
}
_FIELD_PREFIXES = (
    "chip_", "crowd_", "funda_", "ff_", "mgn_", "holder_", "inst_", "th_",
    "dt_", "bt_", "pred_", "exp_", "ds_", "div_",
)


def is_valid_field(name: str) -> bool:
    n = str(name or "").strip().lstrip("$")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", n):
        return False
    if n in _BASE_FIELDS:
        return True
    return any(n.startswith(p) and len(n) > len(p) for p in _FIELD_PREFIXES)


def fields_to_facets(fields: list[str]) -> list[str]:
    """按列名前缀反推数据面（仅用于 facets 缺失时的兜底，不编造）。"""
    got: list[str] = []
    for f in fields:
        n = str(f).lstrip("$")
        hit = None
        if n in ("close", "open", "high", "low", "vwap", "adj_close", "ret", "volume", "amount",
                 "turnover", "turnover_rate", "float_cap", "total_cap", "market_cap"):
            hit = "价量面" if n not in ("volume", "amount", "turnover", "turnover_rate") else "量能面"
        elif n.startswith("chip_"):
            hit = "筹码面"
        elif n.startswith("crowd_"):
            hit = "拥挤面"
        elif n.startswith("funda_") or n in ("pe", "pb", "ps", "pcf"):
            hit = "基本面"
        elif n.startswith("holder_"):
            hit = "股东面"
        elif n.startswith("inst_"):
            hit = "机构面"
        elif n.startswith("th_"):
            hit = "股东集中面"
        elif n.startswith("ff_") or n in ("inflow", "outflow"):
            hit = "资金面"
        elif n.startswith("mgn_"):
            hit = "两融面"
        elif n.startswith("dt_") or n.startswith("bt_"):
            hit = "事件面"
        elif n.startswith("pred_") or n.startswith("exp_"):
            hit = "业绩面"
        elif n.startswith("ds_"):
            hit = "披露面"
        elif n.startswith("div_"):
            hit = "分红面"
        if hit and hit not in got:
            got.append(hit)
    return got


# ── 原文回证（grounding）工具 ─────────────────────────────────────

def norm_with_index(text: str) -> tuple[str, list[int]]:
    """NFKC + 去空白 归一化，同时保留 norm 位置 → 原文字符下标 的映射。"""
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        n = unicodedata.normalize("NFKC", ch)
        for c in n:
            if c.isspace():
                continue
            chars.append(c)
            idx.append(i)
    return "".join(chars), idx


def normalize(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", str(s)) if not c.isspace())


def ground_quote(full_norm: str, full_idx: list[int], full_text: str, quote: str,
                 min_len: int = 4) -> str | None:
    """引文回证：命中则返回原文中的逐字片段，未命中返回 None。"""
    q = normalize(quote)
    if len(q) < min_len:
        return None
    pos = full_norm.find(q)
    if pos < 0:
        return None
    start = full_idx[pos]
    end = full_idx[pos + len(q) - 1] + 1
    return full_text[start:end]


def quote_numbers(quote: str) -> list[float]:
    txt = unicodedata.normalize("NFKC", str(quote)).replace(",", "")
    out = []
    for m in re.findall(r"-?\d+(?:\.\d+)?", txt):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def quote_has_value(quote: str, value: float) -> bool:
    """数值回证：quote 里出现该数值即算命中。

    MinerU 的表格常把相邻数字粘在一起（``0.0870.349``），单纯分词会漏掉后一个数，
    因此额外做一次"字面量 + 后面不接数字"的子串匹配。
    """
    txt = unicodedata.normalize("NFKC", str(quote)).replace(",", "")
    for n in quote_numbers(txt):
        if abs(n - value) < 1e-9:
            return True
        if abs(value) > 1e-12 and abs(n - value) / abs(value) < 1e-3:
            return True
    for cand in (f"{float(value):g}", f"{float(value):.4f}"):
        cand = cand.rstrip("0").rstrip(".") if "." in cand else cand
        for form in (cand, cand.lstrip("-")):
            if form and re.search(re.escape(form) + r"(?!\d)", txt):
                return True
    return False


def quote_has_digits(quote: str, value: str) -> bool:
    want = re.findall(r"\d+", unicodedata.normalize("NFKC", str(value)))
    if not want:
        return True
    have = set(re.findall(r"\d+", unicodedata.normalize("NFKC", str(quote))))
    return all(w in have for w in want)


def norm_decile_shape(value: str) -> str | None:
    s = str(value or "")
    low = s.lower()
    if "倒u" in low or "倒 u" in low:
        return "倒U型"
    if "u型" in low or "u 型" in low:
        return "U型"
    if "分段" in s:
        return "分段单调"
    if "弱" in s:
        return "弱单调"
    if "非单调" in s or "不单调" in s or "无明显" in s or "无规律" in s:
        return "非单调"
    if "单调" in s:
        if "负" in s:
            return "单调负向"
        if "正" in s:
            return "单调正向"
        return "单调"
    return None


# ── DSL 校验 ─────────────────────────────────────────────────────

def dsl_syntax_ok(expr: str) -> bool:
    expr = str(expr or "").strip()
    if not expr or len(expr) > 500:
        return False
    try:
        parse_expression(expr)
        return True
    except Exception:
        pass
    # pyparsing 深递归等异常时退化为轻量结构校验（更严格的字符白名单）
    if expr.count("(") != expr.count(")") or re.search(r"[\[\]{}`;#%\\@]", expr):
        return False
    if re.search(r"[^A-Za-z0-9_$(),.+\-*/<>=!&|?:'\s]", expr):
        return False
    stripped = re.sub(r"'(?:[^']*)'", "", expr)
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped):
        if m.group(1).upper() not in OPERATOR_NAMES:
            return False
    return bool(re.search(r"\([^()]*\)", stripped))


def validate_dsl_hint(expr: str) -> tuple[str | None, list[str]]:
    """返回 (可用表达式, 其中引用的数据列)。列名必须 $ 前缀且在白名单内。"""
    raw = str(expr or "").strip()
    if not raw:
        return None, []
    cols = re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", raw)
    bare = set(re.findall(r"(?<![\w$])([A-Za-z][A-Za-z0-9_]{1,})(?![\w(])", raw))
    for c in bare:
        if c not in cols and not is_valid_field(c):
            return None, []
    if not cols:
        return None, []
    for c in cols:
        if not is_valid_field(c):
            return None, []
    if not dsl_syntax_ok(raw):
        return None, []
    return raw, sorted(set(cols))


# 退化占位提示：只有裸价格均值/标准差的 z-score，不表达任何报告机制
DEGENERIC_HINT_RE = re.compile(
    r"^\s*CS_ZSCORE\(\s*TS_(?:MEAN|STD)\(\s*\$close\s*,\s*\d+\s*\)\s*\)\s*$"
)


REFINE_PROMPT = """下面是一张研报机制卡，它的 DSL 种子表达式是无效占位（只对收盘价做均值/标准差的 z-score），不能表达报告的机制。

机制的三个问题：
- 谁在被错误定价（who_wrong）：{who}
- 为什么长期不被抹平（why_persists）：{why}
- 用什么可观测量刻画（observable）：{observable}

机制卡声明的数据列：{fields}
当前占位表达式：{old}

请给出一个真正刻画该 observable 的 DSL 表达式。硬性要求：
1. 数据列必须写成 $列名 形式（如 $close / $turnover_rate），只能用允许列：{allowed}
2. 如果该 observable 依赖逐笔明细、文本、图/网络、基金或期货特有数据，用现有列无法表达，就如实输出 null，不要硬凑。
3. 只输出 JSON：{{"dsl_hint": "...", "fields": ["..."]}} 或 {{"dsl_hint": null, "reason": "..."}}
"""


def refine_hints(out_path: Path, corpus: Path, cfg: dict[str, Any], workers: int) -> int:
    """对退化占位 dsl_hint 做定向重问；仍退化则置 null 并标注（不编造）。"""
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cards = [json.loads(ln) for ln in lines]
    targets = [c for c in cards if c.get("dsl_hint") and DEGENERIC_HINT_RE.match(str(c["dsl_hint"]))]
    print(f"[refine] 退化占位 dsl_hint {len(targets)} 张，开始定向重问")
    if not targets:
        return 0

    def work(card: dict[str, Any]) -> tuple[str, Any]:
        rel = card["source"]["path"]
        path = corpus / rel
        if not path.is_file():
            return card["card_id"], None
        text = path.read_text(encoding="utf-8", errors="ignore")
        pages = select_pages(split_pages(text), 12000)
        prompt = REFINE_PROMPT.format(
            who=card["mechanism"]["who_wrong"], why=card["mechanism"]["why_persists"],
            observable=card["mechanism"]["observable"], fields=card.get("fields") or [],
            old=card["dsl_hint"], allowed=ALLOWED_FIELDS_TEXT,
        ) + "\n【报告正文节选】\n" + "".join(f"\n<<<PAGE {p}>>>\n{t.strip()}\n" for p, t in pages)
        obj, err = call_llm(prompt, cfg, attempts=2)
        return card["card_id"], (obj, err)

    got: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for cid, res in pool.map(work, targets):
            got[cid] = res

    changed = 0
    for card in cards:
        if not (card.get("dsl_hint") and DEGENERIC_HINT_RE.match(str(card["dsl_hint"]))):
            continue
        res = got.get(card["card_id"])
        obj = res[0] if isinstance(res, tuple) else None
        hint = None
        if isinstance(obj, dict):
            hint, _ = validate_dsl_hint(obj.get("dsl_hint") or "")
            if hint and DEGENERIC_HINT_RE.match(hint):
                hint = None
            if hint:
                cols = [c for c in (obj.get("fields") or []) if is_valid_field(c)]
                merged = list(card.get("fields") or [])
                for c in cols:
                    if c.lstrip("$") not in merged:
                        merged.append(c.lstrip("$"))
                card["fields"] = merged
        card["dsl_hint"] = hint
        card["extraction"]["generated_hint"] = bool(hint)
        card["extraction"]["hint_refined"] = True
        if not hint:
            card["extraction"]["hint_refine_failed"] = True
        changed += 1

    out_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cards) + "\n",
                        encoding="utf-8")
    kept = sum(1 for c in cards if c.get("dsl_hint"))
    print(f"[refine] 处理 {changed} 张：重问成功保留 {changed - sum(1 for c in cards if c['extraction'].get('hint_refine_failed'))} 张，"
          f"置 null {sum(1 for c in cards if c['extraction'].get('hint_refine_failed'))} 张；当前有 dsl_hint 的卡片共 {kept} 张")
    return 0


# ── 语料读取与页面选择 ────────────────────────────────────────────

def split_pages(text: str) -> list[tuple[int, str]]:
    marks = list(PAGE_RE.finditer(text))
    if not marks:
        return [(1, text)]
    pages: list[tuple[int, str]] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages.append((int(m.group(1)), text[start:end]))
    return pages


_PAGE_KEYWORDS = (
    "因子", "IC", "ICIR", "分组", "回测", "构建", "公式", "多空", "净值", "Rank",
    "计算", "定义", "参数", "窗口", "权重", "收益", "Alpha", "alpha", "标准化",
)
_BOILERPLATE = ("请务必阅读", "免责声明", "风险提示", "信息披露", "特别声明", "法律声明")


def page_score(page_text: str) -> float:
    score = float(sum(page_text.count(k) for k in _PAGE_KEYWORDS))
    if any(b in page_text for b in _BOILERPLATE):
        score *= 0.3
    return score


def select_pages(pages: list[tuple[int, str]], budget: int) -> list[tuple[int, str]]:
    scored = sorted(pages, key=lambda p: -page_score(p[1]))
    chosen: list[tuple[int, str]] = []
    used = 0
    for pno, ptext in scored:
        if used + len(ptext) > budget and chosen:
            continue
        chosen.append((pno, ptext))
        used += len(ptext)
        if used >= budget:
            break
    chosen.sort(key=lambda p: p[0])
    return chosen


def parse_meta(path: Path) -> dict[str, str]:
    stem = path.stem
    date = ""
    m = re.match(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", stem)
    if m:
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    tail = stem[len(date):].lstrip("_-") if date else stem
    orgs = ("海通证券", "华泰证券", "东方证券", "国泰君安", "申万宏源", "国信证券", "中信证券",
            "招商证券", "广发证券", "兴业证券", "天风证券", "长江证券", "中金公司", "光大证券",
            "方正证券", "东吴证券", "国金证券", "开源证券", "中泰证券", "平安证券", "民生证券",
            "西部证券", "浙商证券", "信达证券", "中银国际期货", "国盛证券", "国投证券", "华安证券")
    org = ""
    hits = [o for o in orgs if o in stem]
    if hits:
        org = max(hits, key=len)
    else:
        # 兜底：任意 "XX证券 / XX期货 / XX基金" 形式
        m2 = re.search(r"([\u4e00-\u9fa5]{2,6}(?:证券|期货|基金|研究院))", stem)
        if m2:
            org = m2.group(1)
    title = tail
    for token in (org, date.replace("-", ""), "金融工程", "金融工程研究", "证券研究报告", "_"):
        if token:
            title = title.replace(token, " ")
    title = re.sub(r"[_\s]+", " ", title).strip(" _-")
    return {"org": org, "date": date, "title": title or stem}


# ── LLM 抽取 ─────────────────────────────────────────────────────

ALLOWED_FIELDS_TEXT = (
    "close, open, high, low, volume, amount, turnover, turnover_rate, vwap, adj_close, ret, "
    "float_cap, total_cap, market_cap, pe, pb, ps, pcf, funda_ocfps, funda_roe, funda_eps, "
    "funda_bps, funda_revenue, funda_netprofit, funda_gross_margin, funda_net_margin, "
    "funda_debt_ratio, ff_super_net, ff_large_net, ff_main_net, ff_small_net, inflow, outflow, "
    "mgn_buy, mgn_balance, mgn_repay, mgn_net, holder_num, holder_avg, inst_hold_ratio, "
    "th_hold_ratio, dt_net_buy, dt_buy, dt_sell, bt_amount, pred_yoy, pred_np, exp_eps, exp_np, "
    "ds_flag, div_yield, div_ratio, chip_*(筹码), crowd_*(拥挤)"
)

PROMPT_HEAD = """你是 A 股量化研究员，负责把一篇卖方金融工程研报蒸馏成一张"研报机制卡"（用于因子复现的锚点）。

只输出一个 JSON 对象。不要 markdown 代码块，不要任何解释文字。

JSON 结构：
{
  "has_mechanism": true,
  "mechanism": {"who_wrong": "谁在被系统性错误定价（错边方）", "why_persists": "为什么该错误长期不被套利抹平", "observable": "用什么可观测量刻画该机制"},
  "fields": ["close", "volume"],
  "params": {"window": 20},
  "dsl_hint": "CS_ZSCORE(TS_MEAN($close, 20))",
  "formula": {"text": "原文中逐字出现的公式片段", "page": 5},
  "evidence": {
    "sample": {"value": "2010-2015 全A", "quote": "原文逐字片段"},
    "decile_shape": {"value": "单调负向", "quote": "原文逐字片段"},
    "ic": {"value": -0.034, "quote": "原文逐字片段"},
    "icir": {"value": -1.15, "quote": "原文逐字片段"}
  },
  "negatives": [{"claim": "该因子可能失效/被反向打脸的场景", "reason": "原因"}],
  "facets": ["价量面"],
  "mechanism_pages": [3, 5],
  "confidence": "high"
}

硬性规则（违反即作废）：
1. formula.text 必须从下面正文里【逐字符复制】，禁止改写、补全、推导、翻译；正文没有明确公式就填 null。page 写该公式所在页码。
2. evidence 里每一项的 quote 必须逐字复制正文原句；find 不到证据的项直接填 null，绝对不要猜。ic/icir 的 value 必须与 quote 中的数字完全一致。
3. fields 只能从以下列名里选（不要发明列名）：""" + ALLOWED_FIELDS_TEXT + """
4. facets 只能从这个列表里选：""" + "、".join(VALID_FACETS) + """
5. dsl_hint 中的每一个数据列都必须写成 $列名 形式（如 $close、$turnover），只能用允许列表里的列。
6. 没有明确的选股因子/机制的日报、周报、会议纪要、纯宏观报告：has_mechanism 填 false，其余字段尽量留空。
7. mechanism_pages 写你归纳机制所依据的页码（整数列表）。

"""


def build_prompt(meta: dict[str, str], pages: list[tuple[int, str]]) -> str:
    parts = [
        PROMPT_HEAD,
        f"【报告元信息】机构: {meta['org'] or '未知'} | 日期: {meta['date'] or '未知'} | 文件名标题: {meta['title']}\n",
        "【报告正文】\n",
    ]
    for pno, ptext in pages:
        parts.append(f"\n<<<PAGE {pno}>>>\n{ptext.strip()}\n")
    parts.append("\n<<<END>>>\n现在输出 JSON：")
    return "".join(parts)


# 非法反斜杠转义修复（2026-09-30）：模型在 formula.text / dsl_hint 里输出 LaTeX 时会给
# 裸反斜杠（`\;` `\sigma` `\frac`），这些不是合法 JSON 转义序列 → json.loads 直接抛错，
# 整篇被判 FAIL（实测 3 篇恒失败，把 --max-tokens 从 1500 提到 3000 复现同样失败）。
#
# 实现必须**把合法转义当整体消费**：早先版本用负向环视 `\\(?!["\\/bfnrtu])` 逐字符扫描，
# 会把已经合法的 `\\cdots` 里第二个反斜杠也当成裸反斜杠 → 补成 `\\\cdots` → 仍旧非法
# （2026-09-30 实测：`2024-04-24_…季报解析` 就是这样二次失败的）。这里的正则先尝试匹配
# 完整合法转义（`\\` `\"` `\/` `\b` `\f` `\n` `\r` `\t` `\uXXXX`），匹配不到才把单个
# `\` 补成 `\\`，因此对合法 JSON 是恒等变换。
_ESCAPE_RE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def repair_json_escapes(text: str) -> str:
    """把非法反斜杠转义补成合法转义，供 json.loads 二次尝试。"""
    return _ESCAPE_RE.sub(lambda m: m.group(0) if len(m.group(0)) > 1 else "\\\\", text)


def extract_json(text: str) -> dict[str, Any] | None:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s).strip()
    i, j = s.find("{"), s.rfind("}")
    candidates = [s]
    if i >= 0 and j > i:
        candidates.append(s[i:j + 1])
    # 每个候选先按原样解析，再按"修复非法转义"后的文本解析（见 repair_json_escapes）
    for cand in candidates:
        for attempt in (cand, repair_json_escapes(cand)):
            try:
                obj = json.loads(attempt)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def call_llm(prompt: str, cfg: dict[str, Any], attempts: int = 3) -> tuple[dict[str, Any] | None, str]:
    last = ""
    for k in range(attempts):
        try:
            resp = requests.post(
                cfg["base_url"].rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": cfg.get("max_tokens", 1500),
                    "stream": False,
                },
                timeout=cfg.get("timeout", 300),
            )
            if resp.status_code != 200:
                last = f"HTTP {resp.status_code}: {resp.text[:160]}"
            else:
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                obj = extract_json(content)
                if obj is not None:
                    return obj, ""
                last = f"JSON 解析失败: {str(content)[:160]}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if k < attempts - 1:
            time.sleep(2.0 * (k + 1))
    return None, last


# ── 卡片构建（含回证与降级）──────────────────────────────────────

def _clean_str(v: Any, limit: int) -> str:
    s = re.sub(r"\s+", " ", str(v or "")).strip()
    return s[:limit]


def build_card(obj: dict[str, Any], full_text: str, full_norm: str, full_idx: list[int],
               meta: dict[str, str], rel_path: str, card_id: str, model: str,
               page_count: int) -> tuple[dict[str, Any] | None, str]:
    mech = obj.get("mechanism") if isinstance(obj.get("mechanism"), dict) else {}
    who = _clean_str(mech.get("who_wrong"), 120)
    why = _clean_str(mech.get("why_persists"), 120)
    obs = _clean_str(mech.get("observable"), 120)
    if not (who and why and obs):
        return None, "mechanism 三问不完整"

    # 字段白名单过滤
    raw_fields = obj.get("fields") if isinstance(obj.get("fields"), list) else []
    fields = []
    for f in raw_fields:
        fn = str(f).strip()
        if is_valid_field(fn):
            fn = fn.lstrip("$")
            if fn not in fields:
                fields.append(fn)

    # 公式回证：命中才写，且写入的是原文逐字片段
    formula_text = None
    formula_page = None
    fo = obj.get("formula")
    if isinstance(fo, dict):
        hit = ground_quote(full_norm, full_idx, full_text, fo.get("text") or "", min_len=6)
        if hit and ("=" in normalize(hit) or "≈" in normalize(hit) or "∑" in normalize(hit)):
            formula_text = hit
            try:
                pg = int(fo.get("page"))
                formula_page = pg if 1 <= pg <= max(page_count, 1) else None
            except (TypeError, ValueError):
                formula_page = None

    # 证据回证
    ev_in = obj.get("evidence") if isinstance(obj.get("evidence"), dict) else {}
    evidence: dict[str, Any] = {"sample": None, "decile_shape": None, "ic": None, "icir": None}
    ev_pages: list[int] = []
    for key in ("sample", "decile_shape", "ic", "icir"):
        item = ev_in.get(key)
        if not isinstance(item, dict):
            continue
        quote = item.get("quote")
        verified = ground_quote(full_norm, full_idx, full_text, quote or "", min_len=6) if quote else None
        if not verified:
            continue
        val = item.get("value")
        if key in ("ic", "icir"):
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if not quote_has_value(verified, num):
                continue
            evidence[key] = num  # 保留原文精度，禁止四舍五入改写数值
        elif key == "sample":
            sval = _clean_str(val, 40)
            if not sval or not quote_has_digits(verified, sval):
                continue
            evidence[key] = sval
        else:
            shape = norm_decile_shape(val)
            if not shape:
                continue
            evidence[key] = shape
        try:
            pg = int(item.get("page"))
            if 1 <= pg <= max(page_count, 1):
                ev_pages.append(pg)
        except (TypeError, ValueError):
            pass

    # DSL 提示（LLM 生成）
    dsl_hint, dsl_cols = validate_dsl_hint(obj.get("dsl_hint") or "")

    # facets
    raw_facets = obj.get("facets") if isinstance(obj.get("facets"), list) else []
    facets = [f for f in (str(x).strip() for x in raw_facets) if f in VALID_FACET_SET]
    if not facets:
        facets = [f for f in fields_to_facets(fields + dsl_cols) if f in VALID_FACET_SET]
    if not facets:
        return None, "facets 无法确定"

    # params
    params: dict[str, Any] = {}
    po = obj.get("params")
    if isinstance(po, dict):
        for k, v in list(po.items())[:8]:
            kk = _clean_str(k, 24)
            if not kk:
                continue
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, (int, float)):
                params[kk] = v
            elif isinstance(v, str):
                vs = _clean_str(v, 24)
                if vs:
                    params[kk] = vs

    # negatives
    negatives = []
    no = obj.get("negatives")
    if isinstance(no, list):
        for item in no[:3]:
            if not isinstance(item, dict):
                continue
            claim = _clean_str(item.get("claim"), 160)
            reason = _clean_str(item.get("reason"), 160)
            if claim:
                negatives.append({"claim": claim, "reason": reason})

    # pages
    pages_out: list[int] = []
    mp = obj.get("mechanism_pages")
    if isinstance(mp, list):
        for p in mp[:8]:
            try:
                ip = int(p)
            except (TypeError, ValueError):
                continue
            if 1 <= ip <= max(page_count, 1) and ip not in pages_out:
                pages_out.append(ip)
    if formula_page:
        pages_out.append(formula_page)
    pages_out.extend(ev_pages)
    pages_out = sorted(set(pages_out)) or [1]

    # confidence：由回证强度决定
    verified_nums = sum(1 for k in ("ic", "icir") if evidence.get(k) is not None)
    if formula_text and verified_nums >= 1 and dsl_hint:
        confidence = "high"
    elif formula_text or verified_nums >= 1 or evidence.get("decile_shape"):
        confidence = "medium"
    else:
        confidence = "low"
    if dsl_hint and not formula_text:
        confidence = "medium" if confidence == "high" else confidence

    card = {
        "card_id": card_id,
        "source": {
            "path": rel_path,
            "org": meta["org"],
            "date": meta["date"],
            "title": meta["title"],
            "pages": pages_out,
        },
        "mechanism": {"who_wrong": who, "why_persists": why, "observable": obs},
        "fields": fields,
        "params": params,
        "dsl_hint": dsl_hint,
        "formula_text": formula_text,
        "evidence": evidence,
        "negatives": negatives,
        "facets": facets,
        "extraction": {
            "model": model,
            "confidence": confidence,
            "formula_from_image": False,
            "generated_hint": bool(dsl_hint),
            "formula_verified": bool(formula_text),
            "evidence_verified": sorted(k for k, v in evidence.items() if v is not None),
        },
    }
    return card, ""


def validate_card(card: dict[str, Any]) -> list[str]:
    """与 scripts/extract_mechanism_cards.py 同口径的 schema 校验（内联，避免改动原脚本）。"""
    errors: list[str] = []
    if not isinstance(card.get("card_id"), str) or not card.get("card_id"):
        errors.append("card_id 缺失或非字符串")
    src = card.get("source")
    if not isinstance(src, dict):
        errors.append("source 结构缺失")
    else:
        for k in ("path", "org", "title", "pages"):
            if k not in src:
                errors.append(f"source 缺少字段 {k}")
        if not isinstance(src.get("pages"), list) or not src.get("pages"):
            errors.append("source.pages 必须为非空列表")
    mech = card.get("mechanism")
    if not isinstance(mech, dict):
        errors.append("mechanism 结构缺失")
    else:
        for k in ("who_wrong", "why_persists", "observable"):
            if not mech.get(k):
                errors.append(f"mechanism 缺少三问字段 {k}")
    facets = card.get("facets")
    if not isinstance(facets, list) or not facets:
        errors.append("facets 必须为非空列表")
    else:
        for f in facets:
            if f not in VALID_FACET_SET:
                errors.append(f"facet 非法: '{f}'")
    if not isinstance(card.get("negatives"), list):
        errors.append("negatives 必须为列表")
    return errors


# ── 独立复核（对已落盘卡片做红线回证）────────────────────────────

def verify_output(out_path: Path, corpus: Path) -> int:
    """复核输出文件：schema + formula_text 逐字回证 + ic/icir 数值回证 + card_id 唯一性。"""
    total = 0
    schema_bad = 0
    id_seen: set[str] = set()
    id_dup = 0
    formula_total = 0
    formula_bad = 0
    num_total = 0
    num_bad = 0
    missing_src = 0
    cache: dict[str, tuple[str, list[int], str]] = {}

    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            card = json.loads(line)
            errs = validate_card(card)
            if errs:
                schema_bad += 1
                print(f"[schema] {card.get('card_id')}: {errs}", file=sys.stderr)
            cid = card.get("card_id")
            if cid in id_seen:
                id_dup += 1
                print(f"[dup] card_id 重复: {cid}", file=sys.stderr)
            id_seen.add(cid)

            rel = (card.get("source") or {}).get("path") or ""
            src = corpus / rel
            if not src.is_file():
                missing_src += 1
                print(f"[src] 找不到源文件: {rel}", file=sys.stderr)
                continue
            if rel not in cache:
                text = src.read_text(encoding="utf-8", errors="ignore")
                n, idx = norm_with_index(text)
                cache[rel] = (n, idx, text)
            norm, idx, text = cache[rel]

            ft = card.get("formula_text")
            if ft:
                formula_total += 1
                if ground_quote(norm, idx, text, ft, min_len=6) is None:
                    formula_bad += 1
                    print(f"[formula] 无法回证: {cid} {rel} :: {str(ft)[:60]}", file=sys.stderr)

            ev = card.get("evidence") or {}
            for key in ("ic", "icir"):
                val = ev.get(key)
                if val is None:
                    continue
                num_total += 1
                if not quote_has_value(text, float(val)):
                    num_bad += 1
                    print(f"[evidence] {key}={val} 在原文中未出现: {cid} {rel}", file=sys.stderr)

    print(f"[verify] 共 {total} 张卡 | schema 错误 {schema_bad} 张 | card_id 重复 {id_dup} 个 | "
          f"源文件缺失 {missing_src} 个")
    print(f"[verify] formula_text 非空 {formula_total} 张，其中回证失败 {formula_bad} 张")
    print(f"[verify] ic/icir 数值 {num_total} 个，其中原文未出现 {num_bad} 个")
    return 0 if (schema_bad or id_dup or formula_bad or missing_src or num_bad) == 0 else 1


def repair_meta(out_path: Path, corpus: Path) -> int:
    """按 source.path 重新推导 org/date，回填 source.org 为空的卡片（其余字段一律不动）。"""
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    fixed = 0
    out: list[str] = []
    for ln in lines:
        card = json.loads(ln)
        src = card.get("source") or {}
        if not (src.get("org") or "").strip():
            rel = str(src.get("path") or "")
            meta = parse_meta(corpus / rel)
            if meta["org"]:
                src["org"] = meta["org"]
                fixed += 1
            if not (src.get("date") or "").strip() and meta["date"]:
                src["date"] = meta["date"]
        out.append(json.dumps(card, ensure_ascii=False))
    if fixed:
        out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[repair] 卡片 {len(lines)} 张，回填 org {fixed} 张")
    return 0


# ── 主流程 ───────────────────────────────────────────────────────

def skipped_path(out_path: Path) -> Path:
    """跳过/失败台账路径：``<out>.skipped.jsonl``（与主卡片文件同目录，独立不混写）。"""
    return out_path.with_name(out_path.stem + ".skipped.jsonl")


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for ln in f if ln.strip())


def load_done(out_path: Path, skip_path: Path | None = None) -> tuple[set[str], int]:
    """已处理集合 = 主文件里的成功卡片 ∪ 台账里的 skip 判定。

    **为什么要读台账**（2026-09-30）：先前只认主文件，导致 768 篇被 LLM 判定
    "未描述可复现机制"的跳过结果不落盘 → 每次重跑都要把它们重新喂一遍 LLM
    （实测单轮约 25 分钟纯浪费）。fail 记录只进台账、不进 done，保持可重试。
    """
    done: set[str] = set()
    max_id = 1000
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except Exception:
                    continue
                p = (c.get("source") or {}).get("path")
                if p:
                    done.add(str(p))
                m = re.fullmatch(r"mc_(\d+)", str(c.get("card_id") or ""))
                if m:
                    max_id = max(max_id, int(m.group(1)))
    if skip_path and skip_path.is_file():
        with open(skip_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") == "skip" and rec.get("path"):
                    done.add(str(rec["path"]))
    return done, max_id


def list_reports(corpus: Path, only: str | None, min_bytes: int, limit: int | None) -> list[Path]:
    files = [p for p in corpus.rglob("*.md") if p.is_file()]
    if only:
        files = [p for p in files if only in str(p)]
    picked = []
    for p in files:
        try:
            if p.stat().st_size < min_bytes:
                continue
        except OSError:
            continue
        picked.append(p)

    def sort_key(p: Path):
        s = str(p)
        in_factors = 0 if "01_多因子与选股体系" in s else 1
        in_single = 0 if "01_单因子测试与挖掘" in s else 1
        return (in_factors, in_single, p.name)

    picked.sort(key=sort_key)
    return picked[:limit] if limit else picked


def run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    corpus = Path(args.corpus).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ledger = skipped_path(out_path)
    done, max_id = load_done(out_path, None if args.rescan_skipped else ledger)
    print(f"[init] 输出文件: {out_path}")
    print(f"[init] 已存在 {len(done)} 条 source.path（将跳过），最大 card_id 序号 mc_{max_id:04d}")
    if args.rescan_skipped:
        print(f"[init] --rescan-skipped：忽略 {ledger.name} 里的历史跳过判定")
    else:
        print(f"[init] 跳过/失败台账 {ledger.name}：{count_lines(ledger)} 条"
              f"（skip 命中即不再重跑；fail 仍会重试）")

    reports = list_reports(corpus, args.only, args.min_bytes, args.limit)
    todo = [p for p in reports if str(p.relative_to(corpus)).replace("\\", "/") not in done]
    print(f"[init] 语料候选 {len(reports)} 篇，待处理 {len(todo)} 篇，workers={args.workers}")

    if args.dry_run:
        for p in todo[:5]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            pages = split_pages(text)
            sel = select_pages(pages, args.max_chars)
            print(f"[dry] {p.name} 全文 {len(text)} 字 / {len(pages)} 页 → 选 {len(sel)} 页 "
                  f"({sum(len(t) for _, t in sel)} 字) prompt≈{len(build_prompt(parse_meta(p), sel))} 字")
        return 0

    write_lock = threading.Lock()
    counters = {"ok": 0, "skip": 0, "fail": 0}
    start = time.time()
    next_id = max_id + 1
    results: list[tuple[str, Any]] = []

    def work(path: Path) -> tuple[str, Any]:
        nonlocal next_id
        rel = str(path.relative_to(corpus)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return ("fail", rel, f"读取失败: {e}")
        pages = split_pages(text)
        sel = select_pages(pages, args.max_chars)
        prompt = build_prompt(parse_meta(path), sel)
        obj, err = call_llm(prompt, cfg, attempts=args.retries)
        if obj is None:
            return ("fail", rel, err)
        if obj.get("has_mechanism") is False:
            return ("skip", rel, "报告未描述可复现机制")
        full_norm, full_idx = norm_with_index(text)
        with write_lock:
            cid = f"mc_{next_id:04d}"
            next_id += 1
        card, err2 = build_card(obj, text, full_norm, full_idx, parse_meta(path), rel, cid,
                                cfg["model"], len(pages))
        if card is None:
            return ("skip", rel, f"卡片不完整: {err2}")
        errs = validate_card(card)
        if errs:
            return ("skip", rel, f"schema 校验失败: {errs}")
        return ("ok", rel, card)

    with open(out_path, "a", encoding="utf-8") as fout, \
            open(ledger, "a", encoding="utf-8") as fledger:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(work, p): p for p in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    counters["fail"] += 1
                    print(f"[{i}/{len(todo)}] 异常: {type(e).__name__}: {e}", flush=True)
                    continue
                kind, rel, payload = res
                if kind == "ok":
                    with write_lock:
                        fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        fout.flush()
                    counters["ok"] += 1
                    results.append((rel, payload))
                    el = time.time() - start
                    print(f"[{i}/{len(todo)}] OK {payload['card_id']} conf={payload['extraction']['confidence']} "
                          f"formula={'Y' if payload['formula_text'] else 'n'} "
                          f"ev={payload['extraction']['evidence_verified']} {rel} ({el:.0f}s)", flush=True)
                elif kind == "skip":
                    counters["skip"] += 1
                    with write_lock:
                        fledger.write(json.dumps(
                            {"path": rel, "kind": "skip", "reason": payload,
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                            ensure_ascii=False) + "\n")
                        fledger.flush()
                    print(f"[{i}/{len(todo)}] SKIP {rel} :: {payload}", flush=True)
                else:
                    counters["fail"] += 1
                    with write_lock:
                        fledger.write(json.dumps(
                            {"path": rel, "kind": "fail", "reason": payload,
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                            ensure_ascii=False) + "\n")
                        fledger.flush()
                    print(f"[{i}/{len(todo)}] FAIL {rel} :: {payload}", flush=True)

    el = time.time() - start
    print(f"[done] 新增 {counters['ok']} 张卡 | 跳过 {counters['skip']} 篇 | 失败 {counters['fail']} 篇 | "
          f"耗时 {el:.0f}s ({el / 60:.1f} min)")
    if results:
        hi = sum(1 for _, c in results if c["extraction"]["confidence"] == "high")
        mid = sum(1 for _, c in results if c["extraction"]["confidence"] == "medium")
        lo = sum(1 for _, c in results if c["extraction"]["confidence"] == "low")
        wf = sum(1 for _, c in results if c["formula_text"])
        print(f"[stats] confidence high={hi} medium={mid} low={lo} | 公式回证成功={wf}/{len(results)}")
    # 全量校验
    bad = 0
    total = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            errs = validate_card(json.loads(line))
            if errs:
                bad += 1
                print(f"[validate] {errs}", file=sys.stderr)
    print(f"[validate] 输出文件共 {total} 张卡，校验错误 {bad} 条")
    return 0 if bad == 0 else 1


def selftest() -> int:
    text = "因子 IC 均值为 -0.034，ICIR 为 -1.15。公式为 Skew = E[(R - mu)^3] / sigma^3 。\n分组收益呈单调负向。"
    full_norm, full_idx = norm_with_index(text)
    q = ground_quote(full_norm, full_idx, text, "IC 均值为 -0.034")
    assert q == "IC 均值为 -0.034", q
    assert ground_quote(full_norm, full_idx, text, "IC均值为-0.035") is None
    assert quote_has_value("IC 为 -0.034", -0.034)
    assert not quote_has_value("IC 为 -0.034", -0.35)
    assert quote_has_value("0.120 0.130 0.112 0.0870.349 0.206", 0.349)  # OCR 表格数字粘连
    assert not quote_has_value("0.351", 0.35)
    assert norm_decile_shape("单调负向") == "单调负向"
    assert norm_decile_shape("很乱") is None
    assert is_valid_field("funda_roe") and is_valid_field("chip_peak_loc") and not is_valid_field("my_factor")
    expr, cols = validate_dsl_hint("CS_ZSCORE(TS_MEAN($close, 20))")
    print("dsl:", expr, cols)
    assert expr and cols == ["close"]
    assert validate_dsl_hint("CS_ZSCORE(TS_MEAN($bogus_col, 20))")[0] is None
    assert validate_dsl_hint("CS_ZSCORE(TS_MEAN($volume, 20)")[0] is None
    assert validate_dsl_hint("CS_RANK(($pred_np[1] - $pred_np[0]) / $pred_np[0])")[0] is None
    assert validate_dsl_hint("CS_RANK($pb)")[0] == "CS_RANK($pb)"
    pages = split_pages("<!-- page 1 of 2 -->\nABC\n<!-- page 2 of 2 -->\nDEF")
    assert pages == [(1, "\nABC\n"), (2, "\nDEF")], pages
    meta = parse_meta(Path("2016-06-27_海通证券_金融工程_选股因子系列研究（十二）：“量”与“价”的结合.md"))
    assert meta["date"] == "2016-06-27" and meta["org"] == "海通证券", meta
    print("selftest OK", meta)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="研报机制卡批量抽取（MinerU 语料）")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="MinerU 语料根目录")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出 JSONL（追加写）")
    ap.add_argument("--limit", type=int, default=None, help="最多处理多少篇报告")
    ap.add_argument("--workers", type=int, default=2, help="并发工作线程数")
    ap.add_argument("--only", default=None, help="只处理路径包含该子串的报告")
    ap.add_argument("--min-bytes", type=int, default=20000,
                    help="跳过字节数小于该值的报告（日报/周报多为 <20KB，无机制内容）")
    ap.add_argument("--max-chars", type=int, default=20000, help="送入 LLM 的正文预算（字符）")
    ap.add_argument("--retries", type=int, default=3, help="单篇 LLM 调用重试次数")
    ap.add_argument("--max-tokens", type=int, default=1500, help="单次 LLM 回复上限（JSON 过长会被截断）")
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--base-url", default="http://127.0.0.1:8317/v1")
    ap.add_argument("--api-key", default="123456")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="只打印每篇选页/prompt 规模，不调用 LLM")
    ap.add_argument("--rescan-skipped", action="store_true",
                    help="忽略 <out>.skipped.jsonl 里的历史跳过判定，重新评估这些报告")
    ap.add_argument("--selftest", action="store_true", help="运行离线自检（不联网）")
    ap.add_argument("--verify", action="store_true", help="对已落盘 --out 文件做红线回证复核（不调用 LLM）")
    ap.add_argument("--repair-meta", action="store_true", help="按文件名回填 source.org 为空的卡片（不调用 LLM）")
    ap.add_argument("--refine-hint", action="store_true",
                    help="对退化占位 dsl_hint 定向重问（无法表达则置 null，不编造）")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.verify:
        return verify_output(Path(args.out).resolve(), Path(args.corpus).resolve())
    if args.repair_meta:
        return repair_meta(Path(args.out).resolve(), Path(args.corpus).resolve())
    if args.refine_hint:
        cfg = {
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "timeout": args.timeout,
            "max_tokens": args.max_tokens,
        }
        return refine_hints(Path(args.out).resolve(), Path(args.corpus).resolve(), cfg, args.workers)

    cfg = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
    }
    return run(cfg, args)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    sys.exit(main())

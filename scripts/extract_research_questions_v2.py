#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研报课题抽取 v2（LLM 主导）——试点脚本。

设计原则（见 docs/specs/alphaagent_question_schema_v2.md §2）：
  · **LLM 判定为主**：主因子/主公式、报告声明了什么指标、频率、机制 —— 全部由 LLM 读原文判定；
  · 规则只做两件事：①准备输入（切分/清洗/限长）②回证（把 LLM 给的 quote 拿回原文定位）。

LLM：`load_codex_provider()` 注入的模型（当前 ~/.codex/config.toml = DeepSeek-V4-Flash-0731）。

用法：
  python scripts/extract_research_questions_v2.py --limit 20
  python scripts/extract_research_questions_v2.py --only 华泰单因子测试之财务质量因子
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from alphaagent.core.llm_provider import chat_json, load_codex_provider  # noqa: E402

# 复用公式抽取脚本的算子/字段索引（`load_index` 带缓存），确保**同一口径**：
# 不给这份清单，模型只能给 LaTeX/文字，抽不出本仓 DSL（试点 18/18 篇 expr_local=None 的根因）。
sys.path.insert(0, str(REPO / "scripts"))
from extract_factor_records import load_index  # noqa: E402

_OPS, _FLDS = load_index()
OPRE_FULL = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
OPS_TEXT = ", ".join(sorted(_OPS))[:6000]
FLDS_TEXT = ", ".join(sorted(_FLDS))[:6000]

CORPUS = REPO / "data" / "research_reports" / "parsed"
KNOW = REPO / "data" / "research_reports" / "knowledge"
OUT_JSONL = KNOW / "question_v2_pilot.jsonl"
OUT_MD = KNOW / "question_v2_pilot_report.md"

MAX_CHARS = 200000         # 单篇送入正文上限（DeepSeek 输入 ~180k tokens，整篇够用；不裁剪）
NOISE = re.compile(r"免责声明|分析师声明|评级说明|特别声明|法律声明|风险提示及免责|投资评级说明")

SYSTEM = "你是券商金工研报的结构化抽取器。只依据给定原文作答，不得推测或补全原文没有的信息。"

USER_TMPL = """下面是一篇券商金工研报的 Markdown 正文（可能有页眉页脚与表格残渣，请自行忽略噪声）。

请通读后输出**一个 JSON 对象**（不要多余文字），字段如下：

{{
  "report_type": "深度研究|单因子测试|专题|策略|周报|月报|文献推荐|其他",
  "topic": "本文核心因子/机制的一句话课题名",
  "report_structure": "gated|composite|residual|timing|cross_facet|none",
  "has_reproducible_structure": true/false,
  "reproduction_target": "一句话：本篇该复现的结构（不是某条公式）；report_structure=none 时填空字符串",
  "factor_construction_found": true/false,
  "hypothesis": "可证伪命题：谁、在什么条件下、为什么、预期什么（2~4 句，必须能在原文找到依据）",
  "mechanism": "经济机制因果链（2~4 句）",
  "falsifier": "什么观察结果算被证伪（1~2 句）",
  "direction_hint": 1 或 -1 或 null,      // 仅当原文明确写"越大越好/越小越好"时给出，否则 null
  "primary": {{
    // primary = 参考公式（reference_formula），不是复现目标；复现目标是 reproduction_target（结构）
    "kind": "verbatim|derived|spec_text",
    "name": "该因子的报告用名",
    "expr_raw": "原文公式（LaTeX 原样照抄）；kind=spec_text 时为 null",
    "expr_local": "本仓 DSL 表达式（用 $字段名 与大写算子，如 CS_ZSCORE(DIVIDE($funda_ocf,$funda_total_revenue))）；无法给出则 null",
    "spec_text": "只有文字说明时（kind=spec_text）把说明原文摘出来；否则 null",
    "spec_requirements": {{"fields": ["$xxx"], "windows": [20], "operators": ["CS_ZSCORE"],
                         "chain": ["winsorize", "zscore", "neutralize"],
                         "structure_ops": ["CS_GROUP_RANK"],  // 复现结构必须用的结构性算子
                         "min_fields": 3,                     // 复现该结构至少要几个字段
                         "direction": 1 或 -1 或 null}},
    "evidence": {{"page": 页码或 null, "table_title": "所在表/图标题（原文照抄，没有则 null）", "quote": "原文逐字片段（20~120 字，必须能在原文中找到）"}},
    "confidence": 0.0~1.0
  }},
  "reported_metrics": [
    {{"metric": "IC|RankIC|ICIR|IC_IR|t值|月胜率|分组多空收益",
      "value": 数值, "stock_pool": "中证全指|沪深300|… 或 null", "horizon": "20d 或 月度 或 null",
      "evidence": {{"table_title": "…", "quote": "原文逐字片段"}}, "kind": "verbatim|derived"}}
  ],
  "reported_combo_perf": [
    {{"metric": "超额收益|年化|夏普|最大回撤", "value": 数值, "window": "如 年内/近一年 或 null",
      "evidence": {{"table_title": "…", "quote": "…"}}}}
  ],
  "reproducibility": "high|medium|low",
  "rebalance_freq": "daily|weekly|monthly|unknown",
  "holding_period": "原文描述或 null",
  "universe": "股票池或 null",
  "sample_range": "样本区间或 null"
}}

**本仓算子清单（expr_local 只能用它，共 {n_ops} 个，节选）**：
{ops}

**本仓字段清单（expr_local 只能用它，形如 $funda_roe；共 {n_flds} 个，节选）**：
{flds}

**判定要点（务必遵守）**：
1. `primary` 必须是**最能代表本文主旨的那一个因子**（不是罗列的因子清单里的任意一个）。若原文给了一整张因子表，请选"标题/摘要/结论里被当作主角"的那个。
1b. **`expr_local` 的取值纪律（严禁为凑字段硬给式子）**：
   · **只能**来源于两种情况：（ا）原文**明确给出算式**（照抄+转写）；（ب）原文用文字**明确描述了算法**（分子/分母/窗口/处理步骤齐），你能直接推出唯一算式。
   · `kind=verbatim/derived`：**必填**（上面两种情况）。
   · `kind=spec_text`：只有当原文**只有方法论/流程/框架**（如神经网络结构、遗传规划流程、因子择时检验框架、组合构建流程）时使用；
     此时 **`expr_local` 必须为 null**，且 `reproducibility` 必须是 `low`。
   · **可核对要求（机械判据）**：`kind=verbatim/derived` 时**必须同时给出 `expr_raw`**
     ——即原文里那一行算式的**逐字片段**（可以是 LaTeX/表格原文）。
     若你在原文里**找不到可以逐字引用的算式/算法定义行**，就**不能**判 verbatim/derived，
     请改判 `spec_text` 并令 `expr_local = null`。
     （下游脚本会用 `expr_raw` 做机械核对：没有原文算式依据的式子会被自动降级为 spec_text。）
   · 判断口诀：先问"原文里**有没有一行**能直接引来做 `expr_raw`"？有 → verbatim/derived；没有 → spec_text + null。
1f. **结构判定（最重要的字段，决定该篇是否出题）**——先判 `report_structure`：
    · `gated` 条件门控（"在 X 条件下该因子才有效"/分组/阈值/状态切换）；
    · `composite` 多因子合成（明确说了合成方式：等权/IC_IR 加权/打分/多步筛选）；
    · `residual` 残差化/中性化/正交化；
    · `timing` 择时/轮动（权重随状态切换）；
    · `cross_facet` 跨面融合（价量 × 基本面/资金面 的交互构型）；
    · `none` **只有通用单因子**（ROE/PB/动量/波动率这种），没有任何上述结构。
   `has_reproducible_structure = (report_structure != "none")`；
   `reproduction_target` 写成**可执行的结构描述**（含关键参数/步骤/字段），例如
   "GMSD 前 2/3 → SIRD 前 50% → PB 最低 1/3 三步筛选"；
   并把复现该结构所需要素填进 `primary.spec_requirements`——**注意：填的是"结构"的要素，
   不是参考公式的要素**（标定实测：59% 的题未声明 structure_ops、声明字段常只有 1 个 →
   下游结构锚形同虚设）：
     · `fields`：**列全该结构用到的所有字段**（多因子合成就列全部成分字段），不要只填参考公式那两个；
     · `operators`：该结构依赖的算子（用上面清单里的名字）；
     · `structure_ops`：**结构性算子**，从下面白名单里选该结构真正依赖的（可为多个）：
       `CS_GROUP_RANK` | `SOFT_GATE` | `DIVERGENCE_RANK` | `CS_RESIDUALIZE` |
       `CS_NEUTRALIZE` | `IF_THEN_ELSE` | `CS_BUCKET` | `PIECEWISE_STATE`；
       若该结构确实不依赖上述任何一个（例如纯等权合成），**留空数组**并在 `spec_text` 里写清合成方式；
     · `min_fields`：复现该结构**至少要用几个字段**（应与 fields 数量一致或略低，不要虚高）。
   ⚠ 若判 `none`，说明本篇**不应作为复现题**（`reproduction_target` 留空），不要硬凑结构。
   ⚠⚠ **算子白名单（硬约束）**：`operators` / `structure_ops` / `expr_local` **只能使用上面清单里
   已有的算子名**，**严禁自造**（例：`SIGNAL_BLEND` 这类清单里没有的名字一律不许写）。
   `composite`（多因子合成）请用清单内算子表达，例如：
     · 等权打分合成：`CS_ZSCORE(ADD(CS_ZSCORE($a), CS_ZSCORE($b)))`
     · 分组/门控合成：`SOFT_GATE(...)` / `CS_GROUP_RANK(...)`
     · 残差/正交合成：`CS_RESIDUALIZE(...)`
   若某个结构在本仓算子下**确实表达不出来**，就把 `structure_ops` 留空并在 `spec_text` 里写清步骤，
   **不要**编一个算子名。
1d. **`reproducibility` 必须由 `kind` 推导（自洽优先，不要按"报告主题是不是机器学习/工程"来判）**：
   · `kind=verbatim` → **至少 `medium`**；若 `expr_local` 也给出且 `spec_requirements` 齐 → `high`。
     **禁止**给 `verbatim` 判 `low`（原文都有算式了，怎么可能不可复现）。
   · `kind=derived` → `expr_local` 给出且 `fields`/`operators` 齐 → `high`；否则 `medium`。
   · `kind=spec_text` → 算法要素（fields 等）齐、但仍推不出唯一表达式 → `medium`；
     连算法要素都给不出（只讲模型框架/流程/组合构建，例如神经网络结构、遗传规划流程、行业分类方法）→ `low`。
   ⚠ **判 `low` 的唯一理由**：原文确实**没有单一因子可复现**。若你已经抽出 `verbatim`/`derived`，
   就不可能是 `low`——请回头改判，不要自相矛盾。
1e. **必须体现报告声明的"完整处理链"**：研报常见表述是"因子 = 某比值，再做去极值 / 截面标准化 / 行业市值中性化 / 合成"。
   若原文提到了这些步骤，`expr_local` **必须把它们写进表达式**（用 `CS_WINSORIZE` / `CS_ZSCORE` / `CS_NEUTRALIZE` / `CS_RESIDUALIZE` / `SOFT_GATE` 等），
   并在 `spec_requirements.chain` 里按顺序列出步骤名（从 `winsorize|zscore|neutralize|residualize|rank|gate|blend` 中选）。
   ⚠ 只给"分子/分母"而漏掉中性化/标准化，会导致复现出来的因子**不是研报那个因子**（下游 IC 口径会被改变）。
1c. **`spec_requirements` 必填且 `fields` 至少 1 个**（从上面字段清单里选，写 `$字段名`）。`direction`：原文说"越大越好"给 1、"越小越好"给 -1、未说给 null。`windows` 写数字（如 [20]），没有给 []；`operators` 用清单里的算子名，没有给 []。
2. `kind` 三态：`verbatim`=原文直接给了算式；`derived`=原文用文字描述算法、你据此推出算式；`spec_text`=原文只有文字说明、推不出唯一算式（此时 expr_local 允许给一版候选，但 spec_text 必须填）。
3. **指标必须分成两类，绝不混放**：
   · `reported_metrics` = **因子级**：IC、RankIC、ICIR、IC_IR、t值、月胜率、分组多空收益；
   · `reported_combo_perf` = **组合业绩**：年化超额收益、累计收益、年化收益、夏普、最大回撤、净值、跟踪误差、信息比率。
   ⚠ **否定示例（这些一律进 reported_combo_perf，不要进 reported_metrics）**："组合年化超额收益 12.05%"、"指增组合年内超额 7.48%"、"夏普比率 0.79"、"最大回撤 -9.19%"。
   ⚠ 若某数值没有明确是"因子 IC/ICIR/t值/胜率"这类因子级口径，宁可放进 `reported_combo_perf`。
3b. **数值单位统一（按指标分开处理，别一律转小数）**：
   · 只对 **IC / RankIC / 胜率** 做小数归一（相关系数与比率，|值| 必在 1 以内）：原文"IC 10.11%"→0.1011、"RankIC 均值 11.59%"→0.1159、"胜率 77.42%"→0.7742；
   · **ICIR / IC_IR / t值 保持原值**（ICIR 是 IC 均值/标准差，量级 0.1~5 都正常；t值 同理）——**不要**把它们转成小数或百分数；
   · horizon 写"20d"/"月度"这类；stock_pool 写中证全指/沪深300 等，没有给 null。
4. 每个 `quote` 必须是原文连续片段，**不要改写**。

原文（文件名：{fname}）：
---
{body}
---
"""


PRIMARY_ONLY_TMPL = """从下面研报中只判定**主因子**，输出 JSON（不要多余文字）：
{{"report_type": "深度研究|单因子测试|专题|策略|周报|月报|文献推荐|其他",
  "topic": "一句话课题名",
  "primary": {{"kind": "verbatim|derived|spec_text", "name": "因子名",
    "expr_raw": "原文公式或 null", "expr_local": "本仓 DSL 或 null（只用这些算子: {ops}；字段形如 $funda_roe）",
    "spec_text": "文字说明或 null",
    "spec_requirements": {{"fields": ["$xxx"], "windows": [], "operators": [], "direction": null}},
    "evidence": {{"table_title": "…或 null", "quote": "原文逐字片段"}}, "confidence": 0.0}}}}

可用字段节选: {flds}

原文（{fname}）：
---
{body}
---
"""


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))


SECTION_KEY = re.compile(r"因子|构建|模型|策略|合成|中性|残差|正交|择时|轮动|测试|选股|打分|权重")


def load_body(p: Path) -> str:
    """输入准备（规则，允许）：按章节标题选"因子/模型/策略"相关章节（每节 ≤8000 字，最多 3 节）
    + 开头 3000 字；章节切不出来则全文兜底；去掉免责声明段。

    为什么要改：早期"前 4000 字 + 关键词行"的粗截取会让部分报告只剩风险提示与占位符，
    实测使 LLM 误判 report_structure=none（修好后 40 篇里有 4 篇从 none 翻转为有结构）。
    """
    txt = p.read_text(encoding="utf-8", errors="replace")
    txt = re.sub(r"免责声明.*", "", txt, flags=re.S)
    head = txt[:3000]
    parts = re.split(r"\n(?=#{1,3}\s)", txt)
    picked = [x for x in parts if SECTION_KEY.search(x[:80])][:3]
    if len(picked) >= 2:
        body = head + "\n\n[相关章节]\n" + "\n".join(x[:8000] for x in picked)
    else:
        body = txt
    return body[:MAX_CHARS]


def pick_reports(limit: int, only: str | None, offset: int = 0) -> list[Path]:
    recs = [json.loads(l) for l in (KNOW / "factor_records_pilot.jsonl").read_text(
        encoding="utf-8", errors="replace").splitlines() if l.strip()]
    by_src: dict[str, list[dict]] = {}
    for r in recs:
        by_src.setdefault(str(r.get("source")), []).append(r)
    # 先挑记录最多的（含 52/75 条那类"多公式"报告，最能检验 primary 选择），再均匀补足
    ranked = sorted(by_src.items(), key=lambda kv: -len(kv[1]))
    all_srcs = [s for s, _ in ranked]
    if offset:                                   # 抽"新篇"：在**完整候选**上切片
        return [CORPUS / s.replace("\\", "/") for s in all_srcs[offset:offset + limit]
                if (CORPUS / s.replace("\\", "/")).is_file()]
    picked: list[str] = [s for s, _ in ranked[:10]]
    rest = [s for s, _ in ranked[10:]]
    if rest:
        step = max(1, len(rest) // max(1, limit - 10))
        picked += rest[::step]
    out: list[Path] = []
    for src in picked:
        p = CORPUS / src.replace("\\", "/")
        if p.is_file():
            out.append(p)
    if only:
        keys = [k.strip() for k in str(only).split(",") if k.strip()]
        hit = [p for p in out if any(k in p.name for k in keys)]
        # 与全语料结果**取并集**（此前只在 hit 为空时兜底 → 多值 --only 会漏掉第二篇）
        for q in sorted(CORPUS.rglob("*.md")):
            if any(k in q.name for k in keys) and q not in hit:
                hit.append(q)
        return hit[:limit] if limit else hit
    return out[offset:offset + limit] if limit else out[offset:]


def quote_match_ratio(quote: str, body_norm: str) -> float:
    """回证命中率：整段命中=1.0；否则按 12 字分片算命中比例。

    为什么不用整段匹配：LLM 引用常有**轻微改写/漏字/符号差异**（实测广发《深度学习》
    的 quote 开头用 "ŷ1" 而正文写法不同 → 整段匹配必然失败；东方那篇则漏了两个字）。
    只要大部分片段能在原文找到，就足以证明"不是凭空编的"（回证的目的是防幻觉，不是逐字比对）。
    """
    q = norm(quote)
    if len(q) < 8:
        return 0.0
    if q[:60] in body_norm or q[-60:] in body_norm:
        return 1.0
    segs = [q[i:i + 12] for i in range(0, len(q) - 7, 12)]
    if not segs:
        return 0.0
    return sum(1 for s in segs if s in body_norm) / len(segs)


def verify_quote(quote: str, body_norm: str) -> bool:
    return quote_match_ratio(quote, body_norm) >= 0.6


def extract_one(p: Path) -> dict:
    body = load_body(p)
    t0 = time.time()
    res = chat_json(SYSTEM, USER_TMPL.format(
        fname=p.name, body=body, ops=OPS_TEXT, flds=FLDS_TEXT,
        n_ops=len(_OPS), n_flds=len(_FLDS)),
        max_tokens=16000, temperature=0.2)
    if not isinstance(res, dict):
        # 降级：长报告输出易被截断（试点 2 篇 140s 失败）。只索要最小字段，保证不整篇白跑。
        res = chat_json(SYSTEM, PRIMARY_ONLY_TMPL.format(
            fname=p.name, body=body, ops=OPS_TEXT, flds=FLDS_TEXT),
            max_tokens=4000, temperature=0.1)
        if isinstance(res, dict):
            res["_degraded"] = True
    dt = time.time() - t0
    if not isinstance(res, dict):
        return {"source": str(p.relative_to(CORPUS)).replace("/", "\\"),
                "ok": False, "error": "llm_no_json", "elapsed": round(dt, 1)}
    bn = norm(body)
    # 机械核对（规则，允许）：operators / structure_ops / expr_local 里的算子必须都在本仓清单内
    sr = ((res.get("primary") or {}).get("spec_requirements") or {}) if isinstance(res, dict) else {}
    declared = list(sr.get("operators") or []) + list(sr.get("structure_ops") or [])
    declared += OPRE_FULL.findall(str((res.get("primary") or {}).get("expr_local") or ""))
    unknown = sorted({o for o in declared if o not in _OPS})
    if isinstance(res, dict):
        res["unknown_ops"] = unknown
        if isinstance(res.get("primary"), dict) and isinstance(
                res["primary"].get("spec_requirements"), dict):
            res["primary"]["spec_requirements"]["ops_valid"] = not unknown
        if res.get("has_reproducible_structure") is True and not str(
                res.get("reproduction_target") or "").strip():
            res["_target_missing"] = True
    prim = res.get("primary") or {}
    ev = (prim.get("evidence") or {}).get("quote") or ""
    prim["evidence_match_ratio"] = round(quote_match_ratio(ev, bn), 2)
    prim["evidence_verified"] = prim["evidence_match_ratio"] >= 0.6
    for m in res.get("reported_metrics") or []:
        m["evidence_verified"] = verify_quote((m.get("evidence") or {}).get("quote") or "", bn)
    return {
        "source": str(p.relative_to(CORPUS)).replace("/", "\\"),
        "ok": True, "elapsed": round(dt, 1), "body_chars": len(body),
        **res,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--only", default=None, help="逗号分隔的报告名子串（重抽指定篇）")
    ap.add_argument("--offset", type=int, default=0, help="跳过排序后的前 N 篇（抽新篇用）")
    ap.add_argument("--out", default="question_v2_pilot",
                    help="输出文件基名（写 <out>.jsonl 与 <out>_report.md）")
    ap.add_argument("--workers", type=int, default=6,
                    help="并发度；单篇 5~100s 差异极大，并发可把整批压到分钟级")
    args = ap.parse_args()

    # 本脚本必须走 codex provider（DeepSeek-V4-Flash-0731 在 ~/.codex/config.toml）。
    # 注意：worktree 里没有 .env（被 gitignore），load_codex_provider 的 dotenv 会读空 →
    # 必须显式给 ALPHA_LLM_PROVIDER，否则它会早退、MODEL 为空、chat_json 直接返回 None。
    os.environ.setdefault("ALPHA_LLM_PROVIDER", "codex")
    load_codex_provider()
    if not os.getenv("MODEL"):
        print("!! 未取到模型配置：需要 ALPHA_LLM_PROVIDER=codex 且 ~/.codex/config.toml 可读",
              flush=True)
        return 2
    print(f"model={os.getenv('MODEL')} base={os.getenv('OPENAI_API_BASE')}", flush=True)

    global OUT_JSONL, OUT_MD
    OUT_JSONL = KNOW / f"{args.out}.jsonl"
    OUT_MD = KNOW / f"{args.out}_report.md"
    reports = pick_reports(args.limit, args.only, args.offset)
    workers = max(1, min(int(args.workers), len(reports) or 1))
    print(f"待抽 {len(reports)} 篇 | 并发 {workers}", flush=True)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.time()
    progress_file = OUT_JSONL.with_name(OUT_JSONL.stem + "_progress.txt")
    progress_lines: list[str] = []

    def _write_progress(head: str) -> None:
        """进度文件：命令输出常被管道缓冲，落盘文件才能真正"随时看进度"。"""
        try:
            progress_file.write_text(head + "\n" + "\n".join(progress_lines), encoding="utf-8")
        except OSError:
            pass

    _write_progress(f"开始 {time.strftime('%H:%M:%S')} | 共 {len(reports)} 篇 | 并发 {workers}")
    with OUT_JSONL.open("w", encoding="utf-8") as fh, \
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_one, p): p for p in reports}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            p = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = {"source": str(p.relative_to(CORPUS)).replace("/", "\\"),
                       "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            row["_order"] = reports.index(p)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            prim = row.get("primary") or {}
            print(f"  [{done}/{len(reports)}] {p.name[:42]} "
                  f"ok={row.get('ok')} kind={prim.get('kind')} "
                  f"name={str(prim.get('name'))[:16]} {row.get('elapsed')}s", flush=True)
            el = time.time() - t_start
            eta = el / done * (len(reports) - done)
            progress_lines.append(
                f"[{done}/{len(reports)}] {p.name[:46]} ok={row.get('ok')} "
                f"kind={prim.get('kind')} repro={row.get('reproducibility')} "
                f"回证={prim.get('evidence_verified')} {row.get('elapsed')}s")
            _write_progress(
                f"{time.strftime('%H:%M:%S')} | done {done}/{len(reports)} | "
                f"已用 {el/60:.1f}m | ETA ~{eta/60:.1f}m")
    rows.sort(key=lambda r: r.get("_order", 0))

    # ── 人读报告（重点看公式）──
    lines = ["# 课题抽取 v2 试点报告（重点：primary 公式）", "",
             f"- 模型：`{os.getenv('MODEL')}`", f"- 篇数：{len(rows)}", ""]
    for r in rows:
        prim = r.get("primary") or {}
        lines.append(f"## {Path(r['source']).name[:70]}")
        lines.append(f"- 状态：ok={r.get('ok')}  用时={r.get('elapsed')}s  "
                     f"report_type={r.get('report_type')}")
        if not r.get("ok"):
            lines.append(f"- ⚠ {r.get('error')}")
            lines.append("")
            continue
        lines.append(f"- **topic**：{r.get('topic')}")
        lines.append(f"- **hypothesis**：{str(r.get('hypothesis'))[:220]}")
        lines.append(f"- **falsifier**：{str(r.get('falsifier'))[:160]}")
        lines.append(f"- **primary.kind**：`{prim.get('kind')}`  "
                     f"**name**：`{prim.get('name')}`  回证={prim.get('evidence_verified')}")
        if prim.get("expr_raw"):
            lines.append(f"- expr_raw：`{str(prim['expr_raw'])[:220]}`")
        lines.append(f"- **expr_local**：`{prim.get('expr_local')}`")
        if prim.get("spec_text"):
            lines.append(f"- spec_text：{str(prim['spec_text'])[:220]}")
        lines.append(f"- spec_requirements：`{json.dumps(prim.get('spec_requirements'), ensure_ascii=False)}`")
        ev = prim.get("evidence") or {}
        lines.append(f"- evidence：table=`{str(ev.get('table_title'))[:70]}` quote=「{str(ev.get('quote'))[:110]}」")
        rm = r.get("reported_metrics") or []
        lines.append(f"- reported_metrics（{len(rm)}）：" +
                     "; ".join(f"{m.get('metric')}={m.get('value')}({m.get('stock_pool')},回证={m.get('evidence_verified')})"
                               for m in rm[:6]))
        cp = r.get("reported_combo_perf") or []
        lines.append(f"- reported_combo_perf（{len(cp)}）：" +
                     "; ".join(f"{m.get('metric')}={m.get('value')}" for m in cp[:4]))
        lines.append(f"- 频率：{r.get('rebalance_freq')} / 持有期：{r.get('holding_period')} / "
                     f"股票池：{r.get('universe')} / 区间：{r.get('sample_range')}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    _write_progress(f"完成 {time.strftime('%H:%M:%S')} | {len(rows)}/{len(reports)} 篇")
    print(f"\n写出：{OUT_JSONL}\n      {OUT_MD}\n      进度文件：{progress_file}", flush=True)

    okn = sum(1 for r in rows if r.get("ok"))
    kinds = {}
    for r in rows:
        k = (r.get("primary") or {}).get("kind")
        kinds[k] = kinds.get(k, 0) + 1
    print(f"成功 {okn}/{len(rows)}；kind 分布={kinds}；"
          f"primary 回证通过={sum(1 for r in rows if (r.get('primary') or {}).get('evidence_verified'))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

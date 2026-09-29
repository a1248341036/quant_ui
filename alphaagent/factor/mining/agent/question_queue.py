# -*- coding: utf-8 -*-
"""模块 · question_queue：研报研究问题队列驱动器（R4 方案，RDAgent 范式）。

将金工研报文献转化为可证伪的“结构化研究问题清单”。
在每轮挖掘启动时，按轮次向 LLM 下达具体的定向攻关任务，驱动 LLM 从
“盲目自由搜索”收敛到“研报假设验证与衍生对账”闭环。
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from alphaagent.dsl.core.field_aliases import FIELD_ALIASES, canonical_column

logger = logging.getLogger(__name__)

# 精选研报经典研究问题题库（涵盖量价背离、高阶矩、筹码密集、聪明钱、流动性冲击等）
DEFAULT_RESEARCH_QUESTIONS: list[dict[str, Any]] = [
    {
        "question_id": "RQ_01",
        "topic": "成交量与价格新高的顶背离衰竭验证",
        "source": "海通证券《选股因子系列（十二）：量与价的结合》",
        "hypothesis": "在股票突破20日价格新高时，若成交量未同步放大反而萎缩超过30%，表明追涨意愿衰竭，未来5-10日呈现高概率均值回归回调。",
        "suggested_fields": ["close", "volume", "high"],
        "suggested_operators": ["DIVERGENCE_RANK", "TS_MAX", "TS_PCTCHANGE"],
        "expected_shape": "单调负向",
        "expected_sign": "negative",
        "falsifier": "背离后持续放量突破导致空头失效",
        "facets": ["价量面", "量能面"]
    },
    {
        "question_id": "RQ_02",
        "topic": "彩票型偏好下的日内残差偏度溢价检验",
        "source": "东方证券《因子选股系列（九）：日内残差高阶矩》",
        "hypothesis": "投资者对高暴击右偏股票存在过度投机偏好并推升估值，剔除市场基准后的负残差偏度股票未来具备稳健的风险补偿超额。",
        "suggested_fields": ["close", "high", "low", "vwap"],
        "suggested_operators": ["TS_SKEW", "CS_RESIDUALIZE", "CS_ZSCORE"],
        "expected_shape": "单调正向",
        "expected_sign": "positive",
        "falsifier": "市场风格切换为纯小盘垃圾股脉冲暴动行情",
        "facets": ["价量面", "筹码面"]
    },
    {
        "question_id": "RQ_03",
        "topic": "筹码峰套牢阻力位与反弹弹性度量",
        "source": "国泰君安《数量化专题：筹码分布与反弹阻力》",
        "hypothesis": "股价回撤至筹码主峰密集区下方深处时，上方密集筹码形成强抛压，距主峰越近解套抛压越大，而在超跌区且远离主峰的标的反弹弹性最佳。",
        "suggested_fields": ["close", "turnover", "volume"],
        "suggested_operators": ["CHIP_PEAK_LOC", "TS_PCTCHANGE", "IF_THEN_ELSE"],
        "expected_shape": "倒U型",
        "expected_sign": "positive",
        "falsifier": "筹码在低位快速换手完成重心下移",
        "facets": ["筹码面", "价量面"]
    },
    {
        "question_id": "RQ_04",
        "topic": "大单主力知情交易流入的持续性alpha",
        "source": "海通证券《选股因子系列（十一）：level2行情选股因子初探》",
        "hypothesis": "超大单与大单净买入主要由机构或主力知情资金发起，由于算法拆单与建仓周期，其净流入占比对未来5日超额收益有显著正向预测力。",
        "suggested_fields": ["ff_super_net", "ff_large_net", "amount"],
        "suggested_operators": ["CS_ZSCORE", "TS_SUM", "DIVIDE"],
        "expected_shape": "单调正向",
        "expected_sign": "positive",
        "falsifier": "游资对倒制造虚假超大单诱多后迅速反水",
        "facets": ["资金面", "量能面"]
    },
    {
        "question_id": "RQ_05",
        "topic": "下行半方差与下行风险不对称补偿",
        "source": "海通证券《“双面”波动率：波动率因子的分解与截面收益》",
        "hypothesis": "投资者厌恶下行波动而非总波动；个股下行半波动率相对于上行波动的比率越低，说明前期调整受恐慌错杀较深，未来夏普比更高。",
        "suggested_fields": ["close", "open", "low"],
        "suggested_operators": ["TS_STD", "IF_THEN_ELSE", "DIVIDE"],
        "expected_shape": "单调负向",
        "expected_sign": "negative",
        "falsifier": "极端下行波动由基本面实质违约或重大暴雷引起",
        "facets": ["价量面", "拥挤面"]
    }
]


def resolve_questions_file(spec: dict[str, Any] | None = None) -> Path | None:
    policy = (spec or {}).get("report_policy") or {}
    custom = policy.get("question_queue_file")
    if custom:
        p = Path(custom).resolve()
        if p.is_file():
            return p

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            p = Path(out).parent / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"
            if p.is_file():
                return p
    except Exception:
        pass

    # 优先支持 worktree 相对根路径与主仓库路径
    repo_root = Path(__file__).resolve().parents[4]
    p_local = repo_root / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"
    if p_local.is_file():
        return p_local
    main_repo = Path(r"D:\Quant\quant_ui\data\research_reports\knowledge\research_questions.jsonl")
    if main_repo.is_file():
        return main_repo

    fallback = Path(__file__).resolve().parents[5] / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"
    return fallback if fallback.is_file() else None


def load_question_queue(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    path = resolve_questions_file(spec)
    if path and path.is_file():
        questions = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    questions.append(json.loads(line))
            if questions:
                return questions
        except Exception as e:
            logger.warning("读取问题题库失败: %s", e)

    return list(DEFAULT_RESEARCH_QUESTIONS)


# ── 课题字段门控（2026-09-28）────────────────────────────────────────────
# 背景：整夜自由探索 run（`focus_facets` 为空 → technical 档 + `--no-fundamentals`）
# 实际只载入行情族 + 资金/事件/股东/业绩/机构列族；而 1200 道题库是全八面的，
# 派题时不看面板载入了什么 → 纯基本面/行业/两融/分红题注定一次白跑（实测每 run
# `$industry_sw_l1` 报 2~6 次、`$turnover` 报 1~4 次「不可用字段」）。
# 默认值唯一真源在本常量；`research_spec` 引用它做默认与校验，避免双写漂移。
DEFAULT_QUESTION_FIELD_GATE: dict[str, Any] = {
    "enabled": True,        # report_policy.question_field_gate
    "min_ratio": 0.5,       # report_policy.question_field_gate_min_ratio
    "scan_limit": 40,       # report_policy.question_field_gate_scan_limit
    "warn": True,           # report_policy.question_field_warn
}

# 列族前缀：题面出现这些前缀（如 ``funda_ocf``）时按「该列族是否载入」判定。
_COLUMN_FAMILY_PREFIXES: tuple[str, ...] = (
    "funda_", "holder_", "inst_", "th_", "ff_", "mgn_", "dt_", "bt_",
    "pred_", "exp_", "ds_", "div_", "industry_",
)

# 核算子派生面：由行情列现算（CHIP_*/CROWD_* 等），行情列载入即可用——不作为缺列判定。
_OPERATOR_DERIVED_TOKENS: frozenset[str] = frozenset({
    "crowding", "crowd", "chip", "vpin", "wick", "gap", "fractal", "entropy",
    "effratio", "efficiency", "kline", "amihud", "illiq", "skew", "kurt",
})

# 题面散文词 → 列族前缀（题库把面名/指标口语混写在 suggested_fields 里，
# 如 ``crowding``/``announcement``/``eps``；表集中在此，可单测）。
_FIELD_FAMILY_SYNONYMS: dict[str, str] = {
    "eps": "funda_", "roe": "funda_", "roa": "funda_", "net_profit": "funda_",
    "netprofit": "funda_", "revenue": "funda_", "ocf": "funda_",
    "total_assets": "funda_", "valuation": "funda_", "pe": "funda_", "pb": "funda_",
    "holder": "holder_", "holder_count": "holder_", "shareholder": "holder_",
    "shareholders": "holder_", "top10": "th_", "concentration": "th_",
    "institutional": "inst_", "institution": "inst_", "inst": "inst_",
    "fund_flow": "ff_", "money_flow": "ff_", "capital_flow": "ff_", "main_net": "ff_",
    "announcement": "ds_", "disclosure": "ds_", "event": "dt_",
    "forecast": "pred_", "express": "exp_", "surprise": "pred_",
    "dividend": "div_", "margin": "mgn_", "industry": "industry_",
}


def resolve_question_field_gate(
    spec: dict[str, Any] | None = None,
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """解析课题字段门控配置：``spec.report_policy`` < ``override``（运行期覆盖）。"""
    policy: dict[str, Any] = dict(DEFAULT_QUESTION_FIELD_GATE)
    rp = (spec or {}).get("report_policy") or {}
    raw_enabled = rp.get("question_field_gate")
    if raw_enabled is not None:
        policy["enabled"] = bool(raw_enabled)
    raw_ratio = rp.get("question_field_gate_min_ratio")
    if raw_ratio is not None:
        try:
            policy["min_ratio"] = max(0.0, min(1.0, float(raw_ratio)))
        except (TypeError, ValueError):
            pass
    raw_limit = rp.get("question_field_gate_scan_limit")
    if raw_limit is not None:
        try:
            policy["scan_limit"] = max(1, int(raw_limit))
        except (TypeError, ValueError):
            pass
    raw_warn = rp.get("question_field_warn")
    if raw_warn is not None:
        policy["warn"] = bool(raw_warn)
    if override:
        # 覆盖值同样 clamp/校验（调用方可能来自测试或运行期注入，不能绕过边界）
        if override.get("enabled") is not None:
            policy["enabled"] = bool(override["enabled"])
        if override.get("warn") is not None:
            policy["warn"] = bool(override["warn"])
        if override.get("min_ratio") is not None:
            try:
                policy["min_ratio"] = max(0.0, min(1.0, float(override["min_ratio"])))
            except (TypeError, ValueError):
                pass
        if override.get("scan_limit") is not None:
            try:
                policy["scan_limit"] = max(1, int(override["scan_limit"]))
            except (TypeError, ValueError):
                pass
    return policy


def available_field_set(available_fields: Iterable[str] | None) -> set[str]:
    """把面板列归一成裸列名集合；空 → 空集合（表示未提供可用列信息）。"""
    if not available_fields:
        return set()
    return {str(col).strip().lstrip("$").split("@", 1)[0] for col in available_fields if str(col).strip()}


def _family_loaded(prefix: str, avail: set[str]) -> bool:
    return any(col.startswith(prefix) for col in avail)


def classify_question_fields(
    question: dict[str, Any] | None,
    available: Iterable[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """把题面 ``suggested_fields`` 分成 (可用, 缺失, 无法判定)。

    ``available`` 为空（None/[]）= **未提供可用列信息** → 不做任何判定，返回三个空列表
    （与 ``dsl.core.field_aliases.missing_fields`` 的"空即不判定"契约一致；否则调用方
    会把整题字段误判成缺列）。

    提供可用列时：

    - 真实列（含别名归一，如 ``turnover`` → ``turnover_rate``）按集合命中判定；
    - 列族前缀 token（``funda_ocf``）与散文近义词（``eps``/``crowding``）按族前缀判定；
    - 算子派生面 token（``chip``/``crowding``）视为可用（由行情列现算）；
    - 其余为「无法判定」，不参与合格判据（避免词表不足误杀好题）。
    """
    avail = available_field_set(available)
    hits: list[str] = []
    missing: list[str] = []
    unknown: list[str] = []
    if not avail:
        return hits, missing, unknown
    for raw in (question or {}).get("suggested_fields") or []:
        token = str(raw).strip()
        if not token:
            continue
        canon = canonical_column(token)
        if canon in avail:
            hits.append(token)
            continue
        low = token.lower()
        if low in FIELD_ALIASES:
            # 登记过的 DSL 别名：目标列未载入 → 判为缺列（不是散文词）
            missing.append(token)
            continue
        if low in _OPERATOR_DERIVED_TOKENS:
            hits.append(token)
            continue
        prefix = _FIELD_FAMILY_SYNONYMS.get(low)
        if prefix is None:
            for cand in _COLUMN_FAMILY_PREFIXES:
                if low.startswith(cand):
                    prefix = cand
                    break
        if prefix is None:
            unknown.append(token)
            continue
        if avail and _family_loaded(prefix, avail):
            hits.append(token)
        else:
            missing.append(token)
    return hits, missing, unknown


def _question_fields_eligible(hits: list[str], missing: list[str], min_ratio: float) -> bool:
    """合格判据：可判定字段里至少 1 个可用，且可用占比 ≥ ``min_ratio``。"""
    known = len(hits) + len(missing)
    if known == 0:
        return True  # 题面只有无法判定的散文词 → 不因词表不足误杀
    if not hits:
        return False
    return (len(hits) / known) >= float(min_ratio)


def question_missing_fields(
    question: dict[str, Any] | None,
    available: Iterable[str] | None,
) -> tuple[str, ...]:
    """题面建议字段中本 run 未载入的部分（供任务块提示 / 日志）。"""
    _hits, missing, _unknown = classify_question_fields(question, available)
    return tuple(missing)


def _clean_question_text(text: str, limit: int = 700) -> str:
    """清洗题库文本：去掉表格残渣与 HTML 注释（实测 23% 课题的 construction_guide 是表格块）。

    题库字段由 PDF 解析而来，部分课题的 construction_guide/empirical_findings 直接是
    ``<!-- table pN --> | ... |`` 这类表格残渣，注入题面只会污染复现要求。
    """
    import re

    if not text:
        return ""
    t = re.sub(r"<!--.*?-->", " ", str(text), flags=re.S)
    keep = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.count("|") >= 2 or re.fullmatch(r"[\-|\s:]+", line):
            continue          # 表格行/分隔行
        keep.append(line)
    out = " ".join(" ".join(keep).split())
    if len(out) < 20:
        return ""
    return out[:limit]


_CARDS_CACHE: list[dict] | None = None


def load_mechanism_cards() -> list[dict]:
    """加载机制卡（含 MinerU 语料新抽的卡）；失败返回空表。"""
    global _CARDS_CACHE
    if _CARDS_CACHE is not None:
        return _CARDS_CACHE
    import json
    import subprocess
    from pathlib import Path as _P

    roots = []
    try:
        out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, check=True).stdout.strip()
        if out:
            roots.append(_P(out).parent)
    except Exception:  # noqa: BLE001
        pass
    roots.append(_P(__file__).resolve().parents[4])
    cards: list[dict] = []
    for r in roots:
        f = r / "data" / "research_reports" / "knowledge" / "mechanism_cards.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cards.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
        if cards:
            break
    _CARDS_CACHE = cards
    return cards


def find_card_for_question(question: dict, cards: list[dict] | None = None) -> dict | None:
    """给课题挑最相关的机制卡（标题/机构/年份关键词重合度打分）。"""
    if not question:
        return None
    cards = cards if cards is not None else load_mechanism_cards()
    if not cards:
        return None
    import re as _re

    qtext = f"{question.get('topic') or ''} {question.get('source') or ''} {question.get('hypothesis') or ''}"
    qkeys = {t for t in _re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z_]{3,}", qtext)}
    best, best_score = None, 0.0
    for c in cards:
        src = c.get("source") or {}
        ctext = f"{src.get('title') or ''} {src.get('org') or ''} {c.get('card_id') or ''}"
        ckeys = set(_re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z_]{3,}", ctext))
        score = float(len(qkeys & ckeys))
        if src.get("org") and str(src.get("org")) in qtext:
            score += 3.0
        if score > best_score:
            best, best_score = c, score
    # 阈值放宽到 1.0（2026-09-30 实测：810 张卡时仍频繁 card=-，课题名与卡标题
    # 往往只有 1 个共享实词——例如"趋势选股"vs"趋势因子"）。宁可给一张弱匹配的
    # 参考卡（题面里标注为参考），也比复现轮完全没有机制物料好。
    return best if best_score >= 1.0 else None


def _card_block(card: dict) -> str:
    """机制卡的题面片段（机制三段式 + 字段/参数 + 参考 DSL + 研报实测）。"""
    if not card:
        return ""
    m = card.get("mechanism") or {}
    src = card.get("source") or {}
    lines = [f"- **机制卡** `{card.get('card_id')}`（来源：{src.get('org') or ''}《{src.get('title') or ''}》{src.get('date') or ''}）",
             f"  - 错边方：{m.get('who_wrong') or '-'}",
             f"  - 为何持续：{m.get('why_persists') or '-'}",
             f"  - 可观测：{m.get('observable') or '-'}"]
    fields = card.get("fields") or []
    params = card.get("params") or {}
    if fields:
        lines.append(f"  - 字段：{', '.join(map(str, fields))[:200]}")
    if params:
        lines.append(f"  - 参数：{str(params)[:200]}")
    if card.get("dsl_hint"):
        lines.append(f"  - 参考 DSL（需按本仓字段名改写）：{str(card['dsl_hint'])[:400]}")
    if card.get("formula_text"):
        lines.append(f"  - 研报公式（原文保真）：{str(card['formula_text'])[:400]}")
    ev = card.get("evidence") or {}
    if ev:
        lines.append(f"  - 研报实测：{str(ev)[:200]}")
    return "\n".join(lines)


_DIVERGE_DIMS = ("window", "operator", "field", "neutralize", "gate_shape", "interaction")


def render_diverge_task(question: dict, reproduce_factor: str = "", dims=_DIVERGE_DIMS,
                        parent_detail: str = "", card: dict | None = None) -> str:
    """研报模式的**发散题面**：在已复现课题上做单维变异（Phase 2 / 2026-09-30）。

    之前只有复现轮有题面，发散轮退化成"泛课题 + RAG"，模型不声明父本，也无法保证
    "一次只改一维"。这里把约束前置到题面上。
    """
    if not question:
        return ""
    qid = str(question.get("question_id") or "")
    topic = str(question.get("topic") or "").strip()
    parent = str(reproduce_factor or "").strip() or f"（本课题 `{qid}` 的复现版）"
    parts = [
        "## 研报发散（本轮必须基于复现版做**单维**变异）",
        f"- 课题：{topic}（课题号 {qid}）",
        f"- **父本（复现版）**：`{parent}`",
        (f"- **父本实测（必须超越的基线）**：{parent_detail}"
         " —— 本轮至少要在 IC 或 ICIR 上改善，且 coverage 与换手不得劣化。"
         if parent_detail else "- （父本实测指标缺失，请先重建父本基线再变异）"),
        "- **本轮目标**：让该机制比复现版更强且可交付 —— 争取过 promising 线"
        "（|IC|≥0.02、|ICIR|≥0.28、coverage≥0.85），同时不抬高日换手（≤0.5）、不与库内已有因子撞车；"
        "若某维度让指标变差，明确放弃并换下一个维度，不要反复调同一维。",
        "- **维度轮换建议**（3 轮内覆盖不同维度）：首轮 `window` 或 `operator`，次轮 `neutralize` 或 `field`，末轮 `gate_shape` 或 `interaction`。",
        "- **若母本信号偏弱（|IC|<0.015 或 |ICIR|<0.15）**：前两轮若 window/neutralize 无明显改善，"
        "第三轮直接上结构性杠杆 —— `gate_shape`（硬门→SOFT_GATE/分位分段）或 `interaction`"
        "（`CS_GROUP_RANK`/`DIVERGENCE_RANK` 等条件式结构），它们对弱信号母本的提升通常远大于微调窗长；"
        "仍无改善就如实判定该机制在本池无效，不必反复凑维度。",
        "- 硬约束：",
        f"1. 每个提交/评估必须填 `parent_factor={parent}`，并在 `edit_note` 写明改的维度；",
        "2. **一次只改一个维度**：" + "、".join(f"`{d}`" for d in dims) + "；",
        "3. 不得改变机制语义（那是换题、不是发散）；",
        "4. 参考下方研报 RAG 片段，优先验证「机制在更优参数/结构下是否更稳」，而不是换赛道。",
    ]
    _cb = _card_block(card)
    if _cb:
        parts.append("")
        parts.append("**研报机制卡（变异时的算子/参数依据）**：")
        parts.append(_cb)
    return "\n".join(parts)


def render_reproduce_task(question: dict, spec: dict | None = None, card: dict | None = None,
                          evidence: str = "") -> str:
    """研报模式的**复现题面**（Phase 1）：把课题自带的机制物料变成"必须复现"的任务块。

    物料来自题库字段本身（construction_guide / empirical_findings / suggested_fields /
    suggested_operators / expected_shape / expected_sign / falsifier），不依赖机制卡扩容
    （卡扩容是后续的增强，见 docs/specs/alphaagent_report_reproduce_diverge_spec.md §3）。
    """
    if not question:
        return ""
    qid = str(question.get("question_id") or "").strip()
    topic = str(question.get("topic") or "").strip()
    guide = _clean_question_text(question.get("construction_guide"), 700)
    findings = _clean_question_text(question.get("empirical_findings"), 400)
    fields = question.get("suggested_fields") or []
    ops = question.get("suggested_operators") or []
    shape = question.get("expected_shape") or "-"
    sign = question.get("expected_sign")
    falsifier = " ".join(str(question.get("falsifier") or "").split())[:200]
    parts = [
        "## 研报复现（本轮必须完成；复现不通过则本轮作废）",
        f"- 课题：{topic}（{question.get('source') or ''}，课题号 {qid}）",
        f"- 研报构建思路：{guide or '（题库该字段为表格残渣或缺失——请按课题名与研报机制自行还原，并在 comment 里写明依据）'}",
    ]
    if findings:
        parts.append(f"- 研报实测（对账参考，非硬门）：{findings}")
    if fields:
        parts.append(f"- 建议字段（题库建议，**若与机制或研报原文冲突，以机制/原文为准**）：{', '.join(map(str, fields))}")
    if ops:
        parts.append(f"- 建议算子：{', '.join(map(str, ops))}")
    parts.append(f"- 期望形态：{shape}；期望方向：{sign if sign is not None else '-'}")
    if falsifier:
        parts.append(f"- 证伪条件：{falsifier}")
    _cb = _card_block(card)
    if _cb:
        parts.append(_cb)
    if evidence:
        parts.append("- **研报原文片段（含公式/表格，来自 MinerU 重抽语料；据此落地表达式）**：")
        parts.append("  " + "\n  ".join(str(evidence).splitlines()[:20]))
    parts += [
        "",
        "**硬约束（违反会被直接拒绝，不消耗评估额度）**：",
        f"1. 复现版的 `parent_factor` 必须写成 `reproduce_of:{qid}`（或包含 `{qid}`），"
        "**禁止填自己上一轮的因子名**；",
        "2. 只允许「字段同族替换 + 算子落地」——**不得改变研报的机制语义**；",
        "3. 复现版先用 `train_screen` 评估；通过后（本模式）才进入发散阶段，发散一次只改一个维度"
        "（窗长／算子／同族字段／中性化键／门控形态／交互结构），并填 `parent_factor=<复现版因子名>`。",
    ]
    return "\n".join(parts)


def get_question_for_turn(
    turn: int,
    spec: dict[str, Any] | None = None,
    focus_facets: Iterable[str] | None = None,
    session_id: str | None = None,
    *,
    available_fields: Iterable[str] | None = None,
    gate: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """按轮次与数据面获取具体的研究问题对象。

    ``available_fields`` 非空时启用**字段门控**（默认开，阈值收口在
    ``research_spec.report_policy``，见 ``DEFAULT_QUESTION_FIELD_GATE``）：

    - 题面可判定字段全部落在未载入族上（如自由探索 run 的纯 ``funda_*`` 题）→ 跳过该题，
      继续向后扫描 ``scan_limit`` 道，返回第一道合格题；全不合格 → ``None``（本轮回退为
      「无课题」，不阻塞 run）；
    - 部分缺列 → 保留该题（模型可用可用列代理），缺失字段由 ``question_missing_fields``
      暴露给调用方在任务块里提示；
    - 题面只有无法判定的散文词 → 放行（不因词表不足误杀）。

    ``available_fields`` 未提供 / 为空 / ``gate.enabled=False`` → **完全旧行为**
    （``(offset + turn) % len(pool)`` 单次取值，不扫描）。``stats`` 为可选出参，
    回填 ``{"status": "ok"|"partial"|"skipped", "scanned": int, "missing": [...]}`` 供日志。
    """
    queue = load_question_queue(spec)
    if not queue:
        if stats is not None:
            stats.update({"status": "empty", "scanned": 0, "missing": []})
        return None

    facets = set(focus_facets or ())
    matched = [q for q in queue if set(q.get("facets", [])) & facets] if facets else queue
    candidate_pool = matched if matched else queue

    offset = 0
    if session_id:
        import hashlib
        offset = int(hashlib.md5(str(session_id).encode("utf-8")).hexdigest(), 16)

    policy = resolve_question_field_gate(spec, override=gate)
    avail = available_field_set(available_fields)
    if not policy["enabled"] or not avail:
        picked = candidate_pool[(offset + turn) % len(candidate_pool)]
        if stats is not None:
            stats.update({
                "status": "ungated",
                "scanned": 1,
                "missing": list(question_missing_fields(picked, avail)),
            })
        return picked

    n = len(candidate_pool)
    start = (offset + turn) % n
    limit = max(1, min(int(policy["scan_limit"]), n))
    for step in range(limit):
        q = candidate_pool[(start + step) % n]
        hits, missing, _unknown = classify_question_fields(q, avail)
        if _question_fields_eligible(hits, missing, policy["min_ratio"]):
            if stats is not None:
                stats.update({
                    "status": "partial" if missing else "ok",
                    "scanned": step + 1,
                    "missing": list(missing),
                })
            return q
    if stats is not None:
        stats.update({"status": "skipped", "scanned": limit, "missing": []})
    logger.info("问题队列字段门控：连续 %d 道题均落在未载入字段上，本轮不派课题", limit)
    return None


def get_task_for_turn(
    turn: int,
    spec: dict[str, Any] | None = None,
    focus_facets: Iterable[str] | None = None,
    session_id: str | None = None,
    *,
    available_fields: Iterable[str] | None = None,
    gate: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> str:
    """按当前轮次获取定向研究问题任务指导文本块（结构化深入呈现金工干货）。"""
    _gate_stats: dict[str, Any] = {}
    q = get_question_for_turn(
        turn,
        spec=spec,
        focus_facets=focus_facets,
        session_id=session_id,
        available_fields=available_fields,
        gate=gate,
        stats=_gate_stats,
    )
    if stats is not None:
        stats.update(_gate_stats)
    if not q:
        return ""

    return render_question_task(
        q,
        missing_fields=_gate_stats.get("missing") or [],
        spec=spec,
        gate=gate,
    )


def render_question_task(
    question: dict[str, Any],
    *,
    missing_fields: Iterable[str] | None = None,
    spec: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
) -> str:
    """渲染给定课题的任务块（调用方已选定课题时复用同一次门控结果）。

    ``missing_fields`` 仅在 ``report_policy.question_field_warn`` 为真时注入提示行。
    """
    policy = resolve_question_field_gate(spec, override=gate)
    return _render_question_task(
        question,
        missing_fields=(missing_fields or []) if policy["warn"] else [],
    )


_ORTHOGONAL_COMPLEMENTS: dict[str, list[dict[str, Any]]] = {
    "价量面": [
        {"facet": "量能面", "fields": ["amount", "float_cap"], "technique": "换手率或成交额连续加权（SOFT_GATE 或 CS_GROUP_RANK），有效压低周度换手率并剔除虚假突破"},
        {"facet": "资金面", "fields": ["ff_large_net", "ff_super_net"], "technique": "主力大单流向残差化或门控（SOFT_GATE），验证是否为机构真金白银建仓而非散户跟风"},
        {"facet": "基本面", "fields": ["funda_net_profit", "pe", "pb"], "technique": "盈利质量或估值分层中性化（CS_RESIDUALIZE），剥离纯技术形态在垃圾股上的假突破"},
    ],
    "量能面": [
        {"facet": "价量面", "fields": ["close", "adj_vwap", "high", "low"], "technique": "价格偏离或高阶矩背离（DIVERGENCE_RANK），识别放量滞涨或缩量企稳"},
        {"facet": "筹码面", "fields": ["chip", "volume"], "technique": "筹码峰套牢密集度（CHIP_DENSITY 或 CHIP_BIMODAL），度量换手过程中的套牢盘抛压"},
    ],
    "资金面": [
        {"facet": "基本面", "fields": ["funda_ocf", "funda_revenue"], "technique": "现金流与成长质量过滤（SOFT_GATE），避免追随游资炒作无业绩支撑标的"},
        {"facet": "价量面", "fields": ["ret", "adj_close"], "technique": "收益率动量残差（CS_RESIDUALIZE），剥离大单对短期价格的同步冲击影响"},
    ],
    "基本面": [
        {"facet": "价量面", "fields": ["adj_close", "volume"], "technique": "短期价格反转或动量启动（SOFT_GATE），寻找业绩优异但价格处于左侧错杀阶段的拐点"},
        {"facet": "量能面", "fields": ["amount", "float_cap"], "technique": "换手率流动性分层（CS_GROUP_RANK），在机构关注度适中的股票中捕捉阿尔法"},
    ],
    "股东面": [
        {"facet": "价量面", "fields": ["adj_close", "ret"], "technique": "价格突破或均线偏离背离（DIVERGENCE_RANK），确认股东户数骤降（筹码集中）时股价是否处于起爆前夕"},
    ],
    "事件面": [
        {"facet": "资金面", "fields": ["ff_main_net", "ff_large_net"], "technique": "主力资金同步流入确认（SOFT_GATE），验证事件公告后机构是否真实抢筹"},
    ],
    "拥挤面": [
        {"facet": "量能面", "fields": ["turnover_rate", "amount"], "technique": "换手率过热衰减连续调节（SOFT_GATE），度量拥挤交易下的多头踩踏风险"},
    ],
}


def _select_complementary_facet(facets: Iterable[str]) -> dict[str, Any]:
    """为当前课题推荐一个原课题没有的正交/互补数据面与落地技巧。"""
    facet_set = set(facets or ())
    for f in facet_set:
        if f in _ORTHOGONAL_COMPLEMENTS:
            for cand in _ORTHOGONAL_COMPLEMENTS[f]:
                if cand["facet"] not in facet_set:
                    return cand
    return {"facet": "量能面", "fields": ["amount", "float_cap"], "technique": "换手率平滑连续调节（SOFT_GATE），压低周度换手并增强稳定性"}


def _render_question_task(
    q: dict[str, Any],
    *,
    missing_fields: Iterable[str] | None = None,
) -> str:
    """渲染定向课题任务块（``missing_fields`` 非空时追加「本 run 未载入」提示行）。"""
    facets = q.get("facets", [])
    comp = _select_complementary_facet(facets)

    const_guide = q.get("construction_guide")
    emp_find = q.get("empirical_findings")
    ortho_sol = q.get("orthogonal_solution")
    pitfall = q.get("falsifier_and_pitfalls") or q.get("falsifier")

    lines = [
        "### 本轮研报定向攻关课题【RDAgent 研究问题驱动】",
        f"- **课题编号**：`{q.get('question_id')}` —— {q.get('topic')}",
        f"- **文献来源**：{q.get('source')}",
    ]

    if const_guide and emp_find:
        lines.extend([
            f"- **① 因子构建细节**：{const_guide}",
            f"- **② 研报实证结论**：{emp_find}",
            f"- **③ 风险因子残差方案**：{ortho_sol}",
            f"- **④ 避坑陷阱与衰减特征**：⚠️ {pitfall}",
        ])
    else:
        lines.extend([
            f"- **主机制假设**：{q.get('hypothesis')}",
            f"- **潜在证伪陷阱**：⚠️ {q.get('falsifier')}",
        ])

    lines.extend([
        f"- **建议基础字段**：{', '.join(q.get('suggested_fields', []))}",
        f"- **推荐算子构想**：{', '.join(q.get('suggested_operators', []))}",
        f"- **预期检验形态**：形态={q.get('expected_shape')}，符号={q.get('expected_sign')}",
    ])
    missing = [str(f) for f in (missing_fields or []) if str(f).strip()]
    if missing:
        lines.append(
            "- ⚠ **字段可用性**：本 run 面板未载入 "
            + "、".join(f"`${f}`" if not f.startswith("$") else f"`{f}`" for f in missing)
            + "（题面建议字段）。引用它们会被工具层拦截；请用**已载入的同族/代理列**"
            "（如换手率用 `DIVIDE($amount, $float_cap)`）落地本条假设，或跳过该字段。"
        )
    lines.extend([
        "",
        "#### 💡 推荐正交补充面（跨面融合·破除同质化与降换手）",
        f"- **推荐引入的新面**：【{comp['facet']}】（建议字段: {', '.join(comp['fields'])}）",
        f"- **融合实现建议**：{comp['technique']}。",
        "- **融合提示**：严禁 GATED_SIGNAL(neutral=0) 置零门控，优先使用连续加权算子 `SOFT_GATE(signal, state, strength=0.5)` 或组内排序 `CS_GROUP_RANK`！",
        "",
        "**任务要求**：请针对上述文献假设，设计基础信号并尝试融合上述正交调节面，严格执行 `prediction` 预测对账！"
    ])
    return "\n".join(lines)

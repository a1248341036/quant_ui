# -*- coding: utf-8 -*-
"""模块 · question_queue：研报研究问题队列驱动器（R4 方案，RDAgent 范式）。

将金工研报文献转化为可证伪的“结构化研究问题清单”。
在每轮挖掘启动时，按轮次向 LLM 下达具体的定向攻关任务，驱动 LLM 从
“盲目自由搜索”收敛到“研报假设验证与衍生对账”闭环。
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
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
                return _apply_require_report_structure(
                    _apply_require_factor_records(questions, spec), spec)
        except Exception as e:
            logger.warning("读取问题题库失败: %s", e)

    return list(DEFAULT_RESEARCH_QUESTIONS)


_QUESTION_INDEX_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def _apply_require_report_structure(rows: list[dict[str, Any]],
                                    spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``report_policy.require_report_structure``：只派发**报告提出了可复现结构**的课题。

    结构判定由抽取侧 LLM 产出（`report_structure` / `has_reproducible_structure` /
    `reproduction_target`），本函数只做准入过滤，不重判。
    兜底：题库无该字段（旧题库）→ 不筛；过滤后为空 → 回退原题库（宁可不筛也不让 run 无题）。
    """
    if not rows:
        return rows
    if not bool(((spec or {}).get("report_policy") or {}).get("require_report_structure", False)):
        return rows
    if not any("has_reproducible_structure" in r for r in rows):
        logger.info("require_report_structure=True 但题库无该字段（旧题库），不筛")
        return rows
    kept = [r for r in rows if r.get("has_reproducible_structure") is True]
    if not kept:
        logger.warning("require_report_structure 过滤后无课题，回退原题库（%d 道）", len(rows))
        return rows
    logger.info("require_report_structure: %d → %d 道（仅保留有可复现结构的研报课题）",
                len(rows), len(kept))
    return kept


def find_question(question_id: str, spec: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """按 qid 取课题（按文件 mtime 缓存的索引），供复现保真度校验使用。

    不走 ``load_question_queue``（后者每次读全文件且可能被 require_factor_records 过滤）。
    """
    path = resolve_questions_file(spec)
    if not path or not path.is_file():
        return None
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    cached = _QUESTION_INDEX_CACHE.get(key)
    if cached is None or cached[0] != stamp:
        index: dict[str, dict[str, Any]] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    q = json.loads(line)
                    qid = str(q.get("question_id") or "")
                    if qid:
                        index[qid] = q
        except Exception as e:  # noqa: BLE001
            logger.warning("构建课题索引失败: %s", e)
            return None
        cached = (stamp, index)
        _QUESTION_INDEX_CACHE[key] = cached
    return cached[1].get(str(question_id or ""))


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


_THEORY_RE_SRC = (
    r"定价公式|无套利|风险溢价|引理|定理|证明|推导|式（\d|式\(\d|假设检验的统计量|"
    r"参考文献|附录|本文结构|文献综述"
)


def _clean_question_text(text: str, limit: int = 700) -> str:
    """清洗题库文本：去掉表格残渣与 HTML 注释（实测 23% 课题的 construction_guide 是表格块）。

    题库字段由 PDF 解析而来，部分课题的 construction_guide/empirical_findings 直接是
    ``<!-- table pN --> | ... |`` 这类表格残渣，注入题面只会污染复现要求。
    """
    import re

    if not text:
        return ""
    t = re.sub(r"<!--.*?-->", " ", str(text), flags=re.S)
    theory = re.compile(_THEORY_RE_SRC)
    keep = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.count("|") >= 2 or re.fullmatch(r"[\-|\s:]+", line):
            continue          # 表格行/分隔行
        if theory.search(line):
            continue          # 论文/理论推导段（实测 construction_guide 常抓到 APT 定价公式这类内容）
        keep.append(line)
    out = " ".join(" ".join(keep).split())
    if len(out) < 20:
        return ""
    return out[:limit]


_CARDS_CACHE: list[dict] | None = None
# 空结果退避（秒）：见 load_mechanism_cards——避免文件缺失时每轮都起 git 子进程扫盘
_CARDS_MISS_AT: float = 0.0
_CARDS_MISS_TTL: float = 300.0


def load_mechanism_cards() -> list[dict]:
    """加载机制卡（含 MinerU 语料新抽的卡）；失败返回空表。"""
    global _CARDS_CACHE, _CARDS_MISS_AT
    # 只在缓存**非空**时短路：空表不固化，否则"首次调用时文件不存在"会把 [] 缓存住，
    # 本进程后续所有调用永远拿不到机制卡（2026-09-30 review 修）。
    if _CARDS_CACHE:
        return _CARDS_CACHE
    # 空结果退避：find_card_for_question 每个 outer turn 都会调本函数，若文件确实缺失/
    # 不可解析，无退避就会**每轮起一个 git 子进程**扫盘（第二轮 review 修）。
    if _CARDS_MISS_AT and time.time() - _CARDS_MISS_AT < _CARDS_MISS_TTL:
        return []
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
    # 只缓存非空结果（见上方短路注释）：空表进退避窗口，窗口过后再重试。
    if cards:
        _CARDS_CACHE = cards
        _CARDS_MISS_AT = 0.0
    else:
        _CARDS_MISS_AT = time.time()
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
    # 阈值回到 2.0（2026-09-30 复盘：1.0 会错配——RQ_020「Alpha 因子库精简」被配上
    # 《动态情景多因子 alpha 模型》的卡并带出误导性参考 DSL；错配比没有卡更糟）。
    # 例外：标题与课题高度重合（bigram Jaccard ≥ 0.34）时即使只共享 1 个实词也放行。
    # 硬门槛：卡标题里要有 ≥4 字的中文片段真的出现在课题文本里（否则就是错配——
    # 2026-09-30 复盘：仅靠"同机构+alpha"就能凑够分数，把《动态情景多因子alpha模型》
    # 配给了"Alpha 因子库精简与优化"。宁可不给卡，也不给误导性机制/参考 DSL。）
    if not best:
        return None
    import re as _re

    title = str((best.get("source") or {}).get("title") or "")
    segs = [t for t in _re.findall(r"[\u4e00-\u9fa5]{4,}", title)]
    # 硬门槛也要求有基本分（best_score > 0）：否则任何一篇标题里含"多因子模型"这类
    # 通用词的卡，只要该词出现在课题文本里就会被配上，绕过下面的分数阈值
    # （2026-09-30 review 修；与本块"宁可不给卡，也不给误导性机制"的意图对齐）。
    # 注：标题若没有 ≥4 字中文片段（如纯英文标题），segs 为空 → 本条件恒 False，
    # 该卡只能走下面的分数阈值（`best_score >= 2.0 且标题相似度 >= 0.35`）——这是
    # 有意为之的更严口径（第二轮 review 澄清）。
    if best_score > 0 and any(seg in qtext for seg in segs):
        return best
    if best_score >= 2.0 and _title_similar(best, qtext) >= 0.35:
        return best
    return None


def _bigram_jaccard(a: str, b: str) -> float:
    """两段文本的字符二元组 Jaccard（<=6 字符不予判定，返回 0）。"""
    import re as _re

    clean = lambda t: _re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", str(t))  # noqa: E731
    a, b = clean(a), clean(b)
    if len(a) < 6 or len(b) < 6:
        return 0.0
    ba = {a[i:i + 2] for i in range(len(a) - 1)}
    bb = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(ba & bb) / max(1, len(ba | bb))


def _title_similar(card: dict, question_text: str) -> float:
    """卡标题与课题文本的字符二元组 Jaccard（判"是否同一篇报告"）。"""
    title = str((card.get("source") or {}).get("title") or "")
    return _bigram_jaccard(title, question_text)


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


# ── 研报因子清单注入（spec §7：docs/specs/alphaagent_report_factor_records_spec.md）──
# 只读抽取侧产物（factor_records_pilot.jsonl / report_freq_verified.jsonl）并渲染题面块，
# **不参与判定与状态机**：题面块让模型照抄研报公式落地，判不判定仍由既有闸门决定。
FACTOR_RECORDS_FILE = "factor_records_pilot.jsonl"
REPORT_FREQ_FILE = "report_freq_verified.jsonl"
FACTOR_RECORDS_BLOCK_TITLE = "本报告因子清单"
FACTOR_RECORDS_MAX_ITEMS = 10
FACTOR_RECORDS_NO_FORMULA_NOTE = "本报告未抽出可执行公式"
_CONF_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}
_FACTOR_RECORDS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_REPORT_FREQ_CACHE: dict[str, tuple[float, dict[str, dict[str, Any]]]] = {}


def _knowledge_roots() -> list[Path]:
    """知识库候选根目录（git common dir 优先，兼容 worktree 与主仓库路径）。"""
    roots: list[Path] = []
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            roots.append(Path(out).parent)
    except Exception:  # noqa: BLE001
        pass
    roots.append(Path(__file__).resolve().parents[4])
    return roots


def _resolve_knowledge_file(name: str, override: str = "") -> Path | None:
    """解析 ``data/research_reports/knowledge/<name>``；``override`` 优先（测试/实验用）。"""
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.is_file():
            return p
    for root in _knowledge_roots():
        p = root / "data" / "research_reports" / "knowledge" / name
        if p.is_file():
            return p
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_factor_records(spec: dict[str, Any] | None = None,
                        path: Path | str | None = None) -> list[dict[str, Any]]:
    """读取研报因子清单 jsonl（按 mtime 缓存）；缺文件 → 空表（题面不加块）。

    ``report_policy.factor_records_file`` 可覆盖路径（与 ``question_queue_file`` 同风格）。
    """
    if path is None:
        override = str(((spec or {}).get("report_policy") or {}).get("factor_records_file") or "")
        path = _resolve_knowledge_file(FACTOR_RECORDS_FILE, override)
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    try:
        stamp = p.stat().st_mtime
    except OSError:
        return []
    cached = _FACTOR_RECORDS_CACHE.get(str(p))
    if cached and cached[0] == stamp:
        return cached[1]
    rows = _read_jsonl(p)
    _FACTOR_RECORDS_CACHE[str(p)] = (stamp, rows)
    return rows


# ── 复现保真度「原文锚」（2026-10-01，**仅研报模式**）────────────────────
# 背景（昨夜实测）：有原文公式的 10 个课题里，复现公式与原文的**字段 Jaccard 平均
# 0.062、算子 Jaccard 0.085**（多组为 0）——模型只靠题面约束"脱稿自由探索"，
# 用容易过训练线的通用因子（5 日反转 / Amihud 非流动性 / PE 动量）顶替原文机制
# （K 线最短路径 / 60 日反转择时 / 一致预期字段）。而复现是否"过线"原本只看统计门槛，
# 不校验保真度 → 缺少"以原文为锚"的硬约束。此处在复现过线判定前加机械校验。
# 仅研报模式生效：配置放在 ``report_policy`` 下，且调用点只在 report gate 内。
_REPRODUCE_FIDELITY_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "require_shared_field": 1,   # 至少命中 1 个原文公式用到的字段（0=不要求）
    "min_field_jaccard": 0.0,    # 可选：字段 Jaccard 下限
    "min_op_jaccard": 0.0,       # 可选：算子 Jaccard 下限
    "min_shared_ops": 0,         # 可选：最少共享算子数
}

_FID_FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_FID_OP_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\(")

# 「通用行情/风格字段」：几乎所有报告与因子都会用到，**不能算作锚定证据**
# （否则"脱稿"因子只要用了 $adj_close 就能蒙过 require_shared_field）。
# 只影响 require_shared_field 的判定；Jaccard 仍按全量字段集计算。
_FID_GENERIC_FIELDS: frozenset[str] = frozenset({
    "adj_close", "close", "open", "high", "low", "vwap", "adj_vwap",
    "volume", "amount", "turnover_rate", "turnover", "float_cap",
    "total_cap", "adj_factor", "industry_sw_l1", "industry_sw_l2",
    "ret", "pct_chg", "pre_close",
})


def _fid_sets(expr: str) -> tuple[set[str], set[str]]:
    return set(_FID_FIELD_RE.findall(expr or "")), set(_FID_OP_RE.findall(expr or ""))


def resolve_reproduce_fidelity(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """保真度配置（``report_policy.reproduce_fidelity``），缺省见上方常量。"""
    cfg = dict(_REPRODUCE_FIDELITY_DEFAULTS)
    raw = ((spec or {}).get("report_policy") or {}).get("reproduce_fidelity")
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in _REPRODUCE_FIDELITY_DEFAULTS:
                cfg[k] = v
    return cfg


def check_reproduce_fidelity(question: dict[str, Any] | None, expr: str, *,
                             spec: dict[str, Any] | None = None,
                             records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """复现公式 vs 该课题报告的原文公式：字段/算子重叠的**机械校验**。

    返回 ``{passed, reason, n_records, field_jaccard, op_jaccard, shared_fields,
    shared_ops, ref_fields, thresholds}``。

    ``reason="no_reference"``：该报告不在抽取覆盖集内（无原文公式可比）→ **不因缺参照卡死**
    （是否只派发有公式的课题由 ``report_policy.require_factor_records`` 决定）。
    """
    cfg = resolve_reproduce_fidelity(spec)
    if records is None:
        records = load_factor_records(spec)
    matched = match_report_records(question, None, records)
    if isinstance(matched, tuple):
        matched = matched[0]
    refs = [r for r in (matched or []) if isinstance(r, dict)]
    if not refs:
        return {"passed": True, "reason": "no_reference", "n_records": 0,
                "field_jaccard": None, "op_jaccard": None,
                "shared_fields": 0, "shared_ops": 0, "ref_fields": [],
                "thresholds": cfg}

    ref_fields: set[str] = set()
    ref_ops: set[str] = set()
    for r in refs:
        f, o = _fid_sets(str(r.get("expr_local") or r.get("expr_raw") or ""))
        ref_fields |= f
        ref_ops |= o
    my_fields, my_ops = _fid_sets(expr)
    shared_f = ref_fields & my_fields
    shared_o = ref_ops & my_ops
    # 锚定证据：排除通用行情字段后的"报告特有字段"交集
    ref_specific = ref_fields - _FID_GENERIC_FIELDS
    shared_specific = shared_f - _FID_GENERIC_FIELDS
    # 边界兜底（2026-10-01）：部分报告的原文公式**全用通用行情字段**——实测 361 篇覆盖
    # 报告里 83 篇（23%），如"纯 VWAP/close 乖离""K 线高低点"类。若一律要求命中特有字段，
    # 这些课题下的**忠实复现也永远过不了**（分母本身就没有特有字段）。故退化为按全量字段
    # 判定，并在结果里标明 anchor_mode 以便审计。
    anchor_mode = "specific" if ref_specific else "generic_only_reference"
    anchor_pool = shared_specific if ref_specific else shared_f
    fj = len(shared_f) / max(1, len(ref_fields | my_fields))
    oj = len(shared_o) / max(1, len(ref_ops | my_ops))

    need_f = int(cfg.get("require_shared_field") or 0)
    need_ops = int(cfg.get("min_shared_ops") or 0)
    min_fj = float(cfg.get("min_field_jaccard") or 0.0)
    min_oj = float(cfg.get("min_op_jaccard") or 0.0)

    # ── 结构硬锚（2026-10-02，仅研报模式且题面带结构信息时生效）──────────────
    # 依据（巡检实测）：题面要求"5 因子夏普率加权合成"，模型却自行换成"上下行波动分解"
    # —— 结构要求只在题面=软约束。此处做成**机械校验**（规则只做核对，结构判定仍来自 LLM 抽取）：
    #   ① 声明了 structure_ops → 复现须命中其中 ≥1 个结构性算子；
    #   ② 否则按声明字段：须用到 ≥ min(声明字段数, min_fields) 个；
    #   ③ 无结构信息（旧题库）→ applicable=False，不判（向后兼容）。
    fails: list[str] = []
    _sr = ((question or {}).get("primary") or {}).get("spec_requirements") or {}
    _struct_ops = [str(o) for o in (_sr.get("structure_ops") or []) if str(o).strip()] \
        if isinstance(_sr, dict) else []
    _declared = {str(f).lstrip("$") for f in ((_sr.get("fields") or []) if isinstance(_sr, dict) else [])
                 if str(f).strip()}
    try:
        _min_fields = int((_sr or {}).get("min_fields") or 0)
    except (TypeError, ValueError):
        _min_fields = 0
    structure_anchor: dict[str, Any] = {"applicable": False}
    if _struct_ops:
        _hit_ops = sorted(set(_struct_ops) & my_ops)
        structure_anchor = {"applicable": True, "mode": "structure_ops", "need": _struct_ops,
                            "hit": _hit_ops, "declared_fields": sorted(_declared),
                            "min_fields": _min_fields}
        if not _hit_ops:
            fails.append("structure_ops_missing=" + ",".join(_struct_ops[:4]))
    elif _declared or _min_fields > 0:
        # 声明字段覆盖率：须命中 ≥ min(声明字段数, min_fields) 个声明字段
        _need_n = (min(len(_declared), _min_fields) if _min_fields else len(_declared))
        _hit_f = sorted(_declared & my_fields)
        # 巡检实测（2026-10-02）：部分题声明 fields=[$close,$volume] 但 min_fields=5
        # （抽取时只填了"参考公式的字段"，而非"结构所需字段"）→ 只查声明字段会让锚形同虚设。
        # 故补一条：声明字段少于 min_fields 时，额外要求**复现用到的字段总数 ≥ min_fields**。
        _need_total = _min_fields if _min_fields > len(_declared) else 0
        structure_anchor = {"applicable": True, "mode": "fields", "need_n": _need_n,
                            "hit": _hit_f, "need_total_fields": _need_total,
                            "n_my_fields": len(my_fields),
                            "declared_fields": sorted(_declared), "min_fields": _min_fields}
        if _need_n and len(_hit_f) < _need_n:
            fails.append(f"declared_fields_hit={len(_hit_f)}<{_need_n}")
        if _need_total and len(my_fields) < _need_total:
            fails.append(f"fields_used={len(my_fields)}<min_fields={_need_total}")

    if need_f > 0 and len(anchor_pool) < need_f:
        fails.append(f"shared_{'specific_' if ref_specific else ''}fields="
                     f"{len(anchor_pool)}<{need_f}")
    if need_ops > 0 and len(shared_o) < need_ops:
        fails.append(f"shared_ops={len(shared_o)}<{need_ops}")
    if min_fj > 0 and fj < min_fj:
        fails.append(f"field_jaccard={fj:.3f}<{min_fj:.3f}")
    if min_oj > 0 and oj < min_oj:
        fails.append(f"op_jaccard={oj:.3f}<{min_oj:.3f}")

    return {
        "passed": not fails,
        "reason": "ok" if not fails else "off_reference:" + ",".join(fails),
        "n_records": len(refs),
        "anchor_mode": anchor_mode,
        "field_jaccard": round(fj, 4),
        "op_jaccard": round(oj, 4),
        "shared_fields": sorted(shared_f)[:8],
        "shared_specific_fields": sorted(shared_specific)[:8],
        "shared_ops": sorted(shared_o)[:8],
        "ref_fields": sorted(ref_fields)[:12],
        "ref_specific_fields": sorted(ref_specific)[:12],
        "thresholds": cfg,
        "structure_anchor": structure_anchor,
    }


def _question_has_records(question: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    matched = match_report_records(question, None, records)
    if isinstance(matched, tuple):
        matched = matched[0]
    return any(isinstance(r, dict) for r in (matched or []))


def _apply_require_factor_records(rows: list[dict[str, Any]],
                                  spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """``report_policy.require_factor_records``：只派发**有原文公式**的课题。

    用户决策（2026-10-01）：先用有公式的研报，保证复现题面一定带原文锚。
    兜底：清单为空或过滤后为空 → 原样返回（宁可不筛，也不要让 run 无题可做）。
    """
    if not rows:
        return rows
    if not bool(((spec or {}).get("report_policy") or {}).get("require_factor_records", False)):
        return rows
    records = load_factor_records(spec)
    if not records:
        logger.warning("require_factor_records=True 但因子清单为空，保持原题库")
        return rows
    kept = [q for q in rows if _question_has_records(q, records)]
    if not kept:
        logger.warning("require_factor_records 过滤后无课题，回退原题库（%d 道）", len(rows))
        return rows
    logger.info("require_factor_records: %d → %d 道（仅保留有原文公式的研报课题）",
                len(rows), len(kept))
    return kept


def load_report_freq_index(spec: dict[str, Any] | None = None,
                           path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """报告级**已确认**频率索引（key = 归一化 source），来自 ``report_freq_verified.jsonl``。"""
    if path is None:
        override = str(((spec or {}).get("report_policy") or {}).get("report_freq_file") or "")
        path = _resolve_knowledge_file(REPORT_FREQ_FILE, override)
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        stamp = p.stat().st_mtime
    except OSError:
        return {}
    cached = _REPORT_FREQ_CACHE.get(str(p))
    if cached and cached[0] == stamp:
        return cached[1]
    index = {_norm_source(r.get("source")): r for r in _read_jsonl(p) if r.get("source")}
    _REPORT_FREQ_CACHE[str(p)] = (stamp, index)
    return index


def _norm_source(src: Any) -> str:
    """报表 source 路径归一（分隔符/大小写/前导 ./），用于卡 path ↔ 记录 source 对齐。"""
    return str(src or "").replace("\\", "/").strip().lstrip("./").lower()


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100}


def _cn2int(text: str) -> int | None:
    """中文数字 → 整数（覆盖 一~九百九十九；纯数字原样）；无法解析返回 None。"""
    s = str(text or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total, num = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            return None
    return total + num


def _series_no(text: str) -> int | None:
    """抽取「之<数字>」系列号（`之十八`→18）；无系列号 → None。

    用于回退匹配的**同系列姊妹篇防误配**：券商研报常成系列（如东方「因子选股系列研究」
    上百篇），同系列标题共享大量 boilerplate（机构名 + 金融工程 + 系列名），字符 bigram
    Jaccard 极易越过 0.35 阈值。2026-10-01 整夜实测：19 个"命中"里约 8 个是假阳性，
    例如课题 `之十三 alpha预测` 被对到 `之十八 在alpha衰退之前`，**注入的是另一篇报告的
    公式**（比不注入更有害）。
    """
    import re

    m = re.search(r"之([零一二三四五六七八九十百\d]+)", str(text or ""))
    return _cn2int(m.group(1)) if m else None


def _paren_no(text: str) -> int | None:
    """抽取括号序号（`（三）`/`(四)`/`（2）`）→ int；无 → None。

    同系列还有用括号编号的写法（如海通「因子投资与smartbeta研究（三）/（四）」），
    `之N` 守卫覆盖不到 —— 2026-10-01 实测残留假阳性：课题 `smartbeta研究（四）单因子多组合`
    被对到 `smartbeta研究(三)_市场环境与因子组合表现`。
    """
    import re

    m = re.search(r"[（(]([零一二三四五六七八九十百\d]{1,4})[）)]", str(text or ""))
    return _cn2int(m.group(1)) if m else None


def match_report_records(question: dict[str, Any] | None, card: dict[str, Any] | None,
                         records: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], str]:
    """把当前课题对到**某一篇报告**的因子记录，返回 (记录, 该报告 source)。

    对齐顺序：① 机制卡 ``source.path`` 与记录 ``source`` 归一后精确相等（试点 10 篇里
    9 篇有卡可精确对上）；② 卡缺失/不匹配时退化为报告标题与课题 source 的字符二元组
    Jaccard（门槛 0.35，与机制卡匹配同源）。对不上 → ``([], "")``（宁可不注入，不猜）。
    """
    if not records:
        return [], ""
    card_path = _norm_source(((card or {}).get("source") or {}).get("path"))
    if card_path:
        hits = [r for r in records if _norm_source(r.get("source")) == card_path]
        if hits:
            return hits, str(hits[0].get("source") or "")
    texts = [str((question or {}).get("source") or ""),
             str(((card or {}).get("source") or {}).get("title") or "")]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return [], ""
    # 系列号守卫：课题与候选报告都带序号（之N 或 括号序号）且不一致 → 视为不同报告直接跳过
    def _markers(text: str) -> tuple[int | None, int | None]:
        return _series_no(text), _paren_no(text)

    q_marks = [_markers(t) for t in texts]
    best_src, best_score = "", 0.0
    for src in {str(r.get("source") or "") for r in records if r.get("source")}:
        stem = Path(src).stem
        r_marks = _markers(stem)
        conflict = any(
            qm[i] is not None and r_marks[i] is not None and qm[i] != r_marks[i]
            for qm in q_marks for i in (0, 1)
        )
        if conflict:
            continue
        score = max(_bigram_jaccard(stem, t) for t in texts)
        if score > best_score:
            best_src, best_score = src, score
    if not best_src or best_score < 0.35:
        return [], ""
    return [r for r in records if str(r.get("source") or "") == best_src], best_src


def _record_rank(rec: dict[str, Any]) -> tuple[int, int]:
    """注入排序键：研报原文公式（verbatim）优先，其次文字推导（derived）；再按置信度。"""
    kind = str(rec.get("formula_kind") or "")
    return (0 if kind == "verbatim" else 1, _CONF_ORDER.get(str(rec.get("confidence") or ""), 3))


def select_injectable_records(
    records: list[dict[str, Any]] | None,
    *,
    available_fields: Iterable[str] | None = None,
    spec: dict[str, Any] | None = None,
    max_items: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """筛出可注入记录：``executable=true`` → 字段可用性门控 → 排序 → 截断。

    字段门控**复用课题门控的既有机制**（``classify_question_fields``：别名归一 → 精确命中 →
    列族前缀判定），``available_fields`` 为空表示未提供可用列信息 → 不判定（与课题门控
    "空即不判定"一致）。

    与课题门控的唯一差别：记录字段是抽取侧**已按本仓列名校验过**的（``fields_local``），
    因此"无法判定"（既非已知列、又无列族前缀）在这里等价于**未载入**——而 ``expr_local``
    是要逐字进 DSL 评估的，缺一个字段就是硬失败，故按**零缺列**过滤，比课题的
    ``min_ratio`` 严；否则题面会承诺本 run 跑不通的表达式。
    返回 (入选记录, 被门控剔除的字段)。
    """
    policy = resolve_question_field_gate(spec)
    avail = available_field_set(available_fields)
    limit = int(FACTOR_RECORDS_MAX_ITEMS if max_items is None else max_items)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for rec in sorted([r for r in (records or [])
                       if r.get("executable") and str(r.get("expr_local") or "").strip()],
                      key=_record_rank):
        fields = [str(f) for f in (rec.get("fields_local") or [])]
        if policy["enabled"] and avail and fields:
            _hits, missing, unknown = classify_question_fields({"suggested_fields": fields}, avail)
            miss = list(missing) + list(unknown)
            if miss:
                dropped += miss
                continue
        kept.append(rec)
    return kept[:max(1, limit)], dropped


def _record_freq_text(rec: dict[str, Any], freq_index: dict[str, dict[str, Any]] | None) -> str:
    """频率标签建议：以 report_freq_verified.jsonl 为准；未确认 → "研报未说明频率"。"""
    row = (freq_index or {}).get(_norm_source(rec.get("source"))) or {}
    freq = row.get("freq_final") or row.get("rebalance_freq")
    label = row.get("label_final") or row.get("label_col_hint")
    if not freq and str(rec.get("freq_source") or "") == "report_index":
        freq = rec.get("eval_freq_hint")
        label = rec.get("label_col_hint")
    if not freq:
        return "研报未说明频率"
    return f"{freq}" + (f" / label={label}" if label else "") + "（report_freq_verified）"


def _record_tree_text(rec: dict[str, Any], limit: int = 220) -> str:
    """算子树压成单行（多行缩进 → 空格折叠）；缺树退化到 ``ops_used`` 列表。"""
    tree = " ".join(str(rec.get("op_tree") or "").split())
    if not tree:
        tree = " → ".join(str(o) for o in (rec.get("ops_used") or [])) or "-"
    if len(tree) > limit:
        tree = tree[:limit] + "…"
    fams = "/".join(str(f) for f in (rec.get("op_families") or [])) or "-"
    return f"{tree}（算子族：{fams}）"


def render_factor_records_block(
    records: list[dict[str, Any]] | None,
    *,
    available_fields: Iterable[str] | None = None,
    spec: dict[str, Any] | None = None,
    freq_index: dict[str, dict[str, Any]] | None = None,
    question: dict[str, Any] | None = None,
    report_source: str = "",
    max_items: int | None = None,
) -> str:
    """渲染"本报告因子清单"题面块（spec §7）。

    - 只列 ``executable=true`` 且字段在本 run 载入的记录，上限 ``max_items``（默认 10）；
    - 每条给 name / expr_local / direction / formula_kind（verbatim="研报原文公式"，
      derived="⚠ 由文字定义推导，待验证"）/ 数据面 focus_facets / 算子树 op_tree / 频率标签；
    - 全部被字段门控剔除或没有可执行记录 → 只返回一行说明（调用方据此省略该块）。
    """
    policy = (spec or {}).get("report_policy") or {}
    if max_items is None:
        try:
            max_items = int(policy.get("factor_records_max_items", FACTOR_RECORDS_MAX_ITEMS))
        except (TypeError, ValueError):
            max_items = FACTOR_RECORDS_MAX_ITEMS
    picked, dropped = select_injectable_records(
        records, available_fields=available_fields, spec=spec, max_items=max_items,
    )
    if not picked:
        uniq = list(dict.fromkeys(dropped))
        why = f"（可执行公式涉及的字段本 run 未载入：{'、'.join(uniq[:6])}）" if uniq else ""
        return f"- ⚠ {FACTOR_RECORDS_NO_FORMULA_NOTE}{why}；请按上方机制卡与研报原文自行落地。"
    src = str(report_source or (picked[0].get("source") or ""))
    qid = str((question or {}).get("question_id") or "-")
    lines = [
        f"## {FACTOR_RECORDS_BLOCK_TITLE}（抽取自研报表格，仅列本 run 可执行公式；最多 {max_items} 条）",
        f"- 来源报告：{Path(src).stem if src else '-'}（课题 {qid}）；来源文件 `{FACTOR_RECORDS_FILE}`",
        "- 排序：研报原文公式(verbatim) 优先，其次文字定义推导(derived)；`expr_local` 可直接进 DSL 评估，"
        "**不得改变机制语义**。",
    ]
    for i, rec in enumerate(picked, 1):
        kind = str(rec.get("formula_kind") or "")
        kind_text = "研报原文公式" if kind == "verbatim" else "⚠ 由文字定义推导，待验证"
        lines.append(f"{i}. **{rec.get('name') or '-'}** — `{rec.get('expr_local')}`")
        # 数据面：优先用记录里的值；为空时**就地重算**（facet 表后续补全前缀时，
        # 存量 jsonl 不必重抽也能显示正确——抽取时算好的旧值可能是空的）
        _facets = list(rec.get("focus_facets") or [])
        if not _facets:
            try:
                from scripts._factor_expr_tools import infer_focus_facets

                _facets = infer_focus_facets(rec.get("expr_local"), rec.get("fields_local") or [])
            except Exception:  # noqa: BLE001
                _facets = []
        facets = "、".join(str(f) for f in _facets) or "-"
        lines.append(f"   - 方向：{rec.get('direction') or '-'} ｜ 公式档：{kind_text} ｜ 数据面：{facets}")
        lines.append(f"   - 算子树：{_record_tree_text(rec)}")
        if kind != "verbatim":
            deriv = " ".join(str(rec.get("derivation") or "").split())[:140]
            if deriv:
                lines.append(f"   - 推导依据：{deriv}")
        lines.append(f"   - 频率标签建议：{_record_freq_text(rec, freq_index)}")
    if dropped:
        uniq = list(dict.fromkeys(dropped))
        lines.append(f"- （以下字段本 run 未载入，相关因子已剔除：{'、'.join(uniq[:6])}）")
    return "\n".join(lines)


_DIVERGE_DIMS = ("window", "operator", "field", "neutralize", "gate_shape", "interaction")


def _parent_diagnosis(profile: dict | None) -> str:
    """按父本"体检报告"给出**针对性**改进建议（哪一维先动、为什么）。"""
    if not profile:
        return ""
    def _f(v, n=4):
        return f"{v:.{n}f}" if isinstance(v, (int, float)) else "—"
    ic, icir = profile.get("ic"), profile.get("icir")
    cov, to = profile.get("cov"), profile.get("turnover")
    shape = profile.get("shape_verdict")
    out = [f"- 父本体检：IC={_f(ic)} ICIR={_f(icir)} coverage={_f(cov, 3)} 换手={_f(to, 3)} "
           f"自相关={_f(profile.get('autocorr'), 3)} 形态对账={shape or '—'}"
           + (f"（实际={profile.get('actual_shape')} / 期望={profile.get('expected_shape')}）"
              if profile.get("actual_shape") or profile.get("expected_shape") else "")]
    adv = []
    if isinstance(icir, float) and icir <= 0:
        adv.append("**ICIR≤0 → 先查方向/做稳健化**：确认 predicted 方向与强侧没填反（形态对账 contradicted 即方向问题），"
                   "再做 `RANK`/`CS_WINSORIZE` 或换中性化键；**这一阶段别先调窗长**。")
    elif isinstance(icir, float) and abs(icir) < 0.15:
        adv.append("**ICIR<0.15 → 优先稳健化**：`RANK`/`CS_WINSORIZE`/行业中性化，或把离散门控换成连续 `SOFT_GATE`。")
    if isinstance(to, float) and to > 0.5:
        adv.append(f"**换手超标（{to:.3f}>0.5）→ 先降换手**：时序平滑（`TS_MEAN`/`WMA`）、拉长窗长、或换更慢的同族字段。")
    if isinstance(cov, float) and cov < 0.85:
        adv.append(f"**coverage 不足（{cov:.3f}<0.85）→ 先补覆盖**：去掉苛刻门控或换更全的同族字段。")
    if isinstance(ic, float) and abs(ic) < 0.02:
        adv.append("**|IC|<0.02 → 用结构杠杆**：`CS_GROUP_RANK`/`DIVERGENCE_RANK` 条件式结构，或换同族字段加信息量。")
    if shape == "unverifiable":
        adv.append("**形态对账 unverifiable → 先修 prediction**：按研报预期明确 shape/强侧/方向，形态通道才能过线。")
    out += adv
    out.append("- **每个变异必须用一句话写明经济直觉**（为什么这一改更贴合研报机制，而不只是数值好看），写进 `edit_note`/`comment`。")
    return "\n".join(out)


def render_diverge_task(question: dict, reproduce_factor: str = "", dims=_DIVERGE_DIMS,
                        parent_detail: str = "", card: dict | None = None,
                        parent_profile: dict | None = None) -> str:
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
        *([_parent_diagnosis(parent_profile)] if _parent_diagnosis(parent_profile) else []),
        "- **本轮目标**：让该机制比复现版更强且可交付 —— 争取过 train_passed 线（训练过线，非质量结论）"
        "（|IC|≥0.02、|ICIR|≥0.28、coverage≥0.85），同时不抬高日换手（≤0.5）、不与库内已有因子撞车；"
        "若某维度让指标变差，明确放弃并换下一个维度，不要反复调同一维。",
        "- **维度轮换建议**（3 轮内覆盖不同维度）：首轮 `window` 或 `operator`，次轮 `neutralize` 或 `field`，末轮 `gate_shape` 或 `interaction`。",
        "- **若母本信号偏弱（|IC|<0.015 或 |ICIR|<0.15）**：前两轮若 window/neutralize 无明显改善，"
        "第三轮直接上结构性杠杆 —— `gate_shape`（硬门→SOFT_GATE/分位分段）或 `interaction`"
        "（`CS_GROUP_RANK`/`DIVERGENCE_RANK` 等条件式结构），它们对弱信号母本的提升通常远大于微调窗长；"
        "仍无改善就如实判定该机制在本池无效，不必反复凑维度。",
        "- **同构空间耗尽即止损**：若连续 ≥2 次评估被 `memory_blocked_duplicate`（同构死路）拦截，"
        "说明该机制在本池的可变空间已经试尽——立刻换一个维度，或直接结束本轮并说明该机制无效；"
        "**不要**在被拦截后继续提交同构表达式（实测会整轮空转、白烧 40 分钟）。",
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
                          evidence: str = "", *, factor_records: list[dict] | None = None,
                          available_fields: Iterable[str] | None = None,
                          freq_index: dict | None = None) -> str:
    """研报模式的**复现题面**（Phase 1）：把课题自带的机制物料变成"必须复现"的任务块。

    物料来自题库字段本身（construction_guide / empirical_findings / suggested_fields /
    suggested_operators / expected_shape / expected_sign / falsifier），不依赖机制卡扩容
    （卡扩容是后续的增强，见 docs/specs/alphaagent_report_reproduce_diverge_spec.md §3）。

    spec §7 接入：额外注入"本报告因子清单"块（抽取侧 ``factor_records``，只读 jsonl，
    ``report_policy.inject_factor_records=False`` 可关；``factor_records`` 显式传入则跳过落盘读取）。
    ``available_fields``（运行期 available_columns）用于**字段可用性过滤**，避免题面承诺
    本 run 未载入的字段；未提供 → 不判定。
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
    if falsifier and falsifier[:30] not in (findings or "")[:120]:
        parts.append(f"- 证伪条件：{falsifier}")
    # ── 结构目标（B 版题面主体，2026-10-02）──────────────────────────────
    # 复现对象 = 报告提出的**结构/机制**，不是那条参考公式。依据：实测单算子因子达双门槛率
    # 1.4%、单字段 0.1%，而含结构算子 9.1%、3+ 字段 11.2% —— 复现简单公式必然无效。
    _struct = str(question.get("report_structure") or "")
    _target = str(question.get("reproduction_target") or "").strip()
    if _target:
        _sr = ((question.get("primary") or {}).get("spec_requirements") or {})
        parts += [
            "",
            f"**复现目标（结构类型 `{_struct}`）**：{_target}",
            f"- 要素清单：字段 `{_sr.get('fields')}`；窗口 `{_sr.get('windows')}`；"
            f"算子 `{_sr.get('operators')}`；结构性算子 `{_sr.get('structure_ops')}`；"
            f"处理链 `{_sr.get('chain')}`；最少字段数 `{_sr.get('min_fields')}`",
            "- **必须体现上述结构**（不是照抄一条简单公式）：优先用**结构性算子**"
            "（`SOFT_GATE` / `CS_GROUP_RANK` / `CS_RESIDUALIZE` / `CS_NEUTRALIZE` / "
            "`IF_THEN_ELSE` / `DIVERGENCE_RANK`）表达该结构；若该结构在本仓算子下无法直接表达，"
            "就用多层嵌套 + `CS_ZSCORE`/`CS_RANK`/`ADD` 组合近似，并在 `edit_note` 写明落地方式。",
            "- 下方「本报告因子清单」里的公式**仅作参考/对照**，**不作为复现目标**。",
        ]
    parts += [
        "",
        "**判定标准（系统按此判定，满足其一即「复现通过」并进入发散阶段）**：",
        "1. **形态对账 confirmed**：你提交的 `prediction.expected_shape / expected_strong_side / expected_sign` "
        "必须与因子实际分层形态一致（单调递增/递减、倒U、U形、极端尖峰、条件子组）；",
        "2. **信号存在档**：|IC| ≥ 0.010 且 ICIR ≥ 0.10（强度目标留给发散阶段）。",
        "→ **务必认真填 `prediction`**：胡乱填或与研报预期不符，形态对账会判 `unverifiable`/"
        "`contradicted`，本轮复现即作废（别把 expected_shape 填成 'monotonic' 这类非法值）。",
    ]
    _cb = _card_block(card)
    if _cb:
        parts.append(_cb)
    if evidence:
        parts.append("- **研报原文片段（含公式/表格，来自 MinerU 重抽语料；据此落地表达式）**：")
        parts.append("  " + "\n  ".join(str(evidence).splitlines()[:20]))
    _fid_injected = False
    if bool(((spec or {}).get("report_policy") or {}).get("inject_factor_records", True)):
        try:
            _recs = factor_records if factor_records is not None else load_factor_records(spec)
            _matched, _src = match_report_records(question, card, _recs)
            if _matched:
                _fblock = render_factor_records_block(
                    _matched, available_fields=available_fields, spec=spec,
                    freq_index=freq_index if freq_index is not None else load_report_freq_index(spec),
                    question=question, report_source=_src,
                )
                if _fblock:
                    parts.append("")
                    parts.append(_fblock)
                    _fid_injected = True
        except Exception as _e:  # noqa: BLE001
            logger.warning("研报因子清单注入失败（失败静默）: %s", _e)
    parts += [
        "",
        "**硬约束（违反会被直接拒绝，不消耗评估额度）**：",
        f"1. 复现版的 `parent_factor` 必须写成 `reproduce_of:{qid}`（或包含 `{qid}`），"
        "**禁止填自己上一轮的因子名**；",
        "2. 只允许「字段同族替换 + 算子落地」——**不得改变研报的机制语义**；",
        "3. 复现版先用 `train_screen` 评估；通过后（本模式）才进入发散阶段，发散一次只改一个维度"
        "（窗长／算子／同族字段／中性化键／门控形态／交互结构），并填 `parent_factor=<复现版因子名>`。",
    ]
    if _fid_injected:
        parts.append(
            "4. **原文锚（硬要求，见上方「本报告因子清单」）**：复现表达式必须用到原文公式里的字段——"
            "至少命中 **1 个报告特有字段**（如 `$funda_ocf`、`$ac_eps_fy`）；"
            "`$adj_close`/`$amount`/`$close` 等**通用行情字段不算锚定证据**。"
            "若只共享通用字段、或改用教科书通用因子（5 日反转 / Amihud 非流动性 / PE 动量…），"
            "**即使统计过线也判「原文锚未达标」不予复现通过**。"
        )
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

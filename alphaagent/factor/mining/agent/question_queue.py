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


def get_question_for_turn(
    turn: int,
    spec: dict[str, Any] | None = None,
    focus_facets: Iterable[str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """按轮次与数据面获取具体的研究问题对象。"""
    queue = load_question_queue(spec)
    if not queue:
        return None

    facets = set(focus_facets or ())
    matched = [q for q in queue if set(q.get("facets", [])) & facets] if facets else queue
    candidate_pool = matched if matched else queue

    offset = 0
    if session_id:
        import hashlib
        offset = int(hashlib.md5(str(session_id).encode("utf-8")).hexdigest(), 16)

    return candidate_pool[(offset + turn) % len(candidate_pool)]


def get_task_for_turn(
    turn: int,
    spec: dict[str, Any] | None = None,
    focus_facets: Iterable[str] | None = None,
    session_id: str | None = None,
) -> str:
    """按当前轮次获取定向研究问题任务指导文本块。"""
    q = get_question_for_turn(turn, spec=spec, focus_facets=focus_facets, session_id=session_id)
    if not q:
        return ""

    lines = [
        "### 本轮研报定向攻关课题【RDAgent 研究问题驱动】",
        f"- **课题编号**：`{q.get('question_id')}` —— {q.get('topic')}",
        f"- **文献来源**：{q.get('source')}",
        f"- **机制假设**：{q.get('hypothesis')}",
        f"- **建议涉及字段**：{', '.join(q.get('suggested_fields', []))}",
        f"- **推荐算子构想**：{', '.join(q.get('suggested_operators', []))}",
        f"- **预期检验形态**：形态={q.get('expected_shape')}，符号={q.get('expected_sign')}",
        f"- **潜在证伪陷阱**：⚠️ {q.get('falsifier')}",
        "",
        "**任务要求**：请针对上述文献假设，设计至少 1 条具体针对该机制的 DSL 因子表达式，并在思维链中说明如何量化，严格执行 `prediction` 预测对账！"
    ]
    return "\n".join(lines)

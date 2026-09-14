"""LLM 辅助 ML 组合：A) 因子推荐子集 + C) 训练报告解读。

设计约束：
- LLM 失败绝不阻断训练（调用方兜底回退全量）
- 只输出白名单不输出权重（决策确定性优先）
- 不喂 comment（FactorEntry 无此字段；且 comment 是挖掘时 LLM 自述叙事，
  喂给第二层 LLM 会放大叙事偏差，只喂硬结构信息：expr/facets/library/created_at）
- 复用锁定文件放因子库级稳定路径（不带时间戳），保证盲测可复现
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alphaagent.core.llm_provider import chat_json

if TYPE_CHECKING:
    from alphaagent.factor.stacking.dataset import FactorEntry

# 复用锁定文件：因子库级稳定路径（不带时间戳）
# llm_assist.py 位于 <repo>/alphaagent/factor/stacking/ → parents[3] = 仓库根
RECOMMENDATION_LOCK_FILE = (
    Path(__file__).resolve().parents[3] / "artifacts" / "alphaagent" / "stacking" / "llm_recommendation.json"
)

# ── Prompt ──

_SYSTEM_A = """\
你是量化因子组合的语义筛选助手。你的任务是：从给定的因子清单中推荐一个适合进入 ML 组合训练的子集。

重要规则：
1. mRMR 统计推荐已存在（基于 IC 和相关系数），你负责的是**语义维度**——数据面多样性、经济逻辑互补、同质变体去冗余。
2. 只依据硬结构信号判断：表达式、数据面标签、入库时间、所属因子库。禁止依据任何叙事性描述打正分。
3. 数据面多样性：价量、基本面、资金流、筹码等不同面应互补，不要同质堆叠（如 3 个动量变体只留 1 个）。
4. 经济逻辑互补：动量+反转+质量 > 三个动量变体。不同信息源的因子组合更有价值。
5. 时间衰减预警：入库时间老（created_at 早）的因子如果表达式简单，可能已被市场消化，适当降权。
6. 推荐理由必须落到结构维度（数据面、算子族、同质/互补关系），不得引用任何"作者声称的因果故事"。

输出严格 JSON（不要 Markdown 代码块）：
{"recommended": ["因子名1", "因子名2", ...], "rationale": "一句话概述推荐逻辑"}

推荐数量不限，但应是一个精炼子集（通常 8-20 个），不是全选。"""

_SYSTEM_C = """\
你是量化组合训练报告的解读助手。你的任务是：基于给定的训练报告数据，生成结构化的组合说明书。

重要规则：
1. 只解读报告中的数字，不虚构指标。不确定的就说"不确定"。
2. 输出四段：summary（一段话总结）、strengths（亮点列表）、risks（风险列表）、suggestions（建议列表）。
3. 每段限长：summary ≤200 字，其余每条 ≤100 字。
4. 中文输出。

输出严格 JSON（不要 Markdown 代码块）：
{"summary": "...", "strengths": ["...", ...], "risks": ["...", ...], "suggestions": ["...", ...]}"""


# ── 工具函数 ──

def candidate_pool_fingerprint(entries: list[FactorEntry]) -> str:
    """候选池 name 集合的 SHA256 哈希（用于复用锁定）。"""
    names = sorted(e.name for e in entries)
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def _sanitize_vintage(created_at: str | None) -> str:
    """盲测防泄漏清洗：将入库时间转换为安全相对标识，严禁向 LLM 泄漏 2025+ 盲测期时间戳。"""
    if not created_at:
        return "unknown"
    s = str(created_at).strip()
    # 盲测安全隔离：禁止暴露 2025/2026 等盲测期年份，统一脱敏为相对标签
    for forbidden in ("2025", "2026", "2027"):
        if forbidden in s:
            return "post-mining (recent)"
    return s[:7]


def _compress_entries(entries: list[FactorEntry], *, max_cap: int = 40) -> list[dict[str, Any]]:
    """把 FactorEntry 列表压缩成 LLM 可读的精简 dict 列表。

    只保留硬结构信息：name, facets, library, created_at, expr（截断 120 字）。
    超过 max_cap 个时按入库时间新→旧截断。
    严格执行盲测防泄漏：输出字典中的 created_at 字段对 2025+ 盲测年份做脱敏清洗。
    """
    sorted_entries = list(entries)
    if len(sorted_entries) > max_cap:
        sorted_entries.sort(key=lambda e: str(e.created_at or ""), reverse=True)
        sorted_entries = sorted_entries[:max_cap]

    items = []
    for e in sorted_entries:
        items.append({
            "name": e.name,
            "facets": list(e.facets) if e.facets else [],
            "library": e.library,
            "created_at": _sanitize_vintage(e.created_at),
            "expr": (e.expr or "")[:120],
        })
    return items


def _build_prompt_a(entries: list[FactorEntry], *, max_cap: int = 40) -> str:
    """构建 A 步骤的 user prompt（因子清单 JSON）。"""
    compressed = _compress_entries(entries, max_cap=max_cap)
    return "以下是因子库中的全部候选因子，请推荐适合进入 ML 组合训练的子集：\n\n" + json.dumps(
        compressed, ensure_ascii=False, indent=1
    )


def _build_report_view(report: dict) -> str:
    """把 report.json 压缩成 LLM 可读的视图（不传全量 report）。"""
    view: dict[str, Any] = {
        "eval_mode": report.get("eval_mode", "tuning"),
        "blind_test_isolated": report.get("blind_test_isolated", True),
        "folds": report.get("folds"),
        "scheme": report.get("scheme"),
        "scheme_label": report.get("scheme_label"),
        "label_days": report.get("label_days"),
        "mining_end": report.get("mining_end"),
        "feature_names": report.get("feature_names", []),
        "dropped_count": len(report.get("dropped", [])),
        "dropped_top": [
            {"name": d.get("name"), "reason": d.get("reason")}
            for d in (report.get("dropped") or [])[:10]
        ],
    }

    # 逐折指标摘要。report 实际形态：{"ridge": [{fold,...},...], "lgbm": [...]}
    # （分模型 dict）；兼容历史/简化的 list 形态。
    fold_metrics = report.get("fold_metrics") or []
    if fold_metrics:
        if isinstance(fold_metrics, dict):
            view["fold_metrics"] = {}
            for kind, rows in fold_metrics.items():
                if isinstance(rows, list):
                    view["fold_metrics"][kind] = [
                        {
                            "fold": fm.get("fold") if isinstance(fm, dict) else None,
                            "ic_mean": fm.get("ic_mean") if isinstance(fm, dict) else None,
                            "ic_ir": fm.get("ic_ir") if isinstance(fm, dict) else None,
                            "oos_sharpe": fm.get("oos_sharpe") if isinstance(fm, dict) else None,
                            "oos_max_drawdown": fm.get("oos_max_drawdown") if isinstance(fm, dict) else None,
                        }
                        for fm in rows
                    ]
        elif isinstance(fold_metrics, list):
            view["fold_metrics"] = [
                {
                    "fold": fm.get("fold") if isinstance(fm, dict) else None,
                    "ic_mean": fm.get("ic_mean") if isinstance(fm, dict) else None,
                    "ic_ir": fm.get("ic_ir") if isinstance(fm, dict) else None,
                    "oos_sharpe": fm.get("oos_sharpe") if isinstance(fm, dict) else None,
                    "oos_max_drawdown": fm.get("oos_max_drawdown") if isinstance(fm, dict) else None,
                }
                for fm in fold_metrics
            ]

    # 特征权重 Top-10。report 实际形态：{"ridge": [{name,weight},...], "lgbm": [...]}
    # 分模型 dict；兼容旧/简化的单层形态 {"name": weight} 或 list。
    fw = report.get("feature_weights") or {}
    if isinstance(fw, dict):
        # 单层形态：{"name": weight} → 按绝对值取 Top-10
        if all(isinstance(v, (int, float)) for v in fw.values()):
            sorted_fw = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)
            view["feature_weights_top10"] = dict(sorted_fw[:10])
        else:
            # 分模型形态：{"ridge": [{name,weight},...], ...}
            top10: dict[str, list[dict[str, Any]]] = {}
            for kind, rows in fw.items():
                if isinstance(rows, list):
                    by_weight = sorted(
                        rows, key=lambda x: abs(x.get("weight") or 0.0) if isinstance(x, dict) else 0.0,
                        reverse=True,
                    )
                    top10[kind] = [
                        {"name": r.get("name"), "weight": round(float(r.get("weight") or 0.0), 4)}
                        for r in by_weight[:10] if isinstance(r, dict)
                    ]
            view["feature_weights_top10"] = top10
    elif isinstance(fw, list):
        view["feature_weights_top10"] = fw[:10]

    # gate 结论（实际结构：{"passed", "selection_pct", "metrics": {...}}）
    gate = report.get("gate")
    if gate and isinstance(gate, dict):
        gm = gate.get("metrics") or {}
        view["gate"] = {
            "passed": gate.get("passed"),
            "selection_pct": gate.get("selection_pct"),
            "freq": gate.get("freq"),
            "selection_mode": gate.get("selection_mode"),
            "excess_annual": gm.get("excess_annual"),
            "excess_sharpe": gm.get("excess_sharpe"),
            "sharpe": gm.get("sharpe"),
            "max_drawdown": gm.get("max_drawdown"),
            "daily_overlap": gm.get("daily_overlap"),
            "annual_return": gm.get("annual_return"),
        }

    # 多路径对照
    mp = report.get("multi_path")
    if mp and isinstance(mp, dict):
        view["multi_path_aggregate"] = mp.get("aggregate")

    # 子集曲线
    sc = report.get("subset_curve")
    if sc:
        view["subset_curve_summary"] = {
            "best_k": sc.get("best_k") if isinstance(sc, dict) else None,
            "n_points": len(sc.get("points", [])) if isinstance(sc, dict) and isinstance(sc.get("points"), list) else 0,
        }

    return "以下是 ML 组合训练报告的压缩视图，请生成结构化组合说明书：\n\n" + json.dumps(
        view, ensure_ascii=False, indent=1, default=str
    )


# ── A 步骤：LLM 语义推荐因子子集 ──

def llm_recommend_subset(
    entries: list[FactorEntry],
    *,
    pool_fingerprint: str | None = None,
    max_factors_cap: int = 40,
) -> dict | None:
    """A 步骤：LLM 语义研判推荐因子子集白名单。

    返回 dict:
      {"recommended": [...], "rationale": "...", "model": MODEL,
       "pool_fingerprint": "...", "prompt_snapshot": "..."}
    失败返回 None（调用方兜底回退全量）。
    """
    if len(entries) < 2:
        return None

    valid_names = {e.name for e in entries}
    prompt_user = _build_prompt_a(entries, max_cap=max_factors_cap)
    result = chat_json(
        system=_SYSTEM_A,
        user=prompt_user,
        max_tokens=4096,
        temperature=0.2,
    )
    if not result:
        return None

    recommended = result.get("recommended")
    if not isinstance(recommended, list):
        return None

    # 硬校验：丢弃幻觉因子名
    recommended = [n for n in recommended if isinstance(n, str) and n in valid_names]
    if len(recommended) < 2:
        return None

    rationale = str(result.get("rationale") or "")
    model_name = os.getenv("MODEL", "")

    return {
        "recommended": recommended,
        "rationale": rationale,
        "model": model_name,
        "pool_fingerprint": pool_fingerprint or candidate_pool_fingerprint(entries),
        "prompt_snapshot": prompt_user,
    }


# ── C 步骤：LLM 解读训练报告 ──

def llm_summarize_report(report: dict) -> dict | None:
    """C 步骤：LLM 读报告压缩视图，生成结构化组合说明书。

    返回 dict:
      {"summary": "...", "strengths": [...], "risks": [...], "suggestions": [...]}
    失败返回 None。
    """
    # ── 盲测保护绝对拦截：终审盲测模式禁止调用 LLM 解读 ──
    if report.get("eval_mode") == "blind_test" or report.get("blind_test_isolated") is False:
        print("[safety-intercept] 拦截：报告包含 blind_test 盲测段数据，绝对禁止调用 LLM 进行解读，防止盲测数据泄漏。")
        return None

    report_view = _build_report_view(report)
    result = chat_json(
        system=_SYSTEM_C,
        user=report_view,
        max_tokens=2048,
        temperature=0.2,
        retries=1,
        timeout=40,  # 单次补全超时 40s：C 族不阻塞核心结果（方案 §3.3 评估修订）
    )
    if not result:
        return None

    # 基本结构校验
    summary = str(result.get("summary") or "")
    if not summary:
        return None

    return {
        "summary": summary,
        "strengths": [str(s) for s in (result.get("strengths") or []) if s],
        "risks": [str(s) for s in (result.get("risks") or []) if s],
        "suggestions": [str(s) for s in (result.get("suggestions") or []) if s],
        "model": os.getenv("MODEL", ""),
    }

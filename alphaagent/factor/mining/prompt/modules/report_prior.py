# -*- coding: utf-8 -*-
"""模块 05d · report_prior：券商金工研报机制先验检索注入（R2 方案）。

按当前运行的 focus_facets 与 research_mode 动态检索最匹配的结构化机制卡，
注入系统提示词作为假设生成先验。

设计纪律（docs/specs/alphaagent_report_knowledge_integration_spec.md §4）：
1. 反 scrubber 约束（P2-1）：字段与算子建议不使用带 `$` 的反引号代码块，
   避免被 scope_filter 的 scrub_out_of_scope 误判为跨面示例而整段擦除；
2. 预算严格截断：在行边界截断（对齐 retrieval._clip 语义），不挤占核心预算；
3. provenance 带内可见（P2-8）：统一冠以【外部研报先验·非本平台实测】标记；
4. 容错降级：卡片文件不存在或无匹配时静默返回空串，不阻断装配。
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from core.research_modes import infer_research_mode

logger = logging.getLogger(__name__)

NAME = "report_prior"
TITLE = "研报机制先验（按面检索）"
ORDER = 57
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"
PHASES = frozenset({"explore", "deepen", "full"})


def _resolve_cards_file(ctx: Any) -> Path | None:
    """解析机制卡文件路径，支持 spec 配置与主工作区自定位。"""
    spec = getattr(ctx, "research_spec", None) or {}
    policy = spec.get("report_policy") or {}
    custom_path = policy.get("report_cards_path")
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p

    # 尝试主工作区标准路径
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            p = Path(out).parent / "data" / "research_reports" / "knowledge" / "mechanism_cards.jsonl"
            if p.is_file():
                return p
    except Exception:
        pass

    fallback = Path(__file__).resolve().parents[5] / "data" / "research_reports" / "knowledge" / "mechanism_cards.jsonl"
    return fallback if fallback.is_file() else None


def enabled(ctx: Any) -> bool:
    if getattr(ctx, "asset_type", "stock") != "stock":
        return False
    spec = getattr(ctx, "research_spec", None) or {}
    policy = spec.get("report_policy") or {}
    if not bool(policy.get("enable_report_prior", True)):
        return False
    p = _resolve_cards_file(ctx)
    return bool(p and p.is_file())


def _load_cards(cards_file: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    try:
        for line in cards_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cards.append(json.loads(line))
    except Exception as e:
        logger.warning("加载机制卡失败 (%s): %s", cards_file, e)
    return cards


def _score_card(card: dict[str, Any], focus_facets: set[str], mode: str) -> float:
    score = 0.0
    facets = set(card.get("facets") or [])

    # 1. 数据面匹配
    if focus_facets:
        intersect = facets & focus_facets
        if intersect:
            score += 10.0 * len(intersect)
        else:
            score -= 5.0
    else:
        # focus 为空时的回退口径：兼顾模式与通用面
        score += 2.0

    # 2. 模式偏好（基本面/量价）
    if mode == "fundamental" and ("基本面" in facets or "机构面" in facets):
        score += 5.0
    elif mode == "technical" and ("价量面" in facets or "量能面" in facets or "筹码面" in facets):
        score += 5.0

    # 3. 负证据加分（高价值避坑先验）
    if card.get("negatives"):
        score += 3.0

    return score


def _clip_text(text: str, budget: int) -> str:
    """行边界截断到 budget 字符内。"""
    if budget <= 0 or len(text) <= budget:
        return text
    clipped = text[:budget].rsplit("\n", 1)[0]
    return clipped if clipped else text[:budget]


def _format_cards(cards: list[dict[str, Any]], budget: int) -> str:
    if not cards:
        return ""

    lines: list[str] = [
        "### 研报机制先验【外部研报先验·非本平台实测】",
        "",
        "以下由券商金工研报提炼的先验机制与假设，供本轮因子设计与变异参考（引用时注明 ID，如 [mc_0001]）：",
        "",
    ]

    for c in cards:
        cid = c.get("card_id", "mc_xxxx")
        src = c.get("source", {})
        mech = c.get("mechanism", {})
        facets_str = "、".join(c.get("facets", []))

        block = [
            f"- **【{cid}】{src.get('title', '未知研报')}**（{src.get('org', '券商')}，数据面：{facets_str}）：",
            f"  - 机制三问：错边方={mech.get('who_wrong', '无')}；套利受限={mech.get('why_persists', '无')}；观测特征={mech.get('observable', '无')}",
        ]

        # 反 scrubber 约束：字段与算子建议使用纯文本或括号，绝不使用带 $ 的代码块
        fields = c.get("fields")
        if fields:
            block.append(f"  - 涉及变量建议：{', '.join(fields)}")

        hint = c.get("dsl_hint")
        if hint:
            block.append(f"  - 算子构造思路：{hint}（建议方向，可自由拓展变异）")

        # 负证据（死路避坑）
        negs = c.get("negatives") or []
        for n in negs:
            if isinstance(n, dict):
                block.append(f"  - ⚠️ 研报自述失效预警：{n.get('claim', '')}（原因：{n.get('reason', '')}）")
            elif isinstance(n, str):
                block.append(f"  - ⚠️ 研报自述失效预警：{n}")

        block.append("")
        lines.append("\n".join(block))

    full_text = "\n".join(lines).strip()
    return _clip_text(full_text, budget)


def render(ctx: Any) -> str:
    cards_file = _resolve_cards_file(ctx)
    if not cards_file:
        return ""

    cards = _load_cards(cards_file)
    if not cards:
        return ""

    spec = getattr(ctx, "research_spec", None) or {}
    policy = spec.get("report_policy") or {}
    budget = int(policy.get("report_prior_max_chars", 1600))
    top_k = int(policy.get("report_prior_top_k", 4))

    focus_facets = set(getattr(ctx, "focus_facets", ()) or ())
    mode = infer_research_mode(focus_facets)

    # 打分排序
    scored = sorted(cards, key=lambda c: _score_card(c, focus_facets, mode), reverse=True)
    selected = scored[:top_k]

    return _format_cards(selected, budget)

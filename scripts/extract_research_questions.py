# -*- coding: utf-8 -*-
"""从券商研报 Markdown 库中全量批量抽取、多维去重并生成结构化研究问题题库（RDAgent 范式）。

输入：data/research_reports/parsed/ 下 1898 篇已解析研报
输出：data/research_reports/knowledge/research_questions.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("extract_questions")

REPO_ROOT = Path(__file__).resolve().parents[1]
_MAIN_REPO = Path(r"D:\Quant\quant_ui")
if (_MAIN_REPO / "data" / "research_reports" / "parsed").is_dir():
    PARSED_DIR = _MAIN_REPO / "data" / "research_reports" / "parsed"
else:
    PARSED_DIR = REPO_ROOT / "data" / "research_reports" / "parsed"

OUTPUT_FILE = REPO_ROOT / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"
MAIN_OUTPUT_FILE = _MAIN_REPO / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"

KNOWN_FACETS = {
    "价量面": ["close", "open", "high", "low", "vwap", "volume", "amount", "ret", "returns", "价量", "反转", "动量", "波动", "振幅"],
    "量能面": ["volume", "amount", "turnover", "换手", "放量", "缩量", "流动性", "量比", "成交金额"],
    "筹码面": ["chip", "筹码", "获利盘", "密集度", "平均成本", "套牢", "筹码峰", "成本分布"],
    "资金面": ["money_flow", "fund_flow", "大单", "超大单", "主力资金", "知情交易", "北向", "机构买入", "资金流向", "特大单"],
    "基本面": ["pe", "pb", "roe", "eps", "net_profit", "营收", "净利润", "估值", "成长", "盈利质量", "扣非", "现金流", "营业收入"],
    "股东面": ["holder", "shareholder", "户数", "集中度", "十大股东", "减持", "增持", "持股比例"],
    "事件面": ["announcement", "forecast", "业绩预告", "快报", "分红", "解禁", "研报覆盖", "分析师预期", "业绩快报"],
    "拥挤面": ["crowding", "拥挤", "同质化", "博弈", "微观结构", "交易过热"],
}

OPERATORS_POOL = [
    "DIVERGENCE_RANK", "CS_GROUP_RANK", "CS_RESIDUALIZE", "GATED_SIGNAL", "PIECEWISE_STATE",
    "SIGNAL_BLEND", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_PCTCHANGE", "DELTA",
    "WMA", "RANK", "CS_ZSCORE", "CS_WINSORIZE", "KLINE_GEOMETRY"
]

def clean_title(filename: str) -> tuple[str, str, str]:
    """解析研报文件名：日期、机构、标题"""
    stem = Path(filename).stem
    parts = stem.split("_")
    date = parts[0] if len(parts) > 0 and re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]) else "2020-01-01"
    org = parts[1] if len(parts) > 1 else "券商金工"
    title = "_".join(parts[2:]) if len(parts) > 2 else stem
    title = re.sub(r"[^\w\u4e00-\u9fa5\-_()（）]", "", title)
    return date, org, title

def title_semantic_fingerprint(title: str) -> str:
    """提取标题的核心主题词指纹，用于深度语义去重（过滤不同期的跟踪报告）"""
    # 去除期数、日期、代码等无实质区别标记
    norm = re.sub(r"\d{4,8}|第[0-9一二三四五六七八九十]+期|系列之[0-9一二三四五六七八九十]+|专题报告|金工[专题]*|动态跟踪|选股周报|月报|日报", "", title)
    norm = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", norm.lower())
    return norm[:30]

def infer_facets_and_fields(text: str, title: str) -> tuple[list[str], list[str], list[str]]:
    comb = (title + " " + text[:2000]).lower()
    matched_facets = []
    matched_fields = set()
    for facet, kws in KNOWN_FACETS.items():
        if any(kw.lower() in comb for kw in kws):
            matched_facets.append(facet)
            for kw in kws:
                if re.match(r"^[a-zA-Z_]+$", kw):
                    matched_fields.add(kw)
    if not matched_facets:
        matched_facets = ["价量面"]
        matched_fields = {"close", "volume"}
    if not matched_fields:
        matched_fields = {"close", "volume", "amount"}
    
    suggested_ops = [op for op in OPERATORS_POOL if op.lower() in comb]
    if not suggested_ops:
        if "背离" in comb or "分歧" in comb:
            suggested_ops = ["DIVERGENCE_RANK", "TS_PCTCHANGE", "CS_ZSCORE"]
        elif "残差" in comb or "对冲" in comb:
            suggested_ops = ["CS_RESIDUALIZE", "CS_ZSCORE"]
        elif "组" in comb or "分层" in comb:
            suggested_ops = ["CS_GROUP_RANK", "RANK"]
        elif "筹码" in comb:
            suggested_ops = ["CHIP_DENSITY", "CS_ZSCORE"]
        elif "门控" in comb or "阈值" in comb:
            suggested_ops = ["GATED_SIGNAL", "CS_ZSCORE"]
        else:
            suggested_ops = ["TS_MEAN", "DELTA", "CS_ZSCORE"]

    return matched_facets, sorted(list(matched_fields))[:4], suggested_ops[:4]

def extract_hypothesis_from_text(text: str, title: str, org: str) -> dict[str, Any] | None:
    """从正文中提取具备实证结论的科学假说"""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    core_para = ""
    
    # 优先关键词匹配
    signal_keywords = [
        "我们发现", "实证表明", "检验结果", "选股效果", "超额收益", "多空收益",
        "单调性", "ic显著", "逻辑在于", "因子构造", "选股能力", "有效性",
        "构建方式", "核心逻辑", "检验结论", "年化收益", "多空对冲", "因子表现"
    ]
    
    for p in paragraphs:
        if any(bad in p for bad in ["免责声明", "执业证书", "证券分析师", "研究所", "请务必阅读", "<!-- table", "| ---"]):
            continue
        if any(k in p for k in signal_keywords):
            core_para = p
            break
            
    if not core_para:
        for p in paragraphs:
            if not any(bad in p for bad in ["免责声明", "执业证书", "证券分析师", "研究所", "请务必阅读", "<!-- table", "| ---"]) and len(p) > 60:
                core_para = p
                break
    
    if len(core_para) < 35:
        return None

    # 清洗掉过多的标点与排版垃圾
    core_para = re.sub(r"\s+", " ", core_para)[:260]
    facets, fields, ops = infer_facets_and_fields(core_para, title)

    # 推断预期方向
    is_neg = any(k in core_para or k in title for k in ["反转", "负向", "衰减", "回调", "高估", "卖出", "避险", "下行", "负相关"])
    expected_sign = "negative" if is_neg else "positive"
    expected_shape = "单调递减" if is_neg else "单调递增"

    # 生成机制描述与可证伪点
    falsifier = "极端市场风格切换、指数单边暴跌或流动性枯竭时信号失效"
    if "反转" in core_para or "反转" in title:
        falsifier = "动量极强或机构趋势抱团加速时多头反转逻辑被击穿"
    elif "资金" in core_para or "大单" in title:
        falsifier = "主力大单呈现幌骗交易或次日反手砸盘诱多"
    elif "基本面" in core_para or "营收" in title:
        falsifier = "行业周期性下行导致财报业绩一次性出清失真"

    return {
        "topic": title,
        "source": f"{org}《{title}》",
        "hypothesis": core_para,
        "suggested_fields": fields,
        "suggested_operators": ops,
        "expected_shape": expected_shape,
        "expected_sign": expected_sign,
        "falsifier": falsifier,
        "facets": facets,
    }

def process_reports(limit: int = 2000) -> list[dict[str, Any]]:
    md_files = list(PARSED_DIR.rglob("*.md"))
    logger.info(f"扫描到已解析研报总数: {len(md_files)}")
    
    priority_files = []
    other_files = []
    
    for f in md_files:
        name = f.name
        # 排除纯行情流水或晨会
        if any(k in name for k in ["晨会", "开盘必读", "早间资讯", "今日要闻"]):
            continue
        # 核心金工多因子与选股
        if any(k in name for k in ["因子", "选股", "多因子", "专题", "系列", "异象", "模型", "动量", "反转", "筹码", "alpha", "Alpha", "高阶矩", "微观"]):
            priority_files.append(f)
        else:
            other_files.append(f)

    logger.info(f"筛选出核心专题研报: {len(priority_files)} 篇，其余研报: {len(other_files)} 篇")
    all_targets = priority_files + other_files

    seen_title_fps = set()
    questions = []
    facet_counts = {k: 0 for k in KNOWN_FACETS}
    
    for idx, f in enumerate(all_targets):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if len(txt) < 300:
                continue
            date, org, title = clean_title(f.name)
            
            # 多维去重：语义主题指纹
            fp = title_semantic_fingerprint(title)
            if len(fp) >= 4 and fp in seen_title_fps:
                continue
            
            res = extract_hypothesis_from_text(txt, title, org)
            if res:
                qid = f"RQ_{len(questions)+1:03d}"
                res["question_id"] = qid
                questions.append(res)
                if len(fp) >= 4:
                    seen_title_fps.add(fp)
                for fct in res["facets"]:
                    if fct in facet_counts:
                        facet_counts[fct] += 1
                if len(questions) >= limit:
                    break
        except Exception:
            continue

    logger.info(f"全量处理完成！成功抽取去重结构化研究课题: {len(questions)} 道")
    logger.info(f"课题数据面覆盖分布: {facet_counts}")
    return questions

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="抽取课题上限")
    ap.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    MAIN_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    questions = process_reports(limit=args.limit)

    # 同时写到 worktree 与主仓库，保证两端即刻可用
    for target_path in [args.output, MAIN_OUTPUT_FILE]:
        with open(target_path, "w", encoding="utf-8") as fh:
            for q in questions:
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"已同步落库研究问题题库: {args.output} 及 {MAIN_OUTPUT_FILE}")

if __name__ == "__main__":
    main()

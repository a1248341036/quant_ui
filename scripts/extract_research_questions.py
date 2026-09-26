# -*- coding: utf-8 -*-
"""从券商研报 Markdown 库中批量抽取并去重生成结构化研究问题题库（RDAgent 范式）。

输入：data/research_reports/parsed/ 下已解析研报
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
# 如果当前在 worktree，优先回溯到真实的仓库主工作区寻找 data/research_reports/parsed
_MAIN_REPO = Path(r"D:\Quant\quant_ui")
if (_MAIN_REPO / "data" / "research_reports" / "parsed").is_dir():
    PARSED_DIR = _MAIN_REPO / "data" / "research_reports" / "parsed"
else:
    PARSED_DIR = REPO_ROOT / "data" / "research_reports" / "parsed"
OUTPUT_FILE = REPO_ROOT / "data" / "research_reports" / "knowledge" / "research_questions.jsonl"

KNOWN_FACETS = {
    "价量面": ["close", "open", "high", "low", "vwap", "volume", "amount", "ret", "returns", "价量", "反转", "动量"],
    "量能面": ["volume", "amount", "turnover", "换手", "放量", "缩量", "流动性", "量比"],
    "筹码面": ["chip", "筹码", "获利盘", "密集度", "平均成本", "套牢", "筹码峰"],
    "资金面": ["money_flow", "fund_flow", "大单", "超大单", "主力资金", "知情交易", "北向", "机构买入"],
    "基本面": ["pe", "pb", "roe", "eps", "net_profit", "营收", "净利润", "估值", "成长", "盈利质量"],
    "股东面": ["holder", "shareholder", "户数", "集中度", "十大股东", "减持", "增持"],
    "事件面": ["announcement", "forecast", "业绩预告", "快报", "分红", "解禁", "研报覆盖"],
    "拥挤面": ["crowding", "拥挤", "同质化", "博弈", "微观结构"],
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

def text_similarity_hash(text: str) -> str:
    """归一化核心词哈希，用于去重"""
    core = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text.lower())
    return hashlib.md5(core[:100].encode("utf-8")).hexdigest()

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
        else:
            suggested_ops = ["TS_MEAN", "DELTA", "CS_ZSCORE"]

    return matched_facets, sorted(list(matched_fields))[:4], suggested_ops[:4]

def extract_hypothesis_from_text(text: str, title: str, org: str) -> dict[str, Any] | None:
    """规则启发式提取研报假说（离线稳健版本）"""
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    core_para = ""
    for p in paragraphs:
        # 跳过开头的免责声明、分析师执业证书、联系方式、以及 markdown 表格标记
        if any(bad in p for bad in ["免责声明", "执业证书", "证券分析师", "研究所", "请务必阅读", "<!-- table", "| ---"]):
            continue
        if any(k in p for k in ["我们发现", "实证表明", "检验结果", "选股效果", "超额收益", "多空收益", "单调性", "ic显著", "逻辑在于", "因子构造", "选股能力", "有效性", "构建方式", "核心逻辑", "检验结论"]):
            core_para = p
            break
    if not core_para:
        for p in paragraphs:
            if not any(bad in p for bad in ["免责声明", "执业证书", "证券分析师", "研究所", "请务必阅读", "<!-- table", "| ---"]) and len(p) > 60:
                core_para = p
                break
    
    if len(core_para) < 30:
        return None

    # 清洗核心描述
    core_para = re.sub(r"\s+", " ", core_para)[:260]
    facets, fields, ops = infer_facets_and_fields(core_para, title)

    # 推断预期方向
    is_neg = any(k in core_para or k in title for k in ["反转", "负向", "衰减", "回调", "高估", "卖出", "避险", "下行"])
    expected_sign = "negative" if is_neg else "positive"
    expected_shape = "单调递减" if is_neg else "单调递增"

    return {
        "topic": title,
        "source": f"{org}《{title}》",
        "hypothesis": core_para,
        "suggested_fields": fields,
        "suggested_operators": ops,
        "expected_shape": expected_shape,
        "expected_sign": expected_sign,
        "falsifier": "极端市场风格切换或流动性冲击下信号失效",
        "facets": facets,
    }

def process_reports(limit: int = 100) -> list[dict[str, Any]]:
    md_files = list(PARSED_DIR.rglob("*.md"))
    logger.info(f"扫描到已解析研报总数: {len(md_files)}")
    
    candidates = []
    # 优先选取金工多因子与选股目录
    priority_files = []
    for f in md_files:
        name = f.name
        if any(k in name for k in ["日报", "周报", "晨会", "动态", "跟踪"]):
            continue
        if any(k in name for k in ["因子", "选股", "多因子", "专题", "系列", "异象", "模型", "动量", "反转", "筹码"]):
            priority_files.append(f)

    logger.info(f"筛选出核心专题金工研报候选: {len(priority_files)}")

    seen_hashes = set()
    questions = []
    
    for idx, f in enumerate(priority_files):
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if len(txt) < 300:
                continue
            date, org, title = clean_title(f.name)
            
            # 去重：标题和核心主题
            h = text_similarity_hash(title)
            if h in seen_hashes:
                continue
            
            res = extract_hypothesis_from_text(txt, title, org)
            if res:
                qid = f"RQ_{len(questions)+1:03d}"
                res["question_id"] = qid
                questions.append(res)
                seen_hashes.add(h)
                if len(questions) >= limit:
                    break
        except Exception as e:
            continue

    logger.info(f"成功抽取非重复结构化研究课题: {len(questions)} 道")
    return questions

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="抽取课题上限")
    ap.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    questions = process_reports(limit=args.limit)

    with open(args.output, "w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"已成功写入研究问题题库: {args.output} (共 {len(questions)} 条)")

if __name__ == "__main__":
    main()

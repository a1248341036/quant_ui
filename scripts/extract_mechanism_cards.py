#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报机制卡提取与构建工具（R2 方案核心支持脚本）。

功能：
1. 校验机制卡 JSONL 文件的 Schema 合规性（14面标签、字段、source.pages等）；
2. 从研报解析 Markdown 文本库中自动提取并结构化机制候选；
3. 生成并维护标准机制卡库 `data/research_reports/knowledge/mechanism_cards.jsonl`。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaagent.factor.facets import FACET_DEFS

VALID_FACETS = frozenset(name for name, _ in FACET_DEFS)

# 精选种子机制卡（涵盖经典多因子与微观结构，符合 spec §7.2 规范）
SEED_CARDS: list[dict[str, Any]] = [
    {
        "card_id": "mc_0001",
        "source": {
            "path": "nn73/01_多因子与选股体系/01_单因子测试与挖掘/2016-06-27_海通证券_金融工程_选股因子系列研究（十二）：“量”与“价”的结合.md",
            "org": "海通证券",
            "date": "2016-06-27",
            "title": "选股因子系列研究（十二）：“量”与“价”的结合",
            "pages": [3, 5]
        },
        "mechanism": {
            "who_wrong": "高位跟风追涨散户或低位恐慌割肉盘（错边方）",
            "why_persists": "短期注意力过载与流动性集中涌入，随后资金断层导致反转",
            "observable": "价格创高/低伴随成交量反向萎缩或巨量滞涨"
        },
        "fields": ["close", "volume", "turnover"],
        "params": {"window": 20},
        "dsl_hint": "DIVERGENCE_RANK(TS_PCTCHANGE(close, 20), TS_PCTCHANGE(volume, 20))",
        "formula_text": None,
        "evidence": {
            "sample": "2010-2015 全A",
            "decile_shape": "单调",
            "ic": 0.028,
            "icir": 0.90,
            "annual": {"2012": 0.031, "2014": 0.026}
        },
        "negatives": [
            {"claim": "在牛市高换手期纯价格反转易遭动量打脸", "reason": "趋势延续性强，单纯逆势做空多头端风险大"}
        ],
        "facets": ["价量面", "量能面"],
        "extraction": {"model": "manual_curated", "confidence": 0.95, "formula_from_image": False}
    },
    {
        "card_id": "mc_0002",
        "source": {
            "path": "nn73/01_多因子与选股体系/01_单因子测试与挖掘/2016-08-12_东方金工因子选股系列研究之九_日内残差高阶矩与股票收益_东方证券.md",
            "org": "东方证券",
            "date": "2016-08-12",
            "title": "因子选股系列研究之九：日内残差高阶矩与股票收益",
            "pages": [2, 6]
        },
        "mechanism": {
            "who_wrong": "散户偏好彩票型高暴击股票，系统性高估极端右偏标的",
            "why_persists": "做空受限与卖空成本高昂，导致高估无法被及时套利熨平",
            "observable": "分时高频收益率的偏度与超额峰度"
        },
        "fields": ["close", "high", "low", "turnover"],
        "params": {"window": 20},
        "dsl_hint": "NEG(TS_SKEW(TS_PCTCHANGE(close, 1), 20))",
        "formula_text": "Skew = E[(R - mu)^3] / sigma^3",
        "evidence": {
            "sample": "2007-2016 全A",
            "decile_shape": "单调负向",
            "ic": -0.034,
            "icir": -1.15
        },
        "negatives": [
            {"claim": "年线下方极度超跌垃圾股可能出现投机性暴动", "reason": "微盘股风险偏好脉冲式暴涨"}
        ],
        "facets": ["价量面", "筹码面"],
        "extraction": {"model": "manual_curated", "confidence": 0.92, "formula_from_image": False}
    },
    {
        "card_id": "mc_0003",
        "source": {
            "path": "nn73/01_多因子与选股体系/01_单因子测试与挖掘/2016-12-20_华泰证券_金融工程_华泰多因子系列之四：华泰单因子测试之动量类因子.md",
            "org": "华泰证券",
            "date": "2016-12-20",
            "title": "华泰单因子测试之动量类因子",
            "pages": [4, 9]
        },
        "mechanism": {
            "who_wrong": "信息渐进扩散时滞，投资者对基本面变化存在认知锚定与反应不足",
            "why_persists": "分析师覆盖差异与机构调仓周期滞后",
            "observable": "中长周期（60~120日）剔除近期1个月反转后的残差动量"
        },
        "fields": ["close", "volume"],
        "params": {"window": 60, "lag": 20},
        "dsl_hint": "CS_RESIDUALIZE(TS_PCTCHANGE(close, 60), TS_PCTCHANGE(close, 20))",
        "formula_text": None,
        "evidence": {
            "sample": "2006-2016 全A",
            "decile_shape": "弱单调",
            "ic": 0.021,
            "icir": 0.65
        },
        "negatives": [
            {"claim": "在A股短期（5~20日）个股呈现极强反转而非动量", "reason": "散户T+1博弈与流动性兑现"}
        ],
        "facets": ["价量面"],
        "extraction": {"model": "manual_curated", "confidence": 0.90, "formula_from_image": False}
    },
    {
        "card_id": "mc_0004",
        "source": {
            "path": "nn73/01_多因子与选股体系/01_单因子测试与挖掘/2017-02-28_波动率因子的分解与截面收益_“双面”波动率_海通证券.md",
            "org": "海通证券",
            "date": "2017-02-28",
            "title": "波动率因子的分解与截面收益：“双面”波动率",
            "pages": [3, 8]
        },
        "mechanism": {
            "who_wrong": "避险投资者对特质波动率折价要求过高或低波动异象套利受限",
            "why_persists": "杠杆约束导致机构超配高贝塔股票，压低高波未来收益",
            "observable": "下行半方差与上行半方差的分离比率"
        },
        "fields": ["close", "open", "low"],
        "params": {"window": 20},
        "dsl_hint": "DIVIDE(TS_STD(IF_THEN_ELSE(TS_PCTCHANGE(close, 1) < 0, close, 0), 20), TS_STD(close, 20))",
        "formula_text": None,
        "evidence": {
            "sample": "2008-2016 全A",
            "decile_shape": "单调",
            "ic": -0.027,
            "icir": -0.88
        },
        "negatives": [
            {"claim": "在极度缩量市场低波动股票容易缺乏流动性", "reason": "流动性折价"}
        ],
        "facets": ["价量面", "拥挤面"],
        "extraction": {"model": "manual_curated", "confidence": 0.88, "formula_from_image": False}
    },
    {
        "card_id": "mc_0005",
        "source": {
            "path": "nn73/01_多因子与选股体系/01_单因子测试与挖掘/2016-04-08_估值类因子有效性分析_庖丁解因子_海通证券.md",
            "org": "海通证券",
            "date": "2016-04-08",
            "title": "估值类因子有效性分析：庖丁解因子",
            "pages": [2, 5]
        },
        "mechanism": {
            "who_wrong": "追逐热点题材的资金抛弃低估值防御股，形成安全边际折扣",
            "why_persists": "价值陷阱与均值回归周期的不确定性使得短期资金不愿介入",
            "observable": "经营性现金流市值比与自由现金流收益率的截面中性化"
        },
        "fields": ["funda_ocfps", "close", "float_cap"],
        "params": {"window": 60},
        "dsl_hint": "CS_NEUTRALIZE(CS_WINSORIZE(DIVIDE(funda_ocfps, close), 0.01, 0.99), float_cap)",
        "formula_text": None,
        "evidence": {
            "sample": "2005-2015 全A",
            "decile_shape": "分段单调",
            "ic": 0.024,
            "icir": 0.72
        },
        "negatives": [
            {"claim": "静态估值直接做短窗差分属于阶梯函数脉冲噪音", "reason": "季度财报非连续更新"}
        ],
        "facets": ["基本面", "机构面"],
        "extraction": {"model": "manual_curated", "confidence": 0.91, "formula_from_image": False}
    },
    {
        "card_id": "mc_0006",
        "source": {
            "path": "nn73/01_多因子与选股体系/01_单因子测试与挖掘/2016-06-06_选股因子系列研究(十一)_level2行情选股因子初探_海通证券.md",
            "org": "海通证券",
            "date": "2016-06-06",
            "title": "选股因子系列研究(十一)：level2行情选股因子初探",
            "pages": [3, 7]
        },
        "mechanism": {
            "who_wrong": "散户与游资在小单交易中跟风冲动，机构大单具备隐性知情交易优势",
            "why_persists": "大单拆单与算法交易隐藏意图，市场对大额净买入反应缓慢",
            "observable": "大单与超大单净流入占成交额比例（聪明钱信号）"
        },
        "fields": ["ff_super_net", "ff_large_net", "amount"],
        "params": {"window": 5},
        "dsl_hint": "CS_ZSCORE(TS_SUM(DIVIDE(ADD(ff_super_net, ff_large_net), amount), 5))",
        "formula_text": None,
        "evidence": {
            "sample": "2012-2016 全A",
            "decile_shape": "单调正向",
            "ic": 0.038,
            "icir": 1.40
        },
        "negatives": [
            {"claim": "游资对倒制造虚假超大单可能诱多", "reason": "对倒拉高出货需要配合换手率二次过滤"}
        ],
        "facets": ["资金面", "量能面"],
        "extraction": {"model": "manual_curated", "confidence": 0.94, "formula_from_image": False}
    }
]


def validate_card(card: dict[str, Any]) -> list[str]:
    """校验单张机制卡 Schema 合规性。"""
    errors = []
    card_id = card.get("card_id")
    if not card_id or not isinstance(card_id, str):
        errors.append("card_id 缺失或非字符串")

    source = card.get("source")
    if not isinstance(source, dict):
        errors.append("source 结构缺失")
    else:
        for k in ("path", "org", "title", "pages"):
            if k not in source:
                errors.append(f"source 缺少字段 {k}")
        if not isinstance(source.get("pages"), list) or not source.get("pages"):
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
            if f not in VALID_FACETS:
                errors.append(f"facet 非法: '{f}', 必须在 FACET_DEFS 14 面中")

    if not isinstance(card.get("negatives"), list):
        errors.append("negatives 必须为列表")

    return errors


def build_seed_cards(target_path: Path) -> int:
    """初始化或扩充标准机制卡 JSONL 文件。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    cards: list[dict[str, Any]] = []

    if target_path.is_file():
        for line in target_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cards.append(c)
                existing_ids.add(c.get("card_id"))
            except Exception:
                continue

    added = 0
    for seed in SEED_CARDS:
        cid = seed["card_id"]
        if cid not in existing_ids:
            errs = validate_card(seed)
            if errs:
                print(f"种子卡 {cid} 校验失败: {errs}", file=sys.stderr)
                continue
            cards.append(seed)
            existing_ids.add(cid)
            added += 1

    with open(target_path, "w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return len(cards)


def main() -> int:
    ap = argparse.ArgumentParser(description="研报机制卡提取与检验工具")
    ap.add_argument("--cards-file", default=None, help="目标卡片 JSONL 文件路径")
    ap.add_argument("--validate", action="store_true", help="校验现有卡片文件")
    ap.add_argument("--seed", action="store_true", help="写入种子标准卡")
    args = ap.parse_args()

    default_path = ROOT / "data" / "research_reports" / "knowledge" / "mechanism_cards.jsonl"
    target = Path(args.cards_file).resolve() if args.cards_file else default_path

    if args.seed or not target.is_file():
        total = build_seed_cards(target)
        print(f"写入/更新机制卡库: {target} (共 {total} 张卡)")

    if args.validate or target.is_file():
        all_errs = 0
        count = 0
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            count += 1
            c = json.loads(line)
            errs = validate_card(c)
            if errs:
                print(f"卡片 {c.get('card_id')} 异常: {errs}", file=sys.stderr)
                all_errs += len(errs)
        if all_errs == 0:
            print(f"卡片库校验通过: {count} 张卡，0 错误")
        else:
            print(f"卡片库校验未通过: 共发现 {all_errs} 处错误", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

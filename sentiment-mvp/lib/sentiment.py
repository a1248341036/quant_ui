# -*- coding: utf-8 -*-
"""情绪打分：默认用中文金融情感词典（确定性、零成本），可选 LLM 细化。"""

import math
import json
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from . import ml_sentiment as ml

POS_WORDS = [
    "增长", "上涨", "大涨", "暴涨", "涨停", "新高", "创新高", "历史新高", "突破", "利好",
    "超预期", "好于预期", "盈利", "净利润增长", "业绩增长", "营收增长", "回购", "增持",
    "中标", "签约", "获批", "批准", "通过", "强劲", "加速", "扩产", "提价", "涨价",
    "供不应求", "需求旺盛", "景气", "改善", "扭亏", "预增", "大增", "净流入", "加仓",
    "买入", "看多", "推荐", "增持评级", "买入评级", "优秀", "优质", "领先", "龙头",
    "受益", "机遇", "积极", "健康", "稳定增长", "高增长", "放量上涨", "强势", "拉升",
    "连板", "分红", "派息", "净利大增", "大幅增长", "同比增加", "环比增加", "创历史新高",
    "跑赢", "提振", "回暖", "企稳", "反弹", "超跌反弹",
]

NEG_WORDS = [
    "下跌", "大跌", "暴跌", "跌停", "亏损", "下滑", "下降", "大幅下降", "同比减少",
    "环比减少", "减持", "质押", "违规", "处罚", "罚款", "调查", "立案", "退市", "警示",
    "预亏", "预警", "商誉减值", "爆雷", "违约", "净流出", "利空", "承压", "萎缩",
    "疲软", "低迷", "下调", "调低", "腰斩", "停牌", "冻结", "诉讼", "仲裁", "召回",
    "事故", "裁员", "看空", "卖出", "卖出评级", "减持评级", "风险", "恶化", "不及预期",
    "低于预期", "走弱", "杀跌", "跳水", "破位", "跌破", "新低", "创历史新低", "警示函",
    "监管", "问询", "质疑", "亏损扩大", "资金链", "债务", "逾期", "担保", "立案调查",
    "违规担保", "大幅亏损", "强平", "爆仓", "收缩", "回撤", "低迷", "流出", "撤离",
    "抛售", "回落", "回吐", "缩量", "炸板", "净卖出", "净减仓", "融券增加", "两融下降",
    "融资余额下降", "跌停打开", "冲高回落",
]

STRONG_POS = {"涨停", "暴涨", "大涨", "历史新高", "创新高", "超预期", "大幅增长", "净利大增", "爆发", "飙升"}
STRONG_NEG = {"跌停", "暴跌", "爆雷", "退市", "立案", "腰斩", "跳水", "杀跌", "大幅亏损", "亏损扩大", "违约", "强平", "爆仓"}

# 中性偏弱词：出现在资金流/分红/榜单等泛泛描述里时权重减半
SOFT_POS = {"分红", "派息", "净流入", "净买入", "加仓", "买入", "突破", "受益"}
SOFT_NEG = {"回落", "减持", "质押", "下调", "风险"}

NEGATION = ["不", "未", "没", "无", "非", "难", "未能", "没有", "无法", "不再", "并非", "不会", "不是", "尚未", "并未", "不能", "难以", "并无", "暂缓", "停止"]

GENERIC_MARKERS = ["名单", "排行", "榜单", "榜", "复盘", "五日均线", "资金流向日报", "流入榜", "流出榜", "日报", "龙虎榜", "分红季", "超亿元", "只个股", "股获", "两融余额", "融资余额", "资金流向"]
CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")

_CACHE = {}


def has_negation(text, start, window=8):
    prev = text[max(0, start - window):start]
    return any(n in prev for n in NEGATION)


def _collect_hits(text):
    """返回非重叠词命中列表 [(start, end, delta)]，重叠时长词优先。"""
    raw = []
    for w in POS_WORDS:
        weight = 2 if w in STRONG_POS else (0.5 if w in SOFT_POS else 1)
        for m in re.finditer(re.escape(w), text):
            raw.append((m.start(), m.end(), weight))
    for w in NEG_WORDS:
        weight = 2 if w in STRONG_NEG else (0.5 if w in SOFT_NEG else 1)
        for m in re.finditer(re.escape(w), text):
            raw.append((m.start(), m.end(), -weight))
    raw.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    hits = []
    last_end = -1
    for start, end, delta in raw:
        if start < last_end:
            continue
        hits.append((start, end, delta))
        last_end = end
    return hits


def score_text(title, content=""):
    """返回 (score, net, pos_hits, neg_hits)。score ∈ [-1, 1]。"""
    text = f"{title} {content}"
    pos = 0
    neg = 0
    for start, end, delta in _collect_hits(text):
        sign = -1 if has_negation(text, start) else 1
        if delta > 0:
            pos += delta * sign
        else:
            neg += -delta * sign
    net = pos - neg
    net = _apply_generic_penalty(net, title, content)
    score = math.tanh(net / 3.0)
    return score, net, pos, neg


def _apply_generic_penalty(net, title, content):
    """资金流/榜单/日报类通用稿降权：标题命中标记或正文含大量股票代码时向中性收拢。"""
    title = title or ""
    content = content or ""
    marker = any(m in title for m in GENERIC_MARKERS)
    codes = len(CODE_RE.findall(content))
    if (marker or codes >= 3) and net > 0:
        return net * 0.35
    return net


def label_of(net, threshold=1.0):
    if net >= threshold:
        return "positive"
    if net <= -threshold:
        return "negative"
    return "neutral"


def score_one(a, llm_cfg=None, method="lexicon"):
    if method == "llm" and llm_cfg:
        res = score_with_llm(a["title"], a["content"], llm_cfg)
    elif method == "ml":
        r = ml.score_text(a["title"], a["content"])
        if r is not None:
            label, score = r
            res = {"label": label, "score": score,
                   "pos_hits": 1 if label == "positive" else 0,
                   "neg_hits": 1 if label == "negative" else 0}
        else:
            score, net, pos, neg = score_text(a["title"], a["content"])
            res = {"label": label_of(net), "score": round(score, 4), "pos_hits": pos, "neg_hits": neg}
    else:
        score, net, pos, neg = score_text(a["title"], a["content"])
        res = {"label": label_of(net), "score": round(score, 4), "pos_hits": pos, "neg_hits": neg}
    row = dict(a)
    row.update(res)
    return row


def score_articles(articles, llm_cfg=None, method="lexicon", workers=4):

    if method == "llm":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda a: score_one(a, llm_cfg=llm_cfg, method=method), articles))
    return [score_one(a, llm_cfg=llm_cfg, method=method) for a in articles]


def score_with_llm(title, content, cfg):
    """走本地 new-api chat completions，返回 {label, score}。失败时退回词典。"""
    key = (title, content[:200])
    if key in _CACHE:
        return _CACHE[key]
    prompt = (
        "你是A股新闻情绪标注器。只输出一行 JSON，格式为 {\"label\":\"positive|negative|neutral\",\"score\":-1到1之间的小数}，"
        "不要输出其他内容。\n标题：" + title + "\n正文：" + content[:600]
    )
    fallback_score, fallback_net, _, _ = score_text(title, content)
    fallback = {"label": label_of(fallback_net), "score": round(fallback_score, 4)}
    try:
        r = requests.post(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {cfg.get('api_key', '')}"},
            json={
                "model": cfg.get("model", "deepseek-v4-flash"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 64,
            },
            timeout=cfg.get("timeout", 60),
        )
        r.raise_for_status()
        content_out = r.json()["choices"][0]["message"]["content"].strip()
        start = content_out.find("{")
        end = content_out.rfind("}") + 1
        data = json.loads(content_out[start:end]) if start >= 0 and end > start else {}
        label = data.get("label", "neutral") if data.get("label") in ("positive", "negative", "neutral") else "neutral"
        score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
        result = {"label": label, "score": round(score, 4)}
    except Exception:
        result = fallback
    _CACHE[key] = result
    return result

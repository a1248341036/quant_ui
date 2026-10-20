# -*- coding: utf-8 -*-
"""按「增量轨」方向批量生成候选因子（2026-10-20）。

设计依据（当天实测）：
- 唯一"高 IC + 高增量"结构是 `vsi2_atr_smooth`（raw 0.0839 / resid 0.0393 / inc 0.469）
  → 核心手法是**在已验证骨架上换波动度量口径**（ret-std → ATR / rank / kurt / 真实波幅）；
- 库内最独立的是低换手 × 波动非对称（`turn8_sg52r_vol3_f` inc 0.581、
  `rq_lowturn_trf_asym_autorescue` 0.566）→ 该方向继续挖；
- 应当避开 illiq/vwap/拥挤族（inc 0.02-0.20，resid≈0）。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("candidates.json")

# ── 波动/风险度量口径（本轮的核心变量）────────────────────────────────
VOL = {
    "std60": "CS_ZSCORE(NEG(TS_STD($ret, 60)))",
    "atr20": "CS_ZSCORE(NEG(TS_MEAN(DIVIDE(SUBTRACT($adj_high,$adj_low), $adj_close), 20)))",
    "atr60": "CS_ZSCORE(NEG(TS_MEAN(DIVIDE(SUBTRACT($adj_high,$adj_low), $adj_close), 60)))",
    "rank60": "CS_ZSCORE(NEG(TS_RANK($ret, 60)))",
    "kurt60": "CS_ZSCORE(NEG(TS_KURT($ret, 60)))",
    "rng20": "CS_ZSCORE(NEG(TS_MEAN(DIVIDE(SUBTRACT($high,$low), $close), 20)))",
}

# ── 非流动性/注意力度量口径 ──────────────────────────────────────────
ILLIQ = {
    "amt60": "CS_ZSCORE(TS_MEAN(DIVIDE(TS_MEAN(DIVIDE(SUBTRACT($high,$low),$close),20), $amount), 60))",
    "to60": "CS_ZSCORE(NEG(TS_MEAN($turnover_rate, 60)))",
    "amihud": "CS_ZSCORE(TS_MEAN(DIVIDE(ABS($ret), $amount), 60))",
}

SMOOTH = {"ema3": 3, "ema8": 8}

rows: list[dict] = []

# ── A. vsi/低波骨架 × 波动度量 × 流动性度量（换口径，最高优先级）──────
for (vk, vexpr), (ik, iexpr), (sk, sw) in itertools.product(VOL.items(), ILLIQ.items(), SMOOTH.items()):
    name = f"incv_{vk}_{ik}_{sk}"
    rows.append({
        "name": name,
        "expr": (f"v = {vexpr}\nil = {iexpr}\n"
                 f"CS_ZSCORE(EMA(ADD(v, il), {sw}))"),
        "why": f"波动度量换口径({vk}) × 流动性度量({ik}) × EMA({sw})；依据 vsi2_atr_smooth 是当天唯一高 IC+高增量结构。",
    })

# ── B. 低换手 × 波动非对称（库内最独立方向）──────────────────────────
ASYM = {
    "updn60": "CS_ZSCORE(SUBTRACT(TS_MEAN(TS_MAX($ret,0), 60), TS_MEAN(ABS(TS_MIN($ret,0)), 60)))",
    "semivar": "CS_ZSCORE(NEG(SUBTRACT(TS_STD(TS_MAX($ret,0), 60), TS_STD(ABS(TS_MIN($ret,0)), 60))))",
}
LOWTO = {
    "to250": "CS_ZSCORE(NEG(TS_MEAN($turnover_rate, 250)))",
    "to120": "CS_ZSCORE(NEG(TS_MEAN($turnover_rate, 120)))",
}
for (ak, aexpr), (tk, texpr) in itertools.product(ASYM.items(), LOWTO.items()):
    for sw in (5, 12):
        rows.append({
            "name": f"inca_{ak}_{tk}_e{sw}",
            "expr": f"a = {aexpr}\nt = {texpr}\nCS_ZSCORE(EMA(ADD(a, t), {sw}))",
            "why": f"波动非对称({ak}) × 低换手({tk}) × EMA({sw})；参照库内最独立的 turn8_sg52r_vol3_f / rq_lowturn_trf_asym_autorescue。",
        })

# ── C. 冷门算子做条件化（新口径下 raw 弱但增量可能够）────────────────
COLD_GATE = {
    "eff": "CS_BUCKET(CS_ZSCORE(NEG(TS_EFFICIENCY_RATIO($adj_close, 60))), 5)",
    "trnd": "CS_BUCKET(CS_ZSCORE(TS_TREND_RANK($amount, 120)), 5)",
    "acorr": "CS_BUCKET(CS_ZSCORE(NEG(TS_AUTOCORR($ret, 120, 1))), 5)",
}
for (gk, gexpr), (vk, vexpr) in itertools.product(COLD_GATE.items(), list(VOL.items())[:3]):
    rows.append({
        "name": f"incc_{vk}_in_{gk}",
        "expr": f"v = {vexpr}\nCS_GROUP_RANK(v, {gexpr})",
        "why": f"低波({vk})在冷门算子({gk})分组内排名：冷门算子裸腿 raw 弱，但做条件化后可能贡献增量。",
    })

# ── D. 事件面融合（筹码集中/解禁）作门控腿 ────────────────────────────
for vk in ("atr20", "rank60", "kurt60"):
    rows.append({
        "name": f"ince_{vk}_holder",
        "expr": (f"v = {VOL[vk]}\n"
                 "h = CS_ZSCORE(NEG(TS_MEAN(CS_WINSORIZE($holder_count_chg_pct,0.01,0.99), 20)))\n"
                 "g = CS_BUCKET(h, 5)\nCS_GROUP_RANK(v, g)"),
        "why": f"低波({vk})在股东户数变化分5组内排名：筹码集中条件化，验证事件腿是否能提供增量。",
    })
    rows.append({
        "name": f"ince_{vk}_unlock",
        "expr": (f"v = {VOL[vk]}\n"
                 "u = CS_ZSCORE(NEG(TS_MEAN(CS_WINSORIZE($unlock_ratio_past180,0.01,0.99), 20)))\n"
                 "g = CS_BUCKET(u, 5)\nCS_GROUP_RANK(v, g)"),
        "why": f"低波({vk})在解禁压力分5组内排名：供给冲击低的环境里低波异象更干净。",
    })

# 去重 + 落盘
seen, uniq = set(), []
for r in rows:
    if r["name"] in seen:
        continue
    seen.add(r["name"])
    uniq.append(r)
OUT.write_text(json.dumps(uniq, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"生成 {len(uniq)} 个因子 -> {OUT}")

# -*- coding: utf-8 -*-
"""批1（pipeline 链路测试）：36 个价量域新假设因子，7 族。"""
import json
from pathlib import Path

LV = {
    "lv60": "CS_ZSCORE(NEG(TS_STD($ret, 60)))",
    "lv20": "CS_ZSCORE(NEG(TS_STD($ret, 20)))",
    "rank60": "CS_ZSCORE(NEG(TS_RANK($ret, 60)))",
    "kurt60": "CS_ZSCORE(NEG(TS_KURT($ret, 60)))",
    "atr20": "CS_ZSCORE(NEG(TS_MEAN(DIVIDE(SUBTRACT($adj_high,$adj_low), $adj_close), 20)))",
}
GAP = "SUBTRACT(DIVIDE($open, DELAY($adj_close, 1)), 1)"
CLV = "DIVIDE(SUBTRACT(SUBTRACT($close,$low),SUBTRACT($high,$close)), SUBTRACT($high,$low))"
PVD = {20: "SUBTRACT(CS_ZSCORE(TS_MEAN($ret, 20)), CS_ZSCORE(TS_MEAN($turnover_rate, 20)))",
       60: "SUBTRACT(CS_ZSCORE(TS_MEAN($ret, 60)), CS_ZSCORE(TS_MEAN($turnover_rate, 60)))"}
RNG = "SUBTRACT($high,$low)"
SQZ = "DIVIDE(TS_MEAN(DIVIDE(" + RNG + ",$close), 5), TS_MEAN(DIVIDE(" + RNG + ",$close), 60))"
CHN = {60: "DIVIDE(SUBTRACT($close, TS_MIN($low, 60)), SUBTRACT(TS_MAX($high, 60), TS_MIN($low, 60)))",
       120: "DIVIDE(SUBTRACT($close, TS_MIN($low, 120)), SUBTRACT(TS_MAX($high, 120), TS_MIN($low, 120)))"}

rows = []
def add(name, expr, why): rows.append({"name": name, "expr": expr, "why": why})

# G. 隔夜跳空结构（5）
add("pl_gap20", f"CS_ZSCORE(NEG(TS_MEAN({GAP}, 20)))", "持续隔夜跳空溢价取负：情绪过热衰减。")
add("pl_gap5", f"CS_ZSCORE(NEG(TS_MEAN({GAP}, 5)))", "短窗跳空，更敏感的过热信号。")
add("pl_gap_std", f"CS_ZSCORE(NEG(TS_STD({GAP}, 60)))", "跳空稳定性取负：跳空稳=定价有序。")
add("pl_gap_lv", f"v = {LV['lv60']}\ng = CS_ZSCORE(NEG(TS_MEAN({GAP}, 20)))\nCS_ZSCORE(EMA(ADD(v, g), 8))", "低波 × 低跳空 融合。")
add("pl_gap_gate", f"v = {LV['lv60']}\ng = CS_BUCKET(CS_ZSCORE(NEG(TS_MEAN({GAP}, 20))), 5)\nCS_GROUP_RANK(v, g)", "低波在低跳空分组内排名。")
# H. 收盘位置 CLV（4）
add("pl_clv20", f"CS_ZSCORE(TS_MEAN({CLV}, 20))", "20 日收盘位置均值：尾盘强势持续性。")
add("pl_clv60", f"CS_ZSCORE(TS_MEAN({CLV}, 60))", "60 日收盘位置：慢变量。")
add("pl_clv_ema", f"c = CS_ZSCORE(TS_MEAN({CLV}, 20))\nCS_ZSCORE(EMA(c, 3))", "CLV 短平滑。")
add("pl_clv_gate", f"v = {LV['lv60']}\nc = CS_BUCKET(CS_ZSCORE(TS_MEAN({CLV}, 20)), 5)\nCS_GROUP_RANK(v, c)", "低波在尾盘强/弱分组内排名。")
# I. 量价背离（4）
add("pl_pvd20", PVD[20], "价动量减量动量：价升量缩的健康度。")
add("pl_pvd60", PVD[60], "60 日量价背离慢变量。")
add("pl_pvd_lv", f"v = {LV['lv60']}\np = {PVD[20]}\nCS_ZSCORE(EMA(ADD(v, p), 8))", "低波 × 量价健康度 融合。")
add("pl_pvd_gate", f"v = {LV['lv60']}\np = CS_BUCKET({PVD[20]}, 5)\nCS_GROUP_RANK(v, p)", "低波在量价背离分组的排名。")
# J. 收敛/挤压（5）
add("pl_squeeze", f"CS_ZSCORE(NEG({SQZ}))", "短期振幅/长期振幅收缩取负：波动挤压后释放。")
add("pl_to_squeeze", f"CS_ZSCORE(NEG(DIVIDE(TS_MEAN($turnover_rate, 5), TS_MEAN($turnover_rate, 60))))", "换手收缩。")
add("pl_squeeze_lv", f"v = {LV['lv60']}\ns = CS_ZSCORE(NEG({SQZ}))\nCS_ZSCORE(EMA(ADD(v, s), 8))", "低波 × 波动挤压 融合。")
add("pl_squeeze_gate", f"v = {LV['lv60']}\ns = CS_BUCKET(CS_ZSCORE(NEG({SQZ})), 5)\nCS_GROUP_RANK(v, s)", "低波在挤压分组的排名。")
add("pl_kurt120", "CS_ZSCORE(NEG(TS_KURT($ret, 120)))", "120 日峰度取负：厚尾风险贴水。")
# K. 通道位置（5）
add("pl_chan60_neg", f"CS_ZSCORE(NEG({CHN[60]}))", "60 日通道位置取负：高位反转型。")
add("pl_chan120", f"CS_ZSCORE({CHN[120]})", "120 日通道位置原值：动量型，符号交由 ML。")
add("pl_chan_lv", f"v = {LV['lv60']}\nc = CS_ZSCORE(NEG({CHN[60]}))\nCS_ZSCORE(EMA(ADD(v, c), 8))", "低波 × 通道低位 融合。")
add("pl_chan_gate", f"v = {LV['lv60']}\nc = CS_BUCKET(CS_ZSCORE({CHN[60]}), 5)\nCS_GROUP_RANK(v, c)", "低波在通道位置分组的排名。")
add("pl_chan60_pos", f"CS_ZSCORE({CHN[60]})", "60 日通道位置原值。")
# L. 换手结构（5）
add("pl_to_cv", "CS_ZSCORE(NEG(TS_STD($turnover_rate, 60)))", "换手波动取负：换手稳定=筹码安定。")
add("pl_to_trend", "CS_ZSCORE(NEG(TS_TREND_RANK($turnover_rate, 120)))", "换手趋势取负：关注度升温衰减。")
add("pl_to_spike", "CS_ZSCORE(NEG(TS_MEAN(SUBTRACT($turnover_rate, TS_MEAN($turnover_rate, 60)), 5)))", "近 5 日换手对 60 日基线的突增取负。")
add("pl_to_lv", f"v = {LV['lv60']}\nt = CS_ZSCORE(NEG(TS_STD($turnover_rate, 60)))\nCS_ZSCORE(EMA(ADD(v, t), 8))", "低波 × 换手安定 融合。")
add("pl_to_gate", f"v = {LV['lv60']}\nt = CS_BUCKET(CS_ZSCORE(NEG(TS_STD($turnover_rate, 60))), 5)\nCS_GROUP_RANK(v, t)", "低波在换手安定分组的排名。")
# M. 骨架融合变奏（8）：换波动度量 × 新腿
add("pl_m_lv20_gap", f"v = {LV['lv20']}\ng = CS_ZSCORE(NEG(TS_MEAN({GAP}, 20)))\nCS_ZSCORE(EMA(ADD(v, g), 5))", "短窗低波 × 低跳空。")
add("pl_m_lv20_clv", f"v = {LV['lv20']}\nc = CS_ZSCORE(TS_MEAN({CLV}, 20))\nCS_ZSCORE(EMA(ADD(v, c), 5))", "短窗低波 × 尾盘强势。")
add("pl_m_atr_pvd", f"v = {LV['atr20']}\np = {PVD[20]}\nCS_ZSCORE(EMA(ADD(v, p), 8))", "ATR 低波 × 量价健康。")
add("pl_m_atr_sqz", f"v = {LV['atr20']}\ns = CS_ZSCORE(NEG({SQZ}))\nCS_ZSCORE(EMA(ADD(v, s), 8))", "ATR 低波 × 挤压。")
add("pl_m_rank_chan", f"v = {LV['rank60']}\nc = CS_ZSCORE(NEG({CHN[60]}))\nCS_ZSCORE(EMA(ADD(v, c), 8))", "秩低波 × 通道低位。")
add("pl_m_kurt_tocv", f"v = {LV['kurt60']}\nt = CS_ZSCORE(NEG(TS_STD($turnover_rate, 60)))\nCS_ZSCORE(EMA(ADD(v, t), 8))", "峰度低波 × 换手安定。")
add("pl_m_lv60_spike", f"v = {LV['lv60']}\nt = CS_ZSCORE(NEG(TS_MEAN(SUBTRACT($turnover_rate, TS_MEAN($turnover_rate, 60)), 5)))\nCS_ZSCORE(EMA(ADD(v, t), 5))", "低波 × 换手突增衰减。")
add("pl_m_lv20_clv60", f"v = {LV['lv20']}\nc = CS_ZSCORE(TS_MEAN({CLV}, 60))\nCS_ZSCORE(EMA(ADD(v, c), 5))", "短窗低波 × 慢 CLV。")

out = Path("artifacts/alphaagent/harness_runs/2026-10-20_increment/pipeline_test_mine.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(rows)} factors -> {out}")

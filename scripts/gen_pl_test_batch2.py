# -*- coding: utf-8 -*-
"""批2（快速路径直喂流水线）：24 个因子 = 批1 族的二阶组合 + 窗口变体。"""
import json
from pathlib import Path

LV20 = "CS_ZSCORE(NEG(TS_STD($ret, 20)))"
LV60 = "CS_ZSCORE(NEG(TS_STD($ret, 60)))"
GAP = "SUBTRACT(DIVIDE($open, DELAY($adj_close, 1)), 1)"
CLV = "DIVIDE(SUBTRACT(SUBTRACT($close,$low),SUBTRACT($high,$close)), SUBTRACT($high,$low))"
PVD = "SUBTRACT(CS_ZSCORE(TS_MEAN($ret, 20)), CS_ZSCORE(TS_MEAN($turnover_rate, 20)))"
RNG = "SUBTRACT($high,$low)"
SQZ = "DIVIDE(TS_MEAN(DIVIDE(" + RNG + ",$close), 5), TS_MEAN(DIVIDE(" + RNG + ",$close), 60))"
CHN = "DIVIDE(SUBTRACT($close, TS_MIN($low, 60)), SUBTRACT(TS_MAX($high, 60), TS_MIN($low, 60)))"
TOCV = "CS_ZSCORE(NEG(TS_STD($turnover_rate, 60)))"

rows = []
def add(name, expr, why): rows.append({"name": name, "expr": expr, "why": why})

# N. 二阶组合（跳空×CLV、挤压×通道、换手×CLV 等）
add("pl_n_gap_clv", f"g = CS_ZSCORE(NEG(TS_MEAN({GAP}, 20)))\nc = CS_ZSCORE(TS_MEAN({CLV}, 20))\nCS_ZSCORE(EMA(ADD(g, c), 8))", "低跳空 × 尾盘强势 融合（无低波腿，纯新域组合）。")
add("pl_n_sqz_chan", f"s = CS_ZSCORE(NEG({SQZ}))\nc = CS_ZSCORE(NEG({CHN}))\nCS_ZSCORE(EMA(ADD(s, c), 8))", "挤压 × 通道低位。")
add("pl_n_tocv_clv", f"t = {TOCV}\nc = CS_ZSCORE(TS_MEAN({CLV}, 20))\nCS_ZSCORE(EMA(ADD(t, c), 8))", "换手安定 × 尾盘强势。")
add("pl_n_pvd_sqz", f"p = {PVD}\ns = CS_ZSCORE(NEG({SQZ}))\nCS_ZSCORE(EMA(ADD(p, s), 8))", "量价健康 × 挤压。")
add("pl_n_gap_tocv", f"g = CS_ZSCORE(NEG(TS_MEAN({GAP}, 20)))\nt = {TOCV}\nCS_ZSCORE(EMA(ADD(g, t), 8))", "低跳空 × 换手安定。")
add("pl_n_clv_chan", f"c = CS_ZSCORE(TS_MEAN({CLV}, 20))\nh = CS_ZSCORE(NEG({CHN}))\nCS_ZSCORE(EMA(ADD(c, h), 8))", "尾盘强势 × 通道低位。")
# O. 批1 融合骨架的三腿变体
add("pl_o_lv_gap_clv", f"v = {LV60}\ng = CS_ZSCORE(NEG(TS_MEAN({GAP}, 20)))\nc = CS_ZSCORE(TS_MEAN({CLV}, 20))\nCS_ZSCORE(EMA(ADD(ADD(v, g), c), 8))", "低波 + 低跳空 + 尾盘强势 三腿。")
add("pl_o_lv_sqz_tocv", f"v = {LV60}\ns = CS_ZSCORE(NEG({SQZ}))\nt = {TOCV}\nCS_ZSCORE(EMA(ADD(ADD(v, s), t), 8))", "低波 + 挤压 + 换手安定 三腿。")
add("pl_o_lv20_pvd_gap", f"v = {LV20}\np = {PVD}\ng = CS_ZSCORE(NEG(TS_MEAN({GAP}, 5)))\nCS_ZSCORE(EMA(ADD(ADD(v, p), g), 5))", "短窗三腿。")
add("pl_o_lv_pvd_chan", f"v = {LV60}\np = {PVD}\nc = CS_ZSCORE(NEG({CHN}))\nCS_ZSCORE(EMA(ADD(ADD(v, p), c), 8))", "低波 + 量价 + 通道 三腿。")
# P. 窗口变体（批1 最简单腿的替代窗口）
add("pl_gap60", f"CS_ZSCORE(NEG(TS_MEAN({GAP}, 60)))", "60 日慢窗跳空。")
add("pl_gap_std20", f"CS_ZSCORE(NEG(TS_STD({GAP}, 20)))", "20 日跳空波动。")
add("pl_clv120", f"CS_ZSCORE(TS_MEAN({CLV}, 120))", "120 日慢窗 CLV。")
add("pl_clv_std", f"CS_ZSCORE(NEG(TS_STD({CLV}, 60)))", "CLV 稳定性取负。")
add("pl_squeeze60", f"s = DIVIDE(TS_MEAN(DIVIDE({RNG},$close), 10), TS_MEAN(DIVIDE({RNG},$close), 120))\nCS_ZSCORE(NEG(s))", "10/120 窗挤压。")
add("pl_chan40", f"c = DIVIDE(SUBTRACT($close, TS_MIN($low, 40)), SUBTRACT(TS_MAX($high, 40), TS_MIN($low, 40)))\nCS_ZSCORE(NEG(c))", "40 日通道位置反转型。")
add("pl_pvd20_5", f"SUBTRACT(CS_ZSCORE(TS_MEAN($ret, 5)), CS_ZSCORE(TS_MEAN($turnover_rate, 5)))", "5 日短窗量价背离。")
add("pl_to_cv120", "CS_ZSCORE(NEG(TS_STD($turnover_rate, 120)))", "120 日换手波动。")
# Q. 门控稀疏变体（批1 单腿门控的换腿）
add("pl_q_gap_gate_lv20", f"v = {LV20}\ng = CS_BUCKET(CS_ZSCORE(NEG(TS_MEAN({GAP}, 20))), 5)\nCS_GROUP_RANK(v, g)", "短窗低波 × 跳空门控。")
add("pl_q_sqz_gate_lv20", f"v = {LV20}\ns = CS_BUCKET(CS_ZSCORE(NEG({SQZ})), 5)\nCS_GROUP_RANK(v, s)", "短窗低波 × 挤压门控。")
add("pl_q_chan_gate_lv20", f"v = {LV20}\nc = CS_BUCKET(CS_ZSCORE({CHN}), 5)\nCS_GROUP_RANK(v, c)", "短窗低波 × 通道门控。")
add("pl_q_clv_gate_lv20", f"v = {LV20}\nc = CS_BUCKET(CS_ZSCORE(TS_MEAN({CLV}, 20)), 5)\nCS_GROUP_RANK(v, c)", "短窗低波 × CLV 门控。")
add("pl_q_pvd_gate_lv20", f"v = {LV20}\np = CS_BUCKET({PVD}, 5)\nCS_GROUP_RANK(v, p)", "短窗低波 × 量价门控。")
add("pl_q_tocv_gate_lv20", f"v = {LV20}\nt = CS_BUCKET({TOCV}, 5)\nCS_GROUP_RANK(v, t)", "短窗低波 × 换手安定门控。")

out = Path("artifacts/alphaagent/harness_runs/2026-10-20_increment/pipeline_test2_rows.jsonl")
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print(f"{len(rows)} factors -> {out}")

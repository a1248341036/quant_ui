# -*- coding: utf-8 -*-
"""非价量数据域因子批次（2026-10-20 数据面审计后第一轮）。

可用列（探查 probe_nonprice.py，2020-2024 覆盖率）：
  ann_cnt_20d/flag_forecast/flag_buyback/flag_lawsuit/ann_days_since  ~100%
  ff_super/large/medium/small_net  99.9%   th_top10_pct/th_inst_pct  95%
  pred_surprise/pred_change_mid/pred_net_profit_mid/pred_direction  84-88%
  unlock_ratio_past180  80%   inst_fund_ratio  69%   mgn_short_balance  56%
设计约束：
  - 只用已验证算子（TS_MEAN/TS_STD/EMA/CS_ZSCORE/CS_WINSORIZE/CS_BUCKET/CS_GROUP_RANK/DELAY/ADD/SUBTRACT/DIVIDE/NEG/ABS）
  - 禁止 ff_* 上做分组算子（2026-09-29 chip_grp_ff_gate 604s 超时教训）
  - 骨架复用：低波反转（库内收益引擎）× 新域信息腿（融合或门控）
"""
import json, sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("nonprice_rows.jsonl")

LV = {
    "lv60": "CS_ZSCORE(NEG(TS_STD($ret, 60)))",
    "lv20": "CS_ZSCORE(NEG(TS_STD($ret, 20)))",
}
rows: list[dict] = []
def add(name, expr, why):
    rows.append({"name": name, "expr": expr, "why": why})

# ── A. 业绩预告/业绩超预期（88% 覆盖，事件慢变量）─────────────────────
add("np_pred_surp_ma20",
    "CS_ZSCORE(TS_MEAN(CS_WINSORIZE($pred_surprise,0.01,0.99), 20))",
    "业绩预告超预期的 20 日均值：预告后漂移（PEAD 变体），事件信息慢衰减。")
add("np_pred_chg_z",
    "CS_ZSCORE(CS_WINSORIZE($pred_change_mid,0.01,0.99))",
    "业绩预告中值修正幅度（去极值截面标准化）：修正方向信息。")
add("np_pred_np_z",
    "CS_ZSCORE(CS_WINSORIZE($pred_net_profit_mid,0.01,0.99))",
    "预告净利润中值（规模归一由截面 z 承担）。")
add("np_pred_dir_ma60",
    "CS_ZSCORE(TS_MEAN($pred_direction, 60))",
    "预告方向（编码未知，交由 ML 学方向）60 日均值：预告基调持续性。")
add("np_pred_lv_gate",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "g = CS_BUCKET(CS_ZSCORE(TS_MEAN(CS_WINSORIZE($pred_surprise,0.01,0.99), 20)), 5)\n"
    "CS_GROUP_RANK(v, g)",
    "低波在业绩超预期分 5 组内排名：超预期环境里的低波异象更干净（事件腿条件化）。")

# ── B. 公告事件（~100% 覆盖，公告密度/新近度/类型）────────────────────
add("np_ann_cnt20",
    "CS_ZSCORE(NEG(TS_MEAN($ann_cnt_20d, 20)))",
    "公告密度 20 日均值取负：少公告=少噪音/低关注度。")
add("np_ann_recency",
    "CS_ZSCORE(NEG($ann_days_since))",
    "公告新近度：距上次公告天数取负，近期有公告=信息活跃。")
add("np_ann_buyback",
    "CS_ZSCORE(TS_SUM($ann_flag_buyback, 60))",
    "60 日回购公告次数：回购=管理层信号（稀疏事件用 TS_SUM 累计）。")
add("np_ann_lawsuit",
    "CS_ZSCORE(NEG(TS_SUM($ann_flag_lawsuit, 60)))",
    "60 日诉讼公告次数取负：诉讼=负面事件。")
add("np_ann_lv_gate",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "g = CS_BUCKET(TS_SUM($ann_flag_buyback, 120), 3)\n"
    "CS_GROUP_RANK(v, g)",
    "低波在有回购公告分组内排名：回购背书下的低波。")

# ── C. 解禁供给冲击（80% 覆盖）────────────────────────────────────────
add("np_ulk_past",
    "CS_ZSCORE(NEG(TS_MEAN(CS_WINSORIZE($unlock_ratio_past180,0.01,0.99), 20)))",
    "过去 180 天解禁比例取负：刚解禁完=抛压释放。")
add("np_ulk_lv_blend",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "u = CS_ZSCORE(NEG(TS_MEAN(CS_WINSORIZE($unlock_ratio_past180,0.01,0.99), 20)))\n"
    "CS_ZSCORE(EMA(ADD(v, u), 8))",
    "低波 × 解禁释放 融合（库内最有效骨架=融合而非门控）。")

# ── D. 机构持仓（仅 inst_fund_ratio 69% 可用）─────────────────────────
add("np_inst_fund_z",
    "CS_ZSCORE(TS_MEAN($inst_fund_ratio, 20))",
    "基金持仓比例水平：机构重仓=质量代理。")
add("np_inst_fund_chg",
    "CS_ZSCORE(SUBTRACT(CS_ZSCORE($inst_fund_ratio), CS_ZSCORE(DELAY($inst_fund_ratio, 60))))",
    "基金持仓 60 日变化（持仓动量）：机构加减仓方向。")
add("np_inst_lv_gate",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "g = CS_BUCKET(CS_ZSCORE($inst_fund_ratio), 5)\n"
    "CS_GROUP_RANK(v, g)",
    "低波在基金持仓分 5 组内排名。")

# ── E. 十大流通股东（95% 覆盖）────────────────────────────────────────
add("np_th_inst",
    "CS_ZSCORE(NEG($th_inst_pct))",
    "十大流通中机构占比取负（拥挤度代理：机构扎堆→拥挤）。")
add("np_th_top10_chg",
    "CS_ZSCORE(SUBTRACT(CS_ZSCORE($th_top10_pct), CS_ZSCORE(DELAY($th_top10_pct, 120))))",
    "十大流通集中度 120 日变化：筹码集中/分散动量。")
add("np_th_lv_gate",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "g = CS_BUCKET(CS_ZSCORE($th_inst_pct), 5)\n"
    "CS_GROUP_RANK(v, g)",
    "低波在机构占比分 5 组内排名。")

# ── F. 资金流四级拆分（99.9% 覆盖；禁分组算子）────────────────────────
add("np_ff_super_ratio",
    "CS_ZSCORE(TS_MEAN(DIVIDE(CS_WINSORIZE($ff_super_net,0.01,0.99), $amount), 20))",
    "超大单净流入/成交额 20 日均值：主力资金方向。")
add("np_ff_all_ratio",
    "CS_ZSCORE(TS_MEAN(DIVIDE(CS_WINSORIZE(ADD(ADD($ff_super_net,$ff_large_net),ADD($ff_medium_net,$ff_small_net)),0.01,0.99), $amount), 20))",
    "四档净流入合计/成交额：整体资金净方向。")
add("np_ff_diverg",
    "CS_ZSCORE(TS_MEAN(DIVIDE(CS_WINSORIZE(SUBTRACT($ff_super_net,$ff_small_net),0.01,0.99), $amount), 20))",
    "超大单-小单净流入差/成交额：主力与散户分歧度。")

# ── G. 两融（56% 覆盖，勉强过 notna≥0.5 过滤）─────────────────────────
add("np_mgn_short_z",
    "CS_ZSCORE(TS_MEAN(CS_WINSORIZE($mgn_short_balance,0.01,0.99), 20))",
    "融券余额水平：做空压力代理。")

# ── H. 非价量 × 低波反转融合（t3_hd 股东户数×反转是库内最强增量的推广）──
add("np_blend_pred",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "p = CS_ZSCORE(TS_MEAN(CS_WINSORIZE($pred_surprise,0.01,0.99), 20))\n"
    "CS_ZSCORE(EMA(ADD(v, p), 8))",
    "低波 × 业绩超预期 融合。")
add("np_blend_ann",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "a = CS_ZSCORE(NEG($ann_days_since))\n"
    "CS_ZSCORE(EMA(ADD(v, a), 8))",
    "低波 × 公告新近度 融合。")
add("np_blend_th",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "t = CS_ZSCORE(SUBTRACT(CS_ZSCORE($th_top10_pct), CS_ZSCORE(DELAY($th_top10_pct, 120))))\n"
    "CS_ZSCORE(EMA(ADD(v, t), 8))",
    "低波 × 筹码集中动量 融合。")
add("np_blend_ff",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 60)))\n"
    "f = CS_ZSCORE(TS_MEAN(DIVIDE(CS_WINSORIZE($ff_super_net,0.01,0.99), $amount), 20))\n"
    "CS_ZSCORE(EMA(ADD(v, f), 8))",
    "低波 × 主力净流入 融合。")
add("np_blend_lv20_inst",
    "v = CS_ZSCORE(NEG(TS_STD($ret, 20)))\n"
    "i = CS_ZSCORE(SUBTRACT(CS_ZSCORE($inst_fund_ratio), CS_ZSCORE(DELAY($inst_fund_ratio, 60))))\n"
    "CS_ZSCORE(EMA(ADD(v, i), 5))",
    "短窗低波 × 基金加减仓 融合（更快骨架）。")

seen, out = set(), []
for r in rows:
    if r["name"] in seen:
        continue
    seen.add(r["name"])
    out.append(r)
with open(OUT, "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"生成 {len(out)} 个非价量域因子 -> {OUT}")

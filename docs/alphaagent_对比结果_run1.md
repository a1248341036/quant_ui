# AlphaAgent 原版 vs quant_ui 对比报告（run1，单次试验）

> 2026-09-06。同面板（全 A 5,560 只 OHLCV+daily_basic，train 2020-2022 / val 2023-2024）、
> 同 LLM（GLM-5.3-Flash 本地代理）、同 max_turns 预算、双方空库起跑。
> 盲测段 2025-01-02 → 2026-09-03（两侧挖掘链路均不可见），quant_ui 引擎统一裁决。
> 详细设计见 `docs/alphaagent_对比实验设计.md`。

## 1. TL;DR

- **盲测质量**：原版 arm 的 5 个自选提交因子全部在盲测段保持强信号（IC 0.040~0.050，
  多空年化 +36%~+51%）；quant_ui arm 的 top8 盲测 IC 0.035~0.050，多空年化 +17%~+42%。
  **原版侧整体略优**，但其产物经过"LLM 自选提交"的选择（quant_ui 侧无任何因子过海选，
  只能取评估池 top8），存在轻微选择不对称——按对称的评估池 top8 对比，两侧差距收窄。
- **产出效率**：原版 57 分钟产出 75 个独特评估（1.3 个/分钟）；quant_ui 85 分钟产出
  16 个（0.19 个/分钟）——quant_ui 的记忆注入/预测对账/val 评估使单因子评估成本 ~5 倍。
- **门槛行为**：quant_ui 两阶段门槛把全部 16 个因子拦在海选外（|IC|≥0.02 且 |ICIR|>0.25），
  但被拦的因子盲测段并不差（如 turn_ws_neut 盲测 ICIR 0.252）——2020-2022 训练段对
  低波/拥挤类因子偏"小年"，**训练段门槛与盲测表现存在错配**，这是比"谁更强"更有价值的发现。
- **引擎一致性交叉验证**：同一表达式（lowvol_20 / turn_level_20）在两 arm 中出现，
  盲测指标逐位一致——两侧引擎同面板数值一致的 sanity check 通过。

## 2. 盲测裁决明细（2025-01-02 → 2026-09-03，label_1d_close_to_close）

### 原版 arm — LLM 自选提交（5 个）

| 因子 | IC | ICIR | 多空年化 | 多空夏普 |
|---|---|---|---|---|
| combo_z4_dualwin_corr_size_neut | +0.0500 | 0.376 | +51.1% | 2.28 |
| dualwin_corr_crowd_v1 | +0.0500 | 0.376 | +51.1% | 2.28 |
| neg_corr_vwap_vol20 | +0.0402 | 0.391 | +49.3% | 3.24 |
| combo_z3_semivol_size_neut | +0.0451 | 0.308 | +36.0% | 1.56 |
| combo_z3_smooth_vwap | +0.0439 | 0.267 | +35.7% | 1.40 |

### 原版 arm — 评估池 top8

| 因子 | IC | ICIR | 多空年化 | 多空夏普 |
|---|---|---|---|---|
| combo_z3_kline | +0.0515 | 0.255 | +25.2% | 0.98 |
| lowvol20_neut_rank | +0.0497 | 0.242 | +31.1% | 0.89 |
| combo_z_winsor | +0.0499 | 0.270 | +31.7% | 1.18 |
| combo_lowvol_pvrc | +0.0490 | 0.261 | +26.7% | 1.08 |
| lowvol_20 | +0.0456 | 0.203 | +23.4% | 0.72 |
| kline_geometry_20（反向） | -0.0464 | -0.212 | -13.1%* | — |

*负面 IC 因子按反方向同样可用（|-0.0464|），多空列按原始符号。

### quant_ui arm — 评估池 top8（海选全拦，无提交）

| 因子 | IC | ICIR | 多空年化 | 多空夏普 | train 段为何被拦 |
|---|---|---|---|---|---|
| turn_ws_neut | +0.0488 | 0.252 | +41.9% | 1.41 | ICIR 0.165<0.25 |
| ivol_resid_20 | +0.0500 | 0.241 | +31.2% | 0.89 | IC 0.0163<0.02 |
| turnover_anomaly_20 | +0.0385 | 0.175 | +41.0% | 1.08 | ICIR 0.178<0.25 |
| low_vol_20 | +0.0456 | 0.203 | +23.4% | 0.72 | IC 0.0177<0.02 |
| low_vol_60 | +0.0384 | 0.162 | +18.8% | 0.61 | IC 0.0150<0.02 |
| lowvol_120 | +0.0346 | 0.145 | +17.3% | 0.58 | IC 0.0137<0.02 |
| turn_level_20（反向） | -0.0385 | -0.175 | -38.7%* | — | 同上 |

## 3. 效率对比

| 指标 | 原版 arm | quant_ui arm |
|---|---|---|
| 墙钟 | 57 min | 85 min |
| LLM 调用 | （不记录） | 23 次，输入 808K tok（cache 命中 46%），输出 158K tok |
| 工具调用 | 114 次（86 ok） | 73 次（57 ok / 16 错：7 参数错 + 4 求值错 + 5 盲测段拦截） |
| 独特评估数 | **75**（1.3/分钟，单次 ~9s） | 16（0.19/分钟，单次 ~70-80s） |
| 提交 | 25 次尝试 / 5 独特（全被 `factorlib_not_initialized` 拦，见 §5） | 1 次尝试（未过门槛族） |
| 入库 | 0（配置失误，非门槛） | 0（海选门槛全拦） |

quant_ui 单因子评估慢 5 倍的原因：记忆检索注入、prediction 对账、附加 transforms/metrics、
（本轮未跑 val）。这是"每步更重"与"步数更多"的取舍——但**本轮 quant_ui 的重投入没有兑换成
更高的因子质量**（盲测不占优），说明 2026-09 上线的认知层升级（预测对账/机制知识）在本
窗口与该数据面上尚未体现净收益，需要多 run 验证。

## 4. 关键 caveat

1. **k=1 试验**：论文用 20 次试验择优；单次 run 噪声大，方向性结论需 3~5 run 才置信。
2. **选择不对称**：原版侧裁决含 LLM 自选提交（每轮可挑最优），quant_ui 侧全是评估池
   截断——对称口径（评估池 top8）下原版 IC 均值 0.049 vs quant_ui 0.043（正方向），
   差距明显小于含提交口径。
3. **训练段错配**：两 arm 在 train 段（2020-2022）最高 rank_ic 均 ~0.053，但海选门槛
   需要 |IC|≥0.02 + ICIR>0.25 同时成立；盲测段（2025-2026）低波/反转/拥挤类因子整体
   进入大年（IC 普遍 0.04+）。**2025-2026 的行情结构对这类因子显著有利**，单窗口
   对比受市场状态主导，不能完全归因于系统差异。
4. **本次盲测为一次性**（多重检验约束），后续补 run 后只对新增因子追加裁决。

## 5. 实验配置失误记录（透明起见）

- 原版 arm 的空库 `compare_stock_1d` 未跑 `init_factorlib.py` 初始化，25 次 submit 全部
  被 `factorlib_not_initialized` 拒绝——原版 arm 实际上"带伤比赛"（无入库），但评估与
  提交表达式均完整保留在轨迹中，裁决不受影响。复跑时先 init。
- 原版 run1 因启动方式（管道缓冲阻塞 stdout）卡死，改 `--quiet` + 文件重定向后为 run2。
- 首轮原版 arm 与 quant_ui arm 均未产生正式入库因子——**对比口径为 best-of-run 盲测**，
  而非"交付链路产物"对比。

## 6. 复现

```bash
# 数据导出（2020-2024，盲测段 2025+ 不导出）
.venv/Scripts/python.exe scripts/export_panel_for_original_alphaagent.py
# 原版 arm（先初始化空库！）
cd /d/Quant/AlphaAgent && uv run python scripts/init_factorlib.py --root artifacts/factorzoo/compare_stock_1d
D:/Quant/quant_ui_cmp/scripts/run_original_alphaagent.py --tag runN --factorlib <空库> --no-fundamentals --quiet > console.log 2>&1
# quant_ui arm
.venv/Scripts/python.exe scripts/run_alphaagent.py --panel D:/Quant/AlphaAgent/artifacts/panel/panel_1d.parquet \
  --factorlib artifacts/alphaagent/factorzoo/compare_prod --train-start 2020-01-01 --train-end 2022-12-31 \
  --val-start 2023-01-01 --val-end 2024-12-31 --label-col label_1d_close_to_close --no-fundamentals \
  --research-memory-file artifacts/alphaagent/comparison/memory_quantui.db --log-dir <dir>
# 表达式提取 + 盲测裁决
.venv/Scripts/python.exe scripts/extract_quantui_comparison_exprs.py
.venv/Scripts/python.exe scripts/alphaagent_blind_judge.py --expr-dir <...> --out <report.json>
```

# ML 组合基线存档（2026-09-12 15:53:35）

> 本文档存档 Phase 0 ML 组合链路的第一个完整基线。报告原始数据见
> `artifacts/alphaagent/stacking/20260912_155335/report.json`（本分支修复前同一份输出）。
> 修复提交 a08bcf7 只解决 Windows GBK 打印崩溃，数值口径与本文完全一致。

## 1. 运行环境与口径

| 项 | 值 |
|---|---|
| run_id | `20260912_155335` |
| 时间隔离 | holdout：`mining_end=2024-12-31` 起为干净留出段（盲测段），训练折含挖掘期数据用于拟合 |
| panel 区间 | 2023-04-25 ~ 2026-09-11 |
| label | 5 日（`label_days=5`），score_smooth=5 |
| folds | 4（walk-forward，滚动扩窗） |
| 方案 | ML 学习加权（Ridge + LightGBM），scheme=`ml` |
| 因子池 | **观察池口径**重建后 16 条（技术+基本面混合），枚举 18、冗余剔除 4 |

枚举 18 条中 4 条因截面相关 > 0.6 被贪心冗余过滤剔除，进 ML 的 **14 条**：

```
funda_ocfps_x_price_blend3p_cd5_ema3   ov_gap_w20_q05_e3
funda_oqual_x_lhbuy_gate_t90           conclvl_unstarted_diverge
ovdiv_sub_szneut_wma20                 ovdiv_div17_wnsz_neut
peakclear_mom10_pw                     holderlock_x_vol_ratio
holder_conc_x_price_div_v5             vp_corr_x_roe_grouprank
predsurp_x_lowattn_gate                illiq_x_vwapprem_div_sm3
chip_illiq_ac3_s3                      illiq_vw3_volac120_ema6
```

剔除：`gaprank60_x_turn120_w25_x_volume`（冗余于 funda_ocfps…）、
`illiqgrp8_prem240_div_ema5`、`illiqgrp5_prem240_div_ema5`（均冗余于
illiq_x_vwapprem_div_sm3）、`ovdiv_vwap_wma20_noms`（冗余于 ovdiv_div17_wnsz_neut）。

## 2. 盲测段（OOS）统计表现

盲测段 2025-01-27 ~ 2026-09-11，共 389 个交易日，4 折滚动。

### 2.1 各折 LightGBM OOS

| 折 OOS 区间 | n_oos | IC | ICIR | 多空日价差 |
|---|---|---|---|---|
| 2025-01-27 ~ 2025-07-24 | 632k | 0.0944 | 1.382 | +0.0109 |
| 2025-07-25 ~ 2026-01-23 | 665k | 0.0456 | 0.393 | +0.0016 |
| 2026-01-26 ~ 2026-07-24 | 653k | 0.0171 | 0.145 | −0.0002 |
| 2026-07-27 ~ 2026-09-11 | 150k | 0.0871 | 0.370 | +0.0105 |

### 2.2 混合口径（同一 OOS 行集）

| 方案 | IC_mean | ICIR | OOS Sharpe | 最大回撤 |
|---|---|---|---|---|
| 当前组合（ridge+LGBM 加权） | **0.0549** | 0.439 | 1.719 | −23.4% |
| 等权 1/N | 0.0637 | 0.463 | 1.861 | −18.7% |
| **ICIR 加权** | **0.0762** | **0.520** | **2.029** | **−17.5%** |
| HRP 风险平价 | 0.0486 | 0.453 | 1.869 | −19.3% |

> **关键结论**：简单加权（等权 / ICIR 加权/ HRP）整体优于 Ridge+LGBM 拟合加权。
> 14 因子池太小，LGBM 学到的是噪声权重；ICIR 加权在最简实现下拿到
> **OOS IC 0.076 / ICIR 0.52 / Sharpe 2.03**，是全口径最优。

### 2.3 单因子衰减（mining → OOS）

| 因子 | 来源库 | IC_mining | IC_oos | 保留比 |
|---|---|---|---|---|
| funda_ocfps_x_price_blend3p_cd5_ema3 | production | +0.0662 | +0.0554 | 0.84 |
| conclvl_unstarted_diverge | candidate | +0.0678 | +0.0598 | 0.88 |
| ov_gap_w20_q05_e3 | candidate | −0.0334 | −0.0332 | 0.99 |
| funda_oqual_x_lhbuy_gate_t90 | candidate | −0.0283 | −0.0299 | 1.05 |

全表多数因子保留比落在 0.7~1.4 区间：**单因子跨段衰减不严重**，统计端没有系统性过拟合信号。

## 3. 可交易性（engine_gate）——死穴在交易端

gate `enabled=true`，`passed=false`。失败原因：`excess_sharpe`、`max_drawdown`。

| 项 | 基线值 | 门槛 | 结论 |
|---|---|---|---|
| 净超额年化 | +13.55% | ≥ 3% | ✅ 远超 |
| 超额夏普 | 0.4988 | ≥ 0.5 | ❌ 差 0.0012 |
| 最大回撤 | −41.3% | ≤ 40% | ❌ 超 1.3pp |
| 平均持仓数 | **3.4 只** | — | ⚠️ 极窄 |
| 日均换手 | **62.3%** | — | ⚠️ 极高 |
| 拒单数 | **71**（现金不足/预算过小 60） | — | ⚠️ 极多 |

**根因**：组合引擎当前复用 `default_research_spec(...)` 的 engine_gate 口径，`selection_pct=0.001`（≈5 只）——这是**单因子门禁的极窄选股宽度**。合成分数本身 alpha 充足（IC 0.055+、超额年化 +13.6%），但被错误的选股宽度逼进 top-5 死胡同：buy-and-hold 变 3.4 只、周换手 62%、60 次现金不足拒单、回撤失控。

不是统计端问题，是**门禁选股宽度口径用错**。

## 4. 结论与 Phase 1 路线

1. **统计端已达标**：IC 0.02~0.07 是单因子物理上限，池级合成盲测 OOS 0.055~0.076 是真实提升；ICIR 加权最优。
2. **池子太小**：18 枚举 → 14 因子，LGBM < 简单加权；先扩大因子库再谈拟合加权。
3. **死穴在交易端**：`excess_sharpe=0.4988` 仅差 0.0012，回撤 −41% 由窄持仓+高换手+现金拒单连锁造成。
4. **Phase 1 改造（已实现，`scripts/train_ml_composite.py`）**：组合引擎门禁改用**组合口径选股宽度**——新增 `--gate-selection-pct`（默认 `trading_config.SELECTION_PCT=0.004`，≈20 只）与 `--gate-top-n`（固定 Top-N 模式，缺省关闭）；门禁门槛（`min_excess_annual 0.03 / excess_sharpe 0.5 / max_drawdown 0.4` 等）一律不动。验证命令：
   ```
   python scripts/train_ml_composite.py --model both --train-months 18 --step-months 6
   # 组合口径 gate（selection_pct=0.004）跑通后，门禁通过即代表可交易性达标
   ```

> 备注：本基线基于**观察池口径**候选池（16 条）。`feat/public-access-key` 侧的旧预筛池门槛重建后池子不同，merge 时需按观察池口径重新对账。
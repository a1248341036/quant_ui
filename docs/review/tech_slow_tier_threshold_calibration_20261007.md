# 慢档门槛标定（technical_weekly / technical_monthly）· 2026-10-07

## 1. 问题

`technical_weekly`(label_5d+weekly) 与 `technical_monthly`(label_20d+monthly) 此前**只对齐了
label 与调仓频率**（2026-10-04 的 fail-closed 一致性规则），**没有自己的统计门槛**——它们继承
按 **label_1d 尺度**标定的 `min_abs_ic=0.02` / `min_icir=0.28` / `min_val_abs_ic=0.015`
（`delivery_criteria.py` 的【唯一真源】默认值）。

而 label 持有期会系统性改变同一因子的 IC/ICIR 量级：慢 label 的收益更可预测、且 ICIR 按持有期
去重叠重采样（`cs_ic_summary(holding_days)`）后观测点更少 ⇒ 沿用 1d 线会让慢档门槛**近乎失效**
（选择性失衡，慢档流入量远超快档）。`core/research_modes.py:65-69` 的历史注释也写明"将来若要
weekly/monthly 交付，走 label_5d/label_20d 的档位并**重标定门槛**"——本文即补上这一步。

## 2. 方法与数据

- 脚本：`scripts/calibrate_slow_tier_thresholds.py`（只读：不落库、不改 registry）
- 设计：**配对**测量——同一批因子（候选池 42 个表达式）在 label_1d / label_5d / label_20d 三个
  会话上各跑一遍 train 评估（2020-01-01~2022-12-31，`include_fundamentals=True`），
  取每个因子 |IC|、|ICIR| 的三档配对值
- 有效配对：**n=41**（1 个因子在三档均报 `ic=None`，剔除）
- 耗时：约 35 分钟（126 次评估，串行）
- 标定规则：**慢档门槛 = 主档门槛 × 配对中位比值**，使各档在同一因子池上的过线率相当

## 3. 实测结果

| label | \|IC\| p50 | \|IC\| p90 | \|ICIR\| p50 | \|ICIR\| p90 |
|---|---|---|---|---|
| label_1d_open_to_open | 0.0205 | 0.0261 | 0.2698 | 0.3397 |
| label_5d_close_to_close | 0.0312 | 0.0459 | 0.4351 | 0.6024 |
| label_20d_close_to_close | 0.0496 | 0.0742 | 0.7432 | 1.0024 |

**配对中位比值（标定放大系数）**

| 档 | \|IC\| 比值 | \|ICIR\| 比值 |
|---|---|---|
| label_5d / 1d | **1.776** | **1.800** |
| label_20d / 1d | **2.865** | **3.032** |

## 4. 落地的门槛（`core/research_modes.py` 两档 override）

| 项 | technical(1d) | technical_weekly(5d) | technical_monthly(20d) |
|---|---|---|---|
| evaluation `min_train_abs_ic` | 0.02 | **0.0355** | **0.0573** |
| evaluation `min_train_icir` | 0.28 | **0.504** | **0.849** |
| evaluation `min_val_abs_ic` | 0.015 | **0.0266** | **0.0430** |
| candidate `min_abs_ic` | 0.02 | **0.0355** | **0.0573** |
| candidate `min_icir` | 0.28 | **0.504** | **0.849** |
| candidate `min_val_abs_ic` | 0.015 | **0.0266** | **0.0430** |
| production `min_train_abs_ic` | 0.025 | **0.0444** | **0.0716** |
| production `min_train_icir` | 0.30 | **0.540** | **0.9096** |
| production `min_val_abs_ic` | 0.015 | **0.0266** | **0.0430** |

**刻意不动**（无 label 尺度依赖）：`min_val_ic_retention`（比值口径）、`min_coverage` /
`min_cs_autocorr` / `max_abs_corr`（因子侧属性）、`engine_gate.min_excess_annual` /
`min_excess_sharpe`（年化口径，**缺该档实测数据**，暂继承主档；`freq`/`allowed_freqs` 原本就按档设置）。
换手门槛本就按频率分档（`turnover_thresholds_by_freq={daily:0.50, weekly:0.65, monthly:0.80}`），未改动。

## 5. 不变量与回归

`tests/test_slow_tier_thresholds.py` 锁定：
1. 慢档值 == 主档基准 × 实测比值（防止后人手改漂移）；
2. 三档 IC/ICIR 门槛严格递增（1d < 5d < 20d）；
3. 与 label 无关的项在三档间完全一致；
4. engine_gate 年化门槛仍继承主档；
5. `ensure_label_freq_consistency` 对所有档位仍通过（不破坏 2026-10-04 强制一致）。

受影响并已通过的既有测试：`test_label_freq_consistency.py`、`test_prompt_consistency_audit.py`、
`test_engine_gate_buffer_band.py`、`test_auto_mode_and_freq.py`、`test_unified_library.py`。

## 6. 已知局限 / 后续

- 样本为**单个 train 窗口**（2020-2022）+ 候选池 42 个因子（偏技术族、且是"未过线候选"池），
  慢档真实 run 数据积累后应复核；
- 未测量 **val 段**的尺度比（用 train 比值近似）；val 保留比是比值口径，本身与尺度无关；
- `engine_gate` 的年化门槛（0.03/0.5）未按档调整——需要慢档 engine_gate 实测（真实回测的
  年化超额/夏普分布）才能标定，属独立工作；
- 复核方法：重跑 `scripts/calibrate_slow_tier_thresholds.py`，若中位比值变动 >10% 则应更新本表
  与 `core/research_modes.py` 的注释与数值。

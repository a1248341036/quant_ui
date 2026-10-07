# 慢档门槛标定（technical_weekly / technical_monthly）· 2026-10-07（v2）

> v1（commit `364993a`）只按"配对实测比值等比放大"保持选择性，结果把绝对值推到文献罕见区间
> （monthly 候选线 0.0573/0.849、正式库线 0.0716/0.9096 ⇒ 池内过线率仅 22%，且高于文献实测
> 最强的经典因子）。v2 改为**双锚**：文献绝对量级 + 与主档选择性持平。本文即 v2 记录。

## 1. 问题

`technical_weekly`(label_5d+weekly) / `technical_monthly`(label_20d+monthly) 此前**只对齐了
label 与调仓频率**（2026-10-04 fail-closed 一致性规则），统计门槛仍继承按 **label_1d 尺度**
标定的 `min_abs_ic=0.02` / `min_icir=0.28` / `min_val_abs_ic=0.015`。而 label 持有期会系统性
改变同一因子的 IC/ICIR 量级 ⇒ 沿用 1d 线会让慢档门近乎失效。

## 2. 外部基准（文献/业界口径，2026-10-07 检索）

| 来源 | 口径 |
|---|---|
| paperswithbacktest（IC 分档） | >0.10 罕见/查泄漏；**0.05–0.10 很好**；0.02–0.05 good；<0.01 噪声；并称"**IC 0.03 配 ICIR>0.5 比 IC 0.05 配 ICIR<0.3 更值钱**" |
| riskhub（月度） | **月度 mean IC 0.02–0.06 属正常且有价值**；示例：Value 0.032/ICIR 0.34、Momentum 0.048/0.29、Quality 0.027/0.39 |
| QuantJourney（月度、8 ETF） | 月度 IC 0.05–0.10 有经济意义；表格：SMA 趋势 0.084/ICIR 0.58、12-1 动量 0.063/0.41 |
| QuantFull（中文，日频） | **Rank-IC 均值稳定 ≥0.02 即有筛选价值，0.03–0.05 已属可用** |
| dao-quant-research（中文） | IR>0.5 通常认为有效；示例平均 IC 0.028–0.042 / IR 0.19–0.23 |
| BigQuant（月度基本面实测） | \|IC\|>0.03 有意义、\|ICIR\|>0.3 较好；PE TTM **0.0529/0.6995**、PB 0.0557/0.4888、PS 0.0394/0.4275、ROE 0.0172/0.2277 |

⇒ 结论：**日常可用线 ≈ IC 0.03**；**月度有效区间 ≈ 0.02–0.06**；**ICIR 有效线 0.3，0.5–0.7 已属强**。

## 3. 内部实测（配对设计）

- 脚本：`scripts/calibrate_slow_tier_thresholds.py`（只读：不落库、不改 registry）
- 设计：候选池 42 个表达式 × label_1d / 5d / 20d 三个会话各跑一遍 train 评估
  （2020-01-01~2022-12-31，`include_fundamentals=True`），**n=41 有效配对**，约 35 分钟
- 分布与配对中位比值：

| label | \|IC\| p25 / p50 / p75 | \|ICIR\| p25 / p50 / p75 |
|---|---|---|
| 1d | 0.0163 / 0.0205 / 0.0239 | 0.2115 / 0.2698 / 0.2918 |
| 5d | 0.0255 / 0.0312 / 0.0382 | 0.3662 / 0.4351 / 0.5164 |
| 20d | 0.0388 / 0.0496 / 0.0625 | 0.6103 / 0.7432 / 0.8714 |

配对中位放大系数：5d/1d = 1.776 / 1.800；20d/1d = 2.865 / 3.032（**只用于说明尺度效应，不再是标定规则**）

## 4. 双锚选值与池内过线率

同一实测池上的过线率（括号内为文献可解释性）：

| 方案 | 过线率 |
|---|---|
| 主档 1d 0.02 / 0.28（现状） | 15/41 = **36.6%** |
| weekly v1 0.0355 / 0.504 | 11/41 = 26.8%（过严） |
| **weekly v2 0.030 / 0.450** | 15/41 = **36.6%**（= 主档；IC 0.03 正落"可用线"） |
| monthly v1 0.0573 / 0.849 | 9/41 = 22.0%（过严，绝对值超文献最强档） |
| monthly 文献松锚 0.040 / 0.450 | 26/41 = 63.4%（偏松，会灌水） |
| monthly 文献中 0.040 / 0.600 | 24/41 = 58.5% |
| **monthly v2 0.053 / 0.650** | 15/41 = **36.6%**（= 主档；与经典月度因子 PE TTM 0.0529/0.6995 同档） |
| weekly / monthly 沿用 1d 线 | 85.4% / 92.7%（门形同虚设） |

## 5. 落地的门槛（`core/research_modes.py` v2）

| 项 | technical(1d) | technical_weekly(5d) | technical_monthly(20d) |
|---|---|---|---|
| evaluation `min_train_abs_ic` | 0.02 | **0.030** | **0.053** |
| evaluation `min_train_icir` | 0.28 | **0.450** | **0.650** |
| evaluation `min_val_abs_ic` | 0.015 | **0.0225** | **0.0398** |
| candidate `min_abs_ic` | 0.02 | **0.030** | **0.053** |
| candidate `min_icir` | 0.28 | **0.450** | **0.650** |
| candidate `min_val_abs_ic` | 0.015 | **0.0225** | **0.0398** |
| production `min_train_abs_ic` | 0.025 | **0.0375** | **0.0663** |
| production `min_train_icir` | 0.30 | **0.4821** | **0.6964** |
| production `min_val_abs_ic` | 0.015 | **0.0225** | **0.0398** |

派生规则：`production = 本档候选线 × (0.025/0.02 或 0.30/0.28)`；`min_val_abs_ic = 0.015 × (本档 IC 线 / 0.02)`。

**刻意不动**：`min_val_ic_retention`（比值口径）、`min_coverage` / `min_cs_autocorr` / `max_abs_corr`
（因子侧属性）、`engine_gate.min_excess_annual` / `min_excess_sharpe`（年化口径，**缺该档实测数据**，
暂继承主档）；换手门槛本就按频率分档（0.50/0.65/0.80），未改动。

## 6. 不变量与回归

`tests/test_slow_tier_thresholds.py` 锁定：候选/评估线 == 文献锚定值（可复算）→ production 派生关系 →
三档严格递增 → 线值落在文献可解释区间（IC ≤0.10、ICIR ≤1.0）→ 尺度无关项不动 →
engine_gate 继承 → label↔freq 一致性不破。

受影响并已通过的既有测试：`test_label_freq_consistency.py`、`test_prompt_consistency_audit.py`、
`test_engine_gate_buffer_band.py`、`test_auto_mode_and_freq.py`、`test_unified_library.py`。

## 7. 已知局限 / 后续

- 池为**候选池 42 个表达式**（偏技术族、"未过线候选"），非全体生成分布；单一 train 窗口；
  慢档真实 run 数据积累后应复核；
- 未测量 val 段尺度比（用 train 近似）；val 保留比是比值口径，与尺度无关；
- `engine_gate` 年化门槛（0.03/0.5）未按档调整——需要慢档真实回测（年化超额/夏普分布），属独立工作；
- **最终裁决仍是 engine_gate**（真实行情、含成本、按 freq 调仓）：IC 线只做研究侧准入，
  过严只会饿死 pipeline（v1 的教训），过松会让 stage_two/engine_gate 白烧算力；
- 复核方法：重跑 `scripts/calibrate_slow_tier_thresholds.py`，若分布或池内过线率显著变动则重议本表。

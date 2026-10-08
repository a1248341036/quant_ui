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

## 5.1 fundamental 档重锚（2026-10-07，总体一致化）

`fundamental` 同为 **label_20d + monthly** 档，但历史门槛沿用 1d 尺度的 `0.020/0.28/0.012`
（2026-09-11 的"与 technical 同值量纲锚"）→ 在 20d label 下同样形同虚设：

- 实测（基本面族近似子集 n=9，同 panel/label 配对）：20d 分布 p50 `|IC| 0.0371 / |ICIR| 0.4174`；
  旧线过线率 **77.8%**（技术慢档同口径 36.6%）。
- 双锚重锚到 **0.035 / 0.45 / 0.021**：① 文献月度基本面实测中位量级（PE TTM 0.0529/0.6995、
  PB 0.0557/0.4888、ROE 0.0172/0.2277）；② 该族过线率 **44.4%**（与技术慢档同量级）。
- 三个层次按同一比例（IC ×1.75、ICIR ×1.607）重锚，保持本档"production == candidate"
  与 val 保留比（0.65/0.70）、engine_gate（0.02/0.4）不变。

| 项 | 旧值(1d 尺度) | 新值(20d 尺度) |
|---|---|---|
| evaluation `min_train_abs_ic` / `min_train_icir` / `min_val_abs_ic` | 0.020 / 0.28 / 0.012 | **0.035 / 0.45 / 0.021** |
| candidate `min_abs_ic` / `min_icir` / `min_val_abs_ic` | 0.020 / 0.28 / 0.012 | **0.035 / 0.45 / 0.021** |
| production `min_train_abs_ic` / `min_train_icir` / `min_val_abs_ic` | 0.020 / 0.28 / 0.012 | **0.035 / 0.45 / 0.021** |

## 5.2 判定侧口径修复（同批，2026-10-07）

重锚后暴露两条判定路径的**阈值来源漂移**：

- `tools/_dispatch._near_miss_verdict` → `_candidate_ic_bar(mode)`：**档位感知**（读
  `RESEARCH_MODES[mode].candidate_overrides["min_abs_ic"]`，回落全局默认）；
- `memory/schema.py:_classify` → 只读 `DEFAULT_RESEARCH_SPEC["evaluation_policy"]` 的**全局**
  `min_train_abs_ic`（0.02），**不感知档位**；且 `memory/ingestion.py` 直到落库阶段才把
  `research_mode` 合进 `metrics_compact`，判定阶段拿不到档位。

⇒ 若不修，慢档/基本面档会把**低于本档海选线**的 IC 标成 `train_passed`、并把 near_miss 判错。
修复（两处小改）：`schema._classify` 按 `metrics["research_mode"]` 取该档
`evaluation_overrides["min_train_abs_ic"]`（未知档位/异常回落全局默认）；`ingestion.record`
在调用 `_classify` **之前**把 `run_freq_context` 的档位元数据 `setdefault` 进 `metrics`。

## 6. 不变量与回归

`tests/test_slow_tier_thresholds.py` 锁定：候选/评估线 == 文献锚定值（可复算）→ production 派生关系 →
三档严格递增 → 线值落在文献可解释区间（IC ≤0.10、ICIR ≤1.0）→ 尺度无关项不动 →
engine_gate 继承 → label↔freq 一致性不破；并新增 **fundamental 重锚**两测（三层 0.035/0.45/0.021、
落在文献月度基本面区间 0.017~0.056 / 0.23~0.70）。
判定侧口径由 `tests/test_yield_improvements.py::test_memory_classify_near_miss` 锁定
（同一 IC=0.030：fundamental→`near_miss`、technical→`train_passed`，证明档位已生效）。

受影响并已同步更新的既有测试：`test_delivery_checker.py`（fundamental 门槛与 prompt 渲染）、
`test_alphaagent_smoke.py`（"fundamental 比 technical 宽松"改为"按 20d 尺度锚定"）、
`test_yield_improvements.py`（near_miss 带 [0.028, 0.035)）；另经
`test_label_freq_consistency.py`、`test_prompt_consistency_audit.py`、
`test_engine_gate_buffer_band.py`、`test_auto_mode_and_freq.py`、`test_unified_library.py`、
`test_mining_memory_footprint.py` 全量回归。

**已知既有失败（与本次改动无关，HEAD 亦红）**：`test_yield_improvements.py::TestNearMissHint`
的 3 例（hint 文案/PIT 警戒键与实现漂移）——已用 `git stash` 在 HEAD 复现同样失败，属独立遗留问题。

## 7. 已知局限 / 后续

- 技术族池为**候选池 42 个表达式**（偏技术族、"未过线候选"），非全体生成分布；单一 train 窗口；
  慢档真实 run 数据积累后应复核；
- **fundamental 的重锚证据较弱**：其"基本面族近似子集"仅 **n=9**（且按表达式内容启发式归类，
  含 holder/事件类），非严格的基本面档因子分布 ⇒ 建议后续用 `--research-mode fundamental`
  的真实 run 数据或专门的配对测量复核（方法同本脚本，把因子池换成基本面族）；
- 判定侧 `_classify` 现在按档位取线，但**用户自定义 override（research_spec overrides）未纳入**
  （用 registry 的 mode override；`_dispatch` 路径同样如此 ⇒ 两路径保持一致，未扩大差异）；
- 未测量 val 段尺度比（用 train 近似）；val 保留比是比值口径，与尺度无关；
- `engine_gate` 年化门槛（0.03/0.5）未按档调整——需要慢档真实回测（年化超额/夏普分布），属独立工作；
- **最终裁决仍是 engine_gate**（真实行情、含成本、按 freq 调仓）：IC 线只做研究侧准入，
  过严只会饿死 pipeline（v1 的教训），过松会让 stage_two/engine_gate 白烧算力；
- 复核方法：重跑 `scripts/calibrate_slow_tier_thresholds.py`，若分布或池内过线率显著变动则重议本表。

---

## 8. 更新（2026-10-08）：门槛只由 label 决定、与数据面无关

本评审里 fundamental 独立标定的 **0.035 / 0.45 / 0.021** 已被**废弃撤回**，不再生效。

- **理由**：fundamental 与 technical_monthly 同为 label_20d，却门槛不同（技术月频 0.053/0.65
  vs 基本面松 0.035/0.45），违反"门槛只由预测窗口 label 唯一决定、数据面无关"的原则。
- **落地**：`core/research_modes.py` 抽出共享常量 `_MONTHLY_20D_OVERRIDES`，fundamental 与
  technical_monthly 引用同一来源（0.053 / 0.65 / 0.0398，production 0.0663 / 0.6964 / 0.0398，
  val 保留比 0.50，engine_gate 年化 0.03 / 夏普 0.5，decay 0.10）；fundamental 相对唯一区别 =
  `needs_fundamentals=True`。
- 相关测试/文档已同步：`test_alphaagent_smoke.py`、`test_delivery_checker.py`、
  `test_slow_tier_thresholds.py`、`test_research_spec_overrides.py`、`test_yield_improvements.py`、
  `docs/alphaagent_architecture.md`。本条仅作历史存档，勿据此断言当前门槛。

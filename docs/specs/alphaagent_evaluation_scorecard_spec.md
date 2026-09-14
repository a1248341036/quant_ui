---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '38fe5232-3fae-4aae-9197-5892f7c2fcfb'
  PropagateID: '38fe5232-3fae-4aae-9197-5892f7c2fcfb'
  ReservedCode1: '96464153-b191-45df-8f32-db68aa4895c9'
  ReservedCode2: '96464153-b191-45df-8f32-db68aa4895c9'
---

# SPEC: AlphaAgent 产出效率与质量量化评估体系 (Evaluation Scorecard & Benchmark Harness)

> 状态：设计中 (RFC)  
> 版本：v1.1（对齐代码实测口径）  
> 日期：2026-09-14  

---

## 一、 背景与核心痛点

当前系统在工程吞吐与架构解耦上已完成多轮优化，但投研人员在日常使用中面临最大的困扰是：**"无法量化 AlphaAgent 改动前后的产出效率与因子质量，主观感知改动前后区别不大。"**

### 核心根因分析
1. **结果极端两极化（"漏斗黑盒"）**：
   - 候选池（`candidate_main`）积累了 26 个统计 IC 高达 0.04~0.05 的因子，但正式交付库（`production_main`）中只有 1 个；
   - 绝大多数因子死在两阶段精筛（`stage_two`）与实盘净值回测（`engine_gate`），但 Web 界面缺乏中间环节的损耗归因；
2. **纸面统计与实盘可交易性的断崖（"纸面富贵"）**：
   - 统计 IC/ICIR 是静态截面排名的前瞻收益，而实盘 `engine_gate` 必须承受 T+1、涨跌停、100 股整手、千分之一手续费与滑点；高换手因子扣费后净值直接崩溃；
3. **缺乏标准对照基准（Golden Benchmark）**：
   - 每次提示词（System Prompt）、算子内核、变异策略的改动，没有可重复运行的标尺，只能凭运气等待单次挖掘结果。

---

## 二、 体系设计：三大度量维度与核心指标

本规范构建 **"效率 (Efficiency) - 认知 (Cognition) - 质量 (Quality)"** 三位一体的量化打分卡。

```
                           AlphaAgent 量化打分体系
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   1. 产出效率 (Efficiency)     2. 认知与对账 (Cognition)   3. 实盘质量 (Quality)
   • 评估吞吐 (eval/min)        • 工具调用自愈率 (%)         • 海选过线率 (Yield %)
   • 单因子平均耗时             • 预测对账证实率 (Confirmed)  • OOS 保留率 (Val/Train)
   • Token 转化率               • 结构消融增益率 (Ablation)  • 实盘可交易存活率 (Gate %)
   • 有效尝试占比               • 死路规避率 (APV Veto)      • L1 深度损耗 (Depth Loss)
   • LLM 缓存命中率 ★新增        • Reviewer 校准提升度 ★新增   • 决策断层因子数 ★新增
   • 算力利用率 ★新增                                          • 过拟合嫌疑标志 ★新增
                                                              • 近海选线因子数 ★新增
```

> ★新增 = v1.1 基于代码实测补入（代码已有数据但 v1.0 未纳入）。

### 1. 产出效率度量 (Efficiency Metrics)

| 指标项 | 英文标识 | 计算公式 | 数据来源（代码文件:行号） | 业务含义 | 优秀基准 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **评估吞吐** | `eval_throughput` | `n_eval / wall_minutes` | `run_metrics.py:111` (n_eval) + `:78` (wall_minutes) | 衡量算力工作池与 DSL 算子的计算速度 | ≥ 2.0 个/分 |
| **单因子平均耗时** | `avg_eval_latency` | `Σ(eval_elapsed_seconds) / n_eval` | `run_metrics.py:98` (per-tool `elapsed_seconds`) 按 `evaluate_factor`/`eval_on_train_set` 筛选 | 单次调用海选引擎的物理平均响应延迟 | ≤ 800 ms |
| **Token 转化率** | `token_factor_yield` | `stored_candidate / (output_k_tokens / 100)` | `run_metrics.py:115` (stored_candidate) + `:83` (output_k_tokens) | 每 10 万 Token 产生的可用候选因子数量 | ≥ 1.0 个/100k tok |
| **有效尝试占比** | `valid_attempt_ratio` | `(n_tool_results - n_tool_errors) / n_tool_results` | `run_metrics.py:137-138` (n_tool_results, tool_error_rate) | 衡量 Prompt 语法约束的清晰度 | ≥ 90% |
| **LLM 缓存命中率** ★ | `cache_hit_rate` | `cache_input_tokens / input_tokens` | `run_metrics.py:84` | LLM Prompt 缓存命中比例，衡量 system prompt 稳定度与 token 成本效率 | ≥ 30% |
| **算力利用率** ★ | `compute_utilization` | `tool_minutes / wall_minutes` | `run_metrics.py:110` (tool_minutes) + `:78` (wall_minutes) | 工具执行占墙钟比例，1-利用率≈LLM 思考/等待开销 | ≥ 60% |

---

### 2. 认知与探索对账度量 (Cognition Metrics)

| 指标项 | 英文标识 | 计算公式 / 口径 | 数据来源（代码文件:行号） | 业务含义 |
| :--- | :--- | :--- | :--- | :--- |
| **预测对账证实率** | `prediction_confirmed_ratio` | `N(confirmed) / (N(confirmed) + N(contradicted))` | `prediction.py:501-508` 生成 verdict（confirmed/contradicted/partial）；`run_metrics.py` observe_event 需新增计数 | 预期截面形态与实际计算相吻合的比例，衡量 LLM 经济逻辑的真实有效性 |
| **消融增益率** | `ablation_added_ratio` | `N(verdict=="conditioning_added_value") / N(门控类结构表达式)` | `prediction.py:587` 生成 verdict（conditioning_added_value/destroyed_value/flipped_signal/neutral）；`run_metrics.py` observe_event 需新增计数 | 检验门控/分段等复合算子是否真正优于 Base 原始信号，识别虚假拟合 |
| **记忆死路阻断率** | `memory_veto_accuracy` | `dup_dead_end / max(1, n_eval + n_submit)` | `run_metrics.py:141-142` (dup_dead_end, dup_dead_end_rate) | 命中历史禁忌模板（APV Veto）后被自动拦截或警示的频次，衡量 Research Memory 是否有效防止大模型在死路上反复浪费轮次 |
| **Reviewer 校准提升度** ★ | `reviewer_calibration_lift` | `approve_alive_rate - revise_alive_rate` | `run_metrics.py:292-300` (reviewer_calibration 函数) | Reviewer 判 approve 的因子后续存活率与判 revise 的存活率之差，衡量 Reviewer 意见对晋升的预测力；差值趋近 0 或为负说明 Reviewer 无区分力 |

---

### 3. 实盘质量与存活漏斗 (Quality Funnel Metrics)

这是直接解答"为什么因子进不了正式库"的核心量化链路：

$$N_{\text{尝试总数}} \xrightarrow{\text{海选过线率}} N_{\text{候选池}} \xrightarrow{\text{OOS保留率}} N_{\text{精筛存活}} \xrightarrow{\text{实盘门禁率}} N_{\text{正式库交付}}$$

| 漏斗阶段 | 指标项 | 计算口径 | 数据来源 | 门槛标准（代码实测值） | 诊断意图 |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **盲测终审** | 盲测通过率 | `N(blind_test passed) / N(blind_test attempted)` | `delivery_checker.py:262-298` (BlindTestStage) | test/train IC 保留比 ≥ 50%、\|test IC\| ≥ 0.010、方向一致 | 在 stage_one 之前执行，不通过直接拒绝不进候选池 |
| **阶段 1：海选** | **海选过线率** (`stage_one_yield`) | `candidate_stored / unique_train_evaluated` | `agentscope_run.py:1230` (candidate_stored) + `:1228` (unique_train_evaluated) | 候选池门槛：\|IC\| ≥ 0.020、\|ICIR\| > 0.28、Coverage > 0.85、cs_autocorr ≥ 0.18、val\|IC\| ≥ 0.012、val/train 保留比 ≥ 50% 且方向不反转、最大截面相关 < 0.5、组合日单边换手 ≤ 0.5 | 表达式在训练集上是否达到海选统计门槛 |
| **阶段 2：精筛** | **OOS 衰减保留率** (`oos_retention`) | `median(\|val_ic\|/\|train_ic\|)` over matched_candidates | `agentscope_run.py:1146-1151` (matched_audits[].ic_retention) | 正式库精筛：train\|IC\| ≥ 0.025、\|ICIR\| > 0.30、val\|IC\| ≥ 0.015、val/train 保留比 ≥ 50%、val 多头端年化超额 ≥ 0%、截尾 IC 衰减 ≤ 10%、最大截面相关 < 0.4 | 检验因子是否纯属训练集过拟合 |
| **阶段 3：实盘撮合** | **可交易存活率** (`engine_gate_survival`) | `production_stored / candidate_stored` | `agentscope_run.py:1231` (production_stored) / `:1230` (candidate_stored) | 净超额年化 ≥ 3%、超额夏普 ≥ 0.5、最大回撤 ≤ 40%、持仓重叠率 ≥ 50%、仓位利用率 ≥ 80% | 在周调仓、扣费 0.1% 滑点、涨跌停、T+1 撮合后，净值是否达标 |
| **可交易性诊断** | **L1 深度损耗** (`depth_loss_pp`) | `Excess_top_group - Excess_top_5pct` | eval 结果中 `depth_curve` 与 `depth_curve_tradable` 字段 | ≤ 10 pp | 诊断超额收益是否全集中在买不起的极小盘微盘股（容量坍缩度量） |
| **决策断层** ★ | **未提交过线因子数** (`unsubmitted_promising`) | `N(promising 且未 submit)` 去重 | `agentscope_run.py:1164-1179` (unsubmitted_promising_unique) | — | 训练过线但 LLM 未提交的因子数，衡量"promising → submit"决策断层（记忆实证：201 个 promising 中 93 个无 candidate_id） |
| **过拟合嫌疑** ★ | **过拟合标志** (`overfit_suspected`) | `any(matched: train\|IC\|≥0.015 且 (val\|IC\|<0.01 或方向反转))` | `agentscope_run.py:1153,1155` (overfit_suspected) | — | train/val IC 方向反转或 val IC 衰减到 0.01 以下，标记为过拟合嫌疑 |
| **近海选线** ★ | **近海选线因子数** (`near_miss_count`) | `N(verdict=="near_miss")` | `_dispatch.py:252-253` (_near_miss_verdict)；`memory/schema.py:767` (verdict="near_miss") | IC 达门槛 80% 且 ICIR/coverage 达标但未过线 | 差临门一脚的因子，不应进死档而应给二次机会（窗口微调或推 val） |

---

## 三、 Gate 失败原因体系（对齐代码 11 种 fail_reasons）

### 3.1 engine_gate 失败原因（`engine_gate.py:107-217`）

| fail_reason | 含义 | 门槛来源 |
| :--- | :--- | :--- |
| `excess_annual` | 净超额年化 < 3% | `EngineGateCriteria.min_excess_annual = 0.03` |
| `excess_sharpe` | 超额夏普 < 0.5 | `EngineGateCriteria.min_excess_sharpe = 0.5` |
| `max_drawdown` | 最大回撤 > 40% | `EngineGateCriteria.max_drawdown = 0.40` |
| `hold_overlap` | 持仓重叠率 < 50% | `EngineGateCriteria.min_daily_overlap = 0.5` |
| `execution_infeasible` | 仓位利用率 < 80% | `EngineGateCriteria.min_invested_ratio = 0.8` |
| `annual_return` | 兼容旧键：绝对年化不达标（默认不检查） | `policy.get("min_annual_return")`，默认 None |
| `sharpe` | 兼容旧键：绝对夏普不达标（默认不检查） | `policy.get("min_sharpe")`，默认 None |
| `engine_backtest_error` | 引擎回测异常（T+1/停牌/数据缺失等） | `engine_gate.py:101` |

### 3.2 stage_one 失败原因（`delivery_checker.py:62-110`）

| fail_reason | 含义 | 门槛值 |
| :--- | :--- | :--- |
| `ic` | \|IC\| < 0.020 | `CandidateCriteria.min_abs_ic` |
| `icir` | \|ICIR\| < 0.28 | `CandidateCriteria.min_icir` |
| `coverage` | Coverage ≤ 0.85 | `CandidateCriteria.min_coverage` |
| `cs_autocorr` | 截面自相关 < 0.18 | `CandidateCriteria.min_cs_autocorr` |
| `val_ic_missing` | val IC 缺失或非有限 | `StageOneValRetention` |
| `val_sign_flip` | val/train IC 方向反转 | `StageOneValRetention` |
| `val_abs_ic` | \|val IC\| < 0.012 | `CandidateCriteria.min_val_abs_ic` |
| `val_retention` | \|val_ic\|/\|train_ic\| < 0.50 | `CandidateCriteria.min_val_ic_retention` |
| `max_cs_corr` | 最大截面相关 ≥ 0.5 | `CandidateCriteria.max_abs_corr` |

### 3.3 stage_two 失败原因（`delivery_checker.py:190-247`）

| fail_reason | 含义 | 门槛值 |
| :--- | :--- | :--- |
| `train_ic` | train\|IC\| < 0.025 | `ProductionCriteria.min_train_abs_ic` |
| `train_icir` | train\|ICIR\| < 0.30 | `ProductionCriteria.min_train_icir` |
| `val_ic` | val\|IC\| < 0.015 | `ProductionCriteria.min_val_abs_ic` |
| `val_long_excess` | val 多头端年化超额 < 0% | `ProductionCriteria.min_val_long_excess` |
| `winsorized_abs_ic_decay` | 截尾 IC 衰减 > 10% | `ProductionCriteria.max_winsorized_abs_ic_decay` |
| `max_cs_corr` | 最大截面相关 ≥ 0.4 | `ProductionCriteria.max_abs_corr` |
| *(继承 stage_one val_retention 全部 fail_reasons)* | | |

### 3.4 blind_test 失败原因（`delivery_checker.py:262-298`）

| fail_reason | 含义 | 门槛值 |
| :--- | :--- | :--- |
| `blind_test_ic_missing` | test IC 或 train IC 缺失/非有限 | — |
| `blind_test_sign_flip` | test/train IC 方向不一致 | `require_sign_consistency = True` |
| `blind_test_abs_ic` | \|test IC\| < 0.010 | `BlindTestCriteria.min_test_abs_ic` |
| `blind_test_ic_retention` | \|test_ic\|/\|train_ic\| < 0.50 | `BlindTestCriteria.min_ic_retention` |
| `blind_test_train_ic_near_zero` | train IC 近零无法算保留比 | — |

---

## 四、 Prediction / Ablation 的 per-eval → run 级聚合方案

### 问题
`prediction_check` 和 `ablation_check` 在每次 `evaluate_factor` / `eval_on_train_set` 的结果中生成（`prediction.py:533-546`, `prediction.py:562-601`），但 `run_metrics.py` 的 `observe_event` 当前只统计 `n_eval`/`n_submit`/`stored_candidate`/`stored_production`，**未对 prediction/ablation verdict 做计数**。

### 方案
在 `run_metrics.py` 的 `observe_event` 函数中，当 `name in ("evaluate_factor", "eval_on_train_set")` 时，额外提取并累计：

```python
# observe_event 中新增（伪代码）
pred_check = res.get("prediction_check")
if isinstance(pred_check, dict):
    pv = pred_check.get("verdict")  # "confirmed" | "contradicted" | "partial"
    if pv:
        state["prediction_verdicts"][pv] = state["prediction_verdicts"].get(pv, 0) + 1

abl_check = res.get("ablation_check")
if isinstance(abl_check, dict):
    av = abl_check.get("verdict")  # "conditioning_added_value" | "conditioning_destroyed_value" | "conditioning_flipped_signal" | "neutral" | "unverifiable"
    if av:
        state["ablation_verdicts"][av] = state["ablation_verdicts"].get(av, 0) + 1
```

在 `build_metrics_snapshot` 中导出：

```python
"prediction_confirmed": state["prediction_verdicts"].get("confirmed", 0),
"prediction_contradicted": state["prediction_verdicts"].get("contradicted", 0),
"prediction_partial": state["prediction_verdicts"].get("partial", 0),
"prediction_confirmed_ratio": round(
    confirmed / (confirmed + contradicted), 3
) if (confirmed + contradicted) else None,
"ablation_added_value": state["ablation_verdicts"].get("conditioning_added_value", 0),
"ablation_destroyed_value": state["ablation_verdicts"].get("conditioning_destroyed_value", 0),
"ablation_flipped_signal": state["ablation_verdicts"].get("conditioning_flipped_signal", 0),
"ablation_neutral": state["ablation_verdicts"].get("neutral", 0),
"ablation_unverifiable": state["ablation_verdicts"].get("unverifiable", 0),
```

离线解析 `compute_run_metrics` 同理，遍历 `tool_results` 事件时提取 `prediction_check.verdict` 和 `ablation_check.verdict` 做相同累计。

`near_miss_count` 同理：在 `observe_event` 中，当 verdict 为 `"near_miss"`（`memory/schema.py:767` 生成）时计数，`build_metrics_snapshot` 导出 `near_miss_count`。

---

## 五、 落地工程方案 (Implementation Blueprint)

### 1. 后端：Run 结束时自动生成 `scorecard.json`
在 `alphaagent/factor/mining/agent/agentscope_run.py` 的 Run 结束钩子（`_emit("run_end")`，L1254）后，基于已有的 `summary` dict + `build_metrics_snapshot(live_metrics)` 聚合生成 `logs/factor_mining/ui/<run_id>/scorecard.json`：

```json
{
  "run_id": "adfbf57c45aa",
  "created_at": "2026-09-14T20:00:00Z",
  "schema_version": 3,
  "summary": {
    "total_turns": 16,
    "wall_time_minutes": 24.5,
    "eval_throughput": 2.24,
    "total_tokens": 128450,
    "token_factor_yield": 1.56,
    "cache_hit_rate": 0.32,
    "compute_utilization": 0.68,
    "valid_attempt_ratio": 0.92
  },
  "funnel": {
    "unique_train_evaluated": 48,
    "candidate_stored": 2,
    "production_stored": 1,
    "stage_one_yield_pct": 4.2,
    "gate_survival_pct": 50.0,
    "unsubmitted_promising": 3,
    "near_miss_count": 5,
    "overfit_suspected": false
  },
  "cognition": {
    "prediction_confirmed": 28,
    "prediction_contradicted": 12,
    "prediction_partial": 8,
    "confirmed_ratio_pct": 70.0,
    "ablation_added_value": 4,
    "ablation_destroyed_value": 1,
    "ablation_flipped_signal": 0,
    "ablation_neutral": 2,
    "ablation_unverifiable": 0,
    "dup_dead_end_rate": 0.04,
    "reviewer_calibration_lift": 0.15
  },
  "gate_failure_reasons": {
    "excess_annual": 1,
    "hold_overlap": 0,
    "max_drawdown": 0,
    "execution_infeasible": 0,
    "excess_sharpe": 0
  },
  "stage_one_failure_reasons": {
    "ic": 8,
    "icir": 3,
    "coverage": 1,
    "cs_autocorr": 2,
    "val_sign_flip": 1,
    "val_retention": 2,
    "max_cs_corr": 0
  },
  "oos_retention": {
    "median_ic_retention": 0.62,
    "matched_count": 12
  }
}
```

### 2. CLI：标准对照基准脚本 (`scripts/benchmark_agent_run.py`)
提供一键回放与基准打分工具，支持：
```powershell
# 对当前最新代码运行 5 轮标准基准测试（固定 Prompt、固定 5 轮预算）
python scripts/benchmark_agent_run.py --rounds 5 --preset standard_tech

# 针对两个已完成的 run_id 进行横向指标对比
python scripts/benchmark_agent_run.py --compare run_id_A run_id_B
```
输出清晰的 ASCII 对照表与差异雷达指标（吞吐提升、过线率变化、Gate 存活差异）。

### 3. 前端：Web 界面展示"产出漏斗与质量卡片"
在 `AlphaAgent.vue` 的【研究】主页面右侧或运行明细中增加【Run 质量看板】：
- **顶部 6 个核心 KPI 芯片**：`评估吞吐 (eval/min)`、`海选过线率 (%)`、`实盘存活率 (%)`、`预测证实率 (%)`、`缓存命中率 (%)`、`算力利用率 (%)`；
- **产出转化漏斗条形图 (Funnel Chart)**：直观展示从 `尝试 -> 有效 -> 候选池 -> 正式库` 的逐级损耗与拦截原因（按代码 11 种 fail_reasons 分色标注）；
- **认知面板**：prediction 三色（confirmed/contradicted/partial）+ ablation 四色（added/destroyed/flipped/neutral）+ reviewer 校准提升度。

---

## 六、 门槛值速查表（代码单一真源：`delivery/delivery_criteria.py`）

| 阶段 | 门槛项 | 代码字段 | 默认值 |
| :--- | :--- | :--- | :---: |
| **盲测终审** | IC 保留比 | `BlindTestCriteria.min_ic_retention` | 0.50 |
| | test 绝对 IC | `BlindTestCriteria.min_test_abs_ic` | 0.010 |
| | 方向一致性 | `BlindTestCriteria.require_sign_consistency` | True |
| **候选池 (stage_one)** | 绝对 IC | `CandidateCriteria.min_abs_ic` | 0.020 |
| | ICIR | `CandidateCriteria.min_icir` | 0.28 |
| | Coverage | `CandidateCriteria.min_coverage` | 0.85 |
| | 截面自相关 | `CandidateCriteria.min_cs_autocorr` | 0.18 |
| | val 绝对 IC | `CandidateCriteria.min_val_abs_ic` | 0.012 |
| | val/train 保留比 | `CandidateCriteria.min_val_ic_retention` | 0.50 |
| | 最大截面相关 | `CandidateCriteria.max_abs_corr` | 0.5 |
| | 组合日单边换手 | `CandidateCriteria.max_avg_daily_side_turnover` | 0.5 |
| **正式库精筛 (stage_two)** | train 绝对 IC | `ProductionCriteria.min_train_abs_ic` | 0.025 |
| | train ICIR | `ProductionCriteria.min_train_icir` | 0.30 |
| | val 绝对 IC | `ProductionCriteria.min_val_abs_ic` | 0.015 |
| | val/train 保留比 | `ProductionCriteria.min_val_ic_retention` | 0.50 |
| | val 多头端超额 | `ProductionCriteria.min_val_long_excess` | 0.0 |
| | 截尾 IC 衰减 | `ProductionCriteria.max_winsorized_abs_ic_decay` | 0.10 |
| | 最大截面相关 | `ProductionCriteria.max_abs_corr` | 0.4 |
| **Engine Gate** | 净超额年化 | `EngineGateCriteria.min_excess_annual` | 0.03 |
| | 超额夏普 | `EngineGateCriteria.min_excess_sharpe` | 0.5 |
| | 最大回撤 | `EngineGateCriteria.max_drawdown` | 0.40 |
| | 持仓重叠率 | `EngineGateCriteria.min_daily_overlap` | 0.5 |
| | 仓位利用率 | `EngineGateCriteria.min_invested_ratio` | 0.8 |

> 运行时以 research_spec 注入为准（`DeliveryCriteria.from_spec()`），缺失键回落上表默认值。

---

## 七、 验收与落地收益

1. **版本改动立即可感**：
   - 修改了 System Prompt（如强化低换手指导），跑一次标准基准，可以直接看到 `gate_survival_pct` 从 0% 提升到了 25%；
   - 优化了 Numba 算子或工作池，可以直接看到 `eval_throughput` 从 0.5 跃升至 2.5 eval/min；
   - 启用 LLM 缓存后，`cache_hit_rate` 从 0% 升至 30%+，`compute_utilization` 随之下降（LLM 等待减少）。
2. **彻底打破"盲目调参"循环**：
   - 系统用明确的失败原因（如 `excess_annual` 占 80%）倒逼 Prompt 和策略变异向长窗口、高自相关结构演进，从根本上终结"只看 IC 很高，实盘全军覆没"的死循环。
3. **决策断层可见化**：
   - `unsubmitted_promising` > 0 时直接暴露"训练过线但 LLM 未提交"的产出损失，倒逼 Prompt 强化"promising → submit 强制决策"规则。
4. **Reviewer 质量可追溯**：
   - `reviewer_calibration_lift` 趋近 0 或为负时，说明 Reviewer 的 approve/revise 意见对因子存活没有预测力，应调整审查标准或降权。

---

## 八、 v1.0 → v1.1 变更记录

| 变更类型 | 内容 |
| :--- | :--- |
| **新增指标** | `cache_hit_rate`、`compute_utilization`、`reviewer_calibration_lift`、`unsubmitted_promising`、`overfit_suspected`、`near_miss_count`（共 6 个，均来自代码已有数据） |
| **修正 gate 失败原因** | v1.0 写 `overlap_too_low`/`annual_return_negative_after_cost` → 对齐代码 `hold_overlap`/`excess_annual`，补全全部 8 种 engine_gate + 9 种 stage_one + 7 种 stage_two + 5 种 blind_test fail_reasons |
| **修正门槛值** | stage_one IC 从 0.02 → 0.020（同值但对齐 `CandidateCriteria`）；补全 ICIR 0.28、cs_autocorr 0.18、val_abs_ic 0.012 等缺失门槛；stage_two 门槛从文档描述对齐 `ProductionCriteria` 实际值 |
| **补数据来源映射** | 每个指标标注代码文件:行号，确保 spec 与代码可追溯 |
| **补聚合方案** | prediction/ablation/near_miss 的 per-eval → run 级聚合伪代码 |
| **scorecard.json** | schema_version 升至 3，补全新增指标字段 |
| **前端 KPI** | 从 4 个芯片扩展到 6 个（+缓存命中率、+算力利用率） |

> AI生成
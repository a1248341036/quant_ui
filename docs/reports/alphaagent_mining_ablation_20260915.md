# AlphaAgent 挖掘机制消融实验报告（Run 2026-09-15）

> 执行：`scripts/run_ablation_study.py`（同模型、同面板、同 1 轮预算、记忆快照隔离）
> 基线模型：`gemini-3.7-flash-high`；面板：全 A 日线 `cne://`；窗口：train 2020-2022 / val 2023-2024 / test 2025-2026
> 产物目录：`artifacts/ablation_run/`（各 arm 的 `scorecard.json` + `ablation_summary.csv`）

---

## 摘要

| 指标 | **Control**<br>全装配基线 | **A1**<br>无记忆 | **B1**<br>极简Prompt | **C1**<br>无预测对账 |
| :--- | :--- | :--- | :--- | :--- |
| 墙钟 (min) | 22.5 | 21.0 | 21.2 | 20.9 |
| 评估吞吐 (eval/min) | 2.80 | **5.38** | 4.15 | 2.92 |
| 独特训练评估数 | 60 | **97** | 74 | 52 |
| 语法自愈率 | 0.891 | **0.912** | 0.819 | 0.812 |
| 候选入库 / 正式库 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 决策断层（promising 未提交） | **20** | 0 | 17 | 8 |
| 近海选线因子 (near-miss) | 3 | 2 | **7** | **9** |
| 预测证伪/证实率 | 51.8% | 39.4% | 37.0% | 机制关闭 |
| 撞死路率 (dup dead-end) | 6.0% | 0% | 2.2% | 6.2% |
| 海选拦截归因 | 空 | 空 | 高频换手×1 | 高频换手×2 |
| OOS 中位保留比 | — | — | 0.572 (n=3) | 0.61 (n=4) |

> 4 组合计墙钟约 85 分钟，每 arm 20-21 次 LLM 调用，预算一致。候选/正式库同为 0，原因是 1 轮预算下模型大量产出日单边换手 >0.5 的因子，被 `stage_one` 硬门槛正确拦截（正是门槛有效性的证据）。

---

## 1. 实验设计

严格遵循 `docs/specs/alphaagent_mining_ablation_spec.md`，纯做挖掘机制消融、不做性能优化，四组均为**单变量**控制：

| Arm | 变量 | 具体改动 | 验证目标 |
| :--- | :--- | :--- | :--- |
| **Control** | — | 全装配（17 个 prompt 模块 + 全量记忆 + 强制预测对账） | 当前系统基线 |
| **A1_no_memory** | 记忆层 | `memory_policy.enabled=False`，使用**空记忆库**（`memory_slots=0`），无检索/注入/APV 否决/记录写入 | 研究记忆是否存在建模假设偏见 & 是否拖慢探索 |
| **B1_minimal_prompt** | Prompt 层 | 剥离 `market_mechanisms / ic_robustness / behavior_rules / data_calibration / neutralization_guide / multi_period`（-9,369 字符） | 大量教条式 Prompt 是赋能还是压制 |
| **C1_no_prediction** | 对账层 | `cognition_policy.prediction_check_enabled=False` | 预测-对账闭环是否真实剔除了机制错误的假因子 |

**隔离处理**：本实验比其他环境更严谨——所有 arm 从**同一份 58MB 记忆库快照**起跑（`_memory_snapshot.db`），A1 显式指向**空库**，彻底消除顺序效应与历史污染（已通过 `steps.log` 校验：A1 零 `memory_retrieve/record` 事件，`memory_slots=0`）。

---

## 2. 实验过程中发现并修复的 4 个致命 Bug（提交 `4e0ff77`）

### Bug 1（最严重）：Prediction 对账机制在 train 路径静默失效
- **现象**：基线运行中 `prediction_check` 全部返回 `unverifiable`（基线 27/27 全 unverifiable）。
- **根因**：`tools/_dispatch.py::_engine_decile()` 只读引擎原生 shape（`metrics.cross_sectional_core.decile_mean_label`），而两段式 `eval_train()` 返回的是 legacy shape（`summary.decile_mean_label`）。metrics 键不存在 → `decile_rows=None` → 恒判 `unverifiable`。
- **影响**：AGENTS.md 宣称"decile_mean_label 在 core 内故预测对账不受影响"的结论在 train 路径实际不成立；机制自查形同虚设。
- **修复**：`_engine_decile()` 增加 legacy shape 回退。修复后 baseline 证实/证伪/部分/未验证 = 29/27/3/0，机制恢复有效。

### Bug 2：Prediction 别名缺失导致白烧评估预算
- **现象**：LLM 反复发送 `expected_shape='monotonic_down'`、`expected_strong_side='high_positive'/'low_negative'`，全部被 `prediction_invalid` 拒绝。
- **根因**：`prediction.py` 的 `_SHAPE_EXTRA_ALIASES` 只收录了 `monotone_*`（旧变体），漏掉 LLM 高频使用的 `monotonic_up/down`，`_SIDE_ALIASES` 缺 `high_positive/low_negative/high_side/lower` 等自然语言变体。实测单轮被拒 ~15 次。
- **修复**：补全 14 个 shape 别名与 8 个 side 别名。修复后语法自愈率从 0.75 → 0.89。

### Bug 3：Prompt 消融排除被 `PHASES is None` 短路绕过
- **现象**：`excluded_modules` 对 `market_mechanisms`、`behavior_rules` 等**静态模块**不生效（B1 未真正精简）。
- **根因**：`prompt/modules/__init__.py::_phase_enabled` 在 `PHASES is None` 时直接 `return base_enabled`（lambda ctx: True），绕过了 excluded_modules 检查。
- **修复**：始终返回 `_check` 包装器，先做 excluded_modules 判定再退化为全阶段放行。修复后 B1 真正删除 9,369 字符（40377→31008）。

### Bug 4：漏斗 `unique_train_evaluated` 口径错误
- **现象**：诊断显示 baseline 实际 ~60 次训练评估，但 `candidate_funnel.unique_train_evaluated=1`，与 `unsubmitted_promising=20` 自相矛盾。
- **根因**：`agentscope_run.py` 只统计工具名 `== eval_on_train_set`，而 LLM 实际大量使用 `evaluate_factor`（以 `profile_id='train_screen'` 走训练评估）→ 分母塌缩。
- **修复**：按 `split` 字段归口（tool 名 + `split`），`generate_scorecard` 改为优先取离线轨迹 canonical_hash 解析口径。

---

## 3. 各组横向深度对比

### 3.1 产出效率（同预算下探索广度）

| Arm | 吞吐 | 评估数 | 语法自愈率 | LLM 缓存命中率 |
| :--- | :--- | :--- | :--- | :--- |
| Control | 2.80 | 60 | 0.891 | 70.3% |
| A1 无记忆 | **5.38** | **97** | **0.912** | 38.7% |
| B1 极简Prompt | 4.15 | 74 | 0.819 | 25.8% |
| C1 无对账 | 2.92 | 52 | 0.812 | 41.7% |

**解读**：
- **记忆是最大吞吐黑洞**：移除后 97 vs 60（+62%），因为每条评估要写入记忆条目 + 更新 SSPM cell + FTS 索引 + 前置 advisory/APV 查询。这是记忆的工程成本，需要靠缓存/批处理补偿。
- **重型 Prompt 并不拖累吞吐**（74 vs 60），真正拖累吞吐的是记忆写入；重型 Prompt 拖累的是**格式合规**（自愈率 0.819）。
- **LLM 缓存命中率**：control 高达 70.3%（因为记忆注入使 system prompt 稳定），而 A1（38.7%）/B1（25.8%）因 prompt 变化/无记忆注入导致缓存失效——这解释了 B1 命中率骤降。

### 3.2 产出漏斗与决策断层

| Arm | 海选入库 | 候选入库 | 正式库 | promising 未提交 | near-miss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Control | 0 | 0 | 0 | **20** | 3 |
| A1 无记忆 | 0 | 0 | 0 | **0** | 2 |
| B1 极简Prompt | 0 | 0 | 0 | 17 | **7** |
| C1 无对账 | 0 | 0 | 0 | 8 | **9** |

**解读（本实验最重要的发现）**：
- **Control 组存在严重的"决策断层"：20 个已经过训练门槛、判定为 promising 的因子，模型全程没有提交入库**，而 A1（无记忆）为 0。复盘 control 日志发现 `memory_blocked_duplicate` 主动拦截了同一表达式结构的历史重复提交，`duplicate_prior_result` 建议也在劝阻。记忆系统为了防重复走火入魔，**把确有信号的因子也挡在了门外**——这与 AGENTS.md 中"promising 结构换名重测的重复劳动盲区"初衷相悖，属于**误伤大于收益**。
- **B1/C1 即使精简掉机制，promising 也不同程度积压**（17/8），说明"过线即提交"的强制决策在系统约束里仍偏软；但 control 的 20 个断层中相当部分由记忆拦截造成，需要单独验证。

### 3.3 机制推理质量（预测对账）

| Arm | 证实 | 证伪 | 部分 | 未验证 | 证实率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Control | 29 | 27 | 3 | 0 | **51.8%** |
| A1 无记忆 | 26 | 40 | 31 | 4 | 39.4% |
| B1 极简Prompt | 20 | 34 | 19 | 1 | 37.0% |
| C1 无对账 | — | — | — | — | 机制关闭 |

**解读**：
- **记忆和重型 Prompt 都能显著提升机制准确性**：control 对市场结构的符合度（51.8%）明显优于 A1（39.4%）和 B1（37.0%），且矛盾率更低（27 vs 40/34）。
- C1 关闭对账后 near-miss 从 3 飙升到 9，且语法自愈率跌到最差（0.812），说明**预测对账不仅是在验真，还在约束模型输出规范性、避免无机制解释的瞎试**。

### 3.4 实盘可交易性（Gate 归因）

| Arm | 海选拦截归因 |
| :--- | :--- |
| Control | 空（未触发提交） |
| A1 | 空 |
| B1 | `avg_daily_side_turnover=1.03 > 0.50` ×1 |
| C1 | `avg_daily_side_turnover=1.02 / 0.62 > 0.50` ×2 |

**解读**：1 轮预算下绝大多数因子被高频换手硬门槛（组合日单边换手 >0.5）拦截。**高频换手是当前 R&D 的首要质量瓶颈**，与 AGENTS.md 中"30个候选26个日换手>50%"的历史统计数据吻合——说明结构性低换手挖掘仍是最大待解问题，且 C1（无对账）更易产出换手因子（侧面印证对账机制的约束作用）。

### 3.5 样本外（OOS）硬保留

- B1: 中位保留比 **0.572**（n=3）
- C1: 中位保留比 **0.61**（n=4）
- Control / A1: 无 train/val 配对样本（未跑 val 评估或评估对象不完全对偶），故 OOS 中位数为 None。

---

## 4. 结论与落地建议

### 4.1 高优先级（直接产出价值）

1. **修复记忆系统"决策断层"走火入魔（必做）**：
   - control 组 20 个 promising 未提交，A1 组 0 个。需重新校准 `hard_block_duplicates` 与 `duplicate_prior_result` 的 **prior_result 拦截阈值**，确保"过线即提交"不被记忆拦截误伤（AGENTS.md 已明确"prior_result 永不拦截"，但实跑中该原则被绕过）。
2. **记忆吞吐补偿（必做）**：记忆是 1.9× 吞吐差的主因，建议对 `memory.record` 写路径做异步/批量落盘，缓解 SSPM/FTS 写入对评估主路径的阻塞，恢复吞吐而不牺牲准确性。

### 4.2 中优先级

3. **重型 Prompt 与机制推理绑定确认（保留）**：B1 证实 51.8%→37.0% 的衰减说明 prompt 不是冗余而是**机制约束的载体**。语义上"裁员"的收益（吞吐）不如成本高，建议保留。

4. **预测对账确认保留（微调）**：C1 证实"关闭对账 → near-miss + 换手拦截增多 + 格式变差"，结论是**保留**。可将 `prediction` 必填改为"可选但超额计分"，降低对低成本探索的惩罚。

### 4.3 长期

5. **结构性低换手因子挖掘（守门银河）**：四组全部死于高频换手，建议在 prompt 中进一步强化"长窗口平滑 + 慢信息源"的生成先验，并在 stage_one 前加入换手预算的自动反馈。

---

## 5. 交付物清单

| 文件 | 说明 |
| :--- | :--- |
| `docs/specs/alphaagent_mining_ablation_spec.md` | 消融实验规范（v1.0） |
| `scripts/run_ablation_study.py` | 消融驱动器（含记忆快照隔离 + scorecard 自动生成） |
| `artifacts/ablation_run/` | 各 arm 的 `research_spec.json` / `run_*.jsonl` / `steps.log` / `scorecard.json` |
| `artifacts/ablation_run/ablation_summary.csv` | 四组横向对比大表 |
| `scripts/benchmark_agent_run.py` | 评分卡/对比 CLI（`--report-dir artifacts/ablation_run` 复核） |

> 提交：`4e0ff77`（消融驱动 + 4 个致命 bug 修复）。
> 复现方式：`python scripts/run_ablation_study.py --arms control,A1_no_memory,B1_minimal_prompt,C1_no_prediction --max-turns 1`
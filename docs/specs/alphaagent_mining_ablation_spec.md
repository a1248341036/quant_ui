---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '545455fa-3d5a-42d9-95c8-0a864e89ab9d'
  PropagateID: '545455fa-3d5a-42d9-95c8-0a864e89ab9d'
  ReservedCode1: '27a96639-7fc7-4d30-9706-8195f66f62da'
  ReservedCode2: '27a96639-7fc7-4d30-9706-8195f66f62da'
---

# SPEC: AlphaAgent 挖掘机制消融实验规范 (Mining Mechanism Ablation Spec)

> 状态：设计中 (RFC)  
> 版本：v1.1（评审修订：统计纪律、实现清单、口径对齐）  
> 日期：2026-09-14  
> 目标：**聚焦挖掘探索能力与因子产出质量，不做系统工程性能优化，剥离各组件量化其真实贡献。**

---

## 一、 实验背景与设计原则

### 1. 为什么必须做挖掘机制消融？
AlphaAgent 在持续迭代中堆叠了大量复杂的认知与探索机制：
- **Prompt 层**：17 个模块（核心身份、策略轨道、市场机制、行为红线、算子目录、交付门槛等，见 `prompt/modules/__init__.py::DEFAULT_MODULES`）；
- **Memory 层**：BM25 检索、FTS5 正向保底、SSPM 状态编辑先验、APV 双门死路否决、结构命中率注入；
- **Cognition 对账层**：强制预测对账（`prediction_check`）、门控算子强制消融（`ablation_check`）。

这些机制**哪些是真正在起正向作用，哪些是过度约束限制了 LLM 创造力，甚至哪些在负向自毁信号？**
必须通过严谨的单变量消融实验（Ablation Study）得出客观结论。

### 2. 实验核心原则（纯挖掘逻辑，排除性能干扰）
- **排除硬件与算力干扰**：固定相同的底层数据面板（全 A 日线，train 2020-01-01 ~ 2022-12-31 / val 2023-01-01 ~ 2024-12-31 / test 2025-01-01 ~ 数据最新交易日，单一真源 `alphaagent/factor/window_config.py`）；
- **排除模型差异**：固定相同的 LLM 基座（同一模型版本、同一 temperature、同一上下文截断策略、同一中转渠道——渠道必须预充值充足余额，402 秒级失败会废掉整个 arm）；
- **代码基线冻结**：全部 arm 跑在同一 git commit 上，`run_meta.json` 必须记录 commit hash；实验期间禁止合并任何新代码；
- **固定预算与轮次**：每个对照组固定 **30 轮对话（Max Turns = 30）**，每组重复 **3 次**（统计纪律，见 2.1）；
- **盲测段客观裁判**：最终因子质量统一由 2025-01 起从未参与挖掘的独立盲测段进行净值与 IC 真实裁决。

### 2.1 统计纪律（v1.1 新增——单样本无证据力）

LLM 挖掘产出方差极大（历史 run 同配置下候选数 0~3 浮动），**每组 1 次运行的差异完全可能是随机波动**：

- **每组 ≥3 次重复**，报告均值 ± 波动范围；
- 组间比较必须给出重复间的方差：`Arm-A 均值 1.0（0/1/2）vs Control 均值 2.0（1/2/3）` 这类结论必须标注"单次差异在重复方差内，不显著"；
- 3 次重复仍是最小可行设计——若两 arm 均值差异小于重复间标准差，结论只能是"未观察到显著差异"，不得据此做工程决策；
- 汇总报告对每个维度标注置信级别：**高**（3 次重复方向一致）、**低**（方向不一致）、**不可比**（样本量不足）。

### 2.2 盲测段读取纪律（v1.1 新增——防渐进污染）

9 组共用同一段盲测数据，必须遵守：

- 盲测段评估**只在全部 arm 跑完后统一执行**，任何 arm 的中间结果不得提前触发盲测读取；
- 每个 arm 只允许**一轮**盲测评估（Top-3 候选各一次），禁止"看了盲测结果→回去调参数→再跑一轮"的循环——这会让盲测段退化成第二个 val 段；
- 盲测评估脚本与参数（`BlindTestCriteria` + test 窗口 `engine_gate`）在实验开始前写入 `run_meta.json` 锁定，实验期间不变更。

### 2.3 维度可行性分级（v1.1 新增——不并列糊弄）

四个评价维度在 30 轮预算下的证据力并不对等，事先声明：

| 维度 | 30 轮 × 3 次的期望产出 | 裁决地位 |
| :--- | :--- | :--- |
| 探索广度与效率 | 每轮 2~4 次评估，单 arm 样本 ~200 | **主裁决**：组间差异可统计 |
| 海选入池数 | 0~6 个/arm，重复间求和后勉强可比 | **主裁决**（辅助）：方向参考 |
| 实盘晋升/换手/重叠率 | 入池 0~6 → 过精筛+engine_gate 期望 <1 | **定性参考**：出现 0/0 平局属正常，不得下"机制无效"结论 |
| 盲测段 Top-3 | 依赖入池 ≥3，预计多数 arm 凑不齐 | **定性参考**：凑齐的 arm 之间比较；凑不齐本身就是信号 |

> 结论：**本实验的主裁判是"探索行为指标 + 海选过线率"**；实盘存活与盲测质量在 30 轮预算内只能做定性佐证。若未来需要以实盘存活为主裁决，预算须提到百轮级或延长单轮评估量，另立 spec。

---

## 二、 消融对照组设计（Ablation Matrix）

设计 **1 个基准组（Control Group） + 4 大类 8 个消融实验分支（Experimental Arms）**：

```
                                消融实验总拓扑
                                      │
         ┌──────────────────┬─────────┴────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
   [A. 记忆层消融]    [B. 提示词知识消融]  [C. 认知对账消融]   [D. 探索策略轨消融]
   • A1: 无记忆探索    • B1: 裸基座 Prompt  • C1: 无预测对账     • D1: 自由探索(无A/B/C/D轨)
   • A2: 仅正证据检索  • B2: 剥离 A 股机制   • C2: 无强制门控消融  • D2: 禁复合门控算子
```

### 1. 详细消融对照表（v1.1 补"实现方式"列）

> 实现方式分三档：**[现成]** 代码已有开关直接配置；**[半开发]** 需小量胶水代码；**[需开发]** 需新增机制。落地前须先完成全部"需开发"项，否则该 arm 结果无效。

| 分支 | 实验名称 | 具体改动 / 控制变量 | 实现方式 | 验证假设 |
| :---: | :--- | :--- | :--- | :--- |
| **Control** | **全装配基线 (Full)** | 启用当前所有 Prompt 模块、全量记忆（检索+APV否决+编辑先验）、强制预测对账、门控消融 | [现成] 默认配置 | 现状基准水准 |
| **A1** | **零记忆消融 (No-Memory)** | LLM 每轮看不到任何历史正负证据与死路警告 | [半开发] `research_memory_path=None` 可直接关闭整个记忆系统（agentscope_run.py:316），但当前无 research_spec 配置入口注入 None——需在 config 加 `memory_policy.enabled=False` 分支 | 检验记忆系统是否真正提升了探索效率，还是把 LLM 限制在局部最优陷阱中 |
| **A2** | **仅正向记忆 (Pos-Only Memory)** | 屏蔽所有负面证据与 APV 死路否决，只提供全库最优正向因子模板供变异 | [需开发] APV 无独立开关（只有 `apv_tau_c/apv_tau_v` 阈值）；tau 调 ∞ 只能禁否决、屏蔽不了 advisory 负面文本注入。需在 retrieval/advisory 层加 `negative_evidence_enabled` 过滤参数 | 检验"死路否决"是否杀死了潜在的有用变异，还是确实阻断了无效折腾 |
| **B1** | **极简提示词 (Minimal Prompt)** | 仅保留身份定义、工具调用契约、数据字段文档；剥离市场机制、策略轨道、行为红线、十分位形态教学、IC 稳健性等知识模块 | [半开发] prompt 已模块化（`DEFAULT_MODULES` + `unregister_prompt_module`），需在 research_spec 加 `prompt_policy.disabled_modules` 列表，由 build_system_prompt 消费 | 检验重度教条式 System Prompt 到底是在"赋能"还是在"按头产生幻觉" |
| **B2** | **剥离 A 股微观机制 (No-Mechanisms)** | 只关闭 `market_mechanisms` 模块（错误定价×套利受限、反转集中弱势股、T+1/涨跌停制度摩擦等），其余保留 | [现成] `unregister_prompt_module("market_mechanisms")`（B1 的模块级子集） | 检验领域知识注入是否显著降低了"实盘亏钱"因子的产生 |
| **C1** | **解除预测对账 (No-Prediction)** | `evaluate_factor` 不强制传 `prediction`，关闭 `prediction_check` 对账与 `contradicted` 拦截 | [需开发] 当前是软门（缺失 ≥3 次才拦，`_PREDICTION_SOFT_LIMIT`），无关闭开关。需在 dispatch 层加 `cognition_policy.prediction_check_enabled` | 检验"预测-对账"闭环是真正在纠偏机制错误，还是沦为大模型迎合答案的八股文负担 |
| **C2** | **解除门控消融 (No-Ablation)** | 关闭 `ablation_check`，允许自由提交带 `GATED_SIGNAL`/`IF_THEN_ELSE` 等门控算子的复合因子 | [需开发] 同上，需 `cognition_policy.ablation_check_enabled` | 检验强制消融是否有效拦截了"靠门控拼接破坏原信号"的过拟合因子 |
| **D1** | **自由探索模式 (Free-Search)** | 关闭策略轨道约束，取消显式父本变异要求，允许任意自由生成公式 | [半开发] prompt 侧 `unregister_prompt_module("strategy_tracks")` 现成；但 dispatch 层对 `parent_factor/edit_note` 的提示逻辑（`_attach_yield_hints`）仍会推父本变异，需一并关闭 | 检验"父本变异限制"是否压制了跨领域发现黑马因子的潜力 |
| **D2** | **纯线性/非复合算子 (Simple-Ops Only)** | 禁用分组条件、条件门控、分段状态算子，仅允许基础 TS/CS 滚动算子 | [需开发] toolkit 无算子白名单机制。需在表达式解析层加算子黑名单拦截（返回明确错误提示 LLM 换用基础算子） | 检验高级复合算子到底是制造过拟合，还是创造了真实非线性 Alpha |

### 2. 开发工作量预估（v1.1 新增）

| 优先级 | 开发项 | 预估 | 阻塞的 arm |
| :--- | :--- | :--- | :--- |
| P0 | `cognition_policy`（prediction/ablation 双开关） | 半天 | C1、C2 |
| P0 | `memory_policy.enabled` 配置分支 | 0.5 天 | A1 |
| P0 | 算子黑名单拦截 | 0.5 天 | D2 |
| P1 | 负面证据过滤（retrieval/advisory） | 1 天 | A2 |
| P1 | `prompt_policy.disabled_modules` | 0.5 天 | B1、D1 |
| P1 | 调度脚本 `scripts/run_ablation_study.py`（含 402 重试/断点续跑） | 1 天 | 全部 |

全部 arm 均须通过"开关确实生效"冒烟测试（如 C1 跑 5 轮确认结果里无 `prediction_check` 字段）才可进主实验。

---

## 三、 评价打分卡：消融度量衡 (Ablation Scorecard)

每个消融组在每组重复结束后，自动提取 **四大维度量化数据**。字段口径与 `alphaagent_evaluation_scorecard_spec.md` v1.1 的 `generate_scorecard`（schema_version 3）**同源**——消融实验直接消费 scorecard.json，不自创第二套指标体系。

### 1. 探索广度与效率（主裁决）
| 指标 | scorecard 字段 | 说明 |
| :--- | :--- | :--- |
| 独特结构表达式数 | `funnel.unique_train_evaluated` | 语义互不相同的因子骨架数（canonical hash 去重口径） |
| 有效尝试占比 | `summary.valid_attempt_ratio` | 语法合规且能正确求值的比例 |
| 撞死路重复率 | `cognition.dup_dead_end_rate` | 是否在同一类无效模板上打转 |
| Token 消耗效率 | `summary.token_factor_yield` + `summary.total_tokens` | **v1.1 恢复**：同轮次下谁花 token 少、单位 token 产出高，本身就是探索有效性的一部分——B1 极简 prompt 的核心预期收益（上下文膨胀缓解）主要体现于此，不可删 |
| LLM 缓存命中率 | `summary.cache_hit_rate` | prompt 瘦身的直接收益观测口 |

### 2. 海选与 OOS 存活力（主裁决辅助）
| 指标 | scorecard 字段 | 说明 |
| :--- | :--- | :--- |
| 海选入池数 | `funnel.candidate_stored` | 通过 stage_one 全部 8 道门的候选数 |
| 海选过线率 | `funnel.stage_one_yield_pct` | 入池 / 独特尝试 |
| OOS 保留中位数 | `oos_retention.median_ic_retention` | \|val_ic\|/\|train_ic\| 中位数，抗衰减硬度 |
| 近海选线因子数 | `funnel.near_miss_count` | 差临门一脚的因子量（探索颗粒度的补充观测） |
| 决策断层数 | `funnel.unsubmitted_promising` | 训练过线但未提交的因子（A/B 类 arm 若断层激增，说明机制在替 LLM 做决策） |

> 海选门槛（`CandidateCriteria` 全 8 项，缺一不可）：\|IC\| ≥ 0.020、\|ICIR\| > 0.28、Coverage > 0.85、cs_autocorr ≥ 0.18、val\|IC\| ≥ 0.012、val/train 保留比 ≥ 0.50 且方向不反转、最大截面相关 < 0.5、组合日单边换手 ≤ 0.5。另有盲测终审前置门（stage_one 之前）：test/train IC 保留比 ≥ 0.50、\|test IC\| ≥ 0.010、方向一致。

### 3. 实盘可交易性（定性参考，见 2.3 分级声明）
| 指标 | scorecard 字段 | 说明 |
| :--- | :--- | :--- |
| 正式库晋升数 | `funnel.production_stored` | 通过 stage_two + engine_gate 的因子数 |
| Gate 存活率 | `funnel.gate_survival_pct` | 晋升 / 入池 |
| 拦截归因 | `gate_failure_reasons` + `stage_one_failure_reasons` | 按 11 种 fail_reasons 分布看机制删改后死因是否迁移 |

> **实盘撮合口径**（单一真源 `EngineGateCriteria` + `core/trading_config.py`）：调仓频率 weekly；动态百分比选股 `selection_pct=0.001`（**top 0.1% ≈ 5 只股票**——组合宽度极窄，换手/重叠率/回撤的解读必须放在"5 只股组合"语境下）；**滑点 0 bps 不计**（散户口径，与米筐镜像一致）；卖出费率 0.00125（含印花税）；T+1、涨跌停、停牌、整手约束全开；仓位利用率 ≥ 80%、持仓重叠率 ≥ 50%、净超额年化 ≥ 3%、超额夏普 ≥ 0.5、最大回撤 ≤ 40%。

### 4. 盲测段黄金裁判（定性参考，遵守 2.2 纪律）

对每个 arm 的 **Top-3 候选因子**（选择规则见下），送入绝对隔离的盲测段测算：
- 盲测 Rank IC 与 ICIR（`BlindTestStage` 口径）；
- 盲测段完整约束回测（submit.py 已有 `test_gate` 链路：test 窗口跑 `run_engine_gate`，只报告不拦截）：扣全费净值、超额夏普、最大回撤、卡玛比率。

**Top-3 选择规则（v1.1 补——必须可复现）**：
1. 候选池排序键：**val 段 Rank IC 绝对值**（val 是挖掘过程中 LLM 已见过的最强样本外信号；test 段任何参与排序的操作都构成盲测污染）；
2. 同分按 `val_ic_retention` 高者优先；
3. 入池不足 3 个时按实际数量全送，**并在报告中明确标注样本数**——盲测对比只在不同 arm 样本数相同或差异不影响结论方向时进行。

---

## 四、 工程执行设计：自动化实验调度器

提供独立的命令行调度脚本 `scripts/run_ablation_study.py`，支持无人值守批量执行：

```powershell
# 1. 运行完整消融矩阵（Control + 8 arms，各重复 3 次、每次 30 轮，顺序执行）
python scripts/run_ablation_study.py --suite full --rounds 30 --repeats 3

# 2. 仅运行部分消融组
python scripts/run_ablation_study.py --arms control,A1_no_memory,A2_pos_only --rounds 30 --repeats 3

# 3. 汇总当前已完成实验组的消融对比矩阵并打印报告
python scripts/run_ablation_study.py --report artifacts/ablation_20260914/
```

### 调度器硬性要求（v1.1 新增）
- **402/断流重试**：LLM 渠道余额不足会秒级失败，调度器必须预检余额、失败 arm 自动重跑（最多 2 次）、支持 `--resume` 断点续传（已完成重复不重跑）；
- **参数快照**：启动时把全部 arm 的 research_spec 开关组合、commit hash、模型版本写入 `run_meta.json`；
- **盲测隔离**：调度器只负责跑挖掘，**盲测评估是独立第二阶段命令**（`--blind-eval`），强制人工确认全部 arm 完成后才执行；
- **失败隔离**：单 arm 失败不阻断其他 arm，最终报告标注失败原因。

### 实验产出结构：
```
artifacts/ablation_<timestamp>/
├── control/
│   ├── run_meta.json          # commit hash、模型版本、research_spec 开关快照
│   ├── rep1/ scorecard.json  # 3 次重复各自的 scorecard
│   ├── rep2/ scorecard.json
│   ├── rep3/ scorecard.json
│   └── blind_test_report.json # 第二阶段统一生成
├── A1_no_memory/
│   └── ...
└── ablation_summary.csv       # 全 arm × 重复 的对比大表（含均值±方差、置信级别列）
```

---

## 五、 预期结论与产出决策依据

本消融实验的终极目的，是为后续工程**提供双向证据**——无论是"做减法"还是"确认保留"，都需要铁证：

| 假设 | 若消融组 ≥ Control（差异超出重复方差） | 若消融组 < Control（差异超出重复方差） |
| :--- | :--- | :--- |
| **A1 无记忆** | 记忆系统过度引导先入为主偏见 → 大幅简化检索注入量，评估下线 APV 否决 | 记忆系统有效阻断死路 → 保持投入，可优化注入密度 |
| **A2 仅正向** | 死路否决在杀死有用变异 → 放宽 tau 或改为警示不拦截 | 负面证据拦截确实有效 → 保留 |
| **B1 极简 Prompt** | 教条提示词冗长且压制探索 → 深度瘦身，只留核心契约 | 知识注入确有赋能 → 保留，可精简表述 |
| **B2 剥离机制教学** | 领域知识无实盘增益 → 精简 | 机制教学降低亏钱因子产生 → 保留 |
| **C1 无预测对账** | 对账沦为八股负担 → 降为可选参数 | 对账在有效剔除假阳性（尤其看盲测质量是否骤降）→ **永久保留** |
| **C2 无门控消融** | 强制消融无增益 → 关闭省算力 | 门控消融拦截了拼接式过拟合 → 保留 |
| **D1 自由探索** | 父本变异压制了跨领域发现 → 放开探索轨 | 轨道约束有效聚焦 → 保留 |
| **D2 禁复合算子** | 复合算子纯过拟合 → 收紧算子目录 | 复合算子贡献真实非线性 alpha → 保留 |

> **判读纪律**：
> 1. 任何"消融组不差 → 该机制是负担"的单向解读都必须先过统计关（2.1）——差异在重复方差内一律写"未观察到显著差异"，不做工程变更；
> 2. 实盘存活与盲测维度（2.3 定性参考级）只能佐证不能定罪：Control 产出 1 个正式因子、A1 产出 0 个，不构成"A1 更差"的证据；
> 3. 主裁决冲突时（如 A1 探索广度更高但盲测质量更差），以盲测质量为最终取向——探索广度的意义最终要落到样本外存活，但前提是盲测样本数可比。

> AI生成
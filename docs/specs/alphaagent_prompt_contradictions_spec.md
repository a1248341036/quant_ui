---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '5db1b93d-b4f4-407e-a844-6bf9ad1638dd'
  PropagateID: '5db1b93d-b4f4-407e-a844-6bf9ad1638dd'
  ReservedCode1: 'd099cc50-3195-4f93-a0ad-f1c9ac4f1143'
  ReservedCode2: 'd099cc50-3195-4f93-a0ad-f1c9ac4f1143'
---

# SPEC: AlphaAgent System Prompt 规则自相矛盾全面审计与对齐规范

> 状态：Draft / 待评审  
> 版本：v1.0  
> 审计基准：Run 2026-09-18 全装配基线 Prompt（`artifacts/ablation_prompt_20260918/control_rep1/`，40,617 字符）  
> 对应模块：`alphaagent/factor/mining/prompt/modules/`  
> 目标：**彻底消除 System Prompt 中存在的数据打架、规则冲突、行为误导与逻辑自闭环缺陷，恢复 LLM 决策确定性**。

---

## 一、 背景与审计动机

在 2026-09-18 进行的 AlphaAgent 提示词单模块消融实验（15 Arm）中，基线 Control 组与多个消融组呈现出强烈的**“决策断层（Decision Gap）”**与**“同质化死循环”**：
1. **Control 组 27 个已过训练门槛（promising）的因子，LLM 全程未调用 `submit_factor` 提交入库**；
2. 模型中后期大量重复产生围绕 `$adj_close` 与 `$adj_vwap` 的平滑参数微调；
3. 实测所有提交均因日单边换手超标（`avg_daily_side_turnover > 0.50`）被 stage_one 硬拦截。

复盘昨晚消融实验实际注入给模型的完整真实 System Prompt（40,617 字符）后发现，Prompt 内部存在多处**前后打架、上下矛盾、甚至与底层运行时参数背道而驰**的硬性指令。这种规则撕裂直接导致大语言模型产生指令遵循困惑，进而在保守重复与无序探索之间摇摆。

---

## 二、 八大核心自相矛盾点详析

### 矛盾 1：并发调用量冲突（Prompt 强求 12~20 条 vs 底层硬顶 8 条）

- **真实 Prompt 原文**：
  - `strategy_tracks.py`：
    > “轨道 D：新族开拓（**每轮 12~20 条候选中至少一半（≥6 条）**，主轨道）”  
    > “每一轮：优先并行调用 **12~20 次 evaluate_factor(profile_id="train_screen")**...”  
    > “同一批 12~20 条 tool_calls 中，至少覆盖 6 个不同的信号族...”
  - `tool_examples.py`：
    > “建议每轮 12~20 条并行（**与 delivery_interface 的 12~20 次一致**，并行度越高吞吐越高）”
- **底层运行时配置**：
  - `alphaagent_factor_mining.py` 默认参数：`--max-tool-calls-per-round 8`，`--max-tool-workers 8`，`--max-parallel-eval 6`。
  - 昨晚消融实验脚本 `run_ablation_study.py` 未显式覆盖该值，全部以默认 8 条运行。
- **冲突危害**：
  - **规则撕裂**：LLM 如果听从 Prompt 发送 15 个并发 tool_calls，会被底层 ReAct 循环直接截断或报警；
  - **动作变形**：LLM 如果感知到底层限制发 8 条，又直接违反了 Prompt 里“至少 12~20 条、至少覆盖 6 个族”的硬性规定，诱发拒止或困惑。

---

### 矛盾 2：算子可用性红线打架（严禁 MULTIPLY vs 基础四则直用）

- **真实 Prompt 原文**：
  - `facet_focus.py` 与 `strategy_tracks.py`：
    > “**默认禁止 MULTIPLY 乘法交互**：本仓库默认不允许任何形式的算子相乘（含带契约的乘法），`MULTIPLY` 会被直接拦截。需要表达放大、抑制、条件依赖或状态切换时，一律改用结构化交互...”
  - `operator_catalog.py`：
    > “- 基础四则/比较/初等函数（**语义自明，直用**）：`ADD/SUBTRACT/MULTIPLY/DIVIDE(df1, df2)` 逐元素四则；”
- **冲突危害**：
  - 后文算子清单直接将 `MULTIPLY` 标记为“语义自明，直用”的基础四则算子；
  - 模型在缺乏强约束时顺理成章调用 `MULTIPLY` 做特征交叉，随即被 `_dispatch.py` 的消融检查层直接抛异常拦截，无端白耗评估预算。

---

### 矛盾 3：正交相关性门槛阶梯冲突（0.7 vs 0.5 vs 0.4）

- **真实 Prompt 原文**：
  - `strategy_tracks.py`（第三层：正交预判 Orthogonality Guard）：
    > “系统会在 DSL 求值前自动检查新因子与因子库中已有因子的截面 Spearman 相关性。**如果与任何已有因子的相关性 > 0.7，因子会被自动拦截**并返回冗余诊断。”
  - `delivery_interface.py` / `DeliveryCriteria`：
    > “第一阶段（候选登记，预筛池口径）：...与已有因子最大截面相关 **< 0.5**；”  
    > “第二阶段（精筛正式库，双窗口口径）：...与已有因子最大截面相关 **< 0.4**。”
- **冲突危害**：
  - 模型在初筛阶段被告知“相关性 < 0.7 即为合格正交”，因此欣然接受相关性为 0.55 ~ 0.65 的变体；
  - 但一旦调用 `submit_factor`，却立刻被候选池 `< 0.5` 或正式库 `< 0.4` 的严苛门槛无情打回；
  - Prompt 未对 0.7（求值硬拦截）、0.5（候选池准入）、0.4（正式库晋升）三级梯度进行逻辑分层说明，模型误以为 0.7 是最终交付指标。

---

### 矛盾 4：换手率红线口径不一致（行为规则 0.4 vs 引擎准入 0.50）

- **真实 Prompt 原文**：
  - `behavior_rules.py`（Rule 2，换手红线）：
    > “`evaluate_factor` / `eval_on_train_set` 结果里顶层的 `avg_daily_side_turnover`（日单边换手）**> 0.4 的候选不要调用 submit_factor**——消融实验与历史数据证明绝大多数候选均因此止步 stage_one/engine_gate...”
  - `delivery_criteria.py` 与引擎实际拦截日志：
    > 门槛代码真实设定为 `0.50`；  
    > 昨晚实测拦截日志：`avg_daily_side_turnover=0.72 > 0.50 (组合日单边换手过高，实盘不可交付)`。
- **冲突危害**：
  - 行为规则给 LLM 灌输的数值阈值（0.40）与系统真实交付判定（0.50）脱节；
  - 当模型产生一个日换手 0.45 的优秀候选时，本可通过准入门槛，却被行为规则强行恐吓放弃提交，加剧决策断层。

---

### 矛盾 5：变异策略（Exploit）与同质化熔断器机制冲突

- **真实 Prompt 原文**：
  - `strategy_tracks.py`（第二层：探索 与 变异双轨策略）：
    > “**轨 A：参数变异**：保持指定父因子结构不变，替换窗口/分位阈值，如 `TS_MEAN(.., 20)` → `TS_MEAN(.., 40)`；”  
    > “**轨 C：修饰变异**：在父因子外层增加衰减/平滑/中性化，如 `TS_DECAY(父因子, 5)`...”
  - `behavior_rules.py`（Rule 1）与动态 AST 熔断器：
    > “禁止微调已有平滑参数...连续同信号根平滑变体将被 AST 熔断器直接拦截。”
- **冲突危害**：
  - Prompt 的“第二层枷锁”教导模型采用轨 A 和轨 C 对父本做“参数与平滑微调”；
  - 但随后行为规则与系统的 AST 熔断器却将这种微调视作违例并实施硬拦截；
  - 模型无所适从：按规范变异则被拦截，不按规范变异又违反父本变异轨约束。

---

### 矛盾 6：预测周期（1d）与执行周期（Weekly）错配，反向惩罚平滑

- **真实 Prompt 原文**：
  - 数据口径：`label_col` 为 `label_1d_open_to_open`（次日开盘到第三日开盘，严格 1d 前瞻收益）；
  - 交付接口：强制指定 `submit_factor 的 rebalance_freq 必须传 "weekly"`；
  - 行为规则：警告模型“末位 EMA/WMA/TS_MEAN 等平滑算子压低了极端值分布，使得纯多头不可交易”。
- **冲突危害**：
  - 1d label 引导模型捕捉短期微观日频反转；
  - 但 engine_gate 强制以 weekly 频率调仓（持有 5 个交易日），未经平滑的纯 1d 信号在第 2~5 天几乎衰减为纯噪声，换手率自然居高不下（>80%）；
  - Prompt 一方面要求适应 weekly 调仓，另一方面又排斥模型使用长窗平滑（如 EMA20/WMA20）将快信号慢速化，造成实盘逻辑死结。

---

### 矛盾 7：负 IC 价值承认与回测引擎做多方向冲突

- **真实 Prompt 原文**：
  - `ic_robustness.py`：
    > “研究阶段可充分利用负 IC：负 IC 和高 ICIR 同为有效信号，各阶段均以 `abs(IC)` 和 `abs(ICIR)` 判断，不必为转正刻意在字段前取负号。”
  - `delivery_interface.py`：
    > “第二阶段准入：...`val 多头端年化超额 >= 0.0%`，`val/train IC 保留比 >= 50.0% 且方向不反转`。”
- **底层引擎撮合逻辑**：
  - `quantile_portfolio` 与 `engine_gate` 默认按因子值从大到小排序，固定买入 **顶层分位组（D10 / Top 0.1%）**；
  - 若模型保留负 IC 表达式（未套 `NEG()`），D10 恰好是未来收益最差的空头端，做多必定亏损，导致 `engine_gate` 净超额与超额夏普直接暴跌并不予晋升。
- **冲突危害**：
  - 提示词口头承诺“不必取负号”，但真实的交付回测却惩罚未取负号的负 IC 因子；
  - 导致大量优质反转因子因未翻转符号而在交付阶段死于 engine_gate。

---

### 矛盾 8：静态与分阶段裁剪下的悬空引用（Dangling References）

- **真实 Prompt 原文**：
  - `behavior_rules.py`（Rule 1 与 Rule 12）：
    > “输出 DSL 前必须写出经济因果链（严格遵循**第一层经济直觉**）...”  
    > “禁止单信号独走（严格遵循**第二层轨道 D** 硬约束）...”
- **模块剥离现状**：
  - 当进行模块消融（如 P2a 剥离 `strategy_tracks`）或在 `explore` 阶段动态精简时；
  - `strategy_tracks` 已被完全移出上下文，但 `behavior_rules` 仍然在命令模型遵循“第二层轨道 D”，产生不可解析的悬空指引。

---

## 三、 系统性修复与对齐方案 (Repair Spec)

为彻底解决上述 8 大矛盾，需对 `alphaagent/factor/mining/` 下的提示词与配置逻辑实施如下针对性重构：

### 1. 并发度参数化动态注入（修复矛盾 1）
- **改动点**：`alphaagent/factor/mining/prompt/modules/strategy_tracks.py` 与 `tool_examples.py`。
- **实现**：
  - 禁止在 Prompt 中硬编码 `12~20`；
  - 从 `PromptContext` 中获取运行时真实的 `max_tool_calls_per_round`（默认 8，批处理可配置）；
  - 动态渲染为：“每轮建议并发调用 **{batch_size} 次**（其中至少一半（≥{half_batch} 条）用于轨 D 新族开拓）”。

### 2. 算子目录严谨净化（修复矛盾 2）
- **改动点**：`alphaagent/factor/mining/prompt/modules/operator_catalog.py`。
- **实现**：
  - 在“基础四则/比较/初等函数”列表中，彻底移除 `MULTIPLY`；
  - 改为：`ADD/SUBTRACT/DIVIDE(df1, df2)` 逐元素基础算运算；
  - 附注明确提示：“乘法交叉默认关闭，请改用 `GATED_SIGNAL` / `CS_GROUP_RANK` / `CS_RESIDUALIZE` 等结构化算子”。

### 3. 正交性梯度分层 Clarification（修复矛盾 3）
- **改动点**：`alphaagent/factor/mining/prompt/modules/strategy_tracks.py` 第三分部。
- **实现**：
  - 将模糊的“> 0.7 拦截”重写为**三级正交梯度规范**：
    1. **求值硬拦截线（Spearman > 0.70）**：DSL 求值前诊断，超过直接拒绝计算，防无意义算力消耗；
    2. **候选池准入线（Pearson < 0.50）**：`submit_factor` stage_one 门槛，与正式库因子相关度必须小于 0.50；
    3. **正式库晋升线（Pearson < 0.40）**：stage_two 最终精选，必须小于 0.40。

### 4. 换手率口径完全统一（修复矛盾 4）
- **改动点**：`alphaagent/factor/mining/prompt/modules/behavior_rules.py`。
- **实现**：
  - 将 Rule 2 的 `> 0.4` 修正为与系统标准一致的表述：
    “系统交付硬门槛为 `avg_daily_side_turnover <= 0.50`（设计目标建议控制在 `<= 0.40`，留足安全边际；`> 0.50` 必定被拒，请勿直接调用 `submit_factor`）”。

### 5. 澄清变异与同质化平滑界限（修复矛盾 5）
- **改动点**：`strategy_tracks.py` 变异轨与 `behavior_rules.py`。
- **实现**：
  - 明确界定：**有效变异** 是指对“特征源（例如换用筹码/两融数据源）”或“结构机制（例如改用截面残差）”的变异；
  - **严禁变异** 是指在同一价量反转信号根上反复尝试 EMA5 / EMA10 / EMA20 / WMA15 等纯平滑参数网格搜索。

### 6. 持有期与调仓节奏协调指引（修复矛盾 6）
- **改动点**：`market_mechanisms.py` 与 `data_calibration.py`。
- **实现**：
  - 明确向模型解释：虽然使用 `label_1d` 评估日频敏感性，但实盘交付采用 `weekly` 调仓；
  - 引导模型构建**自带时间滞后耐受性**的因子（如中长周期量价背离、成交量钟 VPIN 累积、筹码支撑度），而非单纯依赖次日开盘脉冲的反转。

### 7. 负 IC 因子提交翻转纪律（修复矛盾 7）
- **改动点**：`ic_robustness.py` 与 `delivery_submission.py`。
- **实现**：
  - 明确提示：在调用 `evaluate_factor` 探索期，可自由保留负 IC 观察；
  - **但在调用 `submit_factor` 提交前，若因子预期为负向 alpha（负 IC），必须在表达式顶层套用 `NEG()` 转为正向收益因子**，以确保回测多头分位组 D10 正确买入强收益端。

### 8. 模块间解耦与自包含引用（修复矛盾 8）
- **改动点**：`behavior_rules.py`。
- **实现**：
  - 将 “严格遵循第一层经济直觉”、“第二层轨道 D” 等跨模块章节序号耦合，改为自包含的功能性表述（如“必须写出经济因果机制”、“必须保持新信息源开拓与父本变异的合理配比”），防止模块消融时上下文破损。

---

## 四、 实施清单与防回归验收

### 1. 修改文件清单
- `alphaagent/factor/mining/prompt/modules/strategy_tracks.py`
- `alphaagent/factor/mining/prompt/modules/operator_catalog.py`
- `alphaagent/factor/mining/prompt/modules/behavior_rules.py`
- `alphaagent/factor/mining/prompt/modules/ic_robustness.py`
- `alphaagent/factor/mining/prompt/modules/tool_examples.py`
- `alphaagent/factor/mining/prompt/modules/delivery_interface.py`

### 2. 自动化验收门禁
- **单元测试**：
  - 编写 `tests/test_prompt_consistency_audit.py`，断言生成的完整 System Prompt：
    1. 不包含 `MULTIPLY` 作为“语义自明直用”的描述；
    2. 并发量数字与传入的 `PromptContext.max_tool_calls` 完全相等；
    3. 换手率、相关性、IC 方向等数字全量对齐 `DeliveryCriteria` 的常数值。
- **基线更新**：
  - 运行 `pytest` 并同步刷新 `tests/fixtures/system_prompt_*.txt` 四份黄金测试基线。

---

> AI生成

# AlphaAgent 动态上下文（Prompt）与记忆层注入架构说明

> **更新日期**：2026-09-17  
> **配置基准**：OPT_effective_only（消融实验多轮重复验证落地）  
> **对应目录**：`alphaagent/factor/mining/prompt/` 与 `alphaagent/factor/mining/memory/`

---

## 一、 系统 Prompt 与动态上下文的装配体系

AlphaAgent 注入给 LLM 的上下文分为两层：
1. **静态 System Prompt 模块池 (`prompt/modules/`)**：定义 Agent 角色、算子字典、交易机制知识、工具契约与行为约束。
2. **动态每轮研究记忆注入 (`memory/retrieval.py::context_for`)**：在 ReAct 每轮交互中，根据历史挖掘记录动态检索并注入当轮的记忆块。

在 2026-09-16 ~ 2026-09-17 的组件级消融实验中，实证发现记忆层中存在若干**"过度说教型"**的上下文块，这些块不仅膨胀了 Prompt 长度，且引入了负向诱导（如僵化照抄模板、陷入套路死循环、压制跨领域探索）。

因此，系统全面启用了 **`OPT_effective_only`** 精简注入策略。

---

## 二、 记忆层动态注入模块现状（OPT 配置）

在 `research_spec.json` 的 `memory_policy` 中，各上下文注入块的启用状态如下：

### 1. 活跃注入块（保留的高价值事实资产）

| 注入上下文块 | 对应函数 | 状态 | 注入内容与在 Prompt 中的角色 |
| :--- | :--- | :---: | :--- |
| **已验证因子证据块** | `_evidence_block` | **ON**<br>`enable_factor_retrieval=True` | **客观事实基准**。通过 BM25 + 族亲和度 + 40% 正向配额（跨族保底），注入历史上与当前思路相近的因子表达式、方向与实测 IC/ICIR。作为模型变异和正交对齐的真实依据。 |
| **因子族产出率块** | `_yield_block` | **ON**<br>`enable_yield_block=True` | **宏观导航**。统计历史各类因子族（量价/筹码/两融等）的入池率与成功率，为 LLM 的大方向决策提供统计指引，防止在贫瘠方向过度耗费轮次。 |
| **面覆盖多样性块** | `_diversity_block` | **ON**<br>`enable_diversity_block=True` | **探索面拓宽**。实时统计近期尝试触及的数据面（若不足 3 面则发出警告），引导模型进行跨数据面特征交叉（如量能×股东集中度），防止思维局限在单一价量面。 |

### 2. 已剥离注入块（经消融证伪的负资产/噪声）

以下模块默认**不再向 Prompt / 上下文中注入**（相关开关设为 `False`，阈值设为放行）：

| 剥离模块 | 开关配置 | 为什么从 Prompt 中移除？（实证原因） |
| :--- | :--- | :--- |
| **长期经验块** (`_experience_block`) | `enable_experience_block: false` | 注入"固定成功模式/禁忌模板"后，LLM 倾向于死板照搬模板，导致有效尝试比低下（0.789 vs 0.94+）。移除后模型自由推导逻辑更顺畅。 |
| **变异先验块** (`_edit_prior_block`) | `enable_edit_patterns: false` | 注入微调先验建议（"某参数在某族更优"）严重局限了探索半径。消融实测移除后探索广度提高 3 倍（186 vs 63 次尝试）。 |
| **族饱和度块** (`_saturation_block`) | `enable_saturation_block: false` | 饱和度告警将"尝试次数多"机械视为"拥挤"，误伤了仍在稳定出高质量信号的核心族。 |
| **交互结构命中率** (`_structure_stats_block`) | `enable_structure_stats_block: false` | **关键负资产**。过度宣扬历史高通过率算子（如 `CS_GROUP_RANK`），导致模型无论何种机制都生搬硬套该算子，造成大量无意义撞死路。移除后撞死路率从 11.3% 降至 1.9%。 |
| **APV 双门硬否决** (`_apv_gate`) | `apv_tau_c: 1.0, apv_tau_v: 1.0` | 阈值置 1.0 永不触发硬否决。避免规则层提前杀掉尚未充分尝试的潜在有用变异。 |
| **SSPM 残差写入** (`_update_cell`) | `enable_sspm_write: false` | 停止写入衰减残差，从源头上切断变异先验噪声，降低数据库写竞争。 |

---

## 三、 Prompt 认知与约束机制联动

在精简掉"过度干预型"注入块后，Prompt 的质量把控完全由**三道客观硬核护栏**承担：

1. **预测-对账闭环 (`prediction_check`)**
   - 移除模版说教后，并不意味着放松要求。LLM 在调用 `evaluate_factor` 时**必须提交 `prediction`**（预期单调形态、强侧区间、IC 符号）。
   - 引擎用真实的十分位分布进行对账检验，被证伪（`contradicted`）则禁止继续微调。消融实测显示，精简记忆后对账确认率依然稳定在 **44.5%** 良好基线。
2. **强制消融检验 (`ablation_check`)**
   - 对门控/条件算子强制要求 base-only 对照，剥离虚假伪信号。
3. **换手率显性反馈 (`turnover_visibility`)**
   - 3 次重复实验中，未过线因子绝大部分死在 `avg_daily_side_turnover > 0.50`。Prompt 与反馈流已透传日均单边换手指标，促使模型自发引入 `TS_MEAN` / 衰减平滑算子控制换手。

---

## 四、 配置文件位置与维护

- **当前生效配置文件**：
  - `artifacts/alphaagent/research_specs/technical.json`
  - `artifacts/alphaagent/research_specs/fundamental.json`
- **默认代码回退点**：
  - `alphaagent/factor/mining/research_spec.py::DEFAULT_RESEARCH_SPEC`
- **前端查看与修改**：
  - Quant UI Web 界面右上方点击“策略配置/门槛”，在弹窗的“记忆策略”分栏中即可实时查看与热修改上述 8 个开关。

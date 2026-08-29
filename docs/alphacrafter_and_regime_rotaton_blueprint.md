---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '16fb3f97-9dab-494c-b546-bbbeda510fe4'
  PropagateID: '16fb3f97-9dab-494c-b546-bbbeda510fe4'
  ReservedCode1: 'c7ac96ee-0b02-4cdd-96c9-4ee2e4741cf6'
  ReservedCode2: 'c7ac96ee-0b02-4cdd-96c9-4ee2e4741cf6'
---

# AlphaCrafter 技术要点与 quant_ui 风格轮动实现蓝图

> 论文：Yuan, Y., et al. (2026). "AlphaCrafter: A Full-Stack Multi-Agent Framework for Cross-Sectional Quantitative Trading." arXiv:2605.05580.
>
> 来源：微信公众号"灵度智能"全文解析 + arXiv 摘要交叉验证

---

## 一、AlphaCrafter 架构总览

### 1.1 核心理念

传统量化系统的痛点：因子发现、因子选择、策略执行三段**碎片化**——因子挖掘假设因子有效性跨市场制度持续，执行系统引入行为噪声而非系统理性。AlphaCrafter 把三段统一到一个**闭环自适应**流程中，形成"假设 → 验证 → 执行"循环。

### 1.2 交易环境形式化

环境元组 `E = (M, Z, Π, T, J)`：

| 符号 | 含义 | quant_ui 对应 |
|------|------|---------------|
| M | 市场状态空间（可交易资产 + 宏观信息） | `close/high/low/volume` 面板 + 财务数据 |
| Z | 因子库 | 正式库 `factor_registry` + 候选池 |
| Π | 可允许交易策略空间 | `StrategyDefinition` + `BacktestConfig` |
| T | 交易天数离散集 | 回测时间轴（`window_config.py` 管控） |
| J | 策略表现评估函数（惩罚回撤和波动） | `compute_excess_metrics`（夏普/最大回撤/卡玛） |

**共享内存 H**：所有智能体可访问，定期总结保持持久，让系统随市场变化集体调整行为。
→ 对应 quant_ui 的 `ResearchMemoryStore`（SQLite，存储推荐/禁止/战略洞察），但 AlphaCrafter 的 H 是跨智能体实时共享的，你的研究记忆目前只在挖掘阶段使用。

---

## 二、三智能体详解

### 2.1 Miner（挖掘智能体）

**职责**：自主生成因子 → IC 验证 → 纳入因子库 → 定期重验剔除衰减因子。

**策略 P_M**：
1. 基于当前因子库 Zₜ、资产宇宙 U 及记忆 Hₜ 运行
2. 生成循环：LLM 生成候选因子表达式
3. 用 IC/ICIR 等指标在历史数据上验证
4. 通过的因子附元数据保存到因子库
5. 验证结果记录在案
6. 后续维护阶段：重新验证现有因子，**剔除性能大幅衰减的因子**
7. 当因子质量和库多样性达标时，自主终止探索

**quant_ui 对照**：

| AlphaCrafter Miner | quant_ui 现有 |
|-------------------|--------------|
| LLM 生成候选因子 | FactorMiner AgentScope（DSL → 表达式 → 评估） |
| IC/ICIR 验证 | `compute_ingest_metrics`（IC/ICIR/coverage/autocorr） |
| 纳入因子库 | submit 流程（盲测终审 → stage_one → stage_two → 正式库） |
| 剔除衰减因子 | **缺失**——因子一旦进正式库无退出机制 |
| 自主终止 | mining 记忆 + 质量分门控 |

**差距**：AlphaCrafter 有**因子退役机制**（定期重验 → 衰减剔除），quant_ui 的因子进正式库后没有退出路径。

### 2.2 Screener（筛选智能体）★风格轮动核心★

**职责**：结合因子库信号与市场状况 M，提炼出连贯的因子集合 Eₜ。

**策略 P_S**：
1. 分析个股和指数行为，过滤多方面信息 → **判断市场制度 R̂ₜ**
2. 基于制度诊断评估各因子 f 的**适用性评分**
3. 按制度条件适用性评分对因子排序
4. 考虑相关性降低集中风险
5. 选择性组合形成适应市场动态的集合 Eₜ
6. 输出结构化集合 + 更新记忆 H

**核心逻辑**：
- 不同因子在不同市场制度下表现不同（如动量因子在趋势市强、反转因子在震荡市强）
- Screener 的作用是**根据当前市场状态选择最适配的因子子集 + 权重**
- 消融实验：去掉 Screener → **MDD 增幅最大**，证明 regime 感知选择是风险缓解的关键

**quant_ui 对照**：

| AlphaCrafter Screener | quant_ui 现有 |
|----------------------|--------------|
| 市场制度诊断 R̂ₜ | `SelectionPolicy.regime_adx`（ADX 中位数 < 阈值 → 弱市降仓） |
| 因子适用性评分 | **缺失** |
| 按 regime 选因子子集 | **缺失**——`build_composite_factor` 用静态固定权重 |
| 相关性去冗余 | `build_stacking_dataset` 有贪心冗余过滤（|corr|>0.6 剔除），但只在训练时做一次 |
| 更新记忆 H | 研究记忆只存因子级结论，不存 regime-因子映射 |

**差距**：这是 quant_ui 与 AlphaCrafter **最大的差距**。现有 `regime_adx` 只是静态缩放权重（弱市 ×0.5），不切换因子。`build_composite_factor` 的权重是固定的 `{因子名: 权重}`，不随市场状态变化。

### 2.3 Trader（交易智能体）

**职责**：整合因子集合 Eₜ + 市场制度 R̂ₜ + 风险约束，制定交易策略 πₜ。

**策略 P_T**：
1. 超参数优化：探索参考策略 π_ref 的配置空间 Θ
2. 注入辅助规则和风险敞口约束
3. 历史数据回测评估候选策略
4. 选择使风险调整目标最大化的配置
5. 在实时资产上执行投资组合
6. 根据执行结果更新记忆，实现策略持续优化

**目标函数**：J 对回撤和波动做惩罚，追求稳定资本增值。

**quant_ui 对照**：

| AlphaCrafter Trader | quant_ui 现有 |
|--------------------|--------------|
| 超参数优化 | `walkforward.py` 的 `rolling_train_test_factor`（滚动选最优 top_n/freq） |
| 风险约束 | `engine_gate`（trailing stop / max drawdown / daily overlap / invested ratio） |
| 回测评估 | `run_backtest` + `compute_excess_metrics` |
| 自适应执行 | `SelectionPolicy`（top_n/top_pct/industry_cap/min_score 门控） |
| 策略持续优化 | **缺失**——回测后不反哺策略参数优化 |

---

## 三、Alpha 衰减管理

### 3.1 论文发现

- 2024-01 ~ 2026-01 四个连续半年期 alpha 衰减分析
- 静态因子集（全局前 20）：IC 波动大甚至为负
- 定期更新前 20：通过频繁更新保持高 IC
- 三种智能体方法（RD-Agent、AlphaAgent、AlphaCrafter）：IC 稳定在 0.015-0.025
- **AlphaCrafter 后期衰减最小**：动态因子管理（Miner 重验 + Screener 选择）是关键

### 3.2 quant_ui 现状

- 盲测终审（刚实现）：在 submit 时用 test 段做终审，但这是**入库时的一次性检验**，入库后不再跟踪
- ML holdout 报告显示 OOS IC 波动大（ridge：+0.013 → +0.026 → -0.017 → +0.088），整体混合 IC 仅 0.0032
- 候选池 5 个因子 → 正式库 0 个，说明现有因子在 test 段（2025-01 ~ 2026-08）表现不佳

---

## 四、quant_ui 风格轮动实现路线

### 路线 A：市场 Regime 切换（最轻量，推荐先行）

**改动范围**：`core/selection.py` + `core/engine.py`

**核心思路**：
1. 在 `build_composite_factor` 之前插入 **regime 检测层**
2. 定义 2-3 个市场制度：趋势上行 / 震荡 / 趋势下行
3. 每个 regime 预配一组因子权重（如趋势市偏动量、震荡市偏反转）
4. 按日线计算 regime，切换到对应权重组合

**实现要点**：
```
# 新建 core/regime.py
def detect_regime(close: pd.DataFrame, index_code: str = "000300") -> pd.Series:
    """返回每日 regime: 1=趋势上行, 0=震荡, -1=趋势下行
    
    指标组合：
    - ADX > 25 + 指数 > MA60 → 趋势上行
    - ADX > 25 + 指数 < MA60 → 趋势下行
    - ADX < 25 → 震荡
    """
    
# engine.py build_composite_factor 改造
def build_regime_composite_factor(
    close, am20, turn20,
    regime_weights: dict[int, dict[str, float]],  # {regime_id: {因子名: 权重}}
    regime_series: pd.Series,
    ...
) -> pd.DataFrame:
    """按日线 regime 切换权重组合，合成因子得分。"""
```

**优点**：
- 改动集中，不破坏现有 `build_composite_factor` 接口
- regime 检测可复用现有 `_compute_adx` 函数
- 前端只需加一个"regime 权重配置"面板
- 可直接在 `FactorLab.vue` / `AlphaAgent.vue` 暴露

**预期收益**：
- 趋势市自动加电动量因子，震荡市自动加电反转因子
- 减少"好因子在错的市场制度下亏钱"的问题
- 对应 AlphaCrafter Screener 的核心逻辑（最小可用版）

### 路线 B：滚动 IC 加权（中等）

**改动范围**：`core/engine.py` `build_composite_factor` + 新建 `core/rolling_weighter.py`

**核心思路**：
1. 用 walkforward 框架按滚动窗口（如 60 交易日）计算各因子近期 IC
2. IC 高的因子权重放大，IC 低的权重缩小
3. 每日重新计算权重，动态合成

**优点**：
- 自适应：因子衰减时自动降权，无需硬编码 regime
- 对应 AlphaCrafter "动态跟踪因子库有效因子"的理念
- 可与路线 A 叠加（先 regime 切因子池，再滚动 IC 调权重）

**风险**：
- 滚动 IC 噪声大，需要平滑（EMA）
- 用近期 IC 选因子再评估 = 前视偏差风险，需设计独立验证段

### 路线 C：ML 组合 + Regime 特征（最重，长期演进）

**改动范围**：`alphaagent/factor/stacking/` + `core/engine.py`

**核心思路**：
1. 在 stacking 的特征矩阵中加入 regime 特征（ADX / 波动率 / 指数相对均线位置）
2. 让 ML 模型隐式学习"不同 regime 下用不同因子"
3. walk-forward 训练保证无前视偏差

**优点**：
- 最接近 AlphaCraifter 的端到端自适应
- ML 自动发现 regime-因子映射，无需人工预设

**风险**：
- 特征工程复杂，训练成本高
- 现有 12 个基础因子 + 4 折 walk-forward 数据量有限，容易过拟合

---

## 五、因子退役机制（建议同步实现）

### 问题

当前 quant_ui 因子一旦进正式库就永久存在。AlphaCrafter 的 Miner 有定期重验 + 剔除衰减因子的机制。

### 建议方案

1. **定期重验**：新增 `scripts/revalidate_factors.py`，按月/季对正式库所有因子在最近 N 个交易日重新计算 IC/ICIR
2. **退役规则**：
   - 近 6 个月 IC 均值 < 入库时 IC 的 30% → 标记"衰减"
   - 连续 2 个季度标记衰减 → 移入退役库（不删除，保留历史）
3. **联动 Screener**：衰减因子在 regime 选因子时自动降权或排除

---

## 六、实施优先级建议

| 优先级 | 事项 | 估计工作量 | 依赖 |
|--------|------|-----------|------|
| P0 | commit 当前 34 文件改动 | 10 min | 无 |
| P1 | 路线 A：regime 检测 + 权重切换 | 2-3 天 | 无 |
| P2 | 因子退役机制 | 1 天 | 无 |
| P3 | 路线 B：滚动 IC 加权 | 2-3 天 | P1（可叠加） |
| P4 | 路线 C：ML + regime 特征 | 5-7 天 | P1 + P2 |

---

## 七、参考资源

| 资源 | 链接 | 备注 |
|------|------|------|
| AlphaCrafter 论文 | arXiv:2605.05580 | arxiv.org 直连 403，需 VPN |
| AlphaCrafter 深度解析 | 微信公众号"灵度智能" | 已成功获取全文 |
| RD-Agent（微软） | github.com/microsoft/RD-Agent | NeurIPS 2025，多智能体因子-模型联合优化 |
| AlphaForge | github.com/DulyHao/AlphaForge | AAAI 2025，LLM 驱动因子挖掘 |
| 量化 Agent 论文综述 | 知乎"量化Agent元年" | 61 篇论文六大方向，知乎 403 需缓存 |

## 八、借鉴清单：从 AlphaCrafter 拿什么改 quant_ui

> 以下按 **"值得借鉴且 quant_ui 没有"** → **"已有但不需借鉴"** → **"两者都没做但建议自研"** 三层组织。

### 8.1 值得借鉴（AlphaCrafter 有、quant_ui 没）

| # | 借鉴点 | AlphaCrafter 实现位置 | 价值评级 | 落地难度 | 对你两个痛点的关联 |
|---|--------|----------------------|---------|---------|-------------------|
| 1 | **Screener：regime 感知因子选择 + 动态权重** | `screener.py` 指令 + `factor_screening.md` 技能 | ★★★★★ | 中 | 间接解决"相悖组合"——不同 regime 下自动选不同因子，避免动量+反转同时高位 |
| 2 | **因子退役机制（定期重验 + deprecated 标记）** | `miner.py` 指令：90天周期重验，失败标记 `_deprecated` | ★★★★☆ | 低 | 直接解决因子衰减——入库后持续跟踪，衰减因子自动降权/退场 |
| 3 | **AST 多样性度量（树编辑距离）** | `calculate_diversity.py`（Zhang-Shasha 算法，Φ_intra + Φ_inter） | ★★★★☆ | 中 | 直接解决"父本变异→近亲繁殖"——量化因子表达式结构相似度，变异过近时拦截 |
| 4 | **TF-IDF 因子语义搜索** | `search_factor.py`：因子库 TF-IDF 向量化 + 余弦相似度检索 | ★★★☆☆ | 低 | 可替代现有 BM25 检索，支持更精准的"相似因子已存在"判断 |
| 5 | **多 Miner 并发 + 独立上下文** | `main.py`：默认3个 Miner 并发，各自只看自己的上一轮输出 | ★★★☆☆ | 中 | 缓解近亲繁殖——多个独立进化分支，降低单链路变异趋同 |
| 6 | **Screener 语义去重（TF-IDF 相似度 > 0.8 拒绝）** | `factor_screening.md`：筛选阶段对因子描述做 TF-IDF 去重 | ★★★☆☆ | 低 | 选因子时去冗余，避免选出高度相似的因子组合 |
| 7 | **Trader 超参数 regime 采样** | `strategy_construction.md`：按 regime 采样 N_long/N_short/beta/gamma | ★★★☆☆ | 中 | 不同市场制度下用不同持仓数量和风险敞口，而非固定参数 |
| 8 | **跨智能体共享内存 H（实时同步）** | `main.py`：Miner→Screener→Trader 共享 memory.txt | ★★☆☆☆ | 高 | 你的研究记忆目前只在挖掘阶段用，可考虑扩展到选因子和回测阶段 |

### 8.2 已有能力强项（quant_ui 已做、不需要借鉴）

| quant_ui 已有能力 | 对应 AlphaCrafter 能力 | quant_ui 优势 |
|-------------------|------------------------|--------------|
| **因子族分类 + 饱和度跟踪**（`research_memory.py` `_FAMILY_RULES` 11族 + `compute_saturation()`） | AlphaCrafter 无族分类概念 | 你已按经济逻辑分族并跟踪拥挤度，比 AlphaCrafter 粗粒度的 AST 距离更有经济含义 |
| **新颖性评分 + 门禁**（`factor_reviewer.py` novelty=high/medium/low + `minimum_novelty` 门禁） | AlphaCrafter 无新颖性门禁（AST 多样性是事后分析，不拦截） | 你在入库前就做新颖性拦截，比 AlphaCrafter 事后度量更前置 |
| **规则蒸馏模式记忆**（`distill_batch_patterns()` 推荐/禁止/战略洞察，零 LLM 成本） | AlphaCrafter 用 LLM 写 memory.txt（有成本） | 你的零成本规则蒸馏更适合大规模批量挖掘 |
| **BM25 正负池检索**（`_factor_retrieval_block()` 正负样本独立打分 + 正样本配额） | AlphaCrafter 用 TF-IDF 余弦相似度（单一正样本检索） | BM25 + 正负池更适合"避免类似失败"的场景 |
| **消融矩阵**（`ablation_matrix()` 提交期注入逐腿对照） | AlphaCrafter 无 | 直接暴露"哪些运算腿贡献 alpha" |
| **盲测终审硬门禁**（刚实现，test 段 IC 保留比 + 方向一致性，提前到 stage_one 之前） | AlphaCrafter 的 Miner 重验是事后 90 天，不是入库前盲测 | 更严格——入库前就拦截过拟合因子 |

### 8.3 两者都没做、建议自研（针对你的两个痛点）

| # | 缺失能力 | 解决什么痛点 | 建议方案 | 难度 |
|---|---------|-------------|---------|------|
| A | **经济直觉模板约束** | 缺经济直觉→相悖组合 | 为每族因子定义"经济假设模板"（如动量族：假设近期趋势延续），LLM 生成时必须声明遵循哪个模板，Reviewer 校验生成逻辑与模板一致性 | 高 |
| B | **因子族谱 + 变异深度限制** | 父本变异→近亲繁殖 | 记录每个因子的"父本链"（parent_id 序列），变异深度超过 N 代或与父本 AST 距离 < 阈值时拦截 | 中 |
| C | **因子逻辑一致性校验** | 缺经济直觉→相悖组合 | 提交时校验"因子组合中是否存在经济含义矛盾"（如同时持有动量和反转因子且权重接近） | 中 |
| D | **Regime-因子映射记忆** | 风格轮动缺数据基础 | 研究记忆扩展一类记录："regime X 下因子族 Y 的历史 IC 表现"，为 Screener/路线 A 提供先验 | 低 |

### 8.4 综合建议：分三步走

**第一步（立即可做，1-2天）**：
- 借鉴点 #2 **因子退役机制** → 新建 `scripts/revalidate_factors.py`，月度重验正式库因子，衰减标记 + 联动降权
- 借鉴点 #4 **TF-IDF 因子语义搜索** → 替代/补充现有 BM25，提升"相似因子判断"准确度
- 自研 D **Regime-因子映射记忆** → 扩展研究记忆 schema，记录 regime-因子族 IC 表现

**第二步（1-2周）**：
- 借鉴点 #1 **Screener regime 感知选择** → 实现路线 A（regime 检测 + 权重切换）
- 借鉴点 #3 **AST 多样性度量** → 将 `calculate_diversity.py` 的 Zhang-Shasha 算法移植为 quant_ui 的因子变异门禁
- 借鉴点 #5 **多 Miner 并发** → AlphaAgent 已有双 Agent（FactorMiner + FactorReviewer），可扩展为多 Miner 独立分支
- 自研 B **因子族谱 + 变异深度限制** → 在 submit 流程记录 parent_id 链 + AST 距离门禁

**第三步（长期，3-4周）**：
- 借鉴点 #7 **Trader 超参数 regime 采样** → walkforward 扩展 regime 分组优化
- 自研 A **经济直觉模板约束** + C **逻辑一致性校验** → 需要设计模板体系 + LLM 校验流程
- 借鉴点 #8 **跨智能体共享内存** → 研究记忆从"仅挖掘阶段"扩展到选因子/回测阶段

---

> AI生成
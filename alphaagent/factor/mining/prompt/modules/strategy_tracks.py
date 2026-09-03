# -*- coding: utf-8 -*-
"""模块 02 · strategy_tracks：三层枷锁（经济直觉/双轨策略/交互契约/正交预判）。

分阶段注入：
- explore：第一层经济直觉 + D 轨新族开拓 + 多因子交互约束
- deepen/deliver/full：完整三层枷锁（含 A/B/C 轨变异 + 研究记忆阅读 + 正交预判）
"""

# ── 第一层：经济直觉强制 ──
_LAYER1 = """# 理性枷锁（三条硬性约束，违反即跳过本轮）

本系统通过三层枷锁约束因子生成质量，确保每条因子都有经济意义、继承优秀基因、且与已有因子正交。

## 第一层：经济直觉强制（Economic Intuition First）

**在输出任何 DSL 表达式之前，必须先在思维链中用 50 字以内写出《经济直觉》**：说明为什么这些算子组合在一起能预测未来收益。经济直觉须满足：
- 精确描述信息从何而来（如"上方筹码峰是套牢盘压力位"）
- 精确描述为什么该信息与未来收益有关（如"压力位上方卖压释放后弹性更大"）
- 精确描述算子组合的因果链条（如"CHIP_PEAK_LOC 定位压力位 → TS_PCTCHANGE 量化回撤深度 → 距压力位越远弹性越高"）

**因果链必须是可证伪的**：评估调用时必须同步传 `prediction`（预期十分位形态 expected_shape /
预期 alpha 集中端 expected_strong_side / 预期 IC 符号 expected_sign / 可选证伪条件 falsifier）。
系统自动对账并在结果中返回 `prediction_check`：verdict=contradicted（被证伪）说明机制错误——
**错误的结构调参数不会变对**，应换机制或放弃；partial（部分命中）可继续但须解释形态偏差。

**如果你的经济直觉属于以下任何一种，禁止生成表达式，直接跳过该候选：**
- "A 和 B 可能有关系" → 缺因果链
- "X 是一个好的因子" → 无机制描述
- "类似已有因子 Y" → 无独立逻辑
- "动量/反转/波动率" 等单一标签 → 缺算子级因果
- 无法写出预期十分位形态与证伪条件 → 不可证伪的叙事，视为无机制

**好的经济直觉示例**（可直接用于 comment 字段）：
- ✅ "上方筹码峰（CHIP_PEAK_LOC）是套牢盘压力位；价格回撤越深，距压力位越远，上方卖压越小，后续反弹弹性越大。用 NEG(TS_PCTCHANGE) 量化回撤深度，与筹码峰位置做交互。"
- ✅ "隔夜跳空反映非交易时段信息冲击；VWAP 偏离反映日内主力的成本基准。两者背离时，隔夜信息被日内交易消化但未完全定价，次日开盘存在修正空间。"

"""

# ── 第二层 D 轨：新族开拓 ──
_TRACK_D = """## 第二层：探索 × 变异双轨策略（Explore × Exploit）

搜索 = **新族开拓（D 轨，探索）** 与 **父本变异（A/B/C 轨，深耕）** 两条轨道并行，禁止所有候选都挤在单轨：

### 轨道 D：新族开拓（每轮 12~20 条候选中**至少一半（≥6 条）**，主轨道）

- **D 新族**：一个研究记忆中尚无正/负证据的**信号机制**——不是任何既有父本的变体，核心信息源或经济机制与已评估因子不同。
- 新族同样必须先写 50 字经济直觉因果链，禁止无机制的随机算子拼装。
- **机制驱动探索（先选摩擦，再选算子）**：新族提案必须指名一个市场机制/摩擦，而不是先挑冷门算子再编故事。
  每条 D 轨候选的《经济直觉》必须回答三问：
  1. **谁在系统性错边**（谁在不基于价值地买/卖：追涨散户、被动资金、解禁抛压、注意力驱动的跟风盘…）；
  2. **错误为什么短期修不平**（套利受限来源：不能做空、T+1、涨跌停封板、流动性差、无人覆盖、成本高于错误幅度…）；
  3. **哪个字段能观测到它**（明确到 `$列` 与窗口，而不是"量价关系"）。
  三问缺一 = 该候选按无效跳过。机制方向参考（不限于）：信息扩散时滞（公告→定价渐进）、
  注意力反转（龙虎榜/涨停后的过热回吐）、流动性提供方补偿（套牢盘/被动接盘）、
  供给冲击日历（解禁/增减持/指数调仓）、制度摩擦（T+1/涨跌停/券源稀缺）、
  机构行为日历（季末/申赎/分红）、筹码结构变化（集中度/接力/对倒）。
  - 算子覆盖提示（次要线索）：机制选好后用它匹配信息源——拥挤度 `CROWD_*`、筹码结构 `CHIP_*`、
    影线/形态几何 `WICK_EFFICIENCY`/`KLINE_GEOMETRY`、量钟 `VOLUME_CLOCK_VPIN`、量价互信息
    `MUTUAL_INFO_LAG`、排列熵 `TS_PERMUTATION_ENTROPY`、缺口 `PRICE_GAP_*`、分型 `TS_LAST_*FRACTAL`、
    趋势非参数度量 `TS_TREND_RANK`。冷门算子只是低相关性的候选载体，**不能替代机制三问**。
- **新族晋级**：新族因子 |IC| ≥ 0.02 即成为新父本，纳入 A/B/C 轨深耕；连续 3 个新族 IC < 0.01 → 该机制记入负证据，本轮再换一个机制。
- **禁止低级信号叠加（硬约束）**：D 轨表达式顶层**禁止**使用 `ADD(RANK(x), RANK(y))` 或 `SUBTRACT(RANK(x), RANK(y))` 这种"两个独立信号简单加减"的形式——这只是把两个弱信号拼在一起，没有经济机制上的交互。如果确实需要融合多个信息源，必须使用**至少一层结构化交互算子**：门控 `GATED_SIGNAL`、组内排名 `CS_GROUP_RANK`、残差化 `CS_RESIDUALIZE`、背离 `DIVERGENCE_RANK`、分段状态 `PIECEWISE_STATE`、时序相关 `TS_CORR`/`TS_RANKCORR`、必要条件 `IF_THEN_ELSE`。例外：`ADD(x, RANK(y))` 中 x 本身已经是复合结构（如 `CS_RESIDUALIZE(...)` 输出）时不在此列——拦截的是"两个裸 RANK/TS_ 信号直接相加"。
- **单机制深度优先**：鼓励在单一信息源上构建多层算子链（如 `TS_PCTCHANGE → CS_ZSCORE → CS_NEUTRALIZE → TS_DECAY`），而非拼接多个浅层信号。单机制深度因子有更清晰的因果链条，且与已有因子正交性更好。
- **条件化前先消融**：门控/条件结构（GATED_SIGNAL / IF_THEN_ELSE / PIECEWISE_STATE / CS_GROUP_RANK）默认是"基信号 × 状态过滤"——
  状态过滤经常摧毁基信号而非增强它（反转类信号在"强势股"条件下必然失效）。契约必须传 `base_expr`，
  系统自动跑 base-only 对比并返回 `ablation_check`；`conditioning_destroyed_value` / `conditioning_flipped_signal`
  = 该门控方向错误，删门控或反转条件，不要在这个门控上调参。"""

# ── 第二层 A/B/C 轨：父本变异（探索阶段裁掉） ──
_TRACK_ABC = """

### 轨道 A/B/C：父本变异（每轮合计 ≤ 6 条，只深耕高质量父本）

1. **锁定父本**：从研究记忆中已验证的因子（见下方"长期研究记忆"段）选择 ICIR 绝对值最高、**且 val 保留比 ≥ 0.8 或已 approve/入库** 的 1-2 个作为父本。衰减严重（保留比 < 0.65）或被 reviewer 判 revise 的因子**不得作为父本**——变异它只会复制同样的衰减。
2. **变异饱和冻结**：研究记忆中同一信号族已有 ≥3 条评估记录（无论正负）→ 该族变异冻结，本轮只能以 D 新族开拓新机制。变体堆积不产生新信息。
2. **三种合法变异**：
   - **A. 参数变异**：保持父本算子结构不变，替换窗口/分位参数（如 TS_MEAN(…, 20) → TS_MEAN(…, 40)）
   - **B. 算子变异**：替换核心运算符但不变信息源（如 DIVIDE → CS_RANK，TS_STD → TS_VAR）
   - **C. 修饰变异**：在父本外层叠加衰减/平滑/中性化（如 TS_DECAY(父本, 5)、CS_NEUTRALIZE(父本, 行业)）
3. **父本声明（A/B/C 轨必填）**：变异候选调用 `evaluate_factor` / `eval_on_train_set` / `eval_on_val_set` / `submit_factor` 时必须传 `parent_factor`（父本因子逻辑名）与 `edit_note`（意向编辑，固定格式 `edit=<motif> <参数变化>`，motif 取 `window_rescale`（参数变异）/ `operator_substitute`（算子变异）/ `normalization_change`（修饰变异），如 `edit=window_rescale 10→20`）。comment 中保留一句变异说明。
   - 工具返回的 `memory_advisory` 是研究记忆的硬提醒（同结构死路 / 编辑方向被否决）：**命中后必须换方向**；无视提醒重复提交只会积累更多负证据。"""

# ── 如何阅读研究记忆（探索阶段裁掉） ──
_READ_MEMORY = """

### 如何阅读每轮注入的"长期研究记忆"块

该块出现在每轮任务消息之前，是跨 run 沉淀的历史证据，各段含义与用法：

- **经验记忆**：跨因子蒸馏的结论，优先级最高，两种可执行条目：
  - **【禁止】行（DO NOT）**：给出参数槽模板（如 `RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {w1})))`）与死路因子名单——本轮不得生成该骨架的参数变体，也不得把它们作为父本。
  - **成功模式行**：给出「模板 + 真实示例表达式 + 达标率」——**优先照抄模板骨架、只换参数/修饰算子**，这是历史上验证过的高产路线。
- **编辑方向先验**：A/B/C 轨选择编辑类型时的依据。每行 = 一个「信号族 × 编辑类型 × 父本质量桶」场景的历史成败统计，行尾给出行动指令（优先采用/优先尝试/禁止/谨慎避开）。**仅当该行场景与你正要做的变异匹配时才生效**；「父本质量桶」指统计所基于的父本强弱，你在弱父本上变异时只看 low 桶行即可。
- **已验证 / 有潜力的因子**：候选父本清单——A/B/C 轨从这里挑父本，在其邻近空间做不重复的变异。
- **已否定 / 不足的因子**：具体死路清单——不要重复相同结构，也不要把它们作为父本。
- **饱和度警告**：列出的因子族已拥挤，避开同质微调。

### 任务消息里的「本轮记忆推荐」名额约定

任务消息中若出现 `## 本轮记忆推荐`，前 k 个 `evaluate_factor` 名额**优先**用于推荐方向：按行内给出的父本（`parent_factor`）与编辑类型做恰好一条变异，`edit_note` 按行内格式书写。这是历史统计选出的高残差×高置信方向——比自由探索的期望收益更高；推荐方向与本轮其它假设并行提交，不互斥。

### 轨道切换规则

- 连续 2 个因子 IC < 0.01 → 本轮**全部**转 D 新族开拓（不再是"允许"而是强制）。
- 研究记忆对某机制已有负证据 → 该机制不得再作为 D 新族提出，也不得作为父本。"""

# ── 第二层补充：多因子交互约束 ──
_INTERACTION = """

## 第二层补充：多因子交互必须先选机制，再写公式

**⚠️ 硬性规则：只要表达式中出现以下任何算子名（包括作为中间变量的一部分），
就必须在调用 `evaluate_factor` / `submit_factor` 时同时传入 `interaction` 参数。**

触发拦截的算子：`MULTIPLY`, `TS_CORR`, `TS_COV`, `TS_RANKCORR`, `MUTUAL_INFO_LAG`,
`GATED_SIGNAL`, `CS_GROUP_RANK`, `CS_RESIDUALIZE`, `DIVERGENCE_RANK`, `PIECEWISE_STATE`, `IF_THEN_ELSE`。

**未传 `interaction` 参数 → 工具直接拦截，不会执行表达式，返回错误信息。**

interaction 契约格式：

```json
{
  "interaction_type": "gated_signal",
  "base_signal": "短期反转压力",
  "condition_signal": "流动性/关注度状态",
  "economic_mechanism": "过度反应在套利资金更容易进入的股票中被更快修正",
  "expected_subgroup_pattern": {"high_state": "更强", "low_state": "更弱或无信号"},
  "ablation_required": true,
  "base_expr": "NEG(TS_PCTCHANGE($adj_close, 5))",
  "condition_expr": "RANK(TS_MEAN($amount, 20))"
}
```

`base_expr`（主信号 DSL 子表达式，不含门控）在门控/条件类交互下**必传**：系统自动跑
base-only 消融并在结果中返回 `ablation_check`（条件化增量/破坏的量化对比）；
`condition_expr`（仅条件腿）可选。未传 `base_expr` 时结果会带 `ablation_hint` 提醒，
且你将无法区分"门控带来增量"与"门控摧毁基信号"。

**完整调用示例：**

```
evaluate_factor(
  multi_line_expr="vp = TS_RANKCORR($volume, $adj_close, 20)\nRANK(vp)",
  factor_name="vol_price_rankcorr",
  interaction={
    "interaction_type": "rolling_relation",
    "base_signal": "$adj_close",
    "condition_signal": "$volume",
    "economic_mechanism": "量价相关性反映趋势确认或主力对倒，高相关时动量持续性更强"
  }
)
```

优先使用以下模板；模板中的 `base` / `state` 必须有独立经济含义：

| interaction_type | 用途 | 推荐 DSL 骨架 |
|---|---|---|
| `gated_signal` | 只在状态出现时启用主信号 | `GATED_SIGNAL(base, state, 0.8, true, 0)` |
| `conditional_group_rank` | 同一状态组内比较主信号 | `state_n = CS_BUCKET(state, 5)` → `CS_GROUP_RANK(base, state_n)` |
| `residual_signal` | 剥离控制变量后保留独立信息 | `CS_RESIDUALIZE(base, control)` 或双控制版本 |
| `divergence_signal` | 两个应互相确认的信号背离 | `DIVERGENCE_RANK(signal_a, signal_b)` |
| `rolling_relation` | 两个变量的时序关系本身有信息 | `TS_RANKCORR(x, y, 20)` |
| `piecewise_state` | 主信号在不同状态下方向不同 | `PIECEWISE_STATE(base, state, 0.2, 0.8, 1, -1, 0)` |
| `necessary_condition_signal` | 满足必要条件才启用信号 | 条件面板 + `IF_THEN_ELSE(condition, base, 0)` |
| ~~`multiplication`~~ | **默认禁用**：仅当 ResearchSpec 显式放开时可用 | 须完整消融且组合优于最强单腿 |

**默认禁止 MULTIPLY 乘法交互**：本仓库默认不允许任何形式的算子相乘（含带契约的乘法），
`MULTIPLY` 会被直接拦截。需要表达放大、抑制、条件依赖或状态切换时，一律改用结构化交互：
门控 `GATED_SIGNAL`、组内排名 `CS_GROUP_RANK`、残差化 `CS_RESIDUALIZE`、背离 `DIVERGENCE_RANK`、
分段状态 `PIECEWISE_STATE` 或必要条件 `IF_THEN_ELSE`。
仅当本次 ResearchSpec 的 `interaction_policy.allowed_interaction_types` 显式包含
`"multiplication"` 时才可使用，且必须提供 base-only / condition-only / combined 完整消融，
并证明组合优于最强单腿；"两个 zscore 相乘"永远不算经济创新。"""

# ── 第三层：正交预判（探索阶段裁掉） ──
_LAYER3 = """

## 第三层：正交预判（Orthogonality Guard）

系统会在 DSL 求值前自动检查新因子与因子库中已有因子的截面 Spearman 相关性。如果与任何已有因子的相关性 > 0.7，因子会被自动拦截并返回冗余诊断。因此：
- 你应主动设计与已有因子**不同信息源**的因子，而不仅是改参数。
- 当收到"Too Redundant"预审拦截时，需改变原始变量或核心算子族，而非微调参数。

"""

# ── 全量原文（full 模式等价改造前行为） ──
RAW = _LAYER1 + _TRACK_D + _TRACK_ABC + _READ_MEMORY + _INTERACTION + _LAYER3

# ── 探索阶段精简版 ──
RAW_EXPLORE = _LAYER1 + _TRACK_D + _INTERACTION


NAME = "strategy_tracks"
TITLE = "三层枷锁与双轨策略"
ORDER = 20
REQUIRED = True
SEP_BEFORE = "\n\n"


def render(ctx) -> str:  # noqa: ANN001
    phase = getattr(ctx, "prompt_phase", "full")
    if phase == "explore":
        return RAW_EXPLORE
    return RAW

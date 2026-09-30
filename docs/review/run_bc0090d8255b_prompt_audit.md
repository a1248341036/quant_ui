# 真实 System Prompt 审计报告 — Run bc0090d8255b

- **审计对象**: run `bc0090d8255b` 实际渲染的 3 个 phase system prompt
- **渲染配置**: label_1d_open_to_open / no_fundamentals / focus_facets=["价量面"] / max_tool_calls=6 / research_spec=run 内置
- **git_revision**: `630b934`（含 C6 ICIR 绝对值口径修复）
- **prompt 产物**:
  - `_system_prompt_explore.txt`（33393 chars，turn 0 实际用）
  - `_system_prompt_deepen.txt`（36279 chars，turn 1 实际用）
  - `_system_prompt_deliver.txt`（36850 chars，turn 2 实际用）
- **审计时间**: 2026-09-21 23:25

## 一、模块装配对比（explore vs deepen）

| 模块 | explore | deepen | chars(deepen) |
|---|---|---|---|
| core_identity | ON | ON | 204 |
| strategy_tracks | ON | ON | 7266 |
| delivery_interface | ON | ON | 1992 |
| data_calibration | ON | ON | 1694 |
| data_fields | ON | ON | 637 |
| market_mechanisms | ON | ON | 3271 |
| **multi_period** | **off** | **ON** | **568** |
| operator_catalog | ON | ON | 7520 |
| neutralization_guide | ON | ON | 892 |
| tool_contracts | ON | ON | 2708 |
| ic_robustness | ON | ON | 1615 |
| delivery_submission | off | off | 0 |
| behavior_rules | ON | ON | 6217 |
| tool_examples | ON | ON | 1192 |
| population_mode | off | off | 0 |
| facet_focus | ON | ON | 444 |

**差异**：explore→deepen 仅多启用 `multi_period`（+568 chars）。deliver=deepen（全量）。

## 二、矛盾点检测

### ⚠️ M1：ICIR 比较符号不一致（stage_one `>=` vs stage_two `>`）

**位置**：`alphaagent/factor/mining/delivery/delivery_criteria.py`
- **line 339**（stage_one）：`abs(ICIR) >= {c.min_icir}`（`>=`，门槛 0.28）
- **line 349**（stage_two）：`abs(ICIR) > {p.min_train_icir}`（`>`，门槛 0.30）

**3 个 phase 全部存在此矛盾**（渲染输出一致）：
```
stage_one: abs(ICIR) >= 0.28
stage_two: abs(ICIR) > 0.3
```

**性质**：C6 修复时统一了"绝对值"（`abs(ICIR)`），但**没统一比较符号**（`>=` vs `>`）。

**实际影响**：极小——两阶段数值不同（0.28 vs 0.30），`>=` vs `>` 只在 ICIR 恰好=0.28 或 0.30 的边界值有差异。但属于口径漂移隐患，应统一。

**建议**：stage_two 改为 `abs(ICIR) >= {p.min_train_icir}`，与 stage_one 统一为 `>=`。

### ✅ 无其他矛盾

- **换手率**：3 phase 一致，硬门 `<= 0.5` + 诊断预警 `0.4`，与 spec `max_avg_daily_side_turnover: 0.5` 一致
- **prediction 必填**：3 phase 一致（缺失记账警告、累计 3 次拦截）
- **NEG() 翻转纪律**：3 phase 一致
- **盲测终审**：3 phase 一致（`abs(test IC) >= 0.01` + 保留比 50% + 方向一致）
- **正交门槛**：stage_one `< 0.5` / stage_two `< 0.4`，分层清晰无矛盾
- **Coverage**：stage_one `> 85%` / 两阶段一致

## 三、重复块检测

### 🔴 R1：交互算子清单重复（strategy_tracks.py 内 3 处 + behavior_rules.py 1 处）

**源头**：`alphaagent/factor/mining/prompt/modules/strategy_tracks.py`
- **行 65**（`_LAYER2` D 轨"禁止低级信号叠加"）：列出 7 个结构化交互算子作为替代
- **行 120**（`_INTERACTION` "触发契约要求的算子"）：列出 11 个触发契约的算子（**不同清单**，含 MULTIPLY/TS_CORR/MUTUAL_INFO_LAG 等，不算重复）
- **行 180**（`_INTERACTION` 末尾"MULTIPLY 乘法交互规则"）：再次列出 6 个结构化交互算子作为替代

**+ behavior_rules.py 行 79**：第 4 次出现"禁止裸信号叠加 + 交互算子清单"

**重复内容**（行 65 ≈ 行 180）：
```
门控 `GATED_SIGNAL`、组内排名 `CS_GROUP_RANK`、残差化 `CS_RESIDUALIZE`、
背离 `DIVERGENCE_RANK`、分段状态 `PIECEWISE_STATE`、...、必要条件 `IF_THEN_ELSE`
```

**性质**：**真实重复**。同一份"结构化交互算子清单"在 strategy_tracks 内出现 2 次（行 65 + 行 180），在 behavior_rules 又出现 1 次（行 79）。每次约 80 chars，总冗余 ~160 chars。

**建议**：
- strategy_tracks 行 180 的清单改为引用"见上方第二层 D 轨规则"或直接删除（行 65 已完整列出）
- behavior_rules 行 79 改为引用"见第二层禁止低级信号叠加规则"，不重复清单

### 🟡 R2：消融检查字段重复解释（strategy_tracks 行 57 vs behavior_rules 行 507）

**位置**：
- `strategy_tracks.py` 行 57（"条件化前先消融"规则段）：解释 `conditioning_destroyed_value`/`conditioning_flipped_signal`
- `behavior_rules.py` 行 507（"门控消融 ablation_check"诊断读法段）：再次解释同两个字段

**性质**：**语义重复**——同一概念在"行为规则"和"诊断读法"两处都解释了字段含义。但上下文不同（一个是"应该先消融"，一个是"看到字段怎么办"），属于**可接受的冗余**（强化学习效果），但可精简。

**建议**：behavior_rules 行 507 的字段解释改为"含义见第二层条件化前先消融规则"，只保留诊断读法动作。

### ✅ R3：算子签名相同（误报）

**位置**：`TS_ARGGAP`（行 355）vs `PRICE_GAP_FLOOR`（行 424）签名都是 `(open_df, high_df, low_df, close_df, min_pct=0.0)`

**性质**：**不是重复**——两个不同算子恰好参数列表相同（都接收 OHLC 四列 + min_pct）。误报。

## 四、explore 阶段裁剪验证

explore 裁剪了 `multi_period`（+568 chars），其余模块全启用。与 AGENTS.md 记录的"explore 裁剪扩至 6 模块"**不符**——实际只裁了 1 个模块。

**核查**：AGENTS.md 写"explore 阶段裁剪扩至 6 模块——multi_period/neutralization_guide/ic_robustness/delivery_submission/tool_examples/data_calibration"，但本次渲染 explore 只裁了 `multi_period`，其余 5 个（neutralization_guide/ic_robustness/tool_examples/data_calibration/delivery_submission）全部 ON。

**可能原因**：
1. `delivery_submission` 和 `population_mode` 在 deepen/deliver 也是 off（PHASES 配置不含这些阶段，或 run 配置关闭）
2. `neutralization_guide`/`ic_robustness`/`tool_examples`/`data_calibration` 的 PHASES 配置可能已改（2026-09-12 回退 tool_examples 后未同步文档）

**建议**：核查 `prompt/modules/__init__.py` 的 PHASES 配置，确认 explore 阶段实际裁剪范围与文档是否一致。

## 五、总结

| 类别 | 发现 | 严重度 |
|---|---|---|
| **矛盾** | M1: ICIR 比较符号 `>=` vs `>` 不一致 | 🟡 低（边界值影响，但属口径漂移） |
| **重复** | R1: 交互算子清单 4 处重复 | 🟡 中（~160 chars 冗余，可精简） |
| **重复** | R2: 消融字段解释 2 处 | 🟢 低（可接受冗余，强化学习） |
| **误报** | R3: 算子签名相同 | ✅ 无问题 |
| **文档** | explore 裁剪范围与 AGENTS.md 不符 | 🟡 中（文档过时或配置已改） |

### 与 v2.0 审计 spec 的对照

本次 run 的 prompt 是 `630b934` 修复后渲染，v2.0 spec 记录的 14 点矛盾中：
- **S4 换手率统一**：✅ 已修复（硬门 0.5 + 诊断 0.4，3 phase 一致）
- **S6 1d/weekly 协调**：✅ 已修复（rebalance_freq=weekly 渲染正确，submit_factor 必传 weekly）
- **C6 ICIR 绝对值**：✅ 已修复（abs 口径统一），**但残留 `>=` vs `>` 符号差异**（M1）
- **S3 正交门槛分层**：✅ 已修复（stage_one 0.5 / stage_two 0.4 分层清晰）
- **S7 负 IC 承认**：✅ 已修复（NEG() 翻转纪律 3 phase 一致）

### 建议修复（按优先级）

1. **P2-1**：`delivery_criteria.py:349` stage_two `abs(ICIR) >` → `abs(ICIR) >=`，与 stage_one 统一（M1）
2. **P2-2**：`strategy_tracks.py:180` 交互算子清单改为引用行 65，删除重复（R1）
3. **P2-3**：`behavior_rules.py:79` 交互算子清单改为引用 strategy_tracks（R1）
4. **P3-1**：核查 explore 阶段 PHASES 配置与 AGENTS.md 文档一致性

---
*审计生成: 2026-09-21 23:25 | run_id: bc0090d8255b | git: 630b934*

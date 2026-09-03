# -*- coding: utf-8 -*-
"""模块 12 · behavior_rules：行为准则（12 条硬性行为约束）。原文精确切片。"""

RAW = """### 行为准则

1. **经济直觉先行 + 预测必填**：每个 `evaluate_factor` / `eval_on_train_set` 调用前，在思维链中先写出 50 字以内经济直觉（标准见第一层，违反即跳过），并随调用传 `prediction`（缺失会记账警告、累计 3 次拦截——每次都带上，别依赖宽限）。结果中的 `prediction_check` 优先于门槛阅读：verdict=contradicted（被证伪）→ 换机制或放弃，禁止对被证伪结构做参数变异。`submit_factor` 的 `comment` 中必须包含经济直觉全文。
2. **换手红线（硬约束）**：`evaluate_factor` 结果里的 `quantile_portfolio.avg_daily_side_turnover`（日单边换手）**> 0.4 的候选不要调用 `submit_factor`**——历史数据 26/30 个候选因此止步 stage_two/engine_gate，纯浪费算力。设计期就选低换手结构：CS_ 截面排序类、长窗口平滑（TS_MEDIAN/TS_MEAN ≥20）、慢信息源（基本面 PIT、筹码周频结构）；避免逐日 rank-reversal 式信号（自相关 <0.6 的 TS_ 时序因子大概率高换手）。
3. **双轨标注**：每轮候选因子须明确标注变异类型（A 参数 / B 算子 / C 修饰 / D 新族）和父本来源（D 标注机制名）；A/B/C 必须有高质量父本，D 至少一半（配额细节见第二层）。
3b. **年度稳健性（基本面模式强制）**：基本面因子的 `comment` 必须附**分年度 IC 拆分**（train 按年、val 按年），val 任一年度方向反转或 |IC| < 0.008 的占比过高（>1/3 年度）即不要提交——整体 val IC 达标但靠单年撑起来的因子会在实盘衰减。
4. **每轮先归因上一轮结果，再设计下一代**；避免仅改窗口长度的同质批次。**同一信号族**（如短周期反转 `NEG(TS_PCTCHANGE($adj_close, N))`）在同一轮中**最多出现 1 次**；第 2 个起必须换信号族或换核心变量。
5. **并行候选必须跨越不同信息维度**：同一批 12~20 条 tool_calls 中，至少覆盖 6 个不同的信号族（批量越大，族分布越要分散，避免同族扎堆互相稀释正交性）。可选维度包括但不限于：
   - 价格动量/均值回归（`TS_MEAN($ret, N)`, `TS_PCTCHANGE`）
   - 波动率结构（`TS_STD($ret, N)`, 波动率变化/比率）
   - 量价关系（`TS_CORR($volume, $adj_close, N)`, 量价背离）
   - 流动性/换手（`$amount/$float_cap`, `TS_RANK($volume, N)`）
   - 日内结构（`$adj_open` vs `$adj_close`, `$adj_high`/`$adj_low` 范围, `$adj_vwap` 偏离）
   - 隔夜跳空（`$adj_open` vs `DELAY($adj_close, 1)`）
   - 筹码分布（`CHIP_PEAK_LOC`, `CHIP_ENTROPY`, `CHIP_COM_W_GAP`）
   - 周线结构（`$adj_close@1w` 均线偏离）
   - 截面结构（`RANK`, `CS_ZSCORE`, `CS_NEUTRALIZE` 不同分组键）
6. **连续 2 个因子 IC < 0.01 时，强制切换到完全未尝试过的信号族**，不要在同一信号族上微调。
7. 发起 tool_calls 前完成思考；**不要**停在解释或征询用户下一步。
8. **train+val 统计达标的因子，必须调用 `submit_factor`** 提交候选池；`comment` 须清晰描述因子逻辑和经济直觉，勿空泛。Reviewer 在 validation 阶段的意见仅供参考，不要因此放弃提交。
9. **结束前检查**：若已有 train+val 达标的候选但尚未 `submit_factor`，不得结束；先提交再收尾。
10. **避免过度调参**：除非本轮已产出多个**两两截面相关较低**（机制差异明显）的保留级候选，通常 **提交一个因子后即可结束**，无需对同一机制反复微调窗口或参数。
11. **工具返回的是精简结构化文本**（非完整 JSON），包含 IC/ICIR/诊断建议/批次汇总/同质化警告。请仔细阅读诊断建议和同质化警告，据此调整下一轮方向。
12. **禁止裸信号叠加**：定义与例外见第二层轨道 D 的硬约束——多信息源融合必须走结构化交互算子。
"""

NAME = "behavior_rules"
TITLE = "行为准则"
ORDER = 120
REQUIRED = True
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    return RAW

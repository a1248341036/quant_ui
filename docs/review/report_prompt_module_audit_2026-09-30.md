# 研报模式 Prompt 模块审计（四模式 × 逐模块字符数）

- 日期：**2026-09-30**
- 分支：`feat/report-prompt-plugins`（从 `feat/report-mode` b53f3c5 拉出，独立 worktree）
- 方法：`assemble_system_prompt(DEFAULT_MODULES, PromptContext(...))` 渲染四场景，取逐模块字符数（只读，未改代码）
- 场景：technical（label_1d、无基本面）／fundamental（label_20d、含基本面）／report 复现轮（`report_phase=reproduce`）／report 发散轮（`report_phase=diverge`）

## 1. 实测矩阵（字符数）

| 模块 | technical | fundamental | report 复现 | report 发散 | 归类 |
|---|---:|---:|---:|---:|---|
| operator_catalog | 9753 | 9753 | 9753 | 9753 | 通用（**最大块**） |
| data_fields | 7487 | 9082 | 9082 | 9082 | 通用（内容需按字段族裁剪） |
| **strategy_tracks** | 7792 | 7792 | **7792** | 7792 | **探索类 → 复现轮应关** |
| behavior_rules | 6612 | 6610 | 6612 | 6612 | 通用（复现轮需追加专章） |
| tool_examples | 2673 | 3346 | 3346 | 3346 | 通用（复现轮可精简） |
| **market_mechanisms** | 3239 | 3239 | **3239** | 3239 | **探索类 → 复现轮应关** |
| tool_contracts | 2708 | 2708 | 2708 | 2708 | 通用 |
| delivery_interface | 2046 | 2047 | 2046 | 2046 | 通用 |
| data_calibration | 1682 | 1849 | 1735 | 1735 | 通用 |
| ic_robustness | 1615 | 1615 | 1615 | 1615 | 通用 |
| **report_prior** | 0 | 0 | **1384** | **0** | 复现专属（阶段门禁生效 ✓） |
| report_rag | 1204 | 1204 | 0 | 1204 | **异常：非 report 模式也有 1204** |
| multi_period | 1113 | 1113 | 1113 | 1113 | 通用 |
| neutralization_guide | 902 | 902 | 902 | 902 | 通用 |
| turnover_budget | 714 | 647 | 714 | 714 | 通用（与频率对齐强相关） |
| delivery_submission | 565 | 565 | 565 | 565 | 通用 |
| core_identity | 204 | 204 | 204 | 204 | 通用 |
| **report_mechanisms** | 0 | 0 | **0** | 0 | **异常：复现轮未启用**（机制卡实际走每轮题面注入） |
| facet_focus / model_adaptation / population_mode / extra_instructions | 0 | 0 | 0 | 0 | 当前未启用（条件未满足） |
| **总计** | **50381** | **52748** | **52882** | **52702** | — |

## 2. 结论

1. **report 档没有任何裁剪**：复现轮 52,882 字符 vs technical 50,381（仅多出 `report_prior` 1,384 + 基本面片段），**探索类模块照旧全上**。
2. **探索类 ≈ 11,031 字符（占复现轮 21%）**：`strategy_tracks` 7,792 + `market_mechanisms` 3,239 → 与"忠实复现研报"目标相斥，是"脱稿自由发挥"的素材来源（对应观测：整夜 promising 因子多无 `rq0xx_` 前缀）。
3. **两个异常需修**：
   - `report_mechanisms` 在 report 复现轮渲染 **0 字符** → 阶段化机制卡模块**未接线**（机制卡目前只经 `render_reproduce_task` 注入题面，系统 prompt 里的通道是空的）；
   - `report_rag` 在 **technical/fundamental 也注入 1,204 字符** → 未按模式门禁（要么改名/要么补 `report_flow_enabled` 条件）。
4. `operator_catalog` 9,753 + `data_fields` 9,082 = **18,835 字符（36%）**，是最大的两块，但都是"必需但可裁剪"：算子目录可裁成"本次允许的子集"，字段目录可用既有 `field_family_scope` 白名单按研报字段族裁剪。

## 3. 建议的模式开关矩阵（待确认）

| 模块 | technical | fundamental | report 复现 | report 发散 |
|---|---|---|---|---|
| core_identity / delivery_* / tool_contracts / ic_robustness / data_calibration / turnover_budget / multi_period / neutralization_guide / population_mode / model_adaptation | ✅ | ✅ | ✅ | ✅ |
| operator_catalog | ✅ | ✅ | ✅（可裁子集） | ✅ |
| data_fields | ✅ | ✅ | ✅（`field_family_scope` 裁剪） | ✅ |
| tool_examples | ✅ | ✅ | ⚠️ 精简为复现相关 | ✅ |
| behavior_rules | ✅ | ✅ | ✅ **+ report 行为专章**（禁止改机制语义 / 必须 `reproduce_of` / 单维约束 / 频率标签对齐） | ✅ |
| **strategy_tracks** | ✅ | ✅ | ❌ **关** | ⚠️ 只留 6 维变异指引 |
| **market_mechanisms** | ✅ | ✅ | ❌ **关** | ⚠️ 弱化 |
| facet_focus | 按需 | 按需 | ❌ 关 | 按需 |
| **report_mechanisms** | — | — | ✅ **需修复接线** | ❌ |
| **report_prior** | — | — | ✅ | ✅（结论/频率依据） |
| **report_rag** | ❌（应关） | ❌（应关） | ❌ | ✅ |

## 4. 落地机制（不改框架）

- 在**模式注册表**（`core/research_modes.ResearchModeSpec`，单一真源）新增：
  - `prompt_module_excludes: tuple[str, ...]`（该模式关闭的模块）
  - `prompt_module_phase: dict[str, str]`（阶段化开关，如 `report_rag: diverge`、`report_mechanisms: reproduce`）
- `prompt/modules/__init__.py::_phase_enabled(module, base_enabled)` 读上述两字段 → **模式开关集中一处**，technical/fundamental 行为不变；
- 内容级裁剪复用既有 `prompt/scope_filter.py`（`field_family_scope`）。

## 5. 预期收益（按实测字符数）

| 场景 | 现状 | 裁剪后（估） | 省 |
|---|---:|---:|---:|
| report 复现轮 | 52,882 | ≈ 40,500（关 strategy_tracks+market_mechanisms，裁字段/算子目录） | **−23%** |
| report 发散轮 | 52,702 | ≈ 46,000（弱化探索类，保留 6 维指引） | −13% |

> 说明：以上为**只读审计**结论；开关最终由用户圈定后再落地代码。

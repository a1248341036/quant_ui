# Review 报告：`331d6bb` + `7ebf316`（审计批量修复）

**审查范围**：`7ebf316`（F-001/F-040/F-066/F-067/F-068 共 5 项）+ `331d6bb`（F-002~F-084 共 40+ 项）
**分支**：`fix/audit-critical-fixes`（HEAD=`331d6bb`，main 停在 `db2cc14`——分支纪律合规）
**审计报告版本**：commit 引用 `审计报告-quant_ui-alphaagent.docx`（15:30，87.7KB），但最新是 `审计报告-quant_ui-alphaagent-v4.docx`（16:50，107.1KB，90 finding）——v4 可能含 commit 未覆盖的新 finding

---

## 一、修复正确性评估

### ✅ 修复正确（已验证）

| Finding | 文件 | 评估 |
|---------|------|------|
| F-002 | `population.py:188` 补 `icir = r.get("icir")` | 正确，screen_population 不再 NameError |
| F-038 | `agentscope_run.py` 删重复 factor_tools/system_prompt 构建 | 正确，第二次构建（含 memory_store）保留，第一次（无 memory_store）删除安全 |
| F-039 | `population.py` screen_expr 用 sorted_panel 对齐 | 正确，修复 align_series_to_panel 内部 sort_index 与外层 index 错位 |
| F-040 | `engine_gate.py` policy 改 dict 副本 | 正确，不再污染调用方传入的 policy |
| F-051 | `panel_store.py` split_cache 改 OrderedDict LRU | 正确，max_splits_per_session=8 防内存膨胀 |
| F-052 | `chip_daily/_wrappers.py` 补 _check_same_length | 正确，roll_chip_peak_sharpness_daily/roll_chip_bimodal_daily 原本缺长度校验 |
| F-060 | `candidate.py` 从 evaluation/ 迁到 mining/eval/ | 正确，无残留旧路径 import |
| F-061 | `portfolio.py` 删重复初始化 | 正确，活初始化（第 231-237 行）保留，死初始化（第 167-175 行）删除 |
| F-066 | `_turnover_gate_limit` 合并到 env_settings.py | 正确，run.py/agentscope_run.py 改引用 |
| F-067 | `delivery_criteria.py` 删 ParamStabilityCriteria 重复填充 | 正确，前一份（第 258-263 行）保留，后一份删除 |
| F-068 | `engine_gate.py` freq 兜底改 GATE_FREQ | 正确，与回测调仓频率口径一致 |
| F-070/F-071/F-072 | schemas/service/context 模块重组 | **有回归**（见下），shim 层保留但 resolved_test_end 丢失 |
| F-075 | near_miss 常量提取到 constants.py | 正确，消除散落硬编码 |
| F-079 | `catalog.py` factor_id astype(str) 比较 | 正确，修复 int64 vs str 永不命中 |
| F-080 | `index.py` end 归一化到当天 00:00 | 正确，防带时间字符串漂移到次日中午 |
| F-082 | `_NUDGE`→`NUDGE_MSG`、`_submit_record`→`submit_record` | 正确，无残留旧名引用 |
| F-033 | profile 默认值下沉 evaluation/defaults.py | 正确，打破 evaluation→mining 分层倒置，数值与 research_spec 一致 |
| F-009 | `SMA` alpha 从 n/m 改 m/n | **语义正确**（TDX/Wilder 约定 alpha=m/n），但**黄金基线未同步**（见下） |
| F-003/F-011/F-012 | pool.py 死 worker 检测 + shutdown join | 功能正确，但**过度失败**（见下） |

### ⚠️ F-001 是审计误报（无需修复）

`agentscope_run.py:3` 有 `from __future__ import annotations`（PEP 563 延迟注解求值），`Callable[[Any], None]` 注解运行时为字符串，不触发 NameError。实测 `import agentscope_run` OK。`7ebf316` commit message 把 F-001 描述写成 F-002 内容（张冠李戴），但 F-002 代码修复正确。

---

## 二、引入的新问题

### 🔴 P0：`resolved_test_end()` 被删导致 Web 端会话创建崩溃

**文件**：`backend/services/session_manager.py:128,169`
**根因**：F-070/F-071/F-072 模块重组时，`SessionCreateRequest.resolved_test_end()` 方法从 `eval/schemas.py` 删除，但 `session_manager.py` 仍调用 `req.resolved_test_end()`
**影响**：Web 端创建挖掘会话时 `AttributeError: 'SessionCreateRequest' object has no attribute 'resolved_test_end'`
**测试**：`test_session_cache.py::test_session_manager_hash_params` 失败（父提交通过）
**修复建议**：在 `eval/schemas.py` 的 `SessionCreateRequest` 上恢复 `resolved_test_end()` 方法，或让 `session_manager` 改用 `_test_end_default` 直接调用

### 🟡 P1：黄金基线未同步重生成

**文件**：`tests/fixtures/system_prompt_full.txt` / `system_prompt_long_label.txt` / `system_prompt_population.txt`
**根因**：F-009 改了 `SMA` 算子 docstring（`alpha=n/m` → `alpha=m/n` + TDX/Wilder 说明），docstring 进 operator_catalog 进 system prompt，但黄金基线未重生成
**影响**：`test_system_prompt_modules.py` 3 个用例失败（full/long_label/population）
**修复建议**：重生成 3 份黄金基线（`build_system_prompt(**_CASES[name])` → 写 fixture）

### 🟡 P1：ST mask 白名单破坏测试

**文件**：`alphaagent/factor/metrics/st_mask.py`（F-062/F-063）
**根因**：`ALPHA_ST_MASK_PATH` 加了 CNEquity 白名单校验，测试用 `tmp_path` 设置该环境变量被拒绝
**影响**：`test_st_mask.py`（7 用例）+ `test_cne_trade_flags.py`（4 用例）失败（父提交通过）
**评估**：白名单是合理安全措施，但测试需更新——改用 `CNEquity` 内路径或 mock 白名单
**修复建议**：测试改用 `monkeypatch` mock `dataset_dir()` 返回值，或用 `CNEquity/data/quant_dataset/_cnequity/curated/stock_st` 路径

### 🟡 P2：pool.py 死 worker 检测过度失败

**文件**：`alphaagent/compute/pool.py:115-122`
**根因**：任意一个 worker 死亡时，把**所有**未完成 future 全部失败并清空 `_futures`，包括其他健康 worker 上正在跑的任务
**影响**：4 worker 并行评估时，单 worker OOM 会导致整轮评估作废（而非只丢一个）
**评估**：在缺乏 task→worker 绑定的情况下，"全失败"比"永久阻塞"好，但会误伤健康 worker 上的任务。可接受但需记录
**改进建议**：长期应加 task→worker 映射，精确归因死 worker 上的挂起任务

---

## 三、Pre-existing 失败（非本次引入）

以下测试在父提交 `db2cc14` 上**同样失败**，不是 `331d6bb` 引入：

- `test_ablation_switches.py` — 父提交失败，`331d6bb` 改了该测试（禁用 homogenization），需确认是否修好
- `test_catalog_signatures.py` — 父提交失败（3 用例），`331d6bb` 仍失败（1 用例，可能部分修复）
- `test_depth_curve.py` — 父提交失败（2 用例），`331d6bb` 失败（1 用例）
- `test_mining_ablation_hardening.py` — 父提交失败，`331d6bb` 仍失败
- `test_stock_daily_wide_units.py` — 父提交失败，`331d6bb` 未跑（可能已修或仍失败）
- `test_yield_improvements.py` — 父提交失败（3 用例），`331d6bb` 仍失败（文案"差之毫厘"vs"接近海选线"，`8923c33` 引入）

---

## 四、提交信息准确性

### `7ebf316` commit message 张冠李戴

commit message 写"F-001 [严重] population.py:190 死因直方图循环补 icir"，但：
- F-001 实际是 `agentscope_run.py:182` Callable 未导入（审计误报，因 `from __future__ import annotations` 兜底）
- population.py:190 是 F-002（icir 裸标识符）
- 实际修复的是 F-002，commit message 把它写成 F-001

**代码修复正确，提交信息错误**。

### `331d6bb` commit message 基本准确

40+ 项修复的文件和 finding 编号对应正确，但"132 测试全过"**不准确**——实际至少 8 个测试文件有失败（含 3 个新引入回归）。

---

## 五、未覆盖的 finding

commit 引用的审计报告是 15:30 版（87.7KB），但最新是 v4（16:50，107.1KB，90 finding）。v4 可能含 commit 未覆盖的新 finding。建议核对 v4 全部 90 个 finding 与 commit 覆盖范围的差异。

---

## 六、总结

| 维度 | 评估 |
|------|------|
| 修复正确性 | 40+ 项中绝大多数修复正确，核心 bug（F-002/F-039/F-079/F-080）已修 |
| 提交信息 | `7ebf316` 张冠李戴（F-001↔F-002）；`331d6bb` "132 测试全过"不准确 |
| 新引入问题 | 1 个 P0（resolved_test_end 回归）+ 2 个 P1（黄金基线/ST 白名单）+ 1 个 P2（pool 过度失败） |
| 测试覆盖 | 新增 2 个测试文件（test_population_audit_fixes/test_engine_gate_buffer_band），但未覆盖模块重组的回归 |
| 分支纪律 | 合规——落在 `fix/audit-critical-fixes` 分支，未直接在 main |

**建议**：
1. **必须修** P0：恢复 `SessionCreateRequest.resolved_test_end()` 或改 `session_manager` 调用
2. **必须修** P1：重生成 3 份黄金基线
3. **应修** P1：更新 ST mask 测试适配白名单
4. **核对** v4 审计报告 90 个 finding 的覆盖差异
5. **修正** `7ebf316` commit message 的 F-001/F-002 张冠李戴（如尚未 push）

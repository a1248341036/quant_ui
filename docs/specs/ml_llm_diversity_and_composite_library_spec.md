# ML 组合多样化探索与组合因子库自动沉淀方案规范（修订版）

> 文档编号：SPEC-ML-COMPOSITE-INGEST-20260914-REV2  
> 状态：PROPOSED / REVIEWED  
> 责任模块：`alphaagent/factor/stacking/`、`backend/composite_factor_service.py`、`scripts/train_ml_composite.py`、`static/src/components/alphaagent/`  
> 核心目标：**解决 LLM 推荐同质化** + **打通组合因子库（Composite Factors）自动归档闭环** + **兼顾历史锁定复用与确定性精准复现** + **对齐盲测安全规范**

---

## 1. 现状诊断与历史决策核查

### 1.1 为什么当前每次推荐出的因子完全相同？
- **历史设计背景（2026-09-13）**：  
  在初期实现中，为满足“盲测实验可复现”与“避免重复调用 LLM 产生高额 token 消耗”的诉求，引入了全局单点锁定文件：  
  `RECOMMENDATION_LOCK_FILE = artifacts/alphaagent/stacking/llm_recommendation.json`  
  当候选池未发生增删时（`pool_fp` 指纹一致），系统判定 `reused = True`，**直接复用该文件，不重新调用大模型**。
- **与业务初衷的冲突**：  
  用户引入 LLM 的核心初衷是作为**“智能组合策略师”**，希望在已有因子池上**主动尝试不同维度的组合搭配**。然而全局单点锁定导致只要因子库未变，后续无论发起了多少次不同目的的组合训练，选中的因子名单 **100% 被静态死锁**。
- **低采样随机性**：  
  `llm_assist.py` 中 `temperature=0.2` 硬编码，即使在指纹变更重调时，贪心采样依然倾向于收敛在同一批头部明星因子上。

### 1.2 组合因子库（Composite Factors）当前断路现状
- `scripts/train_ml_composite.py` 执行完 Walk-Forward 拟合与 Gate 裁决后，仅在运行输出目录（`out_dir`）落盘临时的 `report.json` 和 `scores.parquet`，**未串联任何持久化入库逻辑**；
- 唯一的入库途径是用户在前端历史表格的深层展开面板中，手动点击“另存为组合因子”，调用 `backend.composite_factor_service.save_composite_factor`。若无人手动另存，组合因子库永远无法自动沉淀。

### 1.3 关键现存 Bug 核查：`_repro_command` 硬编码 `--modes technical`
- 在 `backend/composite_factor_service.py` 第 81 行中：
  ```python
  parts = [
      ".venv\\Scripts\\python.exe scripts\\train_ml_composite.py",
      "--modes technical",  # <--- 硬编码！
      f"--scheme {report.get('scheme') or 'ml'}",
  ```
  如果一个组合中包含了基本面或跨数据源融合因子，复现命令会遗漏 `--modes fundamental`，导致复现时底层候选特征减少，**打破“100% 确定性精准复现”的承诺**。此 Bug 必须作为本方案的 P0 前置项修复。

---

## 2. 核心架构设计与关键权衡（Trade-offs）

方案在**“自由探索”**、**“成本控制”**、**“精准可复现”**与**“盲测安全”**之间建立清晰的解耦边界：

```
                           ┌────────────────────────────┐
                           │      启动 ML 组合任务      │
                           │  (检查 guidance/theme 输入) │
                           └──────────────┬─────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      【无参数：锁定复用模式】                           【有引导：动态探索模式】
      (保持现状，默认行为不变)                       (显式传入 guidance / theme)
      • 读全局池指纹锁定文件                           • 绕过全局锁，独立调用 LLM
      • 不重调 LLM，零额外 API 消耗                   • Temperature 0.65，激发多样性搭配
      • 推荐落盘至 out_dir 专属目录                   • 产出 LLM 组合命名与投资假设
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                      【执行端：Walk-Forward 拟合 + 门禁裁决】
                      • 强制物理截断至 2024-12-31 (eval_mode=tuning)
                      • 运行 engine_gate 实盘周调仓回测
                                          │
                                          ▼
                      【输出端：组合因子库品质自动沉淀 (Auto-Ingest)】
                      • 仅在 gate_passed == True 时触发（品质入库）
                      • 基于 (scheme, include_factors) 指纹防重复堆砌
                      • 强制打上 eval_mode="tuning"（研发态资产标签）
                      • 动态生成正确的 --modes 显式白名单复现命令
```

---

## 3. 详细修订与执行规范（按优先级）

### 3.1 [P0] 前置修复：`_repro_command` 动态模式生成
- **改动文件**：`backend/composite_factor_service.py`
- **规则**：
  从 `report.get("modes")` 或 `params.get("modes")` 动态提取训练模式列表：
  ```python
  modes = report.get("modes") or params.get("modes") or ["technical"]
  if isinstance(modes, (list, tuple)):
      parts.append("--modes " + " ".join(str(m) for m in modes))
  else:
      parts.append(f"--modes {modes}")
  ```
  确保复现命令与原始运行的因子库环境完全一致。

### 3.2 [P0] 盲测安全规范：入库资产属性与口径声明
- **背景**：自动沉淀的组合条目来自 `tuning` 研发模式（数据物理截断至 `2024-12-31`），其 `scores.parquet` 尚未覆盖 2025+ 盲测段。
- **规则**：
  1. 入库条目的 `provenance_json` 必须显式持久化：
     `"eval_mode": "tuning"`, `"panel_end": report.get("panel_end", "2024-12-31")`；
  2. 条目 `note` 声明：“研发验证态组合（覆盖至 2024-12-31）· 盲测段安全锁定未曝光”；
  3. 严禁 `blind_test` 模式的运行触发自动入库，保持离线审计定位。

### 3.3 [P1] 探索模式参数化解耦（平衡成本与多样性）
- **触发机制**：
  - **静态复用（默认兼容）**：若未传入 `--guidance` / `--theme`，维持现有行为，只要候选池未变，复用 `RECOMMENDATION_LOCK_FILE`，避免非预期的 LLM token 浪费；
  - **动态探索（主动触发）**：当用户传入了 `--guidance`（自定义引导词）或显式指定了探索风格时，判定为**探索模式**：
    - 绕过全局锁定文件；
    - 使用 `temperature=0.65`；
    - 推荐结果单独保存在本次任务的目录中：`<out_dir>/llm_recommendation.json`。
- **主题落地节奏**：
  - **Phase 1**：优先支持 **`custom`（用户自由一句话引导）**，例如：“排除所有均线动量，重点配置筹码分布与资金流异常因子”；
  - **Phase 2**：待候选池规模增长至 50+ 后，再扩充预设固定主题。

### 3.4 [P1] 自动入库（Auto-Ingest）与指纹去重机制
- **触发条件（品质入库）**：
  新增运行参数 `--auto-ingest`，默认 `gate_pass`（只有 `engine_gate.passed == True` 时才自动入库；未通过的保留在训练历史供诊断，绝不污染正式组合库）。可选 `always` / `none`。
- **条目防膨胀去重（Deduplication）**：
  - 自动入库前，计算组合指纹：  
    `combo_fp = sha256(scheme + "|" + ",".join(sorted(include_factors)) + "|" + mining_end)[:16]`
  - 在 `composite_factors` 维表检查是否存在相同 `combo_fp` 的条目：
    - 若已存在同指纹组合，且新运行的 OOS IC / Sharpe 更优，则**更新覆盖**该条目元数据并刷新分数文件；
    - 若新运行表现未跑赢库内存量条目，则跳过入库并打印提示，杜绝同质记录泛滥。

---

## 4. 前端交互与资产迁移设计

1. **新建任务弹窗增强 (`static/src/components/alphaagent/MlPanel.vue`)**：
   - 增加「探索引导（Prompt Guidance）」可选输入框（提示：“告诉 LLM 本次想尝试什么风格的因子组合，留空则沿用标准复用”）；
   - 增加「自动入库」复选框（默认开启：☑️ 达到 Gate 门禁标准后自动归档至组合因子库）。
2. **组合因子库抽屉展示**：
   - 区分显示条目的资产阶段：绿标 `[研发态 · 2024]` 或 `[终审盲测]`；
   - 任务运行完成且触发入库后，抽屉列表实时拉取更新，无需手动刷新。

---

## 5. 验收门禁标准

1. **复现命令有效性测试**：
   - 测试用例：在包含 `fundamental` 模式的环境下生成组合条目，断言 `repro_command` 包含 `--modes` 完整参数列表，重跑输出特征数完全一致。
2. **探索模式参数化测试**：
   - 未传 guidance 时，断言命中复用逻辑（`reused=True`）；
   - 传入 guidance 时，断言绕过全局缓存独立调用，且落盘在 `out_dir`。
3. **品质入库与去重测试**：
   - 模拟 Gate 通过的运行，断言 `composite_factors.db` 成功新增记录且包含 `eval_mode="tuning"`；
   - 连续两次运行完全相同的组合，断言数据库记录数不重复增加。

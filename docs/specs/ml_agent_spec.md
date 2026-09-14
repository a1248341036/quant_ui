# ML 组合智能体（ML Agent）与盲测数据防泄漏规范

> 文档编号：SPEC-ML-AGENT-20260914  
> 状态：PARTIALLY IMPLEMENTED / IN PROGRESS  
> - **§1 ~ §2 盲测数据防泄漏与物理隔离架构**：**已实装并合并至主干（Production / Commit e78da3c, c926caa）**  
> - **§3 ~ §5 流式交互架构、SSE 推流设计与前端渐进迁移**：**规划与设计中（Proposed / RFC）**  
> 责任模块：`alphaagent/factor/stacking/`、`backend/stacking_service.py`、`backend/routers/alphaagent.py`、`static/src/components/alphaagent/`  
> 核心关切：**盲测数据泄漏（Data Snooping）阻断** + **流式 Agent 体验落地路径**

---

## 1. 现状审计：盲测段数据泄漏风险分析【已闭环审计】

### 1.1 两个子系统的口径冲突
在 2026-09-14 改造前，AlphaAgent 单因子挖掘与 ML 组合存在严重的时间窗口割裂：

| 模块 | 训练集（Train） | 验证集（Val） | 盲测锁定段（Holdout Test） | 原始 LLM 能见度 |
|---|---|---|---|---|
| **单因子自主挖掘（AlphaAgent）** | `2020-01-01` ~ `2022-12-31` | `2023-01-01` ~ `2024-12-31` | **`2025-01-01` ~ 至今** | 严格封死在 2024 年底前。盲测段绝对锁定，LLM 完全不可见。 |
| **旧版 ML 组合（train_ml_composite.py）** | 历史至 `mining_end`（2024-12-31） | **无**（直接跳过） | **`2025-01-01` ~ 至今**（被拿来做 Walk-Forward 评估） | **严重隐患**：OOS 折表现与 gate 回测全部算自 2025+，且直接喂给 LLM 说明书。 |

### 1.2 泄漏机理与危害
1. **Walk-Forward 评估区间强行落入盲测期**：  
   旧版第一折 OOS 强制从 `mining_end`（2024-12-31）起步，导致全部折的 OOS IC、ICIR、超额夏普、最大回撤、以及 `engine_gate` 可交易性回测，**全部计算于 2025-01-01 至今的行情数据**。
2. **多重检验烧毁盲测集（Data Snooping Bias）**：  
   生成的 `report.json` 包含 2025+ 的绩效数字。如果用户和 LLM 根据这些结果在前端反复调整因子组合、修改模型、重试训练，**2025+ 的真正盲测段就被当成了验证集（Validation Set）反复调参消耗，导致最终样本外失去诚实性**。
3. **入库时间戳潜在泄漏盲点（P1 修复）**：  
   原 `_compress_entries` 直接将因子元数据中的 `created_at`（如 `2026-09-03`）直接暴露给 Prompt A，使得 LLM 可以根据时间推断哪些因子是在盲测期被挖掘出来的。

---

## 2. 防烧盲测的三层物理隔离架构【生产规范·已实装】

已在 `fix/ml-blind-test-isolation` 分支完成实装，并通过 `tests/test_ml_blind_test_isolation.py` 自动化测试。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段一：组合研发验证态（eval-mode = tuning，默认且唯一交互态）              │
│                                                                             │
│  [2020 ~ 2022 组合训练段] ──► [2023 ~ 2024 组合验证段 (OOS)]                 │
│                                           │                                 │
│                                    右端死线：2024-12-31                      │
│                                           │                                 │
│  ★ 数据加载物理截断：load_panel_from_cne 右端强制封顶在 2024-12-31，           │
│    2025+ 行情与标签数据根本不加载入内存。                                    │
│  ★ 挖掘边界自动对齐：mining_end=auto 时对齐至 2022-12-31，留出 24 个月安全 OOS。 │
│  ★ LLM 绝对安全：Prompt A 对 created_at 做盲测脱敏；Prompt C 仅解读验证段指标。 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 最终定稿，冻结因子组合与权重
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段二：终审盲测定稿态（eval-mode = blind_test，离线归档态）                 │
│                                                                             │
│  [2025-01-01 ~ 最新交易日]                                                 │
│                                                                             │
│  ★ 纯 Python 引擎确定性计算，只用于定稿后一次性出具审计报告。                 │
│  ★ 安全拦截门禁：--eval-mode blind_test 下绝对禁止开启 --llm-assist（违规退出） │
│  ★ LLM 接口拒识：llm_summarize_report 遇盲测报告直接拒绝生成解读（返回 None）。  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 规则与代码对照表
- **右端截断**：`scripts/train_ml_composite.py` 在 `tuning` 模式下强制 `end = min(end, DEFAULT_VAL_END)`。
- **模式互斥门禁**：`args.eval_mode == "blind_test" and args.llm_assist` 触发 `[safety-violation]` 并退出。
- **报告拦截门禁**：`alphaagent/factor/stacking/llm_assist.py` 中 `llm_summarize_report` 检测到 `blind_test` 自动拦截。
- **入库时间脱敏（P1 修复）**：`_compress_entries` 通过 `_sanitize_vintage` 将包含 `2025/2026` 的日期统一转换为安全相对标识 `post-mining (recent)`，消除年份泄漏。
- **前端安全徽章**：`MlPanel.vue` 在配置表单和展开报告头部固定展示 `🛡️ 盲测隔离（≤2024-12-31）`。

---

## 3. ML 组合流式交互架构设计【规划中 / RFC】

### 3.1 现实约束与架构论证（解决 stacking 与 SSE 冲突）

原 SPEC 提出“1 秒内建立 SSE 并渲染 Thinking 打字机”，在现实工程中存在**物理冲突**：
- **现实物理瓶颈**：ML 组合训练启动后需调用 `load_panel_from_cne` 加载 800 万行 Wide 表，即便命中磁盘缓存也需 15~40 秒（冷加载 60~100 秒）；紧接着物化因子矩阵需要 10~30 秒。在此期间，子进程无拟合事件产生。
- **进程间通信（IPC）现实选择**：
  - **方案 A（放弃，成本过高）**：改写 `train_ml_composite.py` 为异步框架，建立 Socket/Queue 双向 IPC，会破坏其作为 CLI 工具的独立可调试性。
  - **方案 B（推荐，轻量日志事件流 Log-tailed SSE）**：
    - `train_ml_composite.py` 在 stdout 打印时，关键节点输出单行 JSON 标记：  
      `[ML_EVENT] {"event": "ml_fold_progress", "fold": 1, "ic": 0.038, ...}`
    - 后端 `stacking_service.py` 维持原有的 `subprocess.Popen`，同时挂载一个轻量级文件流 Tailer，将日志中的 `[ML_EVENT]` 解析并向 SSE 通道 `/api/alphaagent/ml-runs/{id}/events` 广播。
    - **收益**：
      1. 完全兼容离线 CLI 运行（无后端的控制台下依然正常打印人类可读日志）。
      2. 具备天然持久化回放能力：任何时候刷新页面，后端根据已有日志即可重构完整的 timeline 事件，无需额外维护会话内存状态。

### 3.2 阶段时间线预期与体验设计

```
T0: POST /api/alphaagent/ml-runs 启动
 │
 ├─► T+0.5s: SSE 连接建立，推送 session_start，右侧渲染【任务初始化卡片】
 │
 ├─► T+1s ~ T+30s: 子进程加载 CNE Panel，推流 ml_stage{"stage": "panel_loading"}
 │   └── 渲染：Panel 正在加载并校验 2024-12-31 盲测截断线（加载动画，避免白屏焦虑）
 │
 ├─► T+30s: 因子物化完成，调用 LLM 语义推荐，推流 agent_thinking
 │   └── 渲染：Thinking 思考气泡展开，展示对当前统一大库各数据面的互补性推导
 │
 ├─► T+35s: 推荐完成，推流 ml_pool_screened
 │   └── 渲染：【因子筛选卡片】展示 10 个入选因子、剔除列表与推荐理由
 │
 ├─► T+40s ~ T+100s: Walk-Forward 逐折滚动拟合（逐折推流 ml_fold_done）
 │   ├── 折 1 (2023H1) 完成 ──► 动态绘制折 1 柱子与 Top-5 权重
 │   ├── 折 2 (2023H2) 完成 ──► 柱状图追加折 2，更新累积 OOS IC
 │   └── 折 3 (2024H1) 完成 ──► 柱状图追加折 3
 │
 ├─► T+105s: engine_gate 可交易性回测完成，推流 ml_gate_evaluated
 │   └── 渲染：【Gate 门禁卡片】超额年化、夏普、换手率、通过/未过判定
 │
 └─► T+115s: 组合说明书生成完成，推流 ml_summary_generated & session_end
     └── 渲染：【说明书卡片】总结、优势、风险、改进建议；激活底部交互输入框
```

### 3.3 SSE 事件契约规范

```typescript
interface MlEventEnvelope {
  ts: string;
  event: string;
  run_id: string;
  payload: Record<string, any>;
}
```

- `session_start`：`{run_id, eval_mode: "tuning", isolation: "holdout", params}`
- `ml_stage`：`{stage: "panel_loading" | "factor_materializing" | "evaluating"}`
- `agent_thinking`：`{content: "正在分析因子的数据面覆盖..."}`（用于流式思考展示）
- `ml_pool_screened`：`{selected: string[], dropped: {name, reason}[], rationale: string}`
- `ml_fold_done`：`{fold: number, total_folds: number, oos_range: string, ic: number, ir: number, weights: {name, weight}[]}`
- `ml_gate_evaluated`：`{passed: boolean, metrics: {excess_annual, excess_sharpe, max_drawdown, turnover}}`
- `ml_summary_generated`：`{summary, strengths, risks, suggestions}`
- `session_end`：`{status: "completed" | "failed", exit_code: number}`

---

## 4. 前端渐进式重构与资产迁移方案【规划中】

### 4.1 迁移原则：拒绝推倒重来，渐进式组件化
现有 `MlPanel.vue` 已承载大量高质量业务功能（组合因子库管理、多路径对照、累积子集曲线、特征置换重要性）。前端改造遵循**主视窗流式化 + 资产归因沉淀入抽屉**的原则：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ AlphaAgent.vue ──► 子标签 [ML 组合]                                              │
├───────────────────────────────┬─────────────────────────────────────────────────┤
│ 左侧：MlRunSidebar.vue        │ 右侧：MlAgentView.vue                           │
│                               │                                                 │
│ ＋ 新建组合运行 (Modal)       │ 顶部状态栏：[状态徽标] [🛡️盲测隔离] [另存为组合]  │
│ ───────────────────────────── │ ─────────────────────────────────────────────── │
│ 🔘 运行中：phase3_tuning      │ 主视窗：MlAgentThread.vue (流式会话时间线)        │
│    └─ 折 2/4 拟合中           │  ├─ Thinking 思考气泡 (可展开/折叠)             │
│ ⚪ 历史 1：20260914-1130      │  ├─ 阶段状态卡片 (Panel 加载 / 因子筛选画像)    │
│    └─ OOS IC 0.041 (已过)     │  ├─ 实时 Walk-Forward 折叠图表卡片              │
│ ⚪ 历史 2：20260914-0915      │  └─ 组合说明书 & Gate 诊断卡片                  │
│    └─ OOS IC 0.025 (未过)     │ ─────────────────────────────────────────────── │
│ ───────────────────────────── │ 底部：MlComposer.vue                            │
│ 📁 抽屉入口：[组合因子中台库] │  └─ [输入调参指令，如："调高相关性去重阈值"]    │
│ 📊 归因入口：[深度归因分析]   │                                                 │
└───────────────────────────────┴─────────────────────────────────────────────────┘
```

### 4.2 现有资产映射表
1. **组合因子库（Composite Factors）**：
   - 现有的 `compositeList`、`compositeDetail` 和“另存为组合因子”链路，下沉为顶部操作栏或左下角的【组合因子中台库】抽屉，点击从右侧滑出，数据与现有 `/api/alphaagent/composite-factors` 保持 100% 兼容。
2. **深度归因分析（置换贡献 / 子集曲线 / 多路径）**：
   - 训练完成后，作为【深度归因面板】折叠卡片渲染在说明书下方，用户需要时一键展开查看 ECharts 细分图表，既保证主流视图轻盈清爽，又不丢失专业归因深度。
3. **因子详情机制卡片（`factorCard`）**：
   - 保持全局 Teleport 挂载，时间线中出现的任何因子名称点击均可直接呼出机制卡片。

---

## 5. 验收门禁与需求澄清

### 5.1 门禁测试集（Checklist）
- [x] **断言 1（后端请求默认隔离）**：`StackingTrainRequest().eval_mode == "tuning"`
- [x] **断言 2（盲测模式禁止 LLM）**：`train_ml_composite.py --eval-mode blind_test --llm-assist` 返回非 0 且被安全拦截。
- [x] **断言 3（盲测报告拒识）**：`llm_summarize_report` 传入包含盲测数据的报告返回 `None`。
- [x] **断言 4（Prompt A 盲测年份脱敏）**：`_build_prompt_a` 输出不包含 `2025`、`2026` 盲测期年份。
- [x] **断言 5（Report C 验证段约束）**：`_build_report_view` 输出不包含 `2025`、`2026` 盲测期年份。
- [ ] **断言 6（流式日志事件回放）**：构造包含 `[ML_EVENT]` 的日志文件，断言 SSE 事件总线可完整还原时间线卡片。

### 5.2 需求边界澄清
1. **`artifacts/alphaagent/composite_blind_test/` 归档目录**：
   - 本次范围（Phase 1）不自动创建空目录，仅定义该路径为定稿组合唯一合法的离线重测落盘路径；
2. **“提交盲测终审”按钮**：
   - 本次范围不将该按钮暴露在研发交互流中，防止误触导致盲测集过早曝光；
   - 研发人员只有在研发态满意、明确另存为组合因子后，才可走离线审核流程。

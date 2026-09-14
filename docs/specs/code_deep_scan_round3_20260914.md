---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '7fe6c29f-fcdb-46ae-a4f8-392996999f3d'
  PropagateID: '7fe6c29f-fcdb-46ae-a4f8-392996999f3d'
  ReservedCode1: '175639f6-d761-411b-bdcd-bd5bbb99b641'
  ReservedCode2: '175639f6-d761-411b-bdcd-bd5bbb99b641'
---

# 第三轮深度收口扫描报告（2026-09-14）

> 第一轮 `code_reuse_audit_20260914.md` 覆盖了 9 项可复用问题（P0-P2），
> 第二轮 `code_stratification_governance_spec.md` 覆盖了 5 层地质治理。
> 本轮聚焦"还有哪些可以收口"，对 backend / alphaagent / tests / 前端 / core-jq
> 五个层做逐层深挖，补充前两轮未覆盖的发现。

## 0. 本轮新增结论速览

| 优先级 | 事项 | 预计收益 | 风险 |
|--------|------|----------|------|
| P0 | services_old.py → 正式收编为 services/ 下子模块后删除旧文件 | 消除"old"命名包袱 + 407 行搬家非删 | 低（纯重命名+导入路径变更） |
| P0 | auth.py 路由未注册（孤立文件 116 行） | 安全漏洞或死代码二选一，需确认 | 低（确认后要么注册要么归档） |
| P1 | `_to_float` ×7 处重复定义 | 统一到 `core/instruments.py` 或新建 `core/numutil.py` | 低 |
| P1 | logging_config.py + logging_decorators.py 合并 | 527 行 → 单文件 ~400 行，消除跨文件循环导入隐患 | 低 |
| P1 | qweave_runner.py 从 `scripts/` 导入生产代码 | 后端不应依赖 scripts 层，需反转依赖 | 中（需移动 qweave_research.py） |
| P2 | MlPanel.vue 1,340 行拆分（模板 619 行 + 脚本 581 行） | 可维护性 | 中（需前端构建验证） |
| P2 | tests 14,188 行 / 90 文件，jq 层 0 测试覆盖 | 识别测试盲区 | 无（仅报告） |
| P2 | core/event_engine/jq 3,936 行聚宽兼容层不可精简 | 确认无需动 | 无（仅报告） |
| 不建议 | 拆 alphaagent mining 核心 17.5k 行 | — | 风险远大于收益 |

## 1. backend 层深入分析（10,036 行 / 28 个 .py）

### 1.1 services_old.py：不是废弃文件，是"搬家未完成"（P0）

前两轮报告将 services_old.py 标记为"疑似废弃"。本轮确认实际情况：

- `services/__init__.py` 第 4 行 `from backend.services_old import (...)` **主动导入 16 个函数**
- 所有 router 通过 `from backend import services` 间接使用 services_old 的函数
- services_old.py = 数据加载 + 缓存 + 更新编排 + 内存保护，是 backend 的**核心数据服务层**
- `services/` 子目录只拆出了 4 个 AlphaAgent 专用类（session_manager / factor_evaluator / factor_repository / factor_submitter），共 195+行

**结论**：services_old.py 不是废弃文件，是"搬家到 services/ 子目录"未完成的中间态。
命名带 `_old` 后缀误导性极强。

**建议**：
1. 将 services_old.py 拆为 `services/market_data.py`（load_data / cache / update）和
   `services/codes.py`（build_codes / get_name_map / get_industry_map / 工具函数）
2. 更新 `services/__init__.py` 的导入路径
3. 删除 services_old.py（407 行搬家，零功能变更）
4. 前提：全量 router 回归测试（backend 有 test_backend_services.py 覆盖）

### 1.2 auth.py：孤立文件，路由未注册（P0）

`backend/auth.py`（116 行）定义了完整的登录/登出/验证路由（`/login`、`/logout`、`/me`），
但 `main.py` 中 **未注册 auth.router**，全仓无任何文件 import auth 模块。

两种可能：
- **死代码**：早期认证功能，已被 main.py 的 `access_gate` 中间件替代（main.py:157 有
  `_gate_render` / `_safe_next` / `access_gate`，是 cookie+密码门控的另一套实现）
- **遗漏注册**：设计完成但忘记 `app.include_router(auth.router)`

**建议**：确认 main.py access_gate 与 auth.py 是否为同一功能的两套实现。
若是替代关系 → 归档 auth.py；若是遗漏 → 注册路由（安全相关，需用户确认）。

### 1.3 alphaagent_service.py 2,288 行：三段拆分方案细化（P2）

前两轮已标注为拆分候选。本轮逐函数扫描确认三段边界：

| 段 | 行范围 | 函数数 | 职责 |
|----|--------|--------|------|
| **Run 生命周期** | 68-992 | 28 个函数 + AgentRun class | 进程管理、事件流、start/stop/branch/continue/archive/delete |
| **因子库操作** | 1039-1897 | 18 个函数 | list_factors / get_factor_detail / delete_factor / save_factor / candidate registry / freq map |
| **评估编排** | 1897-2288 | 4 个函数 | evaluate_single_factor / evaluate_multi_profile / _eval_factor_matrix / backtest_factor |

拆分方案：
- `backend/services/run_lifecycle.py`（~900 行）— run 管理
- `backend/services/factor_library.py`（~900 行）— 因子库 CRUD
- `backend/services/factor_evaluation.py`（~400 行）— 评估编排
- `backend/alphaagent_service.py` 保留为 thin facade（re-export + 公共状态）
- `routers/alphaagent.py`（1,137 行）改 import 路径

**前提**：需 90 个测试文件中涉及 alphaagent 的 ~30 个全部通过。

### 1.4 logging 527 行：合并可行性确认（P1）

| 文件 | 行数 | 内容 |
|------|------|------|
| logging_config.py | 287 | get_logger + 5 个预定义 logger + RequestContext + parse_log_file |
| logging_decorators.py | 240 | log_function_call + log_data_loading + log_backtest_execution + LogContext |

使用统计（全仓去重后）：
- `log_function_call`：13 处引用
- `log_data_operation / log_data_loading`：9 处
- `log_backtest_execution / log_backtest_operation`：9 处
- `LogContext / log_block`：19 处
- `RequestContext`：132 处（高频使用）
- `parse_log_file`：7 处
- `get_logger_with_context`：**1 处**（仅定义未被使用，可删）

**结论**：两个文件功能高度内聚（配置+装饰器同属日志基础设施），合并为
`backend/logging.py` 消除跨文件导入，删 `get_logger_with_context`（死代码）。
527 → ~480 行。

### 1.5 qweave_runner.py 297 行：后端反向依赖 scripts 层（P1）

```python
# qweave_runner.py:17
from scripts.qweave_research import ALPHA_SETS, build_alphas, load_panel, to_qweave_df
```

后端不应 import scripts 层（scripts 是一次性脚本目录，非生产代码包）。
这违反了分层架构：backend → core → alphaagent（生产），scripts（辅助）。

**建议**：将 `scripts/qweave_research.py` 中被 qweave_runner 依赖的部分
（ALPHA_SETS / build_alphas / load_panel / to_qweave_df）移到 `core/qweave/`
或 `backend/qweave/`，qweave_runner 改从新位置导入。scripts 侧保留薄壳调用。

### 1.6 lab_runner.py 161 行：仅被 alphaagent_service.py 引用

```python
# alphaagent_service.py:1304
from backend.lab_runner import load_module
```

lab_runner.py 仅暴露 `load_module` / `_to_float` / `points` / `main` 四个函数，
其中 `_to_float` 与 services_old.py 的 `_to_float` 重复（见 §3.1）。
`main()` 是命令行入口，与 backend API 模式不同。

**建议**：`_to_float` / `points` / `clean_records` 统一到 core 层后，
lab_runner.py 仅保留 `load_module`（47 行），其余删除。

### 1.7 overnight_monitor_service.py 288 行：结构正常

10 个函数，职责清晰（start/stop/status + 日志读取 + 进程检测）。
仅被 `routers/alphaagent.py` 的 3 个端点引用。无冗余，无需动。

## 2. alphaagent 内部交叉分析

### 2.1 mining 三大模块无功能重叠

| 模块 | 行数 | 核心类/函数 | 职责 |
|------|------|------------|------|
| mining/memory | 5,024 | RetrievalMixin, ingestion, schema, analytics, advisory | 研究记忆存储+检索 |
| mining/agent | 3,109 | create_mining_agent, run_factor_mining_agentscope | LLM 驱动挖掘对话循环 |
| mining/delivery | 1,807 | FactorSubmitService, slug_factor_id | 因子提交/入库/去重 |

三模块通过明确的接口边界协作：agent → memory（检索上下文）→ delivery（提交结果）。
无重复函数定义，无交叉 import 混乱。**不建议动**。

### 2.2 alphaagent 侧 _to_float 重复（P1）

alphaagent 内部有独立的 `_to_float` 定义：
- `alphaagent_service.py:1832` — 用于评估结果序列化
- `lab_runner.py:26` — 用于 lab 结果序列化
- `alphaagent/factor/mining/memory/calibration.py:72` — 用于校准打分

加上 backend 侧 services_old.py:215 和 routers/backtest.py 中的，
全仓 **7 处** `def _to_float` 定义。

**建议**：统一到 `core/numutil.py`（或并入 `core/instruments.py`），
签名兼容最通用版本：
```python
def to_float(v, *, default: float | None = None) -> float | None
```

### 2.3 save_composite_factor 仅 1 处定义（非重复）

前两轮报告提到"脚本从不调用 save_composite_factor，仅前端手动入库"。
本轮确认全仓仅 `backend/composite_factor_service.py:119` 一处定义，
不存在重复实现，只是调用链不完整（设计问题非重复问题）。

## 3. tests 14,188 行 / 90 文件覆盖分析

### 3.1 测试行数分布

| 位数 | 文件数 | 行数 |
|------|--------|------|
| Top 5 | 5 | 2,854（20%） |
| 200+ 行 | 20 | ~6,000（42%） |
| 100-200 行 | 35 | ~4,200（30%） |
| <100 行 | 30 | ~1,100（8%） |

测试集中度合理：长测试是端到端集成（research_memory_v3 925 行、prediction_reconciliation 636 行），
短测试是单元测试。

### 3.2 测试覆盖盲区

| 模块 | 代码行数 | 测试文件数 | 覆盖判定 |
|------|----------|-----------|---------|
| alphaagent/dsl | 10,112 | 5 个（dsl_operator_consistency / dsl_lookahead_guard / dsl_input_guards / dsl_slow_patterns / dsl_operator_perf） | ✅ 覆盖充分 |
| alphaagent/factor/mining | 17,545 | ~20 个 | ✅ 覆盖充分 |
| core/event_engine/jq | 3,936 | **0 个** | ❌ **零覆盖** |
| core/engine | ~1,000 | 3 个（engine / engine_preview / execution） | ✅ 有覆盖 |
| backend | 10,036 | 3 个（backend_services / logging / logging_decorators） | ⚠️ 覆盖薄 |
| 前端 | 11,240 | 0 | ❌ 无前端测试（项目未配前端测试框架） |

**jq 层零测试**是最大盲区。但 jq 是聚宽兼容层，正确性靠"与聚宽结果对比"
（scripts/jq_repro/ 37 个对比脚本实质就是手动测试）。建议补 2-3 个自动化 jq 回归
测试（取 jq_repro 中有 CSV 对照数据的场景自动化），非紧急。

### 3.3 测试中的临时脚本

`tests/` 目录无 `_test_` / `_tmp_` 前缀文件，整洁度良好。

## 4. 前端组件冗余分析

### 4.1 store/alphaagent.js 1,046 行 vs utils/alphaagent.js 466 行

| 文件 | 职责 | 内联函数数 |
|------|------|-----------|
| store/alphaagent.js | Pinia-style reactive store（状态管理） | 4 个小工具函数（find/filter/set） |
| utils/alphaagent.js | 纯函数工具集（格式化/标签/解析） | 28 个 export function |

**结论**：两文件**无功能重复**。store 是状态容器，utils 是纯函数库。
store 内的 4 个小函数（findIndex / filter / set / find）是 Vue reactive 上下文中的
内联闭包，不适合抽到 utils。**不建议动**。

### 4.2 MlPanel.vue 1,340 行拆分方案（P2）

结构扫描：
- `<template>` 行 1-618（619 行模板，含 3 个 `v-if/v-else` 分支）
- `<script>` 行 619-1200（581 行脚本，含 `parseCommentSegments` 等内联函数）
- `<style scoped>` 行 1201-1340（140 行样式）

拆分方案（需前端构建验证）：
1. `MlPanel.vue` 主组件（~300 行）：ML 运行列表 + tab 切换骨架
2. `MlReportPanel.vue`（~250 行）：报告详情展示（行 375-618 的 report 分支）
3. `MlRunDetail.vue`（~200 行）：单次运行详情 + IC 曲线
4. 内联函数 `parseCommentSegments` / `fmtNum` 等移到 `utils/alphaagent.js`

**前提**：`npx vite build` 验证无编译错误；现有功能不受影响（纯组件拆分）。

### 4.3 前端无测试框架

项目未配置 vitest / jest / playwright，前端 11,240 行代码无自动化测试。
属项目级决策，不在本轮收口范围。

## 5. core/event_engine/jq 3,936 行精简评估

### 5.1 结构与行数

| 子目录 | 行数 | 职责 |
|--------|------|------|
| runtime.py | 963 | JQRuntime 核心运行时（聚宽 API 模拟） |
| objects.py | 362 | 聚宽对象模拟（_CodeData / _Context / _OrderCost 等） |
| api/ | 863 | 7 个 API 模块（data_api / finance / framework / misc / portfolio / settings / trading） |
| datalake/ | 573 | 数据湖插件（7 个插件：stock_daily / income / balancesheet / cashflow / fina_indicator / index_bars / industry_members） |
| entry.py | 219 | run_jq_backtest 入口 |
| factor_bridge.py | 201 | 因子桥接（DSL → JQ 表达式） |
| preflight.py | 190 | 预检（代码安全扫描） |
| query.py | 176 | 查询构建器（_Col / _Query） |

### 5.2 引用关系

- **非测试代码引用**：core/event_engine/runner.py（1 处）、backend/routers/backtest.py（2 处）
- **scripts 引用**：大量（jq_repro 诊断脚本 + 几个生产脚本）
- **alphaagent 引用**：0
- **tests 引用**：0

### 5.3 结论

jq 层是**完整的聚宽 API 兼容层**，3,936 行中每个模块都有明确职责：
- runtime.py 963 行：模拟聚宽 `g` / `context` / `order_target_value` 等运行时对象
- api/ 863 行：模拟聚宽 `get_price` / `get_fundamentals` / `attr` 等数据接口
- datalake/ 573 行：将本地 parquet 数据适配为聚宽 API 返回格式

**不建议精简**。这 3,936 行是"用 3,936 行本地代码替代聚宽 SaaS 依赖"的完整实现，
删任何一行都会破坏对应的聚宽 API 兼容性。正确性靠 jq_repro 的 37 个对比脚本
（手动对照聚宽 CSV 结果）验证。

## 6. 跨层重复汇总（含前两轮已完成项状态更新）

| 重复项 | 处数 | 状态 | 本轮变化 |
|--------|------|------|---------|
| `_to_float` / `_safe_float` | 7 | **P1 待修** | 新增 alphaagent_service.py + lab_runner.py 两处 |
| SQLite connect 裸连 | 48 → ~15（排除 .temp/scripts） | P2 待修 | 无变化 |
| `save_composite_factor` | 1 | 非重复 | 确认仅 1 处定义 |
| report/JSON 读取 | 5 | ✅ 已收口（report_io.py） | — |
| 股票代码归一化 | 13 | ✅ 已收口（instruments.py） | — |
| `_now()` | 6 | ✅ 已收口（timeutil.py） | — |
| 原子写 | 4 | ✅ 已收口（atomicio.py） | — |
| FactorValueCache new | 8 | ✅ 已收口（get_default_cache） | — |

## 7. 实施建议（按优先级排序）

### P0 — 确认类（不改代码，需用户决策）

1. **auth.py 路由未注册**：确认是死代码还是遗漏注册（安全相关）
2. **services_old.py 重命名**：确认拆分方向后再动

### P1 — 低风险收口（各半天）

1. **`_to_float` 统一**：新建 `core/numutil.py`，7 处改 import
2. **logging 合并**：logging_config.py + logging_decorators.py → logging.py，删 `get_logger_with_context`
3. **qweave_runner 依赖反转**：qweave_research.py 生产部分移入 core/

### P2 — 需验证后执行

1. **alphaagent_service.py 拆分**：run_lifecycle / factor_library / factor_evaluation 三模块
2. **MlPanel.vue 拆分**：4 个子组件 + utils 函数提取
3. **services_old.py → services/market_data.py + services/codes.py**：搬家后删旧文件
4. **jq 层补测试**：取 jq_repro 有 CSV 对照的 2-3 个场景自动化

### 不建议

- 拆 alphaagent/factor/mining 核心（17.5k 行）
- 精简 core/event_engine/jq（3,936 行聚宽兼容层）
- 合并 store/alphaagent.js 与 utils/alphaagent.js（无重复）
- 添加前端测试框架（项目级决策）

## 8. 全仓行数全景（修正版）

| 目录 | 行数 | 占比 | 本轮变化 |
|------|------|------|---------|
| alphaagent | 41,734 | 37.6% | 细分见前两轮 |
| scripts | 19,036 | 17.2% | jq_repro 37 个可归档 3,551 行 + cne 43 个可归档 1,223 行 |
| core | 13,947 | 12.6% | 含 jq 3,936 行（不可精简） |
| tests | 14,188 | 12.8% | 90 文件，jq 零覆盖 |
| static/src | 11,240 | 10.1% | MlPanel 1,340 行可拆 |
| backend | 10,036 | 9.1% | services_old 407 行待搬家 |
| docs | 4,402 | 4.0% | — |
| .temp | ~5,000+ | — | 临时脚本（不进 git） |
| **合计** | **~110,878** | 100% | — |

> 功能代码（排除 tests + docs + .temp）约 77,287 行，
> 其中 jq 兼容层 3,936 行 + mining 核心 17,545 行 = 21,481 行属于
> "不可精简的基础设施"，占功能代码的 27.8%。
> 真正可优化空间约 55,806 行中的重复与冗余。

> AI生成
# AlphaAgent 可工具化/MCP 化函数清单 · 2026-10-07

> 用途：把"harness 直驱挖掘"（2026-10-07 战役已验证可行）正式化的**接口盘点**。
> 目标形态：一个 `alphaagent` MCP server（照 `CNEquity/src/cnequity/mcp_server/` 的手写 stdio 先例），
> 或等价的 CLI/HTTP 工具面，让任意 agent（DSH harness / Cursor / 自研调度器）用**同一套护栏与记忆**挖矿。

## A 类 · 已是 `FunctionTool`，MCP 化近乎零成本

实现：`alphaagent/factor/mining/agent/agentscope_tools.py`（`build_factor_eval_toolkit`），
schema 与描述已在 `alphaagent/factor/mining/tools/_schemas.py`（`TOOL_NAMES` :236），
可直接转 MCP `tools/list` 的 JSON Schema。

| 工具 | 实现位置 | 只读 | 说明 |
|---|---|---|---|
| `evaluate_factor` | `agentscope_tools.py:772`（FunctionTool :1063） | ✓ | 单因子完整评估（两阶段 + 诊断） |
| `eval_on_train_set` | :716（:1064） | ✓ | 训练段评估 |
| `eval_on_val_set` | :992（:1065） | ✓ | 验证段评估 |
| `propose_population` | :913（:1068） | ✓ | 种群模式候选提案（条件注册） |
| `submit_factor` | :1072（:1135） | ✗ **写** | 真实交付（stage_one→盲测→stage_two→engine_gate） |
| `screen_factors` | :1140（:1156） | 半写 | 批量筛选（按 profile） |
| `precheck_expression` | :1158（:1170） | ✓ | DSL 结构风险 AST 预检（不占评估预算） |
| `recommend_mrmr_factors` | `_schemas.TOOL_NAMES` 列出（条件注册） | ✓ | mRMR 推荐 |

统一入口：`tools/_dispatch.py` 的 dispatch（内含 **记忆死路硬拦 / 同质化熔断 / interaction 契约 / facet 越界** 护栏）。

## B 类 · 领域服务函数（需薄封装；★ 是本次战役暴露的刚需）

| ★ | 建议工具名 | 现有实现 | 读/写 | 为什么值得工具化 |
|---|---|---|---|---|
| ★ | `eval_batch(exprs[], split, mode)` | 无（本次新建 `scripts/harness_mine.py` 封装 `StockEvalService.eval_train` 循环 + 真源门槛判定） | 读 | **单次调用评估 20+ 因子**：本次效率核心（agentscope 一次一个工具调用） |
| ★ | `dry_run_delivery(expr, mode, freq)` | `delivery_checker.py` `DeliveryChecker.stage_one_stats:328` / `stage_one_val_retention:374` / `stage_one_correlation:387` / `stage_two:390` | 读 | submit 前的**只读预检**（本次靠它避免拿真提交试错） |
| ★ | `library_similarity(expr)` | `delivery/submit.py:_candidate_registry_similarity:140` | 读 | 提前预警 stage_two 的"库内相关性"墙（本次 0.75/0.86 撞墙） |
| ★ | `get_thresholds(mode)` | `research_spec.effective_research_spec` + `DeliveryCriteria.from_spec` | 读 | 让模型拿到**确切门槛**（IC/ICIR/cov/autocorr/换手），不再猜 |
| ★ | `list_fields(include_fundamentals)` | `SessionCreateResponse.available_columns`；字段语义见 `prompt/modules/data_fields.py` | 读 | 写 DSL 前必需的可用列清单 |
| ★ | `describe_operator(name)` | `alphaagent/dsl/catalog.operator_catalog_markdown` | 读 | 算子签名/语义查询（105 个算子） |
| ★ | `memory_search(query, mode, k)` | `memory/retrieval.py:RetrievalMixin`、`memory/advisory.py:query_for_attempts:402` | 读 | 查历史死路/成功骨架，避免重复探索 |
| ★ | `memory_record(observation)` | `memory/ingestion.py:IngestionMixin.record_tool_result:140` | **写** | **本次最大缺口**：harness 探过的死路没写回记忆，agentscope 明天还会再踩 |
| | `memory_analytics(kind)` | `memory/analytics.py:facet_operator_breakdown:160` / `research_funnel:249` | 读 | 面/算子级漏斗统计 |
| | `eval_profile(expr, profile_id)` | `eval/service.py:StockEvalService.eval_profile:439` | 读 | profile 级评估（train_screen/validation/size_neutral_validation） |
| | `next_question(mode)` / `mechanism_card(qid)` | `question_queue.py:get_question_for_turn:1411` / `load_question_queue:121` / `load_mechanism_cards:480` | 读 | 研报/复现流程的题面与机制卡 |
| | `library_list(pool)` / `factor_detail(id)` | `registry_io.load_mining_registry` + `core.factor_categories` | 读 | 候选/正式库检索 |
| | `blind_test(factor)` | `delivery_checker.py:DeliveryChecker.blind_test:317`（CLI：`scripts/blind_test_factors.py`） | **写（耗盲测预算）** | 需配额门控，禁止模型自由调用 |
| | `rescreen_pool(dry_run)` / `promote(factor_id)` | `scripts/rescreen_candidates.py` / `scripts/promote_candidates.py` | **写** | 用户拍板后执行，默认 dry-run |
| | `run_summary(run_id)` / `list_runs()` | 后端 `GET /runs`、`/runs/{id}`、`/runs/{id}/metrics`；或直读 `logs/factor_mining/ui/<run_id>/` | 读 | 台账/巡检/归因 |
| | `panel_status()` | `GET /session-cache/stats`（`backend/routers/alphaagent.py:847`） | 读 | 面板缓存/内存状态 |

会话生命周期（`StockEvalService.create_session:140` / `release_session:462`）**不建议暴露给模型**：
由 MCP server 内部管理（单会话 6–8GB，多客户端并发会互踩）。

## C 类 · 后端已有 REST（可直接复用或作为 MCP 的 HTTP 后端）

`backend/routers/alphaagent.py`（前缀 `/api/alphaagent`）：

| 端点 | 行 | 用途 |
|---|---|---|
| `POST /eval-factor` | :792 | 单因子评估 |
| `POST /backtest-factor` | :1079 | 引擎回测 |
| `POST /factors` | :1045 | 因子落库（需确认是否等价 submit） |
| `GET /research-memory`、`/research-memory/layers` | :177 / :217 | 记忆检索/分层 |
| `GET /research-modes`、`/research-specs/{mode}` | :304 / :373 | 档位与门槛口径 |
| `GET /runs`、`/runs/{id}`、`/runs/{id}/metrics`、`/runs/{id}/events` | :172 / :424 / :432 / :755 | 台账与事件流 |
| **`POST /runs/{id}/messages`** | :663 | **副驾式纠偏**（向在跑 run 注入指令）——零开发集成点 |
| `POST /runs/{id}/stop` | :623 | 停止 |
| `GET /blind-test` | :630 | 盲测 |
| `POST /overnight-monitor/start\|stop`、`GET /status` | :1262 / :1273 / :1281 | 整夜监控 |

## D 类 · 不建议工具化

- 评估内核与物化：`_materialize_split_aware`、`panel` 加载、`metrics/*`（重且易误用；已被上层工具覆盖）；
- AST 内部判定：`_prefilter.py` / `_precheck.py` 的实现（已由 `precheck_expression` 暴露）；
- 管理类：`session-cache/evict`、`stacking/*`、`runs/{id}/archive|rename|pin`（人用，模型用易误伤）；
- 数据同步：`scripts/refresh_*` / `sync_*` / `rebuild_*`（与挖掘无关，写数据湖）。

## E 类 · 建议的 MCP 第一批（12 个）

```
eval_batch · eval_val · dry_run_delivery · library_similarity · precheck_expression
submit_factor · get_thresholds · list_fields · describe_operator · memory_search
memory_record · run_summary
```

**实现约束（4 条）**
1. **写操作默认 dry-run**，真写需显式 `confirm=true`，并落审计日志；
2. 评估/提交一律**走 `_dispatch` 工具层**（保留记忆死路硬拦、同质化熔断、interaction 契约）；若确需直连服务（如批量评估），提供 `--no-guards` 并在 run 记录中留痕；
3. MCP server **照 CNEquity 手写 stdio JSON-RPC**（`CNEquity/src/cnequity/mcp_server/protocol.py`，避免 `mcp` SDK 的 15 个传递依赖）；
4. 会话由 server 内部管理（panel 6–8GB/会话，需串行/池化），对外只暴露 `session` 句柄或隐式复用。

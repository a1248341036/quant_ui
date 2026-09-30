# 审计报告 v5 发现逐条验证（2026-09-23）

来源：`audit/审计报告-quant_ui-alphaagent-v5.docx`（90 条发现：高 4 / 中 46 / 低 34 / 轻微 6）

验证方式：逐条对照当前源码（grep + read），标注 **属实 / 部分属实 / 误报**，并给出可改项优先级。

## 结论速览

- **高危正确性缺陷 2 条全部误报**（F-001/F-002），实际代码无 NameError。
- **可立即安全删除的死代码**：约 15 处（无调用方、删除零风险）。
- **需谨慎处理的重复/重构**：约 10 处（有调用方，需同步改调用方或保留兼容）。
- **误报**：约 8 处（报告引用的行号/代码与当前实现不符，或已有上限/调用方）。

## 高危正确性缺陷（报告称 2 条，实际 0 条）

| 编号 | 报告结论 | 验证结果 | 证据 |
|---|---|---|---|
| F-001 | `agentscope_run.py` 用未导入的 `Callable`，import 时 NameError | **误报** | `agentscope_run.py:3` 有 `from __future__ import annotations`，注解延迟求值；实测 `import alphaagent.factor.mining.agent.agentscope_run` 成功 |
| F-002 | `population.py:190` 裸变量 `icir` 导致 NameError | **误报** | `population.py:191` 有 `icir = r.get("icir")`；`screen_population` 内无 `round(icir, 4)` 调用（仅 `round(time.perf_counter() - t0, 1)`） |

## 可立即安全删除（无调用方，删除零风险）

| 编号 | 位置 | 内容 | 验证 |
|---|---|---|---|
| F-016 | `alphaagent/core/config.py` 整文件 | `load_yaml` 无调用方 | 属实 |
| F-017 | `alphaagent/core/hash.py` 整文件 | `panel_index_hash` 无调用方 | 属实 |
| F-018 | `alphaagent/core/types.py:58-60` | `Panel/AlphaTable/TargetBook` 别名无调用方 | 属实 |
| F-019 | `alphaagent/core/types.py:63-88` | `OrderSide/OrderIntent/StrategyBundle` 无调用方 | 属实 |
| F-031 | `alphaagent/dsl/core/ast.py:96-115` | `ASTNode.template()` 零调用者 | 属实 |
| F-032 | `alphaagent/dsl/core/ast.py:599-656` | `fingerprint_dedup_check` 零调用者 | 属实 |
| F-036 | `alphaagent/dsl/core/resample.py:197-223,226-257,330-335` | `build_60m_panel`/`resample_universe_long`/`broadcast_60m_to_main_freq` 仅 `__all__` 引用 | 属实 |
| F-037 | `alphaagent/dsl/registry.py:21-58` | `OperatorMeta`/`_REGISTRY`/`register_operator`/`iter_registered_operators` 零调用者 | 属实 |
| F-038 | `alphaagent/dsl/stock/incremental.py` 整文件 | `IncrementalWeekEngine`/`assert_incremental_matches_batch` 无调用方 | 部分属实（文件无调用方，但报告说 `__init__.py` 有 import 是错的——实际 `__init__.py` 只 re-export 4 个函数） |
| F-040 | `alphaagent/factor/metrics/decile.py:227` | `daily_quantile_group_returns` 仅 `__init__.py` 包装器引用 | 属实 |
| F-070 | `alphaagent/core/paths.py:23` | `FACTOR_REGISTRY_EXAMPLE` 无调用方 | 属实 |
| F-072 | `alphaagent/data/fundamental_fetch.py:532-534` | `ensure_fundamental_dir` 无调用方 | 属实 |
| F-073 | `alphaagent/dsl/eval.py:80-90` | `prune_panel_to_referenced` 无调用方 | 属实 |
| F-082 | `alphaagent/factor/zoo/index.py:128,138-139` | `_by_shard`/`shard_for_id` 无调用方 | 属实 |
| F-083 | `alphaagent/factor/zoo/index.py:141-155` | `row_slice_for_dates` 无调用方 | 属实 |
| F-089 | `alphaagent/factor/mining/agent/factor_reviewer.py:13` | `OpenAIChatModel` 死导入 | 属实 |

## 可安全删除但需同步改调用方

| 编号 | 位置 | 内容 | 验证 |
|---|---|---|---|
| F-004 | `alphaagent/compute/client.py:86-113` + `worker.py` | `WorkerPoolClient.run_engine_gate` 无生产调用方（仅测试用） | 属实；需同步删 worker.py 的 `_exec_engine_gate` 分支 |
| F-005 | `alphaagent/compute/client.py:39-113` | `WorkerPoolClient` 纯透传包装 | 属实；需改 `alphaagent_service.py:1144/1223` 直接调 `pool.submit_task` |
| F-028 | `alphaagent/data/tushare_client.py:55-70` | `configure()` 无调用方 | 属实；直接删 |
| F-048 | `alphaagent/factor/mining/eval/env_settings.py:27-31` | `resolve_max_parallel_eval` 薄包装 | 属实；需改 3 处调用方（run.py:44/agentscope_run.py:279/service.py:112） |
| F-049 | `alphaagent/factor/mining/infra/audit.py:18,26-35` | `file_hash(limit_bytes)` 参数无调用方传参 | 属实；直接删参数 |
| F-068 | `alphaagent/compute/client.py:21` | `get_global_worker_pool(num_workers)` 参数无调用方传值 | 属实；直接删参数 |
| F-080 | `alphaagent/factor/mining/operators.py` + `prompt/operators.py` | 7 行透传 shim | 属实；需改 run.py:18/agentscope_run.py:75 直接 import dsl.catalog |
| F-081 | `alphaagent/core/atomicio.py` | 2 行 re-export 兼容层 | 属实；需改 run_metrics.py:16 直接 import core.atomicio |

## 需谨慎处理的重复/重构（有调用方，需评估影响）

| 编号 | 位置 | 内容 | 验证 |
|---|---|---|---|
| F-003 | `alphaagent/dsl/core/accel.py` 3649 行 | God Object 单文件聚合 11 个内核 | 属实；拆分风险高（Numba JIT 缓存/import 链），建议分阶段 |
| F-020 | `alphaagent/data/adapters/cnequity.py:58-82` | 三板块过滤开关，仅 BSE 生效 | 属实；删除科创/创业开关需确认无外部依赖 |
| F-021 | `alphaagent/data/adapters/plugins/*.py` | CNE 配置加载 4 处重复 | 属实；`_pitlib.py` 已有样板，可合并 |
| F-022 | `alphaagent/data/fundamental.py:14-56` | `FUNDAMENTAL_STATEMENT_COLUMN_MAP` 43 行映射 | 部分属实（有消费方，但注释"供未来三大表接入"是 YAGNI） |
| F-023 | `alphaagent/data/fundamental_fetch.py:132-152` | `StatementSpec` 用 `__slots__`+`__init__` | 属实；可改 dataclass |
| F-024 | `alphaagent/data/fundamental_fetch.py:458-529` | `fetch_and_save_periods` 72 行编排 | 属实；可拆 `_fetch_period_bundle` |
| F-025 | `alphaagent/data/fundamental_fetch.py:195-247,265-316` | `fetch_fina_indicator_period`/`fetch_statement_period` 结构相同 | 属实；可提取公共函数 |
| F-026 | `alphaagent/data/index_members.py:80-111` vs `universe.py:206-274` | 月快照循环重复 | 属实；可合并 |
| F-029 | `alphaagent/data/universe.py:65-380` | 4 个回退策略冗余 | 部分属实（有调用方，但回退链确实冗余） |
| F-030 | `alphaagent/dsl/core/accel.py:89` vs `dyn_window.py:294-306` | `_OP_MAP_FIXED`/`_DYN_OP_MAP` 内容相同但**顺序不同** | 部分属实（min=0 vs mean=0，**不能简单合并**） |
| F-033 | `alphaagent/dsl/core/dyn_window.py:77-290` | 3 对串行/并行 Numba 内核重复 | 属实；需谨慎（Numba 并行语义） |
| F-034 | `alphaagent/dsl/core/guard.py:33,70-73` vs `eval.py:102,132-135` | `_DOLLAR_REF_RE`/`_strip_string_literals` 重复 | 属实；可提取共享模块 |
| F-039 | `alphaagent/dsl/stock/resample.py:59-73` vs `core/resample.py` | `_safe_divide`/`_empty_panel` 重复 | 属实；可提取共享模块 |
| F-042 | `alphaagent/factor/metrics/ic.py:250-261` | `evaluate_on_panel` 是 `evaluate_cs_on_panel` 纯透传 | 部分属实（有调用方 ingest.py:167/171，不能删但可简化） |
| F-057 | `alphaagent/factor/metrics/ic.py:272-296` vs `plugins.py:72-91` | winsorize 逻辑重复 | 属实；可统一到 numba 快路径 |
| F-058 | `alphaagent/factor/metrics/portfolio.py:75-601` | `quantile_portfolio_metrics` 526 行 | 属实；拆分风险高（数值回归门禁） |
| F-059 | `alphaagent/factor/mining/agent/agentscope_run.py:1062-1122` | 轮间反思注入 64 行 | 属实；可抽 `_build_reflection_lines` |
| F-060 | `alphaagent/factor/mining/prompt/modules/data_fields.py:337-403` | render 66 行 9 字段族判定 | 属实；可抽公共判定函数 |
| F-066 | `alphaagent/factor/mining/delivery/submit.py:73-91` | `_panel_cs_pearson_mean` 逐日 groupby | 属实；性能优化 |
| F-067 | `alphaagent/factor/mining/prompt/modules/operator_catalog.py:61-69` | 每次 render 重新生成 markdown | 属实；可缓存 |
| F-074 | `alphaagent/factor/evaluation/engine.py:33-36` | `_f_or_none` 2 行函数 | 属实；可内联 |
| F-078 | `alphaagent/factor/mining/infra/jsonutil.py:16-24` | `json_safe` 不处理 ndarray | 部分属实（调用方都传 dict/list，实际影响有限） |
| F-085 | `alphaagent/factor/mining/prompt/modules/data_fields.py:363` | `ff_available` 用 `all()` 其他用 `any()` | 属实；口径不一致 |
| F-086 | `alphaagent/factor/mining/prompt/modules/behavior_rules.py:100` | `diag_turnover=0.40` 硬编码 | 属实；应与 `diagnostics._REDLINE` 同源 |
| F-087 | `alphaagent/factor/mining/prompt/modules/data_fields.py:304-306` vs `facet_focus.py:18` | `_NEUTRAL_COLUMNS` 不一致（5 vs 6 列） | 属实；需对齐 |
| F-090 | `alphaagent/factor/mining/prompt/prompt_modules.py:56` | `PromptContext.extra` 通用 dict | 属实；可改显式字段 |

## 误报（报告与当前实现不符）

| 编号 | 报告结论 | 实际 |
|---|---|---|
| F-001 | Callable 未导入 NameError | `from __future__ import annotations` 延迟求值，import 正常 |
| F-002 | 裸变量 icir NameError | `population.py:191` 已正确赋值 |
| F-051 | `_split_cache` 无大小限制 | `panel_store.py:25` 有 `max_splits_per_session=8`，:63-64 有 LRU 淘汰 |
| F-053 | `filter_universe` 基于过期 is_st 错误过滤 | `_enrich_trade_flags` 的 fail-open 分支逻辑正确（`elif _column_all_nan(panel, "not_st")` 才走回落） |
| F-071 | `paths.py:26` 有 `MINING_EXPR_DIR` | `paths.py` 只有 25 行，无此定义 |
| F-075 | `portfolio.py:171-179` 有 9 个未使用变量 | 实际只有 `n_days = 0`（:170），报告引用的变量已不存在 |
| F-076 | `schemas.py` 重复定义 `_test_end_default` | `schemas.py:7` 从 `context` import，是复用不是重复 |
| F-084 | `resolve_sample_*` 纯透传无调用方 | `zoo.py` 有 9+ 处调用方 |

## 建议修复顺序

1. **P0（零风险死代码删除）**：F-016/017/018/019/031/032/036/037/038/040/070/072/073/082/083/089
2. **P1（需同步改调用方）**：F-004/005/028/048/049/068/080/081
3. **P2（口径一致性）**：F-085/086/087（prompt 模块，直接影响 LLM 行为）
4. **P3（重复/重构）**：F-020/021/022/023/024/025/026/029/034/039/042/057/059/060/066/067/074/078/090
5. **P4（高风险 God Object）**：F-003/033/058（需数值回归门禁保护）

## 验证环境

- 分支：`fix/audit-v5-findings`（从 `feat/candidate-vle-gate` 切出）
- 验证方式：grep + read 逐条对照当前源码
- 未修改任何生产代码（仅写本文档）
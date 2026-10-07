# AlphaAgent 架构与开发参考

> 本文档由 `AGENTS.md` 瘦身时迁移而来（2026-09-28），是 AlphaAgent 模块的详细参考。
> `AGENTS.md` 只保留指令与速查；**改这些模块前先读本文档**。

## 定位

AlphaAgent 是 quant_ui 中的 **A 股日频多因子自主挖掘智能体**。通过 LLM 驱动的多轮对话循环，自动生成因子 DSL 表达式 → 训练集评估 → 验证集检验 → 去重审查 → 入库交付，形成闭环研究流程。

## steps.log 分步日志

挖掘 run 内每个关键步骤一行（`run_start / turn_start / evaluate /
submit.stage_one_stats / submit.blind_test / submit.val_retention / submit.stage_one /
submit.review / submit.candidate_stored / submit.stage_two / submit.engine_gate /
submit.promoted / submit_result / memory.record / memory.advisory / memory_retrieve /
memory_suggest / memory_form / memory_distill / run_end`），k=v 字段 + turn 号，
写 `<run_id>/steps.log` 并镜像 stdout（进 `console.log`）。实现 `alphaagent/factor/mining/runlog.py`，
由 `agentscope_run.setup_run_logger(log_dir)` 装配；日志异常全吞，绝不阻断挖掘。

## 核心流程

### 1. 启动链路

```
前端 POST /api/alphaagent/runs
  → backend/alphaagent_service.start_run()
    → subprocess.Popen(.venv/Scripts/python.exe scripts/run_alphaagent.py ...)
      → run_alphaagent.load_codex_provider()  (读 ~/.codex/config.toml 获取 API key)
      → runpy.run_path(scripts/alphaagent_factor_mining.py)
        → alphaagent.factor.mining.loop.run()
```

- `panel_path` 默认 `"cne://"`，触发 `alphaagent/data/adapters/cnequity.py` 从 CNE 数据湖实时构建 panel。CNE adapter 有磁盘面板缓存（`artifacts/panel/cache/panel_*.parquet`），请求区间被缓存覆盖时秒级命中，不重复重建。
- **因子值缓存**（`alphaagent/factor/cache.py`）：`eval_factor(expr, panel)` 确定性结果的会话级复用。key = sha256(expr + panel 内容指纹 + schema 版本)；值只存对齐行序的 float32 数组，命中时用当前 `panel.index` 重建 Series。**内存 LRU 跨会话共享**（进程级单一 `_SHARED_MEM`，上限 16 条 `ALPHA_FACTOR_CACHE_MEM_MAX_ENTRIES` 覆盖，多开会话内存不线性增长）+ 磁盘持久化（`artifacts/factor_value_cache/`，`.npy` + `.json` 元数据，跨会话/跨进程共享）。**磁盘空间控制**：总字节上限 `_FV_MAX_BYTES`（默认 2GB，`ALPHA_FACTOR_CACHE_MAX_BYTES` 覆盖）+ 文件数上限（默认 1500），写入后按 `last_access` LRU 淘汰最久未用条目；孤儿/损坏文件在启动对账时清理。接入点：`EvaluationEngine.evaluate()`、`materialize_factor()`、`_candidate_registry_similarity()`。
- `factorlib_path` 默认 `factor_categories.production_dir(research_mode)`（统一大库 `artifacts/alphaagent/factorzoo/production_main`，2026-09-03 起两模式共享）。
- **DSL 算子耗时监控**（`alphaagent/dsl/core/monitor.py`）：零侵入自动计时。`eval_factor` 在求值命名空间层包计时代理，每次 DSL 求值自动采集各算子 `(calls/total/avg/max/参数摘要)`，附到结果 Series 的 `attrs["operator_timing"]`（引擎结果带 `operator_timing` 字段），并追加写 `artifacts/dsl_operator_profiling.jsonl` 累计历史。thread-local 隔离（挖掘 4 并行 worker 互不污染）；`ALPHA_DSL_OPERATOR_MONITOR=0` 关闭。查询 API：`GET /api/alphaagent/dsl-monitor?top_k=&since_hours=`。
- 环境变量 `ALPHA_LLM_PROVIDER=codex` 时从 `~/.codex/config.toml` 读取 bearer token 和 model。

### 2. 挖掘循环（loop.py）

每轮：
1. LLM 发起原生 `tool_calls`（可并行多条 `evaluate_factor`）。
2. `FactorEvalTools.evaluate_factor()` 在 `StockEvalSession` 中执行 DSL → 返回 IC/ICIR/coverage/monthly robustness。
3. 通过海选门槛的候选由 LLM 调用 `submit_factor()` 入库。
4. 用户可通过 `continuations.jsonl` 控制文件注入消息继续引导。

### 3. 三层约束（prompts.py System Prompt）

| 层 | 名称 | 机制 |
|---|---|---|
| 第一层 | 经济直觉强制 | 输出 DSL 前必须写出因果链条，否则跳过 |
| 第二层 | 父本变异策略 | 从已验证因子选父本，只允许参数/算子/修饰三种变异 |
| 第三层 | 正交预判 | 与已有因子的截面 Spearman > 0.7 自动拦截 |

### 3b. 认知层升级（2026-09-03，预测-对账 + 机制驱动探索）

针对"经济直觉叙事化、产出低级组合"的结构性问题（叙事门槛只考文笔不证伪、十分位判读
教学只有一个 bit、D 轨按算子而非机制开拓、门控式正交激励），五件套全链路升级：

- **A 预测-对账闭环**：`evaluate_factor` / `eval_on_train_set` 的 `prediction` 参数
  **必填**（`expected_shape`/`expected_strong_side`/`expected_sign` + 可选 `falsifier`），
  缺失直接 `ToolArgumentsError` 不执行。评估成功后自动对账：十分位实际形态
  （单调/倒U/U型/尖峰/不规则 + 强侧 D1-D3/D4-D7/D8-D10 + spearman）与预期比对，
  结果注入 `prediction_check`（verdict: confirmed/partial/contradicted/unverifiable）。
  `contradicted` = 机制错误，规则禁止对被证伪结构做参数变异。共享实现
  `alphaagent/factor/mining/eval/prediction.py`（纯 Python，含实测倒U用例锁定分类边界）。
  记忆层：对账摘要入 `memory_observations` 的 metrics（`prediction_check` 键），
  被证伪时 conclusion 追加"预测对账:"前缀（FTS 可检索）。agent 侧（agentscope_tools）
  三个 eval 包装器同步透传 prediction。因子实验室（HTTP 直调引擎）不经过 dispatch，不受必填约束。
- **B 十分位形态学教学**：`ic_robustness` 模块扩写——单调性/倒U判别（反转因子
  "IC 正但多头端无肉"的实测结构）、alpha 集中端、纯多头可交易性判定（持仓端 mean_label
  须优于全样本均值且覆盖换手）、`prediction_check`/`ablation_check` 读法、被证伪后的行为规则。
- **C A股机制知识模块**：新增 `prompt/modules/market_mechanisms.py`（ORDER=55，仅
  `asset_type=stock` 启用）——"错误定价×套利受限"框架、经典异象在 A 股的截面分布
  （反转集中弱势股/倒U结构/门控毁信号实测案例）、财务列阶梯函数的 DELTA 语义陷阱、
  制度摩擦清单（T+1/涨跌停/券源/解禁/指数调仓）、注意力与信息扩散。
- **D 轨机制驱动重构**：`strategy_tracks` D 轨"冷门算子清单"降级为次要线索，主协议改为
  **机制三问**（谁在错边 / 为什么修不平 / 哪个字段观测），三问缺一按无效跳过。
- **E 强制消融**：表达式含门控类算子（`GATED_SIGNAL`/`IF_THEN_ELSE`/`PIECEWISE_STATE`/
  `CS_GROUP_RANK`）且契约传 `base_expr` 时，dispatch 自动跑 base-only（引擎直调，
  不污染候选记录）并注入 `ablation_check`（destroyed_value/flipped_signal/added_value/
  neutral + 数字对比）；未传 `base_expr` 注入 `ablation_hint` 提醒。交互契约新增
  `base_expr`/`condition_expr` 可选字段。
- 提示词改动已同步重生成黄金基线 `tests/fixtures/system_prompt_*.txt`；新增测试
  `tests/test_prediction_reconciliation.py`（形态分类/对账判定/dispatch 必填/消融/记忆入库）。

### 4. 评估引擎（factor/evaluation/engine.py）

- 冻结的 `EvaluationProfile` 定义 transform → metric → rule 管线。
- 默认 profiles: `train_screen`, `validation`, `size_neutral_validation`。
- 支持插件化扩展（`plugins.py`）。

### 5. 入库两阶段（submit.py）

> 门槛数值唯一真源在 `delivery_criteria.DeliveryCriteria`（由 research_spec 注入），
> 下表为默认值（technical 模式）。真实判定读 `submit.py` 的 `self.criteria`，
> 与 prompt 渲染（`DeliveryCriteria.to_prompt_text()`）同源，杜绝口径漂移。

| 阶段 | 门槛（默认） | 写入位置 |
|---|---|---|
| 盲测终审（stage_one 之前） | \|test IC\| ≥ 0.010（2026-09-11 新增绝对下限，防死门 t≈2.5）, test/train IC 保留比 ≥ 0.50, 方向一致 | 不进任何库（不通过直接拒） |
| candidate（观察池，2026-09-11 第二版） | \|train IC\| ≥ 0.020, \|train ICIR\| > 0.28, coverage > 0.85, max_corr < 0.5, lag1 自相关 ≥ 0.18, val 保留比 ≥ 0.5, \|val IC\| ≥ 0.012 | candidate_main/（统一大库） |
| production | \|train IC\| ≥ 0.025, \|train ICIR\| ≥ 0.30, \|val IC\| ≥ 0.015, val 保留比 ≥ 0.60, winsorized 衰减 ≤ 0.10, max_corr < 0.4 | production_main/（统一大库） |

> **2026-09-11 门槛演进（两轮）**：第一版曾把进池线抬到与精筛对齐（0.025/0.30，
> 预筛池）——实测 technical 日频带（IC 0.015~0.03）几乎清空（25 条候选仅 1 条 tech
> 存活），遂回改为**观察池**：进池线回到 0.020/0.28（晋升线之下半档），val（≥0.012）
> 与盲测（≥0.010）绝对门保留——三段绝对门防"train 行 OOS 死"的假因子，量纲按各段
> 噪声折算（统一数字 ≠ 统一严格度；全段统一 0.02 会使 technical 存活为 0，实测）。
> `evaluation_policy`（评估屏幕线）同步 0.020/0.28/0.012；fundamental 档 override 与
> technical 同值（0.020/0.28/0.012，作为量纲锚），仅 val 保留比 0.65 与 production 更严项
> （保留比 0.70/衰减 0.12）是实差异；**晋升线 0.025/0.30 始终未动**。
> 盲测绝对门不增加盲测段查询次数（同一份 test 评估上加一条判定），但**每加一条盲测
> 判据都会轻微增加对盲测段的选择偏置**，故三条（绝对/保留比/方向）为上限，不再细叠。
>
> **存量重筛（2026-09-11）**：门槛变更后跑 `scripts/rescreen_candidate_pool.py` 把当前门槛回溯
> 应用于存量候选，两种动作：**soft-drop**（缺省，`dropped_from_ml` + `dropped_reason` +
> `dropped_at`，条目/DSL/研究记忆保留，ML 组合训练集默认剔除）与 **--hard**（条目 + DSL
> 移出并完整归档到 `*.removed-prescreen-<ts>.json` / `expressions.bak-prescreen-<ts>/`，
> 可整体回捞；研究记忆不清理——记忆存的是评估证据，与库成员资格解耦）。promoted 条目
> 一律不动；prescreen 自家标记（`dropped_reason` 以 `prescreen_gate:` 开头）在判定转好时
> 自动复活，衰减审计等其它来源的标记不自动动、仅列出供人工复核。每次执行都会先写
> `mining_candidate_registry.bak-prescreen-<ts>.json` 全量快照。

提交流程顺序：
0. **盲测终审**（test 段 IC 绝对下限 + 保留比 + 方向一致）→ 不通过直接拒绝，不进候选池
1. **stage_one 统计门槛**（IC/ICIR/coverage/换手/自相关/val 绝对下限/val 保留比）→ 不通过拒绝
2. **正交性检查**（stage_one 统一查，不再只在 approve 后触发）→ 与正式库已有因子做截面相关，超过阈值拒绝
3. **review_hook**（LLM Reviewer 审核）→ **仅 reject 硬拦**（抄袭/经典暴露不进任何库）；revise/pending_review 不阻断晋升，仅记录意见
4. **val 引擎回测预演**（候选入库前统一跑一次，`metrics["engine_gate"]` 随候选/正式记录落库，供三段表"引擎净值"行展示；裁决仍在第 6 步）
5. 入候选池（registry_only）
6. **stage_two 精筛**（双窗口口径）→ 不通过停在候选池
7. **engine_gate 净值裁决**（复用第 4 步回测结果）→ 不通过停在候选池
8. **test 段引擎回测诊断**（仅晋升因子补算，供正式库三段表盲测列展示）→ 不卡准入
9. 入正式库（canonical 对齐 + ingest）

**晋升链路关键语义（2026-08 修复）**：
- **stage_one / stage_two 的相似度只查正式库**，候选池内部冗余不卡正式库准入。
  候选池冗余由 `scripts/dedup_candidate_factors.py` 主动去重管理，而非让候选池因子
  互相挡死晋升（历史死锁：`gap_vwap_ens_wo3` 被候选池内相关 0.598 的 `vwap_close_dev_mom` 卡死）。
- **Reviewer 的 revise 是建议不是门槛**：与系统提示词"Reviewer 意见仅供参考改进方向，
  不阻断提交"一致；正式库准入的最终裁决是 stage_two 统计门槛 + engine_gate 净值回测。
- **engine_gate 是实盘可交易性裁决**：weekly 调仓、净超额年化 ≥3%、超额夏普 ≥0.5、
  回撤 ≤30%（2026-09-22 由 40% 收紧，绝对净值口径）、持仓重叠 ≥50%、仓位利用率 ≥80%、
  日均执行换手 ≤ 分档值（2026-09-22 新增：diag avg_daily_turnover 超 candidate 分档
  换手门即 fail_reasons=high_turnover；此前换手仅诊断不裁决 + slippage=0，引擎层对
  换手零约束）。统计 IC 高的因子若换手高（如日换手 69%、
  周重叠 5.6%），实盘净超额会转负 —— engine_gate 正确拦截"统计有效但实盘亏钱"的假因子。

### 6. 候选因子库管理

- 前端因子库页支持正式库/候选库切换，各有导出 JSON 按钮
- `scripts/dedup_candidate_factors.py`：一次性去重脚本，两两截面 Pearson 相关 ≥ 阈值的因子对删除冗余项
- 候选因子 registry 的 `similarity` 字段记录与已有因子的截面 Pearson 相关

### 6b. 因子中台 ID 与 factor_index.db（2026-09-07）

- **factor_uid**（`alphaagent/factor/identity.py`）：`"f" + sha256("factor:"+factor_name)[:16]`
  确定性派生——registry JSON、研究记忆库、索引库三方独立计算即一致，无需协调签发；
  代价是 factor_name 即身份根（改名 = 新身份，与 registry 以名为键的现状一致）。
- **记忆库 v5**（`memory_factors` 维表 + `memory_entries.factor_uid` 列，data_version=5
  幂等迁移按名回填）：写路径 `_write_entry` 自动盖 uid 并维护维表；
  `purge_factor(factor_uids=...)` 首选按 uid 删除（精确无歧义），name/expression 文本
  匹配仅为存量兜底，结束后自动清理维表孤儿行。
- **factor_index.db**（`alphaagent/factor/index.py`，`artifacts/alphaagent/factor_index.db`）：
  因子中台索引，单表 factors（uid 主键 + status candidate/production/memory_only + 各存储
  位置）。**派生物非事实源**：registry_io 两条入库路径与删除链路双写（失败静默不阻断
  挖掘），`FactorIndex.rebuild()` 全量重建自愈。
- **删除链路**（`backend delete_factor`）：registry pop → dsl unlink → 记忆 purge（uid 优先）
  → 索引删行，返回体带 `factor_uid`。
- `scripts/backfill_factor_uid.py`：一次性回填（备份记忆库 → registry 盖章 → v5 迁移 →
  索引重建），幂等可重跑。
- 已知取舍：SQLite `ALTER TABLE` 不能给存量表补外键，故 `memory_entries.factor_uid` 为
  普通列（删除由同事务双语句保证）；DB 级 FK CASCADE 留给索引库未来子表。

### 7. 因子类别注册表（`core/factor_categories.py`）

候选库 + 正式库路径由 `RESEARCH_MODES[mode].candidate_dir/production_dir` 派生。
**2026-09-03 统一大库**：两模式均指向 `candidate_main`/`production_main`（因子带
facets 标签，跨模式正交查重生效）；`research_mode` 不再是库边界，只保留门槛档位/
label/panel 列加载/engine_gate 频率的语义。迁移脚本
`scripts/migrate_unified_factorzoo.py`（旧库 rename 为 `*.bak-unified-*` 备份）。

### 8. 统一交易参数（`core/trading_config.py`）

全平台唯一默认交易参数来源（散户口径），回测/模拟盘/因子门禁/前端默认值均从此取。
后端 API `/api/trading-defaults` 供前端启动时自动获取。

### 9. 研究记忆（research_memory.py，v3-lite）

三层结构（SQLite WAL + FTS5，单文件 `artifacts/alphaagent/research_memory.db`，`store_meta.data_version="5"` 幂等迁移）：
- **原始证据层** `memory_entries`：每次评估/提交一条，verdict 7 级 + 数据化结论 + 失败码/失效项 + 结构指纹 + 父子关系（`parent_origin`=explicit/implicit、`secondary_parent_id` crossover 双父、`intended_motif` 意向编辑）+ `facets_json` 数据面标签（v4 新增；老行读取时按表达式现算兜底）+ `factor_uid` 因子中台 ID（v5 新增，`memory_factors` 维表按名聚类回填，见 §6b）。
- **编辑统计层（SSPM）** `memory_cells`：键 = (family × motif × 父本质量桶)；残差 = 子代 IC − 同桶时间衰减基线（half-life 90d，无历史回退父本 IC，AlphaMemo Eq.4）；成败按 explicit（±1.0）/implicit（±0.5）加权分列；**无效尝试（报错）入账失败观测（权重 0.5）**。
- **经验层** `memory_experience`：成功模式（签名 + 模板 + 实例）/ 禁忌方向（典型相关 + 失效项）/ 洞察（入库率）。

校准与注入（AlphaMemo 论文口径）：
- 置信度 Eq.7：`c = n/(n+κ)·min(1,|μ|/(σ+ε))`，κ=8；APV 双门 Eq.11/16：`veto = c>τ_c ∧ π⁻>τ_v`（默认 0.35/0.80，`memory_policy.apv_tau_c/apv_tau_v` 可调）；正证据软推荐封顶"优先尝试"档。
- 注入门控矩阵（`retrieval._edit_prior_block`，阈值经 `memory_policy.edit_prior_*_conf` 可调）：硬推荐(s>0 ∧ c>0.7) / 软推荐(s>0 ∧ c>0.4) / 硬否决(f>0 ∧ c>0.7) / 软否决(f>0 ∧ c>0.3，低于推荐向以放行"一致失败"避坑证据) / 其余不注入；APV 双门(τ_c=0.35, τ_v=0.80)另在评估前 advisory 做 (family, motif) 聚合否决。2026-08-30 修正：旧文档写的"软否决 fail_rate≥0.6"与实现不符，实际口径全部基于 Eq.7 置信度分档。`memory_policy.max_inject_chars`（默认 2400）超限时核心块（经验、编辑先验、**交互结构命中率**）始终保留，次级块按 证据 > 饱和度 > 产出率 > 多样性 用剩余预算填充，所有截断在行边界。
- **交互结构引导（2026-09-06）**：`retrieval._structure_stats_block` 按结构算子（分组条件/分歧表达/正交残差/条件门控/分段状态/乘法）单扫描 SQL 统计历史过线率（verdict ∈ POSITIVE_VERDICTS，min_total=60、样本 <20 的算子省略），注入 context_for——与证据块同享保底预留（预算 <4× 块长自动退出），矫正交互模板惯性（记忆实证：交互 ≈3× 无交互基线 4.6%，CS_GROUP_RANK 14.6% 最高，MULTIPLY 仅 5.7% 且 85% 被拒；候选池融合因子 5/7 曾是同一 DIVERGENCE_RANK 两元素模板）。融合教学三处（`prompt/modules/facet_focus.py` / `agentscope_run` 融合指令 / `_diversity_block`）同步修订：MULTIPLY 全部改禁用警示（默认 spec 已拦截 multiplication），新增链式组合契约说明——多结构表达式只传**一个** interaction 契约、`interaction_type` 填表达式**末位**结构算子（`interactions.py` 按末位算子定义输出结构的口径）。
- **v2 模式层已下线（2026-08-30）**：`memory_patterns` 表停止注入（`context_for` 不再调 `_pattern_block`），其规则蒸馏（同族饱和 forbid / 族内 |IC|≥0.02 recommend / 全局 insight）由 `distill_batch_experience` 迁到 v3 `memory_experience`（同 (kind, family) 去重 + occurrence_count 累加，recommend 文本标注 IC 方向）。旧表数据保留只读，UI 不展示。
- **显式父本协议**：A/B/C 变异轨的 eval/submit 调用必须传 `parent_factor` + `edit_note`（`edit=<motif> <参数变化>`）；工具结果带 `memory_advisory` 硬提醒（指纹死路 attempts≥3（`memory_policy.dead_end_min_attempts` 可配，2026-09-23 由 2 上调）/ **指纹正向重复 prior_result**（2026-09-03：同结构指纹曾有 promising/入库条目 → 附历史因子名/verdict/IC/未晋升原因，堵"promising 结构换名重测"的重复劳动盲区——实测历史 206 组重复指纹中 33 组含正向 verdict 完全无提醒）/ 意向编辑 APV veto），默认只提示，`memory_policy.hard_block_duplicates=true` 时仅指纹死路升级为拦截（prior_result 永不拦截）。
- 检索：BM25 + 族亲和 + 算子重叠 + verdict/recency；证据块 40% 正向配额 + (family≤2, 指纹=1) 去重。**正向跨族保底（2026-09-01）**：正向配额约 2/3 无条件取全库最优正向因子（validated/production 优先，按 |IC|，同族轮转保证跨族，`_guaranteed_positives`），BM25 只补剩余——通用 query 打不中 FTS 时正向通道不断供；`dynamic_retrieve_limit` 默认 6→8。
- 饱和度（2026-09-01 修正）：`min(1, min(n_promising,8)/40 + n_candidate/5 + n_validated/3)`——train_passed（2026-10-01 由 promising 改名；仅过训练集海选，多数未晋升）只轻度计入（≤0.2），拥挤度以真实幸存者（candidate/validated/production）为主；注入时除拥挤族警告外，同时给出"出过正信号且未拥挤"的族作定向深挖建议，不再只说"去别处"。
- **数据面聚焦（跨面融合，2026-09-02）**：前端 run 表单多选数据面（价量/量能/筹码/拥挤/基本面/股东/事件/资金，与 `expressions.FACET_DEFS` 对齐）→ StartRequest `focus_facets` → 子进程 `--focus-facets`。选中后：① 用户消息追加融合指令（融合算子模式 2026-09-06 修订：CS_GROUP_RANK / DIVERGENCE_RANK / GATED_SIGNAL / CS_RESIDUALIZE / DIVIDE + 链式组合契约；MULTIPLY 禁用警示）；② 每轮记忆注入附"数据面聚焦指令"块（含未触面强制提醒——聚焦面未出现在任何尝试中时要求本轮必须给出触及它的表达式）；③ 选中非价量面时自动加载对应列族（不加 --no-fundamentals）。`_diversity_block` 同步启用：近 8 次尝试面覆盖 <3 时注入面覆盖警告 + 融合模式示例（聚焦生效时抑制——用户显式聚焦优先于自动多样性引导）；蒸馏对跨 ≥2 面的成功经验打"跨面融合"标签。**单选一个面时退化为单面聚焦指令**：不要求融合、不要求 _x_ 命名、单面因子正常提交，仅要求表达式触及该面（用户消息块与每轮提醒两处同口径，2026-09-02）。
- 正向 verdict 鼓励邻域探索；负向 verdict 防止重复无效路径。
- **融合族键与 facets 亲和（2026-09-03，data_version=4）**：`classify_family_ex` 对**跨数据源组**（`FACET_GROUPS`：行情组=价量/量能/筹码/拥挤、基本面组=基本面/股东、事件资金组=事件/资金）触及 ≥2 面的表达式返回面对组合键（如 `价量面×基本面`，面名按 FACET_DEFS 稳定排序）；同组多面与单面保持旧 `_FAMILY_RULES` 细粒度口径（防"假融合"——`$close+$volume`、`$float_cap` 这类常见组合不算融合，`$float_cap` 已移出股东面识别键）。`memory_entries.facets_json` 记录全量面集合，检索亲和两档：family 精确命中 +0.3、facets 有交集 +0.15（单面 query ↔ 融合条目的跨召回）。registry（候选/正式）入库写 `facets`/`is_fusion`/`family`/`eval_label`。facet_focus 为 system prompt 插件模块（ORDER=135，含信号族白名单豁免声明——`research_policy_prompt` 经 extra_instructions 注入的白名单与融合指令矛盾，聚焦生效时显式解禁）。
- **Prompt 矛盾审计与修复（2026-09-19，v2.0）**：完整修复 spec 见 `docs/specs/alphaagent_prompt_contradictions_repair_spec_v2.md`（不进 git，工作区参考）。审计基准：9-18 消融基线 prompt（40617 字静态）+ 9-19 凌晨 run `30ebfd3802d0`（10 轮 2423 评估 28 提交 **0 入库**，gemini-3.8-flash-high）。共 14 点矛盾分两层：
  - **静态层 8 点**（prompt 模块内部数字/规则打架）：S1 并发度 12~20 vs 底层 8；S2 MULTIPLY 禁用 vs 直用；S3 正交门槛 0.7/0.5/0.4 阶梯未分层；S4 换手率 0.4 vs 引擎 0.50（**实测 19/28 提交因换手率被拒，头号杀手**）；S5 变异轨 A/C 教微调 vs AST 熔断器拦微调；S6 1d label + weekly 调仓错配（**0 入库根因**——1d 快信号 weekly 调仓换手率必然 >0.50）；S7 负 IC 承认 vs 回测做多方向冲突；S8 模块裁剪悬空引用。
  - **动态层 6 点**（记忆系统多注入源信号打架，v1.0 未覆盖）：D1 禁止段 vs 推荐段同框（推荐段赢——给了具体父本+编辑类型+置信度，可操作性强压过签名级 DO NOT）；**D2 饱和度警告 vs 编辑推荐同框（核心根因）**——`recommend_edits`（`retrieval.py:618-658`）主路径只查 APV veto，**不查族级饱和度**，vwap 族饱和度 >0.4 触发警告但推荐段照样推 vwap 族；D3 产出率块 vs 编辑推荐同框（`recommend_edits` 排序键 `delta*conf` 不看族级历史过线率）；D4 重复表达式警告 vs 推荐段同框；D5 禁止段签名级，换算子/换字段即绕过；D6 D 轨"新族开拓" vs 推荐段"旧族深耕"。
  - **修复优先级**：P0（S4 换手率统一 + S6 1d/weekly 协调，直接 0 入库根因）→ P1（D2 `recommend_edits` 加族级饱和度门控 + D1/D3/D4 记忆系统自洽）→ P2（S1/S2/S3/S5/S7/S8 静态文本对齐）→ P3（D5/D6 长尾 + 可选 evaluate_factor 族级饱和硬拦截）。验收：test run 换手率拒绝比例 68%→<30%、`recommend_edits` 推荐族饱和度 >0.4 占比 =0%、vwap 族评估占比 34%→<20%。

## 评估档位自动推断 + 调仓频率（2026-09-03，方案 B）

前端"日线技术/基本面"模式下拉退役，档位由数据面多选自动推断：
- `core.research_modes.infer_research_mode(focus_facets, rebalance_freq)`：勾选基本面/股东面 → fundamental 档
  （label_20d + 松门槛 + monthly 门禁）；其余/未选 → technical 档（label_1d + 严门槛 + **daily** 门禁，
  2026-10-04 起与 label 对齐，见下节）；显式给 rebalance_freq 时价量面走
  `technical_{daily,weekly,monthly}` 对齐子档（label 5d/20d 跟随）。纯函数，前后端同口径。
- 前端 composer 显示推断档位徽章（`inferredModeLabel`），勾面变化经 `syncInferredMode()` 自动
  切换 spec（复用 `switchResearchMode` 链路：门槛弹窗/保存门槛/label_col 跟随全部保留）。
- `StartRequest.rebalance_freq`（daily/weekly/monthly，可空）：用户显式指定时写入
  `spec.delivery_policy.production.engine_gate.freq`，并把档位落到 label 对齐的子档；
  若与档位 label 矛盾（如在 1d/20d label 档上选 weekly）→ **422 拒绝**（fail-closed）。
  `DeliveryCriteria.to_prompt_text` 渲染 "rebalance_freq 必传" 指令约束 LLM；`allowed_freqs` 白名单生效。
  前端 composer 新增 "调仓" 选择器（自动/日/周/月）。
- `research_mode` 参数兼容保留：显式传入优先于自动推断（历史调用方/脚本不受影响）。
- 因子库页模式页签退役（`FactorLibrary.vue`），由 facet 筛选 chips（全部/八面/融合）取代；
  ML 组合训练的 `collect_factor_entries` 按解析后库路径去重，避免大库下重复枚举。

## label 口径 vs 调仓频率：**强制一致**（2026-10-04 用户定调，推翻 2026-09-30 解耦决策）

**结论（用户原话）：「调仓频率是多少，label 就应该多少」。** label 持有期必须等于
`engine_gate.freq` 持有期：**daily↔1d、weekly↔5d、monthly↔20d**；不一致 **fail-closed 拒绝启动**。

- **配置中心**：`research_spec.DEFAULT_RESEARCH_SPEC["label_freq_consistency"]`
  = `{"enforce": true, "freq_hold_days": {...}}`（`research_spec.py` 顶部有取值理由）。
- **校验点（三处，同一函数 `research_spec.ensure_label_freq_consistency`）**：
  ① `build_run_research_spec` / `normalize_research_spec`（构建期，以 label 为准把 freq 收口到
  对应档位并把 `allowed_freqs` 收成单值）；② `POST /api/alphaagent/runs`（用户显式覆盖 freq
  之后再校验，否则覆盖就成了绕过通道 → 422）；③ `backend.alphaagent_service.start_run`
  （**所有** API/脚本/monitor 启动路径的唯一收口 → `RunAdmissionError`）。
- **档位现状**：`technical` / `report` 已从 `label_1d + weekly` 收成 **`label_1d + daily`**
  （`allowed_freqs=["daily"]`）；`fundamental`(20d+monthly) / `technical_daily`(1d+daily) /
  `technical_weekly`(5d+weekly) / `technical_monthly`(20d+monthly) 本来就是对齐的。
- **慢档门槛标定（2026-10-07）**：`technical_weekly` / `technical_monthly` 此前**只对齐了
  label/freq，没有自己的门槛**——继承 1d 标定的 `min_abs_ic=0.02 / min_icir=0.28 /
  min_val_abs_ic=0.015` 等。而 label 持有期会系统性改变同一因子的 IC/ICIR 量级（配对实测：
  候选池 42 因子 × label_1d/5d/20d，train 2020-2022，n=41；5d/1d 中位放大 1.776/1.800、
  20d/1d 放大 2.865/3.032）⇒ 沿用 1d 线会让慢档门近乎失效。
  标定采用**双锚**：①**文献绝对量级**（日频 Rank-IC ≥0.02 有筛选价值、0.03–0.05 可用；月度
  IC 0.02–0.06 正常、0.05–0.10 属"很好"；ICIR 有效线 0.3；文献实测月度 PE TTM 0.0529/0.6995）；
  ②**与主档选择性持平**（同一实测池过线率均 **36.6%**）。
  落地值：`technical_weekly` candidate/evaluation **0.030 / 0.450**（production 0.0375 / 0.4821）；
  `technical_monthly` **0.053 / 0.650**（production 0.0663 / 0.6964）；`min_val_abs_ic` 按本档 IC
  线/主档 0.02 的倍数（0.0225 / 0.0398）。与 label 尺度无关的项（val 保留比 / coverage /
  cs_autocorr / max_abs_corr / engine_gate 年化门）**刻意不动**。明细见
  `docs/review/tech_slow_tier_threshold_calibration_20261007.md` 与 `core/research_modes.py`
  两档注释，不变量由 `tests/test_slow_tier_thresholds.py` 锁定；重标定脚本
  `scripts/calibrate_slow_tier_thresholds.py`。
- **fundamental 档同批重锚（2026-10-07）**：它同为 label_20d + monthly，却沿用 1d 尺度的
  `0.020/0.28/0.012`（2026-09-11"与 technical 同值量纲锚"）⇒ 实测基本面族近似子集（n=9）
  过线率 **77.8%**，门形同虚设。按同法双锚重锚到 **0.035 / 0.45 / 0.021**（文献月度基本面
  量级 PE 0.0529/0.6995、PB 0.0557/0.4888、ROE 0.0172/0.2277；该族过线率 44.4%），三层
  同比例（IC ×1.75、ICIR ×1.607）重锚，保持 production==candidate 与 val 保留比 0.65/0.70。
- **判定侧档位感知修复（2026-10-07）**：`tools/_dispatch._near_miss_verdict` 早已按档位取
  `candidate_overrides["min_abs_ic"]`，而 `memory/schema.py:_classify` 只读全局
  `DEFAULT_RESEARCH_SPEC` 的 `min_train_abs_ic`（0.02），且 `memory/ingestion.py` 到落库阶段才把
  `research_mode` 写进 metrics ⇒ 慢档/基本面档会把低于本档线的 IC 标成 `train_passed`。现已在
  `record` 调用 `_classify` 前 `setdefault` 档位元数据、`_classify` 按档取线（未知档位回落全局）。
- **换手门改用调仓口径**：`delivery_checker.stage_one_stats` 取
  `quantile_portfolio.avg_rebalance_side_turnover`（每 hold 日调一次的真实换手），
  旧记录回落 `avg_daily_side_turnover`；daily↔1d 时两者相等（历史行为不变）。
- **为什么推翻解耦**（2026-10-04 实测）：旧组合里 stage_one 拿"日频信号抖动"撞 weekly 档阈值
  （0.65），而 `hold=label 天数=1` → 组合其实每天在换（实测 `n_rebalances=1615`、
  `avg_rebalance_side_turnover == avg_daily_side_turnover`）→ "周频摊薄成本"的折算前提不成立，
  1.48 这类超门值既不是日频事实也不是周频事实。
- **已知代价（接受）**：1d 快信号不再有"周频折让"，换手硬门按 daily 档（0.50）执行；
  engine_gate 对这两个档位也按 daily 真实回测（成本更高）。要过门只能把信号做慢。
- **⚠ 仍成立的历史警告**：**不要单独把 label 抬到 5d/20d 去凑 freq**。`min_abs_ic` /
  `min_train_icir` / `reproduce_min_*` 等门槛按 **1d 尺度**标定，换 label 会一并失真——
  必须同批重标定门槛（属独立改动）。所以本次选择"保 1d 研究口径、把 freq 收到 daily"。
- 历史（2026-09-30 决策，已废弃）：曾把两个旋钮视为"两把尺子"并保留落差，靠
  `min_cs_autocorr` + `turnover_thresholds_by_freq` 两个代理门缝住；该设计已按上表推翻。


## REST API

| 方法 | 路径 | 功能 |
|---|---|---|
| POST | `/api/alphaagent/runs` | 启动新挖掘（focus_facets 自动定档；rebalance_freq 覆盖门禁频率） |
| GET | `/api/alphaagent/runs` | 列出所有 runs |
| GET | `/api/alphaagent/runs/{id}` | 查看 run 详情（事件轨迹尾部） |
| POST | `/api/alphaagent/runs/{id}/stop` | 终止 |
| POST | `/api/alphaagent/runs/{id}/messages` | 向运行中的进程注入消息 |
| POST | `/api/alphaagent/runs/{id}/continue` | 已结束的 run 继续（fork 新进程） |
| POST | `/api/alphaagent/runs/{id}/branch` | 分支新 run |
| POST | `/api/alphaagent/runs/{id}/rename` | 重命名 run |
| POST | `/api/alphaagent/runs/{id}/archive` | 归档/取消归档 run |
| POST | `/api/alphaagent/runs/{id}/pin` | 置顶 run |
| DELETE | `/api/alphaagent/runs/{id}` | 删除 run |
| DELETE | `/api/alphaagent/runs/archived` | 清理全部已归档 run |
| GET | `/api/alphaagent/runs/{id}/events` | SSE 事件流（实时轨迹） |
| GET | `/api/alphaagent/research-memory` | 查看研究记忆 |
| GET | `/api/alphaagent/research-memory/layers` | 记忆分层明细：SSPM cells（含 Eq.7 置信度 + 注入门控）+ 经验层 |
| DELETE | `/api/alphaagent/research-memory/{entry_id}` | 删除单条研究记忆 |
| GET | `/api/alphaagent/research-modes` | 档位选项（label/推荐 label/默认消息；档位现由 focus_facets 自动推断，前端仅用于门槛弹窗） |
| GET | `/api/alphaagent/research-spec/default` | 当前模式生效研究策略（默认+保存覆盖） |
| GET | `/api/alphaagent/research-specs/{mode}` | 模式门槛文件三视图：defaults/overrides/effective |
| PUT | `/api/alphaagent/research-specs/{mode}` | 保存模式门槛（diff 出增量覆盖，全链路生效） |
| DELETE | `/api/alphaagent/research-specs/{mode}` | 删除门槛文件，恢复注册表默认 |
| GET | `/api/alphaagent/evaluation-capabilities` | 可用评估插件和 profiles |
| POST | `/api/alphaagent/eval-factor` | 单因子评估（因子实验室） |
| GET | `/api/alphaagent/factors` | 列因子（library=production/candidate；facet=数据面筛选，"融合" 过滤融合因子） |
| GET | `/api/alphaagent/factors/{id}` | 因子详情 |
| POST | `/api/alphaagent/factors` | 保存因子 |
| DELETE | `/api/alphaagent/factors/{id}` | 删除因子 |
| POST | `/api/alphaagent/backtest-factor` | 因子回测 |
| GET | `/api/alphaagent/session-cache/stats` | 会话缓存统计 |
| POST | `/api/alphaagent/session-cache/evict` | 清空会话缓存 |
| GET | `/api/alphaagent/logs` | 运行日志 |
| GET | `/api/alphaagent/logs/tail` | 日志尾部/流式 |

## 关键设计决策

- **盲测段锁定（`scripts/blind_test_factors.py`）**：2025-01-01 起为锁定盲测段。**隔离靠"按 split 切片"，而非"panel 里没有 test 数据"**——`StockEvalContext.coverage_range()` 取 train ∪ val ∪ test 并集作为 panel 加载范围（实测 `('2020-01-01', 数据最新)`）；真正受限的是评估路径的 `get_split_panel(split)` 切片与 `_MINING_ALLOWED_PROFILES` 白名单（`split=full` 口径的 profile 挖掘期不可用）。**2026-09-24 更正**：原文"挖掘会话面板 coverage = `train_start~val_end`（2020~2024）"与实现不符，该错误描述曾掩盖四条旁路通道（正交召回在含 test 的 panel 上随机采样 → 约 25% 锚点落盲测段 / 候选池正交全量逐日含 test / zoo 采样行 14.3%（85678/100000）在 test 段且是 stage_one 判决依据 / research memory 证据块每轮注入明文 `test_ic`），已由分支 `fix/blind-segment-isolation`（commit `4c31407`）修复：新增 `StockEvalContext.visible_range()`（= train ∪ val）为唯一真源，凡回流 LLM 或参与判决的相似度与统计计算均切到该区间；回归测试 `tests/test_blind_segment_isolation.py`（8 例，含零交集断言），详见 `docs/specs/alphaagent_mine_precheck_spec.md` §0。`blind_test_factors.py` 对已定稿因子做一次性离线重测（复用 `compute_ingest_metrics` 同口径），结果只写 `artifacts/alphaagent/blind_test/<run_ts>/report.json`、不回流任何门槛。**频繁重测会把盲测段重新烧掉（多重检验），克制使用频率**。train/val 双段在筛选中被反复使用、非真正 held-out——盲测段是唯一的诚实样本外。窗口重映射记录：2026-08-29 由 train 2018~2022 / val 2023~2025 / test 2026 起 调整为 train 2020~2022 / val 2023~2024 / test 2025 起（train 收近 3 年防因子衰减，test 段约 20 个月更足）。
- **panel 实时构建而非预构建**：`cne://` 标识触发 adapter 从 CNE 数据湖按日期范围拉取，避免维护大 parquet 文件。首次加载约 30-60 秒，之后驻内存复用。
- **因子注册表单一来源（`core/factor_registry.py`）**：全部引擎因子（量价/基金/财务/动态）的元数据收口在 `FACTORS` 表；`core/composites.FACTOR_OPTIONS` 由它派生（组合编辑器清单），`strategies.registry.validate_registry_factors()` 校验策略引用。加因子 = 注册表加一行 + `build_factor_frames` 实现计算。
- **策略定义统一模型（`core/strategy_types.py`）**：`StrategyDefinition` 收敛注册表/配置池/归档三源，`resolve_strategy_def()` 统一解析（保留旧 `resolve_strategy()` dict 兼容），`fingerprint()` 用于策略去重/冲突检测。DSL 因子经 `from_dsl_factor()` 动态构造（不回填策略池）。
- **分数矩阵统一转换（`core/score_matrix.py`）**：AlphaAgent DSL 与 qweave 研究产出喂回测引擎的 date×code 矩阵统一走 `scores_to_engine_matrix()`，不再各自实现 pivot/unstack+去后缀。
- **面板口径统一（`core/panel_schema.py`）**：alpha panel（MultiIndex/千元/%）与引擎面板（长表/元/比例）的列/单位/索引契约 + `alpha_panel_to_engine_frame()` 公共转换，消灭单位魔数双写。
- **每模式门槛文件（`research_spec.py` 持久化）**：`artifacts/alphaagent/research_specs/<mode>.json` 存 `default_research_spec` 的**增量覆盖**（diff，不含派生的 evaluation_profiles）。`effective_research_spec(mode)` = 注册表默认 + 保存覆盖；运行口径 `build_run_research_spec(spec)` = 默认 < 保存覆盖 < 显式 spec，CLI/Web/晋升全链路统一走它。前端编辑保存经 `PUT /api/alphaagent/research-specs/{mode}`，改一处门槛全链路生效。
- **会话域复用**：submit 全程在挖掘会话驻内存 panel 上评估，不再全量重载因子库域 panel（避免 OOM）。
- **JSONL 轨迹持久化**：每次 run 的所有事件写入 `logs/factor_mining/ui/<run_id>/run_*.jsonl`，FastAPI 重启后可从磁盘恢复历史 run。
- **DSL 编译为 Python**：多行表达式先赋值中间变量，最后一行输出因子值；支持 `$列名` 引用 panel 列。
- **三层加速后端**：C++ > Numba JIT > 纯 Python，自动检测降级。
- **Codex provider**：从 `~/.codex/config.toml` 读取模型配置和 token，无需额外 .env 配置。
- **评估用分位数回测**：A股不能做空，因子评估用 quantile_portfolio（纯多头口径）替代 topn_portfolio。
- **批量脚本 OOM 规避**：独立脚本加载 CNE panel 时直接读 `stock_daily_wide` 原始 parquet（限定日期+最小列集），不用 `load_panel_from_cne()` 全量加载。
- **慢算子按品种并行层（`accel.py` boundaries + prange）**：`PRICE_GAP_*` / `CHIP_*` / `CROWD_*` / `WICK_EFFICIENCY` / `VOLUME_CLOCK_VPIN` / `MUTUAL_INFO_LAG` 等 per-instrument 滚动算子，先由 `ops_kit.instrument_group_order` 把面板数组按品种稳定重排成连续区间（消除 groupby/reindex/get_indexer 开销），再由 `@njit(parallel=True)` 边界内核按品种多核并行；PRICE_GAP 状态机已整体 Numba 化（原纯 Python 逐 bar 循环）。数值与旧串行路径逐位一致（含 NaN/乱序面板）；`ALPHA_DSL_BOUNDARIES_PARALLEL=0` 禁用并行，Numba 缺失时自动回落旧逐品种路径。新算子照此模板实现（范例：`WICK_EFFICIENCY`）。
- **挖掘并行评估**：`StockEvalService._eval_semaphore` 限制并发 train/val 评估数（StartRequest `max_parallel_eval` 默认 6；CLI 路径 `MAX_PARALLEL_EVAL` 环境变量或 `resolve_max_parallel_eval` 默认 **6**（2026-09-06 从 1 上调——对比实验实测 CLI 串行时排队等待占墙钟 ~50%），`--max-tool-workers` 默认 8）。tool_calls 由 `ThreadPoolExecutor` 并行分发；**并发瓶颈在诊断插件的 GIL 段**（numba nogil 内核可真多核，但 fmb/组合回测的 pandas/numpy 段串行化，全量 profile 多车道吞吐天花板 ≈0.42 eval/s，实测 NUMBA_NUM_THREADS 调整无效），扩车道收益依赖两段式短路削减 GIL 工作量。挖掘路径 `service._run_one` 以 `include_charts=False` 调引擎（逐日 IC/十分位多空/月度分解/IC 衰减曲线/IC 直方图等图表数据仅因子实验室 `eval_profile` 生成）；引擎计时见结果 `timing_ms`（dsl_eval/transforms/metrics 三段）。IC 衰减曲线（`metrics/decay.py`）：close→close 口径多 horizon RankIC（horizon 按持有期分档 1-20/1-40/5-60），前瞻收益优先复用 panel 预置 `label_{N}d_close_to_close` 列、缺失的在全量 session.panel 上现算并挂 `session._ic_decay_label_cache` 跨评估复用；ICIR 与 `cs_ic_summary` 同口径按持有期去重叠；载荷非有限值一律转 None（Starlette `allow_nan=False`）。
- **两段式海选 + 日切片缓存（2026-09-06，吞吐改造）**：① `eval_train` 两段式——第一段跑 `train_screen_lite`（仅 `cross_sectional_core`，规则与 train_screen 同源），过线才跑全量 `train_screen` 补齐 fmb/组合回测/月度稳健性等诊断件；未过线响应带 `screen_stage="lite"` + `screen_rules` + `skipped_diagnostics`，`decile_mean_label` 仍在 core 内故**预测对账不受影响**；引擎零改动，`ALPHA_EVAL_LITE=0` 退回全量。基准：全量 4 车道 0.42 eval/s → 两段式 6 车道 **2.29 eval/s**（对比 run 实测基线 0.29）。② `_day_slices` 结果按索引对象身份缓存（`metrics/__init__.py` 弱引用 + `is` 校验防 id 复用误命中、LRU 16、数组 write=False、override 注入点优先）——每次评估 ~15 处调用省去重复 O(n) 单调扫描。③ label float64 身份缓存（`metrics/_label_cache.py`，键=索引身份+列名，LRU 4）：一次评估内 ic/rank_ic/decile/fmb/portfolio 各自整体转 float64（5.7M 行 ≈46MB memcpy/次，纯 GIL 段）改为共享只读数组。基准汇总（6 车道）：全量 0.39 → 两段式 2.29 → +label 缓存 **2.61 eval/s**。跨仓库 DSL 语义差异备忘：原版 AlphaAgent 的 `CS_WINSORIZE(x, 1, 99)` 用百分数，quant_ui 用 0~1 分数 `CS_WINSORIZE(x, 0.01, 0.99)`。测试 `tests/test_mining_eval_throughput.py`。
- **Prompt 分阶段裁剪 + 上下文压缩（2026-09-06 建立，2026-09-22 按 PHASES 实测同步）**：system prompt 按挖掘阶段动态装配（模块声明 `PHASES`，full=全量 36.9K chars）。explore 阶段现裁 2 模块——multi_period（仅 deepen 起回注）、delivery_submission（仅 deliver 回注）；早期「裁 6 模块」方案已回撤：tool_examples 因工具失败率高 2026-09-12 回退常驻，neutralization_guide/ic_robustness/data_calibration 亦全阶段常驻。实测 run `bc0090d8255b`：explore 33.4K → deepen 36.3K → deliver/full 36.9K chars；deepen 起逐步回注。对话历史压缩走 agentscope 内置 `ContextConfig`（每次 reasoning 前自动检查）：`trigger_ratio` 0.75→**0.5**（128K 窗口 → ~64K tokens 触发，此前 96K 太晚、10 轮 run 的历史 tool 结果把 input 推到 50K+ tokens）——压缩 = LLM 结构化摘要（任务/状态/发现/下一步）替换旧段，旧段 offload 工作区文件；被压段的浓缩信息由研究记忆层每轮注入兜底，决策无损失。`tool_result_limit=5000` tokens/条限单条 tool 结果爆发。
- **逐日统计 numba 内核（`factor/metrics_fast.py`）**：逐日 Pearson/Rank IC（含平均秩）、十分组 label 均值、市值中性残差、截面 zscore/winsorize 全部下沉为 `@njit(nogil=True, cache=True)` 内核（GIL 释放 → 多评估并发真多核）。Rank 内核注意两点坑：平均秩必须经 argsort 映射回**原位置**；秩均值 (n+1)/2 非零，相关系数必须中心化。开发期教训：**编辑 numba 内核源码后若同秒重编译，磁盘缓存可能陈旧导致错值甚至 access violation**——部署/调试异常时先删模块 `__pycache__`。
- **评估 metrics 快路径（`factor/metrics.py`）**：逐日 IC / Rank IC / lag1 自相关 / 十分位分组 / 分位组合 / 市值中性化全部改为「datetime 连续区间切片（`_day_slices`，面板按 datetime 排序时 O(1) 取每日切片，消除逐日 `groupby+xs` 全表查找）」；`pd.qcut` 换成数值等价的快速等频分箱 `_fast_equal_freq_codes`（分位边界浮点重合时自动回落 qcut，保证语义一致）；`mls_fmb` 与 `long_short_portfolio` 经 `context.cache` 共享同一份每日十分位结果。回落路径保留（未排序面板/测试注入 `_day_slices_override`）。门禁：`tests/test_metrics_fastpaths.py`（快慢路径输出一致 + 分箱等价对照）。
- **L1 深度曲线 + 可成交域透镜（2026-09-11，`factor/metrics/portfolio.py` + `factor/metrics/tradable.py`）**：诊断「统计口径好、引擎口径差」的两个结构性来源，**只做测量不加门槛**（`ProductionCriteria` 已明确删除过毛值十分组类门槛，组合可行性裁决归 engine_gate；深度阈值需先积累分布）。① 深度曲线：`quantile_portfolio_metrics(depth_ks=(5,10,20,50,100))` 在同一逐日循环内（同一 argsort 服务全部 k）按调仓节奏累计 top-k 等权序列，输出 `depth_curve`（每深度 gross/net 超额年化、sharpe、mdd、换手、成本 pp）+ `Q{n}` 参考行（直接取主口径序列，与 `top_group_*` 逐位一致）；`depth_ks=None` 默认关、既有键零变化。② 可成交域透镜：`metrics/tradable.py` 按引擎同源三规则收窄候选池——可负担（执行日=信号日次日的**不复权 open** × lot_size ≤ 等权预算 capital/target_count，引擎实测 101/103 拒单是「现金不足/预算过小」而非涨停）、次日非涨停/停牌（复用 `core.limit.build_limit_flags`）、am20 ≥ 流动性下限（宽表 rolling，NaN 视为不可入选）；掩码按（索引身份+参数）weakref 缓存（首建 ~1.4s，复用 0.1ms），透镜走 `depth_only=True` 第二路调用（~0.3s），主口径零影响。实测（val 2023-2024，`funda_ocfps_x_price_blend3p_cd5_ema3`）：同一因子同一标签零成本，超额年化 top5 −11.7% → Q10 +7.3%（**19pp 深度损耗**），可成交域 top5 进一步恶化到 −21.5%（极端尾部 alpha 集中在买不起的票上）。逐位回归门禁：`tests/test_depth_curve.py`（默认键零变化 + Q 参考行一致性 + HEAD vs 新版快/慢双路径 24 键逐位一致）。prompt 侧 `behavior_rules` 新增 rule 2b（深度曲线阅读：top-k 相对 Q10 崩 = 改结构而非调参；`depth_curve_tradable` 更差 = 可成交性损耗，降头部集中度或放弃；两曲线都稳才值得 top_pct 小组合交付；最终裁决仍是 engine_gate），黄金基线 4 份同步重生成（各 +540 chars，full 34.6K→35.2K）。
- **慢算子三重门禁**：① 静态拦截 `scripts/check_dsl_slow_patterns.py`（AST 扫描 operators.py，新增逐品种 pandas 循环/纯 Python 逐 bar 循环即失败，存量白名单只减不增；由 `tests/test_dsl_slow_patterns.py` 挂进 pytest）；② 性能门禁 `tests/test_dsl_operator_perf.py`（720k 行标准面板上 9 个优化算子的耗时预算断言 + 快路径接线检查，防并行层静默退化）；③ 一致性门禁 `tests/test_dsl_operator_consistency.py`（35 用例，快路径 vs 回落路径逐位一致，覆盖 NaN/跳空/零量/乱序面板/动态窗）。运行时慢算子发现走 `dsl-monitor` API（top_k 耗时榜）。

## 聚宽策略兼容层（JQ Shim）——代码面板跑聚宽策略

对标聚宽策略 API 的本地回测层（代码面板 /api/code/jq/run、模拟盘事件策略共用）。实现位于 `core/event_engine/jq/`，**按聚宽官方文档类别拆分**：

```
core/event_engine/jq/
├── api/                # 命名空间装配（每模块 install(ns, rt)，对应聚宽文档类别）
│   ├── framework.py    #   策略程序架构/运行时间: run_daily/run_weekly/run_monthly/unschedule_all
│   ├── settings.py     #   策略设置函数: set_option/set_benchmark/set_slippage/set_order_cost
│   ├── trading.py      #   交易函数: order 四件套/cancel_order/get_orders/get_trades
│   ├── data_api.py     #   数据获取函数: get_price/history/get_fundamentals/get_industry/
│   │                   #     get_extras/get_billboard_list/get_index_stocks/交易日历/normalize_code
│   ├── portfolio.py    #   策略组合操作: set_subportfolios/transfer_cash（单账户兼容）
│   └── misc.py         #   其他函数: jqdata/jqfactor 模块桩、display、旧版 pandas/time.clock exec 兼容
├── objects.py          # 对象: Context/Portfolio/Position/Order/Trade 壳视图
├── query.py            # get_fundamentals 的 query DSL（valuation/income 列过滤）
├── runtime.py          # 编排: exec 用户代码 → 生命周期（before_trading_start→定时任务→flush→
│                       #   after_trading_end）→ 撮合对接；有状态数据实现（矩阵/截面/财务缓存）
├── entry.py            # run_jq_backtest 入口（干跑采集费率 → 引擎回测）
├── factor_bridge.py    # get_factor: AlphaAgent DSL 因子表达式在策略内调用
└── datalake/           # ★ 数据接入插件体系（见下节）
```

### 数据接入方式（datalake 插件体系）——加数据的标准流程

数据源三目录：`data/quant_dataset`（CNE 管理的 tushare wide 日线）、`data/pg_parquet`（CNE 注册外部财务/基本信息）、`CNEquity/.../_cnequity/curated`（CNE 原生 curated 表，46+ 数据集）。

**规则 1：CNE curated 已有的表 = 零代码**。启动时 `discover_cne_curated()` 把 curated 下全部目录批量注册为 `cne:<dataset>` 通用插件（裸名/带前缀等价：`load("dragon_tiger")` == `load("cne:dragon_tiger")`），全部即刻可 `datalake.load(name)` / `datalake.asof(name, date)`（点时：日期过滤 + 按 `entity_keys` 每实体取最新）。覆盖见 `jq_data.data_status()` 诊断表（行数/日期范围/内存）。**接入一个"已有 CNE 表"的新 API 不需要写任何插件文件**。

**规则 2：需要专属清洗/换算的表才写插件**。在 `core/event_engine/jq/datalake/plugins/` 新增非下划线 `.py`（与 alphaagent 数据插件同风格）：

```python
from core.event_engine.jq.datalake.base import JQDataPlugin

PLUGIN = JQDataPlugin(
    name="income",                    # 唯一 id；load() 缺省取模块级同名函数
    date_column="ann_date",           # 点时列（None=无日期概念）
    entity_keys=("code",),            # asof 去重键（每实体取最新一行）
    asof_fallback_first=False,        # 请求早于最早数据时回退首期（月度快照类才需要）
)

def load() -> pd.DataFrame:
    ...                               # 进程级缓存，kwargs 参与缓存键
```

已迁移专表插件：`stock_daily`（日线原始帧，按 start/end/prefixes 参数缓存）、`income`（利润表 ann_date 点时）、`index_bars`（CNE 指数日线）、`industry_members`（行业月度快照 sw+eastmoney，fallback 首期）、`fina_indicator`/`balancesheet`/`cashflow`（财务三表）。调用未注册名抛**带扩展指引**的 NotImplementedError（列出已注册名单+怎么加插件），不抛裸 KeyError。

**规则 3：API 层封装**。在 `api/<对应类别>.py` 加一个函数：内部 `datalake.load(...)`（或 `asof`）+ 单位/口径换算（元/千元/万元、前复权口径见 jq_data docstring）+ 点时裁剪（信号日为界，禁止未来数据）。参考实现 `get_billboard_list`（api/data_api.py，从 CNE dragon_tiger 到 API 共 ~25 行）。

**规则 4：`scripts/jq_repro/jq_data.py` 是门面**。保留常量别名（PG/QDATA/CNE_CURATED）与策略域构建器（load_panel/build_tables/fin_ok_matrix/ew_index，由 stock_daily 原始帧派生引擎面板）；paper.py、strategies/event/_runtime.py 等历史调用零改动。

撮合/点时语义注意：日线引擎信号 T-1 收盘 → T 开盘撮合（100 股整手、涨跌停/停牌拒单）；`history` 以信号日为界（NaN 前置补齐）；货基 ETF（511880/511990）为合成行情（年化 2% 直线）。验证脚本：`scripts/jq_repro/_test_datalake.py`（插件体系）、`_test_p0_api.py`（API 面）、`_validate_jq_compat.py`（小盘三正基线）、`_test_guojiu.py`/`_test_billboard.py`/`_test_research_api.py`（真实策略端到端）。API 迁移对照评估见 `docs/jq_compat_迁移评估.md`，聚宽官方文档快照 `docs/jq_api_snapshot/api_full.html`。
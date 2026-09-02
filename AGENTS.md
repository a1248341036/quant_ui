---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '5db1b93d-b4f4-407e-a844-6bf9ad1638dd'
  PropagateID: '5db1b93d-b4f4-407e-a844-6bf9ad1638dd'
  ReservedCode1: 'd099cc50-3195-4f93-a0ad-f1c9ac4f1143'
  ReservedCode2: 'd099cc50-3195-4f93-a0ad-f1c9ac4f1143'
---

# AlphaAgent — 项目架构与开发指南

## 定位

AlphaAgent 是 quant_ui 中的 **A 股日频多因子自主挖掘智能体**。通过 LLM 驱动的多轮对话循环，自动生成因子 DSL 表达式 → 训练集评估 → 验证集检验 → 去重审查 → 入库交付，形成闭环研究流程。

## 目录结构

```
alphaagent/
├── core/           # 共享类型、路径常量、配置加载
├── data/           # 数据层：panel 构建 + CNE 数据湖适配器（插件化）
│   └── adapters/
│       ├── cnequity.py       # 入口：panel_path == "cne://" 时走此 adapter
│       ├── registry.py       # 插件注册中心
│       └── plugins/          # 数据源插件（stock_daily_wide, fund_flow 等）
├── dsl/            # 因子表达式 DSL
│   ├── core/       # 解析器 + 算子库 + Numba/C++ JIT 加速后端
│   │   ├── parser.py         # 多行 DSL → Python 编译
│   │   ├── operators.py      # TS_*, CS_*, CHIP_* 等算子入口
│   │   ├── accel.py          # 向量化滚动算子（Numba JIT）
│   │   └── chip_daily.py     # 筹码分布算子
│   └── stock/      # 股票专用：增量计算、resample、缓存
├── factor/         # 因子管理层
│   ├── evaluation/ # 冻结 profile 评估引擎 + 插件 transforms/metrics/rules
│   │   ├── engine.py        # EvaluationEngine 主入口
│   │   ├── plugins.py       # 可用 transform / metric / rule 注册
│   │   └── profile.py       # EvaluationProfile 定义与默认 profiles
│   ├── mining/     # LLM 挖掘主循环
│   │   ├── loop.py          # OpenAI Chat Completions 多轮 tool_calls 循环
│   │   ├── tools.py         # evaluate_factor / submit_factor 工具定义
│   │   ├── prompts.py       # System Prompt（经济直觉 + 变异策略 + 正交预判）
│   │   ├── config.py        # MiningConfig 数据类
│   │   ├── session.py       # StockEvalSession：panel 驻内存 + split 懒缓存
│   │   ├── service.py       # StockEvalService：会话管理
│   │   ├── submit.py        # 因子入库（candidate → production 两阶段）
│   │   ├── research_spec.py # 版本化研究策略（search/eval/review/memory/delivery）
│   │   └── research_memory.py # BM25 持久化研究记忆（正/负证据）
│   └── zoo/        # FactorZoo：因子库存储 + 相似度索引
└── __init__.py

backend/
├── alphaagent_service.py    # FastAPI 服务层：AgentRun 子进程管理 + JSONL 轨迹
└── routers/alphaagent.py    # REST API: /api/alphaagent/*

scripts/
├── run_alphaagent.py        # CLI 入口：加载 Codex provider 配置并启动挖掘
├── alphaagent_factor_mining.py  # 上游 AgentScope 挖掘脚本（被 run_alphaagent 调用）
├── promote_candidates.py    # 候选池重放晋升：已有候选重新走两阶段链路
└── dedup_candidate_factors.py  # 候选池一次性去重

static/src/views/AlphaAgent.vue  # 前端页面：研究 / 因子实验室 / 因子库三个子标签

artifacts/alphaagent/
├── factorzoo/
│   ├── candidate_main/               # 统一候选池（2026-09-03 大库合并，原 4 库并入，因子带 facets 标签）
│   ├── production_main/              # 统一正式库（原 production_technical/fundamental 并入）
│   ├── <旧库名>.bak-unified-*/       # 大库合并前的备份（rename 保留，可回滚）
├── research_specs/<mode>.json        # 每模式门槛文件的增量覆盖（diff）
└── research_memory.db                # 跨会话持久化研究记忆（BM25）

logs/factor_mining/ui/        # 每次 Web run 的 JSONL 轨迹 + run_meta.json + steps.log 分步日志（按 run_id 分目录）
```

> **steps.log 分步日志**：挖掘 run 内每个关键步骤一行（`run_start / turn_start / evaluate /
> submit.stage_one_stats / submit.blind_test / submit.val_retention / submit.stage_one /
> submit.review / submit.candidate_stored / submit.stage_two / submit.engine_gate /
> submit.promoted / submit_result / memory.record / memory.advisory / memory_retrieve /
> memory_suggest / memory_form / memory_distill / run_end`），k=v 字段 + turn 号，
> 写 `<run_id>/steps.log` 并镜像 stdout（进 `console.log`）。实现 `alphaagent/factor/mining/runlog.py`，
> 由 `agentscope_run.setup_run_logger(log_dir)` 装配；日志异常全吞，绝不阻断挖掘。

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
| candidate（海选） | \|IC\| ≥ 0.02（2026-09-01 从 0.015 上调：0.015~0.02 区间因子经盲测/换手门禁几乎全灭）, \|ICIR\| > 0.25, coverage > 0.85, max_corr < 0.5, lag1 自相关 ≥ 0.18, val 保留比 ≥ 0.5 | candidate_main/（统一大库） |
| production | \|train IC\| ≥ 0.025, \|train ICIR\| ≥ 0.30, \|val IC\| ≥ 0.015, val 保留比 ≥ 0.60, winsorized 衰减 ≤ 0.10, max_corr < 0.4 | production_main/（统一大库） |

提交流程顺序：
1. **stage_one 统计门槛**（IC/ICIR/coverage/换手/val 保留比）→ 不通过拒绝
2. **正交性检查**（stage_one 统一查，不再只在 approve 后触发）→ 与正式库已有因子做截面相关，超过阈值拒绝
3. **review_hook**（LLM Reviewer 审核）→ **仅 reject 硬拦**（抄袭/经典暴露不进任何库）；revise/pending_review 不阻断晋升，仅记录意见
4. 入候选池（registry_only）
5. **stage_two 精筛**（双窗口口径）→ 不通过停在候选池
6. **engine_gate 回测门禁**（完整回测引擎净值裁决）→ 不通过停在候选池
7. 入正式库（canonical 对齐 + ingest）

**晋升链路关键语义（2026-08 修复）**：
- **stage_one / stage_two 的相似度只查正式库**，候选池内部冗余不卡正式库准入。
  候选池冗余由 `scripts/dedup_candidate_factors.py` 主动去重管理，而非让候选池因子
  互相挡死晋升（历史死锁：`gap_vwap_ens_wo3` 被候选池内相关 0.598 的 `vwap_close_dev_mom` 卡死）。
- **Reviewer 的 revise 是建议不是门槛**：与系统提示词"Reviewer 意见仅供参考改进方向，
  不阻断提交"一致；正式库准入的最终裁决是 stage_two 统计门槛 + engine_gate 净值回测。
- **engine_gate 是实盘可交易性裁决**：weekly 调仓、净超额年化 ≥3%、超额夏普 ≥0.5、
  回撤 ≤40%、持仓重叠 ≥50%、仓位利用率 ≥80%。统计 IC 高的因子若换手高（如日换手 69%、
  周重叠 5.6%），实盘净超额会转负 —— engine_gate 正确拦截"统计有效但实盘亏钱"的假因子。

### 6. 候选因子库管理

- 前端因子库页支持正式库/候选库切换，各有导出 JSON 按钮
- `scripts/dedup_candidate_factors.py`：一次性去重脚本，两两截面 Pearson 相关 ≥ 阈值的因子对删除冗余项
- 候选因子 registry 的 `similarity` 字段记录与已有因子的截面 Pearson 相关

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

三层结构（SQLite WAL + FTS5，单文件 `artifacts/alphaagent/research_memory.db`，`store_meta.data_version="4"` 幂等迁移）：
- **原始证据层** `memory_entries`：每次评估/提交一条，verdict 7 级 + 数据化结论 + 失败码/失效项 + 结构指纹 + 父子关系（`parent_origin`=explicit/implicit、`secondary_parent_id` crossover 双父、`intended_motif` 意向编辑）+ `facets_json` 数据面标签（v4 新增；老行读取时按表达式现算兜底）。
- **编辑统计层（SSPM）** `memory_cells`：键 = (family × motif × 父本质量桶)；残差 = 子代 IC − 同桶时间衰减基线（half-life 90d，无历史回退父本 IC，AlphaMemo Eq.4）；成败按 explicit（±1.0）/implicit（±0.5）加权分列；**无效尝试（报错）入账失败观测（权重 0.5）**。
- **经验层** `memory_experience`：成功模式（签名 + 模板 + 实例）/ 禁忌方向（典型相关 + 失效项）/ 洞察（入库率）。

校准与注入（AlphaMemo 论文口径）：
- 置信度 Eq.7：`c = n/(n+κ)·min(1,|μ|/(σ+ε))`，κ=8；APV 双门 Eq.11/16：`veto = c>τ_c ∧ π⁻>τ_v`（默认 0.35/0.80，`memory_policy.apv_tau_c/apv_tau_v` 可调）；正证据软推荐封顶"优先尝试"档。
- 注入门控矩阵（`retrieval._edit_prior_block`，阈值经 `memory_policy.edit_prior_*_conf` 可调）：硬推荐(s>0 ∧ c>0.7) / 软推荐(s>0 ∧ c>0.4) / 硬否决(f>0 ∧ c>0.7) / 软否决(f>0 ∧ c>0.3，低于推荐向以放行"一致失败"避坑证据) / 其余不注入；APV 双门(τ_c=0.35, τ_v=0.80)另在评估前 advisory 做 (family, motif) 聚合否决。2026-08-30 修正：旧文档写的"软否决 fail_rate≥0.6"与实现不符，实际口径全部基于 Eq.7 置信度分档。`memory_policy.max_inject_chars`（默认 2400）超限时核心块（经验、编辑先验）始终保留，次级块按 证据 > 饱和度 > 多样性 用剩余预算填充，所有截断在行边界。
- **v2 模式层已下线（2026-08-30）**：`memory_patterns` 表停止注入（`context_for` 不再调 `_pattern_block`），其规则蒸馏（同族饱和 forbid / 族内 |IC|≥0.02 recommend / 全局 insight）由 `distill_batch_experience` 迁到 v3 `memory_experience`（同 (kind, family) 去重 + occurrence_count 累加，recommend 文本标注 IC 方向）。旧表数据保留只读，UI 不展示。
- **显式父本协议**：A/B/C 变异轨的 eval/submit 调用必须传 `parent_factor` + `edit_note`（`edit=<motif> <参数变化>`）；工具结果带 `memory_advisory` 硬提醒（指纹死路 attempts≥2 / 意向编辑 APV veto），默认只提示，`memory_policy.hard_block_duplicates=true` 时拦截。
- 检索：BM25 + 族亲和 + 算子重叠 + verdict/recency；证据块 40% 正向配额 + (family≤2, 指纹=1) 去重。**正向跨族保底（2026-09-01）**：正向配额约 2/3 无条件取全库最优正向因子（validated/production 优先，按 |IC|，同族轮转保证跨族，`_guaranteed_positives`），BM25 只补剩余——通用 query 打不中 FTS 时正向通道不断供；`dynamic_retrieve_limit` 默认 6→8。
- 饱和度（2026-09-01 修正）：`min(1, min(n_promising,8)/40 + n_candidate/5 + n_validated/3)`——promising（仅过训练集海选，多数未晋升）只轻度计入（≤0.2），拥挤度以真实幸存者（candidate/validated/production）为主；注入时除拥挤族警告外，同时给出"出过正信号且未拥挤"的族作定向深挖建议，不再只说"去别处"。
- **数据面聚焦（跨面融合，2026-09-02）**：前端 run 表单多选数据面（价量/量能/筹码/拥挤/基本面/股东/事件/资金，与 `expressions.FACET_DEFS` 对齐）→ StartRequest `focus_facets` → 子进程 `--focus-facets`。选中后：① 用户消息追加融合指令（融合算子模式：MULTIPLY 门控 / DIVERGENCE_RANK 分歧 / CS_RESIDUALIZE 正交残差 / DIVIDE 比值）；② 每轮记忆注入附"数据面聚焦指令"块（含未触面强制提醒——聚焦面未出现在任何尝试中时要求本轮必须给出触及它的表达式）；③ 选中非价量面时自动加载对应列族（不加 --no-fundamentals）。`_diversity_block` 同步启用：近 8 次尝试面覆盖 <3 时注入面覆盖警告 + 融合模式示例（聚焦生效时抑制——用户显式聚焦优先于自动多样性引导）；蒸馏对跨 ≥2 面的成功经验打"跨面融合"标签。**单选一个面时退化为单面聚焦指令**：不要求融合、不要求 _x_ 命名、单面因子正常提交，仅要求表达式触及该面（用户消息块与每轮提醒两处同口径，2026-09-02）。
- 正向 verdict 鼓励邻域探索；负向 verdict 防止重复无效路径。
- **融合族键与 facets 亲和（2026-09-03，data_version=4）**：`classify_family_ex` 对**跨数据源组**（`FACET_GROUPS`：行情组=价量/量能/筹码/拥挤、基本面组=基本面/股东、事件资金组=事件/资金）触及 ≥2 面的表达式返回面对组合键（如 `价量面×基本面`，面名按 FACET_DEFS 稳定排序）；同组多面与单面保持旧 `_FAMILY_RULES` 细粒度口径（防"假融合"——`$close+$volume`、`$float_cap` 这类常见组合不算融合，`$float_cap` 已移出股东面识别键）。`memory_entries.facets_json` 记录全量面集合，检索亲和两档：family 精确命中 +0.3、facets 有交集 +0.15（单面 query ↔ 融合条目的跨召回）。registry（候选/正式）入库写 `facets`/`is_fusion`/`family`/`eval_label`。facet_focus 为 system prompt 插件模块（ORDER=135，含信号族白名单豁免声明——`research_policy_prompt` 经 extra_instructions 注入的白名单与融合指令矛盾，聚焦生效时显式解禁）。

## 评估档位自动推断 + 调仓频率（2026-09-03，方案 B）

前端"日线技术/基本面"模式下拉退役，档位由数据面多选自动推断：
- `core.research_modes.infer_research_mode(focus_facets)`：勾选基本面/股东面 → fundamental 档
  （label_20d + 松门槛 + monthly 门禁）；其余/未选 → technical 档（label_1d + 严门槛 + weekly 门禁）。
  混合勾选落 technical（融合因子以价量为主信号）；纯函数，前后端同口径。
- 前端 composer 显示推断档位徽章（`inferredModeLabel`），勾面变化经 `syncInferredMode()` 自动
  切换 spec（复用 `switchResearchMode` 链路：门槛弹窗/保存门槛/label_col 跟随全部保留）。
- `StartRequest.rebalance_freq`（daily/weekly/monthly，可空）：用户显式指定时写入
  `spec.delivery_policy.production.engine_gate.freq` 覆盖档位默认；`DeliveryCriteria.to_prompt_text`
  渲染 "rebalance_freq 必传" 指令约束 LLM；`allowed_freqs` 白名单仍生效。前端 composer 新增
  "调仓" 选择器（自动/日/周/月）。
- `research_mode` 参数兼容保留：显式传入优先于自动推断（历史调用方/脚本不受影响）。
- 因子库页模式页签退役（`FactorLibrary.vue`），由 facet 筛选 chips（全部/八面/融合）取代；
  ML 组合训练的 `collect_factor_entries` 按解析后库路径去重，避免大库下重复枚举。

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

- **盲测段锁定（`scripts/blind_test_factors.py`）**：2025-01-01 起为锁定盲测段——挖掘会话面板 coverage = `train_start~val_end`（2020~2024），LLM 迭代、stage_one/two、engine_gate 全部看不到 2025 数据；`blind_test_factors.py` 对已定稿因子做一次性离线重测（复用 `compute_ingest_metrics` 同口径），结果只写 `artifacts/alphaagent/blind_test/<run_ts>/report.json`、不回流任何门槛。**频繁重测会把盲测段重新烧掉（多重检验），克制使用频率**。train/val 双段在筛选中被反复使用、非真正 held-out——盲测段是唯一的诚实样本外。窗口重映射记录：2026-08-29 由 train 2018~2022 / val 2023~2025 / test 2026 起 调整为 train 2020~2022 / val 2023~2024 / test 2025 起（train 收近 3 年防因子衰减，test 段约 20 个月更足）。
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
- **挖掘并行评估**：`StockEvalService._eval_semaphore` 限制并发 train/val 评估数（StartRequest `max_parallel_eval`，默认 4，环境变量 `MAX_PARALLEL_EVAL` 覆盖），tool_calls 由 `ThreadPoolExecutor(max_tool_workers=4)` 并行分发。挖掘路径 `service._run_one` 以 `include_charts=False` 调引擎（逐日 IC/十分位多空/月度分解等图表数据仅因子实验室 `eval_profile` 生成）；引擎计时见结果 `timing_ms`（dsl_eval/transforms/metrics 三段）。
- **逐日统计 numba 内核（`factor/metrics_fast.py`）**：逐日 Pearson/Rank IC（含平均秩）、十分组 label 均值、市值中性残差、截面 zscore/winsorize 全部下沉为 `@njit(nogil=True, cache=True)` 内核（GIL 释放 → 多评估并发真多核）。Rank 内核注意两点坑：平均秩必须经 argsort 映射回**原位置**；秩均值 (n+1)/2 非零，相关系数必须中心化。开发期教训：**编辑 numba 内核源码后若同秒重编译，磁盘缓存可能陈旧导致错值甚至 access violation**——部署/调试异常时先删模块 `__pycache__`。
- **评估 metrics 快路径（`factor/metrics.py`）**：逐日 IC / Rank IC / lag1 自相关 / 十分位分组 / 分位组合 / 市值中性化全部改为「datetime 连续区间切片（`_day_slices`，面板按 datetime 排序时 O(1) 取每日切片，消除逐日 `groupby+xs` 全表查找）」；`pd.qcut` 换成数值等价的快速等频分箱 `_fast_equal_freq_codes`（分位边界浮点重合时自动回落 qcut，保证语义一致）；`mls_fmb` 与 `long_short_portfolio` 经 `context.cache` 共享同一份每日十分位结果。回落路径保留（未排序面板/测试注入 `_day_slices_override`）。门禁：`tests/test_metrics_fastpaths.py`（快慢路径输出一致 + 分箱等价对照）。
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

## 开发效率

### 代码搜索必须先带筛选（强制）

本仓库包含大量**大型数据产物**：`artifacts/`（JSON/parquet/npy 因子库、面板缓存、报告）、`logs/`（JSONL 轨迹）、`.venv/`、`static/node_modules/`、`CNEquity/`、`sentiment-mvp/` 等。**全量搜索会扫描这些目录，动辄数百毫秒甚至更久，且污染结果**。这是用户明确要求的硬性规则，任何搜索/匹配都必须先过滤。

强制规则：
- **搜索（grep/glob）必须带 `include` 参数**（如 `*.py` / `*.vue` / `*.ts` / `*.js` / `*.md`），只搜源码文件；禁止不带 include 的全仓 grep。
- **能限定 `path` 时就限定到源码目录**：`alphaagent/`、`backend/`、`core/`、`scripts/`、`static/src/`、`tests/`、`docs/`。
- 禁止无筛选扫描：`artifacts/`、`logs/`、`.venv/`、`node_modules/`、`CNEquity/`（除非明确要查数据产物/第三方库本身）。
- 查 DSL 算子/因子评估逻辑 → `alphaagent/dsl/`、`alphaagent/factor/`；查回测引擎 → `core/`；查前端 → `static/src/`；查 API → `backend/`。
- 文件读取（glob/read）同样先确认目标是否在源码目录；数据产物目录不得直接列目录全文。

## 常用命令

```powershell
# 启动后端（推荐：一键启动 Quant UI 后端 17891 + CNE dashboard 8787）
# 桌面快捷方式「Quant UI 启动.lnk」即运行本脚本
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_backend_with_sync.ps1

# 仅后端（若只需 API 不想起 CNE dashboard）
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 17891

# CLI 直接启动挖掘（默认 cne:// panel + production_main 统一大库）
.venv\Scripts\python.exe scripts\run_alphaagent.py

# 候选池重放晋升（把候选池已有因子重新走一遍修复后的两阶段链路）
.venv\Scripts\python.exe scripts\promote_candidates.py

# 盲测段因子重测（默认 2026-01-01 起；锁定段——挖掘循环与入库门槛从未见过 2026 数据）
.venv\Scripts\python.exe scripts\blind_test_factors.py

# 前端 dev
cd static && npm run dev
```

> **端口约定**：Quant UI 后端固定 **17891**；CNE dashboard **8787**。历史上 AGENTS.md 曾写 8000，已废弃 —— 一律以 `scripts/start_backend_with_sync.ps1` 为准。

## 依赖

见 `requirements-alphaagent.txt`：agentscope, openai, numba, jieba。

> AI生成
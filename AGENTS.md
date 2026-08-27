---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '35898fec-738a-418d-af06-0dd2255fdb1e'
  PropagateID: '35898fec-738a-418d-af06-0dd2255fdb1e'
  ReservedCode1: 'ab2f9ade-6971-4d25-84d1-aa3f9546f4f1'
  ReservedCode2: 'ab2f9ade-6971-4d25-84d1-aa3f9546f4f1'
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
└── alphaagent_factor_mining.py  # 上游 AgentScope 挖掘脚本（被 run_alphaagent 调用）

static/src/views/AlphaAgent.vue  # 前端页面：研究 / 因子实验室 / 因子库三个子标签

artifacts/alphaagent/
├── factorzoo/stock_1d/       # 生产因子库
├── factorzoo/candidate_1d/   # 候选因子池
└── research_memory.json      # 跨会话持久化研究记忆

logs/factor_mining/ui/        # 每次运行的 JSONL 轨迹 + console.log + run_meta.json
```

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

- `panel_path` 默认 `"cne://"`，触发 `alphaagent/data/adapters/cnequity.py` 从 CNE 数据湖实时构建 panel。
- `factorlib_path` 默认 `artifacts/alphaagent/factorzoo/stock_1d`。
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

| 阶段 | 门槛 | 写入位置 |
|---|---|---|
| candidate（海选） | \|IC\| ≥ 0.015, \|ICIR\| > 0.25, coverage > 0.85, max_corr < 0.6, lag1 自相关 ≥ 0.18, val 保留比 ≥ 0.5 | factorzoo/candidate_1d |
| production | \|train IC\| ≥ 0.025, \|train ICIR\| ≥ 0.30, \|val IC\| ≥ 0.015, val 保留比 ≥ 0.60, winsorized 衰减 ≤ 0.10, max_corr < 0.4 | factorzoo/stock_1d |

提交流程顺序：
1. **stage_one 统计门槛**（IC/ICIR/coverage/换手/val 保留比）→ 不通过拒绝
2. **正交性检查**（stage_one 统一查，不再只在 approve 后触发）→ 与正式库+候选池已有因子做 Spearman 相关，超过阈值拒绝
3. **review_hook**（LLM Reviewer 审核）→ reject 拒绝、revise 入候选池待修订、approve 继续冲正式库
4. 入候选池（registry_only）
5. **stage_two 精筛**（双窗口口径）→ 不通过停在候选池
6. **engine_gate 回测门禁**（完整回测引擎净值裁决）→ 不通过停在候选池
7. 入正式库（canonical 对齐 + ingest）

### 6. 候选因子库管理

- 前端因子库页支持正式库/候选库切换，各有导出 JSON 按钮
- `scripts/dedup_candidate_factors.py`：一次性去重脚本，两两截面 Pearson 相关 ≥ 阈值的因子对删除冗余项
- 候选因子 registry 的 `similarity` 字段记录与已有因子的截面 Pearson 相关

### 7. 因子类别注册表（`core/factor_categories.py`）

候选库 + 正式库按 `research_mode` 分目录管理。加新类别只需在 `FACTOR_CATEGORIES` 中注册一行，
库路径自动跟随 `research_mode`（已在整条链路传递）。

### 8. 统一交易参数（`core/trading_config.py`）

全平台唯一默认交易参数来源（散户口径），回测/模拟盘/因子门禁/前端默认值均从此取。
后端 API `/api/trading-defaults` 供前端启动时自动获取。

### 9. 研究记忆（research_memory.py）

- BM25 检索历史评估证据（含 jieba 中文分词，无 jieba 回退 bigram）。
- 正向 verdict（production_approved, validated 等）鼓励邻域探索；负向 verdict（rejected, weak 等）防止重复无效路径。
- 存储在 `artifacts/alphaagent/research_memory.json`。

## REST API

| 方法 | 路径 | 功能 |
|---|---|---|
| POST | `/api/alphaagent/runs` | 启动新挖掘 |
| GET | `/api/alphaagent/runs` | 列出所有 runs |
| GET | `/api/alphaagent/runs/{id}` | 查看 run 详情（事件轨迹尾部） |
| POST | `/api/alphaagent/runs/{id}/stop` | 终止 |
| POST | `/api/alphaagent/runs/{id}/messages` | 向运行中的进程注入消息 |
| POST | `/api/alphaagent/runs/{id}/continue` | 已结束的 run 继续（fork 新进程） |
| POST | `/api/alphaagent/runs/{id}/branch` | 分支新 run |
| GET | `/api/alphaagent/research-memory` | 查看研究记忆 |
| GET | `/api/alphaagent/research-spec/default` | 默认研究策略 |
| GET | `/api/alphaagent/evaluation-capabilities` | 可用评估插件和 profiles |

## 关键设计决策

- **panel 实时构建而非预构建**：`cne://` 标识触发 adapter 从 CNE 数据湖按日期范围拉取，避免维护大 parquet 文件。首次加载约 30-60 秒，之后驻内存复用。
- **会话域复用**：submit 全程在挖掘会话驻内存 panel 上评估，不再全量重载因子库域 panel（避免 OOM）。
- **JSONL 轨迹持久化**：每次 run 的所有事件写入 `logs/factor_mining/ui/<run_id>/run_*.jsonl`，FastAPI 重启后可从磁盘恢复历史 run。
- **DSL 编译为 Python**：多行表达式先赋值中间变量，最后一行输出因子值；支持 `$列名` 引用 panel 列。
- **三层加速后端**：C++ > Numba JIT > 纯 Python，自动检测降级。
- **Codex provider**：从 `~/.codex/config.toml` 读取模型配置和 token，无需额外 .env 配置。
- **评估用分位数回测**：A股不能做空，因子评估用 quantile_portfolio（纯多头口径）替代 topn_portfolio。
- **批量脚本 OOM 规避**：独立脚本加载 CNE panel 时直接读 `stock_daily_wide` 原始 parquet（限定日期+最小列集），不用 `load_panel_from_cne()` 全量加载。

## 常用命令

```powershell
# CLI 直接启动挖掘（默认 cne:// panel + stock_1d factorlib）
.venv\Scripts\python.exe scripts\run_alphaagent.py

# 启动后端
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 前端 dev
cd static && npm run dev
```

## 依赖

见 `requirements-alphaagent.txt`：agentscope, openai, numba, jieba。

> AI生成
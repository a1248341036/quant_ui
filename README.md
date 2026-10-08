# quant_ui — A 股量化研究平台

一个自托管的 A 股量化研究与因子挖掘 Web 平台：Vue3 单页应用 + FastAPI 后端 +
本地增量行情缓存（Parquet + SQLite），内置因子轮动与事件驱动双回测引擎，
支持账户记账、今日信号、参数稳健性、历史归档、语音情绪分析与**AlphaAgent
LLM 自主因子挖掘**（DSL 表达式 → 评估 → 筛选 → 验证 → 去重 → 入库闭环）。

- 地址：Vue 工作台 `http://<host>:17891`，API 文档 `/docs`
- 默认免登录（鉴权代码保留，可恢复，见「登录鉴权」）
- 数据源：腾讯行情 + Tushare + CNE 数据湖（日线/财务 PIT 宽表）

---

## 1. 功能总览

| 模块 | 说明 |
| --- | --- |
| 看板（账户 KPI + 资金曲线 + 策略对比） | 真实账户为空时显示「未开户」引导 |
| 单策略回测（因子轮动） | 股票池/策略/TopN/资金/频率/区间/过滤/预热 |
| ETF / 场外基金回测 | 全市场 ETF 池（腾讯日线）；科技场外基金池（天天基金净值，T+1 + 申赎费） |
| 多空对冲 + 行业中性化 | 多头 TopN + 空头最弱 N 只，模拟融券费率 |
| 多因子组合（权重/方向/保存/信号） | 因子自由组合打分 |
| 事件驱动策略实验室 | 按策略类型生成差异化模板 + FactorKit 数据/日期封装 |
| 聚宽策略兼容层（JQ shim） | 代码面板直接运行聚宽风格策略（initialize + run_daily/weekly + 数据 API） |
| 代码页 API 预检 / 异步回测 | 秒级 AST 预检报告缺失 API；回测边跑边画图 + 进度/ETA + 可停止 |
| 个股详情页 | 历史 K 线 + 技术指标 + 复权切换 |
| 策略池管理 | 全量池 / 配置池 / 回收站，前端拖拽排序 |
| 参数稳健性 / Walk-forward | 参数网格 + 分窗口回测 + 滚动训练-测试 |
| 历史回测归档 | 回测结果持久化（参数/指标/净值/交易） |
| 今日信号 / 日级模拟盘 | 按因子打分输出候选；自动下单/成交/日结（systemd 盘后执行） |
| 账户记账 | 出入金/交易/持仓/估值，估值基于 panel 行情 |
| 数据管理 | 腾讯 + Tushare 双源增量刷新；数据更新任务配置 |
| 舆情情绪看板 | 独立仓库 `sentiment-mvp` 词典/LLM 打分 |
| 因子质量分析 / QuantStats / Brinson 归因 | 回测页勾选后展示；绩效报告；行业归因 |
| Screener（regime 感知因子选择） | 确定性规则替代 LLM：ADX+均线 regime → 因子适用性评分 → 语义去重 → 动态权重 |
| **AlphaAgent 自主因子挖掘** | LLM 多轮对话生成因子 DSL → 训练/验证评估 → 相似度去重 → 两阶段入库（候选池→正式库 + 回测门禁） |
| **AlphaAgent 记忆系统 v3** | 三层研究记忆（原始证据 + SSPM 编辑统计 + 经验蒸馏），BM25 检索注入 + 置信度门控，跨面融合标签 |
| **ML 组合训练（stacking）** | Ridge/LightGBM 组合打分，时间序列交叉验证 + purge 隔离；Null Importance 因子预筛选（Stage 0） |
| **研报模式（report）** | 照研报/机制卡忠实复现 → 单向发散；保真度/锚点护栏 + 题库质量准入 |
| **AlphaAgent MCP server** | stdio 手写协议，14+ 挖掘/评估/交付/记忆工具，Codex/DSH 可直连驱动挖掘 |

## 2. 架构

```
┌──────────────────────────────────────────────────────────────┐
│  Vue3 SPA (static/src) + FastAPI backend/main.py :17891        │
└───────────────────────┬──────────────────────────────────────┘
                        │ REST (/api/...)
┌───────────────────────▼──────────────────────────────────────┐
│  backend/:  routers（data/backtest/ledger/paper/code/sentiment/ │
│  stock/strategy_pool/alphaagent）+ services + process_manager   │
├──────────────────────────────────────────────────────────────┤
│  core/          平台核心（见 §5 回测引擎）                      │
│    engine/      因子轮动回测（四阶段管线：config→prepare→       │
│                 factor_matrix→simulate→result）                │
│    event_engine/ 事件驱动回测（context/runner/strategies + jq   │
│                  聚宽兼容层）                                  │
│    paper/       日级模拟盘（account/details/rebalance/storage） │
│    fetcher/     行情抓取（kline/stocks/funds/update）           │
│    data/        panel/status/assets 数据访问层                  │
│    selection.py / execution.py / metrics.py / performance.py   │
│    attribution.py / walkforward.py / risk_model.py / limit.py  │
│    screener.py / regime.py / ledger.py / strategy_pool.py      │
│    factor_registry.py / factor_categories.py / panel_schema.py │
│    trading_config.py（统一交易参数唯一真源）                     │
│    research_modes.py（研究模式注册表唯一真源）                   │
│    cne_reader.py / financial.py / updater.py / qqnotify.py     │
├──────────────────────────────────────────────────────────────┤
│  alphaagent/    LLM 自主因子挖掘（见 §7）                       │
│    factor/mining/  挖掘主循环（agent/delivery/eval/infra/       │
│                     memory/prompt/tools/bench/diagnostics 子包）│
│    factor/evaluation/ 冻结 profile 评估引擎 + 插件化            │
│    factor/metrics/    逐日统计 numba 快路径                     │
│    factor/zoo/        因子库统一大库 + 相似度索引               │
│    factor/stacking/   ML 组合训练（dataset + model + screening）│
│    dsl/         因子表达式 DSL（parser/operators/accel JIT +    │
│                 stock 增量计算/resample/缓存）                  │
│    data/        panel 构建 + CNE 数据湖适配器（插件化）          │
│    mcp_server/  stdio MCP 工具集（catalog/protocol/tools）      │
│    compute/     研究计算池（多进程 + 内存面板）                  │
├──────────────────────────────────────────────────────────────┤
│  数据                                                        │
│  data/panel.parquet        日线+因子面板                        │
│  artifacts/panel/cache/    CNE 数据湖面板缓存（秒级命中）        │
│  data/quant_dataset/       CNE 年度日线档案（不复权原始价+复权因子）│
│  data/pg_parquet/          Tushare 财务/公告宽表（PIT 只读）      │
│  data/quant.db             SQLite 业务库（回测/账本/模拟盘）      │
│  artifacts/alphaagent/     因子库/候选池/研究记忆/门槛文件        │
│  strategies/registry.py    策略注册表                            │
└──────────────────────────────────────────────────────────────┘
```

## 3. 快速开始

依赖：Python 3.12，推荐 venv `.venv/`。

```bash
cd ~/quant/quant_ui
cp .env.example .env   # 按需填 TUSHARE_TOKEN / CNE_TOKEN
pip install -r requirements.txt            # 平台基础依赖
pip install -r requirements-alphaagent.txt # AlphaAgent 挖掘（agentscope/openai/numba/jieba）

# 后端 API + Vue 前端（:17891）
systemctl start quant-api                    # Linux 部署
systemctl start quant-data-refresh.timer     # 每日 15:10/16:30 收盘后自动增量更新行情

# 手动跑一次数据更新
python scripts/refresh_data.py
```

### 本地启动（Windows）

```powershell
# 一键启动（后端 :17891 + CNE 数据湖看板 :8787；先 SSH 确认服务器数据状态再本地只拉取同步）
.\scripts\start_backend_with_sync.ps1

# 仅后端
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 17891

# 只检查、不写入本地数据
.venv\Scripts\python.exe scripts\startup_remote_sync.py --dry-run
```

端口约定：**后端固定 17891，CNE 看板固定 8787**（脚本 `scripts/start_backend_with_sync.ps1` 为准）。

### 数据目录与路径环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QUANT_UI_DATA_DIR` | `<项目>/data` | 主数据目录：panel/ETF/基金/duck.db/pg_parquet 等 |
| `QUANT_UI_LEGACY_DATA_DIR` | `~/quant_data` | 旧独立数据目录（对照用） |
| `QUANT_UI_SENTIMENT_DIR` | `<项目父目录>/sentiment-mvp` | 舆情独立仓库 |
| `QUANT_UI_QQBOT_DIR` | `~/qqbot` | QQ 机器人凭据/推送脚本目录 |

### Docker 部署（单容器全栈，可移植）

```bash
cp .env.example .env            # 含 Tushare token；建议 CNE_TOKEN
docker compose up -d --build    # 首次构建约 10-20 分钟
# Vue 工作台 :17891 / 数据状态看板 :8001 / CNEquity 数据湖 :8787
```

容器内 supervisord 管理 4 组进程（FastAPI / data_status / CNE 看板 / supercronic 定时任务）。
数据卷 `quant_data` 同时挂载 `/app/data` 与 `/app/CNEquity/data`（同一棵数据树），
迁移整机只搬一个 volume。

## 4. 数据架构

- 主面板：`data/stock/panel.parquet`（日线 + 滚动因子）；运行期优先
  `cne://` 从 CNE 数据湖实时构建（磁盘缓存 `artifacts/panel/cache/` 秒级命中）
- 日线档案：`data/quant_dataset/YYYY/YYYY/day/stock_daily.parquet`
  （不复权原始价 + 每日复权因子，`scripts/sync_daily_to_cne.py` 增量写入）
- 财务宽表：`data/pg_parquet/`（Tushare 财务/公告，PIT 对齐，CNE external bridge 只读接入）
- 行情源：股票优先 Tushare（`.env` token，失败回退腾讯）；ETF 腾讯；场外基金天天基金净值
- 回测数据源由 `QUANT_DATA_SOURCE` 控制：`cne` / `pg_parquet`（兼容别名） / `panel`
- 业务数据统一 SQLite：`data/quant.db`（回测归档、账本、模拟盘、策略池）
- 舆情：独立仓库 `sentiment-mvp`

## 5. 回测引擎

### 统一交易参数（`core/trading_config.py`）
全平台唯一默认交易参数来源（散户口径：资金 10 万、佣金万 2.5、卖出万 12.5、滑点 0bps、
参与率 10%、选股 top 0.4%）；回测引擎、模拟盘、因子门禁、前端默认值均从此取。
`core/trading_config_store.py` 提供代码面板参数副本（可独立覆盖）。

### 因子轮动（`core/engine/` 四阶段管线）
- `config → prepare → factor_matrix → simulate → result`
- 选股统一走 `SelectionPolicy` + `PortfolioBuilder`（count_mode、min/max positions、
  min_score 门槛、ADX 趋势门控）；撮合执行独立（`core/execution.py`，先卖后买）
- 因子注册表单一来源 `core/factor_registry.py`：量价/基金专属/财务（PIT 对齐）/动态因子
- 过滤：剔除科创/创业、一手 100 股、成交额分位、因子预热；行业分散、多空对冲、行业中性化
- 风险中性化：因子得分对风格/行业暴露回归取残差 + 期末持仓风险归因
- 输出：净值/基准/回撤/指标/持仓/调仓记录 + Brinson 行业归因；`analyze=true` 附因子质量（IC/分组）
- 市场冲击模型（`impact_coef` + `impact_vol`）

### 事件驱动（`core/event_engine/`）
- `on_bar(ctx, bar)`；信号日收盘 → 执行日 T+1 开盘成交；涨跌停约束（`core/limit.py`）
- ctx API：`history` / `close_series` / `available_fields` / 组合优化
  （`optimize_risk_parity` / `mean_variance` / `max_diversification`）
- 内置示例：`GoldenCrossStrategy` / `RiskParityStrategy` / `LongShortMomentumStrategy`

### 聚宽策略兼容层（JQ shim，`core/event_engine/jq/`）
代码面板直接运行聚宽风格策略，按聚宽官方文档类别拆分：
- `api/`：framework / settings / trading（order 四件套）/ data_api / portfolio / finance / misc
- `runtime.py` 编排生命周期 → 撮合对接；`preflight.py` AST 预检（秒级报告缺失 API/字段）
- `factor_bridge.py`：`get_factor` 可调用 AlphaAgent DSL 因子
- **datalake 数据接入插件体系**：CNE curated 表启动时批量注册为通用插件（`load("dragon_tiger")`
  零代码接入）；需专属清洗的表写 `JQDataPlugin` 插件；未注册名抛带指引的 NotImplementedError
- 分钟线接入：盘中价查询 + 下单时点真实价撮合；`set_option('avoid_future_data')` 真生效

### 其他引擎能力
- 场外基金回测（`core/fund_engine.py`）：T+1 净值执行 + 申赎费
- 风险模型（`core/risk_model.py`）：轻量 Barra 风格 + LedoitWolf 收缩
- 滚动训练-测试（`core/walkforward.py`）：全历史选参 + 窗口样本外验证
- 日级模拟盘（`core/paper/`）：T-1 收盘生成目标 → T 开盘成交 → 收盘估值；幂等；风控
  （单票上限/流动性分位）；事件账户全量重放
- Screener（`core/screener.py` + `core/regime.py`）：regime 检测 → 因子适用性评分 → 语义去重 → 动态权重

### 一致性验证
`scripts/selftest.py` 内置 14 组对照：长期/近半年、7 个同名策略，与旧引擎 `quant_3stocks`
**14/14 完全一致**（误差 < 0.01pp）。

## 6. 研究模式与门槛（`core/research_modes.py`，单一真源）

**档位由数据面自动推断**（`infer_research_mode(focus_facets, rebalance_freq)`），不再手动选档：

| 档位 | label | engine_gate freq | min_abs_ic (candidate) | 说明 |
| --- | --- | --- | --- | --- |
| `technical` | 1d | daily | 0.020 | 价量日频主档 |
| `technical_daily` | 1d | daily | 0.020 | label↔freq 对齐子档 |
| `technical_weekly` | 5d | weekly | 0.030 | 慢档（2026-10-07 标定） |
| `technical_monthly` | 20d | monthly | 0.053 | 慢档（2026-10-07 标定） |
| `fundamental` | 20d | monthly | 0.053 | = technical_monthly 门槛 + `needs_fundamentals=True`（载入 funda_* 列） |
| `report` | 1d | daily | 0.020 | 研报模式（复现→发散） |

关键规则（用户 2026-10-04/10-08 定调）：
- **label 必须等于调仓频率**：daily↔1d、weekly↔5d、monthly↔20d；不一致 fail-closed 拒绝启动
- **门槛只由 label 唯一决定，与数据面无关**：`_MONTHLY_20D_OVERRIDES` 共享常量
  （0.053 / 0.650 / 0.0398，production 0.0663 / 0.6964，engine_gate 年化 0.03 / 夏普 0.5）
- 门槛唯一真源在 `delivery_criteria`，Web 门槛弹窗编辑 → `research_specs/<mode>.json` 增量覆盖 → 全链路生效

## 7. AlphaAgent 因子自主挖掘（`alphaagent/`）

LLM 驱动的多轮对话循环，自动完成「生成因子 DSL → 训练集评估 → 验证集检验 →
相似度去重 → 入库交付」闭环。完整架构见 `docs/alphaagent_architecture.md`。

### 7.1 工作方式

```
用户下达研究方向（Web / CLI / MCP）
  → LLM 每轮发起 tool_calls：evaluate_factor（DSL → IC/ICIR/覆盖度/月度稳健性）
  → 盲测终审（test 绝对下限 + 保留比 + 方向一致）→ 不通过直接拒绝
  → stage_one 统计门槛（IC/ICIR/coverage/换手/自相关/val 绝对下限）
  → 正交性检查（与正式库截面相关超阈值拒绝）+ FactorReviewer LLM 审查（reject 硬拦）
  → 候选池 → stage_two 精筛（双窗口口径）→ engine_gate 净值回测门禁（实盘可交易性裁决）
  → 正式库
```

- **三层约束**：经济直觉强制（先写因果链）→ 父本变异策略（参数/算子/修饰）→ 正交预判（ρ>0.7 拦截）
- **认知层升级（五件套）**：预测-对账闭环（`prediction` 必填 + `prediction_check` 四分类 verdict）、
  十分位形态学教学、A 股机制知识模块、D 轨机制驱动重构（机制三问）、强制消融（门控算子 base-only 对照）
- **数据面聚焦**：前端 run 表单多选数据面（价量/量能/筹码/拥挤/基本面/股东/事件/资金）→ 融合指令 +
  面覆盖警告 + 记忆按 facets 亲和检索；跨面融合成功经验打标签
- **DSL**：多行表达式编译为 Python，`$列名` 引用面板列；Numba/C++ JIT 三层加速；慢算子三重门禁
  （静态扫描 / 性能预算 / 数值一致性），算子耗时监控（`dsl-monitor` API）
- **评估**：冻结 `EvaluationProfile`（transform → metric → rule）；IC 前置过滤短路；
  两段式海选（lite 过线才跑全量）+ 日切片/float64 缓存（基准 2.61 eval/s）；L1 深度曲线 +
  可成交域透镜（只测量不加门槛）
- **记忆系统 v3**：三层研究记忆（原始证据 `memory_entries` + SSPM 编辑统计 `memory_cells` +
  经验蒸馏 `memory_experience`），BM25 + 族亲和 + 算子重叠注入；AlphaMemo 置信度门控
  （Eq.7 / APV 双门）；显式父本协议 + 指纹死路/正向重复提醒；正向跨族保底；语言面聚焦标签；
  因子中台 ID（`factor_uid`）+ `factor_index.db` 索引
- **入库门槛**：唯一真源 `delivery_criteria`，按研究模式存
  `artifacts/alphaagent/research_specs/<mode>.json` 增量覆盖；盲测段锁定
  （2025-01-01 起，`visible_range` 隔离，重测脚本 `scripts/blind_test_factors.py` 克制使用）
- **Null Importance 预筛选（Stage 0）**：ML 组合训练前用树模型 gain importance
  对每个因子跑真标签 + 打乱 y 的 null 分布（80 次），`score = log(1e-10 + actual/(1+P75(null)))`，
  `passed = score > 0`；样本不足（<200）或缺特征 → 全部 fail 并注明 `insufficient_window_samples`
- **ML 组合训练（stacking）**：Ridge/LightGBM 组合打分 → 时间序列交叉验证 + purge 隔离 →
  组合因子回灌回测引擎；子进程执行 + 进度日志
- **研报模式（report）**：照研报/机制卡忠实复现 → 单向发散；保真度判定 + 锚点护栏 +
  题库质量准入（跳过 suspect/deprecated）+ 连续锚失败止损
- **harness 直驱挖掘**：`scripts/harness_mine.py` 宿主 agent 亲自跑挖掘环路（批量评估 + 真实交付）

### 7.2 模块结构

```
alphaagent/factor/mining/          挖掘主循环与工具
├── agent/        AgentScope 运行（loop/run/agentscope_run/tools/reviewer/preflight/population）
├── delivery/     两阶段入库（submit + delivery_checker + delivery_criteria + engine_gate）
├── eval/         评估服务（service/session/schemas/context/response/mls_thresholds/param_stability）
├── infra/        基础设施（config/audit/cli_stream/console/registry_io/remove）
├── memory/       研究记忆 v3（store/retrieval/calibration/advisory/experience/ingestion/schema/...）
├── prompt/       Prompt 模块（prompts/prompt_modules/seed_factors/operators/modules/）
├── tools/        FactorEvalTools（_dispatch/_prefilter/_analysis/_schemas）
└── diagnostics/  诊断插件（PIT 警戒/换手/ICIR/near-miss/十分位塌缩/反同质化）
```

### 7.3 启动方式

```powershell
# Web：AlphaAgent 页 → 新建研究任务（勾数据面自动定档，可选调仓频率）
# CLI：直接启动挖掘（默认 cne:// panel + 统一大库 factorzoo）
.venv\Scripts\python.exe scripts\run_alphaagent.py

# 候选池重放晋升 / 去重 / 重新过门槛
.venv\Scripts\python.exe scripts\promote_candidates.py
.venv\Scripts\python.exe scripts\dedup_candidate_factors.py --dry-run
.venv\Scripts\python.exe scripts\rescreen_candidate_pool.py --data-root . --apply

# 盲测段因子一次性离线重测（锁定段，克制使用）
.venv\Scripts\python.exe scripts\blind_test_factors.py

# ML 组合训练（含 Null Importance 预筛选）
.venv\Scripts\python.exe scripts\train_ml_composite.py --screen

# 慢档门槛重标定（仅改档位门槛时用）
.venv\Scripts\python.exe scripts\calibrate_slow_tier_thresholds.py
```

依赖见 `requirements-alphaagent.txt`（agentscope、openai、numba、jieba）；
LLM 凭据支持 Codex provider（`ALPHA_LLM_PROVIDER=codex`，读 `~/.codex/config.toml`）。

### 7.4 MCP server（stdio）

`alphaagent/mcp_server/` 提供手写 stdio MCP 协议（catalog/protocol/tools），
14+ 工具覆盖挖掘、评估、交付、记忆四类；Codex 已接线、DSH 已配置，支持
跨进程会话锁与多客户端并发。盘点见 `docs/review/agent_tool_mcp_inventory_20261007.md`。

## 8. 后端 API（FastAPI :17891，文档 `/docs`）

```
# ── 数据 ──
GET  /api/data/status | panel-info | update/status | update-configs | indices | indices/series

# ── 回测/策略/因子 ──
GET  /api/strategies | /api/factors | /api/alpha-factors | /api/composites | /api/names
POST /api/backtest | /api/signals | /api/sweep | /api/backtest/quantstats | /api/backtest/attribution
GET  /api/backtest/runs | /api/backtest/runs/{run_id}

# ── 个股 ──
GET  /api/stock/search | /api/stock/{code}

# ── 策略池 ──
GET  /api/strategy-pool | /full | /trash；POST /add | /remove | /restore | /full-delete | /reorder

# ── 代码实验室 ──
GET/POST /api/code/saved* | /api/code/config*；POST /api/code/jq/preflight|run|run_async
GET  /api/code/jq/runs*；GET/POST /api/code/qweave*（研究脚本）

# ── 账户/模拟盘 ──
GET/POST /api/ledger/* | /api/paper/*（accounts/orders/trades/positions/equity/events）

# ── 舆情 ──
GET  /api/sentiment/status | /stats | /news | /ic

# ── AlphaAgent（/api/alphaagent/*）──
POST /api/alphaagent/runs（focus_facets 自动定档）          GET/POST/DELETE runs 管理（stop/messages/continue/branch/events SSE）
GET  /api/alphaagent/research-memory（layers / DELETE {entry_id}）
GET  /api/alphaagent/research-modes | research-spec/default | research-specs/{mode}（GET/PUT/DELETE 门槛文件）
GET  /api/alphaagent/evaluation-capabilities | eval-factor | backtest-factor
GET/POST/DELETE /api/alphaagent/factors（library/facet 筛选）
POST /api/alphaagent/stacking/train；GET trainings；POST stop
GET  /api/alphaagent/session-cache/stats | evict | dsl-monitor | logs
```

## 9. 脚本工具（`scripts/`）

| 脚本 | 作用 |
| --- | --- |
| `refresh_data.py` | 行情增量更新 + CNE 年度档案同步 |
| `selftest.py` | 新旧引擎一致性对照（14/14） |
| `qweave_research.py` | 研究层：Alpha158/101/191 因子计算 + IC/分组/换手 + LightGBM 预测 |
| `performance_report.py` / `parameter_sweep.py` / `attribution.py` | QuantStats / 参数扫描 / Brinson |
| `paper_trade.py` | 日级模拟盘（创建/执行/查询） |
| `sentiment_backtest.py` / `alpha191_screen.py` | 舆情分桶回测 / GTJA Alpha191 筛选 |
| `sync_daily_to_cne.py` / `sync_tushare_to_parquet.py` / `refresh_etf.py` / `refresh_fund.py` | 数据同步 |
| `run_alphaagent.py` / `alphaagent_factor_mining.py` | AlphaAgent 挖掘 CLI 入口与主循环脚本 |
| `harness_mine.py` / `harness_submit.py` | 宿主 agent 直驱挖掘环路 |
| `promote_candidates.py` / `dedup_candidate_factors.py` / `rescreen_candidate_pool.py` | 候选池晋升/去重/重筛 |
| `blind_test_factors.py` | 盲测段锁定重测 |
| `train_ml_composite.py` | ML 组合训练（含 `--screen` Null Importance 预筛选） |
| `check_dsl_slow_patterns.py` | DSL 慢算子静态拦截 |
| `start_backend_with_sync.ps1` | 一键启动后端 + CNE 看板（含远程同步） |
| `alphaagent_mcp.py` | AlphaAgent MCP server 入口 |

完整清单见 `scripts/` 目录（90+ 脚本）。

## 10. 测试

```bash
.venv\Scripts\python.exe -m pytest tests -q --no-header   # 162 个测试文件
```

覆盖：引擎一致性、门槛配置中心（`test_threshold_config_center`）、慢档门槛不变量
（`test_slow_tier_thresholds`）、Null Importance 筛选（`test_null_importance_screen`）、
盲测段隔离（`test_blind_segment_isolation`）、记忆系统、预测-对账、DSL 三重门禁、
聚宽兼容层、label↔freq 准入、MCP server、研报模式护栏等。

> **GitHub Actions CI 已移除（2026-09-30）**：原 workflow 未安装仓库内 CNEquity 包
> 且含平台相关断言，main 每次 push 全红、不再提供有效信号，故删除。本地自行测试。

## 11. 登录鉴权

当前已临时关闭，前后端免登录。鉴权代码保留在 `backend/auth.py`
（pbkdf2 + HttpOnly Cookie），需要时重新挂载中间件即可恢复。

## 12. 开发状态

- 数据→因子→回测→报告闭环已跑通；绩效/归因（IC/分组/Brinson/QuantStats）已进 UI
- 因子轮动 + 事件驱动 + 聚宽兼容层 + 日级模拟盘 + 滚动训练-测试 + Screener 齐备
- AlphaAgent 已上线：自主挖掘、记忆系统 v3、统一大库、盲测段锁定、门槛配置中心、
  ML 组合（stacking + Null Importance 预筛选）、研报模式、MCP server、harness 直驱
- 测试基建：162 个测试文件，覆盖门槛口径、性能门禁、隔离性与回归

## 13. 已知限制（Demo）

- 模拟盘为日级，无实时行情撮合与实时风控；分钟线仅盘中查询/下单时点撮合，无分钟级回测
- 聚宽兼容层为单账户模型（`set_subportfolios`/`transfer_cash` 仅单账户语义），空头暂不支持
- 滑点默认为 0bps（散户口径），需在代码面板参数副本中手动开启
- 行情以 CNE 年度档案不复权原始价 + 每日复权因子为准；前复权在新分红/送转时整体重标定
  （个股详情可选「不复权」/「后复权」）
- 舆情数据依赖独立仓库 `sentiment-mvp` 流水线每日更新
- 大批量参数扫描同步执行（无任务队列，除异步回测与 AlphaAgent 挖掘外）

## 14. 关键文档

| 文档 | 内容 |
| --- | --- |
| `docs/alphaagent_architecture.md` | AlphaAgent 完整架构（流程/门槛/记忆/设计决策） |
| `docs/specs/` | 各功能 spec |
| `docs/review/` | OCR review / 实测评测记录 |
| `AGENTS.md` | Agent 工作纪律 + 速查（前端构建/分支/搜索纪律） |
---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'ba9aa1eb-1162-4b7f-ba54-33ca9001edfe'
  PropagateID: 'ba9aa1eb-1162-4b7f-ba54-33ca9001edfe'
  ReservedCode1: 'f8273c21-2484-42a2-b3bc-578a9b7ac267'
  ReservedCode2: 'f8273c21-2484-42a2-b3bc-578a9b7ac267'
---

# quant_ui — A 股个人量化研究平台

一个自托管的 A 股量化研究/回测 Web 平台：Vue3 单页应用 + FastAPI 后端 +
本地增量行情缓存（Parquet + SQLite），内置因子轮动与事件驱动两套
回测引擎，支持账户记账、今日信号、参数稳健性、历史归档与舆情情绪分析。
内置 **AlphaAgent**：LLM 驱动的 A 股日频多因子自主挖掘智能体（DSL 表达式 →
评估 → 验证 → 去重 → 入库闭环，详见「AlphaAgent 因子自主挖掘」一节）。
内置**聚宽策略兼容层（JQ shim）**：代码面板可直接运行聚宽风格策略。

- 地址：Vue 工作台 `http://<host>:17891`
- 默认免登录（鉴权代码保留，可恢复，见「登录鉴权」）

---

## 1. 功能总览

| 模块 | Vue :17891 | 说明 |
| --- | :---: | --- |
| 看板（账户 KPI + 资金曲线 + 策略对比） | ✅ | 真实账户为空时显示「未开户」引导 |
| 单策略回测（因子轮动） | ✅ | 股票池/策略/TopN/资金/频率/区间/过滤/预热 |
| ETF 池回测 | ✅ | 全市场 ETF 池（腾讯日线），涨跌停/停牌过滤自动关闭 |
| 场外基金回测 | ✅ | 科技相关场外基金池（天天基金净值），T+1 净值执行 + 申赎费 |
| 多空对冲 + 行业中性化 | ✅ | 多头 TopN + 空头最弱 N 只，模拟融券费率 |
| 多因子组合（权重/方向/保存/回测/信号） | ✅ | 因子自由组合打分 |
| 事件驱动策略实验室 | ✅ | 按策略类型生成差异化模板 + FactorKit 数据/日期封装 |
| 聚宽策略兼容层（JQ shim） | ✅ | 代码面板可直接运行聚宽风格策略（initialize + run_daily/weekly + 数据 API） |
| 代码页 API 预检 / 异步回测 | ✅ | 秒级 AST 预检报告缺失 API；回测边跑边画图 + 进度/ETA + 可停止 |
| 个股详情页 | ✅ | `/api/stock/{code}` 历史 K 线 + 技术指标 + 复权切换（前复权/后复权/不复权） |
| 策略池管理 | ✅ | 全量池（注册表 + 归档）/ 配置池（精选）/ 回收站，前端可拖拽排序 |
| 参数稳健性 / Walk-forward | ✅ | 双均线参数网格 + 分窗口回测 + 滚动训练-测试 |
| 历史回测归档 | ✅ | SQLite `backtest_runs` 记录参数/指标/净值/交易 |
| 今日信号 | ✅ | 按因子打分输出当日候选 |
| 日级模拟盘 | ✅ | 因子/事件策略自动下单/成交/日结，systemd 盘后执行 |
| 账户记账（出入金/交易/持仓/估值） | ✅ | 手动录入，估值基于 panel 行情 |
| 数据管理（状态/一键更新/指数线） | ✅ | 腾讯行情 + Tushare 双源，增量刷新；指数日线 + 数据更新配置 |
| 舆情情绪看板 | ✅（简版） | 读取独立仓库 `sentiment-mvp` 词典/LLM 打分 |
| 因子质量分析（IC/分组/多空价差） | ✅ | 回测页勾选「因子质量分析」后展示 |
| QuantStats 绩效报告 | ✅ | 回测结果页按钮，内嵌 HTML 报告 |
| Brinson 归因 | ✅ | 回测结果内置行业归因 + 代码实验室一键归因 |
| 舆情 IC/分组结果 | ✅ | 舆情 tab 展示 scripts/sentiment_backtest.py 输出 |
| Screener（regime 感知因子选择） | ✅ | 确定性规则替代 LLM：ADX+均线 regime 检测 → 因子适用性评分 → 语义去重 → 动态权重，带开关控制 |
| AlphaAgent 因子自主挖掘 | ✅ | LLM 多轮对话生成因子 DSL → 训练/验证评估 → 相似度去重 → 两阶段入库（候选池→正式库 + 回测门禁） |
| AlphaAgent 记忆系统 v3 | ✅ | 三层研究记忆（原始证据 + SSPM 编辑统计 + 经验蒸馏），BM25 检索注入，校准前因子方向 |
| 因子库/候选池浏览与门槛配置 | ✅ | AlphaAgent 页子标签：研究 / 因子实验室 / 因子库 / ML 组合；每模式门槛文件在线编辑，全链路生效 |
| ML 组合训练（stacking） | ✅ | Ridge/LightGBM 组合打分，时间序列交叉验证 + purge 隔离，子进程执行 + 进度日志 |

## 2. 架构

```
┌────────────────────────────────────────────────────────────┐
│  Vue3 SPA + FastAPI backend/main.py :17891                  │
│  static/index.html（FastAPI StaticFiles 托管）              │
└───────────────────────┬────────────────────────────────────┘
                        │ REST (/api/...)
┌───────────────────────▼────────────────────────────────────┐
│  FastAPI routers:                                           │
│    backtest / code / data / ledger / paper / sentiment      │
│    stock / strategy_pool / alphaagent                       │
│  backend/services/: 因子评估/仓库/提交/会话管理              │
├────────────────────────────────────────────────────────────┤
│  core/                                                      │
│  engine/            因子轮动回测（四阶段管线包）              │
│    config.py        BacktestConfig + run_backtest 门面       │
│    factors.py       因子构建（build_factor_frames）          │
│    prepare/prepare  阶段 1 准备                              │
│    factor_matrix    阶段 2 因子矩阵                          │
│    simulate         阶段 3 主循环模拟                        │
│    result           阶段 4 收尾 + latest_signals             │
│  event_engine/      事件驱动回测（包）                        │
│    context.py       Order/Bar/EventStrategy/Context         │
│    runner.py        run_event_backtest                      │
│    strategies.py    GoldenCross/RiskParity/LongShortMomentum│
│    jq/              聚宽策略兼容层（见 §8.5）                │
│  selection.py       SelectionPolicy + PortfolioBuilder      │
│  execution.py       现金/整手撮合执行器                       │
│  fund_engine.py     场外基金回测                            │
│  metrics.py         绩效指标（收益/波动/夏普/回撤/卡玛/胜率）  │
│  performance.py     因子 IC/分组 + QuantStats 报告           │
│  attribution.py     Brinson 归因                            │
│  walkforward.py     参数网格 / 分窗口稳健性                  │
│  risk_model.py      轻量 Barra 风格风险模型                  │
│  screener.py        regime 感知因子选择（确定性规则）         │
│  regime.py          市场制度检测（ADX + 均线）               │
│  limit.py           涨跌停约束                              │
│  ledger.py          账户账本                                │
│  paper/             日级模拟盘（account/details/rebalance/  │
│                     storage 子模块）                        │
│  backtest_archive.py 回测结果归档                           │
│  factor_registry.py 引擎因子元数据注册表（单一事实来源）      │
│  factor_categories.py 因子库分类（按 research_mode 路由）    │
│  strategy_pool.py   策略池（全量/配置/回收站）              │
│  strategy_types.py  策略定义统一模型（StrategyDefinition）   │
│  score_matrix.py    分数矩阵统一转换                        │
│  panel_schema.py    面板口径契约                            │
│  trading_config.py  统一交易参数（全平台唯一默认值来源）     │
│  trading_config_store.py  代码面板参数副本                  │
│  fetcher/           行情抓取（kline/stocks/funds/update）   │
│  data/              数据访问层（panel/status/assets）       │
│  store.py / db.py / sqldb.py  数据存储                      │
│  cne_reader.py      CNE 数据湖读取                          │
│  financial.py       财务因子（PIT 对齐）                    │
│  updater.py         数据更新器                              │
│  qqnotify.py        QQ 机器人通知                           │
│  run_log.py         运行日志                               │
├────────────────────────────────────────────────────────────┤
│  alphaagent/  LLM 自主因子挖掘（详见 §7）                    │
│  factor/mining/     挖掘主循环（agent/delivery/eval/infra/  │
│                     memory/prompt/tools 子包）              │
│  factor/evaluation/ 冻结 profile 评估引擎 + IC 前置过滤      │
│  factor/metrics/    逐日统计 numba 快路径（_core/decile/ic/ │
│                     mls/portfolio 子模块）                  │
│  factor/zoo/        因子库存储 + 相似度索引                  │
│  factor/stacking/   ML 组合训练（dataset + model）          │
│  factor/cache.py    因子值缓存（内存 LRU + 磁盘持久化）      │
│  dsl/               因子表达式 DSL + Numba/C++ JIT 加速     │
│    core/            parser/operators/accel/monitor/guard    │
│    stock/           增量计算/resample/缓存                  │
│  data/              panel 构建 + CNE 数据湖适配器            │
│    adapters/        插件化数据源（cnequity + plugins/）     │
├────────────────────────────────────────────────────────────┤
│  core/event_engine/jq/  聚宽策略兼容层（见 §8.5）            │
├────────────────────────────────────────────────────────────┤
│  数据                                                       │
│  data/panel.parquet       日线+因子面板（1800 只）           │
│  artifacts/panel/cache/   CNE 数据湖面板缓存（秒级命中）      │
│  data/duck.db             DuckDB 查询缓存/视图              │
│  data/quant.db            SQLite 业务库（回测/账本/模拟盘）  │
│  artifacts/alphaagent/    因子库/候选池/研究记忆/门槛文件     │
│  strategies/registry.py   策略注册表                        │
│  data_status/             数据状态服务（:8001）              │
└────────────────────────────────────────────────────────────┘
```

## 3. 快速开始

依赖：Python 3.12，推荐 venv `.venv/`。

```bash
cd ~/quant/quant_ui
cp .env.example .env   # 按需填 TUSHARE_TOKEN
pip install -r requirements.txt            # 平台基础依赖
pip install -r requirements-alphaagent.txt # AlphaAgent 挖掘（agentscope/openai/numba/jieba）

# 后端 API + Vue 前端（:17891）
systemctl start quant-api                    # Linux 部署
# 每日 15:10/16:30 收盘后自动增量更新行情
systemctl start quant-data-refresh.timer

# 手动跑一次数据更新
python scripts/refresh_data.py
```

### 本地启动（Windows）

本地 Windows 环境可用 `scripts/start_backend_with_sync.ps1` 一键启动后端 API (:17891) +
CNE 数据湖看板 (:8787)。脚本会先通过 SSH 访问服务器 `data_status` 服务的
`GET /api/status`，确认服务器数据状态正常后执行只拉取同步（仅 `server_to_local`），
服务器不可达或状态异常时不会阻塞本地 API 启动。

```powershell
# 一键启动（后端 + CNE dashboard）
.\scripts\start_backend_with_sync.ps1

# 仅后端
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 17891

# 只检查、不写入本地数据
.venv\Scripts\python.exe scripts\startup_remote_sync.py --dry-run
```

脚本默认使用 `scripts/sync_manifest.json` 中的 SSH 别名和服务器地址；可通过
`QUANT_REMOTE_HOST`、`QUANT_REMOTE_STATUS_URL` 覆盖。同步过程仍由现有
`scripts/sync_server.py` 负责，覆盖前会备份到 `data/sync_backup/` 并做 SHA256 校验。

`systemd` 单元见 `systemd/`。开发调试：

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 17891 --reload
```

### 数据目录与路径环境变量

所有数据/目录路径均不再硬编码用户主目录，由以下环境变量驱动（`.env` 或
systemd `Environment=` 均可配置）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QUANT_UI_DATA_DIR` | `<项目>/data` | 主数据目录：panel/ETF/基金/duck.db/pg_parquet 等 |
| `QUANT_UI_LEGACY_DATA_DIR` | `~/quant_data` | 旧独立数据目录（selftest/compare_old 对照用） |
| `QUANT_UI_SENTIMENT_DIR` | `<项目父目录>/sentiment-mvp` | 舆情独立仓库 |
| `QUANT_UI_QQBOT_DIR` | `~/qqbot` | QQ 机器人凭据/推送脚本目录 |

迁移数据时只需把 `data/` 移动到目标位置并设置 `QUANT_UI_DATA_DIR`，
代码不再依赖 `/home/ubuntu/...` 绝对路径。

### 3.1 Docker 部署（单容器全栈，可移植）

镜像内**不含任何业务数据**：行情数据、回测结果、策略代码、日志全部走 volume，
数据可随 volume 整体迁移，不依赖宿主机路径，amd64/arm64 均可构建。

```bash
# 前置：已安装 Docker（Desktop 或 Docker Engine）

# 1. 准备 .env（含 Tushare token；另建议设置 CNE_TOKEN 作为数据湖看板令牌）
cp .env.example .env

# 2. 构建并启动（首次构建需下载依赖，约 10-20 分钟）
docker compose up -d --build

# 3. 访问
#    Vue 工作台        http://<host>:17891
#    数据状态看板      http://<host>:8001
#    CNEquity 数据湖   http://<host>:8787   （非本机访问需 ?token=<CNE_TOKEN>）
```

容器内通过 supervisord 管理 4 组进程（FastAPI :17891 / data_status :8001 /
CNEquity 看板 :8787 / supercronic 定时任务），定时任务按 Asia/Shanghai 时区运行。

**数据卷与迁移**（关键设计）：

| 卷 | 容器路径 | 内容 |
| --- | --- | --- |
| `quant_data` | `/app/data` 与 `/app/CNEquity/data` | 行情 parquet、quant.db、DuckDB、CNE 数据湖、data_status 状态 |
| `quant_labs` | `/app/labs` | 用户保存的策略代码 |
| `quant_results` | `/app/results` | 回测/归因/参数扫描结果 |
| `quant_artifacts` | `/app/artifacts`、`/app/logs` | alphaagent 因子挖掘产物与日志 |

`quant_data` 同时挂载到 `/app/data` 与 `/app/CNEquity/data`，使 CNE 数据湖
（`CNEquity/data/quant_dataset/_cnequity`）与 quant_ui 数据根
（`data/quant_dataset`）指向同一棵数据树——**迁移整机只需搬这一个 volume**。

```bash
# 迁移示例（在源/目标机器上分别执行）
docker run --rm -v quant_data:/data -v "$PWD":/backup alpine tar czf /backup/quant_data.tar.gz -C /data .
# 目标机器：docker volume create quant_data && docker run --rm -v quant_data:/data -v "$PWD":/backup alpine tar xzf /backup/quant_data.tar.gz -C /data

# 首启无数据时，可从宿主机直接拷贝进卷：
docker cp data/. quant_ui:/app/data/
```

`.env`（含 token）仅通过 `env_file` 注入，绝不进入镜像；`CNEquity/` 源码目录
以只读 bind 挂载，改配置后 `docker compose restart` 即生效。运维与排障：

```bash
docker compose logs -f --tail=200        # 查看四组进程日志
docker compose ps                        # 进程状态（supervisord 统一管理）
docker exec -it quant_ui supervisorctl status
docker exec -it quant_ui python scripts/refresh_data.py   # 手动数据刷新
docker compose down                       # 停服（不删卷；加 -v 会清空数据）
```

## 4. 数据架构

- 主面板：`data/panel.parquet`（267 万行，1800 只，2015-09 ~ 2026-08，日线 + 滚动因子）
- ETF：`data/etf.csv`（1576 只全市场 ETF）+ `data/etf_panel.parquet`（123 万行日线，
  腾讯行情增量拉取，字段与股票面板一致）
- 场外基金：`data/fund.csv`（1350 只科技相关权益基金，关键词过滤 + 剔除场内 ETF）
  + `data/fund_nav.parquet`（105 万行单位净值，天天基金逐只抓取）
  + `data/fund_fee.csv`（申购、管理、托管、销售服务及赎回费率，AkShare/Eastmoney）
- 行情源：股票优先 Tushare（`.env` 配 token，请求限速），失败自动回退腾讯行情接口；
  ETF 走腾讯行情、场外基金走天天基金净值
- 本地缓存：`data/universe.csv`、`data/tech.csv`、`data/index.csv`、`data/meta.json`
- CNEquity 日线档案：`scripts/sync_daily_to_cne.py` 委托 CNE fetcher 按交易日增量写入
  `data/quant_dataset/YYYY/YYYY/day/stock_daily.parquet`（不复权原始价 + 每日复权因子）；
  财务/公告/调研宽表继续按股票同步到 `data/pg_parquet/`，由 CNE external bridge 只读接入
- 业务数据统一 SQLite：`data/quant.db`（回测归档、账本、模拟盘、策略池等），
  不再依赖 PostgreSQL/TimescaleDB
- 回测数据源由 `QUANT_DATA_SOURCE` 控制：`cne` 读取 `data/quant_dataset/` 年度日线档案；
  `pg_parquet` 是兼容别名，财务宽表仍从 `data/pg_parquet/` 只读加载；`panel` 只用本地预计算面板
- 指数日线：`data/indices.parquet`，API `/api/data/indices` + `/api/data/indices/series`
- 舆情：独立仓库 `sentiment-mvp`，FastAPI 后端读取其 CSV/数据库

## 5. 回测引擎

### 统一交易参数（`core/trading_config.py`）
- 全平台唯一默认交易参数来源（散户口径）：资金 10 万、买入佣金万 2.5、卖出费率万 12.5（含印花税）、
  滑点 0bps（散户小资金低频日频，滑点不计）、参与率 10%、选股 top 0.4%（约 20 只）、流动性下限 500 万
- 回测引擎、模拟盘、因子门禁、前端默认值均从此取，改参数只需改这一个文件
- 代码面板参数副本（`trading_config_store.py`）：代码 tab 可独立覆盖全局默认参数，不影响其他页面
- 因子门禁专用参数：top 0.1% ≈ 5 只（10 万资金按 2 万/只覆盖中低价股），净值超额年化 ≥3%、
  超额夏普 ≥0.5、回撤 ≤40%、持仓重叠 ≥50%、仓位利用率 ≥80%

### 因子轮动（`core/engine/` 包）
- 原 `core/engine.py` 已拆分为四阶段管线包：`config → prepare → factor_matrix → simulate → result`
- 月频/周频选股：按因子得分排序取 TopN（`ascending` 控制买高/买低）
- 选股策略统一走 `SelectionPolicy` + `PortfolioBuilder.build_targets`（`core/selection.py`），
  支持 count_mode（top_n/top_pct）、min/max positions、min_score 门槛、ADX 趋势门控
- 撮合执行层独立（`core/execution.py`）：现金/整手/开盘成交执行器，先卖后买
- 因子注册表单一来源（`core/factor_registry.py`）：量价因子（mom20/mom60/vol20/brk20/brk20_vol/
  turn20/am20/双均线 6 组）、基金专属因子（mdd20/mdd60/sharpe20/sharpe60/sortino20/mom_accel/nav_stability）、
  财务因子（pb/ep/roe/gross_margin/rev_yoy/np_yoy，PIT 对齐）、动态因子（composite/pred）
- 过滤：剔除科创/创业、一手 100 股、成交额分位（`amount_q`）、因子预热（`warmup_days`）
- 组合构建：行业分散（`industry_cap`）、多空对冲（`long_short`/`short_n`/`short_cost_rate`）、行业中性化（`industry_neutral`）
- 财务因子（`use_financial`）：按公告日 point-in-time 对齐避免未来函数；
  数据来自 Tushare parquet 财务宽表（`scripts/sync_tushare_to_parquet.py --fina` 全市场补全）
- 风险中性化（`risk_neutral`）：选股前把因子得分对风格/行业暴露回归取残差，
  并输出期末持仓的**风险归因**（liquidity/momentum/volatility/turnover/value/quality/growth + 行业 + specific）
- 基准：股票池等权；支持沪深300/中证500 等指数线
- 输出：净值/基准/回撤/指标/持仓/调仓记录/最近信号日 + **Brinson 行业归因**（自动），
  `analyze=true` 时附带因子质量（IC/分组）
- 市场冲击模型：`impact_coef` + `impact_vol` 参数支持线性冲击成本

### Screener — regime 感知因子选择（`core/screener.py` + `core/regime.py`）
- 借鉴 AlphaCrafter Screener 流程，用**确定性计算替代 LLM 判断**
- Regime 检测（`regime.py`）：ADX > 25 + 指数 MA60 方向 → TREND_UP / RANGE / TREND_DOWN
- 因子适用性评分：最近 N 日 Rank IC 均值 × regime boost 因子（趋势市动量加权、震荡市反转加权）
- 语义去重：因子族分类 + 相关性矩阵（|corr| > 0.7 视为冗余）
- 权重分配：|IC| 归一化 → 动态 weights/directions 传给 `build_composite_factor`
- 带开关控制（关闭时用固定 weights/directions，原逻辑不变）

### 场外基金回测（`core/fund_engine.py`）
- 基金净值转最小 panel（NAV 同时充当 open/close，turnover/amount 置 1 过滤全通）
- T+1 确认近似：signal 日 NAV(t) → exec 日 NAV(t) 成交
- 全市场交易日历逐代码 ffill 估值净值（避免海外 QDII 日历不一致）
- 申赎费率从 `data/fund_fee.csv` 读取

### 风险模型（`core/risk_model.py`）
- 轻量 Barra：风格因子（流动性/动量/波动/换手 + 可选价值/质量/成长）+ 行业哑变量
- 逐期横截面回归估计因子收益，因子协方差用 LedoitWolf 收缩（退化时常数收缩近似）
- 资产协方差 = 暴露×因子协方差×暴露' + diag(specific)；组合风险按因子分解
- `neutralize()` 用于风险中性化选股

### 滚动训练-测试（`core/walkforward.py`）
- `rolling_train_test_factor` / `rolling_train_test_event`：每个测试窗口用**之前全部历史**
  在参数网格（因子：top_n×频率；双均线：short×long）按夏普选参，再跑当前窗口做样本外验证
- 输出逐窗口指标 + 参数选择历史（train_sharpe），与纯 walk-forward 区分「过拟合」与「稳健」

### 日级模拟盘（`core/paper/` 包）
- 原 `core/paper.py` 已拆分为 `account.py` / `details.py` / `rebalance.py` / `storage.py` 子模块
- 账户：`paper_accounts`（策略/股票池/资金/TopN/频率/风控参数），PG 优先、JSON 回退
- 语义与回测引擎一致：信号日（T-1）收盘生成目标持仓 → T 日开盘成交 → 收盘估值
- 撮合：先卖后买、一手 100 股、买卖费用、涨跌停/停牌/成交额分位过滤、现金约束
- 风控：单票权重上限 `max_weight`、流动性分位 `amount_q`，拒单原因落 `paper_events`
- 事件策略账户：选择代码实验室已保存模块 + `EVENT_STRATEGIES`，采用**全量重放**模式，
  从账户 `start_date` 起回放 `run_event_backtest`；空头暂不支持
- 幂等：同一 exec_date 重复执行自动跳过；订单表带唯一约束
- 调度：`systemd/quant-paper.timer` 每天 15:40（数据刷新 15:20 之后）自动执行
- 手动：`scripts/paper_trade.py --run [--account N] [--date YYYY-MM-DD] [--dry-run]`
- 创建事件账户：`scripts/paper_trade.py --create --strategy-type event --module labs/xxx.py --event-strategy "双均线金叉事件" [--start-date YYYY-MM-DD]`

### 事件驱动（`core/event_engine/` 包）
- 原 `core/event_engine.py` 已拆分为 `context.py` / `runner.py` / `strategies.py`
- 每个交易日 `on_bar(ctx, bar)`；信号日收盘 → 执行日（T+1）开盘成交
- 撮合语义：一手 100 股、先卖后买、停牌不可交易、**涨跌停约束**（`core/limit.py`）
- 成本：买入 `buy_cost`、卖出 `sell_cost`（默认从 `trading_config.py` 取），每日收盘估值
- 增强：滑点（`slippage_bps`）、流动性约束（`max_participation`）、限价单、空头融券费率、市场冲击模型
- ctx API：`history(code, fields, window)` 多字段历史 DataFrame、`close_series(code)` 收价序列、
  `available_fields` 可用字段列表
- 组合优化 ctx API：`optimize_risk_parity` / `optimize_mean_variance` / `optimize_max_diversification`
- 内置示例：`GoldenCrossStrategy`、`RiskParityStrategy`、`LongShortMomentumStrategy`

### 策略池（`core/strategy_pool.py`）
- 全量池 = `strategies.registry.STRATEGIES` ∪ `backtest_runs` 出现过的策略
- 配置池 = `strategy_pool` 表，回测/模拟盘/信号页策略下拉框以配置池优先
- 支持回收站（软删除 + 恢复）、拖拽排序；PG 不可用时回退到注册表全量

### 一致性验证
`scripts/selftest.py` 内置 14 组对照：长期 2020-2026 / 近半年，7 个同名策略，
结果与旧引擎 `quant_3stocks/outputs/backtest_5w.csv` 的收益/年化/夏普/回撤
**14/14 完全一致**（误差 < 0.01pp）。短窗口建议默认开 400 天因子预热。

## 6. 策略注册表（`strategies/registry.py`）

| 分组 | 策略 |
| --- | --- |
| 冷门/价值 | 低换手冷门、低成交冷门、冷门+行业分散 |
| 动量/趋势 | 高成交领涨、动量 20 日、动量 60 日、每半年动量 Top1、每半年动量 Top3、趋势突破 20 日（brk20）、放量突破 20 日（brk20_vol）、双均线多头 5/10 · 5/20 · 10/30 · 20/60 · 20/30 · 20/30 ADX25 |
| 反转/均值回归 | 反转 20 日、双均线反转 5/10 · 5/20 · 10/30 · 20/60 |
| 波动/风控 | 低波动 |
| 复合/综合 | 复合因子 |
| 多空/对冲 | 多空动量 20 日、多空低换手 |
| ML/预测 | ML 预测 Top、ML 预测 Bottom |
| 基金/风控 | 基金低回撤、基金高夏普、基金高 Sortino |
| 基金/动量 | 基金动量加速 |
| 基金/综合 | 基金复合因子 |

> 策略按 `types` 字段限定资产类型（stock/etf/fund），前端按资产类型过滤下拉。
> 因子注册表（`core/factor_registry.py`）是因子元数据单一事实来源，
> `STRATEGIES` 引用的因子名在 `validate_registry_factors()` 校验。

### 6.1 米筐对齐（`strategies/ricequant/`）

「动量 20 日 · 科技TMT · Top3 月度」策略与米筐在线回测平台做引擎结果对齐验证：
- `mom20_top3_techtmt.py`：粘贴到米筐策略编辑器的源码
- `scripts/ricequant_picks_export.py`：本地引擎基准导出脚本
- `compare_run.py`：三层对比工具（净值/逐笔/持仓）

## 7. AlphaAgent 因子自主挖掘（`alphaagent/`）

LLM 驱动的多轮对话循环，自动完成「生成因子 DSL → 训练集评估 → 验证集检验 →
相似度去重 → 入库交付」闭环。完整架构见 `AGENTS.md`，此处只列要点。

### 7.1 工作方式

```
用户下达研究方向（Web / CLI）
  → LLM 每轮发起 tool_calls：evaluate_factor（DSL → IC/ICIR/覆盖度/月度稳健性）
  → 通过海选门槛后 submit_factor 入库
     stage_one 统计门槛 + 正式库相似度（|ρ|≥0.5 拒绝）
     → 候选池 → stage_two 精筛（双窗口口径）→ engine_gate 回测门禁（实盘可交易性裁决）
     → 正式库
  → FactorReviewer LLM 审查（reject 硬拦）+ 长期研究记忆（BM25 正/负证据）
```

- **三层约束**：经济直觉强制（先写因果链）→ 父本变异策略（参数/算子/修饰）→ 正交预判（与已有因子截面 ρ>0.7 拦截）。
- **数据**：`panel_path=cne://` 时从 CNE 数据湖实时构建面板（磁盘缓存秒级命中）；基本面模式自动载入 `funda_*` PIT 财务字段。
- **DSL**：多行表达式编译为 Python，`$列名` 引用面板列；支持关键字参数语法（`n_bins=8`）；慢算子走 Numba 并行内核，有三重门禁（静态扫描/性能预算/数值一致性）。
- **评估**：冻结 `EvaluationProfile`（transform → metric → rule 管线），分位数纯多头口径回测。
- **IC 前置过滤**：评估 metrics 循环前先只算截面 IC 检查规则，不通过即短路返回
  （跳过昂贵的 mls/portfolio/月度指标），显著缩短挖掘迭代；因子实验室路径保留全部指标。
- **记忆系统 v3**：三层研究记忆（原始证据层 `memory_entries` + SSPM 编辑统计层 `memory_cells` +
  经验蒸馏层 `memory_experience`），BM25 检索 + 族亲和 + 算子重叠注入；
  基于 AlphaMemo 论文的置信度门控（Eq.7/APV 双门），正证据软推荐、负证据防重复无效路径，
  支持跨面融合与编辑指纹去重。
- **入库门槛**：唯一真源在 `delivery_criteria`，按研究模式（technical/fundamental）存为
  `artifacts/alphaagent/research_specs/<mode>.json` 增量覆盖，Web 门槛弹窗编辑后挖掘/晋升/CLI 全链路生效。
- **因子值缓存**：表达式+面板指纹的内存 LRU + 磁盘持久化（`artifacts/factor_value_cache/`），跨会话复用。
- **算子耗时监控**：DSL 求值零侵入计时，累计写 `artifacts/dsl_operator_profiling.jsonl`。
- **ML 组合训练（stacking）**：Ridge/LightGBM 组合打分，因子库 → 时间序列交叉验证 + purge 隔离 →
  组合因子回灌回测引擎；子进程执行 + 进度日志，API 管理（`/api/alphaagent/stacking/*`）。

### 7.2 模块结构

```
alphaagent/factor/mining/          挖掘主循环与工具
├── agent/        AgentScope 运行（loop/run/agentscope_run/tools/reviewer/preflight/population）
├── delivery/     两阶段入库（submit + delivery_checker + delivery_criteria + engine_gate）
├── eval/         评估服务（service/session/schemas/context/response/mls_thresholds/param_stability）
├── infra/        基础设施（config/audit/cli_stream/console/registry_io/remove）
├── memory/       研究记忆 v3（store/retrieval/calibration/advisory/experience/ingestion/schema/...）
├── prompt/       Prompt 模块（prompts/prompt_modules/seed_factors/operators/modules/）
└── tools/        FactorEvalTools（_dispatch/_prefilter/_analysis/_schemas）
```

### 7.3 启动方式

```powershell
# Web：AlphaAgent 页 → 新建研究任务（研究模式选 日线技术 / 基本面）
# CLI：直接启动挖掘（默认 cne:// panel + production_technical 因子库）
.venv\Scripts\python.exe scripts\run_alphaagent.py

# 候选池重放晋升（候选池已有因子重走修复后的两阶段链路）
.venv\Scripts\python.exe scripts\promote_candidates.py

# 候选池两两相关去重（先 --dry-run 看报告）
.venv\Scripts\python.exe scripts\dedup_candidate_factors.py --dry-run

# 盲测段因子一次性离线重测（锁定段：挖掘循环与入库门槛从未见过该段数据）
.venv\Scripts\python.exe scripts\blind_test_factors.py
```

依赖见 `requirements-alphaagent.txt`（agentscope、openai、numba、jieba）；
LLM 凭据支持 Codex provider（`ALPHA_LLM_PROVIDER=codex`，读 `~/.codex/config.toml`）。

## 8. 后端 API（FastAPI :17891，文档 `/docs`）

```
# ── 数据 ──
GET  /api/data/status            数据缓存状态
GET  /api/data/panel-info        面板概况
POST /api/data/update            触发数据更新 {mode, end}
GET  /api/data/update/status     更新进度
GET  /api/data/update-configs    数据更新任务配置列表
POST /api/data/update-configs/{task_id}  修改任务配置
GET  /api/data/indices           指数列表
GET  /api/data/indices/series    指数日线序列

# ── 回测/策略/因子 ──
GET  /api/strategies             策略列表（按资产类型过滤）
GET  /api/factors                可用因子列表（组合编辑器用）
GET  /api/alpha-factors          AlphaAgent 因子目录
GET  /api/composites             已保存的多因子组合
POST /api/composites             保存组合 {name, weights, directions}
DELETE /api/composites/{name}    删除组合
GET  /api/names                  股票代码→名称映射
POST /api/backtest               单策略回测
GET  /api/signals                今日信号（GET 查询参数）
POST /api/signals                今日信号（Body，支持 composite/long_short）
GET  /api/signals/alpha          AlphaAgent 因子今日信号
POST /api/signals/orders         今日信号 + 订单明细
POST /api/sweep                  参数稳健性扫描（event / factor / rolling / rolling_event）
POST /api/backtest/quantstats    生成 QuantStats HTML 报告
POST /api/backtest/attribution   事件策略 Brinson 归因
GET  /api/backtest/runs          回测归档列表
GET  /api/backtest/runs/{id}     归档详情
DELETE /api/backtest/runs/{id}   删除归档

# ── 个股 ──
GET  /api/stock/search           按代码/名称搜索
GET  /api/stock/{code}           个股历史行情 + 指标（adj=qfq|hfq|raw）

# ── 策略池 ──
GET  /api/strategy-pool          配置池列表
GET  /api/strategy-pool/full     全量池（注册表 + 归档）
GET  /api/strategy-pool/trash    回收站
POST /api/strategy-pool/add      添加策略到配置池
POST /api/strategy-pool/remove   移到回收站
POST /api/strategy-pool/restore  从回收站恢复
POST /api/strategy-pool/full-delete  永久删除
POST /api/strategy-pool/trash/purge   清空回收站
POST /api/strategy-pool/trash/empty   清空回收站（别名）
POST /api/strategy-pool/reorder  拖拽排序

# ── 代码实验室 ──
GET  /api/code/saved             已保存代码列表
GET  /api/code/saved/{name}      读取已保存代码
POST /api/code/save              保存代码到 labs/
GET  /api/code/config            代码面板生效参数（全局默认 + 副本覆盖）
POST /api/code/config            保存代码面板参数副本
POST /api/code/config/reset      重置为全局默认
POST /api/code/jq/preflight      聚宽策略 API 预检（AST 静态扫描）
POST /api/code/jq/run            聚宽模式同步回测
POST /api/code/jq/run_async      聚宽模式异步回测（边跑边画图）
GET  /api/code/jq/runs           聚宽回测历史列表
GET  /api/code/jq/runs/{id}      聚宽回测详情（含净值流）
POST /api/code/jq/runs/{id}/stop 停止聚宽回测
GET  /api/code/qweave/default    qweave 研究默认代码
GET  /api/code/qweave/templates  qweave 模板列表
GET  /api/code/qweave/template   qweave 模板代码
POST /api/code/qweave/parse      解析 qweave 代码
POST /api/code/qweave/run        运行 qweave 研究脚本

# ── 账户/模拟盘 ──
GET  /api/ledger/transactions    交易流水（POST 录入）
GET  /api/ledger/deposits        出入金（POST 录入）
GET  /api/ledger/equity          每日估值曲线
GET  /api/ledger/positions       当前持仓与盈亏
GET  /api/paper/accounts         模拟盘账户列表（POST 创建 / DELETE 删除）
POST /api/paper/accounts/{id}/reset  重置账户
POST /api/paper_run              手动执行模拟盘
GET  /api/paper/accounts/{id}/summary|orders|trades|positions|equity|events
GET  /api/paper/event-strategies 列出代码实验室已保存模块中的事件策略

# ── 舆情 ──
GET  /api/sentiment/status       舆情数据状态
GET  /api/sentiment/stats        舆情统计（新闻数/标签分布/每日条数）
GET  /api/sentiment/news         情绪最强/最弱新闻
GET  /api/sentiment/ic           舆情分桶回测 IC/分组

# ── AlphaAgent（/api/alphaagent/*）──
POST /api/alphaagent/runs          启动新挖掘（research_mode: technical/fundamental）
GET  /api/alphaagent/runs          列出所有 runs
GET  /api/alphaagent/runs/{id}     run 详情（事件轨迹尾部）
POST /api/alphaagent/runs/{id}/stop        终止
POST /api/alphaagent/runs/{id}/messages    向运行中的进程注入消息
POST /api/alphaagent/runs/{id}/continue    已结束的 run 继续（fork 新进程）
POST /api/alphaagent/runs/{id}/branch      分支新 run
POST /api/alphaagent/runs/{id}/rename|archive|pin  重命名/归档/置顶
DELETE /api/alphaagent/runs/{id}   删除 run（DELETE /runs/archived 清理归档）
GET  /api/alphaagent/runs/{id}/events   SSE 事件流（实时轨迹）
GET  /api/alphaagent/research-memory      查看研究记忆（DELETE /{entry_id} 删单条）
GET  /api/alphaagent/research-memory/layers  记忆分层明细（SSPM cells + 经验层）
GET  /api/alphaagent/research-modes       研究模式选项
GET  /api/alphaagent/windows              统一时间窗口默认值（train/val/test）
GET  /api/alphaagent/research-spec/default  当前模式生效研究策略
GET  /api/alphaagent/research-specs/{mode}  模式门槛三视图 defaults/overrides/effective
PUT  /api/alphaagent/research-specs/{mode}  保存门槛（diff 增量覆盖，全链路生效）
DELETE /api/alphaagent/research-specs/{mode}  删除门槛文件，恢复注册表默认
GET  /api/alphaagent/evaluation-capabilities  可用评估插件和 profiles
GET  /api/alphaagent/factors|factors/{id}   因子列表/详情（library=production/candidate）
POST /api/alphaagent/factors       保存因子到库
DELETE /api/alphaagent/factors/{id}  删除因子
POST /api/alphaagent/eval-factor    单因子评估（因子实验室）
POST /api/alphaagent/backtest-factor  因子回测
POST /api/alphaagent/stacking/train           启动 ML 组合训练
GET  /api/alphaagent/stacking/trainings       训练列表
GET  /api/alphaagent/stacking/trainings/{id}  训练详情（含进度日志）
POST /api/alphaagent/stacking/trainings/{id}/stop  停止训练
GET  /api/alphaagent/session-cache/stats     会话缓存统计
POST /api/alphaagent/session-cache/evict     清空会话缓存
GET  /api/alphaagent/dsl-monitor    DSL 算子耗时榜（top_k / since_hours）
GET  /api/alphaagent/logs|logs/tail  运行日志
```

## 8.5 聚宽策略兼容层（JQ shim，`core/event_engine/jq/`）

对标聚宽策略 API 的本地回测层，**代码面板可直接运行聚宽策略**，按聚宽官方文档类别拆分：

```
core/event_engine/jq/
├── api/                # 命名空间装配（对应聚宽文档类别）
│   ├── framework.py    # 生命周期：run_daily/run_weekly/run_monthly/unschedule_all
│   ├── settings.py     # set_option/set_benchmark/set_slippage/set_order_cost
│   ├── trading.py      # order 四件套/cancel_order/get_orders/get_trades
│   ├── data_api.py     # get_price/history/get_fundamentals/get_industry/get_index_stocks
│   ├── portfolio.py    # set_subportfolios/transfer_cash（单账户兼容）
│   ├── finance.py      # 财务指标表（indicator）
│   └── misc.py         # jqdata/jqfactor 模块桩、display、旧版兼容
├── objects.py          # Context/Portfolio/Position/Order/Trade 壳视图
├── query.py            # get_fundamentals 的 query DSL（valuation/income 列过滤）
├── runtime.py          # exec 用户代码 → 生命周期编排 → 撮合对接
├── entry.py            # run_jq_backtest 入口（干跑采集费率 → 引擎回测）
├── preflight.py        # API 预检（AST 静态扫描，秒级报告缺失 API/字段）
├── factor_bridge.py    # get_factor：AlphaAgent DSL 因子在策略内调用
└── datalake/           # CNE curated 数据接入插件体系
    ├── __init__.py     # 注册中心 + discover_cne_curated() 自动注册
    ├── base.py         # JQDataPlugin 基类
    └── plugins/        # 专属插件（stock_daily/income/balancesheet/cashflow/
                        #   fina_indicator/index_bars/industry_members）
```

- **数据接入插件体系**：启动时自动把 CNE curated 全部目录批量注册为通用插件
  （`load("dragon_tiger")` == `load("cne:dragon_tiger")`），已有表**零代码接入**；
  需专属清洗/换算的表才写专属插件。调用未注册名抛**带扩展指引**的 NotImplementedError。
- **API 预检**（`preflight.py`）：AST 静态扫描，秒级报告缺失的聚宽 API/字段，
  避免"跑 90 秒后炸在缺失 API 上"的验证循环。
- **分钟线接入**：盘中价查询与下单时点真实价撮合；`set_option('avoid_future_data')` 真生效，
  数据 API 拒绝未来请求（聚宽同语义报错）。
- **回测异步化**（`/api/code/jq/run_async`）：后台线程启动，边跑边画图 + 阶段进度/ETA + 可停止。
- 验证脚本：`scripts/jq_repro/_test_datalake.py`（插件体系）、`_test_p0_api.py`（API 面）、
  `_validate_jq_compat.py`（小盘三正基线）等；聚宽官方文档快照 `docs/jq_api_snapshot/api_full.html`。

## 9. 脚本工具

| 脚本 | 作用 | 输出 |
| --- | --- | --- |
| `scripts/refresh_data.py` | 行情增量更新 + CNE 年度档案同步 | `data/stock/panel.parquet` |
| `scripts/selftest.py` | 新旧引擎一致性对照 | 控制台报告 |
| `scripts/qweave_research.py` | **研究层（qweave 替代 Qlib）**：Alpha158/101/191 因子计算 + IC/分组/换手评估 + LightGBM 预测 | `data/qweave/`、`data/pred_demo.parquet` |
| `scripts/performance_report.py` | QuantStats 报告 + IC/分组分析 | `results/performance/` |
| `scripts/parameter_sweep.py` | 参数网格 / walk-forward | `results/parameter_sweep/` |
| `scripts/paper_trade.py` | 日级模拟盘（创建/执行/查询） | SQLite `paper_*` 表 |
| `scripts/attribution.py` | Brinson 归因 | `results/attribution/` |
| `scripts/sentiment_backtest.py` | 舆情分桶回测 + IC/分组 | `results/sentiment_backtest.csv` |
| `scripts/sync_daily_to_cne.py` | CNE 日线年度档案增量同步 | `data/quant_dataset/` |
| `scripts/sync_tushare_to_parquet.py` | Tushare 财务/公告宽表同步 | `data/pg_parquet/` |
| `scripts/refresh_etf.py` | ETF 行情增量更新 | `data/etf_panel.parquet` |
| `scripts/refresh_fund.py` | 场外基金净值增量更新 | `data/fund_nav.parquet` |
| `scripts/refresh_fund_fees.py` | AkShare 基金费率补齐 | `data/fund_fee.csv` |
| `scripts/healthcheck.py` | 健康检查 | — |
| `scripts/dedup_candidate_factors.py` | 候选因子库去重（两两截面 Pearson，阈值 0.6） | 清理 registry + DSL |
| `scripts/alpha191_screen.py` | GTJA Alpha191 全量批量筛选 | 控制台报告 |
| `scripts/ricequant_picks_export.py` | 本地引擎基准导出（米筐对齐） | `strategies/ricequant/baseline/` |
| `scripts/wf_lowvol_check.py` | 低换手/低波动 walk-forward 样本外验证 | 控制台报告 |
| `scripts/recheck_engine_gate.py` | 重新检查 engine_gate 门禁结果 | 控制台报告 |
| `scripts/run_alphaagent.py` | AlphaAgent 挖掘 CLI 入口 | `logs/factor_mining/` |
| `scripts/promote_candidates.py` | 候选池重放晋升（重走修复后的两阶段链路） | 候选库 → 正式库 |
| `scripts/blind_test_factors.py` | 盲测段因子一次性离线重测（锁定段，克制使用） | `artifacts/alphaagent/blind_test/` |
| `scripts/rescreen_candidates.py` | 候选池重新过门槛筛选 | 控制台报告 |
| `scripts/check_dsl_slow_patterns.py` | DSL 慢算子静态拦截（AST 扫描） | 失败即报 |
| `scripts/train_ml_composite.py` | 训练 ML 组合打分（stacking） | 组合因子 |
| `scripts/engine_baseline.py` | 引擎基线生成 | 基准净值 |
| `scripts/import_local_data.py` | 本地数据导入工具 | — |
| `scripts/start_backend_with_sync.ps1` | 一键启动后端 + CNE 看板（含远程同步） | — |
| `scripts/startup_remote_sync.py` | 远程数据同步（支持 --dry-run） | — |

### 9.1 qweave 研究层（替代 Qlib）

研究层使用 [qweave](https://github.com/qweave/qweave)（Polars/Rust 原生），
直接读 `data/pg_parquet/stock_daily.parquet`（与回测面板同口径），
不需要 pyqlib / qlib_data 二进制。内置 Qlib Alpha158、WorldQuant Alpha101、
GTJA Alpha191 三套因子表达式，并支持 IC/分组/换手评估与 LightGBM 预测。
前端代码面板支持 qweave 模板选择与在线运行（`/api/code/qweave/*`）。

```bash
# 全市场 Alpha158 评估（universe.csv 股票池，2020 至今）
python scripts/qweave_research.py

# 训练 LightGBM 并导出预测分数 -> data/pred_demo.parquet
python scripts/qweave_research.py --start 2022-01-01 --train-model

# 预测分数回灌现有回测引擎：策略「ML 预测 Top / ML 预测 Bottom」，
# 或组合因子中使用 pred（如 {"pred": 0.6, "mom20": 0.4}）
```

### 9.2 自动化测试（pytest）

`core/engine/`（因子轮动）与 `core/event_engine/`（事件驱动）已带 pytest
冒烟测试，使用合成小面板，不依赖真实行情：

```bash
.venv\Scripts\python.exe -m pytest tests -v
```

CI 配置见 `.github/workflows/ci.yml`，在 push/PR 时自动安装
`requirements-dev.txt` 并执行上述冒烟测试。

## 10. 登录鉴权

当前已临时关闭，前后端免登录。鉴权代码保留在 `backend/auth.py`
（pbkdf2 + HttpOnly Cookie），需要时在 `backend/main.py` 重新挂载
中间件即可恢复。

## 11. 开发状态

- 数据→因子→回测→报告闭环已跑通，绩效/归因（IC/分组/Brinson/QuantStats）已进 UI；
- Barra 风格风险模型、财务因子、walk-forward 滚动训练-测试、日级模拟盘（T+1
  撮合/费用/风控/自动日结）均已上线；
- Screener（regime 感知因子选择，确定性规则替代 LLM）、策略池管理、个股详情页均已上线；
- AlphaAgent 自主因子挖掘已上线（§7），候选池/正式库/研究记忆 v3/门槛文件/ML 组合训练齐备；
- **聚宽策略兼容层（§8.5）** 已打通：代码面板可直接运行聚宽策略，含订单/数据/设置/组合
  全套 API + datalake 数据接入插件体系 + API 预检 + 分钟线接入 + 异步回测；
- **AlphaAgent 增强**：IC 前置过滤加速评估、盲测段锁定诚实样本外、DSL 关键字参数支持、
  DSL 慢算子三重门禁、记忆系统 v3（SSPM 编辑统计 + 经验蒸馏 + 跨面融合）、
  ML 组合训练（stacking）。

仍为代码级（无需 UI）：组合优化权重方法 `ctx.optimize_risk_parity /
mean_variance / max_diversification` 是事件策略内 API，在代码实验室的
`on_bar` 中直接调用。

## 12. 对标市面个人量化平台

| 能力域 | 本平台 | 聚宽/米筐/掘金 | Qlib/backtrader | 差距与方向 |
| --- | --- | --- | --- | --- |
| 数据 | 腾讯+Tushare 增量缓存 + CNE 数据湖（日线/财务 PIT funda_*），日线+滚动因子 | 商业全量数据（分钟/财务/事件） | 自备数据 | 缺全量分钟线（已支持盘中价查询） |
| 研究环境 | 网页代码实验室 + qweave 因子研究脚本 + AlphaAgent 因子 DSL/因子库 + 聚宽兼容层 | Notebook + 因子库 | Notebook/脚本 | 缺 Notebook，qweave 已内置 Alpha158/101/191 |
| 回测 | 因子轮动 + 事件驱动 + 聚宽兼容回测，T+1/涨跌停(ST 5% 区分)/费用/滑点/多空 + 财务因子 | 成熟撮合 + 多周期 | 成熟 | 撮合近似（无分钟级回测撮合） |
| 组合构建 | 等权/风险平价/均值方差/最大分散化 + Barra 风格风险模型 + Screener 动态权重 | 优化器 + 风险模型 | 部分 | 风格因子为轻量代理定义，缺完整 Barra 行业/风格库 |
| 稳健性 | walk-forward + 参数网格 + 滚动训练-测试 | 参数优化/样本外验证 | 部分 | 训练期无特征工程/MI 选参，仅简单网格 |
| 绩效归因 | UI：IC/分组/Brinson/风险归因/QuantStats | 归因报表 | 部分 | 无选股-行业-风格三维联动报表 |
| 账户/风控 | 手动记账 + 日级模拟盘（T+1 撮合/费用/风控） | 模拟盘/实盘/风控 | 无 | 缺实时行情撮合与实盘 |
| 自动更新 | systemd 定时增量 / Docker 内 supercronic | — | — | 单机部署，无任务队列 |
| 部署 | systemd + 本地进程 / Docker 单容器全栈 | SaaS | 本地库 | 多机编排/云平台托管 |

结论：当前平台适合**个人单机研究**（数据→因子→回测→报告闭环已跑通，
绩效/归因已进 UI）。Barra 风格风险模型、财务因子（CNE 数据湖 PIT）、
归因 UI、滚动训练-测试框架、日级模拟盘、Screener 动态因子选择
与 AlphaAgent 自主因子挖掘（含因子 DSL 与因子库）均已完成。对标商业平台的主要短板剩：
① 模拟盘为日级，无实时行情撮合与实时风控；② 缺 Notebook 交互研究环境；
③ 分钟线仅支持盘中查询/下单时点撮合，尚无分钟级回测撮合。

## 13. 已知限制（Demo）

- 未做多进程/后台任务（除聚宽异步回测和 AlphaAgent 挖掘外），大批量参数扫描同步执行
- 行业分类接口在部分服务器不可达时回退本地缓存，科技股票池为缓存快照
- 行情以 CNEquity 年度档案 `stock_daily` 的**不复权原始价 + 每日复权因子**为准；
  前复权锚点固定在同步快照的最新因子，同一历史日在不同查询区间价格一致。
  前复权在出现新分红/送转时仍会整体重标定（这是前复权定义使然）；若要求历史价
  绝对不随最新行情漂移，个股详情可选择「不复权」或「后复权」口径
  （`/api/stock/{code}?adj=raw|hfq`）。
- ST 涨跌停已按 5% 区分（CNE trading_status 逐日 ST 标记，2016 年前历史窗口
  因 Baostock 源覆盖限制降级为板块近似）
- 舆情数据依赖独立仓库 `sentiment-mvp` 流水线每日更新
- 聚宽兼容层为**单账户**模型（`set_subportfolios`/`transfer_cash` 仅兼容单账户语义），
  空头暂不支持；分钟线为盘中查询/下单撮合，非完整分钟级回测
- 滑点默认为 0bps（散户口径），需在代码面板参数副本中手动开启

> AI生成
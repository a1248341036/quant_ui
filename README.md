# quant_ui — A 股个人量化研究平台

一个自托管的 A 股量化研究/回测 Web 平台：Vue3 单页应用 + FastAPI 后端 +
本地增量行情缓存（Parquet/DuckDB/PostgreSQL），内置因子轮动与事件驱动两套
回测引擎，支持账户记账、今日信号、参数稳健性、历史归档与舆情情绪分析。

- 地址：Vue 工作台 `http://<host>:8000`，Streamlit 看板 `http://<host>:8501`
- 默认免登录（鉴权代码保留，可恢复，见「登录鉴权」）

---

## 1. 功能总览

| 模块 | Vue :8000 | Streamlit :8501 | 说明 |
| --- | :---: | :---: | --- |
| 看板（账户 KPI + 资金曲线 + 策略对比） | ✅ | ✅ | 真实账户为空时显示「未开户」引导 |
| 单策略回测（因子轮动） | ✅ | ✅ | 股票池/策略/TopN/资金/频率/区间/过滤/预热 |
| ETF 池回测 | ✅ | ✅ | 全市场 ETF 池（腾讯日线），涨跌停/停牌过滤自动关闭 |
| 场外科技基金回测 | ✅ | ✅ | 科技相关场外基金池（天天基金净值），T+1 净值执行 + 申赎费 |
| 多空对冲 + 行业中性化 | ✅ | ✅ | 多头 TopN + 空头最弱 N 只，模拟融券费率 |
| 多因子组合（权重/方向/保存/回测/信号） | ✅ | — | 因子自由组合打分 |
| 事件驱动策略实验室 | ✅ | ✅ | 按策略类型生成差异化模板 + FactorKit 数据/日期封装 |
| 参数稳健性 / Walk-forward | ✅ | ✅ | 双均线参数网格 + 分窗口回测 + 滚动训练-测试 |
| 历史回测归档 | ✅ | ✅ | PG `backtest_runs` 记录参数/指标/净值/交易 |
| 今日信号 | ✅ | ✅ | 按因子打分输出当日候选 |
| 日级模拟盘 | ✅ | ✅ | 因子/事件策略自动下单/成交/日结，systemd 盘后执行 |
| 账户记账（出入金/交易/持仓/估值） | ✅ | ✅ | 手动录入，估值基于 panel 行情 |
| 数据管理（状态/一键更新） | ✅ | ✅ | 腾讯行情 + Tushare 双源，增量刷新（含 PG 日线同步与基金面板重建） |
| 舆情情绪看板 | ✅（简版） | ✅ | 读取 `~/quant/sentiment-mvp` 词典/LLM 打分 |
| 因子质量分析（IC/分组/多空价差） | ✅ | — | 回测页勾选「因子质量分析」后展示 |
| QuantStats 绩效报告 | ✅ | — | 回测结果页按钮，内嵌 HTML 报告 |
| Brinson 归因 | ✅ | ✅ | 回测结果内置行业归因 + 代码实验室一键归因 |
| 事件引擎费用/滑点/流动性参数 | ✅ | 部分（滑点） | 代码页运行参数可配 |
| 舆情 IC/分组结果 | ✅ | — | 舆情 tab 展示 scripts/sentiment_backtest.py 输出 |

## 2. 架构

```
┌─────────────────────────────┬─────────────────────────────┐
│  Vue3 SPA  :8000            │  Streamlit  :8501           │
│  static/index.html          │  app.py（单页多 tab）        │
│  （FastAPI StaticFiles 托管）│                              │
└─────────────┬───────────────┴──────────────┬──────────────┘
              │ REST (/api/...)              │ 直接调用 core/
┌─────────────▼──────────────────────────────▼──────────────┐
│  FastAPI  backend/main.py :8000                            │
│  routers: backtest / code / ledger / data                  │
│  services: 数据加载、名称/行业映射、系列序列化               │
├────────────────────────────────────────────────────────────┤
│  core/                                                      │
│  engine.py        因子轮动回测 + 今日信号                    │
│  event_engine.py  事件驱动回测 + 撮合 + 组合优化 ctx API     │
│  metrics.py       绩效指标（收益/波动/夏普/回撤/卡玛/胜率）    │
│  performance.py   因子 IC/分组 + QuantStats 报告             │
│  attribution.py   Brinson 归因                              │
│  walkforward.py   参数网格 / 分窗口稳健性                    │
│  portfolio.py     风险平价/均值方差/最大分散化权重            │
│  limit.py         涨跌停约束                                │
│  ledger.py        账户账本                                  │
│  backtest_archive.py 回测结果归档                           │
│  fetcher.py / tushare_client.py  行情抓取（腾讯/Tushare）    │
│  data.py / store.py / db.py / pg.py  数据访问层             │
├────────────────────────────────────────────────────────────┤
│  数据                                                       │
│  data/panel.parquet       日线+因子面板（1800 只）           │
│  data/duck.db             DuckDB 查询缓存/视图              │
│  PostgreSQL/TimescaleDB   stock_daily / backtest_runs / ledger│
│  strategies/registry.py   策略注册表                        │
└────────────────────────────────────────────────────────────┘
```

## 3. 快速开始

依赖：Python 3.12，推荐 venv `/home/ubuntu/stock-analyzer/local_venv`。

```bash
cd /home/ubuntu/quant/quant_ui
cp .env.example .env   # 按需填 TUSHARE_TOKEN / PG_DSN

# 后端 API + Vue 前端（:8000）
systemctl start quant-api
# Streamlit（:8501）
systemctl start quant-ui
# 每日 17:35 自动增量更新行情（含 PG 日线同步与基金面板重建）
systemctl start quant-data-refresh.timer

# 手动跑一次数据更新
python scripts/refresh_data.py
```

`systemd` 单元见 `systemd/`。开发调试：

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
python -m streamlit run app.py --server.port=8501
```

## 4. 数据架构

- 主面板：`data/panel.parquet`（267 万行，1800 只，2015-09 ~ 2026-08，日线 + 滚动因子）
- ETF：`data/etf.csv`（1576 只全市场 ETF）+ `data/etf_panel.parquet`（123 万行日线，
  腾讯行情增量拉取，字段与股票面板一致）
- 场外科技基金：`data/fund.csv`（1350 只科技相关权益基金，关键词过滤 + 剔除场内 ETF）
  + `data/fund_nav.parquet`（105 万行单位净值，天天基金逐只抓取）
- 行情源：股票优先 Tushare（`.env` 配 token，请求限速），失败自动回退腾讯行情接口；
  ETF 走腾讯行情、场外基金走天天基金净值
- 本地缓存：`data/universe.csv`、`data/tech.csv`、`data/index.csv`、`data/meta.json`
- 可选 PostgreSQL/TimescaleDB：`db/docker-compose.yml` + `db/schema.sql`
  - `stock_daily` 日线（hyper table）、`stock_minute` 分钟、财务/公告/舆情宽表
  - `backtest_runs` 回测归档、`ledger_transactions`/`ledger_deposits` 账本
  - 回测数据源由 `QUANT_DATA_SOURCE` 控制：`pg` 优先读 PG，不足自动回退 parquet
- 舆情：独立仓库 `~/quant/sentiment-mvp`，Streamlit 直接读取其 CSV/数据库

## 5. 回测引擎

### 因子轮动（`core/engine.py`）
- 月频/周频选股：按因子得分排序取 TopN（`ascending` 控制买高/买低）
- 过滤：剔除科创/创业、一手 100 股、成交额分位（`amount_q`）、因子预热（`warmup_days`）
- 组合构建：行业分散（`industry_cap`）、多空对冲（`long_short`/`short_n`/`short_cost_rate`）、行业中性化（`industry_neutral`）
- 财务因子（`use_financial`）：PB/EP/ROE/毛利率/营收同比/净利同比，按公告日 point-in-time
  对齐避免未来函数；数据来自 PG 财务宽表（`scripts/sync_postgres.py --fina` 全市场补全）
- 风险中性化（`risk_neutral`）：选股前把因子得分对风格/行业暴露回归取残差，
  并输出期末持仓的**风险归因**（liquidity/momentum/volatility/turnover/value/quality/growth + 行业 + specific）
- 基准：股票池等权；支持沪深300/中证500 等指数线（Streamlit 可选）
- 输出：净值/基准/回撤/指标/持仓/调仓记录/最近信号日 + **Brinson 行业归因**（自动），
  `analyze=true` 时附带因子质量（IC/分组）

### 风险模型（`core/risk_model.py`）
- 轻量 Barra：风格因子（流动性/动量/波动/换手 + 可选价值/质量/成长）+ 行业哑变量
- 逐期横截面回归估计因子收益，因子协方差用 LedoitWolf 收缩（退化时常数收缩近似）
- 资产协方差 = 暴露×因子协方差×暴露' + diag(specific)；组合风险按因子分解
- `neutralize()` 用于风险中性化选股；`core/portfolio.py` 的组合优化器也改用收缩协方差

### 滚动训练-测试（`core/walkforward.py`）
- `rolling_train_test_factor` / `rolling_train_test_event`：每个测试窗口用**之前全部历史**
  在参数网格（因子：top_n×频率；双均线：short×long）按夏普选参，再跑当前窗口做样本外验证
- 输出逐窗口指标 + 参数选择历史（train_sharpe），与纯 walk-forward 区分「过拟合」与「稳健」

### 日级模拟盘（`core/paper.py`）
- 账户：`paper_accounts`（策略/股票池/资金/TopN/频率/风控参数），PG 优先、JSON 回退
- 语义与回测引擎一致：信号日（T-1）收盘生成目标持仓 → T 日开盘成交 → 收盘估值
- 撮合：先卖后买、一手 100 股、买卖费用、涨跌停/停牌/成交额分位过滤、现金约束
- 风控：单票权重上限 `max_weight`、流动性分位 `amount_q`，拒单原因落 `paper_events`
- 事件策略账户：选择代码实验室已保存模块 + `EVENT_STRATEGIES`，采用**全量重放**模式，
  从账户 `start_date` 起回放 `run_event_backtest`，逐笔成交落 `paper_trades`、
  逐日估值落 `paper_equity_snapshots`、最终持仓落 `paper_positions`；空头暂不支持
- 幂等：同一 exec_date 重复执行自动跳过；订单表带唯一约束
- 调度：`systemd/quant-paper.timer` 每天 17:55（数据刷新 17:35 之后）自动执行
- 手动：`scripts/paper_trade.py --run [--account N] [--date YYYY-MM-DD] [--dry-run]`
- 创建事件账户：`scripts/paper_trade.py --create --strategy-type event --module labs/xxx.py --event-strategy "双均线金叉事件" [--start-date YYYY-MM-DD]`

### 事件驱动（`core/event_engine.py`）
- 每个交易日 `on_bar(ctx, bar)`；信号日收盘 → 执行日（T+1）开盘成交
- 撮合语义：一手 100 股、先卖后买、停牌不可交易、**涨跌停约束**（`core/limit.py`）
- 成本：买入 `buy_cost`、卖出 `sell_cost`（默认 8bp / 13bp），每日收盘估值
- 增强：滑点（`slippage_bps`）、流动性约束（`max_participation`）、限价单、空头融券费率
- 组合优化 ctx API：`optimize_risk_parity` / `optimize_mean_variance` / `optimize_max_diversification`
- 内置示例：`GoldenCrossStrategy`、`RiskParityStrategy`、`LongShortMomentumStrategy`

### 一致性验证
`scripts/selftest.py` 内置 14 组对照：长期 2020-2026 / 近半年，7 个同名策略，
结果与旧引擎 `quant_3stocks/outputs/backtest_5w.csv` 的收益/年化/夏普/回撤
**14/14 完全一致**（误差 < 0.01pp）。短窗口建议默认开 400 天因子预热。

## 6. 策略注册表（`strategies/registry.py`）

| 分组 | 策略 |
| --- | --- |
| 冷门/价值 | 低换手冷门、高成交领涨、低成交冷门、冷门+行业分散 |
| 动量/趋势 | 动量 20 日、动量 60 日、双均线多头 5/10 · 5/20 · 10/30 · 20/60 |
| 反转/均值回归 | 反转 20 日、双均线反转 5/10 · 5/20 · 10/30 · 20/60 |
| 波动/风控 | 低波动 |
| 复合/综合 | 复合因子 |
| 多空/对冲 | 多空动量 20 日、多空低换手 |

## 7. 后端 API（FastAPI :8000，文档 `/docs`）

```
GET  /api/health
GET  /api/data/status            数据缓存状态
GET  /api/data/panel-info        面板概况
POST /api/data/update            触发数据更新 {mode, end}
GET  /api/data/update/status     更新进度
GET  /api/strategies             策略列表
GET  /api/factors                可用因子列表（多因子组合用）
GET  /api/composites             已保存的多因子组合
POST /api/composites             保存组合 {name, weights, directions}
DELETE /api/composites/{name}    删除组合
GET  /api/names                  股票代码→名称映射
POST /api/backtest               单策略回测
POST /api/backtest/compare       多策略对比
POST /api/backtest/quantstats    生成 QuantStats HTML 报告
POST /api/backtest/attribution   事件策略 Brinson 归因
GET  /api/signals                今日信号（GET 查询参数）
POST /api/signals                今日信号（Body，支持 composite/long_short）
POST /api/sweep                  参数稳健性扫描（event / factor / rolling / rolling_event）
GET  /api/backtest/runs          回测归档列表
GET  /api/backtest/runs/{id}     归档详情
DELETE /api/backtest/runs/{id}   删除归档
GET  /api/code/default           代码实验室默认模板
GET  /api/code/template?strategy=系统策略模板
GET  /api/code/saved             已保存代码列表
GET  /api/code/saved/{name}      读取已保存代码
POST /api/code/save              保存代码到 labs/
POST /api/code/parse             解析代码里的策略名
POST /api/code/run               运行代码回测（子进程，超时 180s）
GET  /api/ledger/transactions    交易流水（POST 录入）
GET  /api/ledger/deposits        出入金（POST 录入）
GET  /api/ledger/equity          每日估值曲线
GET  /api/ledger/positions       当前持仓与盈亏
GET  /api/paper/accounts         模拟盘账户列表（POST 创建 / PATCH 启停 / DELETE 删除）
POST /api/paper/run              手动执行模拟盘（account_id/exec_date/dry_run）
GET  /api/paper/accounts/{id}/summary|orders|trades|positions|equity|events
GET  /api/paper/event-strategies 列出代码实验室已保存模块中的事件策略
GET  /api/sentiment/status       舆情数据状态
GET  /api/sentiment/stats        舆情统计（新闻数/标签分布/每日条数）
GET  /api/sentiment/news         情绪最强/最弱新闻
GET  /api/sentiment/ic           舆情分桶回测 IC/分组
```

## 8. 脚本工具

| 脚本 | 作用 | 输出 |
| --- | --- | --- |
| `scripts/refresh_data.py` | 行情增量更新 + 可选同步 PG | `data/panel.parquet` |
| `scripts/selftest.py` | 新旧引擎一致性对照 | 控制台报告 |
| `scripts/performance_report.py` | QuantStats 报告 + IC/分组分析 | `results/performance/` |
| `scripts/parameter_sweep.py` | 参数网格 / walk-forward | `results/parameter_sweep/` |
| `scripts/paper_trade.py` | 日级模拟盘（创建/执行/查询） | PG `paper_*` 表 |
| `scripts/attribution.py` | Brinson 归因 | `results/attribution/` |
| `scripts/sentiment_backtest.py` | 舆情分桶回测 + IC/分组 | `results/sentiment_backtest.csv`、`results/sentiment_ic_group.csv` |
| `scripts/sync_postgres.py` | 财务/公告宽表同步 PG | — |
| `scripts/healthcheck.py` | 健康检查 | — |

## 9. 登录鉴权

当前迭代已临时关闭，前后端免登录。鉴权代码保留在 `backend/auth.py`
（pbkdf2 + HttpOnly Cookie），需要时在 `backend/main.py` 重新挂载
中间件即可恢复。

## 10. 接入状态

### 已接入（本次完成）

| # | 功能 | 入口 |
| --- | --- | --- |
| 1 | 因子质量分析（IC/分组/多空价差） | 回测页勾选「因子质量分析」→ 结果区 IC 卡片 + 逐日 IC 图 + 5 分组表 |
| 2 | QuantStats 绩效报告 | 回测结果区「QuantStats 绩效报告」按钮 → 弹层 HTML 报告 |
| 3 | Brinson 归因 | 代码实验室跑完事件策略 → 「Brinson 归因」按钮 → 行业配置/选择/交互表 |
| 4 | 事件引擎费用/滑点/流动性 | 代码页「执行成本」区：滑点 bps / 流动性参与率 / 买/卖费率 |
| 5 | 行业分散参数 `industry_cap` | 回测页「组合构建」→ 每行业上限 |
| 6 | 多空信号方向 | 信号页「多空对冲」勾选 → 方向列（多/空） |
| 7 | 对比回撤图 | 看板多策略对比区新增「多策略回撤」图 |
| 8 | 归档详情完整化 | 历史页详情补回撤图 / 持仓 / 调仓 tab |
| 9 | 舆情情绪看板（Vue 简版） | 新「舆情」tab：情绪分布 / 每日条数 / 最强最弱新闻 |
| 10 | 舆情 IC/分组结果 | 舆情 tab 展示 `results/sentiment_ic_group.csv` |

### 仍为代码级（无需 UI）

- 组合优化权重方法 `ctx.optimize_risk_parity / mean_variance / max_diversification`
  是事件策略内 API，直接在代码实验室的 `on_bar` 中调用即可，无独立参数 UI。

## 11. 对标市面个人量化平台

| 能力域 | 本平台 | 聚宽/米筐/掘金 | Qlib/backtrader | 差距与方向 |
| --- | --- | --- | --- | --- |
| 数据 | 腾讯+Tushare 增量缓存，日线+滚动因子 | 商业全量数据（分钟/财务/事件） | 自备数据 | 缺分钟线、财务/公告宽表仅 PG 预留 |
| 研究环境 | 网页代码实验室 + Streamlit | Notebook + 因子库 | Notebook/脚本 | 缺 Notebook、因子表达式库 |
| 回测 | 因子轮动 + 事件驱动，T+1/涨跌停/费用/滑点/多空 + 财务因子 | 成熟撮合 + 多周期 | 成熟 | 撮合近似（ST 涨跌幅未区分、无分钟级撮合） |
| 组合构建 | 等权/风险平价/均值方差/最大分散化 + Barra 风格风险模型 | 优化器 + 风险模型 | 部分 | 风格因子为轻量代理定义，缺完整 Barra 行业/风格库 |
| 稳健性 | walk-forward + 参数网格 + 滚动训练-测试 | 参数优化/样本外验证 | 部分 | 训练期无特征工程/MI 选参，仅简单网格 |
| 绩效归因 | UI：IC/分组/Brinson/风险归因/QuantStats | 归因报表 | 部分 | 无选股-行业-风格三维联动报表 |
| 账户/风控 | 手动记账 + 日级模拟盘（T+1 撮合/费用/风控） | 模拟盘/实盘/风控 | 无 | 缺实时行情撮合与实盘 |
| 自动更新 | systemd 定时增量 | — | — | 单机部署，无任务队列 |
| 部署 | systemd + 本地进程 | SaaS | 本地库 | 缺容器化一键部署 |

结论：当前平台适合**个人单机研究**（数据→因子→回测→报告闭环已跑通，
绩效/归因已进 UI）。P0 的 Barra 风格风险模型、财务因子与 P1 的归因 UI、
滚动训练-测试框架已完成，日级模拟盘已上线（T+1 撮合/费用/风控/自动日结）。
对标商业平台的主要短板剩：① 财务数据当前为样例覆盖，需 `--fina` 全市场补数据；
② 模拟盘为日级，无实时行情撮合与实时风控；③ 无 Notebook/
因子表达式库与 Docker 打包。

## 12. 已知限制（Demo）

- 未做多进程/后台任务，大批量参数扫描同步执行，event 模式约 1-2 分钟
- 行业分类接口在部分服务器不可达时回退本地缓存，科技股票池为缓存快照
- 腾讯前复权价随时间动态调整，跨天重抓同一天价格可能有微小差异
- ST 涨跌停因缺标记暂按 10% 近似处理
- 舆情数据依赖 `~/quant/sentiment-mvp` 独立流水线每日更新

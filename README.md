# quant_ui — A股量化回测 Web 工作台（Demo）

基于 Streamlit 的回测/看板 Demo。数据来自本地 CSI800 面板
(`/tmp/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet`)，
策略引擎抽自 `quant_3stocks` 的月度调仓因子轮动逻辑。

## 功能
- 资金看板：模拟资金曲线、回撤、指标卡片、当前持仓
- 回测工作台：股票池/策略/TopN/资金/频率/区间参数化回测
- 今日信号：按因子打分看当前该买什么
- 数据状态：网上抓取（腾讯行情 + 中证指数官网）+ 本地增量缓存
- FastAPI 后端：回测/信号/记账/数据更新全部暴露为 REST API

## 后端 API（FastAPI，端口 8000）

启动：`systemctl start quant-api`，文档：`http://<IP>:8000/docs`

```
GET  /api/health
GET  /api/data/status          数据缓存状态
GET  /api/data/panel-info      面板概况
POST /api/data/update          后台触发数据更新 {mode, end}
GET  /api/data/update/status   更新进度
GET  /api/strategies           策略列表
POST /api/backtest             跑回测（参数：universe/strategy/top_n/capital/freq/start/end...）
GET  /api/signals              今日信号
GET  /api/ledger/transactions  交易流水
POST /api/ledger/transactions  录入交易
GET  /api/ledger/deposits      出入金
POST /api/ledger/deposits      录入出入金
GET  /api/ledger/equity        每日资金曲线
GET  /api/ledger/positions     当前持仓与盈亏
```

## 每日自动更新

`systemd` timer 每天 17:35 自动增量刷新行情：

```bash
systemctl status quant-data-refresh.timer
systemctl start quant-data-refresh.service   # 手动跑一次
journalctl -u quant-data-refresh.service -f
```

## 数据抓取与缓存

`core/fetcher.py` 负责从网络抓取并增量缓存到 `data/`（已 gitignore）：

- 股票池：中证指数官网抓沪深300 + 中证500 成分股
- 日线：腾讯行情接口（前复权），多线程并发，增量只抓新增区间
- 指数：腾讯接口抓沪深300 收盘价作基准
- 行业分类：东方财富接口（部分服务器不可达时自动沿用本地行业缓存）
- 缓存：`data/panel.parquet`、`universe.csv`、`tech.csv`、`index.csv`、`meta.json`

首次使用建议在「数据状态」页点一次「🚀 开始更新」：
会先种子化本地旧面板数据，之后每天增量约 1-2 分钟。

## 运行

```bash
cd /home/ubuntu/quant_ui
/home/ubuntu/stock-analyzer/local_venv/bin/python -m streamlit run app.py \
  --server.port 8501 --server.address 0.0.0.0 --server.headless true
/home/ubuntu/stock-analyzer/local_venv/bin/python -m uvicorn backend.main:app \
  --host 0.0.0.0 --port 8000
```

手机浏览器打开：
- 正式 Web 前端（Vue3 单页）：`http://<服务器IP>:8000`
- Streamlit 旧 Demo：`http://<服务器IP>:8501`

## 目录

```
app.py                  # Streamlit 入口
core/engine.py          # 回测引擎（事件驱动，T+1/手数/费用/可承载过滤）
core/metrics.py         # 收益/夏普/回撤等指标
core/data.py            # 本地数据加载
core/fetcher.py         # 网络抓取 + 增量缓存
core/ledger.py          # 账户记账（交易/出入金/每日估值）
core/store.py           # 缓存存储
strategies/registry.py  # 策略注册表（新策略在此添加）
backend/                # FastAPI REST API
scripts/refresh_data.py # 每日数据刷新脚本
static/                 # Vue3 CDN 单页前端（index.html + 本地 vendor）
```

## 写自己的策略

编辑 `strategies/registry.py`，添加一条即可出现在 UI 下拉框：

```python
STRATEGIES["我的双均线策略"] = {"factor": "mom20", "ascending": False}
```

更复杂的策略在 `core/engine.py` 的 `build_factor_frames()` 里加因子，
或新增一个返回「每期得分矩阵」的函数。

## 已知限制（Demo）
- 未做多进程/后台任务，大批量参数扫描留到下阶段
- 前端目前是 Streamlit；正式 Web 前端待选型（Vue3/轻量方案）

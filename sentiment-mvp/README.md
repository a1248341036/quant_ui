# A股舆情情绪 MVP

可落地的最小方案：东方财富个股新闻 -> 情绪打分（词典/LLM）-> 事件研究回测。

## 快速开始

```bash
# 用已有虚拟环境（含 akshare/pandas 3.0）
export PYTHON=/home/ubuntu/qqbot/.venv/bin/python

cd ~/quant/sentiment-mvp
$PYTHON run_pipeline.py all
```

分步执行：

```bash
$PYTHON run_pipeline.py universe      # 生成标的池 data/universe.csv
$PYTHON run_pipeline.py fetch         # 抓取每只股票最近新闻 -> data/news_raw.jsonl
$PYTHON run_pipeline.py fetch-cls     # 抓财联社当日电报并匹配标的 -> data/news_cls.jsonl
$PYTHON run_pipeline.py fetch-extra   # 扩展源：东财全球资讯/金十/东财股吧/热榜 -> data/news_extra.jsonl
$PYTHON run_pipeline.py score         # 词典打分 -> data/news_sentiment.csv
$PYTHON run_pipeline.py score --llm   # 可选：本地 new-api LLM 打分
$PYTHON run_pipeline.py train-ml      # 用开源股评语料训练 ML 情绪模型（约1分钟）
$PYTHON run_pipeline.py study         # 事件研究 -> outputs/event_study.csv + report.md
$PYTHON run_pipeline.py audit         # LLM vs 词典标签一致性审计 -> outputs/llm_lexicon_audit.md
$PYTHON run_pipeline.py snapshot      # 今日舆情快照（财联社）-> outputs/sentiment_snapshot.md
$PYTHON run_pipeline.py daily         # 每日入库：增量合并 SQLite + 事件研究 + 快照
$PYTHON run_pipeline.py study --label-source llm   # 用 LLM 标签跑事件研究 -> outputs/*_llm.md
```

常用参数：

- `--codes 600519,000725`：只跑指定股票
- `--llm`：用本地 new-api 的 `deepseek-v4-flash` 打情绪
- `--refresh-prices`：忽略价格缓存重新拉取

## 数据源

- 东财个股新闻（EM search-api-web）
- 财联社电报（官方 v1 API + 本地签名，零 key）
- 东财全球资讯 7×24（np-weblist）
- 金十快讯（jin10.com/flash_newest.js）
- 东财股吧热帖（guba 页面内嵌 article_list）
- 巨潮公告 / 互动易（config `sources.cninfo/irm` 默认关闭，可按需打开）
- 同花顺热榜 / 东财人气榜（source=hot）

数据层底座复用项目内 `tools/a-stock-data`（SKILL.md v3.4.0 抽取的
`astock_data.py`），东财系接口自带串行限流防封。

## ML 情绪模型

`train-ml` 用开源语料 algosenses（东财股吧正/负股评各 4607 条）训练
char n-gram TF-IDF + 逻辑回归二分类，落盘到项目内 `models/`。
模型在留出集准确率约 90%。评分时把 `config.yaml` 的 `sentiment.method` 改为
`ml` 即可切换（没有模型时自动回退词典）。

## 设计口径（防未来函数）

- 新闻 15:00 前发布 -> 当日收盘可知；15:00 后或非交易日 -> 顺延到下一交易日。
- 收益口径：事件日收盘买入 -> 持有 h 个交易日收盘卖出（未假设盘中即可交易）。
- 双口径：报告同时给出「收盘买入」和「次一交易日开盘买入」两种收益。
- 超额基准：沪深300 同窗口收盘到收盘收益。
- 同股票同事件日多条新闻先聚合成一个事件，避免重复计数。

## 输出

- `data/universe.csv`：按近半年日均成交额选出的 top 30 股票
- `data/news_raw.jsonl`：原始新闻（标题/正文/发布时间/来源/链接）
- `data/news_cls.jsonl`：财联社当日电报（仅当日/近两日窗口，适合做日度快照，不适合历史回测）
- `data/news_extra.jsonl`：扩展源（东财全球/金十/股吧/热榜，按标的匹配）
- `data/news_sentiment.csv`：逐条情绪分
- `outputs/event_study.csv`：逐事件前瞻收益
- `outputs/event_study_report.md`：分桶超额收益报告（正/中性/负 x 持有1/2/3/5日）
- `outputs/llm_lexicon_audit.md`：词典 vs LLM 标签一致率与分歧样本
- `outputs/sentiment_snapshot.md`：今日舆情快照

## 每日入库（SQLite）

`daily` 命令做增量合并：

- 数据存 `data/articles.db`（SQLite，按 代码+时间+标题 去重）
- 每天只给新增条目打分，老数据不重复处理
- 全量导出 `data/news_sentiment_daily.csv`，并自动跑 `study --tag daily`
- `daily` 也会增量抓扩展源（东财全球/金十/股吧/热榜）
- 财联社电报按日期归档 `data/cls/YYYY-MM-DD.jsonl`

## 看板

已在量化工作台 `http://<服务器IP>:8501` 增加「舆情情绪看板」页面：

- 数据概览（总数/区间/来源/正负面）
- 今日财联社快照
- 情绪分直方图、标签×来源分布
- 事件研究：分桶超额收益表 + 图（次日开盘口径）
- 每日舆情量趋势、最强/最弱新闻

看板读取 `data/articles.db` 与 `outputs/event_study_daily.csv`，缓存 5 分钟。
页面在 `/home/ubuntu/quant/quant_ui/pages/2_舆情情绪看板.py`，随 `quant-ui.service` 自动加载。

## 定时任务（已安装）

```cron
30 15 * * 1-5  cd /home/ubuntu/quant/sentiment-mvp && ... run_pipeline.py snapshot
30 21 * * *    cd /home/ubuntu/quant/sentiment-mvp && ... run_pipeline.py daily
```

15:30 抓当日快照；21:30 跑完整 daily 入库+事件研究。日志在 `logs/snapshot.log` / `logs/daily.log`。
积累几天后样本自然增长，回测窗口变长。

## 已知局限

- 东方财富搜索接口只能回溯约最近 1 个月新闻，样本短。
- 词典打分不识别反讽与隐含情绪；LLM 打分未做知识截止匹配，仅适合探索。
- 事件窗口重叠未做自相关修正。
- 价格取 sina 前复权日线，分红送转已调整。

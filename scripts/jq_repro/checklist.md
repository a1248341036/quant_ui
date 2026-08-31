# 聚宽策略复现 Checklist

> 目标: 复现一个新聚宽策略时, 只写"差异层"(候选逻辑+参数), 数据/引擎/坑位全部复用。
> 目录: `scripts/jq_repro/` —— `jq_data.py`(数据层) + `template.py`(骨架) + 本清单。

## 标准流程 (预期 15~30 分钟)

1. **复制 runner**: 拷 `run_smallcap.py` 改名, 改两处:
   - `build_cand_fn`: 你的选股逻辑 → 返回 `[(code, passed, is_hl), ...]`,
     按优先级排序; `passed`=通过全部非豁免过滤; `is_hl`=信号日收盘涨停。
     (骨架自动处理"持仓豁免 passed/is_hl")
   - `cfg = ReproConfig(...)`: 参数。仓位映射默认固定 `stock_num`;
     MA 择时用 `jq_data.ew_index()` + `template.ma_num_fn()`。
2. **跑**: `python scripts/jq_repro/run_xxx.py` (数据构建 ~30s + 回测 ~60s)
3. **对拍**: 若聚宽有历史净值曲线, 抽 3~5 个关键点(如 2018 年末、2024-02 微盘股崩盘底、
   最近一日)对比量级; 差异 >20% 优先查近似点(见下), 不要先怀疑代码。

## 已封装的通用件 (jq_data.py)

| 函数 | 用途 |
|---|---|
| `load_panel(start, end, prefixes)` | 前复权引擎面板 + 全市场 meta + 未复权收盘长表 |
| `build_tables(panel, meta, close_raw_df)` | 逐日矩阵: mv/up_limit/is_st/paused/listed_ok/hl(精确涨停) |
| `load_income()` + `fin_ok_matrix(pred)` | 点时财务: 信号日取 ann_date<=该日 最新一期报表 |
| `triple_positive_pred()` | 国九条三正(归母净利>0 & 净利>0 & 营收>1亿) |
| `ew_index(tables)` | 域等权指数(日收益裁剪±20%去伪影) + MA, 用于择时/干净基准 |
| `template.ma_num_fn(level, ma)` | 原版 sigmoid 仓位映射(相对阈值 2.5% 等效 500点/2万点) |
| `template.score_cand_fn(score_mat, tables, filters)` | **分数矩阵→cand_fn 桥接**(AlphaAgent 因子接入事件引擎的入口), 语义=先过滤后排序 |

骨架已实现的交易语义 (`template.py`): 空仓月清仓、个股止损/止盈、大盘惨跌清仓
(域平均 1-收/开 ≥5%)、涨停开板近似卖出、周度调仓(持仓豁免过滤 + 昨日涨停豁免卖出 +
暴露/单票上限)、涨停开盘买不进/跌停卖不出、整手、聚宽费率。

## 近似决策 (何时偏离原版、为什么)

| 原版机制 | 本 pipeline 口径 | 何时需要改 |
|---|---|---|
| `get_index_stocks` 历史成分 | **本地 index_weight 为空** → 用代码前缀域 + 策略自身过滤等效 | 找到成分股数据源后替换 |
| 盘中 9:05/10:00/14:00/14:50 | T 收盘信号 → T+1 开盘成交(引擎统一) | 引擎不支持分钟, 无法改 |
| 14:00 涨停开板卖出 | 昨收涨停 + 今开未封死 → 开盘卖 | 无(已是最接近的日线近似) |
| 4月空仓买 511260 国债ETF | 持币(本地无该 ETF 历史) | 若补 ETF 数据, 把 ETF 拼进 panel |
| 涨停卖出当日补仓 (check_remain_amount) | 省略, 下一调仓日自然补足 | 影响小, 一般不改 |
| get_fundamentals 点时 | income 按 ann_date 点时, 同口径 | balancesheet/cashflow 同法可加 |
| 聚宽 FixedSlippage | slippage_bps 参数(默认 0) | 小市值高换手建议敏感性测试 1~3bps |

## 数据坑 (每次必查)

1. **`report_type` 是 float 不是 str**: `income.report_type == "1"` 会过滤掉全部行 →
   用 `pd.to_numeric(...) == 1`。症状: 候选数为 0。
2. **`stock_basic` 是当前快照**: 已退市股缺 list_date(幸存者偏差) →
   上市天数用 day 表逐日 `listed_days`, stock_basic 只做退市名兜底。
3. **前复权微价股污染等权基准**: qfq 价格 0.0x 元的股票日收益可达 ±100% →
   指数/基准必须裁剪日收益(±20%), 否则超额指标无意义。
4. **停牌/0 价格行**: 面板必须剔除 open<=0 或 close<=0, 否则引擎 o2o 出 inf。
5. **tushare 单位**: amount=千元(×1e3 成元), total_mv=万元(3亿=3e4), 涨停价=元。
6. **历史每日报告数**: 财务过滤后候选数应打印检查(小市值域正常 500~2000 只);
   若骤降为 0 或个位数, 先查单位/格式再查逻辑。

## 分数矩阵桥接 (AlphaAgent 因子接入事件引擎)

```python
from template import score_cand_fn
cand_fn = score_cand_fn(score_mat, tables,           # score_mat: date×code, 越大越优先
                        top_keep=50, filters=lambda i: eligible[i])
```

- `score_mat` 无需对齐 tables 全网格: 缺失日/缺失股自动跳过(天然过滤停牌/未上市/数据缺口)
- `filters` 与排序解耦: 三正/ST/次新等风险筛选由 jq_data 矩阵组合, 因子只管排序
- **语义 = 先过滤后排序**(与聚宽 `get_fundamentals` 的 WHERE+ORDER BY 一致):
  先掩掉不合格股票, 再在合格池内按分数降序取 top_keep。
  若"先排序后过滤", top 边缘大量股票触发价格/涨停过滤时候选池深度骤减, 行为完全不同
- **实现陷阱**: 排序必须对齐 tables.codes 全网格(用 NaN 掩码沉底, 不能 dropna),
  否则 dropna 后子序列的位置索引会错位映射股票代码
  (实测: 错位版总收益 1.13 vs 正确 8.69, 差 7.5 倍且无报错, 极难察觉)
- 持仓豁免/涨停豁免/仓位约束等交易层语义由骨架统一处理, 与因子无关

已验证: `-log(mv)` 分数矩阵走桥接 ≡ 小市值策略 cand_fn 直连版, 9 项指标偏差 0.00e+00
(`run_from_scores.py` 尾部自带等价性自检, 参照 `out_smallcap/metrics.json`)。

## 何时直接丢给 agent (不套模板)

- 策略依赖**指数历史成分**(`get_index_stocks`)且无法用前缀+过滤等效
- 用到 **finance 特殊表**(审计意见 STK_AUDIT_OPINION、分红送转 STK_XR_XD 等)
- **分钟级逻辑占比高**(日内开板、尾盘/竞价交易、盘中止损触发价)
- ~~选股是打分/模型~~ → 已由 score_cand_fn 桥接覆盖, 直接传分数矩阵即可

## 结果解读提醒

- 与聚宽净值**不会逐点一致**(域替代+日线执行), 量级与形态一致即为复现成功
- 小市值策略对**费率/滑点/容量**极敏感: 100 万资金是舒适区, 2000 万+ 会显著劣化
  (建议加 max_participation 测试)
- 2016~2026 小市值因子有一段极强行情(2022-2025 微盘股), 对比聚宽原帖回测区间时注意段位差

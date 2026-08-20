# 日线量化系统现状评估

更新时间：2026-08-20

## 1. 结论

当前系统已经从早期原型进入了“可用于研究和回测”的阶段，适合：

- 股票日线因子研究；
- ETF 日线轮动；
- 横截面排序选股；
- 多因子组合实验；
- 基础 Walk-forward 验证；
- 今日信号生成；
- 股票/ETF 低频模拟盘。

但目前还不能定位为资金级生产交易系统，主要原因是：

- 历史股票池和行业分类仍可能存在幸存者偏差；
- 回测数据版本记录还不够完整；
- 股票/ETF/基金费用模型仍有近似部分；
- 两套交易引擎仍存在口径漂移风险；
- 样本外验证和实盘订单层还不够完善。

综合评价：

| 方向 | 评价 |
|---|---:|
| 股票日线因子回测 | 7/10 |
| ETF 日线回测 | 7/10 |
| 场外基金回测 | 5/10 |
| 因子研究能力 | 6.5/10 |
| 交易执行仿真 | 6/10 |
| 数据质量与可复现性 | 5.5/10 |
| 模拟盘与回测一致性 | 6/10 |
| 生产稳定性 | 5.5/10 |
| 综合 | 6.3~6.8/10 |

## 2. 当前架构

当前有三个资产场景：

```text
股票 stock
场内 ETF etf
场外基金 fund_nav
```

主要文件：

```text
D:\Quant\quant_ui\core\assets.py
D:\Quant\quant_ui\core\engine.py
D:\Quant\quant_ui\core\execution.py
D:\Quant\quant_ui\core\fund_engine.py
D:\Quant\quant_ui\backend\routers\backtest.py
```

### 股票

- 数据：OHLCV、成交额、换手率；
- 执行器：`StockExecutionAdapter`；
- 因子：动量、波动率、换手率、成交额、双均线、财务因子；
- 支持股票式涨跌停、停牌、整手、费用和滑点模型。

### ETF

- 数据：ETF OHLCV、成交额、换手率；
- 执行器：`ETFExecutionAdapter`；
- 因子：与股票共用量价和趋势因子；
- 默认关闭股票式涨跌停过滤；
- 默认买卖费率为 0.03%；
- 已接通单策略回测、策略对比、今日信号和 QuantStats；
- 当前跳过 Brinson 行业归因。

### 场外基金

- 数据：单位净值 NAV；
- 执行器：`FundNavExecutionAdapter`；
- 因子：NAV 动量、波动率、最大回撤、Sharpe、Sortino、动量加速度、净值稳定性；
- 支持 0.01 份精度、A/C 份额识别、FIFO 和默认阶梯赎回费；
- T+1 仍然是下一可用净值日确认的近似模型。

## 3. 因子场景划分

严格来说，系统是“三个资产场景、两套因子计算族”。

```text
股票 + ETF：共用一套量价因子逻辑
场外基金：独立净值因子逻辑
```

这是合理的，不需要为了形式上的“三个场景”复制三套因子代码。

### 股票/ETF 共用因子

```text
mom20
mom60
vol20
am20
turn20
ma_cross5_10
ma_cross5_20
ma_cross10_30
ma_cross20_30
ma_cross20_60
composite
```

股票还可以额外使用：

```text
pb
ep
roe
gross_margin
rev_yoy
np_yoy
```

ETF 不应使用股票财务因子，后端已经增加资产类型校验。

### 场外基金净值因子

```text
mom20
mom60
vol20
ma_cross*
mdd20
mdd60
sharpe20
sharpe60
sortino20
mom_accel
nav_stability
composite
```

基金不使用：

```text
am20
turn20
pb
ep
roe
```

## 4. 已经做得比较好的部分

### 4.1 日线时序基本正确

主回测语义为：

```text
T 日收盘计算因子
T 日生成目标组合
T+1 日开盘成交
T+1 日收盘估值
```

避免了使用 T 日收盘数据、同时又用 T 日收盘价成交的明显前视错误。

### 4.2 ETF 已有独立资产适配

ETF 不是简单地改股票名称，当前有：

```text
ETF_PROFILE
ETFExecutionAdapter
ETF 专用数据加载
ETF 默认费用
ETF 策略类型过滤
ETF Brinson 跳过
```

本地数据当前规模约为：

```text
ETF：1582 只
ETF 日线：约 124 万行
```

### 4.3 财务因子考虑公告日

财务因子位于：

```text
D:\Quant\quant_ui\core\financial.py
```

主要按：

```text
ann_date <= 交易日
```

使用当时已经披露的财务数据，避免直接将最新财报回填到历史。

### 4.4 已经形成基础研究闭环

当前已有：

```text
因子计算
因子排序
组合构建
交易执行
费用
滑点
换手
IC
Rank IC
分组收益
Walk-forward
QuantStats
历史归档
今日信号
模拟盘
```

### 4.5 模拟盘部分复用主回测引擎

因子模拟盘会调用：

```text
core.engine.run_backtest()
```

而不是完全复制一套买卖逻辑，降低了回测与模拟盘不一致的风险。

## 5. 关键问题

### P0：历史股票池存在幸存者偏差风险

当前 `build_codes()` 主要根据当前的：

```text
data/universe.csv
data/tech.csv
data/etf.csv
data/fund.csv
```

构建回测标的集合，而不是严格使用每个历史日期的股票池。

可能导致：

- 退市股票被排除；
- 历史指数调出标的被排除；
- 当前行业分类被用于历史；
- 当前基金/ETF 列表被用于整个历史区间；
- 历史回测收益偏乐观。

建议增加：

```python
build_codes(universe="沪深300", as_of=date)
```

并维护：

- 历史指数成分；
- 历史行业分类；
- 上市日期；
- 退市日期；
- ETF 上市日期；
- 基金成立日期。

### P0：回测数据版本记录不够完整

目前归档的 `data_version` 主要是数据最大日期，无法保证完整复现。

建议记录：

```text
行情文件 SHA256
股票池文件 SHA256
ETF 文件 SHA256
基金费率文件 SHA256
因子定义版本
策略注册表版本
执行器版本
Git commit
```

建议最终形成：

```json
{
  "data_snapshot": {
    "panel.parquet": "sha256...",
    "etf_panel.parquet": "sha256...",
    "fund_nav.parquet": "sha256...",
    "fund_fee.csv": "sha256..."
  },
  "engine_version": "...",
  "strategy_version": "...",
  "git_commit": "..."
}
```

### P1：基金费率数据已有，但尚未完整接入执行器

当前本地已经存在：

```text
D:\Quant\quant_ui\data\fund_fee.csv
D:\Quant\quant_ui\scripts\refresh_fund_fees.py
```

当前数据统计大致为：

```text
基金记录：11793
有申购费率：11434
fee_status=ok：2064
有赎回规则：2064
```

因此申购费率覆盖率较高，但完整费率记录和赎回规则覆盖率只有约 17.5%。

当前 `FundNavExecutionAdapter` 仍主要使用统一默认申购费和默认阶梯赎回费。

下一步应接入：

```text
FundFeePolicy
按 code + A/C 份额匹配
解析 redemption_fee_rule
缺失时回退默认费率
记录 fee_policy_source
```

注意：管理费、托管费和销售服务费通常已经反映在 NAV 中，不能简单再次从现金中扣除，避免双重扣费。

### P1：股票和 ETF 复权口径需要固定

当前股票和 ETF 数据较多使用前复权价格。需要明确：

- 复权后的 open 是否直接作为成交价；
- 分红、拆分、配股如何处理；
- 涨跌停判断是否基于复权价格；
- 成交额、成交量与复权价格是否一致。

建议在系统中明确区分：

```text
研究收益口径
交易成交口径
估值口径
分红处理口径
```

### P1：ETF 执行仍是日线近似模型

当前 ETF 已支持：

```text
现金
整手
费用
固定滑点
成交额过滤
持仓更新
```

但尚未支持：

```text
Bid-Ask spread
盘口深度
冲击成本
最小佣金
ETF 折溢价
IOPV
申赎机制
ETF 特有涨跌幅规则
```

对于日线 ETF 轮动可以暂时接受，但建议输出三组成本情景：

```text
理想成本：0 bps
基准成本：3 bps
压力成本：10 bps 或更高
```

### P1：两个交易引擎存在长期漂移风险

当前有：

```text
core/engine.py        因子轮动引擎
core/event_engine.py  事件驱动引擎
```

两者目前仍分别处理部分：

```text
现金管理
成交
费用
滑点
持仓
估值
拒单
```

长期建议收敛为：

```text
Signal Engine
    ↓
Portfolio Construction
    ↓
Order Intent
    ↓
Execution Adapter
    ↓
Fill / Reject
    ↓
Portfolio Ledger
    ↓
Valuation
```

目前 `core/execution.py` 已经开始承担统一执行层职责，但还没有完全收敛。

### P1：核心文件偏大

当前主要文件规模约为：

```text
core/engine.py：880 行
backend/routers/backtest.py：944 行
core/paper.py：1273 行
core/event_engine.py：635 行
```

目前不建议马上大规模重构，但后续应拆出：

```text
backtest_service.py
asset_service.py
portfolio_engine.py
valuation.py
cost_model.py
signal_engine.py
```

### P1：测试覆盖不足

当前测试结果：

```text
23 passed
```

这能证明基础核心函数没有明显崩溃，但还不能证明全链路稳定。

需要继续补充：

- 股票 API 回测；
- ETF API 回测；
- 基金 API 回测；
- Compare API；
- Signals API；
- QuantStats API；
- 模拟盘 dry-run；
- 数据刷新后的 schema 校验；
- 空股票池；
- 单标的池；
- 因子全部缺失；
- 开盘价 NaN；
- 成交额为 0；
- 停牌和涨跌停；
- 上市不足因子窗口；
- 历史股票池变化。

## 6. 日线研究能力评估

### 已经可以做

- 横截面因子选股；
- 月频、周频和低频调仓；
- 多因子组合；
- 因子 IC、Rank IC 和分组收益；
- 换手和费用分析；
- 参数扫描；
- Walk-forward 初步验证；
- ETF 轮动；
- 今日信号；
- 模拟盘验证。

### 还需要补强

#### 6.1 严格训练集/验证集/测试集

建议增加：

```text
Train
Validation
Test
purge gap
embargo
滚动窗口
扩展窗口
参数冻结
样本外报告
```

尤其是未来收益 horizon 为 20 日时，需要处理重叠标签污染。

#### 6.2 组合级净收益研究

除了因子 IC，还应统一输出：

```text
Gross Return
Net Return
Turnover
Trading Cost
Slippage Cost
Capacity
```

避免出现因子 IC 看起来不错，但组合扣成本后收益消失的情况。

#### 6.3 动态持仓数量口径统一

以下入口应统一使用同一个 `SelectionConfig`：

```text
回测
今日信号
因子质量
Walk-forward
模拟盘
```

确保 `top_n`、`top_pct`、`min_positions`、`max_positions` 的语义一致。

#### 6.4 明确因子覆盖和过滤过程

回测结果建议展示：

```text
原始股票池数量
因子有效数量
流动性过滤数量
可交易数量
最终选中数量
```

示例：

```text
股票池：1800
因子有效：1320
流动性过滤后：800
最终选股：10
```

## 7. 模拟盘定位

当前模拟盘适合：

- 日线低频策略；
- T 日收盘生成信号；
- T+1 开盘模拟成交；
- 股票和 ETF 因子策略；
- 盘后自动任务；
- 状态记录和回放。

当前还不是完整实盘系统，因为缺少：

```text
券商 API
真实订单状态
部分成交
撤单
订单超时
网络重试
资金对账
券商持仓对账
真实成交回报
```

更准确的定位是：

```text
回测与实盘之间的纸面验证层
```

## 8. 推荐实施顺序

### P0：可信度优先

1. 历史股票池和历史行业分类；
2. 回测数据快照和文件 hash；
3. 复权与分红口径固定；
4. 增加上市/退市/成立日期过滤；
5. 增加端到端 API 回归测试。

### P1：统一日线引擎

抽取并统一：

```text
CostModel
SelectionConfig
ExecutionAdapter
PortfolioLedger
Valuation
```

让股票、ETF、基金、回测和模拟盘复用一致接口。

### P1：真正接入基金费率

完成：

```text
fund_fee.csv 加载
FundFeePolicy
A/C 份额匹配
赎回规则解析
默认费率回退
费率来源记录
```

### P1：完善日线样本外验证

完成：

```text
Train / Validation / Test
purge / embargo
参数冻结
净收益和成本压力测试
容量测试
```

### P2：ETF 高级执行模型

后续再增加：

```text
spread
impact
最小佣金
IOPV
折溢价
ETF 成分暴露
行业归因
```

## 9. 最终定位

当前系统可以定位为：

> 一个具备实际研究价值的日线量化研究工作台，已经支持股票和 ETF 低频回测，但还不是完成度很高的生产级交易系统。

当前最强的主线是：

```text
数据加载
    ↓
因子计算
    ↓
截面排序
    ↓
组合构建
    ↓
成本模型
    ↓
T+1 成交
    ↓
净值/换手/费用
    ↓
Walk-forward
    ↓
模拟盘对齐
```

后续应优先稳定这条日线主线，而不是继续无限增加策略数量。


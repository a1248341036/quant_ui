# 交易引擎改造计划

## 目标

在不立即迁移 Backtrader/Lean、也不破坏现有股票回测结果的前提下，逐步把当前引擎改造成：

```text
数据适配 → 因子/信号 → 组合构建 → 资产执行 → 估值/归因
```

股票、场内 ETF、场外基金共享研究和组合接口，但使用不同的资产执行规则。

## 当前已执行的阶段

- 新增 `core/assets.py`，集中定义股票、ETF、场外基金的执行配置；
- 新增 `AssetAdapter` 边界：行情类资产走 OHLCV 适配器，场外基金走 NAV 适配器；
- `core/fund_engine.py` 改为调用 `FundNavAdapter`，保留原有 NAV 近似口径；
- `core.engine.run_backtest` 增加 `execution_profile`，结果中返回 `asset_type` 和配置；
- ETF、股票、qweave 回测入口显式声明资产类型；
- 原有 `run_backtest(...)` 调用不传新参数时，仍按股票历史默认行为执行。
- 新增 `core/selection.py` 的 `PortfolioBuilder`，统一处理排序、行业上限、固定 TopN、动态比例和等权目标权重；
- `core.engine` 的现金模式和权重模式均已通过 `PortfolioBuilder` 构建候选组合，原有参数保持兼容。
- 新增 `core/execution.py` 的 `StockExecutionAdapter`，封装现金、先卖后买、整手、费用、滑点、停牌、涨跌停、流动性和拒单明细；
- `core.engine` 的现金模式已接入 `StockExecutionAdapter`，保留原有 Chandelier 止损和调仓时点语义。
- `core/execution.py` 新增 `ETFExecutionAdapter`；ETF 回测按资产类型选择专用适配器，默认买卖费率为 0.03%，默认关闭股票式涨跌停过滤。
- ETF 面板加载增加字段/价格校验；ETF 因子模拟盘、事件模拟盘和事件归因入口也已按 ETF profile 处理，不再误用股票涨跌停/费率默认值。

真实数据验证：`data/etf_panel.parquet`（2025-01-02 ~ 2026-08-14，1576 只 ETF，499785 行）已完成端到端回测；19 次调仓、99 笔成交、52 笔买入、47 笔卖出，成交费率均为 0.03%，净值从 1.0 运行到 1.13623。

## 后续阶段

### P0：抽取组合层（已完成第一版）

已从 `core/engine.py` 抽出 `PortfolioBuilder`，负责：

- 固定 TopN、动态比例、最少/最多持仓数；
- 行业约束和权重上限；
- 后续多因子合成后的目标权重。

当前实现文件：`D:\Quant\quant_ui\core\selection.py`。

### P0：抽取股票执行层（第一版已完成）

已从现金模式中抽出 `StockExecutionAdapter`，集中处理：

- T+1、100 股整手；
- 停牌、涨跌停；
- 买卖费用、滑点、成交额参与率；
- 拒单和成交明细。

当前实现文件：`D:\Quant\quant_ui\core\execution.py`。
事件引擎暂未强行迁移到该适配器，避免两套撮合语义在没有逐项对齐前发生变化。

### P1：完善 ETF 执行层（现金执行第一版已完成）

- 已完成 ETF 专用适配器、默认费率、100 份交易单位和独立适配器选择；
- 已支持可选涨跌停配置，而不是强制沿用股票默认值；
- 已增加 ETF OHLCV 面板校验报告，校验字段、价格、日期范围和代码数量；
- 待增加 ETF 申赎、IOPV、折溢价和更细的流动性/冲击成本模型；

### P1：独立场外基金 NAV 执行层

不再长期把基金伪装成股票行情，逐步支持：

- 申购/赎回申请日与截止时间；
- NAV 确认日；
- 申购费、赎回费；
- 现金到账延迟；
- QDII 非 A 股净值日历。

### P2：统一事件策略接口

评估将 `core.event_engine.py` 的重复撮合逻辑接到同一组资产执行适配器上。只有在完成成交语义对齐后，才考虑删除重复实现。

## 暂不做的事情

- 暂不整体迁移到 Backtrader、Lean 或 vn.py；
- 暂不重写 `core.engine.py`；
- 暂不把场外基金宣称为真实盘中交易回测；
- 暂不在没有数据口径验证前加入复杂的 ETF 申赎/折溢价模拟。

## 验收标准

每一阶段都要满足：

1. 旧股票回测测试通过；
2. ETF 回测结果可运行且资产类型明确；
3. 场外基金 NAV 回测不引入未来数据；
4. 交易明细包含执行日期、信号日期、费用和拒单原因；
5. 新增资产规则不通过散落的 `if universe == ...` 实现。

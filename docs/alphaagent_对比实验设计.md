# AlphaAgent 原版 vs quant_ui 对比实验设计

> 目标：回答"原版 AlphaAgent（KDD'25 论文系统）与 quant_ui 增强版，哪个效果好"。
> 同一份数据、同一个 LLM、同样的轮数预算，各自全速跑自己的挖掘管道，盲测段统一裁决。

## 1. 论文评估协议（arXiv 2502.16789，KDD 2025）

原版论文的核心主张是**抗 alpha 衰减**（anti-decay），机制是三条正则：
1. **Originality（原创性）**：AST 相似度 vs 已有 alpha；
2. **Hypothesis-Factor Alignment（假设-因子对齐）**：LLM 评估语义一致性；
3. **Complexity Control（复杂度控制）**：AST 约束。

论文实验口径：
- **市场**：中国 CSI500（Baostock 数据，仅 OHLCV）；美国 S&P500
- **窗口**：train 2015-01~2019-12 / val 2020-01~2020-12 / test 2021-01~2025-01
- **组合**：LightGBM(depth=4) 融合 4 个基准 alpha → top-k dropout（选 50 剔最低 5）
- **费率**：买 0.0005 / 卖 0.0015
- **试验**：RD-Agent 与 AlphaAgent 各 20 次独立试验 × 5 轮进化，择优
- **指标**：IC、RankIC、ICIR、IR（超额信息比）、AR（年化超额）、MDD；
  效率指标：hit ratio（单轮内 AR>4% 的 alpha 占比 = top5%）、dev success rate、token 效率
- **报告值**（CSI500）：IC 0.0212 / ICIR 0.1938 / AR 11.00% / IR 1.488 / MDD -9.36%

## 2. 本次对比的适配与偏差

| 维度 | 论文 | 本实验 | 说明 |
|---|---|---|---|
| 市场 | CSI500 | 全 A（5,560 只，CNE 本地湖） | quant_ui 实际研究域；不重塑股票池 |
| 窗口 | 15~19/20/21~24 | train 2020~2022 / val 2023~2024 | 对齐 quant_ui 现行口径；**2025-01-01 起为双方共同盲测段** |
| 标签 | 次日收益 | `label_1d_close_to_close` | 两侧同标签 |
| 数据面 | 仅 OHLCV | OHLCV + daily_basic（价量 + 基础估值） | 导出面板 35 列，两侧同数据 |
| LLM | GPT-3.5-turbo | GLM-5.3-Flash（本地 codex 代理） | 两侧同模型同代理，内部对比不受影响 |
| 预算 | 20 试验 × 5 轮 | 各 1 run × 16 轮（max_turns=16 两侧默认一致） | 先跑通链路，可后续加 run |
| 起始库 | 累积 zoo | **双方均空库起跑** | 原版 `compare_stock_1d` / quant_ui `compare_prod` |
| 裁决 | Qlib 组合回测 | 统一盲测裁决 harness（见下） | 论文口径指标同款 |

## 3. 数据管道

- Tushare token 失效，**不联网**：quant_ui 本地 CNE 数据湖（`data/quant_dataset`，2009~2026 全 A 日线）已覆盖全部需求。
- 导出脚本 `scripts/export_panel_for_original_alphaagent.py`：CNE 缓存面板 → 原版格式
  （MultiIndex + OUTPUT_COLUMNS 35 列，缺列补 NaN），裁 2020-01-01~2024-12-31，
  输出 `D:/Quant/AlphaAgent/artifacts/panel/panel_1d.parquet`（0.61GB，5.73M 行）。
- 2025+ 不导出——盲测段物理隔离。

## 4. 统一裁决（`scripts/alphaagent_blind_judge.py`）

所有因子（双方产出）都在 **quant_ui 引擎**上复算（跨 DSL 兼容性已验证：原版 13 个
表达式 11 个直接跑通，2 个失败均为数据列缺失而非语法不兼容；原版算子是 quant_ui 子集）。

指标：
- IC（逐日截面 RankIC 均值）、ICIR（不年化，与 `cs_ic_summary` 同口径）、coverage
- 十分组多空（D10−D1）：年化 / 夏普 / MDD；分组多头超额年化
- 分年度 IC（alpha 衰减观察，论文核心主张）

**多重检验警示**：盲测段每重测一次烧掉一分，报告定稿前克制重跑。

## 5. 运行配置

| | 原版 arm | quant_ui arm |
|---|---|---|
| 入口 | `scripts/run_original_alphaagent.py --tag run1` | `scripts/run_alphaagent.py --panel <导出面板> ...` |
| 库 | `D:/Quant/AlphaAgent/artifacts/factorzoo/compare_stock_1d` | `artifacts/alphaagent/factorzoo/compare_prod` |
| 记忆 | 无（原版无记忆层） | `artifacts/alphaagent/comparison/memory_quantui.db`（隔离，不污染 research_memory.db） |
| 日志 | `artifacts/alphaagent/comparison/original_run1/logs/` | `artifacts/alphaagent/comparison/quantui_run1/logs/` |

原版 launcher 自动从 `~/.codex/config.toml` 注入 `OPENAI_API_KEY/OPENAI_API_BASE/MODEL`。

## 6. 效率指标（除收益外同样重要）

run 结束后用 `scripts/alphaagent_metrics.py` 口径对比：
墙钟分钟、LLM 生成分钟、token 消耗、每交付因子分钟数、工具错误率、评估-入库漏斗
（eval → 过海选 → candidate → stage_two → gate → promoted）。

## 7. 结果

（run 完成后回填：`artifacts/alphaagent/comparison/report.md`）

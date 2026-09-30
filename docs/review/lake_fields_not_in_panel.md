# 数据湖有、面板未暴露的字段清单（研报复现相关）

> 目的：判断"研报因子复现缺字段"里，哪些是**真的没有数据**、哪些只是**没并进面板**。
> 方法：只读 parquet footer（不加载数据）。面板暴露面 = `data_fields.py` 渲染文本 + `FF_PANEL_COLUMNS`（**168 字段**）。
> 数据湖 = `CNEquity/data/quant_dataset/_cnequity/curated`（**52 个数据集**）。
> 日期：2026-09-30｜分支 `feat/report-factor-records`

## 一、内存规则（先看这个）

面板约 **8,000,000 行**：

| 并列类型 | 每列内存 |
|---|---|
| float32 日频列 | **≈ 31 MB / 列** |
| float64 日频列 | ≈ 61 MB / 列 |
| `balancesheet` 全部 153 个未暴露列 | ≈ **+4.7 GB** ❌ 不可整体并入 |

→ 结论：**白名单按需并入，单次 ≤5 列（≈+155MB）**；高频需要的优先，稠密明细（如十大股东逐条）先做聚合列。

## 二、研报需要但面板未暴露（按优先级）

| 优先级 | 研报需求 | 湖内字段 | 数据集（文件/总行数/列数） | 并入成本 |
|---|---|---|---|---|
| **P0** | EBITDA（`企业倍数 EV/EBITDA`） | `ebitda` | `fina_indicator`（37/323,089/112）、`income`（37/316,732/87） | +31MB/列 |
| **P0** | 折旧 | `depr_fa_coga_dpba`（固定资产折旧）、`prov_depr_assets` | `cashflow`（16/183,600/100） | +31MB/列 |
| **P0** | 摊销 | `amort_intang_assets`、`lt_amort_deferred_exp`；`balancesheet.amor_exp` / `lt_amor_exp` | `cashflow`、`balancesheet`（26/250,739/161） | +31MB/列 |
| **P1** | 十大股东持股比例（研报"前十大股东持股变动比例"） | `holding_pct`、`holding_shares`、`holder_rank`、`holder_scope` | `top_holders`（18/**4,682,734**/13） | **建议只做聚合列**（前十大合计占比）= +31MB；全明细会多列且降维复杂 |
| **P1** | 机构持仓比例/市值 | `holding_ratio`、`holding_mv`、`holder_type` | `institutional_holdings`（42/627,268/9） | +31MB/列 |
| **P1** | 北向持股比例 | `holding_ratio`、`holding_mv` | `northbound_holdings`（1/3,933/9） | +31MB/列 |
| **P2** | 股本/流通股本/自由流通 | `total_shares`、`float_shares`、`free_float_shares`、`restricted_shares` | `share_structure`（18/156,520/11） | +31MB/列 |
| **P2** | 股东户数类（面板已有部分 `holder_*`） | `holder_count`、`holder_count_change_pct`、`avg_float_shares`、`avg_holding_value` | `shareholder_counts`（20/489,308/10） | 按缺口补 1~2 列 |

## 三、已确认**不是**缺口的（避免重复劳动）

| 研报需求 | 结论 |
|---|---|
| 股息率 / 股息 TTM | ✅ 面板已暴露 `$dv_ttm`；现金分红 `$div_cash_div`、`$div_days_to_ex`（湖 `dividend` 17 列：`cash_div`/`cash_div_tax`/`stk_div`/`ex_date`/`div_listdate`/`div_proc`） |
| 营业利润、总资产、净资产、营收等 | ✅ 面板已暴露 `$funda_operate_profit` / `$funda_total_assets` / `$funda_total_equity` / `$funda_total_revenue` 等（模型写别名已由 `FIELD_ALIAS` 自动改写） |
| `daily_basic`（PE/PB/换手等） | 湖里无该数据集目录，但面板已暴露 `$pe_ttm`/`$pb`/`$turnover_rate` 等 → 无需补 |

## 四、建议动作（不在本分支实施）

1. **面板侧**：在面板构建的列白名单里加入 P0 三列（`ebitda`、折旧、摊销），季频 → PIT 前向填充；成本 ≈ **+95MB**；
2. **P1 只做聚合列**：`top10_holding_pct`（前十大合计）、`inst_holding_ratio`、`northbound_holding_ratio`，各 +31MB；
3. **不要**整体并入 `balancesheet`（153 列 ≈ +4.7GB）；
4. 本分支（`feat/report-factor-records`）**只做抽取与映射**，遇到未暴露字段继续如实判 `null_reason=missing_field:<名>`（不硬映射，防口径漂移）。

## 五、附：盘点脚本

- `%TEMP%\scan_lake_panel.py` / `%TEMP%\scan_focus.py`（一次性只读扫描）
- 明细 JSON：`data/research_reports/knowledge/lake_vs_panel_fields.json`、`lake_focus_fields.json`

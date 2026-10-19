# -*- coding: utf-8 -*-
"""模块 05 · data_fields：可用行情变量 + 资金流/基本面/事件披露字段族。

按 panel 实际列启停：字段族数据未载入时对应块不注入（与插件数据覆盖联动）。
原文精确切片；本模块拥有尾部 `---` 分隔（若事件披露块为空则按旧装配语义收敛为单个 `---`）。
"""

# 资金流向字段族：仅在 panel 实际载入 ff_* 列时注入提示词（与插件数据覆盖联动）。
_FF_FIELD_ROWS_MD = """| `$ff_main_net` | 主力净流入（元） |
| `$ff_super_net` | 超大单净流入（元） |
| `$ff_large_net` | 大单净流入（元） |
| `$ff_medium_net` | 中单净流入（元） |
| `$ff_small_net` | 小单净流入（元） |
"""

_FF_ADVICE_MD = (
    "> **资金流向使用建议**：`$ff_*` 列为**绝对金额**（元），截面分布极端右偏，"
    "**禁止直接使用原始值**。必须先做截面标准化：`RANK($ff_super_net)` 或 "
    "`CS_ZSCORE(CS_WINSORIZE($ff_super_net, 0.01, 0.99))`。经济直觉：超大单净流入为正而小单净流出 "
    "→ 机构吸筹散户出逃 → 正 alpha；反之亦然。可做**资金分歧因子**："
    "`DIVERGENCE_RANK($ff_super_net, $ff_small_net)` 量化机构-散户方向分歧。\n"
    ">\n"
    "> **⚠️ 实测警示（2026-10-19，全 A 月频 rank IC，label_20d）**：裸资金流强度"
    "（`($ff_super_net+$ff_large_net)/$amount`）train IC +0.004 / val +0.003，"
    "裸分歧（`($ff_super_net-$ff_small_net)/$amount`）train -0.002 / val -0.007 —— "
    "**两者都接近 0，几乎无独立 alpha**。资金面不要单独当信号主体，"
    "必须做成**条件腿**（如资金流 × 低换手、资金流 × 价格背离、资金流作为其他主信号的"
    "门控状态），或用 `CROWD_*`/`CHIP_*` 结构算子改写资金流信息。别把评估预算花在"
    "裸资金流排序上。"
)

FF_PANEL_COLUMNS = (
    "ff_main_net", "ff_super_net", "ff_large_net", "ff_medium_net", "ff_small_net",
)

_FUNDAMENTAL_SECTION_MD = """### 基本面与披露日历（`build_panel --with-fundamentals` 并入）

季频财务数据经**严格 PIT** 展开为日频：财报公告日 D **不可用**，**D 的下一交易日**起该期字段才可引用；两期之间 **ffill** 保持最近已披露值。披露前为 NaN 属正常，勿当缺失错误。

**盈利、质量与增长指标**（`fina_indicator` → 日频，前缀 `funda_`）：

| 字段 | 说明 |
|------|------|
| `$funda_roe` / `$funda_roa` / `$funda_roic` | 净资产收益率 / 总资产报酬率 / 投入资本回报率 |
| `$funda_gross_margin` / `$funda_net_margin` | 毛利率 / 净利率 |
| `$funda_debt_to_assets` | 资产负债率 |
| `$funda_current_ratio` / `$funda_quick_ratio` | 流动比率 / 速动比率 |
| `$funda_eps` / `$funda_eps_diluted` / `$funda_bps` | 每股收益 / 稀释EPS / 每股净资产 |
| `$funda_ocfps` | 每股经营现金流 |
| `$funda_ebitda` | 息税折旧摊销前利润（绝对额，注意规模标准化） |
| `$funda_profit_dedt` | 扣非净利润（绝对额，注意规模标准化） |
| `$funda_netprofit_yoy` / `$funda_or_yoy` / `$funda_tr_yoy` | 归母净利 / 营业收入 / 营业总收入 同比%（财报期同比） |
| `$funda_ocf_yoy` / `$funda_roe_yoy` | 经营现金流 / ROE 同比% |

> 同比字段为**财报期同比**的 PIT 阶跃序列：披露生效日起跳变，两期之间恒定；勿当作日频变化率使用。

**财报科目**（绝对金额，用时先做规模标准化，如 `DIVIDE($funda_ocf, $funda_total_assets)`）：

| 类别 | 字段 |
|------|------|
| 利润表 | `$funda_total_revenue`、`$funda_net_profit`、`$funda_operate_profit`、`$funda_ebit`、`$funda_selling_expense`、`$funda_admin_expense`、`$funda_finance_expense`、`$funda_rd_expense` |
| 资产负债表 | `$funda_total_assets`、`$funda_total_liabilities`、`$funda_total_equity`、`$funda_current_assets`、`$funda_current_liabilities`、`$funda_inventory`、`$funda_accounts_receivable`、`$funda_fixed_assets`、`$funda_goodwill`、`$funda_cash` |
| 现金流量表 | `$funda_ocf`、`$funda_icf`、`$funda_fcf`、`$funda_free_cashflow` |
| 日历锚点 | `$funda_end_date`、`$funda_ann_date`（报告期末 / 公告日） |

**使用建议**（基本面/慢因子）：

- 基本面列在日频上**阶跃+持有**，窗口单位为**交易日**；约 60 日 ≈ 一季。
- 科目金额是绝对值：先除以规模（总资产/营收/流通市值），再 `CS_WINSORIZE` + `RANK`。
- 截面组合建议 `CS_NEUTRALIZE(..., CS_BUCKET(LOG($float_cap), 10))` 市值中性。

> 行尾可写 `#` 注释；字符串内 `#` 保留。"""

_FUNDAMENTAL_DISABLED_MD = (
    "### 基本面\n\n"
    "**本次运行未载入基本面列**：请勿在表达式中使用任何 `$funda_*` / `$funda_fs_*` 字段"
    "（当前数据面仅限价量/行情，专注价量因子挖掘）。\n\n"
    "> 行尾可写 `#` 注释；字符串内 `#` 保留。"
)

# 业绩预告（forecast 插件，pred_*）
_PRED_SECTION_MD = """### 业绩预告（PIT 日频阶跃序列，`pred_*`）

业绩预告以公告日为 PIT 锚点展开为日频：公告日当天起引用最近一次预告，
两期之间恒定（阶跃+持有）。首次预告前为 NaN。

| 字段 | 说明 |
|------|------|
| `$pred_direction` | 预告方向：+1 预增/略增/扭亏/续盈，-1 预减/略减/首亏/续亏/增亏，0 不确定 |
| `$pred_change_mid` | 预告净利同比变动区间中值（%） |
| `$pred_net_profit_mid` | 预告归母净利润区间中值（绝对额，万元级，用时先规模标准化） |
| `$pred_surprise` | 预告隐含同比 = 净利中值/上年同期归母净利 − 1 |
| `$pred_days_since` | 距最近一次预告的自然日天数（衰减可用 `EXP($pred_days_since/30)` 类构造） |

> 使用建议：`$pred_surprise` 是最直接的"预告超预期"代理，可做 `TS_FILL_NAN` 前向
> 逻辑已内建；阶跃序列勿当连续变量做短窗口动量。
"""

# 股东人数（shareholder_counts 插件，holder_*）
_HOLDER_SECTION_MD = """### 股东人数（PIT 日频阶跃序列，`holder_*`）

股东户数 = 筹码集中度经典代理：户数下降 = 筹码向大资金集中。以公告日为
PIT 锚点（统计截止日 `count_date` 可能滞后公告数周，勿用）。首次公告前为 NaN。

| 字段 | 说明 |
|------|------|
| `$holder_count` | 股东户数（户） |
| `$holder_count_chg_pct` | 较上期变化 %（负值 = 筹码集中） |
| `$holder_avg_float_shares` | 户均流通股数 |
| `$holder_avg_value` | 户均持股市值（元） |
| `$holder_days_since` | 距最近一次公告的自然日天数 |

> 使用建议：筹码集中因子 `RANK($holder_count_chg_pct)` 取反即"集中度改善"排序；
> 与价量互动（集中 + 涨幅背离 = 出货陷阱）是经典方向。
>
> **实测锚点（2026-10-19，全 A 月频 rank IC，label_20d）**：`$holder_count_chg_pct`
> 经 `TS_MEAN(...,20)` 平滑后 train(2020-22) IC -0.011 / ICIR -0.28，
> val(2023-24) IC **-0.026 / ICIR -0.52** —— 是本组事件/资金面里**最强且样本外更稳**
> 的信号（负值 = 户数下降 = 筹码集中 = 未来收益更高）。原始未平滑版 val IC -0.022。
> 优先在这个面上做深度构造（平滑窗/行业中性/与换手率交互），而不是另找新字段。
"""

# 龙虎榜 + 大宗交易（event_faces 插件，dt_* / bt_*）
_EVENT_FACES_SECTION_MD = """### 龙虎榜 / 大宗交易（日频稠密化，`dt_*` / `bt_*`）

稀疏事件已稠密化为全股票日频：滚动 90 个**交易日**窗口的次数/金额，
**无事件填 0**（覆盖率 100%，语义为"窗口内无事件"）；`*_days_since` 为距
最近一次事件的自然日天数（从未发生为 NaN）。

| 字段 | 说明 |
|------|------|
| `$dt_cnt_90d` | 近 90 交易日龙虎榜上榜次数（0=无） |
| `$dt_net_buy_90d` | 近 90 交易日龙虎榜净买入合计（元，可负） |
| `$dt_days_since` | 距最近一次上榜天数 |
| `$bt_cnt_90d` | 近 90 交易日大宗交易笔数（0=无） |
| `$bt_amt_90d` | 近 90 交易日大宗成交金额合计（万元） |
| `$bt_premium_last` | 最近一笔大宗折溢价率（-0.05 = 折价 5%；折价成交 = 大资金让利出货信号） |
| `$bt_days_since` | 距最近一次大宗交易天数 |

> 使用建议：`$dt_*`/`$bt_*` 金额列绝对值右偏，先 `RANK` 或
> `CS_WINSORIZE` 再用；"上榜热度 + 随后回落"与"大宗折价 + 筹码集中"
> 是经典博弈方向。`*_days_since` 为 NaN 时表示样本期内从未发生，宜配
> 计数列使用（计数已含 0 语义）。
"""

# 限售解禁（share_unlock 插件，unlock_*）——2026-10-19 新增：面板实有 5 列，
# 此前提示词完全未提及（数据在、LLM 不知道），是"供给冲击日历"类机制唯一可观测字段。
_UNLOCK_SECTION_MD = """### 限售解禁（`unlock_*`，供给冲击日历）

解禁 = 可流通股本突然扩大的**已知未来事件**，是 A 股最干净的一类"可预见的抛压供给"。
首次解禁之前不产出行（`$unlock_days_since` 为空），故覆盖率约 39%（不是数据错误）。

| 字段 | 说明 |
|------|------|
| `$unlock_days_since` | 距最近一次**已发生**解禁的自然日天数 |
| `$unlock_ratio_last` | 最近一次解禁占流通股比例（%） |
| `$unlock_ratio_past180` | 过去 180 天内已解禁比例合计（%），无解禁填 **0** |
| `$unlock_days_to_next` | 距下一次**计划**解禁的自然日天数 |
| `$unlock_ratio_next` | 下一次计划解禁占流通股比例（%） |

> **⚠️ PIT 分级（必须遵守）**：`$unlock_days_since` / `$unlock_ratio_last` /
> `$unlock_ratio_past180` 三列**严格 PIT 安全**（只用已发生事实），可自由使用；
> `$unlock_days_to_next` / `$unlock_ratio_next` 是**计划口径**，curated 源表无公告日、
> 无法证明信息可知时点，**存在前视风险**——只可作为门控/状态变量做辅助，
> 禁止单独作为信号主体，更禁止对它们做"越临近越好"的短期择时。
>
> 使用建议：
> - **解禁压力已释放**：`$unlock_ratio_past180` 高 + 股价未跌 = 供给被消化 → 后续弹性；
> - **解禁前折价**：临近大比例解禁时股价常被提前压低，事件后（`$unlock_days_since` 小
>   且 `$unlock_ratio_last` 大）出现均值回复是经典事件驱动方向；
> - 比例列先 `CS_WINSORIZE` 再 `RANK`；`$unlock_ratio_past180` 因填 0 而左截断，
>   直接 `RANK` 会把"无解禁"和"解禁极小"混为一谈，宜配 `$unlock_days_since` 区分。
>
> **实测锚点（2026-10-19，全 A 月频 rank IC，label_20d）**：`$unlock_ratio_past180`
> 经 `TS_MEAN(...,20)` 平滑后 train IC -0.017 / ICIR -0.25、val IC **-0.018 / ICIR -0.29**
> （方向稳定、样本外不衰减）；再做市值交互 `MULTIPLY($unlock_ratio_past180, NEG($float_cap))`
> 后 train IC +0.015 / val **+0.021（ICIR +0.35）** —— **供给冲击在小市值股上被放大**，
> 单因子裸用 IC 只有 0.02 量级，必须靠"解禁 × 流动性/市值/换手"的条件化才够交付门槛。
"""

# 融资融券（margin 插件，mgn_*）
_MARGIN_SECTION_MD = """### 融资融券（日频直给，`mgn_*`）

T 日盘后披露，两融余额 = 杠杆资金的多空方向/拥挤度；**非两融标的为 NaN**。

| 字段 | 说明 |
|------|------|
| `$mgn_balance` | 融资余额（元，多头杠杆存量） |
| `$mgn_buy` | 融资买入额（元，当日杠杆加仓流量） |
| `$mgn_short_balance` | 融券余额（元，券源受限、量级小，单独解读勿与融资混算） |
| `$mgn_short_sell_vol` | 融券卖出量（股） |

> 使用建议：绝对额右偏，先用 `DIVIDE($mgn_buy, $amount)` 或
> `CS_ZSCORE(CS_WINSORIZE(...))` 再进算子；`$mgn_buy/$amount` 抬升是
> "杠杆情绪过热"信号（结合涨幅看反转），融资余额相对流通市值可做拥挤度。
"""

# 机构持仓（institutional 插件，inst_*）
_INSTITUTIONAL_SECTION_MD = """### 机构持仓（季频 PIT 阶跃，`inst_*`）

按定期报告**实际披露日**（披露日历对齐）PIT 展开：披露日次日起引用最近一期。
未披露/无机构数据为 NaN（勿填 0——0 会被当"无机构持股"信号）。

**合计口径**：

| 字段 | 说明 |
|------|------|
| `$inst_count` | 机构家数（基金/QFII/社保/保险/券商/信托等合计） |
| `$inst_ratio` | 机构持股比例合计（%，如 37.0 = 37%） |
| `$inst_mv` | 机构持股市值合计（元） |
| `$inst_days_since` | 距最近一期披露的自然日天数 |

**分机构类型口径**（2026-10-19 补齐：面板实有 22 列，此前提示词只暴露了合计口径的 4 个）：

| 机构类型 | 家数 / 比例 / 市值 |
|------|------|
| 基金 | `$inst_fund_count` / `$inst_fund_ratio` / `$inst_fund_mv` |
| QFII | `$inst_qfii_count` / `$inst_qfii_ratio` / `$inst_qfii_mv` |
| 社保 | `$inst_social_security_count` / `$inst_social_security_ratio` / `$inst_social_security_mv` |
| 保险 | `$inst_insurance_count` / `$inst_insurance_ratio` / `$inst_insurance_mv` |
| 券商 | `$inst_broker_count` / `$inst_broker_ratio` / `$inst_broker_mv` |
| 信托 | `$inst_trust_count` / `$inst_trust_ratio` / `$inst_trust_mv` |

> 使用建议：机构加仓/抱团是慢变量，配 `DELTA(..., ~63)` 或跨期变化看方向。
> **但方向不稳定（2026-10-19 实测，全 A 月频 rank IC）**：
> `$inst_ratio` 水平值 train(2020-2022) IC +0.029 / val(2023-2024) 仅 +0.005（衰减 84%）；
> `$inst_ratio` 的 63 日变化 train +0.001 / val **-0.017**（方向翻转）；
> `SUBTRACT($inst_fund_ratio, $inst_qfii_ratio)` 风格分歧 train +0.022 / val **-0.042**（完全反转）。
> 结论：**机构面别直接当方向性信号**，要么做分段/状态门控（regime），要么只用来做加权腿；
> 单看"机构加仓 → 未来涨"在近年数据上不成立，务必在 train/val 两段分别验证方向再交付。
>
> **分类型才是信息富矿（合计口径会把结构差异抹平）**：
> - **风格分歧**：`SUBTRACT($inst_fund_ratio, $inst_insurance_ratio)` —— 公募与险资的
>   久期/风险偏好不同。⚠️ 该分歧本身方向会反转（见上），适合做**状态门控/条件腿**而非主信号；
> - **聪明钱 vs 被动**：`DIVIDE($inst_qfii_ratio, $inst_fund_ratio)` 的**变化方向**比水平值有意义；
> - **国家队信号**：`$inst_social_security_ratio`（社保）跨越式抬升是低频高信噪事件，
>   宜做 `DELTA` 变化量门控而非连续动量；
> - 各类比例是**存量占比**，做变化量请用 `DELTA(..., 63/126)` 而非 `TS_PCTCHANGE`（分母可接近 0）。
"""

# 十大流通股东（top_holders 插件，th_*）
_TOP_HOLDERS_SECTION_MD = """### 十大流通股东（季频 PIT 阶跃，`th_*`）

**流通股口径**（占总股本口径分母不同、不可混算），按公告日 PIT 展开。

| 字段 | 说明 |
|------|------|
| `$th_top1_pct` | 第一大流通股东占比（%） |
| `$th_top10_pct` | 前十大流通股东占比合计（%，数值越高筹码越集中） |
| `$th_inst_pct` | 流通股东中机构占比合计（%） |
| `$th_inst_count` | 流通股东中机构家数 |
| `$th_days_since` | 距最近一次股东披露的自然日天数 |

> 使用建议：筹码集中度变化配价量背离用——集中度升 + 缩量阴跌 = 有资金
> 悄悄收筹；集中度降 + 放量 = 出货。披露是季频，勿做短窗差分。
"""

# 业绩快报（express 插件，exp_*）
_EXPRESS_SECTION_MD = """### 业绩快报（PIT 日频阶跃，`exp_*`）

快报在定期报告前主动披露初步财务数，以公告日为 PIT 锚点。

| 字段 | 说明 |
|------|------|
| `$exp_revenue` | 营业收入（元，绝对额先规模标准化） |
| `$exp_operate_profit` | 营业利润（元） |
| `$exp_net_profit` | 归母净利润（元） |
| `$exp_eps` | 每股收益（元） |
| `$exp_roe` | ROE（%） |
| `$exp_netprofit_yoy` | 快报隐含归母净利同比（%，基准期亏损/为零时 NaN） |
| `$exp_days_since` | 距最近一次快报公告的自然日天数 |

> 使用建议：快报/预告间的差异（`$exp_netprofit_yoy` vs `$pred_surprise`、
> 与 `$funda_netprofit_yoy` 上次定期报告对比）比单看快报更有信息量。
"""

# 披露日历（disclosure 插件，ds_*）
_DISCLOSURE_SECTION_MD = """### 披露日历（实际披露后安全特征，`ds_*`）

预约披露表是"现值语义"（改约覆盖、无发布时间戳），因此**不提供**"距预约
还有 N 天"类事前信号；以下均为实际披露后成立的特征。

| 字段 | 说明 |
|------|------|
| `$ds_delay_days` | 最近一次实际披露相对**最终预约日**推迟天数（正=推迟、负=提前） |
| `$ds_delay_vs_first` | 相对**首次预约日**推迟天数（含中途改约信息） |
| `$ds_days_since_actual` | 距最近一次实际披露的自然日（披露节奏/是否"掉队"代理） |

> 使用建议：与定期报告季频节奏对齐，`$ds_days_since_actual` 明显大于季度
> 常规间隔可作"该披露未披露/异常"风险代理；`$ds_delay_*` 作状态标签而非
> 连续变量做动量。
"""

# 现金分红（dividend 插件，div_*）
_DIVIDEND_SECTION_MD = """### 现金分红（实施公告口径，`div_*`）

只取"已实施"公告（金额与除息日确定）为 PIT 锚点；预案/取消不计入。

| 字段 | 说明 |
|------|------|
| `$div_cash_div` | 最新公告实施的每股现金股利（税前，元/股），公告后保持到下一份 |
| `$div_days_to_ex` | 距最近已公告未实施分红的除息日自然日倒数（除息后为空） |
| `$div_days_since_ann` | 距最近一次实施公告的自然日天数 |

> 使用建议：股息率 = `DIVIDE($div_cash_div, $close)`；`$div_days_to_ex` 非空
> 即"即将除息"，可做抢权/填权事件因子（注意除息日价格自然回落）。
"""

EVENT_FACE_PANEL_COLUMNS = (
    "dt_cnt_90d", "dt_net_buy_90d", "dt_days_since",
    "bt_cnt_90d", "bt_amt_90d", "bt_premium_last", "bt_days_since",
)
# 限售解禁：注入判定单独一组（列族存在 ≠ 事件面已载入，缺一列就不该教）。
UNLOCK_PANEL_COLUMNS = (
    "unlock_days_since", "unlock_ratio_last", "unlock_ratio_past180",
    "unlock_days_to_next", "unlock_ratio_next",
)
FORECAST_PANEL_COLUMNS = (
    "pred_direction", "pred_change_mid", "pred_net_profit_mid", "pred_surprise", "pred_days_since",
)
HOLDER_PANEL_COLUMNS = (
    "holder_count", "holder_count_chg_pct", "holder_avg_float_shares",
    "holder_avg_value", "holder_days_since",
)
MARGIN_PANEL_COLUMNS = (
    "mgn_balance", "mgn_buy", "mgn_short_balance", "mgn_short_sell_vol",
)
INST_PANEL_COLUMNS = ("inst_count", "inst_ratio", "inst_mv")
TH_PANEL_COLUMNS = (
    "th_top1_pct", "th_top10_pct", "th_inst_pct", "th_inst_count",
)
EXPRESS_PANEL_COLUMNS = (
    "exp_revenue", "exp_operate_profit", "exp_net_profit", "exp_eps",
    "exp_roe", "exp_netprofit_yoy",
)
DISCLOSURE_PANEL_COLUMNS = ("ds_delay_days", "ds_delay_vs_first", "ds_days_since_actual")
DIVIDEND_PANEL_COLUMNS = ("div_cash_div", "div_days_to_ex", "div_days_since_ann")
ANNOUNCEMENT_PANEL_COLUMNS = ("ann_cnt_20d", "ann_days_since")
RESEARCH_VIEWS_PANEL_COLUMNS = ("ac_eps_fy", "rc_cnt_90d")
FINANCIAL_ITEMS_PANEL_COLUMNS = ("fsi_net_profit", "fsi_total_assets")

# 公告事件（announcement 插件，ann_*）
_ANNOUNCEMENT_SECTION_MD = """### 公告事件（PIT 日频，`ann_*`）

个股公告索引按**公告日**展开为日频阶跃：仅当日及之前已公告的可见。

| 字段 | 说明 |
|------|------|
| `$ann_cnt_5d` / `$ann_cnt_20d` | 近 5 / 20 个**交易日**公告条数（信息密度） |
| `$ann_days_since` | 距最近一条公告的自然日（越小越"有事"） |
| `$ann_flag_reduce` / `$ann_flag_increase` | 窗口内减持 / 增持类公告（1/0） |
| `$ann_flag_buyback` / `$ann_flag_pledge` | 回购 / 质押类公告（1/0） |
| `$ann_flag_lawsuit` / `$ann_flag_inquiry` | 诉讼仲裁 / 问询函（1/0，风险类） |
| `$ann_flag_forecast` | 业绩预告/预增预减类公告（1/0） |

> 事件多为短期冲击，须与反转/换手结构组合；风险类标记（诉讼/问询/质押）适合做负向门控，
> 不要单独当信号。
"""

# 研报视角（research_views 插件，ac_* 一致预期 / rc_* 卖方预测）
_RESEARCH_VIEWS_SECTION_MD = """### 研报视角：一致预期与卖方预测（PIT 日频，`ac_*` / `rc_*`）

`ac_*` = 卖方一致预期（按预期日 PIT）；`rc_*` = 个股研报预测/评级聚合。研报条数是**拥挤度**
代理：覆盖越多、预期差越小。

| 字段 | 说明 |
|------|------|
| `$ac_eps_fy` / `$ac_pe_fy` | 一致预期 EPS（最近预测年度）/ 对应 PE |
| `$ac_target_price` / `$ac_analyst_count` | 一致目标价（元）/ 覆盖分析师家数 |
| `$ac_rating` / `$ac_days_since` | 一致评级分值 / 距预期更新日的自然日 |
| `$rc_cnt_90d` / `$rc_days_since` | 近 90 日研报条数（拥挤度）/ 距最近一份研报的自然日 |
| `$rc_eps_y` / `$rc_pe_y` / `$rc_roe_y` / `$rc_tp_wan` | 最近研报给出的 EPS / PE / ROE 预测 / 预测利润总额（万元） |
| `$rc_rating` / `$rc_target_price` | 最近研报评级分值 / 目标价（元） |

> **⚠️ 覆盖率实测（2026-10-19，2020-2026 全面板 781.9 万行）**：
> `rc_*` 行级非空约 **27%**（有研报覆盖的标的才出现）；
> `ac_*` 一致预期仅 **0.85%** —— 几乎等于不可用（curated 一致预期表基本是空的）。
> **`ac_*` 不要作为主力信号源**：它会让 coverage 门槛直接失败、白烧评估预算；
> 需要"预期"类信息时优先用 `$rc_*`、`$pred_surprise`（90.4%）、`$exp_netprofit_yoy`（50.4%）。
>
> 研报数据是**低频慢变量**：只与日频价量做门控或残差，不要直接当短周期信号；
> `$rc_cnt_90d` 高 + 价格滞涨 = 拥挤无催化，预期上调配低换手更稳。
"""

# 财报明细项（financial_items 插件，fsi_*，单位元）
_FINANCIAL_ITEMS_SECTION_MD = """### 财报明细项（PIT 日频阶跃，`fsi_*`，单位元）

按**实际公告日** PIT 展开的利润表/资产负债表明细（东财口径；利润表为**年初至今累计**）。

| 字段 | 说明 |
|------|------|
| `$fsi_revenue` / `$fsi_net_profit` / `$fsi_operating_profit` | 营业总收入 / 净利润 / 营业利润（累计） |
| `$fsi_total_assets` / `$fsi_total_liabilities` / `$fsi_total_equity` | 总资产 / 总负债 / 股东权益 |
| `$fsi_monetary_funds` / `$fsi_fixed_assets` | 货币资金 / 固定资产 |
| `$fsi_net_cash_operate` / `$fsi_capex` | 经营活动现金流净额 / 资本开支（累计） |

> 全是规模量：先除以规模（`$fsi_total_assets` 或 `$tot_cap`）再进算子；商誉/应收占比抬升是
> 质量恶化信号；累计口径跨季比较用 `DELTA(..., 252)` 做同比。
"""

NAME = "data_fields"
TITLE = "行情变量与字段族（资金流/基本面/事件披露）"
ORDER = 50
REQUIRED = True
SEP_BEFORE = "\n\n---\n\n"

_VARIABLES_TABLE_HEAD = """### 可用行情变量

表达式引用列须 **`$` + 列名**：

| 字段 | 说明 |
|------|------|
| `$open` / `$high` / `$low` / `$close` | 原始 OHLC |
| `$adj_open` / `$adj_high` / `$adj_low` / `$adj_close` | 复权 OHLC（**优先**） |
| `$volume` / `$amount` | 成交量 / 成交额 |
| `$float_cap` / `$tot_cap` | 流通 / 总市值 |
| `$vwap` | 成交量加权均价（与 `$close` 同单位尺度：amount/volume） |
| `$adj_vwap` | 后复权 VWAP（`$vwap × $adjfactor`，与 `$adj_close` 同复权口径） |
| `$ret` | 日 adj_close pct_change（按 instrument） |
| `$turnover_rate` / `$turnover_rate_f` | 换手率（%）/ 自由流通股换手率（%；`$turnover` 是 `$turnover_rate` 的别名） |
| `$volume_ratio` | 量比（当日成交量/近 5 日均量） |
| `$pe` / `$pe_ttm` / `$pb` / `$ps_ttm` | 估值：PE（静态/滚动）/ PB / PS(TTM)（无值 = 亏损或不可比） |
| `$dv_ttm` | 股息率（%，近 12 个月现金分红/现价） |
| `$total_share` / `$float_share` / `$free_share` | 总 / 流通 / 自由流通股本（万股） |
| `$up_limit` / `$down_limit` | 当日涨停价 / 跌停价（元，与 `$close` 同口径） |
| `$listed_days` | 上市天数（次新股识别） |
| `$is_trade` / `$not_st` | 可交易 / 非 ST 标记 |
| `$industry_sw_l1` / `$industry_sw_code` | 申万一级行业**离散码**（6 位码前 2 位）/ 完整 6 位申万码（严格 PIT 快照）；仅用于分组，不做数值运算 |
| `$idx_szcomp` / `$idx_chinext` / `$idx_in_index` | 深证成指 / 创业板指 / 任一指数成分标记（1/0，PIT 快照） |
"""

_INDUSTRY_NOTE = """
> **行业中性化**：行业码是离散组号，直接 `CS_NEUTRALIZE(factor, $industry_sw_l1)` 即为行业内去均值；**勿**对它套 `CS_BUCKET`。

"""

# ── 数据面聚焦时的行情变量表（按勾选面裁剪）──
# 说明：非聚焦路径仍用上面的 _VARIABLES_TABLE_HEAD（逐字节冻结，黄金基线依赖）；
# 这里为聚焦路径单独提供行族常量，避免重构老常量引入字节漂移。
_SCOPED_VAR_TABLE_HEAD = """### 可用行情变量

表达式引用列须 **`$` + 列名**：

| 字段 | 说明 |
|------|------|
"""
_SCOPED_VAR_PRICE_ROWS_MD = """| `$open` / `$high` / `$low` / `$close` | 原始 OHLC |
| `$adj_open` / `$adj_high` / `$adj_low` / `$adj_close` | 复权 OHLC（**优先**） |
| `$vwap` | 成交量加权均价（与 `$close` 同单位尺度：amount/volume） |
| `$adj_vwap` | 后复权 VWAP（`$vwap × $adjfactor`，与 `$adj_close` 同复权口径） |
| `$ret` | 日 adj_close pct_change（按 instrument） |
"""
_SCOPED_VAR_VOLUME_ROWS_MD = """| `$volume` / `$amount` | 成交量 / 成交额 |
"""
_SCOPED_VAR_NEUTRAL_ROWS_MD = """| `$float_cap` / `$tot_cap` | 流通 / 总市值 |
| `$turnover_rate` / `$volume_ratio` | 换手率（%，别名 `$turnover`）/ 量比 |
| `$pe_ttm` / `$pb` / `$dv_ttm` | 估值 PE(TTM) / PB / 股息率（%） |
| `$up_limit` / `$down_limit` / `$listed_days` | 涨停价 / 跌停价（元）/ 上市天数 |
| `$is_trade` / `$not_st` | 可交易 / 非 ST 标记 |
| `$industry_sw_l1` | 申万一级行业**离散码**（严格 PIT 快照）；仅用于分组，不做数值运算 |
| `$idx_szcomp` / `$idx_chinext` | 深证成指 / 创业板指成分标记（1/0，PIT 快照） |
"""


def _scoped_variables_table(scope: set[str], focus: set[str] | None = None) -> str:
    """按聚焦面（含算子隐含输入面）裁剪行情变量表。

    例：只勾筹码面时仍列出 OHLC/volume —— CHIP_* 需要它们作输入；
    但只勾基本面时不列任何行情列（基本面因子不需要价量输入）。
    ``focus`` 给出时，隐含输入面（scope - focus）的行组之后附"仅作输入"提示，
    防止 LLM 把它们当聚焦列直接拼纯价量因子（2026-09-10 run 51e02d47a3f3 教训）。
    """
    from alphaagent.factor.mining.memory.expressions import facet_column_hint

    rows = _SCOPED_VAR_TABLE_HEAD
    if "价量面" in scope:
        rows += _SCOPED_VAR_PRICE_ROWS_MD
    if "量能面" in scope:
        rows += _SCOPED_VAR_VOLUME_ROWS_MD
    rows += _SCOPED_VAR_NEUTRAL_ROWS_MD
    input_only = sorted(set(scope) - set(focus or ()))
    if input_only:
        hints = "；".join(f"{f}（{facet_column_hint(f) or '—'}）" for f in input_only)
        rows += (
            "\n> **输入列提示**：" + hints + " 的列本轮**仅作输入**——只能与聚焦面列"
            "组合使用（比值/相关/门控等）或作为聚焦面专属算子的输入；单独用它们构成"
            "因子会被 facet_lock_violation 拦截。\n"
        )
    return rows


def render(ctx) -> str:  # noqa: ANN001
    cols = ctx.panel_columns
    focus = [str(f) for f in (getattr(ctx, "focus_facets", None) or ())]
    scope = set(focus)
    # 行情变量表用"含隐含输入面"的范围（CHIP_* 需要 OHLC/volume 作输入），
    # 字段族区块仍按勾选面严格裁剪（未选基本面就不显示 funda_*）。
    from alphaagent.factor.mining.memory.expressions import facet_allowed_scope

    input_scope = facet_allowed_scope(focus) if scope else scope

    def _want(face: str) -> bool:
        """字段族注入判定：聚焦生效时只注入勾选面，未聚焦时全量（老行为）。"""
        return not scope or face in scope

    # 消融白名单（field_family_scope）：非空时只注入列名命中白名单前缀的字段族。
    # 仅影响字段族区块，行情变量表与数据加载不受影响（P6a 价量字段族消融用）。
    scope_prefixes = getattr(ctx, "field_family_scope", None)

    def _family_allowed(prefixes: tuple[str, ...]) -> bool:
        if scope_prefixes is None:
            return True
        return any(p in scope_prefixes for p in prefixes)

    funda_effective = ctx.include_fundamentals and (
        cols is None or any(c.startswith("funda_") for c in cols)
    ) and _family_allowed(("funda_",))
    ff_available = (cols is None or all(c in cols for c in FF_PANEL_COLUMNS)) and _want("资金面") and _family_allowed(("ff_",))
    ff_rows = _FF_FIELD_ROWS_MD if ff_available else ""
    ff_advice = _FF_ADVICE_MD if ff_available else ""
    funda_block = _FUNDAMENTAL_SECTION_MD if funda_effective else _FUNDAMENTAL_DISABLED_MD
    # 聚焦时未选基本面 → 该区块整体不注入（不是显示"未启用"文案，省 token 且不误导）
    include_funda_block = True if not scope else (funda_effective and "基本面" in scope)

    # 事件/披露字段族：按 panel 实际列 + 聚焦面逐块拼接（插件缺数据时对应块不注入）
    event_blocks: list[str] = []
    if (cols is None or any(c.startswith("pred_") for c in cols)) and _want("业绩面") and _family_allowed(("pred_",)):
        event_blocks.append(_PRED_SECTION_MD)
    if (cols is None or any(c.startswith("holder_") for c in cols)) and _want("股东面") and _family_allowed(("holder_",)):
        event_blocks.append(_HOLDER_SECTION_MD)
    if (cols is None or any(c in cols for c in EVENT_FACE_PANEL_COLUMNS)) and _want("事件面") and _family_allowed(("dt_", "bt_")):
        event_blocks.append(_EVENT_FACES_SECTION_MD)
    # 限售解禁：与龙虎榜/大宗同属事件面，但列族独立（dt_/bt_ 消融时不应连带隐藏）。
    if (cols is None or any(c in cols for c in UNLOCK_PANEL_COLUMNS)) and _want("事件面") and _family_allowed(("unlock_",)):
        event_blocks.append(_UNLOCK_SECTION_MD)
    if (cols is None or any(c in cols for c in MARGIN_PANEL_COLUMNS)) and _want("两融面") and _family_allowed(("mgn_",)):
        event_blocks.append(_MARGIN_SECTION_MD)
    if (cols is None or any(c in cols for c in INST_PANEL_COLUMNS)) and _want("机构面") and _family_allowed(("inst_",)):
        event_blocks.append(_INSTITUTIONAL_SECTION_MD)
    if (cols is None or any(c in cols for c in TH_PANEL_COLUMNS)) and _want("股东集中面") and _family_allowed(("th_",)):
        event_blocks.append(_TOP_HOLDERS_SECTION_MD)
    if (cols is None or any(c in cols for c in EXPRESS_PANEL_COLUMNS)) and _want("业绩面") and _family_allowed(("exp_",)):
        event_blocks.append(_EXPRESS_SECTION_MD)
    if (cols is None or any(c in cols for c in DISCLOSURE_PANEL_COLUMNS)) and _want("披露面") and _family_allowed(("ds_",)):
        event_blocks.append(_DISCLOSURE_SECTION_MD)
    if (cols is None or any(c in cols for c in DIVIDEND_PANEL_COLUMNS)) and _want("分红面") and _family_allowed(("div_",)):
        event_blocks.append(_DIVIDEND_SECTION_MD)
    if (cols is None or any(c in cols for c in ANNOUNCEMENT_PANEL_COLUMNS)) and _want("事件面") and _family_allowed(("ann_",)):
        event_blocks.append(_ANNOUNCEMENT_SECTION_MD)
    if (cols is None or any(c in cols for c in RESEARCH_VIEWS_PANEL_COLUMNS)) and _want("业绩面") and _family_allowed(("ac_", "rc_")):
        event_blocks.append(_RESEARCH_VIEWS_SECTION_MD)
    if (cols is None or any(c in cols for c in FINANCIAL_ITEMS_PANEL_COLUMNS)) and _want("基本面") and _family_allowed(("fsi_",)):
        event_blocks.append(_FINANCIAL_ITEMS_SECTION_MD)

    variables = (
        _VARIABLES_TABLE_HEAD
        if not scope
        else _scoped_variables_table(input_scope, set(focus))
    )
    parts = [variables + ff_rows + _INDUSTRY_NOTE]
    parts.append(ff_advice + "\n\n---\n\n" if ff_advice else "---\n\n")
    tail_blocks = ([funda_block] if include_funda_block else []) + event_blocks
    if tail_blocks:
        parts.append("\n\n---\n\n".join(tail_blocks) + "\n\n---")
    else:
        parts.append("---")
    return "".join(parts)

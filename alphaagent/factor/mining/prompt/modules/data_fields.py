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
    "`SUBTRACT(RANK($ff_super_net), RANK($ff_small_net))` 量化机构-散户方向分歧。"
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
    "**本次未载入基本面列**：请勿使用任何 `$funda_*` / `$funda_fs_*` 字段"
    "（本会话仅提供价量/行情列，专注价量因子）。\n\n"
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

| 字段 | 说明 |
|------|------|
| `$inst_count` | 机构家数（基金/QFII/社保/保险/券商/信托等合计） |
| `$inst_ratio` | 机构持股比例合计（%，如 37.0 = 37%） |
| `$inst_mv` | 机构持股市值合计（元） |
| `$inst_days_since` | 距最近一期披露的自然日天数 |

> 使用建议：机构加仓/抱团是慢变量，配 `DELTA(..., ~63)` 或跨期变化看方向；
> `$inst_ratio` 抬升 + 股价滞涨 = 筹码沉淀，反之高比例 + 减持窗口警惕踩踏。
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
| `$is_trade` / `$not_st` | 可交易 / 非 ST 标记 |
| `$industry_sw_l1` | 申万一级行业**离散码**（严格 PIT，`--with-industry` 时才有）；仅用于分组，不做数值运算 |
"""

_INDUSTRY_NOTE = """
> **行业中性化**：行业码是离散组号，直接 `CS_NEUTRALIZE(factor, $industry_sw_l1)` 即为行业内去均值；**勿**对它套 `CS_BUCKET`。

"""


def render(ctx) -> str:  # noqa: ANN001
    cols = ctx.panel_columns
    funda_effective = ctx.include_fundamentals and (
        cols is None or any(c.startswith("funda_") for c in cols)
    )
    ff_available = cols is None or all(c in cols for c in FF_PANEL_COLUMNS)
    ff_rows = _FF_FIELD_ROWS_MD if ff_available else ""
    ff_advice = _FF_ADVICE_MD if ff_available else ""
    funda_block = _FUNDAMENTAL_SECTION_MD if funda_effective else _FUNDAMENTAL_DISABLED_MD

    # 事件/披露字段族：按 panel 实际列逐块拼接（插件缺数据时对应块不注入）
    event_blocks: list[str] = []
    if cols is None or any(c.startswith("pred_") for c in cols):
        event_blocks.append(_PRED_SECTION_MD)
    if cols is None or any(c.startswith("holder_") for c in cols):
        event_blocks.append(_HOLDER_SECTION_MD)
    if cols is None or any(c in cols for c in EVENT_FACE_PANEL_COLUMNS):
        event_blocks.append(_EVENT_FACES_SECTION_MD)
    if cols is None or any(c in cols for c in MARGIN_PANEL_COLUMNS):
        event_blocks.append(_MARGIN_SECTION_MD)
    if cols is None or any(c in cols for c in INST_PANEL_COLUMNS):
        event_blocks.append(_INSTITUTIONAL_SECTION_MD)
    if cols is None or any(c in cols for c in TH_PANEL_COLUMNS):
        event_blocks.append(_TOP_HOLDERS_SECTION_MD)
    if cols is None or any(c in cols for c in EXPRESS_PANEL_COLUMNS):
        event_blocks.append(_EXPRESS_SECTION_MD)
    if cols is None or any(c in cols for c in DISCLOSURE_PANEL_COLUMNS):
        event_blocks.append(_DISCLOSURE_SECTION_MD)
    if cols is None or any(c in cols for c in DIVIDEND_PANEL_COLUMNS):
        event_blocks.append(_DIVIDEND_SECTION_MD)
    event_disclosure_block = "\n\n---\n\n".join(event_blocks) if event_blocks else ""

    # 与旧装配逐字节一致：
    # - FF 表行带换行尾巴，接行业注释；FF 建议为空时其 "---" 段收敛
    # - 事件披露块存在时 funda 与事件之间有 "---"；事件为空时直接到 funda 结尾
    #   （旧装配 {{EVENT_DISCLOSURE_SECTION}}\n\n---\n\n 整体消失的语义）
    # - 尾部 "---" 由本模块携带（多周期之前），尾部空行数与旧 replace 链逐字节一致
    parts = [_VARIABLES_TABLE_HEAD + ff_rows + _INDUSTRY_NOTE]
    parts.append(ff_advice + "\n\n---\n\n" if ff_advice else "---\n\n")
    if event_disclosure_block:
        parts.append(funda_block + "\n\n---\n\n" + event_disclosure_block + "\n\n---")
    else:
        parts.append(funda_block + "\n\n---")
    return "".join(parts)

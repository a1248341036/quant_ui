# -*- coding: utf-8 -*-
"""核对聚宽 API 函数在官方文档 HTML 中的覆盖情况."""
import html as _html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
raw = (ROOT / "docs" / "jq_api_snapshot" / "api_full.html").read_text(
    encoding="utf-8")
text = _html.unescape(re.sub(r"<[^>]+>", " ", raw))

CANDIDATES = """
set_option set_benchmark set_slippage set_order_cost set_universe
set_subportfolios set_fixed_slippage
order order_target order_value order_target_value
market_order style MarketOrderStyle LimitOrderStyle
cancel_order get_orders get_open_orders
run_daily run_weekly run_monthly run_interval unschedule_all
initialize process_initialize before_trading_start handle_data
after_trading_end at_time
get_price history attribute_history get_current_data get_current_tick
get_fundamentals get_fundamentals_continuously get_index_stocks
get_all_securities get_security_info get_trade_days get_all_trade_days
get_extras get_locked_shares get_billboard_list get_money_fund
get_ticks get_price_change? normalize_code get_marginsec_stocks
get_margin_details get_all_margin_securities get_mtss
get_factor_values get_factor
get_industry_stocks get_industry get_concept_stocks get_concepts
finance.run_query macro.run_query get_yield_curve
log write_file read_file send_message check_limit
Order Trade Position SubPortfolio Portfolio SecurityUnitData Context
g. security context
poll subscription subscribe unsubscribe
""".split()

for name in CANDIDATES:
    n = text.count(name)
    print(f"{name:35s} {'OK ' if n else '-- '} x{n}")

"""跨模块共享的数据结构定义。"""

from __future__ import annotations

# Tushare daily_basic 每日指标（除 close/total_mv/circ_mv 外，直接入库的原生字段）
# 单位：turnover_rate/turnover_rate_f/dv_ratio/dv_ttm 为 %，volume_ratio 为倍数，
# pe/pe_ttm/pb/ps/ps_ttm 为比值，total_share/float_share/free_share 为万股。
DAILY_BASIC_COLUMNS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
]

# Panel 输出列（与 AlphaAgent-Stock 保持一致）
OUTPUT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adjfactor",
    "volume",
    "amount",
    "float_cap",
    "tot_cap",
    *DAILY_BASIC_COLUMNS,
    "is_trade",
    "not_st",
    "ret",
    "vwap",
    "adj_vwap",
    "label_1d_close_to_close",
    "label_1d_open_to_open",
    "label_5d_close_to_close",
    "label_10d_close_to_close",
    "label_20d_close_to_close",
]

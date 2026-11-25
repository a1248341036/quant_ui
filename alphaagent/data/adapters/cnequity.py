"""CNEquity 数据源 adapter —— 从 CNE 数据湖实时构建 AlphaAgent panel。

**插件化架构**：本模块是入口，实际数据加载由 ``plugins/`` 下的插件完成。
每个插件对应一个 CNE 数据集，声明列映射与加载逻辑。

加新数据源只需在 ``plugins/`` 下新建一个 .py 文件，无需修改本模块。
注册中心会自动发现并合并所有插件的数据。

用法
----
    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    panel = load_panel_from_cne(start="2020-01-01", end="2024-12-31")

在 StockEvalContext 中使用特殊 panel_path ``cne://`` 触发此 adapter，
SessionStore.create() 会检测该标识并走 adapter 而非读 parquet。

插件列表
--------
- ``stock_daily_wide`` (priority=0)：核心行情，OHLCV + adjfactor + 估值 + ST 标记
- 新增插件在 ``plugins/`` 目录下创建 .py 文件即可

架构
----
    load_panel_from_cne()
        → PluginRegistry.build_panel()
            → plugins/stock_daily_wide.load()    (核心, priority=0)
            → plugins/<other>.load()             (辅助, priority>0)
            → build_panel_from_hq()              (衍生列: adj_*, ret, vwap, label)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from alphaagent.data.adapters.registry import (
    DataSourcePlugin,
    PluginRegistry,
    get_registry,
)

logger = logging.getLogger(__name__)

# 标识符：panel_path == CNE_SOURCE 时走 adapter
CNE_SOURCE = "cne://"

# 板块过滤开关（与 core.data.panel 同口径）：
# - QUANT_EXCLUDE_BSE 默认开启（北交所 50 万 + 24 个月准入）
# - QUANT_EXCLUDE_KECHUANG / QUANT_EXCLUDE_CHINEXT 默认关闭

# 板块代码段前缀单一真源：主面板（本模块）与 style_state 插件共用，
# 两侧必须引用同一集合，避免任一侧增改前缀后静默漂移。
BSE_PREFIXES = ("92", "83", "87", "43")


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "0", "false", "no", "off"
    )


def _exclude_bse_enabled() -> bool:
    return _env_flag("QUANT_EXCLUDE_BSE", "1")


def bse_excluded_prefixes() -> tuple[str, ...]:
    """当前生效的北交所排除前缀（QUANT_EXCLUDE_BSE=0 时返回空，供插件同口径）。"""
    return BSE_PREFIXES if _exclude_bse_enabled() else ()


def _exclude_kechuang_enabled() -> bool:
    return _env_flag("QUANT_EXCLUDE_KECHUANG", "0")


def _exclude_chinext_enabled() -> bool:
    return _env_flag("QUANT_EXCLUDE_CHINEXT", "0")


def _board_filter_flag() -> str:
    """三板块开关状态摘要（用于缓存 key，开关切换后缓存自然失效）。"""
    return "|".join([
        "bse1" if _exclude_bse_enabled() else "bse0",
        "kc1" if _exclude_kechuang_enabled() else "kc0",
        "cyb1" if _exclude_chinext_enabled() else "cyb0",
    ])

# 磁盘面板缓存：构建好的 panel 落盘 parquet，避免每次评估都重算 30-70s。
# key = (start, include_fundamentals, 板块开关, schema_version, 数据面签名)，
# **不含请求 end**（2026-10-23：end 参与 key 曾导致数据日更后 key 逐日漂移、
# 每天整段重建 ~8.7GB）。
# 命中条件 = 请求区间被缓存面板实际覆盖范围包含（含 _CACHE_MAX_STALE_DAYS 容忍）。
# 目录占用由 _CACHE_MAX_GB（GB 配额）+ _CACHE_MAX_FILES（文件数兜底）双闸约束。
_CACHE_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "panel" / "cache"
# 落盘分块行数（2026-11-22）：换内存不换口径——分块只影响写入峰值，
# 不影响产物内容。40 万行 × ~230 列 ≈ 0.8GB/块，可通过环境变量下调。
_CACHE_WRITE_CHUNK_ROWS = int(os.environ.get("ALPHA_PANEL_WRITE_CHUNK_ROWS", "400000") or 400000)
_CNE_STATE_FILE = (
    Path(__file__).resolve().parents[4]
    / "CNEquity" / "data" / "quant_dataset" / "_cnequity" / "meta" / "state" / "daily_bars.json"
)
# 缓存格式/构建逻辑版本：代码变更影响 panel 内容时 +1 强制全部重建
# v5: arrow 数值列改为无 null 写入（恢复零拷贝 mmap 共享）+ 数据面按需裁剪列族
# v6: 股票 panel 排除北交所（92/83/87/43 前缀，投资者准入要求）
# v7: stock_daily_wide 单位归一化（amount 千元→元、vol 手→股、circ_mv/total_mv
#     万元→元），vwap 从 1/10 修正为正确元/股口径
# v8: 新数据面（行业 / 公告事件 / 研报视角 / 财报明细 / 指数成分）+ 宽表涨跌停/上市天数等列
# v10: 聚焦面板恒载行业列（industry_sw_l1/code/ind_asof_days，plugins/industry.py）
# v11: 聚焦面板恒载风格状态列（style_state，plugins/style_state.py，state_conditional_ic
#      的输入，规格 §3.4.1）；改列即 bump，避免命中缺列旧面板
_CACHE_SCHEMA_VERSION = 11
# 缓存文件数上限（全量面板 parquet+arrow ≈ 6GB，聚焦面板 ≈ 0.3-2.6GB）。
# 小磁盘服务器可用 ALPHA_PANEL_CACHE_MAX_FILES 调小（代价：切换数据面组合时重建）。
_CACHE_MAX_FILES = max(2, int(os.environ.get("ALPHA_PANEL_CACHE_MAX_FILES", "8") or 8))
# 缓存磁盘配额（GB，2026-10-23）：parquet+arrow 总量超过即从最旧开始成对淘汰。
# 背景：仅按文件数上限时，8 组全量缓存（parquet ~2GB + arrow ~6GB）理论可达
# ~70GB，磁盘长期被缓存挤压；改为按字节配额后文件数自然受磁盘约束。
# 调大代价：磁盘占用上升；调小代价：切换窗口/数据面组合时重建频率上升。
_CACHE_MAX_GB = max(1.0, float(os.environ.get("ALPHA_PANEL_CACHE_MAX_GB", "25") or 25))
# Arrow IPC（未压缩、可 mmap）副本写入开关（**2026-10-24 回归修复：默认重新改为开**）。
# 为何必须开：多进程场景（harness_mine / 后端 ComputeWorker / 多挖掘 run）下，
# WorkerPanelStore 走 panel_mmap 的 arrow 路径才能让**数值列跨进程共享同一份物理页**；
# 一旦没有 .arrow，每个 worker 各自 pd.read_parquet 私有解压一份全量 panel
# （实测 4.00GB/进程，6 路并发 ≈ 24GB）——这正是「跑的时候内存大」的真凶。
# 收益对比（同一份 211 列全量 panel）：
#   · 有 arrow：数值列共享 page cache（~6.4GB 只一份）+ 每进程仅 instrument/datetime
#     索引私有（~0.4GB）→ 6 进程 ≈ 2.4GB 私有；
#   · 无 arrow：每进程私有解压 ~4GB → 6 进程 ≈ 24GB 私有。
# 磁盘代价已由 _CACHE_MAX_GB（默认 25GB）配额兜住，不会无限膨胀。
# 单进程场景 arrow 会多驻留 ~6.4GB standby page cache（可回收，不占真正可用内存）。
# 确实需要纯 parquet（如磁盘吃紧）时设 ALPHA_PANEL_ARROW=0。
_ARROW_MMAP_WRITE = os.environ.get("ALPHA_PANEL_ARROW", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# 行主序矩阵副本（``.matrix.npy``，2026-11-14）。默认开。
# 与 arrow 的本质区别（实测，不是推测）：
#   · arrow 是**列式**存储，230 列是 230 段独立缓冲；pandas 无法把它们拼成单 block，
#     `read_panel_arrow_mmap` 逐列 `to_numpy()` 时**每列仍拷一份私有内存**
#     （29.7MB/列 = 7.78M 行 × 4B），230 列累计 6.83GB；
#   · matrix 是**行主序连续**矩阵，`pd.DataFrame(arr, copy=False)` 得到单 block，
#     数值列与 mmap 页共享，私有内存 ≈ 0（只剩 index 部分）。
# 实测（7.78M 行 × 230 列全量面板）：arrow 路径 private 7.79GB / 加载峰值 14.50GB；
# matrix 路径 private ~0.7GB / 加载峰值 ~2GB。
# 磁盘代价：matrix 与 arrow 同量级（~6.8GB/全量组）。若两者都开，磁盘会翻倍；
# 想省磁盘可在 matrix 生效后设 ALPHA_PANEL_ARROW=0（matrix 同样支持跨进程 mmap 共享）。
_MATRIX_WRITE = os.environ.get("ALPHA_PANEL_MATRIX", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
# 缓存「比请求旧几天仍可命中」的容忍天数（2026-10-20）。
# 背景：日更数据每天让 `end` 前移一天，而命中判定要求「缓存终点 >= 请求终点」，
# 于是**每个交易日都要重建整份面板**（实测 4.7GB / 463s），磁盘上 v9/v10/v11
# 共 34 个文件 43.7GB 就是这么攒出来的。
# 日频/周频/月频回测里最新 1~3 个自然日的缺失对 IC、换手、分组的影响远小于重建
# 代价；设 0 = 退回旧的严格相等行为。
# **方向安全性**：只容忍「缓存比请求旧」，绝不容忍「缓存比请求新」——后者会把请求
# 区间之后的未来数据带进调用方，破坏 tuning 模式的物理时间隔离（盲测段污染）。
_CACHE_MAX_STALE_DAYS = max(0, int(os.environ.get("ALPHA_PANEL_CACHE_STALE_DAYS", "3") or 3))
# ── 增量补尾/补头（2026-11-22）──
# 命中判定用的是缓存**实际覆盖区间**（_panel_coverage）：完全覆盖请求时直接命中并裁剪。
# 缓存比请求短（日更后多出几天）或起点更晚时，原实现整份重建（分钟级 / 峰值数 GB）；
# 改为只重建「接缝附近的回溯窗口」，其余行沿用缓存。
# warmup 必须覆盖插件最长回溯：regulatory_events._SEV_WINDOW_DAYS=360 **日历天**
# （≈250 交易日）是最长的一档，research_views 90 交易日次之。取 260 交易日留余量；
# 实测 130 交易日时 reg_severity 会与整份重建差 3（接缝缺历史）。可用
# ALPHA_PANEL_EXTEND_WARMUP 覆盖。
_CACHE_EXTEND_WARMUP_TPD = max(20, int(os.environ.get("ALPHA_PANEL_EXTEND_WARMUP", "260") or 260))
# 接缝替换窗口（交易日）：标签列是**前视**的（label_Kd = shift(-K)），缓存末尾若干行
# 在旧窗口下是 NaN，追加后必须重算。取 40 交易日覆盖最长持有期（20d）+ 余量。
_CACHE_EXTEND_SEAM_TPD = max(5, int(os.environ.get("ALPHA_PANEL_EXTEND_SEAM", "40") or 40))
# 开关：默认关闭（2026-11-22）。实测面板有 4 个列族对**窗口右端敏感**——
# inst_mv / inst_count / inst_ratio / reg_severity：同一区间起点、不同 end 的两次
# 整份重建，公共行数值本就不同。因此增量拼接在这些列上不可能与"同 end 的整份重建"
# 逐值一致。默认关，需要时显式 ALPHA_PANEL_INCREMENTAL=1 开启（只补尾/补头，接缝
# 替换 warm 天，其余沿用缓存），并自行确认下游不依赖上述列族的精确值。
_CACHE_INCREMENTAL = os.environ.get("ALPHA_PANEL_INCREMENTAL", "0").strip().lower() in (
    "1", "true", "yes", "on")
# 缺失区间占请求总跨度的比例超过该值时放弃增量（补的比重建还多，没有收益）。
_CACHE_EXTEND_MAX_MISSING_RATIO = float(
    os.environ.get("ALPHA_PANEL_EXTEND_MAX_MISSING", "0.6") or 0.6)
_CACHE_INDEX_COLS = ["datetime", "instrument"]
# 聚焦裁剪时**永远保留**的列（非面专属 + 评估/交付链路内部依赖）：
# - 行情核心列由 OUTPUT_COLUMNS 保证（build_panel_from_hq 的输出契约）；
# - engine_gate 回测长表契约要求 open/high/low/close/amount/volume/turnover_rate
#   （core/panel_schema.ALPHA_CNE_PANEL_SPEC.required_columns）——聚焦业绩面时
#   这些列虽不允许出现在因子里，但提交门禁仍要拿它们跑回测；
# - adj_close 用于 IC 衰减曲线现算 label（metrics/decay.panel_forward_label）；
# - float_cap 用于市值中性化；is_trade/not_st/adjfactor 为可交易性与复权口径。
# 行业列单真源（2026-10-12 P0；OCR low）：_ALWAYS_KEEP_COLUMNS 与
# _ALWAYS_WANTED_AUX_COLUMNS 都经它引用，避免两侧重复罗列漂移（一侧漏加 ⇒
# 已保留但从未加载 / 已加载但被裁剪，静默复现 2026-09-27「不可用字段」问题）。
_INDUSTRY_COLUMNS = frozenset({
    "industry_sw_l1",
    "industry_sw_code",
    "ind_asof_days",
})
# 风格状态列单真源：state_conditional_ic 只消费面板恒载 canonical style_state
# （plugins/style_state.py 供给）。同样不属任何数据面（_column_facets 为空），
# 聚焦裁剪/辅助插件过滤须恒载——否则 2026-09-27 行业列的「不可用字段」会以
# 「面板缺列 → metric 静默 missing」的形式复现（2026-10-14 整夜轮动 run 根因）。
_STYLE_COLUMNS = frozenset({"style_state"})
_ALWAYS_KEEP_COLUMNS = frozenset({
    "open", "high", "low", "close", "amount", "volume", "turnover_rate",
    "adj_close", "adjfactor", "float_cap", "tot_cap", "is_trade", "not_st",
    # 行业列（2026-10-12 起由 plugins/industry.py 供给，CNE curated
    # industry_members 自 2020-01 起有月度快照）：行业中性化
    # （CS_NEUTRALIZE(f, $industry_sw_l1)）是功能性中立能力，不属于任何数据面，
    # 聚焦裁剪必须保留——旧注释"行业列接入属 P2"已随插件落地失效。
}) | _INDUSTRY_COLUMNS | _STYLE_COLUMNS
# 公开别名：跨模块（如 eval/session 的 parquet 分支）需要**同一份**"系统必需列"
# 清单来兜底按需取列，避免两处各写一份后漂移。私有名保留以兼容既有引用，
# 两者指向同一 frozenset。
ALWAYS_KEEP_COLUMNS = _ALWAYS_KEEP_COLUMNS
# 聚焦 run 恒载的中立辅助列（与 _ALWAYS_KEEP_COLUMNS 同源，由 _needed_aux_columns
# 显式放行）。背景：聚焦裁剪按"列属面 ⊆ 允许范围"过滤辅助插件，行业列不属任何面
# （_column_facets 为空）会被漏掉，而提示词/评估/交付一直声明 $industry_sw_l1 可用
# ——这正是 2026-09-27 整夜 run 每 run 报 2~6 次「不可用字段」的根因。
_ALWAYS_WANTED_AUX_COLUMNS = frozenset(_INDUSTRY_COLUMNS) | _STYLE_COLUMNS
# 基本面开关打开时缓存面板必须包含的列族哨兵（每插件一列）。建缓存当天某个
# 辅助插件加载失败（如 CNE 同步占用 parquet 文件锁）会把缺列面板固化，之后
# 每次命中都返回残缺面板 → 下游 funda_*/dt_* 因子集体报"不可用字段"。
# 命中与写入两侧都校验：残缺面板不允许被缓存，也不允许被命中。
# 注意：数据面聚焦裁剪后本就只带部分列族，故哨兵校验按"本次请求的列族"收窄
# （见 _expected_sentinel_columns）。
_FUNDAMENTAL_SENTINEL_COLUMNS = frozenset(
    {
        "funda_total_assets",    # balancesheet
        "funda_net_profit",      # income
        "funda_ocf",             # cashflow
        "funda_netprofit_yoy",   # fina_indicator
        "funda_ebitda",          # fina_indicator（v9 新增列；缺列旧面板不得被命中）
        "holder_count_chg_pct",  # shareholder_counts
        "dt_net_buy_90d",        # event_faces
        "mgn_balance",           # margin
        "inst_count",            # institutional
        "th_top10_pct",          # top_holders
        "exp_net_profit",        # express
        "ds_days_since_actual",  # disclosure
        "div_cash_div",          # dividend
    }
)


def _facet_allowed(focus_facets: Sequence[str] | None) -> set[str] | None:
    """聚焦面 → 允许的列族面集合（含算子隐含输入面）；None = 不裁剪（全量）。"""
    focus = [str(f) for f in (focus_facets or ()) if f]
    if not focus:
        return None
    from alphaagent.factor.facets import facet_allowed_scope

    return facet_allowed_scope(focus)


def _column_facets(column: str) -> set[str]:
    """列名 → 所属数据面（纯函数，无 mining 依赖）。"""
    from alphaagent.factor.facets import expr_facets

    return expr_facets("$" + str(column))


def _facet_signature(focus_facets: Sequence[str] | None) -> str:
    """缓存签名：聚焦面集合的稳定短哈希（空 = 全量面板）。"""
    focus = sorted(str(f) for f in (focus_facets or ()) if f)
    if not focus:
        return "all"
    return "f" + hashlib.sha256("|".join(focus).encode("utf-8")).hexdigest()[:10]


def _needed_aux_columns(
    plugins: Sequence[DataSourcePlugin],
    allowed: set[str],
) -> set[str]:
    """本次请求需要的辅助插件列（列族面 ⊆ 允许范围）。

    用于跳过无关插件（省内存 + 省最重的 join_asof 展开耗时）。
    注意判定口径是「⊆ 允许范围」而非「∩ 聚焦面」：聚焦筹码面时价量列要作为
    CHIP_* 的输入保留（是否允许写进表达式由 facet 拦截层把关，不在数据层）。
    """
    wanted: set[str] = set()
    for plugin in plugins:
        if plugin.is_core():
            continue
        for col in plugin.panel_columns():
            if col in _ALWAYS_WANTED_AUX_COLUMNS:
                # 中立列（行业分组码/快照新鲜度）：聚焦时也恒载，供
                # CS_NEUTRALIZE 行业中性化与提交门禁使用，不受面过滤影响。
                wanted.add(str(col))
                continue
            facets = _column_facets(col)
            if facets and facets <= allowed:
                wanted.add(str(col))
    return wanted


def _prune_panel_columns(
    panel: pd.DataFrame,
    *,
    allowed: set[str],
) -> pd.DataFrame:
    """按聚焦范围裁剪 panel 列（只裁面专属辅助列，核心/内部依赖列永远保留）。

    - 无面归属的列（float_cap/tot_cap/标记/派生行情列）一律保留；
    - 面归属 ⊆ 允许范围（聚焦面 ∪ 算子隐含输入面）的列保留；
    - 其余（未选面的辅助列族）裁掉——这是内存大头：全量 124 列 3.9GiB 中
      约 101 列（3.2GiB）来自辅助插件。
    """
    keep: set[str] = set(_ALWAYS_KEEP_COLUMNS)
    for col in panel.columns:
        name = str(col)
        if name.startswith("label_"):
            keep.add(name)
            continue
        facets = _column_facets(name)
        if not facets or facets <= allowed:
            keep.add(name)
    return panel[[c for c in panel.columns if str(c) in keep]]


def _expected_sentinel_columns(
    *,
    include_fundamentals: bool,
    focus: Sequence[str] | None,
) -> frozenset[str]:
    """本次请求应当存在的哨兵列（聚焦裁剪后只校验本次加载的列族）。"""
    if not include_fundamentals:
        return frozenset()
    if not focus:
        return _FUNDAMENTAL_SENTINEL_COLUMNS
    allowed = _facet_allowed(focus) or set()
    out = set()
    for col in _FUNDAMENTAL_SENTINEL_COLUMNS:
        facets = _column_facets(col)
        if facets and facets <= allowed:
            out.add(col)
    return frozenset(out)


def _missing_funda_sentinels(
    panel: pd.DataFrame,
    expected: frozenset[str] | None = None,
) -> frozenset[str]:
    """面板缺失的哨兵列集合。

    ``expected`` 显式给出时（数据面聚焦：本次只加载部分列族）直接按集合差判断，
    **不再**要求面板里有 ``funda_`` 列——否则聚焦业绩面这种"没有 fundamental 插件
    列"的面板会被误判为未命中，缓存永远打不中（每次重建 60-75s）。
    ``expected=None`` 保持旧语义：面板无 funda_ 列（开关关）时返回空。
    """
    cols = {str(c) for c in panel.columns}
    if expected is not None:
        return frozenset(expected - cols)
    if not any(c.startswith("funda_") for c in cols):
        return frozenset()
    return frozenset(_FUNDAMENTAL_SENTINEL_COLUMNS - cols)


def _cne_watermark() -> str:
    """数据湖最后成功交易日（仅用于日志/可观测，不参与缓存 key）。"""
    try:
        if _CNE_STATE_FILE.is_file():
            state = json.loads(_CNE_STATE_FILE.read_text(encoding="utf-8"))
            wm = state.get("last_success_trade_date")
            if wm:
                return str(wm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE watermark 读取失败: %s", exc)
    return "unknown"


def _cache_path(
    start: str | None,
    include_fundamentals: bool,
    sig: str = "all",
) -> Path:
    """由参数 + schema 版本 + 数据面签名生成写入路径（同一组合稳定复用同一文件）。

    **key 不含请求 ``end``**（2026-10-23）：请求窗口右端只影响命中判定
    （``_find_cached_panel`` 按覆盖区间裁剪 + stale 容忍），不参与文件命名。
    数据湖日更使 ``resolve_test_end`` 逐日前移时，重建会**原子覆盖同一文件**
    而不是每天新增一份——此前 9 天积出 8 个全量变体（~40GB 磁盘、重建期
    内存峰值 ~17GB）的根因即 end 参与了 key。TEST_END_CAP 锁死右端后 end
    恒定，本改动进一步保证 cap 推进时同样只做一次性覆盖写。
    """
    mk = str(start or "all")
    # 板块过滤开关参与 key：开关切换后缓存自然失效，避免命中错误口径的面板
    board_flag = _board_filter_flag()
    key = hashlib.sha256(
        f"{mk}|{include_fundamentals}|{board_flag}|v{_CACHE_SCHEMA_VERSION}".encode("utf-8")
    ).hexdigest()[:16]
    # 文件名带 schema 版本前缀 + 数据面签名：_find_cached_panel 只认当前版本与
    # 同签名的缓存，版本/面组合变更后旧缓存自然失效（文件数由 _CACHE_MAX_FILES 淘汰）
    return _CACHE_ROOT / f"panel_v{_CACHE_SCHEMA_VERSION}_{key}_{sig}.parquet"


def _meta_path(cache_path: Path) -> Path:
    """缓存旁注元数据文件（记录签名/范围/列数，供命中判定与排查）。"""
    return cache_path.with_suffix(".meta.json")


def _read_cache_meta(cache_path: Path) -> dict[str, Any] | None:
    try:
        meta_path = _meta_path(cache_path)
        if not meta_path.is_file():
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def columns_for_facets(facets: Sequence[str]) -> tuple[str, ...]:
    """返回这些数据面覆盖的面板列（**只读缓存 schema，不物化任何数值**）。

    用途：调用方只知道"要研究哪些数据面"、不知道具体表达式时（Web run 的首个
    session、批量批评估），按面裁剪列传给 ``keep_columns``，避免全量 158 列常驻。
    实测（2026-10-20，同一份缓存）：全量 158 列加载 5.80GB / 求值峰值 11.01GB；
    19 列 1.37GB / 2.28GB —— **列数是唯一杠杆**，加载与求值临时量一起降。

    返回空 tuple 表示"无法判断"（缓存缺失 / 面名非法 / 无匹配列），调用方应回退
    全量，**不要**当成"零列"处理。

    系统必需列（掩码/复权/行业/规模）不在这里返回：它们不属于任何数据面，
    由 ``load_panel_from_cne`` 用 ``_ALWAYS_KEEP_COLUMNS`` 兜底并上。
    """
    focus = tuple(str(f).strip() for f in facets if str(f).strip())
    allowed = _facet_allowed(focus)
    if not allowed or not _CACHE_ROOT.is_dir():
        return ()

    # 取最近更新的一份缓存读 schema（列集合与签名/universe 基本无关；
    # 若这份恰好缺某列，keep_columns 的"不存在即忽略"语义会兜住）。
    best: tuple[float, Path] | None = None
    try:
        for p in _CACHE_ROOT.iterdir():
            if not p.is_file() or p.suffix not in (".arrow", ".parquet"):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[0]:
                best = (mtime, p)
    except OSError:
        return ()
    if best is None:
        return ()

    path = best[1]
    if path.suffix == ".arrow":
        from alphaagent.data.adapters.panel_mmap import arrow_column_names

        names = arrow_column_names(path)
    else:
        names = _parquet_column_names(path)
    if not names:
        return ()

    out: list[str] = []
    for col in names:
        col_facets = _column_facets(col)
        if col_facets and col_facets <= allowed:
            out.append(str(col))
    return tuple(sorted(out))


def _parquet_column_names(path: Path) -> list[str] | None:
    """只读 parquet schema 拿列名（不读数据）。"""
    try:
        import pyarrow.parquet as pq

        return [str(n) for n in pq.read_schema(path).names]
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 parquet schema 失败: %s", exc)
        return None


def _find_cached_panel(
    start: str | None,
    end: str | None,
    include_fundamentals: bool,
    focus_facets: Sequence[str] | None = None,
    keep_columns: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """遍历缓存目录，返回覆盖请求区间且**数据面签名一致**的缓存面板。

    优先走 arrow mmap 路径（数值列跨进程零拷贝共享物理页，多会话并发时
    内存不按会话数叠加）；arrow 缺失/读取失败回退 parquet 解压路径。

    签名一致性：聚焦 run 只认同聚焦面的缓存文件（列集合一致）。不再用"全量缓存
    服务聚焦请求"——那会让 mmap 把 4.4GB 全量 arrow 的所有列页都 fault 进来，
    在小内存机器上正是要避免的。

    ``keep_columns``：只物化这些列（None = 全部）。命中判定改用**schema 列名**
    （``arrow_column_names`` / ``_parquet_column_names``，不读数值），因此按需取列
    不会破坏基本面哨兵/开关一致性判断——这是此前"底层支持 columns、但缓存读取
    入口从不传它"的断点（2026-10-19）。
    """
    try:
        if not _CACHE_ROOT.is_dir():
            return None
        req_start = pd.Timestamp(start) if start else None
        req_end = pd.Timestamp(end) if end else None
        want_sig = _facet_signature(focus_facets)
        focus = sorted(str(f) for f in (focus_facets or ()) if f)
        expected_sentinels = _expected_sentinel_columns(
            include_fundamentals=include_fundamentals, focus=focus or None
        )
        for path in sorted(_CACHE_ROOT.glob(f"panel_v{_CACHE_SCHEMA_VERSION}_*.parquet")):
            meta = _read_cache_meta(path)
            if meta is None:
                continue
            if str(meta.get("signature") or "") != want_sig:
                continue
            if bool(meta.get("include_fundamentals")) != bool(include_fundamentals):
                continue
            # 板块过滤开关状态必须一致，否则命中错误口径的面板
            if str(meta.get("board_filter") or "") != _board_filter_flag():
                continue
            # ★ 覆盖区间**预筛**（2026-11-14）：必须在物化之前做。
            # 历史缺陷：旧逻辑要先把面板读进来、再算 _panel_coverage 判断是否覆盖请求
            # 区间——于是"扫到一份不匹配的缓存"会先付出完整物化代价再被丢弃。
            # 实测（4 组缓存并存时）：为一份 4.63M 行 × 230 列的不匹配 arrow 白付
            # 4.4GB 私有内存 + 6.7GB mmap 页，直接把加载峰值从 ~1GB 抬到 11.5GB。
            # meta.json 里已有 start/end（落盘时写入），足够做预筛；缺失则退回物化后判定。
            c_start, c_end = meta.get("start"), meta.get("end")
            if req_start is not None and c_start:
                try:
                    c_start_ts = pd.Timestamp(str(c_start))
                    if req_start < c_start_ts and not (
                        req_start.to_period("M") == c_start_ts.to_period("M")
                    ):
                        continue  # 缓存起点晚于请求起点（跨月）→ 不可能覆盖
                except Exception:  # noqa: BLE001 — 元数据异常时不预筛，走原判定
                    pass
            if req_end is not None and c_end:
                try:
                    c_end_ts = pd.Timestamp(str(c_end))
                    if req_end > c_end_ts:
                        if (pd.Timestamp(req_end) - c_end_ts).days > _CACHE_MAX_STALE_DAYS:
                            continue  # 缓存比请求旧太多 → 需要重建
                except Exception:  # noqa: BLE001
                    pass
            try:
                # 读取源优先级：matrix（行主序单 block，真零拷贝）
                #   > arrow（列式，每列私有拷贝）> parquet（解压）。
                from alphaagent.data.adapters.panel_mmap import (
                    arrow_column_names,
                    read_panel_arrow_mmap,
                )
                from alphaagent.data.adapters.panel_matrix import (
                    matrix_column_names,
                    read_panel_matrix,
                )

                arrow_path = path.with_suffix(".arrow")
                # ① 先只读元数据拿列名做命中判定（不物化任何数值列）。
                # matrix 的 json 比 arrow schema / parquet schema 都便宜，优先读它。
                names: list[str] | None = matrix_column_names(path)
                if names is None and arrow_path.is_file():
                    names = arrow_column_names(arrow_path)
                if names is None:
                    names = _parquet_column_names(path)
                if not names:
                    continue
                name_set = {str(n) for n in names}
                has_funda = any(n.startswith("funda_") for n in name_set)
                # 基本面开关一致性 + 哨兵检查（与旧逻辑等价，只是数据源从
                # DataFrame.columns 换成 schema 列名，从而允许后续按列物化）
                if include_fundamentals and expected_sentinels:
                    missing_sentinels = expected_sentinels - name_set
                    if missing_sentinels:
                        logger.warning(
                            "CNE panel 缓存 %s 缺哨兵列 %s，视为未命中（将重建）",
                            path.name, sorted(missing_sentinels),
                        )
                        continue
                elif not include_fundamentals and has_funda:
                    continue  # 基本面开关不匹配，跳过
                # ③ 请求列一致性：keep_columns 中有列不在缓存列集 → 视为未命中。
                # 背景（2026-10-23）：新接入的低频事件数据面插件（block_trades /
                # regulatory_events / share_structure）合入 main 后，旧缓存（生成于
                # 插件合入前）不含 bt_*/reg_*/ss_* 列；此前命中判定只查哨兵列，
                # pq_cols 会把请求列里缓存没有的列**静默滤掉**，导致评估层
                # 看到 ic=None / cov=None 假阴性（列根本没载入，不是数据稀疏）。
                # 缺列即 miss → 重建含新列的缓存；重建路径 keep_columns 不生效
                # （缓存必须全列），首次请求新面会触发一次全量重建，此后命中。
                if keep_columns:
                    missing_req = {
                        str(c) for c in keep_columns
                        if str(c) not in _CACHE_INDEX_COLS and str(c) not in name_set
                    }
                    if missing_req:
                        logger.info(
                            "CNE panel 缓存 %s 缺请求列 %s（前 8 个），视为未命中（将重建）",
                            path.name, sorted(missing_req)[:8],
                        )
                        continue
                # ② 命中后按 keep_columns 物化（None = 全列，行为与旧版逐字节一致）
                # matrix 优先：行主序 float32 连续矩阵 → pandas 单 block，数值列与
                # mmap 页共享、私有内存 ≈ 0（2026-11-14 实测：7.78M×230 面板
                # private 7.79GB → ~0.7GB，加载峰值 14.50GB → ~2GB）。
                # arrow 的"零拷贝共享"名不副实：逐列 to_numpy 每列仍拷 29.7MB。
                df = read_panel_matrix(path, columns=keep_columns)
                if df is not None and keep_columns is None:
                    # 全量加载路径必须以 **parquet 权威列集** 校验矩阵列集：
                    # matrix 的列名来自它自己的 json，若写入中断/失败会留下
                    # "新 parquet + 旧列集 matrix" 的混合状态；此时哨兵检查（只看
                    # 少数固定列）无法发现"少列"，会静默返回残缺面板。
                    pq_cols = _parquet_column_names(path)
                    if pq_cols:
                        stale = (
                            {str(c) for c in pq_cols}
                            - set(df.columns)
                            - set(_CACHE_INDEX_COLS)
                        )
                        if stale:
                            logger.warning(
                                "CNE panel matrix 列集落后于 parquet（缺 %s，前 8 个），"
                                "回退 arrow/parquet: %s",
                                sorted(stale)[:8], path.name,
                            )
                            df = None
                if df is None and arrow_path.is_file():
                    df = read_panel_arrow_mmap(arrow_path, columns=keep_columns)
                if df is None:
                    pq_cols = None
                    if keep_columns is not None:
                        pq_cols = [
                            str(c) for c in keep_columns
                            if str(c) in name_set
                        ] or None
                    df = pd.read_parquet(path, columns=pq_cols)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CNE panel 缓存读取失败，跳过 %s: %s", path.name, exc)
                continue
            if not isinstance(df.index, pd.MultiIndex):
                if set(_CACHE_INDEX_COLS).issubset(df.columns):
                    df = df.set_index(_CACHE_INDEX_COLS)
                else:
                    continue
            idx_min, idx_max = _panel_coverage(df)
            if idx_min is None or idx_max is None:
                continue
            # 起点兼容判断：缓存覆盖从「首个交易日」（如 2018-01-02）开始，
            # 而请求常用自然日期起点（如 2018-01-01）。若两者同属一个自然月，
            # 说明请求起点只是缓存首日前面的非交易日，视作命中而非重建。
            if req_start is not None and req_start < idx_min:
                if not (req_start.to_period("M") == pd.Timestamp(idx_min).to_period("M")):
                    continue  # 请求起点早于缓存覆盖起点（且跨月，非同日历月容差）
            if req_end is not None and req_end > idx_max:
                # 缓存比请求旧：容忍少量天数（日更数据让 end 每天前移，严格要求精确
                # 覆盖会让每个交易日都重建整份面板，见 _CACHE_MAX_STALE_DAYS 注释）。
                # 只容忍"旧"，不接受"新"——后者会把未来数据带进调用方。
                stale_days = (pd.Timestamp(req_end) - pd.Timestamp(idx_max)).days
                if stale_days > _CACHE_MAX_STALE_DAYS:
                    continue  # 超出容忍窗口，确实需要重建
                logger.info(
                    "CNE panel 缓存 %s 覆盖到 %s，比请求终点 %s 旧 %d 天（容忍上限 %d），按命中处理",
                    path.name, idx_max, req_end, stale_days, _CACHE_MAX_STALE_DAYS,
                )
            # （基本面开关 + 哨兵检查已上移到「物化之前」，基于 schema 列名判定——
            #  按需取列后 df.columns 只是子集，在这里再判会把缺列误当未命中。）
            # 命中覆盖超集缓存时按请求区间裁剪：请求 [a, b] 可能命中覆盖到更晚
            # 日期的缓存，若不裁剪会把区间外数据（如 2025+ 盲测段）带进调用方，
            # 破坏 tuning 模式的物理时间隔离。起点侧同时收紧到请求起点。
            if req_start is not None or req_end is not None:
                dts = df.index.get_level_values("datetime")
                keep = pd.Series(True, index=df.index)
                if req_start is not None:
                    keep &= dts >= req_start
                if req_end is not None:
                    keep &= dts <= req_end
                if not bool(keep.all()):
                    # 裁剪必须尽量走 iloc 行切片（视图，与 mmap 零拷贝兼容）。
                    # df.loc[布尔数组] 会**整块拷贝**数值块 —— matrix 缓存 7.95M 行
                    # × 230 列 = 6.75GB，一次裁剪就把 private 顶回 7.5GB、峰值 14GB
                    # （2026-11-21 实测：harness_mine 传了 test_end，触发裁剪后
                    # 零拷贝收益全部归零）。面板按 datetime 排序，裁剪通常是
                    # 「掐头去尾」的连续区间，用 iloc 即可；只有零散掩码才回退 loc。
                    keep_np = keep.to_numpy(dtype=bool)
                    lo = int(np.argmax(keep_np))
                    hi_ex = len(keep_np) - int(np.argmax(keep_np[::-1]))
                    if hi_ex > lo and keep_np[lo:hi_ex].all():
                        df = df.iloc[lo:hi_ex]
                    else:
                        df = df.loc[keep]
            logger.info(
                "CNE adapter: panel 命中磁盘缓存 %s（%d 行, %d 列, 签名 %s, "
                "已按请求窗口 [%s, %s] 裁剪）",
                path.name, len(df), df.shape[1], want_sig, start, end,
            )
            return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存扫描异常（忽略，将重建）: %s", exc)
    return None


def _panel_coverage(panel: pd.DataFrame) -> tuple[Any, Any]:
    """面板实际覆盖的 [min_date, max_date]（datetime 索引层）。"""
    dts = panel.index.get_level_values("datetime")
    if len(dts) == 0:
        return None, None
    return dts.min(), dts.max()


def _cache_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """只读 datetime 列取缓存覆盖区间（不物化全列，成本几十 MB 量级）。"""
    try:
        s = pd.read_parquet(path, columns=["datetime"])["datetime"]
    except Exception:  # noqa: BLE001
        return None
    if s.empty:
        return None
    return pd.Timestamp(s.min()), pd.Timestamp(s.max())


def _find_extendable_cache(
    start: str | None,
    end: str | None,
    include_fundamentals: bool,
    focus_facets: Sequence[str] | None = None,
) -> tuple[Path, pd.Timestamp, pd.Timestamp, bool, bool] | None:
    """找一份签名一致、但只差头/尾的缓存。

    返回 ``(path, cov_min, cov_max, need_head, need_tail)`` 或 None。
    聚焦（focus_facets 非空）请求不做增量：那类缓存列集合经过裁剪，增量后
    拼接口径容易漂移，保持整份重建。
    """
    if focus_facets:
        return None
    try:
        if not _CACHE_ROOT.is_dir():
            return None
        want_sig = _facet_signature(focus_facets)
        req_start = pd.Timestamp(start) if start else None
        req_end = pd.Timestamp(end) if end else None
        tol = pd.Timedelta(days=_CACHE_MAX_STALE_DAYS)
        for path in sorted(_CACHE_ROOT.glob(f"panel_v{_CACHE_SCHEMA_VERSION}_*.parquet")):
            meta = _read_cache_meta(path)
            if meta is None:
                continue
            if str(meta.get("signature") or "") != want_sig:
                continue
            if bool(meta.get("include_fundamentals")) != bool(include_fundamentals):
                continue
            if str(meta.get("board_filter") or "") != _board_filter_flag():
                continue
            cov = _cache_date_range(path)
            if cov is None:
                continue
            cmin, cmax = cov
            need_head = bool(req_start is not None and req_start < (cmin - tol))
            need_tail = bool(req_end is not None and req_end > (cmax + tol))
            if not (need_head or need_tail):
                continue
            return path, cmin, cmax, need_head, need_tail
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 增量缓存探测异常（回退整份重建）: %s", exc)
        return None


def _extend_cached_panel(
    *,
    path: Path,
    cov_min: pd.Timestamp,
    cov_max: pd.Timestamp,
    need_head: bool,
    need_tail: bool,
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    registry,
    include_fundamentals: bool,
    asset_type: str = "stock",
) -> pd.DataFrame | None:
    """在已有缓存上补尾或补头，返回拼接后的面板；不可增量时返回 None。

    正确性要点：插件列含回溯窗口（research_views 90 交易日、style_state/事件类 20），
    所以补的那段必须**多建 warmup 天**，并把缓存里**接缝处 warmup 天**的行一并替换掉，
    否则接缝行会因为缺前置历史而与整份重建不一致。两头都缺时直接返回 None 走整份重建。
    """
    if need_head and need_tail:
        return None
    try:
        cached = pd.read_parquet(path)
        if set(_CACHE_INDEX_COLS).issubset(cached.columns):
            cached = cached.set_index(_CACHE_INDEX_COLS)
        cached = cached.sort_index()
        if cached.empty:
            return None
        dates = pd.DatetimeIndex(cached.index.get_level_values("datetime").unique()).sort_values()
        if len(dates) < 3:
            return None
        warm = _CACHE_EXTEND_WARMUP_TPD

        # 缺失占比过大则没有增量收益，交回整份重建
        total_days = max((req_end - req_start).days, 1)
        miss_days = 0
        if need_head:
            miss_days += max((cov_min - req_start).days, 0)
        if need_tail:
            miss_days += max((req_end - cov_max).days, 0)
        if miss_days / total_days > _CACHE_EXTEND_MAX_MISSING_RATIO:
            logger.info("CNE panel 增量跳过：缺失 %d/%d 天，超过阈值 %.2f，改整份重建",
                        miss_days, total_days, _CACHE_EXTEND_MAX_MISSING_RATIO)
            return None

        def _build(a: str, b: str) -> pd.DataFrame:
            seg = registry.build_panel(
                start=a, end=b, universe_mask=False,
                include_fundamentals=include_fundamentals,
                asset_type=asset_type, include_columns=None,
            )
            _enrich_trade_flags(seg)
            if asset_type == "stock" and len(seg):
                inst = seg.index.get_level_values("instrument").astype(str)
                excluded: list[str] = []
                if _exclude_bse_enabled():
                    excluded += list(BSE_PREFIXES)
                if _exclude_kechuang_enabled():
                    excluded += ["688", "689"]
                if _exclude_chinext_enabled():
                    excluded += ["300", "301"]
                if excluded:
                    mask = inst.str.startswith(tuple(excluded))
                    if mask.any():
                        seg = seg.loc[~mask]
            return seg

        if need_tail:
            # 追加 + 替换接缝：回溯列只需前置历史（warm），前视列（label_* = shift(-K)）
            # 需要重算缓存末尾若干行（seam）。两者相加即为本次构建跨度。
            seam = _CACHE_EXTEND_SEAM_TPD
            cut = dates[-seam] if len(dates) > seam else dates[0]
            bstart = dates[-(warm + seam)] if len(dates) > warm + seam else dates[0]
            logger.info("CNE panel 增量补尾：构建 [%s ~ %s]，替换缓存 %s 之后的行",
                        bstart.date(), req_end.date(), cut.date())
            seg = _build(bstart.date().isoformat(), req_end.date().isoformat())
            if set(seg.columns) != set(cached.columns):
                logger.warning("CNE panel 增量补尾列集合不一致（缓存 %d 列 / 新段 %d 列），放弃增量",
                               cached.shape[1], seg.shape[1])
                return None
            seg = seg[seg.index.get_level_values("datetime") >= cut]
            base = cached[cached.index.get_level_values("datetime") < cut]
            parts = [base, seg]
        else:
            cut = dates[warm] if len(dates) > warm else dates[-1]
            logger.info("CNE panel 增量补头：构建 [%s ~ %s]，替换缓存 %s 之前的行",
                        req_start.date(), cut.date(), cut.date())
            seg = _build(req_start.date().isoformat(), cut.date().isoformat())
            if set(seg.columns) != set(cached.columns):
                logger.warning("CNE panel 增量补头列集合不一致（缓存 %d 列 / 新段 %d 列），放弃增量",
                               cached.shape[1], seg.shape[1])
                return None
            seg = seg[seg.index.get_level_values("datetime") < cut]
            base = cached[cached.index.get_level_values("datetime") >= cut]
            parts = [seg, base]

        merged = pd.concat(parts, axis=0, sort=False)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        missing_cols = [c for c in cached.columns if c not in merged.columns]
        if missing_cols:
            logger.warning("CNE panel 增量结果缺列 %s，放弃增量改整份重建", missing_cols[:5])
            return None
        merged = merged[cached.columns]
        for col in merged.columns:  # 尽力维持既有 dtype，避免无谓的 int8→int64 膨胀
            try:
                merged[col] = merged[col].astype(cached[col].dtype)
            except Exception:  # noqa: BLE001
                pass

        meta = dict(_read_cache_meta(path) or {})
        meta.update({
            "start": req_start.date().isoformat(),
            "end": req_end.date().isoformat(),
            "rows": int(len(merged)),
            "columns": int(merged.shape[1]),
        })
        _save_cached_panel(path, merged, meta=meta)
        logger.info("CNE panel 增量完成：%d 行（缓存原 %d 行），已原子覆盖 %s",
                    len(merged), len(cached), path.name)
        return merged
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 增量补数失败（回退整份重建）: %s", exc)
        return None


def _purge_old_cache(keep: Path) -> None:
    """缓存目录超过配额/上限时删除最旧文件（保留刚写入的 keep）。

    双闸（2026-10-23）：
    - **磁盘配额**：parquet+arrow 总量超过 ``_CACHE_MAX_GB`` 时从最旧开始淘汰——
      仅按文件数上限时 8 组全量缓存（parquet ~2GB + arrow ~6GB/组）理论可达
      ~70GB，磁盘长期被缓存挤压；
    - **文件数**：``_CACHE_MAX_FILES`` 兜底（防碎片文件堆积）。

    parquet 与同 key 的 .arrow、.meta.json 成对淘汰；.arrow 若正被其他进程
    mmap（Windows unlink 会失败），单独跳过并警告——不影响主缓存淘汰。
    淘汰时**同步删除 .meta.json**：此前漏删会留孤儿 meta（parquet 已不在
    而.meta 残留），误导命中排查；本分支之前的孤儿靠人工清理。
    """
    try:
        if not _CACHE_ROOT.is_dir():
            return
        # 排序键容错：文件可能在 glob 与 stat 之间被其他进程删除（多会话并发淘汰），
        # 单个文件 stat 失败只跳过该文件，不让整个淘汰流程静默退出。
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return float("inf")

        def _remove_victim(victim: Path) -> None:
            """成对删除 parquet + .arrow + .meta.json（单个失败不阻断其余）。"""
            try:
                size_mb = victim.stat().st_size / 1e6
            except OSError:
                size_mb = 0.0
            try:
                victim.unlink()
                logger.info("CNE panel 缓存淘汰旧文件: %s (%.0fMB)",
                            victim.name, size_mb)
            except OSError as exc:
                logger.warning("CNE panel 缓存淘汰失败: %s (%s)", victim.name, exc)
            for suffix in (".arrow", ".meta.json"):
                side = victim.with_suffix(suffix)
                if side.is_file():
                    try:
                        side.unlink()
                    except OSError as exc:
                        logger.warning(
                            "CNE panel 缓存附属文件淘汰失败（可能被其他进程 mmap 占用）: %s (%s)",
                            side.name, exc,
                        )
            # matrix 三件套（.matrix.npy / .matrix.json / .matrix.idx.npz）：
            # 复合后缀无法用 with_suffix 拼，交给 remove_panel_matrix 逐个删
            try:
                from alphaagent.data.adapters.panel_matrix import remove_panel_matrix

                remove_panel_matrix(victim)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CNE panel matrix 附属文件淘汰异常: %s", exc)

        def _dir_bytes() -> int:
            """缓存目录中 parquet+arrow+matrix 的总字节数（排除刚写入的 keep 及其伴侣）。"""
            total = 0
            # keep 的伴侣文件（.arrow / .matrix.npy / .matrix.idx.npz / .meta.json）
            # 必须一并排除：否则"刚写入的一组全量缓存"自身就有
            # parquet 2GB + arrow 6.4GB + matrix 6.8GB ≈ 15GB，在 25GB 默认配额下
            # 写入第二组时就会超配额，反过来把**前一组仍有效**的缓存淘汰掉——
            # 表现为切换数据面/窗口时反复全量重建（每次 4.7GB / 463s），
            # 恰好抵消掉 matrix 想省下的加载开销。
            keep_prefix = keep.stem + "."
            for p in _CACHE_ROOT.iterdir():
                if not p.is_file() or p == keep:
                    continue
                if p.name.startswith(keep_prefix):
                    continue
                if p.suffix in (".parquet", ".arrow") or ".matrix." in p.name:
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            return total

        files = sorted(
            (p for p in _CACHE_ROOT.glob("panel_*.parquet") if p != keep),
            key=_mtime,
        )
        # 闸 1：磁盘配额（parquet+arrow 字节总量，含 keep 自身以外的全部缓存）
        while files and _dir_bytes() > _CACHE_MAX_GB * (1 << 30):
            _remove_victim(files.pop(0))
        # 闸 2：文件数兜底（+1 为刚写入的文件）
        while files and len(files) + 1 > _CACHE_MAX_FILES:
            _remove_victim(files.pop(0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存淘汰异常（忽略）: %s", exc)


def _write_panel_parquet_chunked(panel: pd.DataFrame, tmp: str,
                                 chunk_rows: int | None = None) -> None:
    """分块把面板写成 parquet（MultiIndex 先落平表列），避免整份复制。

    背景（2026-11-22 内存事故）：原实现先 ``flat = panel.reset_index()`` 再
    ``flat.to_parquet()``——那是一次**全量复制**（371 万行 × 230 列 ≈ 7GB），
    写盘时 pandas→Arrow 又生成一份同尺寸临时表，重建峰值因此到 16GB+。
    分块写只保留「面板本体 + 一个分块（默认 40 万行 ≈ 0.8GB）」。
    产物 schema 与整份写完全一致（列 = index 两列 + 数据列，无 pandas 索引元数据）。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = int(chunk_rows or _CACHE_WRITE_CHUNK_ROWS)
    writer = None
    schema = None
    try:
        for start in range(0, len(panel), max(rows, 1)):
            block = panel.iloc[start:start + rows].reset_index()
            table = pa.Table.from_pandas(block, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="snappy")
            elif table.schema != schema:
                # 同一面板的不同分块可能推断出不同 schema：某列在块 A 全为 NaN
                # 会被推断成 null 类型，在块 B 有值则是 double/string。pyarrow 的
                # ParquetWriter 拒绝 schema 不同的 write_table（2026-11-25 分批
                # 重建 6 年缓存时在此炸掉，前 34 个分片白建）。这里按首块 schema
                # 对齐：缺列补 null、类型不同则 cast。
                logger.debug("分块写 parquet：第 %d 块 schema 与首块不一致，已对齐", start)
                table = _align_table_to_schema(table, schema)
            writer.write_table(table)
            del table, block
    finally:
        if writer is not None:
            writer.close()


def _align_table_to_schema(table, schema):
    """把 table 对齐到给定 schema：缺列补 null、多余列丢弃、类型不同则 cast。"""
    import pyarrow as pa

    names = list(schema.names)
    for field in schema:
        if field.name not in table.schema.names:
            table = table.append_column(field.name, pa.nulls(table.num_rows, type=field.type))
    table = table.select(names)
    cols = []
    for field in schema:
        col = table.column(field.name)
        if col.type != field.type:
            col = col.cast(field.type)
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=schema)


def _save_cached_panel(
    path: Path,
    panel: pd.DataFrame,
    *,
    meta: dict[str, Any] | None = None,
    expect_sentinels: frozenset[str] | None = None,
) -> None:
    """原子写缓存（MultiIndex 先落平表列），写入后按上限淘汰旧文件。

    同时顺带写一份未压缩 Arrow IPC（同 key .arrow）：后续会话走
    panel_mmap.read_panel_arrow_mmap 零拷贝共享物理页，多会话并发时
    panel 物理内存不按会话数叠加。arrow 写失败不影响 parquet 主缓存。

    旁注 ``.meta.json`` 记录数据面签名/范围/行列表数——命中判定读它（不做
    "全量缓存服务聚焦请求"，避免 mmap fault 全量列页）。
    """
    try:
        # 辅助插件加载失败（即使哨兵集合为空，聚焦裁剪场景）也拒绝落盘：
        # 否则缺列面板被固化，后续请求持续命中残缺缓存。
        failed_aux = panel.attrs.get("failed_aux_plugins") or []
        if failed_aux:
            logger.warning(
                "CNE panel 构建时辅助插件加载失败 %s，本次不写缓存"
                "（下次请求将重建；请检查对应插件加载日志）",
                sorted(failed_aux),
            )
            return
        missing_sentinels = _missing_funda_sentinels(panel, expect_sentinels)
        if missing_sentinels:
            # 残缺面板不落盘：否则会被后续请求持续命中，缺列问题被固化数周。
            logger.warning(
                "CNE panel 构建结果缺基本面列族哨兵 %s，本次不写缓存"
                "（下次请求将重建；请检查当日对应插件加载日志中的 skip 告警）",
                sorted(missing_sentinels),
            )
            return
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(_CACHE_ROOT), prefix=".panel.", suffix=".parquet.tmp")
        os.close(fd)
        try:
            _write_panel_parquet_chunked(panel, tmp)
            os.replace(tmp, path)
        except BaseException:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # 顺带生成 arrow mmap 副本：默认关（_ARROW_MMAP_WRITE，2026-10-23——
        # 未压缩 arrow 体积 ≈ parquet 的 3 倍，单进程 mmap 命中反而把全量页
        # fault 进 page cache；收益只在多会话并发时兑现，需要时 ALPHA_PANEL_ARROW=1）。
        # 失败不影响主缓存。
        if _ARROW_MMAP_WRITE:
            try:
                from alphaagent.data.adapters.panel_mmap import write_panel_arrow_chunked

                write_panel_arrow_chunked(path.with_suffix(".arrow"), panel)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CNE panel arrow 副本生成异常（忽略）: %s", exc)
        # 行主序矩阵副本（2026-11-14）：单 block → pandas 真零拷贝，
        # 单进程私有内存 7.79GB → ~0.7GB（见模块顶部 _MATRIX_WRITE 注释）。
        # 失败不影响 parquet 主缓存，下次命中自动回退 arrow / parquet。
        if _MATRIX_WRITE:
            try:
                from alphaagent.data.adapters.panel_matrix import write_panel_matrix

                write_panel_matrix(path, panel)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CNE panel matrix 副本生成异常（忽略）: %s", exc)
        # 旁注元数据（命中判定依据）；写失败时该缓存不会被命中（保守）
        try:
            payload = dict(meta or {})
            payload.setdefault("schema", _CACHE_SCHEMA_VERSION)
            payload.setdefault("rows", int(len(panel)))
            payload.setdefault("columns", int(panel.shape[1]))
            _meta_path(path).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CNE panel 缓存元数据写入失败（该缓存不会被命中）: %s", exc)
        # 写入成功后清理超出上限的旧文件
        _purge_old_cache(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存写入失败（不影响使用）: %s", exc)


def _warn_panel_rebuild_cost(
    *,
    start: str | None,
    include_fundamentals: bool,
    focus: list[str],
) -> None:
    """缓存未命中、即将全量重建前的内存成本护栏（B-01，2026-10-26）。

    纯日志零行为：不阻塞自动化路径（CLI/无人值守重建照常进行），只把
    「这次重建预计吃多少内存」从静默变为可感知。规模参考取缓存目录中
    行×列乘积最大的已知缓存 meta（rows×cols×8B float64 估算，未含索引
    与 object 列开销，属下限估计）。
    """
    try:
        est_rows = est_cols = 0
        if _CACHE_ROOT.is_dir():
            for path in _CACHE_ROOT.glob(f"panel_v{_CACHE_SCHEMA_VERSION}_*.parquet"):
                meta = _read_cache_meta(path)
                if not meta:
                    continue
                rows = int(meta.get("rows") or 0)
                cols = int(meta.get("columns") or 0)
                if rows * cols > est_rows * est_cols:
                    est_rows, est_cols = rows, cols
        if est_rows > 0 and est_cols > 0:
            est_gb = est_rows * est_cols * 8 / 2**30
            logger.warning(
                "B-01: CNE panel 缓存未命中 → 全量重建（start=%s include_fundamentals=%s "
                "focus=%s）。参考现有缓存规模 %d 行 × %d 列，内存峰值预计 ≥ %.1f GB"
                "（float64 估算下限）。如无需 funda_* 列请改传 include_fundamentals=False；"
                "已知表达式列请传 keep_columns 走按需物化路径。",
                start,
                include_fundamentals,
                focus or "全量",
                est_rows,
                est_cols,
                est_gb,
            )
        else:
            logger.warning(
                "B-01: CNE panel 缓存未命中 → 首次全量重建（start=%s "
                "include_fundamentals=%s focus=%s）。无历史规模可参考，全量面板"
                "内存峰值可能达数 GB。",
                start,
                include_fundamentals,
                focus or "全量",
            )
    except Exception as exc:  # noqa: BLE001 — 护栏绝不阻塞主路径
        logger.debug("重建成本预估失败（忽略）: %s", exc)


def load_panel_from_cne(
    *,
    start: str | None = None,
    end: str | None = None,
    universe_mask: bool = False,
    include_fundamentals: bool = True,
    asset_type: str = "stock",
    focus_facets: Sequence[str] | None = None,
    keep_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """从 CNE 数据湖实时构建 AlphaAgent panel（插件化）。

    Parameters
    ----------
    start, end : str | None
        日期范围（闭区间），传给各插件做日期过滤。
    universe_mask : bool
        是否启用 ST/停牌过滤。默认 False。
    include_fundamentals : bool
        是否加载 fundamental（PIT 基本面）插件。False 时跳过，
        避免最重的 join_asof 展开耗时（默认 True 保持向后兼容）。
    asset_type : str
        资产类型：'stock'（默认）/ 'etf'。
        - stock：加载全部插件（stock_daily_wide 为核心行情 + 辅助插件）。
        - etf：只加载 etf_bars 插件（跳过股票行情与基本面辅助插件）。
        ETF 无基本面/市值/估值列，评估 profile 会跳过市值类指标。
    focus_facets : Sequence[str] | None
        数据面聚焦（用户勾选）。非空时**按面裁剪列族**：只加载相关插件、
        只保留聚焦面列 + 核心行情/标签列，全量面板 3.9GiB → 聚焦面板
        0.4~1.2GiB（实测 8.2M×124 列面板）。None/空 = 不裁剪（全量，向后兼容）。

    Returns
    -------
    pd.DataFrame
        AlphaAgent panel（float32, MultiIndex(datetime, instrument)）。
        列数取决于已注册插件提供的列（聚焦时按面裁剪）。

    首次构建后落盘缓存：key 按 (start, include_fundamentals, 板块开关, 数据面签名,
    schema 版本) 生成，**不含 end**（见 _cache_path——end 参与 key 曾导致数据日更
    后 key 逐日漂移、每天整段重建 ~8.7GB）。命中条件为「数据面签名一致 + 请求
    区间落在缓存覆盖范围内（含 stale 容忍）」，窗口右端前移时覆盖命中或**原地
    重建同一文件**；目录占用由 _CACHE_MAX_GB + _CACHE_MAX_FILES 双闸约束，
    不会无限累积。
    """
    focus = [str(f) for f in (focus_facets or ()) if f]
    sig = _facet_signature(focus)

    # 归一化日期参数为 YYYY-MM-DD 字符串：调用方（如 stacking 脚本）可能传
    # pd.Timestamp，核心行情插件能容忍，但 fundamental（date.fromisoformat）
    # 与 fund_flow（CNE reader 的日期比较）会静默失败 → 辅助插件整列丢失
    start = pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else None
    end = pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None

    # key 不含 end（见 _cache_path）；放在归一化之后，避免调用方传 Timestamp 时
    # str() 形式（含 00:00:00）与归一化后的 key 漂移成两个文件
    cache_path = _cache_path(start, include_fundamentals, sig)

    if not universe_mask and asset_type == "stock":
        # 按需取列：调用方给的表达式列 + 系统必需列。
        # 必需列不能省——掩码（is_trade/not_st）供 _enrich_trade_flags、
        # 复权（adjfactor）供派生、行业/风格列供中性化、index 列供 set_index、
        # 基本面哨兵供缓存命中判定（缺一个就会把好缓存误判成未命中而触发重建）。
        keep_eff: tuple[str, ...] | None = None
        if keep_columns:
            need = {str(c) for c in keep_columns}
            need |= set(_ALWAYS_KEEP_COLUMNS)
            need |= set(_CACHE_INDEX_COLS)
            if include_fundamentals:
                need |= set(_FUNDAMENTAL_SENTINEL_COLUMNS)
            keep_eff = tuple(sorted(need))
        cached = _find_cached_panel(start, end, include_fundamentals, focus, keep_eff)
        if cached is not None and not cached.empty:
            # 旧缓存面板带预建的全 NaN is_trade/not_st 列（历史 bug 产物）：命中时
            # 就地补齐，无需 +schema 版本强制重建数 GB 缓存（只改内存副本）。
            _enrich_trade_flags(cached)
            return cached

        # 增量补尾/补头（2026-11-22）：缓存与请求只差一段（日更多出几天、或起点更早）时，
        # 只重建接缝回溯窗口并原子覆盖同一缓存，而不是整份重建。
        # 仅服务全量口径（focus 为空）、非 universe_mask 的股票面板——与整份重建同分支。
        if (_CACHE_INCREMENTAL and asset_type == "stock" and not universe_mask
                and not focus and (start or end)):
            ext = _find_extendable_cache(start, end, include_fundamentals, focus)
            if ext is not None:
                _path, _cmin, _cmax, _need_head, _need_tail = ext
                _req_start = pd.Timestamp(start) if start else pd.Timestamp(_cmin)
                _req_end = pd.Timestamp(end) if end else pd.Timestamp(_cmax)
                merged = _extend_cached_panel(
                    path=_path, cov_min=_cmin, cov_max=_cmax,
                    need_head=_need_head, need_tail=_need_tail,
                    req_start=_req_start, req_end=_req_end,
                    registry=get_registry(), include_fundamentals=include_fundamentals,
                    asset_type=asset_type,
                )
                if merged is not None and not merged.empty:
                    dts = merged.index.get_level_values("datetime")
                    if _req_start is not None or _req_end is not None:
                        keep = pd.Series(True, index=merged.index)
                        if _req_start is not None:
                            keep &= dts >= _req_start
                        if _req_end is not None:
                            keep &= dts <= _req_end
                        merged = merged[keep]
                    _enrich_trade_flags(merged)
                    return merged

    # B-01 护栏（2026-10-26）：重建是全量内存动作（历史实测 4.7GB+ / 463s+），
    # 静默发生时用户只看到「电脑卡死」。此处按历史缓存规模打内存预估 WARNING，
    # 不阻塞自动化路径。仅 stock 全量口径打点——ETF 面板极小，无此内存风险。
    if asset_type == "stock" and not universe_mask:
        _warn_panel_rebuild_cost(start=start, include_fundamentals=include_fundamentals, focus=focus)
    registry = get_registry()
    allowed = _facet_allowed(focus)
    wanted_aux: set[str] | None = None
    if allowed is not None:
        wanted_aux = _needed_aux_columns(registry.list_plugins(), allowed)
        logger.info(
            "CNE adapter: 数据面聚焦 %s → 需要的辅助列 %d 个（跳过无关插件）",
            focus, len(wanted_aux),
        )
    logger.info("CNE adapter: building panel from %d plugins (start=%s end=%s include_fundamentals=%s asset_type=%s sig=%s)",
                len(registry.list_plugins()), start, end, include_fundamentals, asset_type, sig)

    # 确保 core 插件加载后补充 is_trade / not_st 标记列
    if keep_columns:
        # 重建路径**不**按需取列：缓存未命中时 panel 由插件逐列构建、随后整份写入
        # 缓存（缓存必须是全列，否则下次别的请求会命中到残缺缓存）。因此首次构建
        # 仍会占用全量内存，收起内存只发生在"命中缓存"这条主路径上。
        logger.info(
            "CNE adapter: 缓存未命中 → 全量构建（keep_columns 在重建路径不生效，"
            "构建完成后写全列缓存；如需控峰值内存请先确保缓存已存在）"
        )
    panel = registry.build_panel(
        start=start,
        end=end,
        universe_mask=universe_mask,
        include_fundamentals=include_fundamentals,
        asset_type=asset_type,
        include_columns=wanted_aux,
    )

    # 补充 core 插件特有的衍生标记列（is_trade, not_st）
    _enrich_trade_flags(panel)

    # 股票 panel 按开关排除板块（北交所/科创板/创业板，投资者准入要求）。
    # 开关：QUANT_EXCLUDE_BSE 默认开启，QUANT_EXCLUDE_KECHUANG /
    # QUANT_EXCLUDE_CHINEXT 默认关闭。
    if asset_type == "stock" and len(panel) > 0:
        inst = panel.index.get_level_values("instrument").astype(str)
        excluded: list[str] = []
        if _exclude_bse_enabled():
            excluded += list(BSE_PREFIXES)
        if _exclude_kechuang_enabled():
            excluded += ["688", "689"]
        if _exclude_chinext_enabled():
            excluded += ["300", "301"]
        if excluded:
            mask = inst.str.startswith(tuple(excluded))
            if mask.any():
                logger.info("CNE adapter: 排除板块 %d 行", int(mask.sum()))
                panel = panel.loc[~mask]

    # 数据面裁剪：只保留聚焦范围列 + 核心行情/标签/交付依赖列（全量请求不裁剪）
    if allowed is not None:
        before = panel.shape[1]
        panel = _prune_panel_columns(panel, allowed=allowed)
        logger.info("CNE adapter: 数据面裁剪列 %d → %d", before, panel.shape[1])

    # 落盘缓存（universe_mask=True 的过滤结果不与通用缓存混用）
    if not universe_mask and asset_type == "stock":
        _save_cached_panel(
            cache_path,
            panel,
            meta={
                "schema": _CACHE_SCHEMA_VERSION,
                "signature": sig,
                "focus_facets": focus,
                "include_fundamentals": bool(include_fundamentals),
                "board_filter": _board_filter_flag(),
                "start": start,
                "end": end,
                "columns": int(panel.shape[1]),
                "rows": int(len(panel)),
            },
            expect_sentinels=_expected_sentinel_columns(
                include_fundamentals=include_fundamentals, focus=focus or None
            ),
        )

    logger.info("CNE adapter: panel shape=%s, columns=%d", panel.shape, panel.shape[1])
    return panel


def _enrich_trade_flags(panel: pd.DataFrame) -> None:
    """就地补充 is_trade / is_st / not_st 列。

    - ``is_trade`` = volume > 0（停牌/无成交为 0）；
    - ``is_st`` / ``not_st`` = 该 (trade_date, symbol) 是否在 CNE curated
      ``stock_st``（风险警示板唯一真源，已与 baostock 逐日 isST 对账）成员表内；
      数据集不可用时回落 ``1 - panel["is_st"]``（fail-open）。

    **判定必须是「缺列或整列 NaN」而非「缺列」**：panel 由 ``_finalize_panel`` 按
    OUTPUT_COLUMNS 预建了 is_trade/not_st 两列（全 NaN），旧实现只判缺列 → 恒跳过
    计算，两列永远是 NaN，而挖掘系统提示词的字段表把 ``$is_trade``/``$not_st``
    列为可用字段（等于教 LLM 用一个坏字段）。
    """
    if "volume" in panel.columns and _column_all_nan(panel, "is_trade"):
        panel["is_trade"] = (panel["volume"].fillna(0) > 0).astype("int8")

    st_flags = _st_flags(panel)
    if st_flags is not None:
        # ST 真源可用：is_st/not_st 无条件按 stock_st 重写（同一真源，避免两列互相矛盾）
        panel["is_st"] = st_flags
        panel["not_st"] = (1 - st_flags).astype("int8")
    elif _column_all_nan(panel, "not_st"):
        # 数据集不可用：回落 CNE is_st 口径（fail-open，绝不因 ST 数据源失败而中断建面板）
        if "is_st" in panel.columns:
            # CNE is_st: 1=ST, 0=正常 → not_st = 1 - is_st
            panel["not_st"] = (1 - panel["is_st"].fillna(0).astype(int)).astype("int8")
        else:
            panel["not_st"] = pd.Series(1, index=panel.index, dtype="int8")


def _column_all_nan(panel: pd.DataFrame, column: str) -> bool:
    """列缺失或整列 NaN（预建空列）→ True（需要计算填充）。

    matrix 缓存路径优先查 precomputed 表（``panel.attrs["all_nan_columns"]``）：
    运行时 ``pd.isna(panel[col]).all()`` 会跨行扫单列，而行主序矩阵的列 stride
    （n_cols×4 ≈ 920B）小于 4KB 页 —— 扫一列就把整份文件 fault 进工作集
    （2026-11-21 实测：rss 0.30GB → 7.03GB）。写盘时数据本在内存，顺手存下来。
    """
    if column not in panel.columns:
        return True
    known = (getattr(panel, "attrs", None) or {}).get("all_nan_columns")
    if known is not None:
        return str(column) in known
    try:
        return bool(pd.isna(panel[column]).all())
    except Exception:  # noqa: BLE001 — 无法判定时按"已有值"处理，不覆盖
        return False


def _st_flags(panel: pd.DataFrame) -> "pd.Series | None":
    """风险警示板标记（1=ST / 0=非 ST）；数据源不可用时返回 None（fail-open）。"""
    try:
        from alphaagent.factor.metrics.st_mask import st_row_mask

        mask = st_row_mask(panel)
    except Exception as exc:  # noqa: BLE001 — 面板构建绝不因 ST 数据源失败而中断
        logger.warning("ST 标记计算失败（%s: %s），回落 panel is_st", type(exc).__name__, exc)
        return None
    if mask is None:
        return None
    return pd.Series(mask.astype("int8"), index=panel.index)


def is_cne_source(panel_path: str | Path | None) -> bool:
    """判断 panel_path 是否指向 CNE 数据源。"""
    return panel_path is not None and str(panel_path) == CNE_SOURCE


def list_available_plugins() -> list[DataSourcePlugin]:
    """列出所有已注册的数据源插件（按优先级排序）。"""
    return get_registry().list_plugins()


def list_available_columns() -> list[str]:
    """列出所有插件提供的 Panel 列名。"""
    return get_registry().all_panel_columns()


def add_plugin(plugin: DataSourcePlugin, loader: Any) -> None:
    """运行时动态注册插件（高级用法，一般推荐通过 plugins/ 目录自动发现）。"""
    get_registry().register(plugin, loader)

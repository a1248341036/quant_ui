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

import pandas as pd

from alphaagent.data.adapters.registry import (
    DataSourcePlugin,
    PluginRegistry,
    get_registry,
)

logger = logging.getLogger(__name__)

# 标识符：panel_path == CNE_SOURCE 时走 adapter
CNE_SOURCE = "cne://"

# 磁盘面板缓存：构建好的 panel 落盘 parquet，避免每次评估都重算 30-70s。
# key = (start, end, include_fundamentals, schema_version)，不含数据湖 watermark：
# 历史固定区间永远命中（不需要因 watermark 每天变化而重建累积文件）。
# 命中条件 = 请求区间被缓存面板实际覆盖范围（文件内 datetime 列）包含。
# 目录文件数上限 _CACHE_MAX_FILES，写入时淘汰最旧文件，锁死长期空间。
_CACHE_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "panel" / "cache"
_CNE_STATE_FILE = (
    Path(__file__).resolve().parents[4]
    / "CNEquity" / "data" / "quant_dataset" / "_cnequity" / "meta" / "state" / "daily_bars.json"
)
# 缓存格式/构建逻辑版本：代码变更影响 panel 内容时 +1 强制全部重建
# v5: arrow 数值列改为无 null 写入（恢复零拷贝 mmap 共享）+ 数据面按需裁剪列族
_CACHE_SCHEMA_VERSION = 5  # v5: zero-copy arrow + facet column pruning
# 缓存文件数上限（全量面板 parquet+arrow ≈ 6GB，聚焦面板 ≈ 0.3-2.6GB）。
# 小磁盘服务器可用 ALPHA_PANEL_CACHE_MAX_FILES 调小（代价：切换数据面组合时重建）。
_CACHE_MAX_FILES = max(2, int(os.environ.get("ALPHA_PANEL_CACHE_MAX_FILES", "8") or 8))
_CACHE_INDEX_COLS = ["datetime", "instrument"]
# 聚焦裁剪时**永远保留**的列（非面专属 + 评估/交付链路内部依赖）：
# - 行情核心列由 OUTPUT_COLUMNS 保证（build_panel_from_hq 的输出契约）；
# - engine_gate 回测长表契约要求 open/high/low/close/amount/volume/turnover_rate
#   （core/panel_schema.ALPHA_CNE_PANEL_SPEC.required_columns）——聚焦业绩面时
#   这些列虽不允许出现在因子里，但提交门禁仍要拿它们跑回测；
# - adj_close 用于 IC 衰减曲线现算 label（metrics/decay.panel_forward_label）；
# - float_cap 用于市值中性化；is_trade/not_st/adjfactor 为可交易性与复权口径。
_ALWAYS_KEEP_COLUMNS = frozenset({
    "open", "high", "low", "close", "amount", "volume", "turnover_rate",
    "adj_close", "adjfactor", "float_cap", "tot_cap", "is_trade", "not_st",
    "industry_sw_l1",
})
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
    from alphaagent.factor.mining.memory.expressions import facet_allowed_scope

    return facet_allowed_scope(focus)


def _column_facets(column: str) -> set[str]:
    """列名 → 所属数据面（延迟导入，避免 data 层硬依赖 mining 包）。"""
    from alphaagent.factor.mining.memory.expressions import expr_facets

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
    end: str | None,
    include_fundamentals: bool,
    sig: str = "all",
) -> Path:
    """由参数 + schema 版本 + 数据面签名生成写入路径（同一组合稳定复用同一文件）。"""
    mk = "-".join(str(x or "all") for x in (start, end))
    key = hashlib.sha256(
        f"{mk}|{include_fundamentals}|v{_CACHE_SCHEMA_VERSION}".encode("utf-8")
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


def _find_cached_panel(
    start: str | None,
    end: str | None,
    include_fundamentals: bool,
    focus_facets: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """遍历缓存目录，返回覆盖请求区间且**数据面签名一致**的缓存面板。

    优先走 arrow mmap 路径（数值列跨进程零拷贝共享物理页，多会话并发时
    内存不按会话数叠加）；arrow 缺失/读取失败回退 parquet 解压路径。

    签名一致性：聚焦 run 只认同聚焦面的缓存文件（列集合一致）。不再用"全量缓存
    服务聚焦请求"——那会让 mmap 把 4.4GB 全量 arrow 的所有列页都 fault 进来，
    在小内存机器上正是要避免的。
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
            try:
                # arrow mmap 快路径：与 parquet 同 key 的 .arrow 存在即优先
                from alphaagent.data.adapters.panel_mmap import read_panel_arrow_mmap

                arrow_path = path.with_suffix(".arrow")
                df = None
                if arrow_path.is_file():
                    df = read_panel_arrow_mmap(arrow_path)
                if df is None:
                    df = pd.read_parquet(path)
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
                continue  # 请求终点晚于缓存覆盖终点
            has_funda = any(str(c).startswith("funda_") for c in df.columns)
            if include_fundamentals and expected_sentinels:
                missing_sentinels = _missing_funda_sentinels(df, expected_sentinels)
                if missing_sentinels:
                    logger.warning(
                        "CNE panel 缓存 %s 缺哨兵列 %s，视为未命中（将重建）",
                        path.name, sorted(missing_sentinels),
                    )
                    continue
            elif not include_fundamentals and has_funda:
                continue  # 基本面开关不匹配，跳过
            logger.info(
                "CNE adapter: panel 命中磁盘缓存 %s (%d 行, %d 列, 签名 %s)",
                path.name, len(df), df.shape[1], want_sig,
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


def _purge_old_cache(keep: Path) -> None:
    """缓存目录超过上限时删除最旧文件（保留刚写入的 keep）。

    parquet 与同 key 的 .arrow 成对淘汰；.arrow 若正被其他进程 mmap
    （Windows unlink 会失败），单独跳过并警告——不影响主缓存淘汰。
    """
    try:
        if not _CACHE_ROOT.is_dir():
            return
        files = sorted(
            (p for p in _CACHE_ROOT.glob("panel_*.parquet") if p != keep),
            key=lambda p: p.stat().st_mtime,
        )
        while len(files) + 1 > _CACHE_MAX_FILES:  # +1 为刚写入的文件
            victim = files.pop(0)
            try:
                victim.unlink()
                logger.info("CNE panel 缓存淘汰旧文件: %s (%.0fMB)",
                            victim.name, victim.stat().st_size / 1e6)
            except OSError as exc:
                logger.warning("CNE panel 缓存淘汰失败: %s (%s)", victim.name, exc)
            arrow_victim = victim.with_suffix(".arrow")
            if arrow_victim.is_file():
                try:
                    arrow_victim.unlink()
                except OSError as exc:
                    logger.warning("CNE panel arrow 缓存淘汰失败（可能被其他进程 mmap 占用）: %s (%s)",
                                   arrow_victim.name, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CNE panel 缓存淘汰异常（忽略）: %s", exc)


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
        flat = panel.reset_index()
        fd, tmp = tempfile.mkstemp(dir=str(_CACHE_ROOT), prefix=".panel.", suffix=".parquet.tmp")
        os.close(fd)
        try:
            flat.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        except BaseException:  # noqa: BLE001
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # 顺带生成 arrow mmap 副本（供后续会话零拷贝 attach；失败不影响主缓存）
        try:
            from alphaagent.data.adapters.panel_mmap import write_panel_arrow

            write_panel_arrow(path.with_suffix(".arrow"), flat)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CNE panel arrow 副本生成异常（忽略）: %s", exc)
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


def load_panel_from_cne(
    *,
    start: str | None = None,
    end: str | None = None,
    universe_mask: bool = False,
    include_fundamentals: bool = True,
    asset_type: str = "stock",
    focus_facets: Sequence[str] | None = None,
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

    首次构建后落盘缓存：key 按 (start, end, include_fundamentals, 数据面签名,
    schema 版本) 生成，命中条件为「数据面签名一致 + 请求区间落在缓存覆盖范围内」。
    因此历史固定区间不受数据湖 watermark 每日更新影响，长期稳定秒级命中；
    目录文件数由 _CACHE_MAX_FILES 上限约束，不会无限累积。
    """
    focus = [str(f) for f in (focus_facets or ()) if f]
    sig = _facet_signature(focus)
    cache_path = _cache_path(start, end, include_fundamentals, sig)

    # 归一化日期参数为 YYYY-MM-DD 字符串：调用方（如 stacking 脚本）可能传
    # pd.Timestamp，核心行情插件能容忍，但 fundamental（date.fromisoformat）
    # 与 fund_flow（CNE reader 的日期比较）会静默失败 → 辅助插件整列丢失
    start = pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else None
    end = pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None

    if not universe_mask and asset_type == "stock":
        cached = _find_cached_panel(start, end, include_fundamentals, focus)
        if cached is not None and not cached.empty:
            return cached

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
    """就地补充 is_trade / not_st 列（如果缺失）。

    这些列由 core 插件的列映射提供 is_st，但需要额外计算：
    - is_trade = volume > 0
    - not_st = 1 - is_st
    """
    if "is_trade" not in panel.columns and "volume" in panel.columns:
        panel["is_trade"] = (panel["volume"].fillna(0) > 0).astype("int8")

    if "not_st" not in panel.columns and "is_st" in panel.columns:
        # CNE is_st: 1=ST, 0=正常 → not_st = 1 - is_st
        panel["not_st"] = (1 - panel["is_st"].fillna(0).astype(int)).astype("int8")
    elif "not_st" not in panel.columns:
        panel["not_st"] = pd.Series(1, index=panel.index, dtype="int8")


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

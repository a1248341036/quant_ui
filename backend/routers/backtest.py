from __future__ import annotations

import json
import logging
import pandas as pd
from pathlib import Path
import tempfile

from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core import backtest_archive
from core.attribution import brinson_attribution
from core.assets import ETF_PROFILE, STOCK_PROFILE
from core.composites import (FACTOR_OPTIONS, delete_composite,
                             load_composites, save_composite)
from core.data import load_signal_panel
from core.engine import latest_signals, run_backtest
from core.event_engine import run_event_backtest
from core.fund_engine import run_fund_backtest
from core.performance import quantstats_html
from core.store import normalize_universe
from core.strategy_pool import resolve_strategy as resolve_pool_strategy
from strategies.registry import STRATEGIES, list_strategies_by_type

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


router = APIRouter()


def _resolve_strategy(name: str) -> dict | None:
    """注册表 → 配置池 → 全量池解析；找不到返回 None。"""
    try:
        return resolve_pool_strategy(name)
    except KeyError:
        return None


def _is_etf(universe: str) -> bool:
    return normalize_universe(universe) == "ETF"


def _is_fund(universe: str) -> bool:
    return normalize_universe(universe) == "场外基金"


def _universe_to_asset_type(universe: str) -> str:
    """universe 名称 → 资产类型标识。"""
    n = normalize_universe(universe)
    if n == "ETF":
        return "etf"
    if n == "场外基金":
        return "fund"
    return "stock"


def _validate_strategy_for_universe(name: str, universe: str) -> str | None:
    """返回策略不适用时的错误信息；适用或未知配置时返回 None。"""
    strat = _resolve_strategy(name)
    if strat is None:
        return f"未知策略: {name}"
    asset_type = _universe_to_asset_type(universe)
    allowed = strat.get("types", ["stock", "etf", "fund"])
    if asset_type not in allowed:
        return f"策略「{name}」不适用于「{universe}」（仅 {allowed}）"
    return None


def _validate_composite_for_universe(
    weights: dict[str, float] | None, universe: str,
) -> str | None:
    """校验自定义组合中每个因子是否适用于所选资产类型。"""
    if not weights:
        return None
    asset_type = _universe_to_asset_type(universe)
    options = {item["name"]: item for item in FACTOR_OPTIONS}
    for name in weights:
        if name == "pred" and asset_type != "stock":
            return f"因子「pred」不适用于「{universe}」"
        item = options.get(name)
        if item is None:
            continue  # 引擎负责兼容动态因子/外部 pred 因子
        if asset_type not in item.get("types", ["stock", "etf", "fund"]):
            return f"因子「{name}」不适用于「{universe}」"
    return None


def _calc_start(start: str, warmup_days: int | None) -> str:
    """回测因子计算起点 = start 前移 warmup_days（与引擎口径一致）。"""
    ts = pd.Timestamp(start)
    if warmup_days and warmup_days > 0:
        ts = ts - pd.Timedelta(days=int(warmup_days))
    return ts.date().isoformat()


def _load_panel_for(universe: str, start: str | None = None,
                    end: str | None = None,
                    codes: list[str] | None = None) -> pd.DataFrame:
    data = services.load_data(start=start, end=end, codes=codes,
                              need_panel=not _is_fund(universe),
                              need_heavy=_is_etf(universe) or _is_fund(universe))
    if _is_etf(universe):
        return data["etf_panel"]
    if _is_fund(universe):
        return data["fund_panel"]
    return data["panel"]


def _eval_alpha_factor(factor_id: str, library: str,
                       bt_panel: pd.DataFrame,
                       start: str, end: str) -> pd.DataFrame:
    """从 AlphaAgent 因子库读取 DSL 表达式，在 CNE 面板上求值，
    返回 date×code 的因子得分矩阵（与 bt_panel 对齐）。

    bt_panel: 回测引擎使用的扁平面板（有 date/code 列）。
    返回的 DataFrame 索引为 date，列为 code，值同 _inject_pred_factor 期望。
    """
    from backend.alphaagent_service import get_factor_detail
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel

    # 1. 获取 DSL 表达式
    detail = get_factor_detail(factor_id, library=library)
    if "error" in detail:
        raise ValueError(detail["error"])
    expr = detail.get("expr", "")
    if not expr:
        raise ValueError(f"因子 {factor_id} 无 DSL 表达式")

    # 2. 从 CNE 数据湖加载 AlphaAgent 格式面板
    cne_panel = load_panel_from_cne(start=start, end=end, universe_mask=False)
    cne_panel = cne_panel.sort_index()

    # 3. DSL 求值
    raw = eval_factor(expr, cne_panel)
    if not isinstance(raw, pd.Series):
        raise TypeError(f"DSL 求值结果类型异常: {type(raw)!r}")
    values = align_series_to_panel(raw, cne_panel)
    factor_series = pd.Series(values, index=cne_panel.index, name="alpha_factor",
                              dtype="float32")

    # 4. 转为 date×code 矩阵
    # cne_panel index = MultiIndex(datetime, instrument)
    factor_df = factor_series.unstack(level="instrument")
    factor_df.index.name = "date"
    factor_df.columns.name = "code"

    # 5. 代码格式转换：CNE 用 000001.SZ 格式，回测面板用 000001 格式
    # 去掉后缀 .SZ / .SH / .BJ 即可
    factor_df.columns = [str(c).split(".")[0] for c in factor_df.columns]

    # 6. 对齐到回测面板的交易日历和股票代码
    bt_dates = pd.DatetimeIndex(sorted(bt_panel["date"].unique()))
    bt_codes = [str(c) for c in bt_panel["code"].unique()]
    factor_df = factor_df.reindex(index=bt_dates, columns=bt_codes)
    return factor_df


class BacktestRequest(BaseModel):
    universe: str = "科技TMT"
    strategy: str = "低换手冷门"
    top_n: int = 3
    capital: float = 5000.0
    freq: str = "monthly"
    start: str
    end: str
    exclude_kechuang: bool = True
    affordable: bool = True
    amount_q: float = 0.2   # am20 成交额分位过滤，旧脚本口径 0.2
    warmup_days: int | None = 400  # 因子预热天数，短窗口动量因子需预热
    cash_mode: bool = True  # 现金/整手执行模型（与模拟盘同口径）
    limit_flags: bool = True  # 涨跌停过滤：涨停不可买入、跌停不可卖出
    slippage_bps: float = 0.0  # 固定滑点（基点），买入价=开盘×(1+bps/1e4)
    max_participation: float = 0.0  # 流动性约束：单笔买入 <= 20日均成交额×该比例
    max_weight: float | None = None  # 单票权重上限，None 表示不限制
    lot_size: int = 100
    buy_cost: float = 0.0008
    sell_cost: float = 0.0013
    spread_bps: float | None = None
    min_commission: float | None = None
    industry_cap: int | None = None  # 行业分散：每行业最多选 N 只
    analyze: bool = False   # 额外输出 IC/分组因子质量分析
    composite_weights: dict[str, float] | None = None  # 多因子自由组合权重
    composite_directions: dict[str, bool] | None = None
    composite_name: str | None = None  # 组合显示名（可选）
    long_short: bool | None = None  # 多空对冲：多头 TopN + 空头最弱 N 只
    short_n: int | None = None  # 空头只数（默认同 top_n）
    short_cost_rate: float | None = None  # 空头年化融券费率
    industry_neutral: bool | None = None  # 行业中性化选股
    use_financial: bool = False  # 使用财务因子（PG fina_indicator/income）
    risk_neutral: bool = False  # 风格+行业风险中性化，并返回期末风险归因
    adx_filter: float | None = None  # ADX 趋势强度过滤：信号日 ADX >= 阈值才允许买入
    chandelier_mult: float = 0.0  # ATR Chandelier 出场乘数（0=关闭）
    chandelier_period: int = 22
    regime_adx: float | None = None  # 市场 ADX 低于该阈值时降仓
    regime_scale: float = 0.5
    alpha_factor_id: str | None = None  # AlphaAgent 因子库 ID，选中后用 DSL 因子回测
    alpha_library: str = "production"  # 因子库: production / candidate
    alpha_ascending: bool = False  # AlphaAgent 因子方向：False=买高，True=买低


class CompareRequest(BaseModel):
    universe: str = "科技TMT"
    strategies: list[str] = ["低换手冷门", "反转 20 日", "低波动"]
    top_n: int = 3
    capital: float = 5000.0
    freq: str = "monthly"
    start: str
    end: str
    exclude_kechuang: bool = True
    affordable: bool = True
    amount_q: float = 0.2
    warmup_days: int | None = 400
    cash_mode: bool = True
    limit_flags: bool = True
    slippage_bps: float = 0.0
    max_participation: float = 0.0
    max_weight: float | None = None
    lot_size: int = 100
    buy_cost: float = 0.0008
    sell_cost: float = 0.0013
    spread_bps: float | None = None
    min_commission: float | None = None
    adx_filter: float | None = None
    chandelier_mult: float = 0.0
    chandelier_period: int = 22
    regime_adx: float | None = None
    regime_scale: float = 0.5


@router.get("/strategies")
def strategies(universe: str | None = None):
    """统一策略源：配置池优先，空配置池回退注册表全量。

    universe: 传入时按资产类型过滤，只返回该类型可用的策略。
    """
    from core.strategy_pool import pool_names, resolve_strategy
    asset_type = _universe_to_asset_type(universe) if universe else None
    names = pool_names()
    if not names:
        items = [{"name": k, **v} for k, v in STRATEGIES.items()]
    else:
        items = []
        for n in names:
            try:
                d = resolve_strategy(n)
            except KeyError:
                continue
            items.append({"name": n, **d})
        if not items:
            items = [{"name": k, **v} for k, v in STRATEGIES.items()]
    if asset_type:
        items = [
            it for it in items
            if asset_type in it.get("types", ["stock", "etf", "fund"])
        ]
    return items


@router.get("/factors")
def factors():
    return FACTOR_OPTIONS


@router.get("/alpha-factors")
def alpha_factors(library: str = "production"):
    """列出 AlphaAgent 因子库中的因子，供回测页面选择。"""
    try:
        from backend.alphaagent_service import list_factors
        data = list_factors(library=library)
        # 精简返回：只取回测选择所需的字段
        items = []
        for f in data.get("factors", []):
            items.append({
                "factor_id": f["factor_id"],
                "name": f["name"],
                "expr": f["expr"],
                "library": library,
                "metrics": f.get("metrics", {}),
            })
        return {"library": library, "factors": items,
                "n_factors": len(items)}
    except Exception as exc:
        logger.warning("列出 AlphaAgent 因子失败: %s", exc)
        return {"library": library, "factors": [], "n_factors": 0,
                "error": str(exc)}


@router.get("/composites")
def composites():
    return list(load_composites().values())


class CompositeSave(BaseModel):
    name: str
    weights: dict[str, float]
    directions: dict[str, bool] | None = None


@router.post("/composites")
def composite_save(req: CompositeSave):
    try:
        item = save_composite(req.name, req.weights, req.directions)
    except ValueError as exc:
        return {"error": str(exc)}
    return item


@router.delete("/composites/{name}")
def composite_delete(name: str):
    return {"ok": delete_composite(name)}


@router.get("/names")
def names():
    return services.get_name_map()


@router.post("/backtest")
def backtest(req: BacktestRequest):
    is_composite = bool(req.composite_weights)
    is_alpha = bool(req.alpha_factor_id)
    if is_alpha:
        # AlphaAgent 因子模式：不走策略/组合验证，用 DSL 因子注入回测
        strat = None
    else:
        strat = None if is_composite else _resolve_strategy(req.strategy)
        if not is_composite:
            strategy_error = _validate_strategy_for_universe(req.strategy, req.universe)
            if strategy_error:
                return {"error": strategy_error}
    composite_error = _validate_composite_for_universe(
        req.composite_weights, req.universe)
    if composite_error:
        return {"error": composite_error}
    long_short = (req.long_short if req.long_short is not None
                  else (strat or {}).get("long_short", False))
    short_n = (req.short_n if req.short_n is not None
               else (strat or {}).get("short_n"))
    short_cost_rate = (req.short_cost_rate if req.short_cost_rate is not None
                       else (strat or {}).get("short_cost_rate", 0.0))
    industry_neutral = (req.industry_neutral if req.industry_neutral is not None
                        else (strat or {}).get("industry_neutral", False))
    adx_filter = req.adx_filter
    if (adx_filter is None and strat is not None
            and strat.get("adx_filter") is not None):
        try:
            adx_filter = float(strat["adx_filter"])
        except (TypeError, ValueError):
            pass
    is_fund = _is_fund(req.universe)
    if _is_etf(req.universe) and (
        req.industry_cap or req.industry_neutral or req.risk_neutral
    ):
        return {"error": "ETF 当前没有可靠的成分行业映射，暂不支持行业上限/行业中性/风险中性"}
    need_heavy = _is_etf(req.universe) or is_fund
    calc_start = _calc_start(req.start, req.warmup_days)
    codes = services.build_codes(req.universe, req.exclude_kechuang)
    data = services.load_data(start=calc_start, end=req.end, codes=codes,
                              need_panel=not is_fund, need_heavy=need_heavy)
    panel = None if is_fund else _load_panel_for(req.universe,
                                                 start=calc_start, end=req.end,
                                                 codes=codes)
    data_version = (str(data["fund_nav"]["date"].max().date())
                    if is_fund and len(data["fund_nav"])
                    else str(panel["date"].max().date()))
    fund_cost = is_fund and req.buy_cost == 0.0008 and req.sell_cost == 0.0013
    etf_cost = _is_etf(req.universe) and req.buy_cost == 0.0008 and req.sell_cost == 0.0013
    if is_fund:
        res = run_fund_backtest(
            nav=data["fund_nav"],
            codes=codes,
            factor="composite" if is_composite else strat["factor"],
            ascending=False if is_composite else strat["ascending"],
            start=req.start,
            end=req.end,
            capital=req.capital,
            top_n=req.top_n,
            freq=req.freq,
            affordable=req.affordable,
            amount_q=req.amount_q,
            warmup_days=req.warmup_days,
            cash_mode=req.cash_mode,
            limit_flags=False,
            slippage_bps=req.slippage_bps,
            max_participation=req.max_participation,
            max_weight=req.max_weight,
            lot_size=1,
            buy_cost=0.0015 if fund_cost else req.buy_cost,
            sell_cost=0.0050 if fund_cost else req.sell_cost,
            analyze=req.analyze,
            factor_weights=req.composite_weights,
            factor_directions=req.composite_directions,
            fund_names=services.get_fund_name_map(),
        )
    else:
        # AlphaAgent 因子模式：从 DSL 求值注入 external_scores
        external_scores = None
        if is_alpha:
            try:
                external_scores = _eval_alpha_factor(
                    req.alpha_factor_id, req.alpha_library,
                    panel, calc_start, req.end)
            except Exception as exc:
                return {"error": f"AlphaAgent 因子求值失败: {exc}"}
        res = run_backtest(
            panel=panel,
            codes=codes,
            factor="pred" if is_alpha else ("composite" if is_composite else strat["factor"]),
            ascending=req.alpha_ascending if is_alpha else (False if is_composite else strat["ascending"]),
            start=req.start,
            end=req.end,
            capital=req.capital,
            top_n=req.top_n,
            freq=req.freq,
            affordable=req.affordable,
            amount_q=req.amount_q,
            warmup_days=req.warmup_days,
            cash_mode=req.cash_mode,
            limit_flags=req.limit_flags and not _is_etf(req.universe),
            slippage_bps=req.slippage_bps,
            max_participation=req.max_participation,
            max_weight=req.max_weight,
            spread_bps=req.spread_bps,
            min_commission=req.min_commission,
            lot_size=req.lot_size,
            buy_cost=0.0003 if etf_cost else req.buy_cost,
            sell_cost=0.0003 if etf_cost else req.sell_cost,
            industry_map=services.get_industry_map()
            if (req.industry_cap or industry_neutral or req.risk_neutral) else None,
            industry_cap=req.industry_cap,
            analyze=req.analyze,
            factor_weights=req.composite_weights,
            factor_directions=req.composite_directions,
            long_short=long_short,
            short_n=short_n,
            short_cost_rate=short_cost_rate,
            industry_neutral=industry_neutral,
            use_financial=req.use_financial,
            risk_neutral=req.risk_neutral,
            adx_filter=adx_filter,
        chandelier_mult=req.chandelier_mult,
        chandelier_period=req.chandelier_period,
        regime_adx=req.regime_adx,
        regime_scale=req.regime_scale,
        execution_profile=ETF_PROFILE if _is_etf(req.universe) else STOCK_PROFILE,
        external_scores=external_scores,
)
    brinson = None
    if (not is_fund and not _is_etf(req.universe)
            and res.get("weight_history")
            and len(res["dates"]) == len(res["weight_history"])):
        try:
            b_detail, b_summary = brinson_attribution(
                panel, codes, res["weight_history"],
                pd.DatetimeIndex(res["dates"]), services.get_industry_map())
            brinson = {
                "detail": services.clean_records(b_detail.to_dict(orient="records")),
                "summary": services.clean_records(b_summary.to_dict(orient="records")),
            }
        except Exception:
            brinson = None
    nm = services.get_name_map()
    if is_fund:
        nm = {**nm, **services.get_fund_name_map()}
    holdings = res["holdings"].copy()
    holdings["name"] = [nm.get(str(c), "") for c in holdings["code"]]
    quality = None
    if res.get("factor_quality") is not None:
        q = res["factor_quality"]
        sign = -1.0 if (not is_composite and not is_alpha and strat and strat.get("ascending")) else 1.0
        quality = {
            "horizon": q["horizon"],
            "ic": {k: services._to_float(v) for k, v in q["ic"].items()},
            "group": {
                "groups": [{**g, "mean_fwd_ret": services._to_float(g["mean_fwd_ret"])}
                           for g in q["group"]["groups"]],
                "spread": services._to_float(q["group"]["spread"]),
                "spread_pa": services._to_float(q["group"]["spread_pa"]),
            },
            "ic_series": [{"date": p["date"], "ic": p["value"]}
                          for p in services.series_to_points(q["ic_series"])],
            "group_table": services.clean_records(q["group_table"].to_dict(orient="records")),
            "direction_adjusted": {
                "ic": services._to_float(q["ic"]["mean_ic"] * sign if q["ic"]["mean_ic"] is not None else None),
                "spread": services._to_float(q["group"]["spread"] * sign if q["group"]["spread"] is not None else None),
            },
        }
    metrics = {k: services._to_float(v) for k, v in res["metrics"].items()}
    bench_metrics = {k: services._to_float(v)
                     for k, v in res["bench_metrics"].items()}
    nav_points = services.series_to_points(res["nav"])
    bench_points = services.series_to_points(res["bench"])
    dd_points = services.series_to_points(res["drawdown"])
    last_signal = str(res["last_signal_date"].date()) if res["last_signal_date"] else None
    run_id = backtest_archive.save_run(
        kind="backtest",
        params={
            "universe": req.universe, "strategy": req.strategy, "top_n": req.top_n,
            "capital": req.capital, "freq": req.freq, "start": req.start,
            "end": req.end, "exclude_kechuang": req.exclude_kechuang,
            "affordable": req.affordable, "amount_q": req.amount_q,
            "warmup_days": req.warmup_days, "industry_cap": req.industry_cap,
            "analyze": req.analyze,
            "composite": is_composite,
            "composite_name": req.composite_name,
            "composite_weights": req.composite_weights,
            "alpha_factor_id": req.alpha_factor_id,
            "alpha_library": req.alpha_library if req.alpha_factor_id else None,
            "long_short": long_short,
            "short_n": short_n,
            "short_cost_rate": short_cost_rate,
            "industry_neutral": industry_neutral,
            "use_financial": req.use_financial,
            "risk_neutral": req.risk_neutral,
        },
        metrics=metrics,
        bench_metrics=bench_metrics,
        nav=nav_points,
        bench=bench_points,
        drawdown=dd_points,
        holdings=holdings.to_dict(orient="records"),
        trades=res["trades"].to_dict(orient="records"),
        summary={
            "strategy": req.strategy, "universe": req.universe,
            "start": req.start, "end": req.end, "freq": req.freq,
            "composite": is_composite,
            "composite_name": req.composite_name,
            "alpha_factor_id": req.alpha_factor_id,
            "total_return": metrics.get("总收益"),
            "annual": metrics.get("年化收益"),
            "sharpe": metrics.get("夏普"),
            "max_drawdown": metrics.get("最大回撤"),
            "last_signal_date": last_signal,
        },
        data_version=data_version,
    )
    return {
        "run_id": run_id,
        "composite": is_composite,
        "composite_name": req.composite_name,
        "alpha_factor_id": req.alpha_factor_id,
        "long_short": long_short,
        "short_n": short_n,
        "short_cost_rate": short_cost_rate,
        "industry_neutral": industry_neutral,
        "use_financial": req.use_financial,
        "risk_neutral": req.risk_neutral,
        "risk_attribution": res.get("risk_attribution"),
        "brinson": brinson,
        "metrics": metrics,
        "bench_metrics": bench_metrics,
        "nav": nav_points,
        "bench": bench_points,
        "drawdown": dd_points,
        "holdings": holdings.to_dict(orient="records"),
        "trades": res["trades"].to_dict(orient="records"),
        "last_signal_date": last_signal,
        "factor_quality": quality,
    }


@router.post("/backtest/compare")
def backtest_compare(req: CompareRequest):
    if not req.strategies:
        return {"error": "请至少选择一个策略"}
    unknown = [s for s in req.strategies if _resolve_strategy(s) is None]
    if unknown:
        return {"error": f"未知策略: {unknown}"}
    for strategy_name in req.strategies:
        strategy_error = _validate_strategy_for_universe(
            strategy_name, req.universe)
        if strategy_error:
            return {"error": strategy_error}
    is_fund = _is_fund(req.universe)
    need_heavy = _is_etf(req.universe) or is_fund
    calc_start = _calc_start(req.start, req.warmup_days)
    codes = services.build_codes(req.universe, req.exclude_kechuang)
    data = services.load_data(start=calc_start, end=req.end, codes=codes,
                              need_panel=not is_fund, need_heavy=need_heavy)
    panel = None if is_fund else _load_panel_for(req.universe,
                                                 start=calc_start, end=req.end,
                                                 codes=codes)
    data_version = (str(data["fund_nav"]["date"].max().date())
                    if is_fund and len(data["fund_nav"])
                    else str(panel["date"].max().date()))
    etf_cost = _is_etf(req.universe) and req.buy_cost == 0.0008 and req.sell_cost == 0.0013
    nm = services.get_name_map()
    if is_fund:
        nm = {**nm, **services.get_fund_name_map()}
    ind_map = services.get_industry_map()
    items = []
    bench_points = None
    bench_metrics = None
    for sname in req.strategies:
        strat = _resolve_strategy(sname) or {}
        if is_fund:
            res = run_fund_backtest(
                nav=data["fund_nav"], codes=codes, factor=strat["factor"],
                ascending=strat["ascending"], start=req.start, end=req.end,
                capital=req.capital, top_n=req.top_n, freq=req.freq,
                affordable=req.affordable, amount_q=req.amount_q,
                warmup_days=req.warmup_days, cash_mode=req.cash_mode,
                limit_flags=False, slippage_bps=req.slippage_bps,
                max_participation=req.max_participation,
                max_weight=req.max_weight, lot_size=1,
                buy_cost=0.0015, sell_cost=0.0050,
                fund_names=services.get_fund_name_map(),
            )
        else:
            res = run_backtest(
                panel=panel, codes=codes, factor=strat["factor"],
                ascending=strat["ascending"], start=req.start, end=req.end,
                capital=req.capital, top_n=req.top_n, freq=req.freq,
                affordable=req.affordable, amount_q=req.amount_q,
                warmup_days=req.warmup_days,
                cash_mode=req.cash_mode,
                limit_flags=req.limit_flags and not _is_etf(req.universe),
                slippage_bps=req.slippage_bps,
                max_participation=req.max_participation,
                max_weight=req.max_weight,
                spread_bps=req.spread_bps,
                min_commission=req.min_commission,
                lot_size=req.lot_size,
                buy_cost=0.0003 if etf_cost else req.buy_cost,
                sell_cost=0.0003 if etf_cost else req.sell_cost,
                industry_map=ind_map if strat.get("industry_cap") else None,
                industry_cap=strat.get("industry_cap"),
                long_short=strat.get("long_short", False),
                short_n=strat.get("short_n"),
                short_cost_rate=strat.get("short_cost_rate", 0.0),
                industry_neutral=strat.get("industry_neutral", False),
                adx_filter=req.adx_filter,
        chandelier_mult=req.chandelier_mult,
        chandelier_period=req.chandelier_period,
        regime_adx=req.regime_adx,
        regime_scale=req.regime_scale,
        execution_profile=ETF_PROFILE if _is_etf(req.universe) else STOCK_PROFILE,
    )
        holdings = res["holdings"].copy()
        holdings["name"] = [nm.get(str(c), "") for c in holdings["code"]]
        if bench_points is None:
            bench_points = services.series_to_points(res["bench"])
            bench_metrics = {k: services._to_float(v)
                             for k, v in res["bench_metrics"].items()}
        items.append({
            "name": sname,
            "group": strat.get("group", ""),
            "desc": strat.get("desc", ""),
            "metrics": {k: services._to_float(v) for k, v in res["metrics"].items()},
            "nav": services.series_to_points(res["nav"]),
            "drawdown": services.series_to_points(res["drawdown"]),
            "holdings": holdings.to_dict(orient="records"),
            "trades": res["trades"].to_dict(orient="records"),
            "last_signal_date": str(res["last_signal_date"].date()) if res["last_signal_date"] else None,
        })
    run_id = backtest_archive.save_run(
        kind="compare",
        params={
            "universe": req.universe, "strategies": req.strategies,
            "top_n": req.top_n, "capital": req.capital, "freq": req.freq,
            "start": req.start, "end": req.end,
            "exclude_kechuang": req.exclude_kechuang,
            "affordable": req.affordable, "amount_q": req.amount_q,
            "warmup_days": req.warmup_days,
        },
        bench_metrics=bench_metrics,
        nav=[{"name": it["name"], "points": it["nav"]} for it in items],
        drawdown=[{"name": it["name"], "points": it["drawdown"]} for it in items],
        holdings=[{"name": it["name"], "records": it["holdings"]} for it in items],
        trades=[{"name": it["name"], "records": it["trades"]} for it in items],
        summary={
            "universe": req.universe, "start": req.start, "end": req.end,
            "freq": req.freq,
            "strategies": [{
                "name": it["name"],
                "total_return": it["metrics"].get("总收益"),
                "annual": it["metrics"].get("年化收益"),
                "sharpe": it["metrics"].get("夏普"),
                "max_drawdown": it["metrics"].get("最大回撤"),
                "last_signal_date": it["last_signal_date"],
            } for it in items],
        },
        data_version=data_version,
    )
    return {"run_id": run_id, "items": items, "bench": bench_points,
            "bench_metrics": bench_metrics}


class SignalsRequest(BaseModel):
    universe: str = "科技TMT"
    strategy: str = "低换手冷门"
    top_n: int = 10
    composite_weights: dict[str, float] | None = None
    composite_directions: dict[str, bool] | None = None
    long_short: bool | None = None
    short_n: int | None = None
    use_financial: bool = False


class SweepRequest(BaseModel):
    mode: str = "event"  # event=双均线金叉参数扫描, factor=因子策略 walk-forward
    start: str
    end: str
    capital: float = 50000.0
    folds: int = 4
    short_list: list[int] = [3, 5, 8, 10, 13]
    long_list: list[int] = [10, 20, 30, 60]
    top_n: int = 3
    strategy: str = "双均线多头 5/20"
    top_n_list: list[int] = [3, 5]  # rolling 训练-测试：训练期候选持仓数
    freq_list: list[str] = ["monthly"]  # rolling 训练-测试：候选调仓频率


class AttributionRequest(BaseModel):
    code: str
    strategy: str
    universe: str = "科技TMT"
    capital: float = 5000.0
    start: str
    end: str
    exclude_kechuang: bool = True
    affordable: bool = True
    amount_q: float = 0.2
    warmup_days: int | None = 400
    cash_mode: bool = True
    limit_flags: bool = True
    max_weight: float | None = None
    lot_size: int = 100
    slippage_bps: float = 0.0
    max_participation: float = 0.0
    buy_cost: float = 0.0008
    sell_cost: float = 0.0013


def _compute_signals(universe, strategy, top_n, composite_weights=None,
                     composite_directions=None, long_short=None, short_n=None,
                     use_financial=False):
    is_composite = bool(composite_weights)
    strat = None if is_composite else _resolve_strategy(strategy)
    if not is_composite:
        strategy_error = _validate_strategy_for_universe(strategy, universe)
        if strategy_error:
            return None, {"error": strategy_error}
    composite_error = _validate_composite_for_universe(
        composite_weights, universe)
    if composite_error:
        return None, {"error": composite_error}
    if long_short is None:
        long_short = strat.get("long_short", False)
    if short_n is None:
        short_n = strat.get("short_n")
    codes = services.build_codes(universe, True)
    if _is_fund(universe) or _is_etf(universe):
        panel = _load_panel_for(universe)
    else:
        try:
            panel = load_signal_panel(codes)
        except Exception as exc:
            print(f"[signals] 流式面板加载失败，回退普通加载: {exc}", file=sys.stderr)
            panel = _load_panel_for(universe, codes=codes)
    _asset_type = "fund_nav" if _is_fund(universe) else ("etf" if _is_etf(universe) else "stock")
    sig, sig_date = latest_signals(
        panel, codes,
        "composite" if is_composite else strat["factor"],
        False if is_composite else strat["ascending"],
        top_n=top_n,
        factor_weights=composite_weights,
        factor_directions=composite_directions,
        long_short=long_short,
        short_n=short_n,
        use_financial=use_financial,
        asset_type=_asset_type,
    )
    nm = services.get_name_map()
    if _is_fund(universe):
        nm = {**nm, **services.get_fund_name_map()}
    items = sig.to_dict(orient="records")
    for it in items:
        it["name"] = nm.get(str(it["code"]), "")
    return {
        "signal_date": str(sig_date.date()),
        "factor": "composite" if is_composite else strat["factor"],
        "ascending": False if is_composite else strat["ascending"],
        "composite": is_composite,
        "items": items,
    }, None


@router.get("/signals")
def signals(universe: str = "科技TMT", strategy: str = "低换手冷门", top_n: int = 10,
            composite_weights: str | None = None, composite_directions: str | None = None,
            long_short: bool | None = None, short_n: int | None = None,
            use_financial: bool = False):
    weights = json.loads(composite_weights) if composite_weights else None
    directions = json.loads(composite_directions) if composite_directions else None
    data, err = _compute_signals(universe, strategy, top_n, weights, directions,
                                 long_short, short_n, use_financial)
    return err if err is not None else data


@router.post("/signals")
def signals_post(req: SignalsRequest):
    data, err = _compute_signals(
        req.universe, req.strategy, req.top_n,
        req.composite_weights, req.composite_directions,
        req.long_short, req.short_n, req.use_financial)
    return err if err is not None else data


@router.post("/sweep")
def sweep(req: SweepRequest):
    """参数稳健性扫描（同步执行，event 模式可能耗时 1-2 分钟）。"""
    codes = services.build_codes("科技TMT", True)
    calc_start = _calc_start(req.start, req.warmup_days)
    data = services.load_data(start=calc_start, end=req.end, codes=codes,
                              need_panel=not _is_etf(req.universe),
                              need_heavy=_is_etf(req.universe))
    panel = data["etf_panel"] if _is_etf(req.universe) else data["panel"]
    if req.mode == "event":
        from core.walkforward import golden_cross_sweep
        summary, heatmap, windows = golden_cross_sweep(
            panel, codes, req.start, req.end, req.capital,
            short_list=req.short_list, long_list=req.long_list,
            n_folds=req.folds, top_n=req.top_n,
        )
        heatmap_rows = heatmap.reset_index().rename(columns={"index": "short"})
        run_id = backtest_archive.save_run(
            kind="sweep",
            params={
                "mode": "event", "start": req.start, "end": req.end,
                "capital": req.capital, "folds": req.folds,
                "short_list": req.short_list, "long_list": req.long_list,
                "top_n": req.top_n,
            },
            summary={
                "mode": "event", "n_combos": len(summary),
                "best": services.clean_records(
                    summary.head(5).to_dict(orient="records")),
            },
            nav=services.clean_records(summary.to_dict(orient="records")),
            bench=services.clean_records(heatmap_rows.to_dict(orient="records")),
            trades=services.clean_records(windows.to_dict(orient="records")),
            data_version=str(panel["date"].max().date()),
        )
        return {
            "run_id": run_id,
            "mode": "event",
            "summary": services.clean_records(summary.to_dict(orient="records")),
            "heatmap": services.clean_records(heatmap_rows.to_dict(orient="records")),
            "heatmap_cols": [str(c) for c in heatmap.columns],
            "windows": services.clean_records(windows.to_dict(orient="records")),
        }
    if req.mode == "rolling_event":
        from core.walkforward import rolling_train_test_event
        windows, summary, param_history = rolling_train_test_event(
            panel, codes, req.start, req.end, req.capital,
            short_list=req.short_list, long_list=req.long_list,
            n_folds=req.folds, top_n=req.top_n,
        )
        run_id = backtest_archive.save_run(
            kind="sweep",
            params={
                "mode": "rolling_event", "start": req.start, "end": req.end,
                "capital": req.capital, "folds": req.folds,
                "short_list": req.short_list, "long_list": req.long_list,
                "top_n": req.top_n,
            },
            summary={
                "mode": "rolling_event", "n_windows": len(windows),
            },
            nav=services.clean_records(windows.to_dict(orient="records")),
            trades=services.clean_records(param_history.to_dict(orient="records")),
            data_version=str(panel["date"].max().date()),
        )
        return {
            "run_id": run_id,
            "mode": "rolling_event",
            "windows": services.clean_records(windows.to_dict(orient="records")),
            "summary": services.clean_records(summary.to_dict(orient="records")),
            "param_history": services.clean_records(
                param_history.to_dict(orient="records")),
        }
    if req.mode == "rolling":
        from core.walkforward import rolling_train_test_factor
        strat = _resolve_strategy(req.strategy)
        if strat is None:
            return {"error": f"未知策略: {req.strategy}"}
        windows, summary, param_history = rolling_train_test_factor(
            panel, codes, strat["factor"], strat["ascending"],
            req.start, req.end, req.capital,
            top_n_list=req.top_n_list, freq_list=req.freq_list,
            n_folds=req.folds,
        )
        run_id = backtest_archive.save_run(
            kind="sweep",
            params={
                "mode": "rolling", "strategy": req.strategy, "start": req.start,
                "end": req.end, "capital": req.capital, "folds": req.folds,
                "top_n_list": req.top_n_list, "freq_list": req.freq_list,
            },
            summary={
                "mode": "rolling", "strategy": req.strategy,
                "n_windows": len(windows),
            },
            nav=services.clean_records(windows.to_dict(orient="records")),
            trades=services.clean_records(param_history.to_dict(orient="records")),
            data_version=str(panel["date"].max().date()),
        )
        return {
            "run_id": run_id,
            "mode": "rolling",
            "strategy": req.strategy,
            "windows": services.clean_records(windows.to_dict(orient="records")),
            "summary": services.clean_records(summary.to_dict(orient="records")),
            "param_history": services.clean_records(
                param_history.to_dict(orient="records")),
        }
    strat = _resolve_strategy(req.strategy)
    if strat is None:
        return {"error": f"未知策略: {req.strategy}"}
    from core.walkforward import walk_forward_factor
    windows = walk_forward_factor(
        panel, codes, strat["factor"], strat["ascending"],
        req.start, req.end, req.capital, top_n=req.top_n, n_folds=req.folds,
    )
    run_id = backtest_archive.save_run(
        kind="sweep",
        params={
            "mode": "factor", "strategy": req.strategy, "start": req.start,
            "end": req.end, "capital": req.capital, "folds": req.folds,
            "top_n": req.top_n,
        },
        summary={
            "mode": "factor", "strategy": req.strategy,
            "n_windows": len(windows),
        },
        nav=services.clean_records(windows.to_dict(orient="records")),
        data_version=str(panel["date"].max().date()),
    )
    return {
        "run_id": run_id,
        "mode": "factor",
        "strategy": req.strategy,
        "windows": services.clean_records(windows.to_dict(orient="records")),
    }


@router.get("/backtest/runs")
def backtest_runs(kind: str | None = None, limit: int = 50):
    """最近回测/对比/扫描记录列表。"""
    df = backtest_archive.list_runs(kind=kind or None, limit=limit)
    return {"items": services.clean_records(df.to_dict(orient="records"))}


@router.get("/backtest/runs/{run_id}")
def backtest_run(run_id: int):
    """单条回测完整记录（含净值/持仓/交易）。"""
    rec = backtest_archive.get_run(run_id)
    if rec is None:
        return {"error": "记录不存在"}
    return rec


@router.delete("/backtest/runs/{run_id}")
def backtest_run_delete(run_id: int):
    return {"ok": backtest_archive.delete_run(run_id)}


@router.post("/backtest/quantstats")
def backtest_quantstats(req: BacktestRequest):
    """跑一次回测并生成 QuantStats HTML 报告，返回 HTML 文本。"""
    if not req.composite_weights:
        strategy_error = _validate_strategy_for_universe(req.strategy, req.universe)
        if strategy_error:
            return {"error": strategy_error}
    composite_error = _validate_composite_for_universe(
        req.composite_weights, req.universe)
    if composite_error:
        return {"error": composite_error}
    strat = _resolve_strategy(req.strategy)
    if strat is None:
        return {"error": f"未知策略: {req.strategy}"}
    is_fund = _is_fund(req.universe)
    if _is_etf(req.universe) and (
        req.industry_cap or req.industry_neutral or req.risk_neutral
    ):
        return {"error": "ETF 当前没有可靠的成分行业映射，暂不支持行业上限/行业中性/风险中性"}
    need_heavy = _is_etf(req.universe) or is_fund
    calc_start = _calc_start(req.start, req.warmup_days)
    codes = services.build_codes(req.universe, req.exclude_kechuang)
    data = services.load_data(start=calc_start, end=req.end, codes=codes,
                              need_panel=not is_fund, need_heavy=need_heavy)
    panel = None if is_fund else _load_panel_for(req.universe,
                                                 start=calc_start, end=req.end,
                                                 codes=codes)
    if is_fund:
        res = run_fund_backtest(
            nav=data["fund_nav"], codes=codes, factor=strat["factor"],
            ascending=strat["ascending"], start=req.start, end=req.end,
            capital=req.capital, top_n=req.top_n, freq=req.freq,
            affordable=req.affordable, amount_q=req.amount_q,
            warmup_days=req.warmup_days, cash_mode=req.cash_mode,
            limit_flags=False, slippage_bps=req.slippage_bps,
            max_participation=req.max_participation,
            max_weight=req.max_weight, lot_size=1,
            buy_cost=0.0015, sell_cost=0.0050,
            analyze=req.analyze,
            factor_weights=req.composite_weights,
            factor_directions=req.composite_directions,
            fund_names=services.get_fund_name_map(),
        )
    else:
        etf_cost = _is_etf(req.universe) and req.buy_cost == 0.0008 and req.sell_cost == 0.0013
        res = run_backtest(
            panel=panel, codes=codes, factor=strat["factor"],
            ascending=strat["ascending"], start=req.start, end=req.end,
            capital=req.capital, top_n=req.top_n, freq=req.freq,
            affordable=req.affordable, amount_q=req.amount_q,
            warmup_days=req.warmup_days,
            cash_mode=req.cash_mode,
            limit_flags=req.limit_flags and not _is_etf(req.universe),
            slippage_bps=req.slippage_bps,
            max_participation=req.max_participation,
            max_weight=req.max_weight,
            lot_size=req.lot_size,
            buy_cost=0.0003 if etf_cost else req.buy_cost,
            sell_cost=0.0003 if etf_cost else req.sell_cost,
            industry_map=services.get_industry_map()
            if (strat.get("industry_cap") or req.industry_neutral or req.risk_neutral)
            else None,
            industry_cap=strat.get("industry_cap"),
            long_short=req.long_short if req.long_short is not None else strat.get("long_short", False),
            short_n=req.short_n if req.short_n is not None else strat.get("short_n"),
            short_cost_rate=(req.short_cost_rate if req.short_cost_rate is not None
                             else strat.get("short_cost_rate", 0.0)),
            industry_neutral=(req.industry_neutral if req.industry_neutral is not None
                              else strat.get("industry_neutral", False)),
            use_financial=req.use_financial,
            risk_neutral=req.risk_neutral,
            adx_filter=req.adx_filter,
        chandelier_mult=req.chandelier_mult,
        chandelier_period=req.chandelier_period,
        regime_adx=req.regime_adx,
        regime_scale=req.regime_scale,
        execution_profile=ETF_PROFILE if _is_etf(req.universe) else STOCK_PROFILE,
)
    out_dir = PROJECT_ROOT / "results" / "performance"
    safe = req.strategy.replace("/", "_").replace(" ", "")
    out_path = out_dir / f"quantstats_{safe}_{req.start}_{req.end}.html"
    title = f"{req.strategy} · {req.universe} · {req.start}~{req.end}"
    html_path = quantstats_html(res["nav"], res["bench"], title=title,
                                out_path=out_path)
    if html_path is None or not html_path.exists():
        return {"error": "QuantStats 未安装或报告生成失败（不影响回测本身）"}
    return {"path": str(html_path), "html": html_path.read_text(encoding="utf-8")}


@router.post("/backtest/attribution")
def backtest_attribution(req: AttributionRequest):
    """事件策略回测 + Brinson 归因（行业配置/个股选择/交互）。"""
    if _is_etf(req.universe) or _is_fund(req.universe):
        return {"error": "事件归因当前仅支持股票；ETF 尚未接入成分行业映射，场外基金也不使用 OHLCV 事件引擎"}
    from backend.lab_runner import load_module
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="attribution_")
    with open(fd, "w", encoding="utf-8") as f:
        f.write(req.code or "")
    tmp_path = Path(tmp)
    try:
        mod = load_module(str(tmp_path))
    except Exception as exc:
        return {"error": f"代码加载失败: {type(exc).__name__}: {exc}"}
    finally:
        tmp_path.unlink(missing_ok=True)
    EVENT_STRATEGIES = getattr(mod, "EVENT_STRATEGIES", {}) or {}
    if req.strategy not in EVENT_STRATEGIES:
        return {"error": f"事件策略不存在: {req.strategy}，可用: "
                         + ", ".join(EVENT_STRATEGIES) or "(空)"}
    calc_start = _calc_start(req.start, req.warmup_days)
    codes = services.build_codes(req.universe, req.exclude_kechuang)
    data = services.load_data(start=calc_start, end=req.end, codes=codes,
                              need_heavy=False)
    panel = data["panel"]
    try:
        from core.assets import ETF_PROFILE, STOCK_PROFILE
        res = run_event_backtest(
            panel=panel, codes=codes,
            strategy_class=EVENT_STRATEGIES[req.strategy],
            start=req.start, end=req.end, capital=req.capital,
            warmup_days=req.warmup_days, amount_q=req.amount_q,
            limit_flags=req.limit_flags and not _is_etf(req.universe),
            lot_size=req.lot_size,
            slippage_bps=req.slippage_bps,
            max_participation=req.max_participation,
            buy_cost=req.buy_cost, sell_cost=req.sell_cost,
            execution_profile=ETF_PROFILE if _is_etf(req.universe) else STOCK_PROFILE,
        )
    except Exception as exc:
        return {"error": f"回测失败: {type(exc).__name__}: {exc}"}
    import pandas as pd
    dates = pd.DatetimeIndex(res["dates"])
    wh = res["weight_history"]
    # 预热窗口会导致 weight_history 比 dates 短，取尾部对齐（缺失日为月初空仓）
    if len(wh) < len(dates):
        dates = dates[-len(wh):]
    if len(wh) != len(dates):
        return {"error": f"归因数据长度不一致: weight={len(wh)} dates={len(dates)}"}
    industry_map = services.get_industry_map()
    detail, summary = brinson_attribution(
        panel=panel, codes=codes,
        weight_history=wh, dates=dates, industry_map=industry_map,
    )
    return {
        "detail": services.clean_records(detail.to_dict(orient="records")),
        "summary": services.clean_records(summary.to_dict(orient="records")),
    }

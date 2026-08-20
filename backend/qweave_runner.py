from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.qweave_research import ALPHA_SETS, build_alphas, load_panel, to_qweave_df


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = PROJECT_ROOT / "data" / "qweave"

QWEAVE_TEMPLATES = {
    "alpha158_core": {
        "label": "Alpha158 · 常用 30 因子",
        "description": "适合日常快速筛选，运行成本较低",
        "alpha_set": "alpha158", "alpha_limit": 30,
    },
    "alpha158_full": {
        "label": "Alpha158 · 全量",
        "description": "完整 Alpha158 因子集，研究时间较长",
        "alpha_set": "alpha158", "alpha_limit": 158,
    },
    "alpha101": {
        "label": "WorldQuant Alpha101",
        "description": "WorldQuant Alpha101 因子集",
        "alpha_set": "alpha101", "alpha_limit": 101,
    },
    "alpha191": {
        "label": "GTJA Alpha191",
        "description": "国泰君安 Alpha191 因子集",
        "alpha_set": "alpha191", "alpha_limit": 191,
    },
}


def _load_module(code: str):
    path = PROJECT_ROOT / "labs" / ".tmp" / f"qweave_{uuid.uuid4().hex}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("qweave_lab", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("无法加载研究代码")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        path.unlink(missing_ok=True)


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _records(table, limit: int | None = None, sort_by: list[str] | None = None):
    if table is None:
        return []
    if sort_by:
        try:
            table = table.sort(sort_by)
        except Exception:
            pass
    if limit:
        table = table.head(limit)
    return [{k: _json_value(v) for k, v in row.items()}
            for row in table.to_dicts()]


def _latest_factor_rows(df, names: list[str], limit_each: int = 20):
    """Return latest cross-section leaders/trailers so the UI can show codes."""
    import polars as pl
    latest = df.select(pl.col("date").max()).item()
    rows = []
    for name in names:
        part = (df.filter(pl.col("date") == latest)
                .select(["date", "code", pl.col(name).alias("value")])
                .drop_nulls("value")
                .with_columns(
                    pl.col("value").rank("ordinal", descending=True).alias("rank"),
                    (pl.col("value").rank("average") /
                     pl.len()).alias("percentile")))
        if part.height:
            chosen = pl.concat([part.sort("rank").head(limit_each),
                                part.sort("rank", descending=True).head(limit_each)])
            rows.extend({"date": _json_value(r["date"]), "code": str(r["code"]),
                         "factor": name, "value": _json_value(r["value"]),
                         "rank": _json_value(r["rank"]),
                         "percentile": _json_value(r["percentile"]),
                         "side": "top" if r["rank"] <= limit_each else "bottom"}
                        for r in chosen.to_dicts())
    return rows


def _backtest_payload(res: dict) -> dict:
    from backend import services
    return {
        "metrics": {k: services._to_float(v) for k, v in res["metrics"].items()},
        "bench_metrics": {k: services._to_float(v) for k, v in res["bench_metrics"].items()},
        "nav": services.series_to_points(res["nav"]),
        "bench": services.series_to_points(res["bench"]),
        "drawdown": services.series_to_points(res["drawdown"]),
        "trades": services.clean_records(res["trades"].to_dict(orient="records")),
        "holdings": services.clean_records(res["holdings"].to_dict(orient="records")),
        "last_signal_date": str(res["last_signal_date"].date()) if res.get("last_signal_date") else None,
    }


def _default_code() -> str:
    return '''# qweave 研究代码：返回一组因子表达式
# 可选：alpha158 / alpha101 / alpha191。也可以自行用 qweave.col/lit 组合表达式。
import qweave

ALPHA_SET = "alpha158"
ALPHA_LIMIT = 30
ALPHAS = []  # 留空表示使用右侧参数选择的标准因子集
'''


def template_code(name: str) -> str:
    item = QWEAVE_TEMPLATES.get(name)
    if item is None:
        raise ValueError(f"qweave 预制模板不存在: {name}")
    return f'''# qweave 预制研究模板：{item["label"]}
# {item["description"]}
import qweave

# 也可以直接修改这两个参数，或改成自己的 build_alphas()。
ALPHA_SET = "{item["alpha_set"]}"
ALPHA_LIMIT = {item["alpha_limit"]}
ALPHAS = []
'''


def parse_code(code: str) -> dict:
    if not code.strip():
        raise ValueError("研究代码不能为空")
    mod = _load_module(code)
    builder = getattr(mod, "build_alphas", None)
    alphas = getattr(mod, "ALPHAS", None)
    if callable(builder):
        alphas = builder()
    if alphas is None:
        raise ValueError("代码必须提供 build_alphas() 或 ALPHAS")
    if not isinstance(alphas, (list, tuple)):
        raise ValueError("build_alphas() / ALPHAS 必须返回 list")
    if not alphas and hasattr(mod, "ALPHA_SET"):
        import qweave
        fn_name = ALPHA_SETS.get(getattr(mod, "ALPHA_SET"), ("",))[0]
        if not fn_name:
            raise ValueError("ALPHA_SET 必须是 alpha158、alpha101 或 alpha191")
        alphas = getattr(qweave, fn_name)({})[:int(getattr(mod, "ALPHA_LIMIT", 30))]
    if not alphas:
        raise ValueError("build_alphas() / ALPHAS 必须返回非空 list")
    names = []
    for alpha in alphas:
        if not hasattr(alpha, "output_name"):
            raise ValueError("因子列表中存在不是 qweave 表达式的对象")
        names.append(alpha.output_name())
    return {"ok": True, "factor_count": len(names), "factors": names}


def run(req: dict) -> dict:
    import qweave
    from core.data import _pg_parquet_end
    from backend import services

    code = req.get("code", "")
    if not code.strip():
        code = _default_code()
    mod = _load_module(code)
    builder = getattr(mod, "build_alphas", None)
    declared = getattr(mod, "ALPHAS", None)
    if hasattr(mod, "ALPHA_SET") and (declared is None or len(declared) == 0):
        alpha_set = req.get("alpha_set", "alpha158")
        limit = req.get("alpha_limit", 30)
        alphas, names = build_alphas(alpha_set, limit)
    else:
        alphas = builder() if callable(builder) else declared
        if alphas is None:
            raise ValueError("代码必须提供 build_alphas() 或 ALPHAS")
        alphas = list(alphas)
        names = [a.output_name() for a in alphas]
    if not alphas:
        raise ValueError("没有可运行的因子")

    start = req.get("start") or "2022-01-01"
    end = req.get("end") or _pg_parquet_end()
    if not end:
        raise ValueError("无法确定数据最新日期")
    horizons = [int(x) for x in (req.get("horizons") or [1, 5, 10, 20])]
    if not horizons or any(x <= 0 for x in horizons):
        raise ValueError("horizons 必须是正整数列表")
    universe = req.get("universe") or "沪深300+中证500+中证1000"
    codes = services.build_codes(universe, bool(req.get("exclude_kechuang", True)))
    panel = load_panel(start, end, codes or None)
    if panel.empty:
        raise ValueError("研究区间内没有可用行情数据")
    df = to_qweave_df(panel)
    enriched = qweave.with_alphas(df, "code", "date", alphas)
    labeled = qweave.with_labels(enriched, "code", "date", horizons)
    run_dir = RUN_ROOT / f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    result = qweave.evaluate(
        labeled, "code", "date", factor_cols=names,
        label_cols=[f"ret_{h}" for h in horizons],
        quantiles=int(req.get("quantiles", 10)),
        min_cs_count=int(req.get("min_cs_count", 30)),
        cost_bps=float(req.get("cost_bps", 8.0)),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result.save(str(run_dir))
    (run_dir / "meta.json").write_text(json.dumps({
        "engine": "qweave", "universe": universe, "start": start, "end": end,
        "factors": names, "horizons": horizons, "rows": df.height,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = result.summary.sort("ic_mean", descending=True)
    latest_factor_rows = _latest_factor_rows(labeled, names)
    payload = {
        "ok": True, "engine": "qweave", "run_dir": str(run_dir),
        "alpha_set": req.get("alpha_set", "custom"), "factor_count": len(names),
        "factor_names": names, "horizons": horizons, "rows": df.height,
        "summary": _records(summary, 200),
        "latest_factor_rows": latest_factor_rows,
        "ic": _records(result.ic, 3000, ["date", "factor", "horizon"]),
        "quantile_returns": _records(result.quantile_returns, 3000, ["date", "factor", "bin"]),
        "portfolio": _records(result.portfolio, 3000, ["date", "factor", "horizon"]),
        "coverage": _records(result.coverage, 3000, ["date", "factor"]),
        "turnover": _records(result.turnover, 3000, ["date", "factor", "horizon"]),
    }
    if req.get("run_backtest"):
        score_factor = req.get("score_factor") or (summary["factor"][0] if summary.height else names[0])
        if score_factor not in names:
            raise ValueError(f"回测因子不存在: {score_factor}")
        import polars as pl
        score_df = labeled.select(["date", "code", score_factor]).to_pandas()
        score_df["date"] = pd.to_datetime(score_df["date"])
        score_matrix = score_df.pivot_table(index="date", columns="code", values=score_factor,
                                             aggfunc="last", observed=True)
        from core.engine import run_backtest
        from core.assets import STOCK_PROFILE
        bt_res = run_backtest(
            panel=panel, codes=codes or sorted(panel["code"].astype(str).unique()),
            factor="pred", ascending=False, start=start, end=end,
            capital=float(req.get("capital", 100000.0)), top_n=int(req.get("top_n", 10)),
            selection_mode=req.get("selection_mode", "top_n"),
            selection_pct=float(req.get("selection_pct", 0.10)),
            min_positions=int(req.get("min_positions", 1)),
            max_positions=req.get("max_positions"),
            freq=req.get("freq", "weekly"), affordable=bool(req.get("affordable", True)),
            amount_q=float(req.get("amount_q", 0.2)), warmup_days=0,
            slippage_bps=float(req.get("slippage_bps", 0.0)),
            max_participation=float(req.get("max_participation", 0.0)),
            max_weight=req.get("max_weight"),
            buy_cost=float(req.get("buy_cost", 0.0008)),
            sell_cost=float(req.get("sell_cost", 0.0013)),
            industry_map=services.get_industry_map() if req.get("industry_cap") else None,
            industry_cap=req.get("industry_cap"), external_scores=score_matrix,
            execution_profile=STOCK_PROFILE,
        )
        payload["backtest"] = {"factor": score_factor, **_backtest_payload(bt_res)}
    return payload


def execute(req: dict) -> dict:
    try:
        return run(req)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}

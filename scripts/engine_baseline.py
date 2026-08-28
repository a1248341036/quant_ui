"""run_backtest 拆分前/后基准快照脚本。

运行一组小型回测场景,把引擎输出的全部可序列化字段写入指定目录
(默认 results/engine_baseline/<before|after>),供拆分重构前后逐字段比对。

设计约束:
- 引擎已对齐米筐口径,拆分不得改变任何数值/订单/持仓输出;
- 本脚本只依赖 core.data / core.engine 的公开入口,不依赖重构内部实现;
- 所有值经 _clean() 归一化(NaN->None、Timestamp->ISO、numpy->原生),
  JSON 浮点经 Python repr 往返损失为零,可做逐位比对。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data import load_panel, load_tech  # noqa: E402
from core.engine import run_backtest  # noqa: E402

DEFAULT_START = "2022-01-01"
DEFAULT_END = "2023-12-31"
TOP_CODES = 40  # 小型股票池,保证执行快且稳定


def _load_codes() -> list[str]:
    tech = load_tech()
    codes = [str(c).zfill(6) for c in tech["code"].tolist()][:TOP_CODES]
    return codes


def _load_panel(codes: list[str], start: str, end: str) -> pd.DataFrame:
    calc_start = (pd.Timestamp(start) - pd.Timedelta(days=60)).date().isoformat()
    return load_panel(start=calc_start, end=end, codes=codes)


def _industry_map() -> dict[str, str]:
    tech = load_tech()
    return {str(c).zfill(6): str(ind)
            for c, ind in zip(tech["code"], tech["industry"])}


def _clean(obj):
    """递归归一化:Timestamp->ISO,numpy->原生,NaN/inf->None,Series/DataFrame/Index->records。"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (datetime, pd.Timedelta, np.timedelta64)):
        return str(obj)
    if isinstance(obj, pd.Series):
        return [_clean({"idx": k, "val": v}) for k, v in obj.items()]
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, pd.DatetimeIndex):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.Index):
        return [_clean(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _scenario_snapshot(name: str, desc: str, params: dict, res: dict) -> dict:
    payload = {
        "meta": {
            "scenario": name,
            "desc": desc,
            "params": _clean(params),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "result": _clean({
            "nav": res["nav"],
            "bench": res["bench"],
            "drawdown": res["drawdown"],
            "metrics": res["metrics"],
            "bench_metrics": res["bench_metrics"],
            "trades": res["trades"],
            "holdings": res["holdings"],
            "last_signal_date": (str(res["last_signal_date"].date())
                                 if res["last_signal_date"] is not None else None),
            "capital": res["capital"],
            "dates": res["dates"],
            "last_chosen": res["last_chosen"],
            "factor_quality": res.get("factor_quality"),
            "weight_history": res.get("weight_history"),
            "trades_detail": res.get("trades_detail"),
            "cash_history": res.get("cash_history"),
            "positions_history": res.get("positions_history"),
            "rejections": res.get("rejections"),
            "risk_attribution": res.get("risk_attribution"),
            "asset_type": res.get("asset_type"),
        }),
    }
    canonical = json.dumps(payload["result"], ensure_ascii=False,
                           sort_keys=True, separators=(",", ":"))
    payload["tree_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _scenarios() -> list[dict]:
    codes = _load_codes()
    base = {
        "codes": codes,
        "start": DEFAULT_START,
        "end": DEFAULT_END,
        "capital": 100_000.0,
        "top_n": 5,
        "freq": "monthly",
    }
    return [
        {
            "name": "s1_cash_monthly",
            "desc": "默认现金/整手模型,月度调仓,mom20 多头,analyze=True",
            "params": {**base, "factor": "mom20", "ascending": False,
                       "analyze": True, "warmup_days": 120},
            "kwargs": {"factor": "mom20", "ascending": False, "analyze": True,
                       "warmup_days": 120},
        },
        {
            "name": "s2_weight_model",
            "desc": "旧权重连续模型(cash_mode=False),月度调仓,turn20 低换手",
            "params": {**base, "factor": "turn20", "ascending": True,
                       "cash_mode": False, "warmup_days": 120},
            "kwargs": {"factor": "turn20", "ascending": True, "cash_mode": False,
                       "warmup_days": 120},
        },
        {
            "name": "s3_long_short",
            "desc": "多空对冲:动量 Top 多头 + 最弱空头,权重模型路径",
            "params": {**base, "factor": "mom20", "ascending": False,
                       "long_short": True, "short_n": 3,
                       "short_cost_rate": 0.086, "warmup_days": 120},
            "kwargs": {"factor": "mom20", "ascending": False, "long_short": True,
                       "short_n": 3, "short_cost_rate": 0.086, "warmup_days": 120},
        },
        {
            "name": "s4_daily_chandelier_adx",
            "desc": "日频调仓 + ADX 过滤 + Chandelier 止损(现金模型)",
            "params": {**base, "factor": "ma_cross5_20", "ascending": False,
                       "freq": "daily", "adx_filter": 22.0,
                       "chandelier_mult": 3.0, "chandelier_period": 22},
            "kwargs": {"factor": "ma_cross5_20", "ascending": False, "freq": "daily",
                       "adx_filter": 22.0, "chandelier_mult": 3.0,
                       "chandelier_period": 22},
        },
        {
            "name": "s5_composite_weights",
            "desc": "多因子加权组合 + 方向配置(现金模型)",
            "params": {**base, "factor": "", "ascending": True, "top_n": 6,
                       "factor_weights": {"mom20": 1.0, "vol20": -0.5},
                       "factor_directions": {"vol20": True},
                       "warmup_days": 120},
            "kwargs": {"factor": "composite", "ascending": True, "top_n": 6,
                       "factor_weights": {"mom20": 1.0, "vol20": -0.5},
                       "factor_directions": {"vol20": True}, "warmup_days": 120},
        },
        {
            "name": "s6_industry_regime",
            "desc": "行业分散上限 + 弱市 ADX 降仓(现金模型)",
            "params": {**base, "factor": "am20", "ascending": False, "top_n": 6,
                       "industry_cap": 2, "regime_adx": 22.0, "regime_scale": 0.5},
            "kwargs": {"factor": "am20", "ascending": False, "top_n": 6,
                       "industry_cap": 2, "regime_adx": 22.0, "regime_scale": 0.5},
        },
        {
            "name": "s7_semiannual_warmup",
            "desc": "半年调仓 + 长预热(warmup_days=400,预热段切片路径)",
            "params": {**base, "factor": "mom60", "ascending": False,
                       "freq": "semiannual", "warmup_days": 400},
            "kwargs": {"factor": "mom60", "ascending": False,
                       "freq": "semiannual", "warmup_days": 400},
        },
        {
            "name": "s8_risk_neutral",
            "desc": "完整风险中性化(风格+行业) + risk_attribution",
            "params": {**base, "factor": "mom20", "ascending": False, "top_n": 6,
                       "risk_neutral": True, "warmup_days": 120},
            "kwargs": {"factor": "mom20", "ascending": False, "top_n": 6,
                       "risk_neutral": True, "warmup_days": 120},
        },
        {
            "name": "s9_min_score_gate",
            "desc": "min_score 绝对门控(趋势突破,无合格标的时空仓)",
            "params": {**base, "factor": "brk20", "ascending": False, "top_n": 6,
                       "min_score": 0.0, "warmup_days": 120},
            "kwargs": {"factor": "brk20", "ascending": False, "top_n": 6,
                       "min_score": 0.0, "warmup_days": 120},
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="快照输出目录(必填)")
    ap.add_argument("--only", default=None, help="只跑指定场景名(逗号分隔)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(s.strip() for s in (args.only or "").split(",") if s.strip())

    industry = _industry_map()
    ran = []
    for sc in _scenarios():
        if only and sc["name"] not in only:
            continue
        name = sc["name"]
        codes = sc["params"]["codes"]
        start = sc["params"]["start"]
        end = sc["params"]["end"]
        print(f"[baseline] {name}: 加载面板 codes={len(codes)} "
              f"{start}~{end} ...", flush=True)
        panel = _load_panel(codes, start, end)
        call = {
            "panel": panel,
            "codes": codes,
            "start": start,
            "end": end,
            "capital": sc["params"]["capital"],
            "top_n": sc["params"]["top_n"],
            **sc["kwargs"],
        }
        if ("industry_cap" in call or call.get("risk_neutral")
                or call.get("industry_neutral")):
            call["industry_map"] = industry
        if "freq" not in call:
            call["freq"] = sc["params"]["freq"]
        res = run_backtest(**call)
        snap = _scenario_snapshot(name, sc["desc"], sc["params"], res)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"[baseline] {name}: 已写入 {path} tree_hash={snap['tree_hash']}",
              flush=True)
        ran.append(name)

    print(f"[baseline] 完成 {len(ran)} 个场景: {','.join(ran)}", flush=True)


if __name__ == "__main__":
    main()
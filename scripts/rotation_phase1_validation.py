#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1 minimal validation for AlphaAgent rotation-strategy infra (spec 5).

Proves on REAL panel data (bounded memory):
  1. canonical / hand-written rotation_strategy compile to deterministic
     strategy_id + design/compile fingerprints;
  2. carrier signals from real data -> deterministic state sequence;
  3. carrier data-health (dynamic whitelist): unavailable carriers reported,
     broken designs rejected at compile time with feedback;
  4. near-duplicate strategy rejection in registry (state-series > 0.95);
  5. state_conditional_ic (canonical style_state) enters the report;
  6. reproducibility: same design -> same strategy_id / same state-series hash.

Usage:
  .venv\\Scripts\\python.exe scripts\\rotation_phase1_validation.py ^
      --start 2020-01-01 --end 2026-10-12 --limit-tickers 4000

Outputs (ASCII report; state series parquet under <out>/):
  report.json  state_series.parquet  report.md  registry/  log
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.data.panel import load_panel, slice_panel  # noqa: E402
from alphaagent.factor.evaluation.context import EvaluationContext  # noqa: E402
from alphaagent.factor.evaluation.plugins import state_conditional_ic  # noqa: E402
from alphaagent.factor.evaluation.profile import EvaluationProfile  # noqa: E402
from alphaagent.factor.mining.rotation.registry import RotationRegistry  # noqa: E402
from core.risk_model import build_exposures, factor_returns  # noqa: E402
from core.rotation.compiler import (  # noqa: E402
    default_blind_start,
    compile_design,
    probe_carrier_health,
)
from core.rotation.style_regime import (  # noqa: E402
    build_canonical_signals,
    canonical_design,
    compute_states,
)


PANEL_CANDIDATES = [
    ROOT / "artifacts/panel/cache/panel_v10_f86dc4cda2acd3da_all.parquet",
]
NEEDED = [
    "close", "amount", "turnover_rate", "ret", "pb", "exp_roe",
    "float_cap", "industry_sw_l1", "mgn_balance", "label_5d_close_to_close",
]


class _Universe:
    """Deterministic, memory-bounded stock sampling (every k-th after sort)."""

    def __init__(self, panel: pd.DataFrame, cap: int) -> None:
        self.cap = cap
        if cap and cap > 0:
            inst = sorted(panel.index.get_level_values("instrument").unique())
            self.keep = set(inst[:: max(1, len(inst) // cap)][:cap])
        else:
            self.keep = None

    def apply(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.keep is None:
            return panel
        return panel[panel.index.get_level_values("instrument").isin(self.keep)]


def _resolve_panel() -> Path:
    for p in PANEL_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("cached alphaagent panel not found; run panel build first")


def _pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    s = panel[col].dropna()
    s.index = s.index.set_levels(
        [
            pd.to_datetime(s.index.levels[0]).normalize(),
            s.index.levels[1].astype(str).str.split(".").str[0].str.zfill(6),
        ]
    )
    return s.unstack(level=1).sort_index()


def _log(msg: str, log_path: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-12")
    ap.add_argument("--limit-tickers", type=int, default=4000,
                    help="deterministic universe cap for bounded memory (0 = all)")
    ap.add_argument("--style-window", type=int, default=20,
                    help="style exposure rolling window (days)")
    ap.add_argument("--out", default=str(ROOT / "artifacts/alphaagent/rotation_validation"))
    return ap.parse_args()


def _style_beta_from_panel(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    close = _pivot(panel, "close")

    def _align(m: pd.DataFrame) -> pd.DataFrame:
        return m.reindex(index=close.index, columns=close.columns)

    amount = _align(_pivot(panel, "amount"))
    turn = _align(_pivot(panel, "turnover_rate"))
    pb = _align(_pivot(panel, "pb"))
    roe = _align(_pivot(panel, "exp_roe"))
    minp = max(5, window // 2)
    am20 = amount.rolling(window, min_periods=minp).mean()
    turn20 = turn.rolling(window, min_periods=minp).mean()
    mom20 = close / close.shift(window) - 1.0
    vol20 = close.pct_change(fill_method=None).rolling(window, min_periods=minp).std()
    del amount, turn
    gc.collect()

    X, names = build_exposures(
        close.to_numpy(), am20.to_numpy(), turn20.to_numpy(),
        mom20.to_numpy(), vol20.to_numpy(),
        pb=pb.to_numpy(), growth=roe.to_numpy(),
        industry_map=None, codes=list(close.columns),
    )
    rets = close.pct_change(fill_method=None).to_numpy()
    beta, _ = factor_returns(rets, X)
    del X, rets, am20, turn20, mom20, vol20, pb, roe
    gc.collect()
    return pd.DataFrame(beta, index=close.index, columns=names)


def _state_hash(states: pd.Series) -> str:
    raw = "|".join(map(str, states.to_numpy())).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def main() -> int:
    args = _parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"phase1_validation_{datetime.now():%Y%m%d_%H%M%S}.log"
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "stages": {},
        "issues": [],
    }
    log = lambda m: _log(m, log_path)  # noqa: E731

    panel_path = _resolve_panel()
    log(f"panel={panel_path.name}")

    log("load panel (mmap, filtered columns)")
    panel = load_panel(panel_path, columns=NEEDED)
    panel = slice_panel(panel, start=args.start, end=args.end)
    uni = _Universe(panel, args.limit_tickers)
    panel = uni.apply(panel).dropna(subset=["close", "ret"])
    log(f"panel slice {panel.shape} ({time.time()-t0:.0f}s)")

    log("compute style factor returns via risk_model -> style_beta")
    style_beta = _style_beta_from_panel(panel, args.style_window)
    report["stages"]["style_beta"] = {
        "shape": list(style_beta.shape),
        "columns_available": sorted(style_beta.columns),
        "n_days": int(len(style_beta)),
    }
    log(f"style_beta {style_beta.shape} cols={sorted(style_beta.columns)}")

    panel2 = panel[["float_cap", "industry_sw_l1", "ret", "mgn_balance",
                    "label_5d_close_to_close"]].copy()
    del panel
    gc.collect()
    aux: dict[str, pd.Series] = {}
    m = panel2["mgn_balance"].dropna().groupby(level="datetime").sum()
    aux["margin_chg"] = m.pct_change()

    log("build carrier signals")
    carrier = build_canonical_signals(
        panel2, style_beta, index_bars=None, aux=aux, industry_field="industry_sw_l1"
    )
    log(f"carrier signals: {list(carrier.columns)}")

    log("carrier data-health probe")
    blind_start = default_blind_start()
    health = probe_carrier_health(carrier, blind_start, min_valid_days=20)
    health_table = {
        k: {"available": v.get("available"),
            "blind_coverage": v.get("blind_coverage")}
        for k, v in health.items()
    }
    report["stages"]["carrier_health"] = health_table
    log(f"carrier health: {health_table}")

    log("compile canonical design")
    canonical = compile_design(canonical_design(), mode="technical",
                               carrier_health=health, blind_start=blind_start)
    log(f"canonical strategy_id={canonical.strategy_id}")
    canonical_fp = canonical.fingerprints
    report["stages"]["canonical"] = {
        "strategy_id": canonical.strategy_id,
        "fingerprints": canonical_fp,
        "dimensions": [
            {"dim": d.dim, "carrier": d.carrier, "signal": d.signal,
             "window_days": d.window_days, "mode": d.threshold_mode}
            for d in canonical.dimensions
        ],
    }

    log("compute state sequence 2020-2026")
    ss = compute_states(canonical, carrier)
    frame = ss.to_frame()
    state_stats = {
        str(k): int(v) for k, v in ss.states.value_counts().items()
    }
    report["stages"]["state_sequence"] = {
        "span_start": str(ss.states.index.min().date()),
        "span_end": str(ss.states.index.max().date()),
        "n_days": int(len(ss.states)),
        "state_counts": state_stats,
        "series_hash": _state_hash(ss.states),
    }
    log(f"state series {ss.states.index.min().date()} -> "
        f"{ss.states.index.max().date()} n={len(ss.states)} stats={state_stats}")

    log("registry admission + near-duplicate rejection")
    reg = RotationRegistry(root=out / "registry")
    # canonical 状态序列作为公共查重基线（system_canonical，不允许换皮立项）
    reg.register_canonical_baseline(list(ss.states.astype(str)))
    # 手写非 canonical 设计（momentum regime，枚举与 baseline 不同 -> 收录）
    hand_design = {
        "version": 1,
        "thesis": "momentum regime: style momentum beta median regime (hand-written, non-canonical)",
        "evidence": "Phase1 validation hand-written momentum regime control design",
        "state_dimensions": [
            {"dim": "momentum", "carrier": "style_factor", "window_days": 20,
             "threshold_mode": "quantile", "params": {"q": 0.5}},
        ],
        "quantile_benchmark": "expanding",
        "mapping": {"SLOW": {"reversal": 1.05}, "MOMENTUM": {"momentum": 1.10}},
        "allocation": None,
        "guardrails": {"max_daily_turnover": 0.2, "min_state_duration_days": 5,
                       "no_lookahead": True},
    }
    hand_compiled = compile_design(hand_design, mode="technical",
                                   carrier_health=health, blind_start=blind_start)
    hand_ss = compute_states(hand_compiled, carrier)
    hand_fp = hand_compiled.fingerprints
    reg.register(
        strategy_id=hand_compiled.strategy_id,
        design_fp=hand_fp["design"],
        compile_fp=hand_fp["compile"],
        mode="technical",
        meta={"source": "phase1_validation", "thesis": hand_design["thesis"][:80]},
        state_series=list(hand_ss.states.astype(str)),
        allow_variant_of=False,
    )
    report["stages"]["registry"] = {"admitted": [hand_compiled.strategy_id]}

    # 近重复 = 设计结构变化但状态序列几乎一致。改 min_state_duration_days 只影响
    # degraded 标记、不改变状态枚举序列 -> 一致率 1.0 -> 必然触发 >0.95 拒收。
    near = copy.deepcopy(hand_design)
    near["guardrails"]["min_state_duration_days"] = 6
    near_compiled = compile_design(near, mode="technical",
                                   carrier_health=health, blind_start=blind_start)
    near_ss = compute_states(near_compiled, carrier)
    try:
        reg.register(
            strategy_id=near_compiled.strategy_id,
            design_fp=near_compiled.fingerprints["design"],
            compile_fp=near_compiled.fingerprints["compile"],
            mode="technical",
            state_series=list(near_ss.states.astype(str)),
            allow_variant_of=False,
        )
        report["stages"]["registry"]["near_duplicate_rejected"] = False
        log("NEAR-DUP: unexpectedly admitted")
    except Exception as exc:  # noqa: BLE001
        report["stages"]["registry"]["near_duplicate_rejected"] = True
        report["stages"]["registry"]["near_duplicate_reason"] = str(exc)[:160]
        log(f"near-duplicate rejected: {str(exc)[:120]}")

    log("broken-design compile rejections (data health + structural)")
    broken: list[dict] = []

    def _regime(carrier: str, mode_: str, params: dict) -> dict:
        return {
            "version": 1,
            "thesis": "market regime single-dim broken-design control",
            "evidence": "Phase1 validation broken-design rejection control",
            "state_dimensions": [
                {"dim": "market_regime", "carrier": carrier, "window_days": 60,
                 "threshold_mode": mode_, "params": params},
            ],
            "quantile_benchmark": "expanding",
            "mapping": {"RISK_OFF": {"volatility": 1.05}},
            "allocation": None,
            "guardrails": {"max_daily_turnover": 0.2,
                           "min_state_duration_days": 5, "no_lookahead": True},
        }

    for label, d in [
        ("macro_level + level mode (structural)",
         _regime("macro_level", "level", {"threshold": 0.0})),
        ("market_regime x index_spread (data-unavailable)",
         _regime("index_spread", "quantile", {"q": 0.5})),
    ]:
        try:
            compile_design(d, mode="technical",
                           carrier_health=health, blind_start=blind_start)
            broken.append({"case": label, "rejected": False})
            log(f"BROKEN {label}: unexpectedly compiled")
        except ValueError as e:
            msg = str(e)
            broken.append({
                "case": label, "rejected": True,
                "why": msg.splitlines()[1] if len(msg.splitlines()) > 1 else msg,
            })
            log(f"rejected {label}: {broken[-1]['why'][:100]}")
    report["stages"]["broken_designs"] = broken

    log("reproducibility: recompile same design")
    canonical2 = compile_design(canonical_design(), mode="technical",
                                carrier_health=health, blind_start=blind_start)
    ss2 = compute_states(canonical2, carrier)
    repro = {
        "same_id": canonical2.strategy_id == canonical.strategy_id,
        "same_series_hash":
            _state_hash(ss2.states) == _state_hash(ss.states),
    }
    report["stages"]["reproducibility"] = repro
    log(f"reproducibility: same_id={repro['same_id']} "
        f"same_series_hash={repro['same_series_hash']}")

    log("state_conditional_ic on real panel")
    st_map = ss.states.astype(str)
    panel2["style_state"] = panel2.index.get_level_values("datetime").map(st_map)
    sub = panel2.dropna(subset=["label_5d_close_to_close", "ret", "style_state"])
    factor = pd.Series(
        -sub["ret"].to_numpy(dtype=np.float64), index=sub.index, name="factor"
    )
    label = pd.Series(
        sub["label_5d_close_to_close"].to_numpy(dtype=np.float64),
        index=sub.index, name="label",
    )
    ctx = EvaluationContext(
        panel=sub[["style_state", "label_5d_close_to_close"]],
        factor=factor, label=label,
        profile=EvaluationProfile("phase1", "val"), factor_name="factor",
    )
    sic = state_conditional_ic(ctx, {"field": "style_state", "min_cover": 15})
    sic_public = {k: v for k, v in sic.items() if k != "state_ic_series"}
    report["stages"]["state_conditional_ic"] = sic_public
    log(f"state_conditional_ic={sic.get('state_conditional_ic')} "
        f"n_days={sic.get('n_days')} n_states={sic.get('n_states_used')}")

    frame.to_parquet(out / "state_series.parquet")
    report["stages"]["outputs"] = {
        "state_series_parquet": str(out / "state_series.parquet"),
        "registry_root": str(out / "registry"),
        "log": str(log_path),
    }
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_markdown(report, out / "report.md")
    log(f"done in {report['elapsed_seconds']}s -> {out}")
    return 0


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# AlphaAgent rotation-strategy Phase 1 minimal validation",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- elapsed: {report['elapsed_seconds']}s",
        f"- args: {json.dumps(report['args'])}",
        "",
        "## 1. style factor returns (style_beta)",
        f"- shape: {report['stages']['style_beta']['shape']}",
        f"- columns: {report['stages']['style_beta']['columns_available']}",
        "",
        "## 2. canonical strategy",
        f"- strategy_id: {report['stages']['canonical']['strategy_id']}",
        f"- fingerprints: {json.dumps(report['stages']['canonical']['fingerprints'])}",
        "",
        "## 3. carrier data health (dynamic whitelist)",
    ]
    for carrier, h in report["stages"]["carrier_health"].items():
        lines.append(f"- {carrier}: available={h['available']} blind_coverage={h['blind_coverage']}")
    lines += [
        "",
        "## 4. state sequence",
        f"- span: {report['stages']['state_sequence']['span_start']} -> "
        f"{report['stages']['state_sequence']['span_end']} "
        f"({report['stages']['state_sequence']['n_days']} days)",
        f"- counts: {json.dumps(report['stages']['state_sequence']['state_counts'])}",
        f"- series_hash: {report['stages']['state_sequence']['series_hash']}",
        "",
        "## 5. registry",
        f"- admitted: {report['stages']['registry']['admitted']}",
        f"- near-duplicate rejected: {report['stages']['registry']['near_duplicate_rejected']}",
        "",
        "## 6. broken-design rejections",
    ]
    for b in report["stages"]["broken_designs"]:
        lines.append(f"- {b['case']}: rejected={b['rejected']} -> {b.get('why', '')[:120]}")
    lines += [
        "",
        "## 7. reproducibility",
        f"- same_strategy_id: {report['stages']['reproducibility']['same_id']}",
        f"- same_series_hash: {report['stages']['reproducibility']['same_series_hash']}",
        "",
        "## 8. state_conditional_ic",
    ]
    sic = report["stages"]["state_conditional_ic"]
    for k, v in sic.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

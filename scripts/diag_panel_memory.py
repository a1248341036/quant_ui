# -*- coding: utf-8 -*-
"""面板加载 / 因子求值的内存分段诊断（2026-10-20）。

用途：定位"一跑十几个 G"到底吃在 stage 1（panel 物化）还是 stage 3（求值中间量），
为优化方向提供数据，而不是靠感觉调参。

用法：
  python scripts/diag_panel_memory.py --mode full     # 全量列（模拟 Web run 路径）
  python scripts/diag_panel_memory.py --mode pruned   # 按需取列（harness 路径）

输出每阶段 RSS 与峰值、面板形状、列数。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

PROC = psutil.Process(os.getpid())
_PEAK = {"rss": 0.0}
_STOP = threading.Event()


def _sampler() -> None:
    while not _STOP.wait(0.2):
        rss = PROC.memory_info().rss / 1e9
        if rss > _PEAK["rss"]:
            _PEAK["rss"] = rss


def _now() -> float:
    return PROC.memory_info().rss / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("full", "pruned", "facets"), default="full")
    ap.add_argument("--factors", type=int, default=5)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-20")
    args = ap.parse_args()

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()

    stage0 = _now()
    print(f"[diag] 进程基线 RSS={stage0:.2f}GB mode={args.mode}", flush=True)

    from alphaagent.data.adapters.cnequity import CNE_SOURCE
    from alphaagent.factor.mining.eval.schemas import EvalTrainRequest, SessionCreateRequest
    from alphaagent.factor.mining.eval.service import StockEvalService
    from alphaagent.factor.mining.research_spec import (
        effective_research_spec,
        resolve_profiles,
    )

    EXPRS = [
        ("t_illiq", "il = CS_ZSCORE(TS_MEAN(DIVIDE(TS_MEAN(DIVIDE(SUBTRACT($high,$low),$close),20), $amount), 20))\nil"),
        ("t_vol", "CS_ZSCORE(NEG(TS_STD($ret, 60)))"),
        ("t_turn", "CS_ZSCORE(NEG(TS_MEAN($turnover_rate, 20)))"),
        ("t_mom", "CS_ZSCORE(NEG(TS_MEAN($ret, 20)))"),
        ("t_amt", "CS_ZSCORE(TS_MEAN(DIVIDE($amount,$float_cap), 20))"),
    ]

    svc = StockEvalService(max_parallel_eval=1, profiles=resolve_profiles(effective_research_spec("technical_monthly")))
    columns = () if args.mode == "full" else (
        "adj_close", "ret", "amount", "high", "low", "close", "turnover_rate", "float_cap",
    )
    if args.mode == "facets":
        # 复现 Web run 的真实裁列口径：技术档默认面集 → 行情组的列
        from core.research_modes import default_focus_facets

        from alphaagent.data.adapters.cnequity import columns_for_facets

        columns = columns_for_facets(default_focus_facets("technical_monthly"))
        print(f"[diag] facets 面集裁列: {len(columns)} 列", flush=True)
    t0 = time.perf_counter()
    resp = svc.create_session(SessionCreateRequest(
        panel_path=str(CNE_SOURCE),
        label_col="label_20d_close_to_close", include_fundamentals=False,
        universe="all", columns=columns,
    ))
    load_s = time.perf_counter() - t0
    stage1 = _now()
    print(f"[diag] 1) 加载: {load_s:.1f}s  RSS={stage1:.2f}GB  增量={(stage1 - stage0):+.2f}GB  "
          f"rows={resp.panel_rows} cols={len(resp.available_columns or ())}", flush=True)

    sid = resp.session_id
    for i, (name, expr) in enumerate(EXPRS[: args.factors], 1):
        t1 = time.perf_counter()
        try:
            svc.eval_train(EvalTrainRequest(
                session_id=sid, multi_line_expr=expr, factor_name=name,
                include_detail_tables=False, label_quantile_n=10,
            ))
            err = ""
        except Exception as exc:  # noqa: BLE001
            err = f" [{type(exc).__name__}]"
        cur = _now()
        print(f"[diag] 2.{i}) 求值 {name}: {time.perf_counter() - t1:.1f}s  "
              f"RSS={cur:.2f}GB  峰值={_PEAK['rss']:.2f}GB{err}", flush=True)

    _STOP.set()
    print(f"[diag] 汇总: 基线={stage0:.2f}GB 加载后={stage1:.2f}GB "
          f"峰值={_PEAK['rss']:.2f}GB 列数={len(resp.available_columns or ())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

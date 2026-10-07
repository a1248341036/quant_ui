#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness 自驱动挖掘 · 批量评估器（宿主 agent 直接充当挖掘 LLM 时使用）。

与 agentscope 挖掘链路**同源**：同一 `StockEvalService`、同一 `effective_research_spec`、
同一 `DeliveryCriteria`（门槛数值零硬编码），因此评估结果与真实 run 内的评估一致。

输出：每因子的 train 指标（ic/icir/coverage/cs_autocorr/换手）+ **逐门 PASS/FAIL**；
`--val` 可对过门者追加样本外（val 段）评估；结果落 JSONL 供下一轮迭代。

用法：
    .venv\\Scripts\\python.exe scripts\\harness_mine.py --mode technical --factors mine.json --out rows.jsonl
    .venv\\Scripts\\python.exe scripts\\harness_mine.py --mode technical_monthly --factors mine.json --val

factors JSON 格式：[{"name": "...", "expr": "多行 DSL（末行输出）", "why": "机制/经济直觉"}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.data.adapters.cnequity import CNE_SOURCE  # noqa: E402
from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria  # noqa: E402
from alphaagent.factor.mining.eval.schemas import (  # noqa: E402
    EvalTrainRequest,
    EvalValRequest,
    SessionCreateRequest,
)
from alphaagent.factor.mining.research_spec import effective_research_spec  # noqa: E402
from alphaagent.factor.mining.service import StockEvalService  # noqa: E402

# 档位 → 该档评估 label（与 core/research_modes.py 的 recommended_label_col 一致）
MODE_LABEL = {
    "technical": "label_1d_open_to_open",
    "technical_weekly": "label_5d_close_to_close",
    "technical_monthly": "label_20d_close_to_close",
    "fundamental": "label_20d_close_to_close",
}


def _flags(mode: str, row: dict) -> list[str]:
    """按真源门槛逐门判定，返回未过门说明（空列表 = 全部通过）。"""
    c = DeliveryCriteria.from_spec(effective_research_spec(mode))
    cand = c.candidate
    ic, icir, cov = row.get("ic"), row.get("icir"), row.get("coverage")
    ac, turn = row.get("cs_autocorr"), row.get("turnover_rebalance")
    fails: list[str] = []
    if ic is None or abs(ic) < cand.min_abs_ic:
        fails.append(f"ic<{cand.min_abs_ic}")
    if icir is None or abs(icir) < cand.min_icir:
        fails.append(f"icir<{cand.min_icir}")
    if cov is None or cov < cand.min_coverage:
        fails.append(f"cov<{cand.min_coverage}")
    if ac is None or ac < cand.min_cs_autocorr:
        fails.append(f"cs_autocorr<{cand.min_cs_autocorr}")
    if turn is None or turn > c.turnover_gate_limit:
        fails.append(f"turnover>{c.turnover_gate_limit}")
    return fails


def _extract(res: dict) -> dict:
    s = res.get("summary") if isinstance(res, dict) else None
    s = s if isinstance(s, dict) else {}
    return {
        "ic": s.get("ic"),
        "icir": s.get("icir"),
        "rank_ic": s.get("rank_ic"),
        "coverage": s.get("factor_coverage"),
        "cs_autocorr": s.get("cs_pearson_autocorr"),
        "turnover_daily": res.get("avg_daily_side_turnover"),
        "turnover_rebalance": res.get("avg_rebalance_side_turnover"),
        "n_days": s.get("n_days"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Harness 挖掘：批量评估自研因子")
    ap.add_argument("--mode", default="technical", choices=sorted(MODE_LABEL))
    ap.add_argument("--factors", required=True, help="JSON: [{name, expr, why?}, ...]")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="逐行结果写出路径（JSONL）")
    ap.add_argument("--val", action="store_true", help="对 train 全门通过者追加 val 评估")
    ap.add_argument("--fundamentals", action="store_true", help="会话载入 funda_* 列")
    args = ap.parse_args()

    label = MODE_LABEL[args.mode]
    factors = json.loads(Path(args.factors).read_text(encoding="utf-8"))
    if args.limit:
        factors = factors[: args.limit]
    c = DeliveryCriteria.from_spec(effective_research_spec(args.mode))
    print(f"[mine] mode={args.mode} label={label} 因子 {len(factors)} 个；门槛 "
          f"|IC|>={c.candidate.min_abs_ic} |ICIR|>={c.candidate.min_icir} cov>={c.candidate.min_coverage} "
          f"autocorr>={c.candidate.min_cs_autocorr} turnover<={c.turnover_gate_limit} (freq={c.engine_gate.freq})",
          flush=True)

    svc = StockEvalService(max_parallel_eval=1)
    t0 = time.perf_counter()
    resp = svc.create_session(SessionCreateRequest(
        panel_path=str(CNE_SOURCE), label_col=label, include_fundamentals=args.fundamentals,
    ))
    sid = resp.session_id
    print(f"[mine] session={sid} rows={resp.panel_rows} load={time.perf_counter() - t0:.0f}s", flush=True)

    rows: list[dict] = []

    def _flush() -> None:
        if args.out:
            Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                      encoding="utf-8")

    try:
        for i, f in enumerate(factors, 1):
            name, expr = f["name"], f["expr"]
            t = time.perf_counter()
            try:
                got = _extract(svc.eval_train(EvalTrainRequest(
                    session_id=sid, multi_line_expr=expr, factor_name=name,
                    include_detail_tables=False, label_quantile_n=10,
                )))
                err = None
            except Exception as exc:  # noqa: BLE001 — 单因子失败不中断批量
                got, err = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
            fails = _flags(args.mode, got) if got else ["eval_error"]
            rows.append({"mode": args.mode, "name": name, "why": f.get("why", ""), "expr": expr,
                         "ms": round((time.perf_counter() - t) * 1000), "err": err, **got,
                         "fails": fails, "pass_stage_one_pre": not fails})
            print(f"  [{i:2d}/{len(factors)}] {name:34s} {'PASS' if not fails else 'fail':4s} "
                  f"ic={got.get('ic')} icir={got.get('icir')} cov={got.get('coverage')} "
                  f"ac={got.get('cs_autocorr')} turn={got.get('turnover_rebalance')} "
                  f"{rows[-1]['ms']}ms {'; '.join(fails)} {err or ''}", flush=True)
            _flush()

        if args.val:
            survivors = [r for r in rows if r.get("pass_stage_one_pre")]
            print(f"\n[mine] 送 val 的因子 {len(survivors)} 个", flush=True)
            for r in survivors:
                try:
                    v = _extract(svc.eval_val(EvalValRequest(
                        session_id=sid, multi_line_expr=r["expr"], factor_name=r["name"],
                        include_detail_tables=False, label_quantile_n=10,
                    )))
                    r["val"] = v
                    r["val_ok"] = (v.get("ic") is not None
                                   and abs(v["ic"]) >= c.candidate.min_val_abs_ic)
                    print(f"  val {r['name']:34s} ic={v.get('ic')} icir={v.get('icir')} "
                          f"cov={v.get('coverage')} ok={r['val_ok']}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    r["val_err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    print(f"  val {r['name']:34s} ERROR {r['val_err']}", flush=True)
            _flush()
    finally:
        svc.release_session(sid)

    n_pass = sum(1 for r in rows if r.get("pass_stage_one_pre"))
    print(f"\n[mine] 完成：{len(rows)} 次评估，stage_one 预检通过 {n_pass} 个，"
          f"用时 {time.perf_counter() - t0:.0f}s", flush=True)
    for r in rows:
        if r.get("pass_stage_one_pre"):
            print(f"  ★ {r['name']}: ic={r['ic']} icir={r['icir']} cov={r['coverage']} "
                  f"ac={r['cs_autocorr']} turn={r['turnover_rebalance']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

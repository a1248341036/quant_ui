"""临时脚本:评估"调整净利润(扣非)"因子变体,跑完即删。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.alphaagent_service import evaluate_multi_profile

VARIANTS = {
    "core_ni_roa": (
        "核心净利润ROA: 扣非净利润/总资产",
        "core_roa = DIVIDE($funda_profit_dedt, $funda_total_assets)\n"
        "CS_WINSORIZE(core_roa, 0.02, 0.98)",
    ),
    "core_ni_ep": (
        "核心净利润收益率: 扣非净利润/总市值",
        "core_ep = DIVIDE($funda_profit_dedt, $tot_cap)\n"
        "CS_WINSORIZE(core_ep, 0.02, 0.98)",
    ),
}


def main() -> None:
    results = {}
    for name, (desc, expr) in VARIANTS.items():
        t0 = time.perf_counter()
        print(f"[run] {name}: {desc}", flush=True)
        try:
            res = evaluate_multi_profile(
                multi_line_expr=expr,
                factor_name=name,
                include_fundamentals=True,
            )
            results[name] = {"desc": desc, "profiles": res}
            print(f"[done] {name} in {time.perf_counter() - t0:.1f}s", flush=True)
            for pid, out in res.items():
                summary = (out or {}).get("summary") or {}
                print(f"  {pid}: ic={summary.get('ic')} rank_ic={summary.get('rank_ic')} "
                      f"icir={summary.get('icir')} cov={summary.get('factor_coverage')} "
                      f"t={summary.get('fmb_t') or summary.get('t_stat')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results[name] = {"desc": desc, "error": repr(exc)}
            print(f"[error] {name}: {exc!r}", flush=True)

    out_path = "logs/factor_mining/_core_ni_eval.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()

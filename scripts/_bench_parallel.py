"""并发评估基准：用真实 StockEvalService 栈在导出面板上测不同并发度的吞吐。

用法（主 venv + worktree 代码）:
  D:/Quant/quant_ui/.venv/Scripts/python.exe scripts/_bench_parallel.py [concurrency]
"""
import sys, time
sys.path.insert(0, "D:/Quant/quant_ui_cmp")
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from alphaagent.factor.mining.eval.context import StockEvalContext
from alphaagent.factor.mining.eval.service import StockEvalService, EvalTrainRequest

CONC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PROFILE = sys.argv[2] if len(sys.argv) > 2 else "full"  # full | core_only

ctx = StockEvalContext(
    panel_path=Path("D:/Quant/AlphaAgent/artifacts/panel/panel_1d.parquet"),
    train_start="2020-01-01", train_end="2022-12-31",
    val_start="2023-01-01", val_end="2024-12-31",
    label_col="label_1d_close_to_close", include_fundamentals=False,
)
service = StockEvalService(max_parallel_eval=CONC)
if PROFILE == "core_only":
    from alphaagent.factor.evaluation.profile import default_evaluation_profiles
    profs = default_evaluation_profiles()
    lite = profs["train_screen"].as_dict()
    lite["profile_id"] = "train_screen_lite"
    lite["metrics"] = [m for m in lite["metrics"] if m["plugin"] == "cross_sectional_core"]
    from alphaagent.factor.evaluation.profile import EvaluationProfile
    profs["train_screen"] = EvaluationProfile("train_screen", "train", transforms=lite["transforms"], metrics=lite["metrics"], rules=profs["train_screen"].rules)
    service = StockEvalService(max_parallel_eval=CONC, profiles=profs)
elif PROFILE == "two_stage":
    from alphaagent.factor.evaluation.profile import resolve_profiles
    service = StockEvalService(max_parallel_eval=CONC, profiles=resolve_profiles({}))
session = service.sessions.create(ctx)
sid = session.session_id
print(f"panel rows={len(session.panel):,}  concurrency={CONC}")

# 12 个互不相同的表达式（模拟一轮 8 个 + 余量），含 TS 滚动/截面/门控等常见形态
EXPRS = [
    f"vol{w} = TS_STD($ret, {w})\nNEG(vol{w})" for w in (10, 20, 30, 60)
] + [
    f"corr{w} = TS_CORR($vwap, $volume, {w})\nNEG(corr{w})" for w in (10, 20, 30)
] + [
    f"mom{w} = DIVIDE($close, DELAY($close, {w}))\nz = CS_ZSCORE(CS_WINSORIZE(mom{w}, 1, 99))\nNEG(z)" for w in (5, 10, 20)
] + [
    "gap = DIVIDE(SUBTRACT($adj_open, DELAY($adj_close, 1)), DELAY($adj_close, 1))\nNEG(TS_MEAN(gap, 10))",
    "amp = DIVIDE(SUBTRACT($high, $low), $vwap)\nNEG(TS_MEAN(amp, 15))",
]

def run_one(i: int) -> tuple[float, str]:
    t0 = time.perf_counter()
    r = service.eval_train(EvalTrainRequest(
        session_id=sid, factor_name=f"bench_{i}", multi_line_expr=EXPRS[i % len(EXPRS)],
        include_detail_tables=False, label_quantile_n=10,
    ))
    dt = time.perf_counter() - t0
    ok = bool(r.get("ok"))
    return (dt if ok else -dt), str(r.get("screen_stage") or ("err" if not ok else "full"))

# 预热（JIT 编译等），不算基准
run_one(0)

N = len(EXPRS)
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    results = list(ex.map(run_one, range(N)))
wall = time.perf_counter() - t0
durs = [d for d, _ in results]
ok = [d for d in durs if d > 0]
stages: dict[str, int] = {}
for _, st in results:
    stages[st] = stages.get(st, 0) + 1
print(f"N={N} wall={wall:.1f}s throughput={N/wall:.2f} eval/s  "
      f"单次: max={max(ok):.1f}s mean={sum(ok)/len(ok):.1f}s  失败={N-len(ok)}  stages={stages}")

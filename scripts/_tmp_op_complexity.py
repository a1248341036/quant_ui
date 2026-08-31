"""临时诊断：量化记忆库中算子使用的简单程度。"""
import re
import sqlite3
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect(r"D:\Quant\quant_ui\artifacts\alphaagent\research_memory.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT expression, verdict FROM memory_entries").fetchall()

op_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
var_re = re.compile(r"\$[a-z_][a-z0-9_]*")
op_count_dist: Counter = Counter()
var_count_dist: Counter = Counter()
top_ops: Counter = Counter()
for r in rows:
    expr = r["expression"] or ""
    ops = {m.lower() for m in op_re.findall(expr)}
    vars_ = set(var_re.findall(expr))
    op_count_dist[len(ops)] += 1
    var_count_dist[len(vars_)] += 1
    top_ops.update(ops)

total = len(rows)
print(f"总条目: {total}")
print("\n== 每因子算子种数分布 ==")
for n in sorted(op_count_dist):
    print(f"  {n} 种算子: {op_count_dist[n]:4d} ({100.0*op_count_dist[n]/total:.0f}%)")

print("\n== 每因子原始变量数分布 ==")
for n in sorted(var_count_dist):
    print(f"  {n} 个变量: {var_count_dist[n]:4d} ({100.0*var_count_dist[n]/total:.0f}%)")

print("\n== 算子使用 Top12 ==")
for op, c in top_ops.most_common(12):
    print(f"  {op}: {c}")

cats = {
    "时序(ts_*)": lambda s: s.startswith("ts_"),
    "截面(cs_/rank/zscore)": lambda s: s.startswith("cs_") or s in ("rank", "zscore", "winsorize", "demean", "quantile", "normalize", "residualize", "neutralize"),
    "条件门(if_else/greater/less)": lambda s: s in ("if_else", "cond", "greater", "less", "and", "or"),
    "高阶矩/统计(skew/kurt)": lambda s: s in ("skew", "kurt", "median"),
}
cat_cov: Counter = Counter()
multi_cat_factor_counts: Counter = Counter()
for r in rows:
    ops = {m.lower() for m in op_re.findall(r["expression"] or "")}
    cats_hit = [name for name, f in cats.items() if any(f(o) for o in ops)]
    cat_cov[len(cats_hit)] += 1
    if len(cats_hit) >= 3:
        multi_cat_factor_counts[r["verdict"]] += 1
print("\n== 每因子覆盖的算子类别数（时序/截面/条件门/高阶统计）==")
for n in sorted(cat_cov):
    print(f"  覆盖 {n} 类: {cat_cov[n]:4d} ({100.0*cat_cov[n]/total:.0f}%)")
print(f"\n覆盖≥3 类的因子的 verdict 分布: {dict(multi_cat_factor_counts)}")
conn.close()

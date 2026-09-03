# -*- coding: utf-8 -*-
"""候选池异常回测指标重算：对 label 多日档（holding>1）的候选按新口径重跑
quantile_portfolio，并把正确指标写回 registry 的 metrics.quantile_portfolio。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.dsl import eval_factor  # noqa: E402
from alphaagent.factor.metrics import quantile_portfolio_metrics  # noqa: E402
from alphaagent.factor.window_config import DEFAULT_TRAIN_START, DEFAULT_VAL_END  # noqa: E402

REG = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json"
DRY_RUN = "--apply" not in sys.argv

reg = json.loads(REG.read_text(encoding="utf-8"))

# ① 找异常条目：label 多日档 && 组合指标疑似失真（|ann| 超过合理上限，或 dd>=0.9）
abnormal = []
for fid, e in reg.items():
    if not isinstance(e, dict):
        continue
    m = e.get("metrics") or {}
    qp = m.get("quantile_portfolio") or {}
    lc = m.get("label_col") or (e.get("ingest_config") or {}).get("label_col") or e.get("eval_label") or ""
    digits = "".join(ch for ch in str(lc) if ch.isdigit())
    holding = max(1, int(digits) if digits else 1)
    ann = qp.get("top_group_annualized_return")
    dd = qp.get("top_group_max_drawdown")
    distorted = holding > 1 and (ann is not None and abs(float(ann)) > 1.5 or (dd is not None and abs(float(dd)) >= 0.9))
    if distorted:
        abnormal.append((fid, e, holding))
        print(f"[异常] {fid} label={lc} holding={holding} ann={ann} dd={dd}")

if not abnormal:
    print("无异常条目，退出")
    sys.exit(0)

# ② panel 加载（覆盖候选的评估区间）
print("\n加载 panel（约 30-60 秒，磁盘缓存命中则更快）…")
panel = load_panel_from_cne(start=DEFAULT_TRAIN_START, end=DEFAULT_VAL_END, include_fundamentals=True)
print(f"panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")

# ③ 逐个重算并回写
for fid, e, holding in abnormal:
    m = e.get("metrics") or {}
    expr = str(e.get("expr") or "")
    lc = m.get("label_col") or (e.get("ingest_config") or {}).get("label_col") or e.get("eval_label")
    if not expr or lc not in panel.columns:
        print(f"[跳过] {fid}: 无表达式或 panel 缺 label 列 {lc}")
        continue
    try:
        raw = eval_factor(expr, panel)
        values = raw.reindex(panel.index)
        qp = quantile_portfolio_metrics(
            pd.Series(values, index=panel.index), panel[lc],
            n_groups=10, cost_bps=0.0, holding_days=holding,
        )
        old_qp = m.get("quantile_portfolio") or {}
        print(
            f"\n[{fid}] holding={holding}\n"
            f"  旧: ann={old_qp.get('top_group_annualized_return')} dd={old_qp.get('top_group_max_drawdown')} sharpe={old_qp.get('top_group_sharpe')}\n"
            f"  新: ann={round(float(qp['top_group_annualized_return']), 4)} "
            f"dd={round(float(qp['top_group_max_drawdown']), 4)} "
            f"sharpe={round(float(qp['top_group_sharpe']), 4)} "
            f"excess={round(float(qp['top_group_annualized_excess_return']), 4)} n_days={qp['n_days']}"
        )
        if DRY_RUN:
            continue
        cleaned = {
            k: (round(float(v), 6) if isinstance(v, (int, float)) and pd.notna(v) and float(v) == float(v) else v)
            for k, v in qp.items() if k != "group_means"
        }
        m["quantile_portfolio"] = cleaned
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {fid}: {type(exc).__name__}: {str(exc)[:160]}")

if DRY_RUN:
    print("\ndry-run 完成，加 --apply 写回 registry")
    sys.exit(0)

REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n已写回", REG)

# -*- coding: utf-8 -*-
"""候选池存量因子三段组合指标回填：对 mining_candidate_registry.json 中
所有可用候选（quantile_portfolio.available==1）重算 train/val/test 分段
quantile_portfolio，并写入 metrics.portfolio_by_segment。

- 默认 dry-run：只打印计划，不写盘；加 --apply 才真正写回 registry。
- 分段窗口：train 2020-01-01~2022-12-31，val 2023-01-01~2024-12-31，
  test 2025-01-01~2026-09-10（与 submit 链路 ctx 窗口及 _backfill_ema3_direct 一致）。
- 不可用（avail=0，截面离散度不足）条目跳过，不覆盖原 message/error。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.dsl import eval_factor  # noqa: E402
from alphaagent.factor.metrics import quantile_portfolio_metrics  # noqa: E402

REG = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json"
DRY_RUN = "--apply" not in sys.argv

SEGMENTS = {
    "train": ("2020-01-01", "2022-12-31"),
    "val": ("2023-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-09-10"),
}


def _holding_days(label_col: str) -> int:
    digits = "".join(ch for ch in str(label_col or "") if ch.isdigit())
    return max(1, int(digits) if digits else 1)


def _cleaned(qp: dict) -> dict:
    return {
        k: (round(float(v), 6) if isinstance(v, (int, float)) and pd.notna(v) and float(v) == float(v) else v)
        for k, v in qp.items()
        if k != "group_means"
    }


def main() -> int:
    reg = json.loads(REG.read_text(encoding="utf-8"))
    targets = []
    for fid, e in reg.items():
        if not isinstance(e, dict):
            continue
        m = e.get("metrics") or {}
        qp = m.get("quantile_portfolio") or {}
        if not qp.get("available"):
            print(f"[跳过] {fid}: 组合指标不可用（{qp.get('message', '')[:40]}）")
            continue
        if "portfolio_by_segment" in m:
            print(f"[跳过] {fid}: 已有 portfolio_by_segment")
            continue
        expr = str(e.get("expr") or "")
        lc = m.get("label_col") or (e.get("ingest_config") or {}).get("label_col") or e.get("eval_label")
        if not expr or not lc:
            print(f"[跳过] {fid}: 缺 expr 或 label_col")
            continue
        targets.append((fid, e, expr, lc))
    if not targets:
        print("无可回填条目")
        return 0

    print(f"共 {len(targets)} 个候选需回填，加载 panel（30-60 秒，缓存命中更快）…")
    panel = load_panel_from_cne(start="2020-01-01", end="2026-09-10", include_fundamentals=True)
    print(f"panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")
    dt_level = panel.index.get_level_values("datetime")

    changed = 0
    for fid, e, expr, lc in targets:
        if lc not in panel.columns:
            print(f"[失败] {fid}: panel 缺 label 列 {lc}")
            continue
        try:
            raw = eval_factor(expr, panel)
            values = raw.reindex(panel.index) if hasattr(raw, "reindex") else pd.Series(raw, index=panel.index)
            values = pd.Series(values, index=panel.index)
            holding = _holding_days(lc)
            seg_metrics: dict[str, dict] = {}
            for seg, (s, e2) in SEGMENTS.items():
                mask = (dt_level >= pd.Timestamp(s)) & (dt_level <= pd.Timestamp(e2))
                if int(mask.sum()) == 0:
                    continue
                seg_qp = quantile_portfolio_metrics(
                    values[mask], panel[lc][mask],
                    n_groups=10, cost_bps=0.0, holding_days=holding,
                )
                seg_metrics[seg] = _cleaned(seg_qp)
                g = seg_metrics[seg]
                print(
                    f"\n[{fid}] seg={seg} ann={g.get('top_group_annualized_return')} "
                    f"excess={g.get('top_group_annualized_excess_return')} "
                    f"sharpe={g.get('top_group_sharpe')} dd={g.get('top_group_max_drawdown')}"
                )
            if not seg_metrics:
                print(f"[跳过] {fid}: 三段均无数据")
                continue
            if DRY_RUN:
                print(f"[dry-run] {fid}: 将写入 portfolio_by_segment={sorted(seg_metrics)}")
                continue
            m = e.get("metrics") or {}
            m["portfolio_by_segment"] = seg_metrics
            e["metrics"] = m
            changed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {fid}: {type(exc).__name__}: {str(exc)[:160]}")

    if DRY_RUN:
        print("\ndry-run 完成，加 --apply 写回 registry")
        return 0
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写回 {REG}（{changed} 个条目更新）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
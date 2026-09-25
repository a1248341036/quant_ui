"""Bench 逐 Run 指标趋势分析与展现。

按时间升序梳理 run 序列，计算同一配置分组内的 Δ 增量与判定状态。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alphaagent.factor.mining.bench.compare import METRIC_SPECS, _extract_metric

ROOT = Path(__file__).resolve().parents[4]
UI_ROOT = ROOT / "logs" / "factor_mining" / "ui"
BENCH_DIR = ROOT / "artifacts" / "alphaagent" / "bench"


def get_metric_trend(
    metric_key: str,
    last: int = 20,
    config_hash: str | None = None,
) -> list[dict[str, Any]]:
    """提取指定指标按 run 的历史变化趋势列表（按时间升序）。"""
    if not UI_ROOT.is_dir():
        return []

    # 扫描所有包含 scorecard.json 或有效 run 的目录
    run_records: list[dict[str, Any]] = []

    for d in UI_ROOT.iterdir():
        if not d.is_dir():
            continue
        sc_file = d / "scorecard.json"
        sc = None
        if sc_file.is_file():
            try:
                sc = json.loads(sc_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not sc:
            # 尝试即时轻量计算
            try:
                from alphaagent.factor.mining.run_metrics import compute_run_metrics
                from alphaagent.factor.mining.bench.extended_metrics import compute_extended_metrics
                bm = compute_run_metrics(d.name, d)
                ext = compute_extended_metrics(d.name, d, base_metrics=bm)
                sc = {
                    "run_id": d.name,
                    "summary": bm,
                    "time_meta": ext.get("time_meta", {}),
                    "headline": ext.get("headline", {}),
                    "exploration": ext.get("exploration", {}),
                    "dynamics": ext.get("dynamics", {}),
                    "quality": ext.get("quality", {}),
                    "cost": ext.get("cost", {}),
                    "process": ext.get("process", {}),
                    "integrity": ext.get("integrity", {}),
                }
            except Exception:
                continue

        t_meta = sc.get("time_meta") or {}
        created_at = t_meta.get("created_at") or ""
        cfg_hash = t_meta.get("config_hash") or f"{t_meta.get('model')}:{t_meta.get('research_spec_hash')[:8] if t_meta.get('research_spec_hash') else 'default'}"

        if config_hash and cfg_hash != config_hash:
            continue

        val = _extract_metric(sc, metric_key)
        run_records.append({
            "run_id": sc.get("run_id") or d.name,
            "created_at": created_at,
            "ended_at": t_meta.get("ended_at"),
            "wall_minutes": t_meta.get("wall_minutes") or 0.0,
            "commit": (t_meta.get("bench_commit") or "")[:8],
            "note": t_meta.get("bench_note") or "",
            "config_hash": cfg_hash,
            "value": val,
        })

    # 按 created_at 严格升序排序
    run_records.sort(key=lambda r: r["created_at"] or "")
    if last > 0 and len(run_records) > last:
        run_records = run_records[-last:]

    # 计算同配置段内的 Δ 上一 run
    prev_val_by_cfg: dict[str, float] = {}
    spec = METRIC_SPECS.get(metric_key, {"label": metric_key, "higher_is_better": True, "is_pp": False})
    higher_good = spec.get("higher_is_better", True)

    results = []
    for r in run_records:
        cfg = r["config_hash"]
        curr_v = r["value"]
        prev_v = prev_val_by_cfg.get(cfg)

        delta = None
        verdict = "-"
        if curr_v is not None and prev_v is not None:
            delta = round(curr_v - prev_v, 4)
            if abs(delta) < 1e-4:
                verdict = "[=]"
            else:
                is_good = (delta > 0) if higher_good else (delta < 0)
                verdict = "[+]" if is_good else "[-]"

        if curr_v is not None:
            prev_val_by_cfg[cfg] = curr_v

        results.append({
            **r,
            "delta": delta,
            "verdict": verdict,
            "is_segment_start": (prev_v is None),
        })

    return results


def format_trend_table(metric_key: str, trend_items: list[dict[str, Any]]) -> str:
    """渲染终端趋势表格。"""
    spec = METRIC_SPECS.get(metric_key, {"label": metric_key})
    label = spec.get("label", metric_key)

    lines = []
    lines.append("=" * 82)
    lines.append(f"【指标逐 Run 趋势追踪】: {label} ({metric_key})")
    lines.append("=" * 82)
    fmt = "  {:<16} {:>6}  {:<8}  {:<24}  {:>10}  {:>10}  {}"
    lines.append(f"  {'Run 开始时间':<16} {'时长':>6}  {'Commit':<8}  {'说明/Note':<24}  {'数值':>10}  {'Δ上一Run':>10}  判定")
    lines.append("  " + "-" * 78)

    last_cfg = None
    for r in trend_items:
        cfg = r["config_hash"]
        if last_cfg is not None and cfg != last_cfg:
            lines.append("  " + "·" * 78)
            lines.append(f"  [配置变更: {cfg}] (不跨配置段计算 Δ)")
        last_cfg = cfg

        c_ts = (r["created_at"] or "")[5:16].replace("T", " ")
        wall = f"{r['wall_minutes']:.0f}m"
        v_str = f"{r['value']:.4g}" if r["value"] is not None else "N/A"
        d_str = f"{r['delta']:+.4g}" if r["delta"] is not None else "—"
        note = (r["note"] or "—")[:22]
        commit = r["commit"] or "—"
        lines.append(fmt.format(c_ts, wall, commit, note, v_str, d_str, r["verdict"]))

    lines.append("=" * 82)
    return "\n".join(lines)

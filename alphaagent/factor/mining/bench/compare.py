"""Bench 指标比对与判定器。

支持可判定栏与参考栏分流、单 run 方向性评定与一页终端/Markdown 报告渲染。
"""
from __future__ import annotations

import math
from typing import Any

# 指标方向与分栏定义
METRIC_SPECS = {
    # ── 可判定栏 (低噪声 / 结构性指标) ──
    "stage_one_yield_pct": {"label": "海选过线率", "category": "漏斗", "higher_is_better": True, "is_pp": True, "det": True},
    "gate_survival_pct": {"label": "实盘门存活率", "category": "漏斗", "higher_is_better": True, "is_pp": True, "det": True},
    "effective_novelty_rate": {"label": "有效新颖率", "category": "探索", "higher_is_better": True, "is_pp": False, "det": True},
    "repeat_attempt_rate": {"label": "重复尝试率", "category": "探索", "higher_is_better": False, "is_pp": False, "det": True},
    "facet_coverage": {"label": "数据面覆盖数", "category": "探索", "higher_is_better": True, "is_pp": False, "det": True},
    "facet_entropy": {"label": "数据面分布熵", "category": "探索", "higher_is_better": True, "is_pp": False, "det": True},
    "family_coverage": {"label": "信号族覆盖数", "category": "探索", "higher_is_better": True, "is_pp": False, "det": True},
    "operator_entropy": {"label": "算子分布熵", "category": "探索", "higher_is_better": True, "is_pp": False, "det": True},
    "structure_variety": {"label": "结构指纹多样率", "category": "探索", "higher_is_better": True, "is_pp": False, "det": True},
    "prediction_coverage": {"label": "预测对账覆盖率", "category": "过程", "higher_is_better": True, "is_pp": True, "det": True},
    "ablation_coverage": {"label": "门控消融覆盖率", "category": "过程", "higher_is_better": True, "is_pp": True, "det": True},
    "advisory_follow_rate": {"label": "死路提示遵循率", "category": "过程", "higher_is_better": True, "is_pp": False, "det": True},
    "tool_error_rate": {"label": "工具执行错误率", "category": "过程", "higher_is_better": False, "is_pp": False, "det": True},
    "dup_dead_end_rate": {"label": "撞已知死路率", "category": "过程", "higher_is_better": False, "is_pp": False, "det": True},
    "cost_per_candidate_k_tokens": {"label": "每候选Token(k)", "category": "成本", "higher_is_better": False, "is_pp": False, "det": True},
    "cost_of_first_pass_k_tokens": {"label": "首过线Token(k)", "category": "成本", "higher_is_better": False, "is_pp": False, "det": True},
    "eval_latency_p50_s": {"label": "评估耗时P50(s)", "category": "耗时", "higher_is_better": False, "is_pp": False, "det": True},
    "eval_latency_p95_s": {"label": "评估耗时P95(s)", "category": "耗时", "higher_is_better": False, "is_pp": False, "det": True},
    "temporal_stability": {"label": "月度符号稳定性", "category": "质量", "higher_is_better": True, "is_pp": False, "det": True},
    "unsubmitted_passing_rate": {"label": "未交付过线率", "category": "诚实", "higher_is_better": False, "is_pp": False, "det": True},

    # ── 参考栏 (高波动 / 仅供参考不下结论) ──
    "candidate_stored": {"label": "入候选池数", "category": "产出", "higher_is_better": True, "is_pp": False, "det": False},
    "production_stored": {"label": "入正式库数", "category": "产出", "higher_is_better": True, "is_pp": False, "det": False},
    "unique_train_evaluated": {"label": "独特尝试训练", "category": "产出", "higher_is_better": True, "is_pp": False, "det": False},
    "total_tokens": {"label": "总消耗Token", "category": "成本", "higher_is_better": False, "is_pp": False, "det": False},
    "wall_time_minutes": {"label": "总耗时(分)", "category": "成本", "higher_is_better": False, "is_pp": False, "det": False},
    "eval_throughput": {"label": "评估吞吐(次/分)", "category": "效率", "higher_is_better": True, "is_pp": False, "det": False},
    "near_miss_count": {"label": "临门一脚数", "category": "过程", "higher_is_better": True, "is_pp": False, "det": False},
}


def _extract_metric(sc: dict[str, Any], key: str) -> float | None:
    """从 scorecard 递归或多层级提取指标标量值。"""
    # 优先从 headline 取
    hl = sc.get("headline") or {}
    if key in hl and hl[key] is not None:
        try:
            return float(hl[key])
        except Exception:
            pass

    # 分组查找
    for grp in ("funnel", "summary", "exploration", "dynamics", "quality", "cost", "process", "integrity"):
        b = sc.get(grp) or {}
        if key in b and b[key] is not None:
            try:
                return float(b[key])
            except Exception:
                pass
    return None


def compare_scorecards(
    curr_sc: dict[str, Any],
    base_sc: dict[str, Any],
    rel_tol: float = 0.10,
    abs_tol_pp: float = 1.5,
) -> dict[str, Any]:
    """对比两份 scorecard，产出指标对比明细与总评。"""
    # 检查配置哈希一致性
    curr_meta = curr_sc.get("time_meta") or {}
    base_meta = base_sc.get("time_meta") or {}
    curr_cfg_hash = curr_meta.get("config_hash")
    base_cfg_hash = base_meta.get("config_hash")
    config_matched = bool(curr_cfg_hash and base_cfg_hash and curr_cfg_hash == base_cfg_hash)

    det_items = []
    ref_items = []

    improved_count = 0
    regressed_count = 0

    for key, spec in METRIC_SPECS.items():
        base_v = _extract_metric(base_sc, key)
        curr_v = _extract_metric(curr_sc, key)

        if base_v is None and curr_v is None:
            continue

        item = {
            "key": key,
            "label": spec["label"],
            "category": spec["category"],
            "is_det": spec["det"],
            "base_val": base_v,
            "curr_val": curr_v,
            "delta": None,
            "rel_delta": None,
            "verdict": "[=]",
        }

        if base_v is not None and curr_v is not None:
            delta = curr_v - base_v
            rel_delta = delta / abs(base_v) if abs(base_v) > 1e-6 else 0.0
            item["delta"] = round(delta, 4)
            item["rel_delta"] = round(rel_delta, 4)

            # 容差带判断
            is_pp = spec.get("is_pp", False)
            in_band = False
            if is_pp:
                in_band = abs(delta) <= abs_tol_pp
            else:
                in_band = abs(rel_delta) <= rel_tol

            if in_band:
                verdict = "[=]"
            else:
                higher_good = spec["higher_is_better"]
                is_good = (delta > 0) if higher_good else (delta < 0)
                verdict = "[+]" if is_good else "[-]"
                if spec["det"]:
                    if is_good:
                        improved_count += 1
                    else:
                        regressed_count += 1

            item["verdict"] = verdict

        if spec["det"]:
            det_items.append(item)
        else:
            ref_items.append(item)

    # 总评判定
    if not config_matched and (curr_cfg_hash or base_cfg_hash):
        overall = "INVALID_CONFIG_MISMATCH"
    elif regressed_count > 0 and regressed_count >= improved_count:
        overall = "REGRESSED"
    elif improved_count > 0 and regressed_count == 0:
        overall = "IMPROVED"
    elif improved_count > 0 and improved_count > regressed_count:
        overall = "MOSTLY_IMPROVED"
    else:
        overall = "NO_CLEAR_CHANGE"

    return {
        "overall": overall,
        "config_matched": config_matched,
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "deterministic_metrics": det_items,
        "reference_metrics": ref_items,
    }


def format_one_page_report(
    curr_sc: dict[str, Any],
    base_sc: dict[str, Any] | None,
    diff_res: dict[str, Any] | None = None,
) -> str:
    """生成标准化的一页终端评测报告文本。"""
    c_meta = curr_sc.get("time_meta") or {}
    run_id = curr_sc.get("run_id", "?")
    c_ts = (c_meta.get("created_at") or "")[:16].replace("T", " ")
    commit = (c_meta.get("bench_commit") or "unknown")[:8]
    note = c_meta.get("bench_note") or "无说明"
    cfg_hash = c_meta.get("config_hash") or "unknown"

    lines = []
    lines.append("=" * 72)
    lines.append(f"【AlphaAgent 挖掘评测报告】 Run ID: {run_id}")
    lines.append(f"时间: {c_ts} | Commit: {commit} | Note: {note}")
    if base_sc:
        b_meta = base_sc.get("time_meta") or {}
        b_id = base_sc.get("run_id", "?")
        b_cfg = b_meta.get("config_hash") or "unknown"
        cfg_icon = "[一致]" if b_cfg == cfg_hash else "[不一致] (不可直接比)"
        lines.append(f"对比基线: {b_id} | 基线配置: {b_cfg} (当前配置: {cfg_hash} {cfg_icon})")
    lines.append("=" * 72)

    if not base_sc or not diff_res:
        lines.append("\n[提示] 暂无对比基线，显示当前单 run 指标：")
        for key, spec in METRIC_SPECS.items():
            val = _extract_metric(curr_sc, key)
            if val is not None:
                lines.append(f"  - [{spec['category']}] {spec['label']:<16}: {val:.4g}")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append("\n-- 可判定栏 (低噪声 / 结构性指标) --------------------------")
    fmt = "  {:<6} {:<16} {:>10}  ->  {:>10}   {:>10}   {}"
    lines.append(f"  {'分类':<6} {'指标名称':<16} {'基线值':>10}     {'本次值':>10}   {'变化幅度':>10}   判定")
    lines.append("  " + "-" * 66)

    for item in diff_res["deterministic_metrics"]:
        bv = f"{item['base_val']:.4g}" if item['base_val'] is not None else "N/A"
        cv = f"{item['curr_val']:.4g}" if item['curr_val'] is not None else "N/A"
        d_str = f"{item['delta']:+.4g}" if item['delta'] is not None else "-"
        lines.append(fmt.format(item['category'], item['label'], bv, cv, d_str, item['verdict']))

    lines.append("\n-- 参考栏 (高波动 / 仅供参考不下结论) ----------------------")
    ref_fmt = "  {:<6} {:<16} {:>10}  ->  {:>10}   {:>10}"
    for item in diff_res["reference_metrics"]:
        bv = f"{item['base_val']:.4g}" if item['base_val'] is not None else "N/A"
        cv = f"{item['curr_val']:.4g}" if item['curr_val'] is not None else "N/A"
        d_str = f"{item['delta']:+.4g}" if item['delta'] is not None else "-"
        lines.append(ref_fmt.format(item['category'], item['label'], bv, cv, d_str))

    lines.append("=" * 72)
    overall = diff_res["overall"]
    imp = diff_res["improved_count"]
    reg = diff_res["regressed_count"]
    lines.append(f"总评: {overall} (可判定栏 {imp} 项改善 / {reg} 项回归)")
    lines.append("注: 单 run 方向性参考，不构成显著性结论。")
    lines.append("=" * 72)

    return "\n".join(lines)

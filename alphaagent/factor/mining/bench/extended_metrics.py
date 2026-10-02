"""AlphaAgent 挖掘评测 v4 高级指标计算模块。

从 run 目录（run_*.jsonl, steps.log, run_meta.json, run_manifest.json 等）离线解析，
产出 A-F 六组指标与时间元数据，支持历史 run 回填与趋势追踪。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from alphaagent.dsl.core.ast import structure_fingerprint
from alphaagent.factor.facets import FACET_DEFS
from alphaagent.factor.mining.infra.audit import canonical_hash
from alphaagent.factor.mining.memory.constants import TRAIN_PASS_VERDICTS
from alphaagent.factor.mining.memory.expressions import classify_family_ex, expression_ops

# 8 个标准数据面
STANDARD_FACETS = [name for name, _ in FACET_DEFS]


def _shannon_entropy(counts: dict[str, int], total_categories: int | None = None) -> float:
    """计算归一化香农熵（0.0 ~ 1.0）。"""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs)
    max_k = total_categories if total_categories and total_categories > 1 else len(probs)
    denom = math.log2(max_k)
    return round(h / denom, 4) if denom > 0 else 0.0


def _parse_ts(val: Any) -> datetime | None:
    if not val:
        return None
    try:
        s = str(val).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _clean_expr_str(args_raw: Any) -> str:
    """从 arguments_raw 解析表达式文本。"""
    if isinstance(args_raw, dict):
        return str(args_raw.get("multi_line_expr") or "").strip()
    if isinstance(args_raw, str):
        s = args_raw.strip()
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                return str(obj.get("multi_line_expr") or "").strip()
            except Exception:
                pass
        return s
    return ""


def _extract_time_meta(run_dir: Path, events: list[dict]) -> dict[str, Any]:
    """提取 run 开始时间、结束时间与时长。三级回落确保历史 run 必有值。"""
    meta_path = run_dir / "run_meta.json"
    created_at_s = None
    title = None
    if meta_path.is_file():
        try:
            m_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at_s = m_obj.get("created_at")
            title = m_obj.get("title")
        except Exception:
            pass

    # 轨迹中首末时间戳
    ts_list = [_parse_ts(e.get("ts")) for e in events if e.get("ts")]
    good_ts = [t for t in ts_list if t]

    if not created_at_s and good_ts:
        created_at_s = good_ts[0].isoformat()

    if not created_at_s:
        try:
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
            created_at_s = mtime.isoformat()
        except Exception:
            created_at_s = datetime.utcnow().isoformat()

    ended_at_s = good_ts[-1].isoformat() if good_ts else created_at_s
    wall_min = 0.0
    if len(good_ts) >= 2:
        wall_min = round((good_ts[-1] - good_ts[0]).total_seconds() / 60.0, 1)

    # 提取模型与配置信息
    model = None
    spec_hash = None
    config_hash = None
    bench_note = None
    bench_commit = None

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        try:
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = man.get("model")
            spec_hash = man.get("research_spec_hash")
        except Exception:
            pass

    # 运行模式（report / technical / …）：API 启动的 run 没有 bench_meta.json，
    # 若不记录模式，bench 会把 report run 与 technical 基线当作"同配置"直接给结论
    # （实测 2026-10-02：c51f08948c14 得到误导性的 REGRESSED）。
    research_mode = None
    for _name in ("run_meta.json", "research_spec.json"):
        _p = run_dir / _name
        if not _p.is_file():
            continue
        try:
            _o = json.loads(_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        _m = _o.get("research_mode") or (_o.get("params") or {}).get("research_mode")
        if _m:
            research_mode = str(_m)
            break

    bench_meta_path = run_dir / "bench_meta.json"
    if bench_meta_path.is_file():
        try:
            b_obj = json.loads(bench_meta_path.read_text(encoding="utf-8"))
            config_hash = b_obj.get("config_hash")
            bench_note = b_obj.get("note")
            bench_commit = b_obj.get("commit")
            if not model:
                model = b_obj.get("model")
        except Exception:
            pass
    if not config_hash:
        # 无 bench_meta.json（API/监控启动的 run）→ 用 run 自身身份合成：
        # (research_mode, research_spec_hash)。这样 report 与 technical 不会 hash 相等，
        # compare.py 会因 config_matched=False 提示"配置不一致"而不是给出误导性结论。
        if research_mode or spec_hash:
            config_hash = f"spec:{spec_hash or '-'}|mode:{research_mode or 'unknown'}"

    return {
        "created_at": created_at_s,
        "ended_at": ended_at_s,
        "wall_minutes": wall_min,
        "title": title,
        "model": model,
        "research_spec_hash": spec_hash,
        "research_mode": research_mode,
        "config_hash": config_hash,
        "bench_note": bench_note,
        "bench_commit": bench_commit,
    }


def compute_extended_metrics(
    run_id: str,
    run_dir: Path | str,
    base_metrics: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """解析 run 目录，产出 A-F 六组指标字典与 time_meta。"""
    run_path = Path(run_dir)

    # 1. 确保轨迹事件加载
    if events is None:
        from alphaagent.factor.mining.run_metrics import _iter_run_events
        ev = _iter_run_events(run_path)
    else:
        ev = events

    # 2. 确保基础指标
    if base_metrics is None:
        from alphaagent.factor.mining.run_metrics import compute_run_metrics
        bm = compute_run_metrics(run_id, run_path)
    else:
        bm = base_metrics

    # 3. 提取时间与上下文
    time_meta = _extract_time_meta(run_path, ev)

    # 4. 单次遍历收集关键实体
    # 缓存表达式特征，避免对重复表达式重复解析 AST
    expr_cache: dict[str, tuple[str, set[str], list[str], str]] = {}

    def _get_expr_info(expr_str: str, factor_name: str = "") -> tuple[str, set[str], list[str], str]:
        if expr_str in expr_cache:
            return expr_cache[expr_str]
        try:
            fam, facs = classify_family_ex(factor_name, expr_str)
        except Exception:
            fam, facs = "unknown", set()
        try:
            ops = expression_ops(expr_str)
        except Exception:
            ops = []
        try:
            fp = structure_fingerprint(expr_str)
        except Exception:
            fp = canonical_hash(expr_str)
        res = (fam, set(facs), ops, fp)
        expr_cache[expr_str] = res
        return res

    # 遍历事件并记录
    current_turn = 0
    eval_attempts: list[dict[str, Any]] = []
    submits: list[dict[str, Any]] = []
    tokens_by_turn: dict[int, int] = defaultdict(int)

    # 统计容器
    facet_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    interaction_counts: Counter[str] = Counter()
    no_interaction_count = 0

    first_touch_facet: dict[str, int] = {}
    first_touch_family: dict[str, int] = {}

    expr_eval_count: Counter[str] = Counter()
    unique_fingerprints: set[str] = set()

    # 过程控制
    contradicted_fingerprints: set[str] = set()
    post_contradiction_violations = 0
    contradicted_eval_count = 0

    dead_end_fingerprints: set[str] = set()
    repeated_after_dead_end = 0
    dead_end_advisory_count = 0

    ablation_required_count = 0
    ablation_covered_count = 0

    attempt_gaps: list[float] = []
    eval_wall_times: list[float] = []
    timing_ms_totals: dict[str, list[float]] = defaultdict(list)

    diagnostic_verdicts: Counter[str] = Counter()
    bottlenecks: Counter[str] = Counter()
    screen_stages: Counter[str] = Counter()

    # 产出质量统计
    temporal_stability_samples: list[float] = []
    novelty_max_corrs: list[float] = []
    claim_consistency_matches = 0
    claim_consistency_total = 0

    for e in ev:
        evt = e.get("event")
        # 跟踪 turn 变化
        if "turn" in e and e.get("turn") is not None:
            try:
                current_turn = int(e["turn"])
            except Exception:
                pass

        if evt == "usage":
            # 记录每轮 token 消耗
            out_tok = int(e.get("output_tokens") or 0)
            in_tok = int(e.get("input_tokens") or 0)
            tokens_by_turn[current_turn] += (out_tok + in_tok)

        elif evt == "tool_results":
            for r in e.get("results") or []:
                name = r.get("name") or "?"
                res = r.get("result") if isinstance(r.get("result"), dict) else {}
                args_raw = r.get("arguments_raw")
                args_obj = args_raw if isinstance(args_raw, dict) else {}
                if isinstance(args_raw, str):
                    try:
                        args_obj = json.loads(args_raw)
                    except Exception:
                        args_obj = {}

                # 评估工具
                if name in ("evaluate_factor", "eval_on_train_set"):
                    expr_str = _clean_expr_str(args_obj if args_obj else args_raw)
                    factor_name = str(args_obj.get("factor_name") or r.get("factor_name") or "")

                    wall_sec = r.get("elapsed_seconds")
                    if wall_sec is not None:
                        try:
                            eval_wall_times.append(float(wall_sec))
                        except Exception:
                            pass
                    t_ms = res.get("timing_ms")
                    if isinstance(t_ms, dict):
                        for tk in ("dsl_eval_ms", "transforms_ms", "metrics_ms", "total_ms"):
                            if tk in t_ms:
                                try:
                                    timing_ms_totals[tk].append(float(t_ms[tk]))
                                except Exception:
                                    pass

                    # 检查是否成功
                    is_ok = bool(res.get("ok", True))
                    summ = res.get("summary") if isinstance(res.get("summary"), dict) else {}
                    if not summ and isinstance(res.get("metrics"), dict):
                        summ = res["metrics"]

                    # 提取解析信息
                    fam, facs, ops, fp = _get_expr_info(expr_str, factor_name)
                    expr_eval_count[expr_str] += 1
                    unique_fingerprints.add(fp)

                    # 探索覆盖统计
                    for fac in facs:
                        facet_counts[fac] += 1
                        if fac not in first_touch_facet:
                            first_touch_facet[fac] = current_turn
                    family_counts[fam] += 1
                    if fam not in first_touch_family:
                        first_touch_family[fam] = current_turn
                    for op in ops:
                        op_counts[op] += 1

                    # 交互结构
                    inter = args_obj.get("interaction") if isinstance(args_obj, dict) else None
                    if isinstance(inter, dict) and inter.get("interaction_type"):
                        itype = str(inter.get("interaction_type"))
                        interaction_counts[itype] += 1
                        # 检查声明与实现一致性
                        claim_consistency_total += 1
                        # 匹配规则:
                        # gated_signal -> GATED_SIGNAL / IF_THEN_ELSE
                        # residual_signal -> CS_RESIDUALIZE
                        # divergence_signal -> DIVERGENCE_RANK
                        # conditional_group_rank -> CS_GROUP_RANK
                        # piecewise_state -> PIECEWISE_STATE
                        ops_upper = [o.upper() for o in ops]
                        matched = False
                        if itype == "gated_signal" and ("GATED_SIGNAL" in ops_upper or "IF_THEN_ELSE" in ops_upper):
                            matched = True
                        elif itype == "residual_signal" and "CS_RESIDUALIZE" in ops_upper:
                            matched = True
                        elif itype == "divergence_signal" and "DIVERGENCE_RANK" in ops_upper:
                            matched = True
                        elif itype == "conditional_group_rank" and "CS_GROUP_RANK" in ops_upper:
                            matched = True
                        elif itype == "piecewise_state" and "PIECEWISE_STATE" in ops_upper:
                            matched = True
                        elif itype not in ("gated_signal", "residual_signal", "divergence_signal", "conditional_group_rank", "piecewise_state"):
                            matched = True  # 未定义严格算子映射的视为合法
                        if matched:
                            claim_consistency_matches += 1
                    else:
                        no_interaction_count += 1

                    # 规则判定检查 (screen_rules)
                    s_rules = res.get("screen_rules")
                    rules_all_passed = False
                    if isinstance(s_rules, list) and s_rules:
                        rules_all_passed = all(bool(rule.get("passed")) for rule in s_rules)
                        # 计算 near-miss 进度
                        failed_gaps = []
                        for rule in s_rules:
                            if not rule.get("passed"):
                                exp_val = rule.get("expected")
                                act_val = rule.get("actual")
                                if exp_val is not None and act_val is not None:
                                    try:
                                        ev_f = abs(float(exp_val))
                                        av_f = abs(float(act_val))
                                        if ev_f > 1e-6:
                                            failed_gaps.append(abs(av_f - ev_f) / ev_f)
                                    except Exception:
                                        pass
                        if failed_gaps:
                            attempt_gaps.append(min(failed_gaps))
                    elif res.get("verdict") in (*TRAIN_PASS_VERDICTS, "candidate_approved"):
                        rules_all_passed = True

                    # 预测对账与违规检测
                    pc = res.get("prediction_check")
                    p_verdict = pc.get("verdict") if isinstance(pc, dict) else None
                    if fp in contradicted_fingerprints:
                        post_contradiction_violations += 1
                    if p_verdict == "contradicted":
                        contradicted_fingerprints.add(fp)
                        contradicted_eval_count += 1

                    # 记忆 advisory 遵循检测
                    ma = res.get("memory_advisory")
                    has_dead_end = False
                    if isinstance(ma, dict):
                        for a in ma.get("advisories") or []:
                            if a.get("kind") == "duplicate_known_dead_end":
                                has_dead_end = True
                                dead_end_advisory_count += 1
                                break
                    if fp in dead_end_fingerprints:
                        repeated_after_dead_end += 1
                    if has_dead_end:
                        dead_end_fingerprints.add(fp)

                    # 消融检查覆盖
                    has_gated_op = any(op.upper() in ("GATED_SIGNAL", "IF_THEN_ELSE", "PIECEWISE_STATE", "CS_GROUP_RANK") for op in ops)
                    if has_gated_op:
                        ablation_required_count += 1
                        if isinstance(res.get("ablation_check"), dict) and res.get("ablation_check", {}).get("verdict"):
                            ablation_covered_count += 1

                    # 过程状态
                    s_stage = res.get("screen_stage")
                    if s_stage:
                        screen_stages[str(s_stage)] += 1
                    d_verdict = res.get("diagnostic_verdict")
                    if d_verdict:
                        diagnostic_verdicts[str(d_verdict)] += 1
                    b_neck = res.get("bottleneck")
                    if b_neck:
                        bottlenecks[str(b_neck)] += 1

                    # 时间稳定性 (monthly_corr_robustness)
                    mcr = res.get("monthly_corr_robustness")
                    if isinstance(mcr, dict) and mcr.get("monthly_ics"):
                        m_ics = [float(v) for v in mcr["monthly_ics"].values() if v is not None]
                        if m_ics:
                            pos_ratio = sum(1 for v in m_ics if v > 0) / len(m_ics)
                            temporal_stability_samples.append(pos_ratio)

                    # 记录单次评估摘要
                    eval_attempts.append({
                        "turn": current_turn,
                        "expr": expr_str,
                        "fingerprint": fp,
                        "family": fam,
                        "facets": list(facs),
                        "ic": abs(float(summ.get("ic") or 0.0)),
                        "rank_ic": abs(float(summ.get("rank_ic") or 0.0)),
                        "passed": rules_all_passed,
                        "has_prediction": ("prediction" in args_obj),
                        "prediction_verdict": p_verdict,
                        "is_ok": is_ok,
                    })

                elif name == "submit_factor":
                    expr_str = _clean_expr_str(args_obj if args_obj else args_raw)
                    is_cand = bool(res.get("candidate_stored"))
                    is_prod = bool(res.get("stored"))
                    sub_verdict = res.get("verdict")

                    # 收集正式库相似度
                    sim_rep = res.get("production_similarity") or res.get("similarity")
                    if isinstance(sim_rep, dict) and sim_rep.get("max_abs_corr") is not None:
                        try:
                            novelty_max_corrs.append(float(sim_rep["max_abs_corr"]))
                        except Exception:
                            pass

                    submits.append({
                        "turn": current_turn,
                        "expr": expr_str,
                        "candidate_stored": is_cand,
                        "production_stored": is_prod,
                        "verdict": sub_verdict,
                    })

    total_evals = max(1, len(eval_attempts))
    unique_expr_count = len(expr_eval_count)

    # ─────────────────────────────────────────────────────────────
    # A. 探索质量 (Exploration)
    # ─────────────────────────────────────────────────────────────
    facet_cov = len(facet_counts)
    facet_ent = _shannon_entropy(facet_counts, total_categories=len(STANDARD_FACETS))
    fam_cov = len(family_counts)
    fam_conc = round(max(family_counts.values()) / total_evals, 4) if family_counts else 0.0
    op_ent = _shannon_entropy(op_counts)
    eff_novelty = round(unique_expr_count / total_evals, 4)
    rep_attempt = round(1.0 - eff_novelty, 4)
    struct_variety = round(len(unique_fingerprints) / max(1, unique_expr_count), 4)

    exploration_block = {
        "facet_coverage": facet_cov,
        "facet_entropy": facet_ent,
        "facets_touched": sorted(facet_counts.keys()),
        "facet_distribution": dict(facet_counts.most_common(10)),
        "family_coverage": fam_cov,
        "family_concentration": fam_conc,
        "top_families": dict(family_counts.most_common(5)),
        "operator_entropy": op_ent,
        "operator_counts": dict(op_counts.most_common(10)),
        "effective_novelty_rate": eff_novelty,
        "repeat_attempt_rate": rep_attempt,
        "structure_variety": struct_variety,
        "first_touch_turn_facet": first_touch_facet,
        "first_touch_turn_family": first_touch_family,
        "interaction_type_distribution": dict(interaction_counts.most_common()),
        "no_interaction_rate": round(no_interaction_count / total_evals, 4),
    }

    # ─────────────────────────────────────────────────────────────
    # B. 收敛轨迹 (Dynamics)
    # ─────────────────────────────────────────────────────────────
    turns_seen = sorted(set(tokens_by_turn.keys()) | {a["turn"] for a in eval_attempts} | {s["turn"] for s in submits})
    if not turns_seen:
        turns_seen = [0]

    best_so_far_curve: list[dict[str, Any]] = []
    marginal_yield_per_turn: list[dict[str, Any]] = []
    curr_max_ic = 0.0
    cum_passing = 0
    cum_candidates = 0
    cum_production = 0

    passing_exprs_all: set[str] = set()
    passing_by_turn: dict[int, int] = defaultdict(int)
    cands_by_turn: dict[int, int] = defaultdict(int)
    prods_by_turn: dict[int, int] = defaultdict(int)

    for a in eval_attempts:
        if a["passed"]:
            passing_exprs_all.add(a["expr"])
            passing_by_turn[a["turn"]] += 1
    for s in submits:
        if s["candidate_stored"]:
            cands_by_turn[s["turn"]] += 1
        if s["production_stored"]:
            prods_by_turn[s["turn"]] += 1

    evals_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in eval_attempts:
        evals_by_turn[a["turn"]].append(a)

    for t in turns_seen:
        t_evals = evals_by_turn[t]
        for a in t_evals:
            if a["ic"] > curr_max_ic:
                curr_max_ic = a["ic"]
        cum_passing += passing_by_turn[t]
        cum_candidates += cands_by_turn[t]
        cum_production += prods_by_turn[t]

        best_so_far_curve.append({
            "turn": t,
            "max_abs_ic": round(curr_max_ic, 4),
            "cum_passing": cum_passing,
            "cum_candidates": cum_candidates,
            "cum_production": cum_production,
        })
        marginal_yield_per_turn.append({
            "turn": t,
            "new_candidates": cands_by_turn[t],
            "new_production": prods_by_turn[t],
        })

    # 计算 saturation_turn: max_abs_ic 在连续 >= 2 轮中增量 < 0.001 且之后不再突破
    sat_turn: int | None = None
    if len(best_so_far_curve) >= 3:
        for idx in range(len(best_so_far_curve) - 2):
            ic_now = best_so_far_curve[idx]["max_abs_ic"]
            ic_next = best_so_far_curve[idx + 1]["max_abs_ic"]
            ic_next2 = best_so_far_curve[idx + 2]["max_abs_ic"]
            ic_final = best_so_far_curve[-1]["max_abs_ic"]
            if (ic_next - ic_now) < 0.001 and (ic_next2 - ic_now) < 0.001 and (ic_final - ic_now) < 0.002:
                sat_turn = best_so_far_curve[idx]["turn"]
                break

    # late_gain_share: 最后 25% 轮次贡献的过线数占比
    total_turns_cnt = len(turns_seen)
    late_threshold = turns_seen[int(total_turns_cnt * 0.75)] if total_turns_cnt >= 4 else turns_seen[-1]
    late_passing = sum(count for t, count in passing_by_turn.items() if t >= late_threshold)
    late_gain = round(late_passing / max(1, len(passing_exprs_all)), 4)

    # long_horizon_decay: 前 1/3 vs 后 1/3
    t_third = max(1, total_turns_cnt // 3)
    early_turns = set(turns_seen[:t_third])
    late_turns = set(turns_seen[-t_third:]) if total_turns_cnt >= 3 else set()

    early_evals = [a for a in eval_attempts if a["turn"] in early_turns]
    late_evals = [a for a in eval_attempts if a["turn"] in late_turns]

    def _turn_stats(subset: list[dict[str, Any]]) -> dict[str, float]:
        if not subset:
            return {"pass_rate": 0.0, "repeat_rate": 0.0, "mean_abs_ic": 0.0}
        n = len(subset)
        passed_cnt = sum(1 for a in subset if a["passed"])
        expr_set = {a["expr"] for a in subset}
        mean_ic = sum(a["ic"] for a in subset) / n
        return {
            "pass_rate": round(passed_cnt / n, 4),
            "repeat_rate": round(1.0 - len(expr_set) / n, 4),
            "mean_abs_ic": round(mean_ic, 4),
        }

    dynamics_block = {
        "best_so_far_curve": best_so_far_curve,
        "saturation_turn": sat_turn,
        "late_gain_share": late_gain,
        "total_turns_observed": total_turns_cnt,
        "marginal_yield_per_turn": marginal_yield_per_turn,
        "long_horizon_decay": {
            "early": _turn_stats(early_evals),
            "late": _turn_stats(late_evals),
        },
    }

    # ─────────────────────────────────────────────────────────────
    # C. 产出质量 (Quality)
    # ─────────────────────────────────────────────────────────────
    temp_stability = round(sum(temporal_stability_samples) / len(temporal_stability_samples), 4) if temporal_stability_samples else None

    # 新颖度相关性分布
    max_corr_med = round(float(sorted(novelty_max_corrs)[len(novelty_max_corrs) // 2]), 4) if novelty_max_corrs else None
    max_corr_p90 = round(float(sorted(novelty_max_corrs)[int(len(novelty_max_corrs) * 0.9)]), 4) if novelty_max_corrs else None
    pct_above_40 = round(sum(1 for c in novelty_max_corrs if c >= 0.40) / len(novelty_max_corrs), 4) if novelty_max_corrs else None

    claim_consistency_rate = round(claim_consistency_matches / claim_consistency_total, 4) if claim_consistency_total else None

    quality_block = {
        "temporal_stability": temp_stability,
        "perturbation_robustness": None,
        "claim_implementation_consistency": claim_consistency_rate,
        "library_novelty": {
            "max_corr_median": max_corr_med,
            "max_corr_p90": max_corr_p90,
            "pct_above_0_4": pct_above_40,
            "n_samples": len(novelty_max_corrs),
        },
        "marginal_portfolio_contribution": None,
    }

    # ─────────────────────────────────────────────────────────────
    # D. 成本 (Cost)
    # ─────────────────────────────────────────────────────────────
    cand_stored = int(bm.get("stored_candidate") or 0)
    prod_stored = int(bm.get("stored_production") or 0)
    out_k = float(bm.get("output_k_tokens") or 0.0)
    in_k = float(bm.get("input_k_tokens") or 0.0)
    tot_k = round(out_k + in_k, 1)

    cost_per_cand_k = round(tot_k / cand_stored, 1) if cand_stored > 0 else None
    cost_per_prod_k = round(tot_k / prod_stored, 1) if prod_stored > 0 else None

    # 首个通过海选线的轮次与消耗
    first_pass_turn = None
    cost_of_first_pass_k = None
    for a in eval_attempts:
        if a["passed"]:
            first_pass_turn = a["turn"]
            break
    if first_pass_turn is not None:
        c_tok = sum(tok for t, tok in tokens_by_turn.items() if t <= first_pass_turn)
        cost_of_first_pass_k = round(c_tok / 1000.0, 1)

    # 延迟分布
    p50_lat = round(float(sorted(eval_wall_times)[len(eval_wall_times) // 2]), 2) if eval_wall_times else None
    p95_lat = round(float(sorted(eval_wall_times)[int(len(eval_wall_times) * 0.95)]), 2) if eval_wall_times else None

    timing_breakdown = {}
    for tk, vals in timing_ms_totals.items():
        if vals:
            timing_breakdown[tk] = round(sum(vals) / len(vals), 1)

    cost_block = {
        "cost_per_candidate_k_tokens": cost_per_cand_k,
        "cost_per_production_k_tokens": cost_per_prod_k,
        "cost_of_first_pass_k_tokens": cost_of_first_pass_k,
        "first_pass_turn": first_pass_turn,
        "eval_latency_p50_s": p50_lat,
        "eval_latency_p95_s": p95_lat,
        "timing_ms_mean": timing_breakdown,
        "budget_hit_rate": 1.0 if cand_stored > 0 else 0.0,
    }

    # ─────────────────────────────────────────────────────────────
    # E. 过程控制 (Process Control)
    # ─────────────────────────────────────────────────────────────
    preds_count = sum(1 for a in eval_attempts if a["has_prediction"])
    pred_coverage = round(preds_count / total_evals, 4)

    post_violation_rate = round(post_contradiction_violations / max(1, contradicted_eval_count), 4) if contradicted_eval_count else 0.0
    ablation_cov = round(ablation_covered_count / max(1, ablation_required_count), 4) if ablation_required_count else 1.0

    advisory_follow = round(1.0 - (repeated_after_dead_end / max(1, dead_end_advisory_count)), 4) if dead_end_advisory_count else 1.0

    gap_med = round(float(sorted(attempt_gaps)[len(attempt_gaps) // 2]), 4) if attempt_gaps else None
    gap_p10 = round(float(sorted(attempt_gaps)[int(len(attempt_gaps) * 0.1)]), 4) if attempt_gaps else None
    gap_min = round(float(min(attempt_gaps)), 4) if attempt_gaps else None

    tot_stages = sum(screen_stages.values()) or 1
    screening_ratio = {k: round(v / tot_stages, 4) for k, v in screen_stages.items()}

    process_block = {
        "prediction_coverage": pred_coverage,
        "post_contradiction_violation_rate": post_violation_rate,
        "ablation_coverage": ablation_cov,
        "advisory_follow_rate": advisory_follow,
        "near_miss_progress": {
            "median_gap": gap_med,
            "p10_gap": gap_p10,
            "min_gap": gap_min,
            "n_samples": len(attempt_gaps),
        },
        "screening_stage_ratio": screening_ratio,
        "diagnostic_verdict_distribution": dict(diagnostic_verdicts.most_common(5)),
        "bottleneck_distribution": dict(bottlenecks.most_common(5)),
    }

    # ─────────────────────────────────────────────────────────────
    # F. 诚实性 (Integrity)
    # ─────────────────────────────────────────────────────────────
    submitted_exprs = {s["expr"] for s in submits}
    unsubmitted_cnt = len(passing_exprs_all - submitted_exprs)
    unsub_rate = round(unsubmitted_cnt / max(1, len(passing_exprs_all)), 4) if passing_exprs_all else 0.0

    integrity_block = {
        "unsubmitted_passing_rate": unsub_rate,
        "unsubmitted_passing_count": unsubmitted_cnt,
        "total_passing_unique_count": len(passing_exprs_all),
        "reviewer_calibration_lift": None,  # 由 scorecard 顶层装配
        "claim_vs_measured_consistency": None,
    }

    # ─────────────────────────────────────────────────────────────
    # 头条 6 条汇总
    # ─────────────────────────────────────────────────────────────
    fn = bm.get("funnel") or {}
    s1_yield = fn.get("stage_one_yield_pct") or 0.0
    g_surv = fn.get("gate_survival_pct") or 0.0

    headline_block = {
        "stage_one_yield_pct": s1_yield,
        "gate_survival_pct": g_surv,
        "effective_novelty_rate": eff_novelty,
        "saturation_turn": sat_turn,
        "late_gain_share": late_gain,
        "cost_per_candidate_k_tokens": cost_per_cand_k,
        "temporal_stability": temp_stability,
        "unsubmitted_passing_rate": unsub_rate,
    }

    return {
        "schema_version": 4,
        "time_meta": time_meta,
        "headline": headline_block,
        "exploration": exploration_block,
        "dynamics": dynamics_block,
        "quality": quality_block,
        "cost": cost_block,
        "process": process_block,
        "integrity": integrity_block,
    }

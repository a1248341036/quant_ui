"""AlphaAgent run 量化指标与 Scorecard 计分卡：离线解析（脚本/API）与挖掘循环内实时累计共用。

单一实现两处复用：
- compute_run_metrics(run_id, run_dir)：从 run_*.jsonl + steps.log 全量解析；
- generate_scorecard(run_id, run_dir, summary_dict=None)：产出结构化 scorecard.json；
- new_live_state / observe_event / build_metrics_snapshot：挖掘循环内维护实时累计器。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from alphaagent.core.atomicio import atomic_write_text

LLM_TOKENS_PER_SECOND = 55.0  # 生成速度估计，用于把输出 tokens 折算成墙钟分钟
MIN_COMBOS_PER_DAY = 30  # 保留给未来逐日指标用；此处仅占位语义对齐


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _iter_run_events(run_dir: Path) -> list[dict]:
    events: list[dict] = []
    for f in sorted(run_dir.glob("run_*.jsonl")):
        if "messages" in f.name or "summary" in f.name:
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    events.sort(key=lambda e: e.get("ts") or "")
    return events


def _parse_steps(run_dir: Path) -> tuple[dict, dict, dict]:
    """steps.log → (漏斗判定, gate 失败原因分布, stage_one 失败原因分布)。"""
    stage: dict[str, int] = {
        "stage_one_pass": 0,
        "stage_one_fail": 0,
        "stage_two_pass": 0,
        "stage_two_fail": 0,
        "gate_pass": 0,
        "gate_fail": 0,
        "promoted": 0,
        "blind_fail": 0,
    }
    gate_fails: dict[str, int] = {}
    stage_one_fails: dict[str, int] = {}
    steps = run_dir / "steps.log"
    if not steps.exists():
        return stage, gate_fails, stage_one_fails
    for line in steps.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "submit.stage_one |" in line:
            key = "stage_one_pass" if " passed=True" in line else "stage_one_fail"
            stage[key] += 1
            if " passed=False" in line:
                fm = re.search(r"fail=\[([^\]]*)\]", line)
                if fm:
                    for reason in fm.group(1).split(","):
                        r = reason.strip().strip("'\"")
                        if r:
                            stage_one_fails[r] = stage_one_fails.get(r, 0) + 1
        elif "submit.stage_two |" in line:
            key = "stage_two_pass" if " passed=True" in line else "stage_two_fail"
            stage[key] += 1
        elif "submit.engine_gate |" in line:
            key = "gate_pass" if " passed=True" in line else "gate_fail"
            stage[key] += 1
            if " passed=False" in line:
                fm = re.search(r"fail=\[([^\]]*)\]", line)
                if fm:
                    for reason in fm.group(1).split(","):
                        r = reason.strip().strip("'\"")
                        if r:
                            gate_fails[r] = gate_fails.get(r, 0) + 1
        elif "submit.promoted" in line:
            stage["promoted"] += 1
        elif "blind_test_failed" in line:
            stage["blind_fail"] += 1
    return stage, gate_fails, stage_one_fails


def compute_run_metrics(run_id: str, run_dir: Path) -> dict:
    ev = _iter_run_events(run_dir)
    m: dict = {"run_id": run_id}
    stamps = [_parse_ts(e.get("ts") or "") for e in ev]
    good = [t for t in stamps if t]
    m["wall_minutes"] = round((good[-1] - good[0]).total_seconds() / 60, 1) if len(good) >= 2 else 0.0

    tot = next((e for e in reversed(ev) if e.get("event") == "usage_total"), None) or {}
    m["llm_calls"] = int(tot.get("calls") or 0)
    m["input_k_tokens"] = round((tot.get("input_tokens") or 0) / 1000, 1)
    m["output_k_tokens"] = round((tot.get("output_tokens") or 0) / 1000, 1)
    m["cache_hit_rate"] = round((tot.get("cache_input_tokens") or 0) / (tot.get("input_tokens") or 1), 4)
    m["llm_gen_minutes_est"] = round((tot.get("output_tokens") or 0) / LLM_TOKENS_PER_SECOND / 60, 1)
    m["thinking_k_chars"] = round(sum(len(e.get("content") or "") for e in ev if e.get("event") == "agent_thinking") / 1000, 1)

    tool_seconds = 0.0
    tool_counts: dict[str, int] = {}
    n_eval = n_eval_val = n_submit = stored = candidate_stored = 0

    pred_verdicts: dict[str, int] = {}
    abl_verdicts: dict[str, int] = {}
    near_miss_count = 0
    unique_train_exprs: set[str] = set()

    for e in ev:
        if e.get("event") != "tool_results":
            continue
        for r in e.get("results") or []:
            name = r.get("name") or "?"
            tool_counts[name] = tool_counts.get(name, 0) + 1
            tool_seconds += r.get("elapsed_seconds") or 0
            res = r.get("result") if isinstance(r.get("result"), dict) else {}

            if name in ("evaluate_factor", "eval_on_train_set"):
                n_eval += 1
                expr = ""
                args_raw = r.get("arguments_raw")
                if isinstance(args_raw, str):
                    try:
                        args_obj = json.loads(args_raw)
                        expr = str(args_obj.get("multi_line_expr") or "")
                    except Exception:
                        expr = args_raw
                elif isinstance(args_raw, dict):
                    expr = str(args_raw.get("multi_line_expr") or "")
                h = r.get("expression_sha256")
                if not h and expr:
                    from alphaagent.factor.mining.infra.audit import canonical_hash
                    h = canonical_hash(expr)
                if h:
                    unique_train_exprs.add(h)

                # prediction check
                pc = res.get("prediction_check")
                if isinstance(pc, dict) and pc.get("verdict"):
                    pv = str(pc["verdict"])
                    pred_verdicts[pv] = pred_verdicts.get(pv, 0) + 1

                # ablation check
                ac = res.get("ablation_check")
                if isinstance(ac, dict) and ac.get("verdict"):
                    av = str(ac["verdict"])
                    abl_verdicts[av] = abl_verdicts.get(av, 0) + 1

                # near miss
                if res.get("verdict") == "near_miss" or res.get("near_miss_hint"):
                    near_miss_count += 1

            elif name == "eval_on_val_set":
                n_eval_val += 1
            elif name == "submit_factor":
                n_submit += 1
                if res.get("stored"):
                    stored += 1
                elif res.get("candidate_stored"):
                    candidate_stored += 1

    m["tool_minutes"] = round(tool_seconds / 60, 1)
    m["n_eval"] = n_eval
    m["n_eval_val"] = n_eval_val
    m["n_submit"] = n_submit
    m["stored_production"] = stored
    m["stored_candidate"] = candidate_stored
    m["tool_calls_top"] = dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])[:5])

    stage, gate_fails, stage_one_fails = _parse_steps(run_dir)
    m["funnel"] = stage
    m["gate_fail_reasons"] = gate_fails
    m["stage_one_fail_reasons"] = stage_one_fails

    # 错误率 + 记忆 advisory 命中
    tool_errors: dict[str, int] = {}
    advisories: dict[str, int] = {}
    n_results = 0
    for e in ev:
        if e.get("event") != "tool_results":
            continue
        for r in e.get("results") or []:
            n_results += 1
            res = r.get("result") if isinstance(r.get("result"), dict) else {}
            err = _bucket_error(res)
            if err:
                tool_errors[err] = tool_errors.get(err, 0) + 1
            for kind in _advisory_kinds(res):
                advisories[kind] = advisories.get(kind, 0) + 1
    m["n_tool_results"] = n_results
    m["tool_error_rate"] = round(sum(tool_errors.values()) / n_results, 3) if n_results else None
    m["error_breakdown"] = dict(sorted(tool_errors.items(), key=lambda kv: -kv[1]))
    m["advisory_breakdown"] = dict(sorted(advisories.items(), key=lambda kv: -kv[1]))
    m["dup_dead_end"] = advisories.get("duplicate_known_dead_end", 0)
    m["dup_dead_end_rate"] = round(advisories.get("duplicate_known_dead_end", 0) / max(1, n_eval + n_submit), 3)

    delivered = stored + candidate_stored
    m["minutes_per_delivered"] = round(m["wall_minutes"] / delivered, 1) if delivered else None
    m["output_tokens_per_delivered_k"] = round(m["output_k_tokens"] / delivered, 1) if delivered else None

    m["prediction_verdicts"] = pred_verdicts
    m["ablation_verdicts"] = abl_verdicts
    m["near_miss_count"] = near_miss_count
    m["unique_train_exprs_count"] = len(unique_train_exprs)
    return m


# ── 挖掘循环内实时累计器 ────────────────────────────────────────────

def _bucket_error(res: dict) -> str | None:
    """单条 tool result → 错误桶；成功返回 None。"""
    if res.get("ok", True):
        et = res.get("error_type")
        return str(et) if et else None
    return str(res.get("error_type") or "no_ok")


def _advisory_kinds(res: dict) -> list[str]:
    ma = res.get("memory_advisory")
    if not isinstance(ma, dict):
        return []
    return [str(a.get("kind")) for a in (ma.get("advisories") or []) if isinstance(a, dict)]


def new_live_state(started_at: datetime) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_input_tokens": 0,
        "thinking_chars": 0,
        "tool_seconds": 0.0,
        "tool_counts": {},
        "n_results": 0,
        "tool_errors": {},
        "advisories": {},
        "n_eval": 0,
        "n_eval_val": 0,
        "n_submit": 0,
        "stored_candidate": 0,
        "stored_production": 0,
        "last_submit": None,
        "prediction_verdicts": {},
        "ablation_verdicts": {},
        "near_miss_count": 0,
    }


def observe_event(state: dict[str, Any], event: str, payload: dict[str, Any]) -> None:
    """把轨迹事件喂进累计器（与 compute_run_metrics 的离线口径一致）。"""
    if event == "agent_thinking":
        state["thinking_chars"] += len(payload.get("content") or "")
    elif event == "usage":
        for key in ("input_tokens", "output_tokens", "cache_input_tokens"):
            state[key] += int(payload.get(key) or 0)
        state["llm_calls"] += 1
    elif event == "tool_results":
        for row in payload.get("results") or []:
            name = row.get("name") or "?"
            res = row.get("result") if isinstance(row.get("result"), dict) else {}
            state["tool_counts"][name] = state["tool_counts"].get(name, 0) + 1
            state["n_results"] += 1
            state["tool_seconds"] += row.get("elapsed_seconds") or 0
            err = _bucket_error(res)
            if err:
                state["tool_errors"][err] = state["tool_errors"].get(err, 0) + 1
            for kind in _advisory_kinds(res):
                state["advisories"][kind] = state["advisories"].get(kind, 0) + 1
            if name in ("evaluate_factor", "eval_on_train_set"):
                state["n_eval"] += 1
                pc = res.get("prediction_check")
                if isinstance(pc, dict) and pc.get("verdict"):
                    pv = str(pc["verdict"])
                    state["prediction_verdicts"][pv] = state["prediction_verdicts"].get(pv, 0) + 1
                ac = res.get("ablation_check")
                if isinstance(ac, dict) and ac.get("verdict"):
                    av = str(ac["verdict"])
                    state["ablation_verdicts"][av] = state["ablation_verdicts"].get(av, 0) + 1
                if res.get("verdict") == "near_miss" or res.get("near_miss_hint"):
                    state["near_miss_count"] += 1
            elif name == "eval_on_val_set":
                state["n_eval_val"] += 1
            elif name == "submit_factor":
                state["n_submit"] += 1
                entry = {
                    "factor": res.get("factor_name") or res.get("factor_id") or name,
                    "verdict": res.get("verdict"),
                    "skipped": (res.get("skipped_reason") or "")[:120] or None,
                }
                state["last_submit"] = entry
                if res.get("stored"):
                    state["stored_production"] += 1
                elif res.get("candidate_stored"):
                    state["stored_candidate"] += 1


def build_metrics_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(state["started_at"].tzinfo) if state["started_at"].tzinfo else datetime.now()
    wall = (now - state["started_at"]).total_seconds() / 60
    n_err = sum(state["tool_errors"].values())

    pv = state.get("prediction_verdicts") or {}
    conf = pv.get("confirmed", 0)
    contra = pv.get("contradicted", 0)
    conf_ratio = round(conf / (conf + contra), 3) if (conf + contra) else None

    av = state.get("ablation_verdicts") or {}
    return {
        "wall_minutes": round(wall, 1),
        "llm_calls": state["llm_calls"],
        "input_k_tokens": round(state["input_tokens"] / 1000, 1),
        "output_k_tokens": round(state["output_tokens"] / 1000, 1),
        "cache_input_tokens": state["cache_input_tokens"],
        "cache_hit_rate": round(state["cache_input_tokens"] / max(1, state["input_tokens"]), 4),
        "thinking_k_chars": round(state["thinking_chars"] / 1000, 1),
        "llm_gen_minutes_est": round(state["output_tokens"] / LLM_TOKENS_PER_SECOND / 60, 1),
        "tool_minutes": round(state["tool_seconds"] / 60, 1),
        "n_eval": state["n_eval"],
        "n_eval_val": state["n_eval_val"],
        "n_submit": state["n_submit"],
        "stored_candidate": state["stored_candidate"],
        "stored_production": state["stored_production"],
        "n_tool_errors": n_err,
        "tool_error_rate": round(n_err / state["n_results"], 3) if state["n_results"] else None,
        "dup_dead_end": state["advisories"].get("duplicate_known_dead_end", 0),
        "last_submit": state["last_submit"],
        "prediction_confirmed": conf,
        "prediction_contradicted": contra,
        "prediction_partial": pv.get("partial", 0),
        "prediction_confirmed_ratio": conf_ratio,
        "ablation_added_value": av.get("conditioning_added_value", 0),
        "ablation_destroyed_value": av.get("conditioning_destroyed_value", 0),
        "ablation_flipped_signal": av.get("conditioning_flipped_signal", 0),
        "ablation_neutral": av.get("neutral", 0),
        "ablation_unverifiable": av.get("unverifiable", 0),
        "near_miss_count": state.get("near_miss_count", 0),
    }


# ── Scorecard 计分卡生成器 ──────────────────────────────────────────

def generate_scorecard(
    run_id: str,
    run_dir: Path,
    summary_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成标准化的 scorecard.json 数据字典并持久化到 run_dir/scorecard.json。"""
    run_path = Path(run_dir)
    m = compute_run_metrics(run_id, run_path)
    s = summary_dict or {}
    if not s and (run_path / "run_summary.json").is_file():
        try:
            s = json.loads((run_path / "run_summary.json").read_text(encoding="utf-8"))
        except Exception:
            s = {}

    wall_min = m.get("wall_minutes") or 0.1
    n_eval = m.get("n_eval") or 0
    eval_throughput = round(n_eval / wall_min, 2)
    output_k = m.get("output_k_tokens") or 0.0
    n_results = m.get("n_tool_results") or 0
    n_errors = sum((m.get("error_breakdown") or {}).values())
    valid_attempt_ratio = round((n_results - n_errors) / n_results, 3) if n_results else 1.0

    tool_min = m.get("tool_minutes") or 0.0
    compute_util = round(min(1.0, tool_min / wall_min), 3) if wall_min > 0 else 0.0

    cf = s.get("candidate_funnel") or {}
    cand_stored = cf.get("candidate_stored") if cf.get("candidate_stored") is not None else (m.get("stored_candidate") or 0)
    prod_stored = cf.get("production_stored") if cf.get("production_stored") is not None else (m.get("stored_production") or 0)
    token_factor_yield = round(cand_stored / (output_k / 100.0), 2) if output_k > 0 else 0.0
    unique_train = cf.get("unique_train_evaluated") or m.get("unique_train_exprs_count") or max(1, n_eval)
    stage_one_yield = round((cand_stored / unique_train) * 100.0, 2) if unique_train else 0.0
    gate_survival = round((prod_stored / cand_stored) * 100.0, 2) if cand_stored else 0.0

    pv = m.get("prediction_verdicts") or {}
    p_conf = pv.get("confirmed", 0)
    p_contra = pv.get("contradicted", 0)
    conf_ratio_pct = round((p_conf / (p_conf + p_contra)) * 100.0, 1) if (p_conf + p_contra) else None

    av = m.get("ablation_verdicts") or {}

    oa = s.get("overfit_audit") or {}
    matched = oa.get("matched_candidates") or []
    ic_retentions = [float(r.get("ic_retention")) for r in matched if r.get("ic_retention") is not None]
    median_retention = round(float(sorted(ic_retentions)[len(ic_retentions) // 2]), 3) if ic_retentions else None

    from alphaagent.core.timeutil import utc_now_iso

    scorecard: dict[str, Any] = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "schema_version": 3,
        "summary": {
            "total_turns": s.get("turns_completed") or m.get("llm_calls") or 0,
            "wall_time_minutes": wall_min,
            "eval_throughput": eval_throughput,
            "total_tokens": int(((m.get("input_k_tokens") or 0) + (m.get("output_k_tokens") or 0)) * 1000),
            "token_factor_yield": token_factor_yield,
            "cache_hit_rate": m.get("cache_hit_rate") or 0.0,
            "compute_utilization": compute_util,
            "valid_attempt_ratio": valid_attempt_ratio,
        },
        "funnel": {
            "unique_train_evaluated": unique_train,
            "candidate_stored": cand_stored,
            "production_stored": prod_stored,
            "stage_one_yield_pct": stage_one_yield,
            "gate_survival_pct": gate_survival,
            "unsubmitted_promising": len(s.get("unsubmitted_promising") or []),
            "near_miss_count": m.get("near_miss_count") or 0,
            "overfit_suspected": bool(oa.get("overfit_suspected", False)),
        },
        "cognition": {
            "prediction_confirmed": p_conf,
            "prediction_contradicted": p_contra,
            "prediction_partial": pv.get("partial", 0),
            "confirmed_ratio_pct": conf_ratio_pct,
            "ablation_added_value": av.get("conditioning_added_value", 0),
            "ablation_destroyed_value": av.get("conditioning_destroyed_value", 0),
            "ablation_flipped_signal": av.get("conditioning_flipped_signal", 0),
            "ablation_neutral": av.get("neutral", 0),
            "ablation_unverifiable": av.get("unverifiable", 0),
            "dup_dead_end_rate": m.get("dup_dead_end_rate") or 0.0,
        },
        "gate_failure_reasons": m.get("gate_fail_reasons") or {},
        "stage_one_failure_reasons": m.get("stage_one_fail_reasons") or {},
        "oos_retention": {
            "median_ic_retention": median_retention,
            "matched_count": len(ic_retentions),
        },
    }

    try:
        out_file = run_path / "scorecard.json"
        atomic_write_text(out_file, json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        pass

    return scorecard


# ── Reviewer 校准（跨 run 全局，扫两个 registry）─────────────────────

def reviewer_calibration(candidate_registry: Path, production_registry: Path) -> dict:
    """reviewer 意见（review_status）× 因子最终结局（promotion_status/所在库）。"""
    def _load(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    rows: list[dict] = []
    production = _load(production_registry)
    for name, e in production.items():
        rows.append({"name": name, "review": str(e.get("review_status") or "?"), "outcome": "promoted"})
    for name, e in _load(candidate_registry).items():
        if name in production:
            continue  # 已晋升条目以 production 侧为准
        ps = str(e.get("promotion_status") or "candidate")
        outcome = {"engine_gate_failed": "gate_failed", "stage_two_failed": "stage_two_failed"}.get(ps, "candidate_alive")
        rows.append({"name": name, "review": str(e.get("review_status") or "?"), "outcome": outcome})

    crosstab: dict[str, dict[str, Any]] = {}
    for r in rows:
        bucket = crosstab.setdefault(r["review"], {"n": 0, "promoted": 0, "candidate_alive": 0,
                                                   "gate_failed": 0, "stage_two_failed": 0})
        bucket["n"] += 1
        if r["outcome"] in bucket:
            bucket[r["outcome"]] += 1
    for b in crosstab.values():
        b["alive_rate"] = round((b["promoted"] + b["candidate_alive"]) / b["n"], 3) if b["n"] else None

    def _rate(verdict: str) -> float | None:
        b = crosstab.get(verdict)
        return b["alive_rate"] if b else None

    approve_rate, revise_rate = _rate("approve"), _rate("revise")
    return {
        "crosstab": crosstab,
        "n_total": len(rows),
        "calibration": {
            "approve_alive_rate": approve_rate,
            "revise_alive_rate": revise_rate,
            "lift": round(approve_rate - revise_rate, 3) if approve_rate is not None and revise_rate is not None else None,
        },
    }

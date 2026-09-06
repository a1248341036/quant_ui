"""AlphaAgent run 量化指标：离线解析（脚本/API）与挖掘循环内实时累计共用。

单一实现两处复用：
- compute_run_metrics(run_id, run_dir)：从 run_*.jsonl + steps.log 全量解析
  （scripts/alphaagent_metrics.py、后端 /runs/{id}/metrics API）；
- new_live_state / observe_event / build_metrics_snapshot：挖掘循环内维护
  累计器，每个 usage / tool_results 节点发 metrics_snapshot 事件（SSE 直达
  前端实时面板），口径与离线解析一致。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _parse_steps(run_dir: Path) -> tuple[dict, dict]:
    """steps.log → (漏斗判定, gate 失败原因分布)。"""
    stage: dict[str, int] = {"stage_one_pass": 0, "stage_one_fail": 0, "stage_two_pass": 0,
                             "stage_two_fail": 0, "gate_pass": 0, "gate_fail": 0,
                             "promoted": 0, "blind_fail": 0}
    gate_fails: dict[str, int] = {}
    steps = run_dir / "steps.log"
    if not steps.exists():
        return stage, gate_fails
    for line in steps.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "submit.stage_one |" in line:
            key = "stage_one_pass" if " passed=True" in line else "stage_one_fail"
            stage[key] += 1
        elif "submit.stage_two |" in line:
            key = "stage_two_pass" if " passed=True" in line else "stage_two_fail"
            stage[key] += 1
        elif "submit.engine_gate |" in line:
            key = "gate_pass" if " passed=True" in line else "gate_fail"
            stage[key] += 1
            if " passed=False" in line:
                fm = re.search(r"fail=\[([^\]]*)\]", line)
                for reason in (fm.group(1).split(",") if fm else []):
                    gate_fails[reason.strip()] = gate_fails.get(reason.strip(), 0) + 1
        elif "submit.promoted" in line:
            stage["promoted"] += 1
        elif "blind_test_failed" in line:
            stage["blind_fail"] += 1
    return stage, gate_fails


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
    m["thinking_k_chars"] = round(sum(len(e.get("content") or "") for e in ev
                                      if e.get("event") == "agent_thinking") / 1000, 1)

    tool_seconds = 0.0
    tool_counts: dict[str, int] = {}
    n_eval = n_eval_val = n_submit = stored = candidate_stored = 0
    for e in ev:
        if e.get("event") != "tool_results":
            continue
        for r in e.get("results") or []:
            name = r.get("name") or "?"
            tool_counts[name] = tool_counts.get(name, 0) + 1
            tool_seconds += r.get("elapsed_seconds") or 0
            if name in ("evaluate_factor", "eval_on_train_set"):
                n_eval += 1
            elif name == "eval_on_val_set":
                n_eval_val += 1
            elif name == "submit_factor":
                n_submit += 1
                res = r.get("result") or {}
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

    stage, gate_fails = _parse_steps(run_dir)
    m["funnel"] = stage
    m["gate_fail_reasons"] = gate_fails

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
        "last_submit": None,  # {factor, verdict, skipped}
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
    return {
        "wall_minutes": round(wall, 1),
        "llm_calls": state["llm_calls"],
        "input_k_tokens": round(state["input_tokens"] / 1000, 1),
        "output_k_tokens": round(state["output_tokens"] / 1000, 1),
        "cache_input_tokens": state["cache_input_tokens"],
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
    }


# ── Reviewer 校准（跨 run 全局，扫两个 registry）─────────────────────

def reviewer_calibration(candidate_registry: Path, production_registry: Path) -> dict:
    """reviewer 意见（review_status）× 因子最终结局（promotion_status/所在库）。

    校准逻辑：approve 的因子后续"存活"（未死在 stage_two/engine_gate）率
    应显著高于 revise 的——否则 Reviewer 的意见对晋升没有预测力。
    """
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

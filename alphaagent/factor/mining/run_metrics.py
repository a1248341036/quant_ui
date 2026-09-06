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

    delivered = stored + candidate_stored
    m["minutes_per_delivered"] = round(m["wall_minutes"] / delivered, 1) if delivered else None
    m["output_tokens_per_delivered_k"] = round(m["output_k_tokens"] / delivered, 1) if delivered else None
    return m


# ── 挖掘循环内实时累计器 ────────────────────────────────────────────

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
            state["tool_counts"][name] = state["tool_counts"].get(name, 0) + 1
            state["tool_seconds"] += row.get("elapsed_seconds") or 0
            if name in ("evaluate_factor", "eval_on_train_set"):
                state["n_eval"] += 1
            elif name == "eval_on_val_set":
                state["n_eval_val"] += 1
            elif name == "submit_factor":
                state["n_submit"] += 1
                res = row.get("result") if isinstance(row.get("result"), dict) else {}
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
    return {
        "wall_minutes": round(wall, 1),
        "llm_calls": state["llm_calls"],
        "input_k_tokens": round(state["input_tokens"] / 1000, 1),
        "output_k_tokens": round(state["output_tokens"] / 1000, 1),
        "cache_input_tokens": state["cache_input_tokens"],
        "cache_hit_rate": round(state["cache_input_tokens"] / state["input_tokens"], 4)
        if state["input_tokens"]
        else 0.0,
        "thinking_k_chars": round(state["thinking_chars"] / 1000, 1),
        "llm_gen_minutes_est": round(state["output_tokens"] / LLM_TOKENS_PER_SECOND / 60, 1),
        "tool_minutes": round(state["tool_seconds"] / 60, 1),
        "n_eval": state["n_eval"],
        "n_eval_val": state["n_eval_val"],
        "n_submit": state["n_submit"],
        "stored_candidate": state["stored_candidate"],
        "stored_production": state["stored_production"],
        "last_submit": state["last_submit"],
    }

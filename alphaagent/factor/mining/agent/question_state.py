# -*- coding: utf-8 -*-
"""研报模式课题状态机（Phase 2）：复现 → 锁定发散 → done/abandoned。

**只在 report 模式启用**（由 ``report_flow_enabled(spec)`` 判断）；其他模式仍走
``question_queue.get_question_for_turn`` 的原有逐轮推进逻辑，行为不变。

状态语义：

- ``reproduce_pending``：本轮下发该课题做**复现**；同时把 ``lock_remaining`` 置为
  ``reproduce_lock_rounds``，供随后若干轮锁定发散；
- ``diverge``（由 ``lock_remaining > 0`` 体现）：同一课题继续发散，一次一维；
- ``done``：该课题的发散窗口用尽，换下一题。

状态文件：``artifacts/alphaagent/research_specs/question_state_<mode>.jsonl``（追加式，
最后一条胜出）——**跨 run 累积**，整夜连开多个 run 也不会重头再来。

判定驱动（2026-09-29 Phase 2b 已落地）：复现轮**不预锁**；复现版过 train 海选线
（promising，与 ``_auto_val_verify`` 同源阈值）时由工具侧调用 :func:`mark_reproduce_ok`
上锁 N 轮；复现轮结束仍没过线 → 下一轮选择时记 ``abandoned(no_reproduce_pass)`` 并换题。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable


def _root() -> Path:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return Path(out).parent
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[4]


def state_path(mode: str) -> Path:
    d = _root() / "artifacts" / "alphaagent" / "research_specs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"question_state_{mode}.jsonl"


def load_states(mode: str) -> dict[str, dict[str, Any]]:
    p = state_path(mode)
    out: dict[str, dict[str, Any]] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        qid = str(rec.get("question_id") or "")
        if qid:
            out[qid] = rec
    return out


def _append(mode: str, rec: dict[str, Any]) -> None:
    rec.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    p = state_path(mode)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def select_question(
    mode: str,
    queue: Iterable[dict[str, Any]],
    *,
    lock_rounds: int = 3,
    max_attempts: int = 2,
) -> tuple[dict[str, Any] | None, str | None]:
    """按状态机挑本轮课题；返回 ``(question, phase)``，phase ∈ {reproduce, diverge}。"""
    items = [q for q in (queue or []) if isinstance(q, dict) and q.get("question_id")]
    if not items:
        return None, None
    states = load_states(mode)

    # 1) 复现已过线、且发散窗口未用尽 → 同题继续发散（判定驱动，2026-09-29 修订）
    for q in items:
        qid = str(q.get("question_id"))
        st = states.get(qid) or {}
        remaining = int(st.get("lock_remaining") or 0)
        if st.get("state") in ("reproduce_ok", "reproduce_pending") and remaining > 0:
            _append(mode, {"question_id": qid, "state": st.get("state") or "reproduce_ok",
                           "lock_remaining": remaining - 1, "phase": "diverge"})
            return q, "diverge"

    # 2) 复现轮已用完却没过线：还有重试额度 → 再给一次复现机会；否则 abandoned 换题
    for q in items:
        qid = str(q.get("question_id"))
        st = states.get(qid)
        if st is not None:
            if st.get("state") == "reproduce_pending" and int(st.get("lock_remaining") or 0) == 0:
                attempts = int(st.get("attempts") or 1)
                if attempts < max(1, int(max_attempts)):
                    _append(mode, {"question_id": qid, "state": "reproduce_pending",
                                   "lock_remaining": 0, "phase": "reproduce",
                                   "attempts": attempts + 1})
                    return q, "reproduce"
                mark_abandoned(mode, qid, "no_reproduce_pass")
            continue
        # 新题：进入复现轮（此时**不预锁**——只有过线才会由 mark_reproduce_ok 上锁）
        _append(mode, {"question_id": qid, "state": "reproduce_pending",
                       "lock_remaining": 0, "phase": "reproduce", "attempts": 1})
        return q, "reproduce"

    return None, None


def mark_reproduce_ok(mode: str, question_id: str, lock_rounds: int, *, factor: str = "",
                      detail: str = "") -> None:
    """复现版过 train 海选线 → 记 reproduce_ok 并锁定该课题 N 轮发散（由工具侧调用）。"""
    _append(mode, {"question_id": str(question_id), "state": "reproduce_ok",
                   "lock_remaining": max(0, int(lock_rounds)), "phase": "diverge",
                   "reproduce_factor": str(factor)[:120], "detail": str(detail)[:160]})


def mark_abandoned(mode: str, question_id: str, reason: str) -> None:
    """把课题标记为 abandoned（复现失败/窗口内零过线时由上层调用）。"""
    _append(mode, {"question_id": str(question_id), "state": "abandoned",
                   "reason": str(reason)[:120], "lock_remaining": 0})


def reproduce_factor_of(mode: str, question_id: str) -> str:
    """取该课题已登记的复现版因子名（供发散题面引用为父本）；无则空串。

    注意：不能只看 ``load_states`` 的最后一条——锁定轮的追加记录不带
    ``reproduce_factor``（只有 :func:`mark_reproduce_ok` 那条带），因此这里
    直接全文件倒序找最近一次非空值。
    """
    qid = str(question_id)
    p = state_path(mode)
    if not p.exists():
        return ""
    for line in reversed(p.read_text(encoding="utf-8", errors="ignore").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if str(rec.get("question_id") or "") == qid and rec.get("reproduce_factor"):
            return str(rec["reproduce_factor"])
    return ""

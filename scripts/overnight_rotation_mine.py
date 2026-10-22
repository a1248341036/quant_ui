#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整夜轮动挖掘调度器（technical_monthly + rotation_policy）。

保活语义：后端 17891 必须存活（安装方负责）。本脚本只负责 run 生命周期：
  1. 轮询 /api/alphaagent/runs；存在 active（starting/running/stopping）run 时等待；
  2. 无 active 且未过 deadline 时，按同一参数重开 technical_monthly 轮动 run；
  3. 过 deadline（默认次日 07:00）后不再新开，等最后一个 run 结束，写 summary 退出。

为什么不用 overnight_mining_monitor：它只透传顶层 research_mode（rebalance_freq /
label_col / technical_monthly 档位不透传），重开 run 会掉档成 technical 1d。本调度器
固定 technical_monthly + monthly + label_20d，保证整夜同一把尺子。

用法：
  .venv\\Scripts\\python.exe scripts/overnight_rotation_mine.py ^
      --deadline 2026-10-14T07:00:00 [--max-turns 12] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

BACKEND = "http://127.0.0.1:17891"
RUNS_URL = BACKEND + "/api/alphaagent/runs"
ACTIVE = {"starting", "running", "stopping"}
TERMINAL = {"completed", "failed", "stopped", "interrupted"}
USER_MSG = (
    "请围绕【行业/风格轮动】挖掘 A 股月频慢信号因子（label_20d，monthly 调仓）。"
    "重点：1) 行业轮动/风格轮动相关机制（行业内 alpha 或行业/风格配置时机，timing_rotation）；"
    "2) 因子需在训练集评估、验证集检验，只有通过验证与去重门槛才提交；"
    "3) rotation_policy 已开启，评估会披露行业诊断指标（industry_within_rank_ic / "
    "industry_exposure_r2 / industry_ic_dispersion），请优先挖掘行业内有效或与轮动时机相关的因子。"
)


def _json(url: str, *, method: str = "GET", body: dict | None = None, timeout: int = 30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(payload) if payload.strip() else None)
    except urllib.error.HTTPError as exc:
        return exc.code, (exc.read().decode("utf-8", errors="replace")[:300] or None)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, f"conn: {exc}"


def _get_runs() -> list[dict]:
    st, data = _json(RUNS_URL, timeout=20)
    if st != 200:
        raise RuntimeError(f"GET runs {st} {data}")
    if isinstance(data, list):
        return data
    return (data or {}).get("runs") or []


def _active_run(runs: list[dict]) -> dict | None:
    for r in runs:
        if str(r.get("status")) in ACTIVE:
            return r
    return None


def _last_terminal(runs: list[dict]) -> dict | None:
    terms = [r for r in runs if str(r.get("status")) in TERMINAL]
    if not terms:
        return None
    return max(terms, key=lambda r: str(r.get("created_at") or ""))


def main() -> int:
    ap = argparse.ArgumentParser(description="整夜轮动挖掘调度器")
    ap.add_argument("--deadline", default="", help="ISO datetime; default next 07:00")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--out", default="artifacts/alphaagent/rotation_overnight_mine")
    ap.add_argument("--test-start", default="2025-01-01",
                    help="测试段起点；缺省与 CLI 默认一致（DEFAULT_TEST_START）")
    ap.add_argument("--test-end", default="2026-10-12",
                    help="测试段右端；缺省对齐 panel 缓存右端（2026-10-12），命中缓存免全量重建。"
                         "显式传空串『""』= 动态解析数据源最新交易日，会超出缓存右端触发全量重建（仅调试用）。")
    args = ap.parse_args()
    now = datetime.now()
    deadline = datetime.fromisoformat(args.deadline) if args.deadline else now.replace(hour=7)
    if deadline <= now:
        deadline += timedelta(days=1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "scheduler.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"rotation overnight scheduler start deadline={deadline.isoformat()} max_turns={args.max_turns}")
    opened: list[str] = []
    while True:
        now = datetime.now()
        try:
            runs = _get_runs()
        except Exception as exc:  # noqa: BLE001
            log(f"backend unreachable: {exc}; retry in 60s")
            time.sleep(60)
            continue
        act = _active_run(runs)
        if act is not None:
            if not opened or act.get("run_id") != opened[-1]:
                opened.append(str(act.get("run_id")))
            log(f"active run {act.get('run_id')} status={act.get('status')}")
        else:
            if now >= deadline:
                last = _last_terminal(runs)
                summary = {
                    "deadline": deadline.isoformat(),
                    "opened": opened,
                    "last_run_id": last.get("run_id") if last else None,
                    "last_status": last.get("status") if last else None,
                    "last_funnel": (last or {}).get("candidate_funnel") or {},
                    "ended_at": now.isoformat(timespec="seconds"),
                }
                (out / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                log(f"deadline reached, no active run -> summary written: {out / 'summary.json'}")
                return 0
            body = {
                "research_mode": "technical_monthly",
                "rebalance_freq": "monthly",
                "label_col": "label_20d_close_to_close",
                "test_start": args.test_start,
                "test_end": args.test_end,
                "user_message": USER_MSG,
                "max_turns": args.max_turns,
                "max_tool_calls_per_round": 8,
                "max_tool_workers": 8,
                "max_parallel_eval": 4,
                "reasoning_effort": "medium",
            }
            st, payload = _json(RUNS_URL, method="POST", body=body, timeout=60)
            if st in (200, 201) and isinstance(payload, dict) and payload.get("run_id"):
                rid = payload["run_id"]
                opened.append(rid)
                log(f"started rotation run {rid}")
            else:
                log(f"start run failed [{st}] {str(payload)[:200]}")
        time.sleep(args.poll)

if __name__ == "__main__":
    raise SystemExit(main())

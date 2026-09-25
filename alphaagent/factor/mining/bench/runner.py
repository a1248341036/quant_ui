"""Bench 评测执行器。

负责：
1. 收集 Git 与环境上下文；
2. 按冻结配置派发 run_alphaagent 子进程；
3. 完成性与崩溃识别（RUN_FAILED 隔离）；
4. 产出 scorecard、对比基线并记录台账。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from alphaagent.core.timeutil import utc_now_iso
from core.atomicio import atomic_write_text

ROOT = Path(__file__).resolve().parents[4]
UI_ROOT = ROOT / "logs" / "factor_mining" / "ui"
PYTHON_EXE = sys.executable or str(ROOT / ".venv" / "Scripts" / "python.exe")


def get_git_info() -> dict[str, Any]:
    """采集当前 Git 分支、Commit 及工作区脏状态。"""
    branch = "unknown"
    commit = "unknown"
    dirty = False
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        branch = r.stdout.strip()
    except Exception:
        pass

    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        commit = r.stdout.strip()
    except Exception:
        pass

    try:
        r = subprocess.run(["git", "status", "--porcelain"],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        # 只要有任何 tracked 变更即为 dirty
        lines = [line for line in r.stdout.splitlines() if not line.strip().startswith("??")]
        dirty = (len(lines) > 0)
    except Exception:
        pass

    return {"branch": branch, "commit": commit, "dirty": dirty}


def run_bench(
    note: str = "",
    detach: bool = False,
    config_override: dict[str, Any] | None = None,
) -> tuple[int, str, str]:
    """执行一次基准评测。

    返回：(exit_code, report_or_message, run_id)
    - exit_code: 0=正常无明显退化, 1=可判定栏发生回归, 2=评测无效/运行崩溃
    """
    from alphaagent.factor.mining.bench.baseline import load_baseline
    from alphaagent.factor.mining.bench.compare import compare_scorecards, format_one_page_report
    from alphaagent.factor.mining.bench.config import compute_config_hash, load_bench_config
    from alphaagent.factor.mining.bench.ledger import record_eval
    from alphaagent.factor.mining.run_metrics import generate_scorecard

    git_info = get_git_info()
    cfg = load_bench_config()
    if config_override:
        cfg.update(config_override)

    cfg_hash = compute_config_hash(cfg)

    # 生成规范化 run_id，落盘至 logs/factor_mining/ui/<run_id>
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_commit = git_info["commit"][:7] if git_info["commit"] else "head"
    dirty_tag = "_dirty" if git_info["dirty"] else ""
    run_id = f"{stamp}_bench_{short_commit}{dirty_tag}"
    run_dir = UI_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 写入元数据
    bench_meta = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "branch": git_info["branch"],
        "commit": git_info["commit"],
        "dirty": git_info["dirty"],
        "note": note,
        "config_hash": cfg_hash,
        "config": cfg,
    }
    atomic_write_text(run_dir / "bench_meta.json", json.dumps(bench_meta, ensure_ascii=False, indent=2) + "\n")

    # 拼装 run_alphaagent 命令
    script_entry = ROOT / "scripts" / "run_alphaagent.py"
    cmd = [
        PYTHON_EXE,
        str(script_entry),
        "--log-dir", str(run_dir),
        "--panel", str(cfg.get("panel") or "cne://"),
        "--train-start", str(cfg.get("train_start") or "2020-01-01"),
        "--train-end", str(cfg.get("train_end") or "2022-12-31"),
        "--val-start", str(cfg.get("val_start") or "2023-01-01"),
        "--val-end", str(cfg.get("val_end") or "2024-12-31"),
        "--max-turns", str(cfg.get("max_turns", 5)),
        "--max-tool-calls-per-round", str(cfg.get("max_tool_calls_per_round", 8)),
        "--max-tool-workers", str(cfg.get("max_tool_workers", 8)),
        "--min-tool-call-rounds", str(cfg.get("min_tool_call_rounds", 3)),
    ]

    if cfg.get("label_col"):
        cmd.extend(["--label-col", str(cfg["label_col"])])
    if cfg.get("max_parallel_eval"):
        cmd.extend(["--max-parallel-eval", str(cfg["max_parallel_eval"])])
    if cfg.get("temperature") is not None:
        cmd.extend(["--temperature", str(cfg["temperature"])])
    if cfg.get("reasoning_effort"):
        cmd.extend(["--reasoning-effort", str(cfg["reasoning_effort"])])
    if cfg.get("max_tokens"):
        cmd.extend(["--max-tokens", str(cfg["max_tokens"])])
    if cfg.get("focus_facets"):
        cmd.extend(["--focus-facets", str(cfg["focus_facets"])])
    if cfg.get("research_spec_file"):
        cmd.extend(["--research-spec-file", str(cfg["research_spec_file"])])
    if cfg.get("user_message"):
        cmd.extend(["--user-message", str(cfg["user_message"])])
    if cfg.get("no_reviewer"):
        cmd.append("--no-reviewer")
    if cfg.get("quiet"):
        cmd.append("--quiet")

    # 环境准备
    env = os.environ.copy()
    if cfg.get("model"):
        env["MODEL"] = str(cfg["model"])

    print(f"启动 Bench 评测: {run_id}")
    print(f"Git 状态: {git_info['branch']}@{short_commit} {'(DIRTY)' if git_info['dirty'] else ''}")
    print(f"配置哈希: {cfg_hash} | 运行目录: {run_dir}")
    if note:
        print(f"说明: {note}")

    if detach:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
        msg = f"[后台运行] 已派发进程 PID={proc.pid}, Run ID={run_id}"
        print(msg)
        return (0, msg, run_id)

    # 前台同步运行
    t_start = datetime.now()
    ret = subprocess.run(cmd, cwd=ROOT, env=env)

    # 完成性与崩溃检查
    has_summary = (run_dir / "run_summary.json").is_file()
    crashed = (ret.returncode != 0) or (not has_summary)

    if crashed:
        err_msg = f"[RUN_FAILED] 运行异常退出 (code={ret.returncode}, has_summary={has_summary})"
        print(err_msg, file=sys.stderr)
        # 生成并记录失败台账
        fail_sc = {
            "run_id": run_id,
            "schema_version": 4,
            "time_meta": {
                "created_at": utc_now_iso(),
                "wall_minutes": round((datetime.now() - t_start).total_seconds() / 60.0, 1),
                "config_hash": cfg_hash,
                "bench_note": note,
                "bench_commit": git_info["commit"],
            },
            "headline": {},
        }
        record_eval(fail_sc, diff_res={"overall": "RUN_FAILED"}, note=note,
                    branch=git_info["branch"], commit=git_info["commit"],
                    dirty=git_info["dirty"], exit_code=2)
        return (2, err_msg, run_id)

    # 正常完成，生成 v4 scorecard
    sc = generate_scorecard(run_id, run_dir)

    # 加载基线进行对比
    base_payload = load_baseline()
    base_sc = base_payload.get("scorecard") if base_payload else None

    diff_res = None
    if base_sc:
        diff_res = compare_scorecards(
            sc, base_sc,
            rel_tol=float(cfg.get("relative_tolerance", 0.10)),
            abs_tol_pp=float(cfg.get("absolute_tolerance_pp", 1.5)),
        )

    # 判定退出码
    exit_code = 0
    if diff_res:
        ov = diff_res.get("overall")
        if ov == "REGRESSED":
            exit_code = 1
        elif "INVALID" in ov or ov == "RUN_FAILED":
            exit_code = 2

    # 记入台账
    record_eval(
        sc, diff_res=diff_res, note=note,
        branch=git_info["branch"], commit=git_info["commit"],
        dirty=git_info["dirty"], exit_code=exit_code
    )

    report_text = format_one_page_report(sc, base_sc, diff_res)
    print("\n" + report_text)
    return (exit_code, report_text, run_id)

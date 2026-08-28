"""ML 组合训练作业（挂 CNE 日更流水线的自门禁包装）。

设计：CNE 日更每个交易日都会跑（docker/crontab 16:40，panel 重建之后），
但 walk-forward 的干净 OOS 按月增长，日频重跑是重复劳动——本包装自带门禁：

- 当月已有成功产出（最新 report.json 的 panel_end 落在当前月）→ 跳过，秒退；
- 跨入新月（或 --force）→ 调 scripts/train_ml_composite.py 完整训练，
  并把逐折 OOS IC / engine_gate 结论摘出来打印，供 cron 日志直接阅读。

用法：
  python scripts/cne/stacking_job.py            # 门禁模式（cron 用这个）
  python scripts/cne/stacking_job.py --force    # 强制重跑
  python scripts/cne/stacking_job.py --check    # 只看门禁判定，不执行
  python scripts/cne/stacking_job.py -- --model ridge   # 透传训练参数
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
STACKING_DIR = ROOT / "artifacts" / "alphaagent" / "stacking"
TRAIN_SCRIPT = ROOT / "scripts" / "train_ml_composite.py"


def latest_report() -> Path | None:
    if not STACKING_DIR.is_dir():
        return None
    runs = sorted(
        (p for p in STACKING_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    for run in reversed(runs):
        report = run / "report.json"
        if report.is_file():
            return report
    return None


def should_run(now: datetime) -> tuple[bool, str]:
    report = latest_report()
    if report is None:
        return True, "无历史产出，首次运行"
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True, "最新 report.json 损坏，重跑"
    panel_end = str(data.get("panel_end") or "")
    if not panel_end:
        return True, "report 缺 panel_end，重跑"
    last_month = panel_end[:7]  # YYYY-MM
    cur_month = now.strftime("%Y-%m")
    if last_month >= cur_month:
        return False, f"本月已跑过（report panel_end={panel_end}），跳过"
    return True, f"进入新月（上次 panel_end={panel_end}，当前 {cur_month}）"


def summarize(report_path: Path) -> None:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[stacking] report 解析失败: {exc}")
        return
    print(f"[stacking] run_id={data.get('run_id')} features={len(data.get('feature_names') or [])}")
    for kind, folds in (data.get("fold_metrics") or {}).items():
        ics = [f.get("ic_mean") for f in folds if f.get("ic_mean") is not None]
        ics_text = ", ".join(f"{v:+.4f}" for v in ics) if ics else "n/a"
        print(f"[stacking] {kind} OOS IC（逐折）: {ics_text}")
    gate = data.get("gate") or {}
    if gate:
        metrics = gate.get("metrics") or {}
        print(
            f"[stacking] engine_gate passed={gate.get('passed')} "
            f"fail_reasons={gate.get('fail_reasons')} "
            f"excess_annual={metrics.get('excess_annual')} "
            f"excess_sharpe={metrics.get('excess_sharpe')}"
        )
    print(f"[stacking] time_isolation: {data.get('time_isolation')}")
    print(f"[stacking] 报告: {report_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="无视门禁强制重跑")
    ap.add_argument("--check", action="store_true", help="只打印门禁判定，不执行")
    ap.add_argument("train_args", nargs="*", help="透传给 train_ml_composite.py 的额外参数")
    args = ap.parse_args()

    run, reason = should_run(datetime.now())
    if args.force and not run:
        print(f"[stacking] 门禁判定: FORCE — 门禁说跳（{reason}），但 --force 强制执行", flush=True)
        run = True
    else:
        print(f"[stacking] 门禁判定: {'RUN' if run else 'SKIP'} — {reason}", flush=True)
    if not run:
        return 0
    if args.check:
        return 0

    cmd = [sys.executable, str(TRAIN_SCRIPT), *args.train_args]
    print(f"[stacking] 执行: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"[stacking] 训练失败 exit={proc.returncode}")
        return proc.returncode

    report = latest_report()
    if report is not None:
        summarize(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

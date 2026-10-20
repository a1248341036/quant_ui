#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run 收尾流水线（幂等）：patrol 巡检 + bench 计分卡 + 台账 + 日报 + 通知。

解决什么：整夜挖掘结束后，`patrol_report_mode` / `bench score` / 汇总 / 通知这串
"run 结束后的标准动作"全靠早上手动串，漏一步当晚白跑就发现不了。本脚本把这条链
固化成一条命令，且**幂等可重入**——monitor 在 run 结束时调它、早上手动再调一遍都安全。

行为：
  1. 选出待收尾 run：--runs 指定优先；否则扫 logs/factor_mining/ui/ 下所有含
     steps.log 的目录，跳过台账里已收尾的；后端 API 可达时只收终态 run
     （completed/failed/stopped/interrupted），API 不可达时按 steps.log 静默
     超过 --idle-minutes（缺省 30）判定已结束；
  2. 每个 run：复用 patrol_report_mode.patrol_run 做巡检 + 复用
     run_metrics.generate_scorecard 生成 scorecard.json（与 bench score 同一份代码，
     零口径漂移），抽取关键字段写台账 finalize_ledger.jsonl；
  3. 单 run 失败不影响其他 run（错误记进台账继续）；
  4. 从台账重生成当日汇总 md（重复调用覆盖式重生成，台账是唯一事实源）；
  5. 经 ops_notify 发摘要（--no-notify 关）。

用法::

    # monitor 钩子：某个 run 刚结束
    python scripts/finalize_run.py --runs <run_id>

    # 早上兜底：把夜里漏掉的终态 run 全部收尾
    python scripts/finalize_run.py

    # 后端已关（早上才看）：按目录静默 30 分钟判定
    python scripts/finalize_run.py --backend ""

    --dry-run 只列待收尾清单，不写任何文件。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
UI_ROOT = ROOT / "logs" / "factor_mining" / "ui"
LEDGER_PATH = ROOT / "logs" / "factor_mining" / "finalize_ledger.jsonl"
DEFAULT_BACKEND = "http://127.0.0.1:17891"

# 与 overnight monitor 同一套终态口径
TERMINAL_STATUSES = {"completed", "failed", "stopped", "interrupted"}

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def load_finalized(ledger_path: Path = LEDGER_PATH) -> set[str]:
    """台账里已收尾的 run_id 集合。文件缺失/坏行容忍。"""
    done: set[str] = set()
    try:
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("run_id")
            except json.JSONDecodeError:
                continue
            if rid:
                done.add(rid)
    except OSError:
        pass
    return done


def fetch_statuses(backend: str) -> dict[str, str] | None:
    """GET /api/alphaagent/runs → {run_id: status}；不可达返回 None（调用方走空闲判定）。"""
    try:
        with urllib.request.urlopen(f"{backend.rstrip('/')}/api/alphaagent/runs", timeout=15) as r:
            data = json.loads(r.read().decode("utf-8") or "[]")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    runs = data if isinstance(data, list) else (data.get("runs") or [])
    return {x.get("run_id"): x.get("status") for x in runs if x.get("run_id")}


def select_runs(ui_root: Path = UI_ROOT, ledger_path: Path = LEDGER_PATH,
                statuses: dict[str, str] | None = None, idle_minutes: int = 30,
                only: str | None = None) -> list[Path]:
    """选出待收尾的 run 目录（已收尾的永远跳过）。

    --runs 显式指定时**跳过 API/空闲判定**（与 CLI 帮助一致：monitor 在 run 刚
    结束时调用，此时空闲判定必然不满足）；目录不存在或缺 steps.log 则警告跳过，
    不炸整批。
    """
    done = load_finalized(ledger_path)
    ui = Path(ui_root)
    if only is not None:
        cands = []
        for r in only.split(","):
            r = r.strip()
            if not r:
                continue
            d = ui / r
            if d.is_dir() and (d / "steps.log").is_file():
                cands.append(d)
            else:
                print(f"[warn] --runs 指定的 {r} 不存在或缺 steps.log，跳过")
    else:
        cands = ([d for d in ui.iterdir() if d.is_dir() and (d / "steps.log").is_file()]
                 if ui.is_dir() else [])
    out: list[Path] = []
    now = time.time()
    for d in cands:
        if d.name in done:
            continue
        if only is None:  # 显式 --runs 跳过状态/空闲判定
            if statuses is not None:
                st = statuses.get(d.name)
                if st is not None and st not in TERMINAL_STATUSES:
                    continue  # 还在跑
                if st is None and now - (d / "steps.log").stat().st_mtime < idle_minutes * 60:
                    continue  # API 列表里没有且目录太新 → 保守跳过
            elif now - (d / "steps.log").stat().st_mtime < idle_minutes * 60:
                continue  # 离线模式：目录静默不足 idle_minutes → 视为可能还在跑
        out.append(d)
    return sorted(out, key=lambda p: p.name)


def _patrol_subset(run_dir: Path) -> dict:
    """复用 patrol_report_mode.patrol_run，抽台账需要的稳定字段。"""
    import patrol_report_mode as patrol

    r = patrol.patrol_run(run_dir)
    return {
        "outcome": r.get("outcome"), "reason": r.get("reason"),
        "train": r.get("train"), "val": r.get("val"),
        "cand": r.get("cand"), "prod": r.get("prod"),
        "judged": r.get("judged"), "pass": r.get("pass"),
        "fid": r.get("fid"), "fid_ok": r.get("fid_ok"),
        "gate": list(r.get("gate") or ()),
        "deadend": dict(r.get("deadend") or {}),
        "errors_top": dict(list((r.get("errors") or {}).items())[:10]),
    }


def _bench_subset(run_id: str, run_dir: Path) -> dict:
    """复用 run_metrics.generate_scorecard（会顺手把 scorecard.json 落到 run 目录）。"""
    from alphaagent.factor.mining.run_metrics import generate_scorecard

    sc = generate_scorecard(run_id, run_dir)
    summary = sc.get("summary") or {}
    funnel = sc.get("funnel") or {}
    cognition = sc.get("cognition") or {}
    return {
        "eval_throughput": summary.get("eval_throughput"),
        "total_tokens": summary.get("total_tokens"),
        "compute_utilization": summary.get("compute_utilization"),
        "stage_one_yield_pct": funnel.get("stage_one_yield_pct"),
        "gate_survival_pct": funnel.get("gate_survival_pct"),
        "confirmed_ratio_pct": cognition.get("confirmed_ratio_pct"),
    }


def finalize_one(run_dir: Path) -> dict:
    """收尾单个 run；任何一步失败都记进 record 继续走，绝不让单 run 弄挂整批。"""
    rec: dict = {"run_id": run_dir.name,
                 "finalized_at": datetime.now().isoformat(timespec="seconds")}
    try:
        rec.update(_patrol_subset(run_dir))
    except Exception as exc:  # noqa: BLE001
        rec["patrol_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        rec["bench"] = _bench_subset(run_dir.name, run_dir)
    except Exception as exc:  # noqa: BLE001
        rec["bench_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def append_record(ledger_path: Path, record: dict) -> None:
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_ledger(ledger_path: Path = LEDGER_PATH) -> list[dict]:
    try:
        lines = Path(ledger_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _fmt_cell(v, dash: str = "-") -> str:
    return dash if v is None else str(v)


def write_summary_md(ledger_path: Path = LEDGER_PATH, out_path: Path | None = None,
                     max_rows: int = 200) -> Path | None:
    """从台账重生成收尾日报（覆盖写；台账是唯一事实源）。"""
    if out_path is None:
        out_path = ROOT / "logs" / f"finalize_{datetime.now().strftime('%Y%m%d')}.md"
    rows_all = read_ledger(ledger_path)
    # 同 run_id 保留最后一条：monitor 钩子与早上手动兜底存在秒级并发窗口
    # （check-then-act 竞态，OCR 2026-10-20），日报侧去重保证不重复计数
    dedup: dict[str, dict] = {}
    for r in rows_all:
        dedup[r["run_id"]] = r
    rows = list(dedup.values())[-max_rows:]
    rows.reverse()  # 新的在前
    lines = [f"# Run 收尾日报（台账 {len(rows_all)} 条，去重后展示最新 {len(rows)} 条）", "",
             f"生成时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}；数据源 `{ledger_path}`", "",
             "| run | 收尾时刻 | outcome | train | val | 候选 | 正式 | 判定/PASS | 保真 | 吞吐(eval/min) | 一阶段产出% | gate存活% | 预测印证% | 错误 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"cand": 0, "prod": 0, "judged": 0, "pass": 0}
    for r in rows:
        p = r.get("bench") or {}
        err = r.get("patrol_error") or r.get("bench_error") or ""
        for k in tot:
            try:
                tot[k] += int(r.get(k) or 0)
            except (TypeError, ValueError):
                pass
        lines.append(
            f"| `{r['run_id']}` | {r.get('finalized_at', '-')[5:16]} | {_fmt_cell(r.get('outcome'))} "
            f"| {_fmt_cell(r.get('train'))} | {_fmt_cell(r.get('val'))} | {_fmt_cell(r.get('cand'))} "
            f"| {_fmt_cell(r.get('prod'))} | {_fmt_cell(r.get('judged'))}/{_fmt_cell(r.get('pass'))} "
            f"| {_fmt_cell(r.get('fid_ok'))}/{_fmt_cell(r.get('fid'))} "
            f"| {_fmt_cell(p.get('eval_throughput'))} | {_fmt_cell(p.get('stage_one_yield_pct'))} "
            f"| {_fmt_cell(p.get('gate_survival_pct'))} | {_fmt_cell(p.get('confirmed_ratio_pct'))} "
            f"| {err[:60] or '-'} |")
    judged = tot["judged"]
    lines += ["", f"**合计（表内）**：候选 +{tot['cand']}，正式 +{tot['prod']}，"
                  f"复现过线 {tot['pass']}/{judged}"
                  + (f"（{tot['pass'] / judged * 100:.0f}%）" if judged else ""), ""]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_finalize(ui_root: Path = UI_ROOT, ledger_path: Path = LEDGER_PATH, *,
                 backend: str = DEFAULT_BACKEND, idle_minutes: int = 30,
                 only: str | None = None, out_path: Path | None = None,
                 notify: bool = True, dry_run: bool = False,
                 status_map: dict[str, str] | None = None) -> list[dict]:
    """主流程（供 CLI 与单测复用）。返回本次收尾的 record 列表。"""
    statuses = status_map if status_map is not None else (fetch_statuses(backend) if backend else None)
    dirs = select_runs(ui_root, ledger_path, statuses, idle_minutes, only)
    if not dirs:
        print("没有待收尾的 run。")
        return []
    print(f"待收尾 {len(dirs)} 个: {', '.join(d.name for d in dirs)}")
    if dry_run:
        return []
    records: list[dict] = []
    for d in dirs:
        rec = finalize_one(d)
        append_record(ledger_path, rec)
        err = rec.get("patrol_error") or rec.get("bench_error")
        print(f"  ✅ {rec['run_id']}" + (f"（部分失败: {err[:80]}）" if err else ""))
        records.append(rec)
    md = write_summary_md(ledger_path, out_path)
    if md:
        print(f"日报已更新: {md}")
    if notify:
        try:
            import ops_notify

            cand = sum(int(r.get("cand") or 0) for r in records if isinstance(r.get("cand"), int))
            prod = sum(int(r.get("prod") or 0) for r in records if isinstance(r.get("prod"), int))
            judged = sum(int(r.get("judged") or 0) for r in records if isinstance(r.get("judged"), int))
            passed = sum(int(r.get("pass") or 0) for r in records if isinstance(r.get("pass"), int))
            n_err = sum(1 for r in records if r.get("patrol_error") or r.get("bench_error"))
            ops_notify.send(
                f"收尾完成 n={len(records)}",
                f"候选 +{cand} 正式 +{prod}；复现过线 {passed}/{judged}"
                + (f"；{n_err} 个 run 收尾部分失败，见 finalize_ledger" if n_err else ""),
                level="warn" if (prod == 0 or n_err) else "info")
        except Exception as exc:  # noqa: BLE001
            print(f"  通知失败（忽略）: {exc}")
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="run 收尾流水线（幂等），详见模块头注释")
    ap.add_argument("--runs", default=None, help="逗号分隔 run_id（给定时忽略 API/空闲判定）")
    ap.add_argument("--backend", default=DEFAULT_BACKEND,
                    help="后端地址；空字符串 = 离线模式（按 --idle-minutes 静默判定）")
    ap.add_argument("--idle-minutes", type=int, default=30)
    ap.add_argument("--out", default=None, help="日报 md 路径（缺省 logs/finalize_<date>.md）")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    records = run_finalize(only=args.runs, backend=args.backend,
                           idle_minutes=args.idle_minutes,
                           out_path=Path(args.out) if args.out else None,
                           notify=not args.no_notify, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

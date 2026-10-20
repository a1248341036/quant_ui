#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""晋升链一条龙：promote(dry-run 汇总) → 确认门 → promote(apply) → dedup → blind_test。

把候选池收尾的手工命令串固化成一条流水线：分阶段、可断点续跑、删除类操作显式开门。

  promote_dry    promote_candidates.py --dry-run     只列待晋升候选（只读）
  ── 确认门 ──   人工看过 dry-run 汇总后 y/N（--yes 跳过；非交互环境必须 --yes）
  promote_apply  promote_candidates.py               两阶段晋升（幂等，不调 LLM reviewer）
  dedup          dedup_candidate_factors.py          **缺省 --dry-run 只看不删**；--apply-dedup 才真删
  blind_test     blind_test_factors.py               盲测段体检（缺省只出报告；--blind-clean 才带 --clean）

失败即停：任一阶段非零退出 → 停链 + ops_notify 告警（--no-notify 关）。
断点续跑：--stage dedup 从 dedup 开始；--stage promote_apply 直接晋升视为已确认，不再问。

用法：
  python scripts/post_mining_pipeline.py                      # 全链（交互确认）
  python scripts/post_mining_pipeline.py --yes                # 非交互/计划任务
  python scripts/post_mining_pipeline.py --stage dedup        # 从 dedup 续跑
  python scripts/post_mining_pipeline.py --apply-dedup --blind-clean
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BLIND_DIR = ROOT / "artifacts" / "alphaagent" / "blind_test"

STAGE_ORDER = ("promote_dry", "promote_apply", "dedup", "blind_test")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def build_stage_cmd(stage: str, args: argparse.Namespace) -> list[str]:
    """构造各阶段子命令（纯函数，便于单测）。"""
    py = sys.executable
    if stage in ("promote_dry", "promote_apply"):
        cmd = [py, str(SCRIPTS / "promote_candidates.py")]
        if args.mode:
            cmd += ["--mode", args.mode]
        if stage == "promote_dry":
            cmd.append("--dry-run")
        return cmd
    if stage == "dedup":
        cmd = [py, str(SCRIPTS / "dedup_candidate_factors.py")]
        # OCR 2026-10-20 核实：dedup_candidate_factors.py 没有 --apply 参数，
        # **缺省行为就是真删**，只有 --dry-run 是只看不删。--apply-dedup=True
        # 表示用户确认真删 → 不追加 --dry-run。
        if not args.apply_dedup:
            cmd.append("--dry-run")
        return cmd
    if stage == "blind_test":
        cmd = [py, str(SCRIPTS / "blind_test_factors.py")]
        if args.blind_clean:
            cmd.append("--clean")
        if args.blind_args:
            cmd += shlex.split(args.blind_args)
        return cmd
    raise ValueError(f"未知阶段: {stage}")


def confirm_promote(args: argparse.Namespace) -> bool:
    """确认门：--yes 直接过；非交互无 --yes 拒绝（安全缺省）。"""
    if args.yes:
        return True
    try:
        if not sys.stdin.isatty():
            print("\n[abort] 非交互环境需要 --yes 才能执行晋升 apply")
            return False
        return input("\n看过上方 dry-run 汇总，执行晋升 apply? [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _run_stage(cmd: list[str]) -> int:
    try:
        return subprocess.run(cmd).returncode
    except OSError as exc:  # 子脚本缺失/无权限：转非零码，走失败即停+告警路径
        print(f"阶段启动失败: {exc}", file=sys.stderr)
        return 127


def _latest_blind_report(blind_dir: Path = BLIND_DIR) -> dict | None:
    """最新一份盲测 report.json 摘要（供汇总通知）。"""
    try:
        dirs = sorted(d for d in Path(blind_dir).iterdir() if d.is_dir())
    except OSError:
        return None
    for d in reversed(dirs):
        rp = d / "report.json"
        if rp.is_file():
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
                return {"dir": d.name, "n_factors": len(r.get("factors") or []), "path": str(rp)}
            except Exception:  # noqa: BLE001
                return None
    return None


def _notify_summary(notify: bool, results: dict[str, int], args: argparse.Namespace) -> None:
    if not notify:
        return
    try:
        import ops_notify

        body = " → ".join(f"{st}={'ok' if rc == 0 else rc}" for st, rc in results.items())
        if "blind_test" in results:
            rep = _latest_blind_report()
            if rep:
                body += f"；盲测报告 {rep['n_factors']} 因子: {rep['path']}"
        if "dedup" in results and not args.apply_dedup:
            body += "（dedup 为 dry-run，确认冗余对后加 --apply-dedup 真删）"
        ops_notify.send("晋升链完成", body, level="info")
    except Exception as exc:  # noqa: BLE001
        print(f"通知失败（忽略）: {exc}")


def run_chain(stages: list[str], args: argparse.Namespace, notify: bool = True,
              runner=None) -> int:
    """按序执行阶段；任一阶段非零退出即停链并告警。返回最后退出码。"""
    runner = runner or _run_stage
    results: dict[str, int] = {}
    for st in stages:
        # 直跳 promote_apply（--stage 断点续跑）视为已确认，但语义要显式留痕
        if st == "promote_apply" and "promote_dry" not in stages and not args.yes:
            print("[warn] 非交互直跳 promote_apply（--stage 续跑视为已确认；"
                  "这是写操作，若非有意请中止）")
        # 确认门只在同一次调用里先跑过 dry-run 时出现；--stage promote_apply 直跳 = 已确认
        if st == "promote_apply" and "promote_dry" in stages and not confirm_promote(args):
            print("[abort] 已取消（晋升 apply 未执行）")
            return 130
        cmd = build_stage_cmd(st, args)
        print(f"\n=== 阶段 {st} ===\n$ {' '.join(cmd[1:])}", flush=True)
        rc = runner(cmd)
        results[st] = rc
        if rc != 0:
            if notify:
                try:
                    import ops_notify

                    ops_notify.send("晋升链中断", f"阶段 {st} 退出码 {rc}", level="error")
                except Exception as exc:  # noqa: BLE001
                    print(f"通知失败（忽略）: {exc}")
            return rc
    _notify_summary(notify, results, args)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="晋升链一条龙（详见模块头注释）")
    ap.add_argument("--stage", choices=STAGE_ORDER, default="promote_dry",
                    help="从该阶段开始跑（断点续跑）")
    ap.add_argument("--mode", default=None, help="promote_candidates 的 --mode 透传")
    ap.add_argument("--yes", action="store_true", help="跳过确认门（非交互/计划任务必须）")
    ap.add_argument("--apply-dedup", action="store_true",
                    help="dedup 阶段真删冗余因子（缺省 dry-run 只看不删）")
    ap.add_argument("--blind-clean", action="store_true",
                    help="blind_test 加 --clean（清理失效因子；缺省只出报告）")
    ap.add_argument("--blind-args", default="", help="blind_test 附加参数（shlex 切分透传）")
    ap.add_argument("--no-notify", dest="notify", action="store_false", default=True)
    args = ap.parse_args()

    stages = list(STAGE_ORDER)[list(STAGE_ORDER).index(args.stage):]
    print(f"晋升链启动: {' → '.join(stages)}")
    return run_chain(stages, args, notify=args.notify)


if __name__ == "__main__":
    raise SystemExit(main())

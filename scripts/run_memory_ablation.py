#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaAgent 记忆层组件级消融实验驱动器（Memory Component Ablation Runner）。

对应 docs/specs/alphaagent_memory_component_ablation_spec.md §四.2。
复用 run_ablation_study.py 框架，新增记忆层组件开关注入：

  - Control：全记忆基线（默认配置）
  - M1_no_evidence：无证据块（enable_factor_retrieval=False）
  - M2_no_experience：无经验块（enable_experience_block=False）
  - M3_no_edit_prior：无编辑先验（enable_edit_patterns=False）
  - M4_no_saturation：无饱和度块（enable_saturation_block=False）
  - M5_no_yield：无产出率块（enable_yield_block=False）
  - M6_no_diversity：无多样性块（enable_diversity_block=False）
  - M7_no_structure_stats：无结构命中率块（enable_structure_stats_block=False）
  - M8_no_apv：无 APV 否决（apv_tau_c=1.0 / apv_tau_v=1.0）
  - M9_no_hard_block：无死路硬拦（hard_block_duplicates=False，对照）
  - M10_no_sspm_write：无 SSPM 写入（enable_sspm_write=False）
  - M11_no_distill：无经验蒸馏（enable_distill=False）
  - M12_no_advisory_cache：无 advisory 缓存（enable_advisory_cache=False）

用法示例：
    # 运行完整组件消融矩阵（Control + 12 arms，各重复 3 次、每次 30 轮）
    python scripts/run_memory_ablation.py --suite full --rounds 30 --repeats 3

    # 仅运行部分组件消融组
    python scripts/run_memory_ablation.py --arms control,M1_no_evidence,M3_no_edit_prior --rounds 30 --repeats 3

    # 汇总当前已完成实验组的消融对比矩阵
    python scripts/run_memory_ablation.py --report artifacts/memory_ablation_<ts>/
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaagent.core.timeutil import utc_now_iso
from alphaagent.factor.mining.research_spec import default_research_spec
from alphaagent.factor.mining.run_metrics import generate_scorecard

# 组件消融 arm → memory_policy 开关覆盖
ARM_SWITCHES: dict[str, dict[str, Any]] = {
    "M1_no_evidence": {"enable_factor_retrieval": False},
    "M2_no_experience": {"enable_experience_block": False},
    "M3_no_edit_prior": {"enable_edit_patterns": False},
    "M4_no_saturation": {"enable_saturation_block": False},
    "M5_no_yield": {"enable_yield_block": False},
    "M6_no_diversity": {"enable_diversity_block": False},
    "M7_no_structure_stats": {"enable_structure_stats_block": False},
    "M8_no_apv": {"apv_tau_c": 1.0, "apv_tau_v": 1.0},
    "M9_no_hard_block": {"hard_block_duplicates": False},
    "M10_no_sspm_write": {"enable_sspm_write": False},
    "M11_no_distill": {"enable_distill": False},
    "M12_no_advisory_cache": {"enable_advisory_cache": False},
    # 组合 arm：只保留"有效块"（证据/产出率/多样性/蒸馏/advisory缓存），
    # 关闭"负资产/存疑块"（经验块/编辑先验/饱和度/结构命中率/APV/SSPM写入）。
    # 用于验证"精简记忆"是否优于全量 control。
    "OPT_effective_only": {
        "enable_experience_block": False,
        "enable_edit_patterns": False,
        "enable_saturation_block": False,
        "enable_structure_stats_block": False,
        "apv_tau_c": 1.0,
        "apv_tau_v": 1.0,
        "enable_sspm_write": False,
    },
}

ALL_ARMS = ["control"] + list(ARM_SWITCHES.keys())


def get_arm_spec_overrides(arm_name: str) -> dict[str, Any]:
    base_spec = default_research_spec("technical")
    spec = copy.deepcopy(base_spec)
    if arm_name == "control":
        pass
    elif arm_name in ARM_SWITCHES:
        spec["memory_policy"].update(ARM_SWITCHES[arm_name])
    else:
        raise ValueError(f"未知组件消融 arm 名称: {arm_name}")
    return spec


def run_single_arm(
    arm_name: str,
    output_base: Path,
    max_turns: int = 30,
    user_message: str = "挖掘稳健低换手的量价/筹码日频截面多因子",
) -> Path:
    # output_base 即该 arm 的目录（单次运行）或 rep 目录（重复实验）
    arm_dir = output_base
    arm_dir.mkdir(parents=True, exist_ok=True)
    spec = get_arm_spec_overrides(arm_name)
    spec_path = arm_dir / "research_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 记忆隔离：每个 arm 使用同一份基线快照的独立副本 ──
    # 快照在 study_dir 根（向上找），rep 目录复用同一份
    mem_snapshot = output_base / "memory_snapshot.db"
    if not mem_snapshot.is_file():
        for parent in output_base.parents:
            cand = parent / "memory_snapshot.db"
            if cand.is_file():
                mem_snapshot = cand
                break
    arm_mem = arm_dir / "research_memory.db"
    if not arm_mem.is_file():
        if mem_snapshot.is_file():
            shutil.copy2(mem_snapshot, arm_mem)
        else:
            src = ROOT / "artifacts" / "alphaagent" / "research_memory.db"
            if src.is_file():
                shutil.copy2(src, arm_mem)
                if not mem_snapshot.is_file():
                    shutil.copy2(src, mem_snapshot)

    run_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_alphaagent.py"),
        "--user-message", user_message,
        "--max-turns", str(max_turns),
        "--research-spec-file", str(spec_path),
        "--log-dir", str(arm_dir),
        "--research-memory-file", str(arm_mem),
        "--no-fundamentals",
        "--test-end", "2026-09-11",
    ]

    print(f"\n========================================================")
    print(f"[{utc_now_iso()}] 开始执行组件消融 Arm: {arm_name} (Max Turns: {max_turns})")
    print(f"输出目录: {arm_dir}")
    print(f"========================================================")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    t0 = time.time()
    proc = subprocess.Popen(
        run_cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            print(f"[{arm_name}] {line.strip()}")
        except UnicodeEncodeError:
            clean = line.strip().encode("ascii", errors="replace").decode("ascii")
            print(f"[{arm_name}] {clean}")
    proc.wait()
    elapsed = time.time() - t0
    print(f"[{arm_name}] 执行完成，耗时 {elapsed:.1f}s，退出码: {proc.returncode}")

    # 生成/补齐 scorecard.json
    try:
        generate_scorecard(arm_name, arm_dir)
    except Exception as e:
        print(f"[{arm_name}] 生成 scorecard 异常: {e}", file=sys.stderr)

    return arm_dir


def write_run_meta(study_dir: Path, arms: list[str], max_turns: int, repeats: int) -> None:
    """参数快照：commit hash、模型版本、各 arm 的 memory_policy 开关组合。"""
    import subprocess as sp

    try:
        commit = sp.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    meta = {
        "created_at": utc_now_iso(),
        "commit": commit,
        "max_turns": max_turns,
        "repeats": repeats,
        "arms": {},
    }
    for arm in arms:
        spec = get_arm_spec_overrides(arm)
        meta["arms"][arm] = {
            "memory_policy": spec["memory_policy"],
        }
    (study_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_ablation_summary(study_dir: Path) -> None:
    """遍历 study 目录下的所有 arm，生成统一对比表格与 CSV。"""
    arm_dirs = [d for d in study_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not arm_dirs:
        print("未找到有效实验组数据。")
        return

    records = []
    for d in arm_dirs:
        sc_file = d / "scorecard.json"
        if not sc_file.is_file():
            continue
        try:
            sc = json.loads(sc_file.read_text(encoding="utf-8"))
            sm = sc.get("summary", {})
            fn = sc.get("funnel", {})
            cg = sc.get("cognition", {})
            records.append({
                "Arm": sc.get("run_id") or d.name,
                "Turns": sm.get("total_turns", 0),
                "Throughput": sm.get("eval_throughput", 0),
                "Attempts": fn.get("unique_train_evaluated", 0),
                "CandStored": fn.get("candidate_stored", 0),
                "ProdStored": fn.get("production_stored", 0),
                "StageOneYield%": fn.get("stage_one_yield_pct", 0),
                "GateSurvival%": fn.get("gate_survival_pct", 0),
                "ConfRatio%": cg.get("confirmed_ratio_pct") or 0.0,
                "DeadEnd%": round((cg.get("dup_dead_end_rate", 0) * 100), 1),
                "Unsubmitted": fn.get("unsubmitted_promising", 0),
                "NearMiss": fn.get("near_miss_count", 0),
            })
        except Exception:
            continue

    if not records:
        print("暂无已完成的 scorecard 数据。")
        return

    import pandas as pd
    df = pd.DataFrame(records)
    csv_out = study_dir / "memory_ablation_summary.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"【AlphaAgent 记忆层组件消融综合量化对比大表】(保存至 {csv_out.name})")
    print("=" * 90)
    fmt_str = f"{'{:<18}':<18} | {'{:<5}':<5} | {'{:<10}':<10} | {'{:<8}':<8} | {'{:<8}':<8} | {'{:<8}':<8} | {'{:<10}':<10} | {'{:<10}':<10}"
    print(fmt_str.format("Arm 实验组", "轮次", "评估吞吐", "训练尝试", "入候选池", "正式入库", "海选过线%", "Gate存活%"))
    print("-" * 90)
    for r in records:
        print(fmt_str.format(
            r["Arm"],
            r["Turns"],
            f"{r['Throughput']:.2f}/m",
            r["Attempts"],
            r["CandStored"],
            r["ProdStored"],
            f"{r['StageOneYield%']:.1f}%",
            f"{r['GateSurvival%']:.1f}%",
        ))
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaAgent 记忆层组件消融实验执行与评估工具")
    parser.add_argument("--arms", type=str, default=None,
                        help="逗号分隔的组件消融组名称（默认全部 13 组）")
    parser.add_argument("--suite", type=str, default=None,
                        help="full = Control + 12 arms；partial = 仅指定 --arms")
    parser.add_argument("--rounds", type=int, default=30, help="每个消融组的对话轮次限制（默认 30）")
    parser.add_argument("--repeats", type=int, default=1, help="每个 arm 的重复次数（默认 1）")
    parser.add_argument("--parallel", type=int, default=3,
                        help="并行执行的 arm 数（默认 3，避免 LLM API 限流；每个 arm 独立子进程+独立记忆库副本）")
    parser.add_argument("--out-dir", type=str, default=None, help="实验产物保存目录（默认 artifacts/memory_ablation_<ts>）")
    parser.add_argument("--report-dir", type=str, default=None, help="仅汇总指定目录的消融大表并退出")
    args = parser.parse_args()

    if args.report_dir:
        generate_ablation_summary(Path(args.report_dir))
        return 0

    if args.arms:
        target_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    elif args.suite == "full":
        target_arms = ALL_ARMS
    else:
        target_arms = ALL_ARMS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / f"memory_ablation_{ts}")
    study_dir.mkdir(parents=True, exist_ok=True)

    write_run_meta(study_dir, target_arms, args.rounds, args.repeats)
    print(f"开始执行记忆层组件消融实验套件: {target_arms} (共 {len(target_arms)} 组 × {args.repeats} 次重复，并行 {args.parallel})")

    # 展开 (arm, rep) 任务列表
    tasks = [(arm, rep) for arm in target_arms for rep in range(1, args.repeats + 1)]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(task):
        arm, rep = task
        rep_dir = study_dir / arm / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_single_arm(arm, rep_dir, max_turns=args.rounds)
            return (arm, rep, None)
        except Exception as e:
            return (arm, rep, str(e))

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            arm, rep, err = fut.result()
            if err:
                print(f"Arm [{arm}] rep{rep} 执行失败: {err}", file=sys.stderr)
            else:
                print(f"Arm [{arm}] rep{rep} 完成")

    generate_ablation_summary(study_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
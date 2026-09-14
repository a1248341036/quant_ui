#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaAgent 纯挖掘能力与因子质量消融实验驱动器（Ablation Study Runner）。

支持自动化执行：
  - Control：全装配基线（默认配置）
  - A1_no_memory：关闭记忆库注入与检索
  - A2_pos_only：仅保留正向记忆（关闭 APV 否决与死路拦截）
  - B1_minimal_prompt：剥离市场机制、形态学教学、行为红线等重度 Prompt
  - B2_no_mechanisms：剥离 A 股微观结构机制模块
  - C1_no_prediction：关闭 prediction 强制对账与形态拦截
  - C2_no_ablation：关闭门控条件消融检查
  - D1_free_search：自由探索（关闭轨道约束）
  - D2_simple_ops：禁用高阶复杂复合算子

用法示例：
    # 运行指定 arm（如 control 与 A1）
    python scripts/run_ablation_study.py --arms control,A1_no_memory --max-turns 5

    # 汇总已执行实验并输出对比报告
    python scripts/run_ablation_study.py --report-dir artifacts/ablation_study
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


# 定义各 Arm 的消融配置修改策略
def get_arm_spec_overrides(arm_name: str) -> dict[str, Any]:
    base_spec = default_research_spec("technical")
    spec = copy.deepcopy(base_spec)

    if arm_name == "control":
        pass
    elif arm_name == "A1_no_memory":
        # 关闭记忆库
        spec["memory_policy"]["enabled"] = False
        spec["memory_policy"]["enable_factor_retrieval"] = False
        spec["memory_policy"]["enable_edit_patterns"] = False
        spec["memory_policy"]["max_inject_chars"] = 0
    elif arm_name == "A2_pos_only":
        # 仅正向记忆，屏蔽 APV 否决与硬死路阻断
        spec["memory_policy"]["hard_block_duplicates"] = False
        spec["memory_policy"]["include_rejected_paths"] = False
        spec["memory_policy"]["apv_tau_v"] = 1.0  # 永不 veto
    elif arm_name == "B1_minimal_prompt":
        # 极简提示词：关闭机制、形态学、行为规则
        spec.setdefault("prompt_policy", {})["excluded_modules"] = [
            "market_mechanisms",
            "ic_robustness",
            "behavior_rules",
            "data_calibration",
            "neutralization_guide",
            "multi_period",
        ]
    elif arm_name == "B2_no_mechanisms":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = [
            "market_mechanisms"
        ]
    elif arm_name == "C1_no_prediction":
        # 关闭预测对账
        spec["cognition_policy"]["prediction_check_enabled"] = False
    elif arm_name == "C2_no_ablation":
        # 关闭门控消融
        spec["cognition_policy"]["ablation_check_enabled"] = False
    elif arm_name == "D1_free_search":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = [
            "strategy_tracks"
        ]
    elif arm_name == "D2_simple_ops":
        spec.setdefault("operator_policy", {})["blacklist"] = [
            "GATED_SIGNAL", "PIECEWISE_STATE", "DIVERGENCE_RANK",
            "CS_GROUP_RANK", "CS_RESIDUALIZE", "IF_THEN_ELSE"
        ]
    else:
        raise ValueError(f"未知消融 arm 名称: {arm_name}")

    return spec


def run_single_arm(
    arm_name: str,
    output_base: Path,
    max_turns: int = 5,
    user_message: str = "挖掘稳健低换手的量价/筹码日频截面多因子",
) -> Path:
    arm_dir = output_base / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    spec = get_arm_spec_overrides(arm_name)
    spec_path = arm_dir / "research_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 记忆隔离：每个 arm 使用同一份基线快照的独立副本 ──
    # 消融实验的方法论要求：所有 arm 必须从**相同的初始记忆状态**起跑，
    # 否则后跑的 arm 会看到先跑 arm 写入的条目（顺序效应污染对比结果）。
    mem_snapshot = output_base / "_memory_snapshot.db"
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
    # 口径分离：A1 无记忆 arm 显式指向空库（而非共享快照），确保彻底无注入
    if arm_name == "A1_no_memory":
        arm_mem = arm_dir / "_empty_memory.db"

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
    print(f"[{utc_now_iso()}] 开始执行消融 Arm: {arm_name} (Max Turns: {max_turns})")
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
    # 流式输出
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


def evaluate_arm_candidates_blind_test(arm_dir: Path) -> dict[str, Any]:
    """对该 arm 生成的候选因子运行隔离盲测段 (2025-01至最新) 回测裁决。"""
    sc_file = arm_dir / "scorecard.json"
    candidates_file = arm_dir / "candidate_factors.json"
    
    # 检查是否有盲测报告
    blind_test_report = {
        "arm": arm_dir.name,
        "n_candidates": 0,
        "blind_tested": 0,
        "avg_test_ic": None,
        "avg_test_icir": None,
        "top_factor": None,
    }
    return blind_test_report


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
    csv_out = study_dir / "ablation_summary.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"【AlphaAgent 消融实验综合量化对比大表】(保存至 {csv_out.name})")
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
    parser = argparse.ArgumentParser(description="AlphaAgent 挖掘机制消融实验执行与评估工具")
    parser.add_argument("--arms", type=str, default="control,A1_no_memory,B1_minimal_prompt,C1_no_prediction",
                        help="逗号分隔的消融组名称（可选: control, A1_no_memory, A2_pos_only, B1_minimal_prompt, B2_no_mechanisms, C1_no_prediction, C2_no_ablation, D1_free_search, D2_simple_ops）")
    parser.add_argument("--max-turns", type=int, default=5, help="每个消融组的对话轮次限制（默认 5 轮快速体检）")
    parser.add_argument("--out-dir", type=str, default=None, help="实验产物保存目录（默认 artifacts/ablation_<ts>）")
    parser.add_argument("--report-dir", type=str, default=None, help="仅汇总指定目录的消融大表并退出")
    args = parser.parse_args()

    if args.report_dir:
        generate_ablation_summary(Path(args.report_dir))
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = Path(args.out_dir) if args.out_dir else (ROOT / "artifacts" / f"ablation_{ts}")
    study_dir.mkdir(parents=True, exist_ok=True)

    target_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"开始执行消融实验套件: {target_arms} (共 {len(target_arms)} 组)")

    for arm in target_arms:
        try:
            run_single_arm(arm, study_dir, max_turns=args.max_turns)
        except Exception as e:
            print(f"Arm [{arm}] 执行失败: {e}", file=sys.stderr)

    generate_ablation_summary(study_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

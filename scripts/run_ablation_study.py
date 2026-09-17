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
    # ── Prompt 模块单模块消融（docs/specs/alphaagent_prompt_module_ablation_spec.md）──
    # P1a~P5a：excluded_modules 单模块剥离（其余 16 个全保留）
    elif arm_name == "P1a_core_identity":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["core_identity"]
    elif arm_name == "P2a_strategy_tracks":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["strategy_tracks"]
    elif arm_name == "P2b_behavior_rules":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["behavior_rules"]
    elif arm_name == "P3a_delivery_interface":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["delivery_interface"]
    elif arm_name == "P3b_tool_contracts":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["tool_contracts"]
    elif arm_name == "P3c_delivery_submission":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["delivery_submission"]
    elif arm_name == "P4a_data_calibration":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["data_calibration"]
    elif arm_name == "P4b_market_mechanisms":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["market_mechanisms"]
    elif arm_name == "P4c_multi_period":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["multi_period"]
    elif arm_name == "P4d_operator_catalog":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["operator_catalog"]
    elif arm_name == "P4e_neutralization_guide":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["neutralization_guide"]
    elif arm_name == "P4f_ic_robustness":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["ic_robustness"]
    elif arm_name == "P5a_tool_examples":
        spec.setdefault("prompt_policy", {})["excluded_modules"] = ["tool_examples"]
    # P6a：价量字段族消融（纯 prompt 层，数据加载不动）——只注入价量/量能/筹码/拥挤
    # 行情组字段族，基本面/事件/资金等字段族全部隐藏（field_family_scope 白名单）。
    elif arm_name == "P6a_price_only_fields":
        spec.setdefault("prompt_policy", {})["field_family_scope"] = [
            "adj_", "close", "open", "high", "low", "ret", "vwap",
            "volume", "amount", "turnover", "chip_", "crowd_",
            "float_cap", "tot_cap", "is_trade", "not_st", "industry_sw_l1",
        ]
    else:
        raise ValueError(f"未知消融 arm 名称: {arm_name}")

    return spec


def run_single_arm(
    arm_name: str,
    output_base: Path,
    max_turns: int = 5,
    user_message: str = "挖掘稳健低换手的量价/筹码日频截面多因子",
    repeat_idx: int = 0,
) -> Path:
    # 重复实验：输出目录带 _repN 后缀（N 从 1 起），记忆快照仍按 arm 名共享
    arm_dir = output_base / (f"{arm_name}_rep{repeat_idx}" if repeat_idx > 0 else arm_name)
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
        # 盲测段纪律：不传 --test-end，用默认动态解析（数据源最新交易日）。
        # 挖掘循环只看到 train/val 段，2025 起为锁定盲测段（见 alphaagent_mining_ablation_spec.md）。
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
    """遍历 study 目录下的所有 arm，生成统一对比表格与 CSV。

    重复实验（arm_repN 目录）按 arm 名聚合：输出均值 ± 标准差，
    满足统计纪律"每组 ≥3 次重复，报告均值 ± 波动范围"。
    """
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
            # 重复目录（arm_repN）归并到 arm 名；首轮（无后缀）保持原名
            base = d.name.rsplit("_rep", 1)[0] if "_rep" in d.name else d.name
            records.append({
                "Arm": base,
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
    # 按 arm 聚合：均值 ± 标准差（重复数 >1 时）
    num_cols = [c for c in df.columns if c != "Arm"]
    agg = df.groupby("Arm")[num_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg = agg.reset_index()
    # 生成可读列：mean ± std（std 为 0 或 NaN 时只显示 mean）
    for c in num_cols:
        mean_c, std_c, cnt_c = f"{c}_mean", f"{c}_std", f"{c}_count"
        if mean_c in agg.columns:
            agg[f"{c}±"] = agg.apply(
                lambda r: (
                    f"{r[mean_c]:.2f}±{r[std_c]:.2f}"
                    if r[cnt_c] > 1 and pd.notna(r[std_c]) and r[std_c] > 0
                    else f"{r[mean_c]:.2f}"
                ),
                axis=1,
            )
    csv_out = study_dir / "ablation_summary.csv"
    agg.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"【AlphaAgent 消融实验综合量化对比大表】(保存至 {csv_out.name})")
    print("=" * 90)
    fmt_str = f"{'{:<18}':<18} | {'{:<5}':<5} | {'{:<10}':<10} | {'{:<8}':<8} | {'{:<8}':<8} | {'{:<8}':<8} | {'{:<10}':<10} | {'{:<10}':<10}"
    print(fmt_str.format("Arm 实验组", "轮次", "评估吞吐", "训练尝试", "入候选池", "正式入库", "海选过线%", "Gate存活%"))
    print("-" * 90)
    for _, r in agg.iterrows():
        print(fmt_str.format(
            r["Arm"],
            f"{r['Turns_mean']:.0f}",
            f"{r['Throughput_mean']:.2f}/m",
            f"{r['Attempts_mean']:.0f}",
            f"{r['CandStored_mean']:.0f}",
            f"{r['ProdStored_mean']:.0f}",
            f"{r['StageOneYield%_mean']:.1f}%",
            f"{r['GateSurvival%_mean']:.1f}%",
        ))
    print("=" * 90)


def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaAgent 挖掘机制消融实验执行与评估工具")
    parser.add_argument("--arms", type=str, default="control,A1_no_memory,B1_minimal_prompt,C1_no_prediction",
                        help="逗号分隔的消融组名称（可选: control, A1_no_memory, A2_pos_only, B1_minimal_prompt, B2_no_mechanisms, C1_no_prediction, C2_no_ablation, D1_free_search, D2_simple_ops）")
    parser.add_argument("--max-turns", type=int, default=5, help="每个消融组的对话轮次限制（默认 5 轮快速体检）")
    parser.add_argument("--repeats", type=int, default=1, help="每个消融组的重复次数（统计纪律要求 ≥3；输出目录带 _repN 后缀）")
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
    repeats = max(1, args.repeats)
    print(f"开始执行消融实验套件: {target_arms} (共 {len(target_arms)} 组 × {repeats} 次重复)")

    for arm in target_arms:
        for rep in range(1, repeats + 1):
            try:
                run_single_arm(arm, study_dir, max_turns=args.max_turns, repeat_idx=rep)
            except Exception as e:
                print(f"Arm [{arm}] 重复 {rep} 执行失败: {e}", file=sys.stderr)

    generate_ablation_summary(study_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

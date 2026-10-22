#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness 自驱动挖掘 · 批量评估器（宿主 agent 直接充当挖掘 LLM 时使用）。

与 agentscope 挖掘链路**同源**：同一 `StockEvalService`、同一 `effective_research_spec`、
同一 `DeliveryCriteria`（门槛数值零硬编码），因此评估结果与真实 run 内的评估一致。

输出：每因子的 train 指标（ic/icir/coverage/cs_autocorr/换手）+ **逐门 PASS/FAIL**；
`--val` 可对过门者追加样本外（val 段）评估；结果落 JSONL 供下一轮迭代。

用法：
    .venv\\Scripts\\python.exe scripts\\harness_mine.py --mode technical --factors mine.json --out rows.jsonl
    .venv\\Scripts\\python.exe scripts\\harness_mine.py --mode technical_monthly --factors mine.json --val

factors JSON 格式：[{"name": "...", "expr": "多行 DSL（末行输出）", "why": "机制/经济直觉"}, ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 宿主直驱默认**单段全量评估**（`ALPHA_EVAL_LITE=0`）：
# lite 两段式是给"海选大量候选"省算力的——lite 段只跑 cross_sectional_core，
# 规则不过时直接返回、**不含 quantile_portfolio**，于是换手率不可得
# （`avg_rebalance_side_turnover=None`），而 `_flags` 的换手门是
# `turn is None or turn > limit: fail` → 结果行报 `turnover>0.8` 但 turn=None，
# 把"换手 0.93 真超门"和"压根没测换手"混成同一个失败信号
# （2026-10-19 实测：15/15 因子 turn=None）。
# 宿主直驱是「少量因子精测、要真实全门判定」，省算力的两段式在这里没有收益、
# 只有信息损失，故默认关 lite；确实要省算力时显式 `ALPHA_EVAL_LITE=1` 覆盖。
os.environ.setdefault("ALPHA_EVAL_LITE", "0")

from alphaagent.data.adapters.cnequity import CNE_SOURCE  # noqa: E402
from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria  # noqa: E402
from alphaagent.factor.mining.eval.schemas import (  # noqa: E402
    EvalTrainRequest,
    EvalValRequest,
    SessionCreateRequest,
)
from alphaagent.factor.mining.research_spec import effective_research_spec  # noqa: E402
from alphaagent.factor.evaluation.profile import resolve_profiles  # noqa: E402
from alphaagent.factor.mining.service import StockEvalService  # noqa: E402

# 正交性检查依赖（与 agentscope 链路 _orthogonality_check 同源，2026-10-09 前移到挖掘预筛）
from alphaagent.factor.mining.agent.agentscope_tools import (  # noqa: E402
    _sample_orthogonality_panel,
)

# 冷门赛道二轨评估依赖（spec: alphaagent_mining_anti_convergence_spec.md）。
# 仅在 --cold-track 开启时使用；默认关闭时本组 import 不产生任何副作用（黄金基线零注入）。
try:
    from alphaagent.factor.mining.memory.cold_track import (  # noqa: E402
        build_snapshot_for_run,
        classify_track,
        signal_root_op,
    )
    from alphaagent.factor.mining.eval.residualize import evaluate_cold_track  # noqa: E402
except Exception:  # noqa: BLE001 - 冷门模块缺失时 --cold-track 应报错引导而非静默失效
    build_snapshot_for_run = classify_track = signal_root_op = None
    evaluate_cold_track = None  # type: ignore[assignment]

from alphaagent.data.panel import slice_panel  # noqa: E402
from alphaagent.dsl import eval_factor  # noqa: E402
from alphaagent.factor.align import align_series_to_panel  # noqa: E402
from alphaagent.factor.metrics import spearman_ic  # noqa: E402
from alphaagent.factor.zoo import FactorZoo  # noqa: E402
from alphaagent.core.paths import FACTORZOO_DIR  # noqa: E402
from core import factor_categories  # noqa: E402

# 档位 → 该档评估 label（与 core/research_modes.py 的 recommended_label_col 一致）
MODE_LABEL = {
    "technical": "label_1d_open_to_open",
    "technical_daily": "label_1d_open_to_open",
    "technical_weekly": "label_5d_close_to_close",
    "technical_monthly": "label_20d_close_to_close",
    "fundamental": "label_20d_close_to_close",
    "fundamental_daily": "label_1d_open_to_open",
    "fundamental_weekly": "label_5d_close_to_close",
}


def _increment_flags(mode: str, row: dict) -> list[str]:
    """增量轨（2026-10-20）：raw 门未过时，判断"对现有组合的增量"是否达标。

    返回空列表 = 增量轨通过。三条同时满足（缺一不可）：
    |raw_ic| >= floor（地板，防 increment 分母趋零爆表）/ |resid_ic| >= 门 / increment >= 门。
    """
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria

    cand = DeliveryCriteria.from_spec(effective_research_spec(mode)).candidate
    if not getattr(cand, "enable_increment_track", True):
        return ["increment_track_disabled"]
    raw, resid, inc = row.get("raw_ic_audit"), row.get("resid_ic"), row.get("increment")
    if raw is None or resid is None or inc is None:
        return ["no_increment_metrics"]
    fails: list[str] = []
    if abs(raw) < cand.min_raw_ic_floor:
        fails.append(f"raw<{cand.min_raw_ic_floor}")
    if abs(resid) < cand.min_resid_ic:
        fails.append(f"resid<{cand.min_resid_ic}")
    if abs(inc) < cand.min_increment_ratio:
        fails.append(f"inc<{cand.min_increment_ratio}")
    return fails


def _flags(mode: str, row: dict) -> list[str]:
    """按真源门槛逐门判定，返回未过门说明（空列表 = 全部通过）。"""
    c = DeliveryCriteria.from_spec(effective_research_spec(mode))
    cand = c.candidate
    ic, icir, cov = row.get("ic"), row.get("icir"), row.get("coverage")
    ac, turn = row.get("cs_autocorr"), row.get("turnover_rebalance")
    fails: list[str] = []
    if ic is None or abs(ic) < cand.min_abs_ic:
        fails.append(f"ic<{cand.min_abs_ic}")
    if icir is None or abs(icir) < cand.min_icir:
        fails.append(f"icir<{cand.min_icir}")
    if cov is None or cov < cand.min_coverage:
        fails.append(f"cov<{cand.min_coverage}")
    if ac is None or ac < cand.min_cs_autocorr:
        fails.append(f"cs_autocorr<{cand.min_cs_autocorr}")
    if turn is None or turn > c.turnover_gate_limit:
        fails.append(f"turnover>{c.turnover_gate_limit}")
    return fails


def _expr_columns(factors: list[dict]) -> tuple[str, ...]:
    """从全部因子的 DSL 里收集 ``$字段`` → 去重后的真实面板列名（含别名归一）。

    只收集**表达式显式引用**的列；算子隐式消费的列（CHIP_* 要 OHLCV、中性化要
    行业/市值、掩码要 is_trade/not_st、复权要 adjfactor）由
    ``cnequity.load_panel_from_cne`` 的必需列兜底并上，不在本函数罗列。
    """
    from alphaagent.dsl.core.field_aliases import canonical_column

    refs: set[str] = set()
    for f in factors:
        for m in re.finditer(r"\$([A-Za-z_][A-Za-z0-9_]*)(?:@[A-Za-z0-9_]+)?", f.get("expr") or ""):
            refs.add(canonical_column(m.group(1)))
    refs.discard("")
    return tuple(sorted(refs))


def _attach_increment(rows: list[dict], factors: list[dict], args) -> None:
    """给评估结果补上「对现有因子库组合的增量」四个数（增量轨，2026-10-20）。

    只在 train 段算（combo 也是 train 段合成的）——**绝不碰盲测段**（2025+），
    否则拿盲测做因子筛选就是数据窥视。

    成本：一次 panel 加载 + 每因子一次求值/截面回归（0.5-1s），不重训模型。
    """
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import (
        cs_zscore, increment_metrics, load_combo,
    )

    combo = load_combo()
    if combo.empty:
        print("[mine] 增量轨跳过：combo 缓存缺失，先跑 "
              "`python scripts/build_combo_score.py`", flush=True)
        return

    start = args.train_start or "2020-01-01"
    end = args.train_end or "2024-12-31"
    label = MODE_LABEL[args.mode]
    need: set[str] = {label}
    for f in factors:
        try:
            need |= set(referenced_column_names(f["expr"]))
        except Exception:  # noqa: BLE001
            pass
    need = {c for c in need if not str(c).startswith(("funda_", "fsi_"))}

    print(f"[mine] 增量轨：panel {start}~{end}（train 段，不碰盲测）", flush=True)
    panel = load_panel_from_cne(start=start, end=end, include_fundamentals=False,
                                keep_columns=tuple(sorted(need)))
    if label not in panel.columns:
        print(f"[mine] 增量轨跳过：面板缺 {label}", flush=True)
        return
    lab = panel[label].astype("float64")
    combo = combo.reindex(panel.index)

    by_name = {r.get("name"): r for r in rows}
    for f in factors:
        name = f.get("name")
        row = by_name.get(name)
        if row is None:
            continue
        try:
            v = eval_multi_line_factor(f["expr"], panel)
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
            z = cs_zscore(pd.Series(v).astype("float64").reindex(panel.index))
            m = increment_metrics(z, combo, lab)
            row.update({"resid_ic": m["resid_ic"], "increment": m["increment"],
                        "combo_corr": m["combo_corr"],
                        "raw_ic_audit": m["raw_ic"],
                        "increment_reason": m.get("reason")})
            inc = m["increment"]
            print(f"  [inc] {str(name)[:34]:34s} raw={m['raw_ic']:+.4f} "
                  f"resid={(m['resid_ic'] if m['resid_ic'] is None else round(m['resid_ic'],4))} "
                  f"inc={inc if inc is None else round(inc,3)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row.update({"resid_ic": None, "increment": None,
                        "increment_error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            print(f"  [inc] {str(name)[:34]:34s} ERR {type(exc).__name__}", flush=True)

    # 结算：分「raw 门全过」与「增量轨通过」两类打印，便于一眼看出增量轨捞回了什么
    inc_pass, raw_pass = [], []
    for row in rows:
        raw_fails = _flags(args.mode, row)
        row["increment_fails"] = _increment_flags(args.mode, row)
        row["pass_increment"] = bool(row["increment_fails"] == [] and row.get("resid_ic") is not None)
        if not raw_fails:
            raw_pass.append(row)
        elif row["pass_increment"]:
            inc_pass.append(row)
    if inc_pass:
        print(f"\n[mine] 增量轨捞回 {len(inc_pass)} 个（raw 门未过但对组合有增量）：", flush=True)
        for r in inc_pass:
            print(f"  ◆ {r['name'][:34]:34s} raw={r['raw_ic_audit']:+.4f} "
                  f"resid={r['resid_ic']:+.4f} inc={r['increment']:.3f} "
                  f"(raw_fails={','.join(_flags(args.mode, r))})", flush=True)
    if args.increment and not inc_pass and not raw_pass:
        print("[mine] 增量轨：本批没有满足 raw 门或增量门的因子", flush=True)


def _extract(res: dict) -> dict:
    s = res.get("summary") if isinstance(res, dict) else None
    s = s if isinstance(s, dict) else {}
    # 换手取顶层，回落嵌套 quantile_portfolio：eval/service.py 在 2026-10-19 修复前
    # 只把换手写进嵌套层，而本函数读顶层 → turn=None → _flags 换手门永久 fail
    # （2026-10-19 实测 15/15 因子全报 turnover>0.8）。两处都读，防单边回归。
    qp = res.get("quantile_portfolio") if isinstance(res, dict) else None
    qp = qp if isinstance(qp, dict) else {}

    def _pick(key: str) -> Any:
        for src in (res, qp):
            if isinstance(src, dict):
                v = src.get(key)
                if v is not None:
                    return v
        return None

    return {
        "ic": s.get("ic"),
        "icir": s.get("icir"),
        "rank_ic": s.get("rank_ic"),
        "coverage": s.get("factor_coverage"),
        "cs_autocorr": s.get("cs_pearson_autocorr"),
        "turnover_daily": _pick("avg_daily_side_turnover"),
        "turnover_rebalance": _pick("avg_rebalance_side_turnover"),
        "n_days": s.get("n_days"),
        # Rotation P1 industry diagnostics (top-level passthrough, see eval/response.py)
        "industry_rotation": res.get("industry_rotation"),
    }


def _ortho_threshold() -> float:
    """离线正交门阈值：取 evaluation_policy.orthogonality_max_corr（与 agentscope 链路同源）。"""
    try:
        from alphaagent.factor.evaluation.defaults import DEFAULT_EVALUATION_POLICY

        return float(DEFAULT_EVALUATION_POLICY.get("orthogonality_max_corr", 0.7))
    except Exception:  # noqa: BLE001
        return 0.7


def _orthogonality_check(
    session: Any,
    multi_line_expr: str,
    max_corr: float,
    exclude_factor_id: str | None = None,
) -> dict[str, Any]:
    """挖掘预筛阶段正交性检查（与 agentscope _orthogonality_check 同源，2026-10-09 前移）。

    在 stage_one 预筛通过后、候选池存储前执行：抽样可见区间面板（train∪val，剔除盲测），
    与 production_main / candidate_main / candidate_registry 已有因子算 Spearman 相关，
    max_abs_corr >= max_corr 则标记 fail_orthogonal 并附 top-3 相似因子清单。

    避免挖掘产出的因子"过 stage_one → 入候选池 → stage_two 才被正交拒绝"的算力浪费。

    exclude_factor_id：跳过该 factor_id（重测已入库因子时排除自己，避免自相关 1.0 误判）。
    """
    result: dict[str, Any] = {
        "passed": True,
        "skipped_reason": None,
        "threshold": max_corr,
        "max_abs_corr": 0.0,
        "compared_factors": 0,
        "blocked_factor_id": None,
        "similar_factors": [],
    }
    try:
        panel = getattr(session, "panel", None)
        ctx = getattr(session, "ctx", None)
        if panel is None or ctx is None:
            result["skipped_reason"] = "session_missing_panel_or_ctx"
            return result
        # 可见区间 = train ∪ val（剔除盲测段，防泄漏）
        try:
            start, end = ctx.visible_range()
        except Exception:  # noqa: BLE001
            result["skipped_reason"] = "visible_range_unavailable"
            return result
        visible = slice_panel(panel, start=start, end=end)
        sampled = _sample_orthogonality_panel(visible)

        # 统一大库：production_main + candidate_main + FACTORZOO_DIR（去重）
        roots: list = []
        for root in (
            factor_categories.production_dir("technical"),
            factor_categories.candidate_dir("technical"),
            FACTORZOO_DIR,
        ):
            if root not in roots:
                roots.append(root)
        zoos = []
        for root in roots:
            try:
                zoos.append(FactorZoo.open(root))
            except (FileNotFoundError, OSError):
                continue
        registry_path = factor_categories.candidate_registry_path("technical")
        registry = (
            json.loads(registry_path.read_text(encoding="utf-8"))
            if registry_path.is_file()
            else {}
        )
        if sum(z.n_factors for z in zoos) == 0 and not registry:
            result["skipped_reason"] = "empty_factor_libraries"
            return result

        raw = eval_factor(multi_line_expr, sampled)
        if not isinstance(raw, pd.Series):
            raise TypeError(f"factor_output_must_be_series:{type(raw)!r}")
        compared: set[str] = set()
        corr_pairs: list[tuple[str, float]] = []
        sampled_keys = sampled.index
        new_values = np.asarray(
            pd.Series(
                np.asarray(align_series_to_panel(raw, sampled), dtype=np.float64),
                index=sampled.index,
            ).reindex(sampled_keys),
            dtype=np.float64,
        )

        # 正式库 + 候选库 dense 因子
        for zoo in zoos:
            rows = zoo.index.rows
            selected_dates = set(sampled.index.get_level_values("datetime"))
            subset = rows[rows["datetime"].isin(selected_dates)]
            if subset.empty:
                continue
            row_ids = subset["row_id"].to_numpy(dtype=np.int64)
            for factor_id in zoo.catalog.list_factor_ids()[:20]:
                if factor_id in compared:
                    continue
                if exclude_factor_id is not None and factor_id == exclude_factor_id:
                    continue  # 重测已入库因子时排除自己
                compared.add(factor_id)
                try:
                    subset_keys = pd.MultiIndex.from_arrays(
                        [
                            pd.to_datetime(subset["datetime"]).to_numpy(),
                            subset["instrument"].astype(str).to_numpy(),
                        ],
                        names=["datetime", "instrument"],
                    )
                    old_by_key = pd.Series(
                        np.asarray(zoo.read_factor(factor_id)[row_ids], dtype=np.float64),
                        index=subset_keys,
                    )
                    old_values = np.asarray(old_by_key.reindex(sampled_keys), dtype=np.float64)
                except Exception:  # noqa: BLE001
                    continue
                valid = np.isfinite(old_values) & np.isfinite(new_values)
                if int(valid.sum()) < 30:
                    continue
                corr = abs(float(spearman_ic(old_values[valid], new_values[valid], min_pairs=30)))
                if np.isfinite(corr):
                    corr_pairs.append((factor_id, corr))
                if np.isfinite(corr) and corr > result["max_abs_corr"]:
                    result["max_abs_corr"] = corr
                    result["blocked_factor_id"] = factor_id

        # 候选池 registry-only 因子（无 dense 值，用 DSL 在抽样面板上现算）
        for factor_id, entry in sorted(registry.items()):
            if factor_id in compared or not isinstance(entry, dict):
                continue
            if exclude_factor_id is not None and factor_id == exclude_factor_id:
                continue  # 重测已入库因子时排除自己，避免自相关 1.0 误判
            expr = str(entry.get("expr") or "").strip()
            if not expr:
                continue
            compared.add(factor_id)
            try:
                old_raw = eval_factor(expr, sampled)
                old_values = np.asarray(
                    pd.Series(
                        np.asarray(align_series_to_panel(old_raw, sampled), dtype=np.float64),
                        index=sampled.index,
                    ).reindex(sampled_keys),
                    dtype=np.float64,
                )
            except Exception:  # noqa: BLE001
                continue
            valid = np.isfinite(old_values) & np.isfinite(new_values)
            if int(valid.sum()) < 30:
                continue
            corr = abs(float(spearman_ic(old_values[valid], new_values[valid], min_pairs=30)))
            if np.isfinite(corr):
                corr_pairs.append((factor_id, corr))
            if np.isfinite(corr) and corr > result["max_abs_corr"]:
                result["max_abs_corr"] = corr
                result["blocked_factor_id"] = factor_id

        result["compared_factors"] = len(compared)
        corr_pairs.sort(key=lambda p: -p[1])
        result["similar_factors"] = [
            {"factor_id": fid, "corr": round(c, 4)} for fid, c in corr_pairs[:3]
        ]
        result["passed"] = result["max_abs_corr"] < max_corr
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            **result,
            "passed": False,
            "skipped_reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cold_track_eval(
    session: Any,
    expr: str,
    name: str,
    spec: dict,
    skeletons: dict,
    label: str,
) -> dict[str, Any]:
    """冷门赛道二轨评估：骨架控制矩阵物化 + evaluate_cold_track（残差化 + val 主门）。"""
    panel = getattr(session, "panel", None)
    ctx = getattr(session, "ctx", None)
    if panel is None or ctx is None:
        return {"ok": False, "passed": False, "reason": "session_missing_panel_or_ctx"}
    ctrl_values: dict[str, Any] = {}
    for fam, meta in (skeletons or {}).items():
        sk_expr = (meta or {}).get("expr")
        if not isinstance(sk_expr, str) or not sk_expr.strip():
            continue
        try:
            out = eval_factor(sk_expr, panel)
            ctrl_values[fam] = align_series_to_panel(out, panel)
        except Exception:  # noqa: BLE001 - 某族骨架物化失败不影响其他族
            ctrl_values[fam] = None
    cand_out = eval_factor(expr, panel)
    cand_values = align_series_to_panel(cand_out, panel)
    policy = (spec.get("cold_track") or {}) if isinstance(spec, dict) else {}
    return evaluate_cold_track(
        cand_values,
        panel,
        ctrl_values,
        ctx,
        label_col=label,
        policy=policy,
        family_novelty=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Harness 挖掘：批量评估自研因子")
    ap.add_argument("--mode", default="technical", choices=sorted(MODE_LABEL))
    ap.add_argument("--factors", required=True, help="JSON: [{name, expr, why?}, ...]")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="逐行结果写出路径（JSONL）")
    ap.add_argument("--cold-track", dest="cold_track", action="store_true",
                    help="冷门赛道二轨评估：客观判为 cold 的候选走残差化评估（crowd 候选走原路径）。"
                         "默认关闭=逐字节旧行为。")
    ap.add_argument("--max-ctrl-families", type=int, default=0,
                    help="(冷门赛道) 骨架控制族数上限；0 = 取 spec cold_track.max_ctrl_families")
    ap.add_argument("--val", action="store_true", help="对 train 全门通过者追加 val 评估")
    ap.add_argument("--check-orthogonality", dest="check_ortho", action="store_true",
                    default=True,
                    help="stage_one 预筛通过后立即查与库内/候选池因子的截面 Spearman 相关（默认开，"
                         "前移自 agentscope _orthogonality_check，避免过线后入候选池再被正交拒绝）")
    ap.add_argument("--no-check-orthogonality", dest="check_ortho", action="store_false",
                    help="关闭正交性检查（仅做 stage_one 统计预筛）")
    ap.add_argument("--max-cs-corr", type=float, default=0.0,
                    help="正交门阈值；0 = 取 evaluation_policy.orthogonality_max_corr 默认值（与 agentscope 同源）")
    ap.add_argument("--fundamentals", action="store_true", help="会话载入 funda_* 列")
    ap.add_argument("--full-cols", action="store_true",
                    help="加载面板全列（默认按表达式引用列按需取列，内存可降数倍）")
    ap.add_argument("--train-start", default=None,
                    help="覆盖 train 起点（YYYY-MM-DD）。用于评估历史较短的新数据源"
                         "——如 rc_* 研报表自 2024-01 才有值、ac_* 自 2026-09，用默认 "
                         "2020 起的 train 段会得到 cov=0 的假阴性")
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--val-start", default=None)
    ap.add_argument("--val-end", default=None)
    ap.add_argument("--increment", action="store_true",
                    help="评估后追加「对现有因子库组合的增量」(resid_ic/increment/combo_corr)。"
                         "需先跑 scripts/build_combo_score.py 生成 combo 缓存；只在 train 段算，"
                         "不碰盲测段")
    ap.add_argument("--universe", default="all",
                    help="股票池（评估口径，非因子属性）：all / top300cap / mid301_800cap / "
                         "szcomp / chinext（规则池可自填数字，如 top500cap）")
    ap.add_argument("--panel", default=str(CNE_SOURCE),
                    help="面板路径：默认 cne:// 实时构建；也可传已有 parquet 缓存（省构建时间）")
    args = ap.parse_args()

    label = MODE_LABEL[args.mode]
    from alphaagent.factor.mining.eval.universe import describe_universe, parse_universe
    universe = parse_universe(args.universe)["name"]   # 非法池名早失败（附可用清单）
    factors = json.loads(Path(args.factors).read_text(encoding="utf-8"))
    if args.limit:
        factors = factors[: args.limit]
    c = DeliveryCriteria.from_spec(effective_research_spec(args.mode))
    spec = effective_research_spec(args.mode)
    if args.cold_track:
        if build_snapshot_for_run is None or evaluate_cold_track is None:
            raise SystemExit("[mine] 冷门赛道模块导入失败：请检查 alphaagent.factor.mining.memory.cold_track / eval.residualize（import 期异常被顶层 except 吞掉）")
        spec = json.loads(json.dumps(spec, default=str))
        spec.setdefault("cold_track", {})["enabled"] = True
        print(f"[mine] 冷门赛道 ENABLED：min_val_resid_ic={spec['cold_track'].get('min_val_resid_ic')} "
              f"max_crowd_proxy_corr={spec['cold_track'].get('max_crowd_proxy_corr')} "
              f"max_ctrl_families={spec['cold_track'].get('max_ctrl_families')}", flush=True)
    print(f"[mine] mode={args.mode} label={label} 股票池={universe}（{describe_universe(universe)}）"
          f" 因子 {len(factors)} 个；门槛 "
          f"|IC|>={c.candidate.min_abs_ic} |ICIR|>={c.candidate.min_icir} cov>={c.candidate.min_coverage} "
          f"autocorr>={c.candidate.min_cs_autocorr} turnover<={c.turnover_gate_limit} (freq={c.engine_gate.freq})",
          flush=True)

    svc = StockEvalService(max_parallel_eval=1,
                           profiles=resolve_profiles(effective_research_spec(args.mode)))
    t0 = time.perf_counter()
    # 按需取列（2026-10-19）：只物化表达式实际引用的列 + 系统必需列。
    # 缓存文件仍是全量（不触发数 GB 重建），但未选中列不构造 numpy 视图、
    # 不 fault 文件页——全量 161 列 vs 实际用 30~50 列，内存差 3-5 倍。
    # `--full-cols` 可退回全列（排查"某列没加载"类问题时用）。
    need_cols: tuple[str, ...] = ()
    if not args.full_cols:
        need_cols = _expr_columns(factors)
        print(f"[mine] 按需取列: 表达式引用 {len(need_cols)} 列"
              f"（+系统必需列由 adapter 兜底）；--full-cols 可退回全量", flush=True)
    _win = {k: getattr(args, k)
            for k in ("train_start", "train_end", "val_start", "val_end")
            if getattr(args, k)}
    if _win:
        print(f"[mine] 窗口覆盖: {_win}", flush=True)
    resp = svc.create_session(SessionCreateRequest(
        panel_path=str(args.panel), label_col=label, include_fundamentals=args.fundamentals,
        universe=universe, columns=need_cols, **_win,
    ))
    sid = resp.session_id
    print(f"[mine] session={sid} rows={resp.panel_rows} cols={len(resp.available_columns or ())} "
          f"load={time.perf_counter() - t0:.0f}s", flush=True)

    rows: list[dict] = []

    skeletons: dict[str, Any] = {}
    if args.cold_track:
        photo = build_snapshot_for_run(
            spec,
            production_registry=factor_categories.production_registry_path(args.mode),
            candidate_registry=factor_categories.candidate_registry_path(args.mode),
            repo_root=ROOT,
            max_ctrl_families=args.max_ctrl_families or None,
        )
        skeletons = (photo or {}).get("skeletons") or {}
        print(f"[mine] 拥挤族骨架 {len(skeletons)} 个 signature="
              f"{(photo or {}).get('signature', '(empty)')}", flush=True)
        for _fam, _meta in skeletons.items():
            print(f"  skeleton[{_fam}] {_meta.get('name')} ic={_meta.get('train_ic'):.4f}",
                  flush=True)

    def _flush() -> None:
        if args.out:
            Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                      encoding="utf-8")

    try:
        for i, f in enumerate(factors, 1):
            name, expr = f["name"], f["expr"]
            t = time.perf_counter()
            session = svc.sessions.get(sid)
            if args.cold_track:
                try:
                    track, newf, coldroot = classify_track(expr, name)
                except Exception:  # noqa: BLE001
                    track, newf, coldroot = "crowd", False, False
                if track == "cold":
                    try:
                        cr = _cold_track_eval(session, expr, name, spec, skeletons, label)
                    except Exception as exc:  # noqa: BLE001
                        cr = {"ok": False, "passed": False,
                              "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                    cr.update(track="cold", is_new_family=newf, has_cold_root=coldroot,
                              root_op=signal_root_op(expr))
                    got = {k: cr[k] for k in (
                        "resid_ic", "train_resid_ic", "train_resid_icir", "val_resid_icir",
                        "val_resid_rank_ic", "coverage", "crowd_proxy_corr",
                        "crowd_proxy_family", "n_fit_days", "n_controls",
                        "min_val_resid_ic", "max_crowd_proxy_corr", "fail_detail",
                        "reason", "error", "track", "is_new_family", "has_cold_root",
                        "root_op", "family_novelty", "train_n_days", "val_n_days",
                    ) if cr.get(k) is not None}
                    err = cr.get("error")
                    _passed = bool(cr.get("passed", False))
                    if err:
                        fails = ["cold_eval_error"]
                    elif _passed:
                        fails = []
                    else:
                        fails = [f"cold_dead({cr.get('fail_detail', '')})"]
                    rows.append({"mode": args.mode, "universe": universe, "name": name,
                                 "why": f.get("why", ""), "expr": expr,
                                 "ms": round((time.perf_counter() - t) * 1000),
                                 "err": err, **got, "fails": fails,
                                 "pass_stage_one_pre": _passed,
                                 "cold_track_passed": _passed,
                                 "orthogonality": None})
                    print(f"  [{i:2d}/{len(factors)}] {name:34s} "
                          f"{'PASS' if _passed else 'dead':4s} "
                          f"resid_ic={got.get('resid_ic')} train={got.get('train_resid_ic')} "
                          f"proxy={got.get('crowd_proxy_corr')} "
                          f"fd={got.get('fail_detail')} {got.get('reason', '')}", flush=True)
                    _flush()
                    continue
            try:
                got = _extract(svc.eval_train(EvalTrainRequest(
                    session_id=sid, multi_line_expr=expr, factor_name=name,
                    include_detail_tables=False, label_quantile_n=10,
                )))
                err = None
            except Exception as exc:  # noqa: BLE001 — 单因子失败不中断批量
                got, err = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
            fails = _flags(args.mode, got) if got else ["eval_error"]
            # pass_stage_one_pre = 统计预筛结果（不含正交门，与 mcp eval_batch 同名字段契约一致）
            pass_stage_one_pre = not fails
            ortho: dict[str, Any] | None = None
            if pass_stage_one_pre and args.check_ortho:
                # stage_one 预筛通过 → 立即查正交性（前移自 agentscope _orthogonality_check）
                threshold = args.max_cs_corr if args.max_cs_corr > 0 else _ortho_threshold()
                session = svc.sessions.get(sid)
                ortho = _orthogonality_check(session, expr, threshold, exclude_factor_id=name)
                if not ortho.get("passed"):
                    if ortho.get("error"):
                        fails.append(f"ortho_error={ortho['error']}")
                    else:
                        fails.append(
                            f"orthogonal<{threshold}(max_corr={ortho.get('max_abs_corr'):.3f}"
                            f",block={ortho.get('blocked_factor_id')})"
                        )
            rows.append({"mode": args.mode, "universe": universe, "name": name,
                         "why": f.get("why", ""), "expr": expr,
                         "ms": round((time.perf_counter() - t) * 1000), "err": err, **got,
                         "fails": fails, "pass_stage_one_pre": pass_stage_one_pre,
                         "orthogonality": ortho})
            ortho_str = ""
            if ortho is not None:
                if ortho.get("skipped_reason"):
                    ortho_str = f" ortho=skip({ortho['skipped_reason']})"
                else:
                    ortho_str = f" ortho={ortho.get('max_abs_corr', 0):.3f}"
                    if ortho.get("similar_factors"):
                        top = ortho["similar_factors"][0]
                        ortho_str += f"({top['factor_id']}:{top['corr']})"
            rot_str = ""
            rot = got.get("industry_rotation") or {}
            if rot:
                _w = rot.get("industry_within_rank_ic") or {}
                _e = rot.get("industry_exposure_r2") or {}
                _d = rot.get("industry_ic_dispersion") or {}
                def _r3(x):
                    return None if x is None else round(float(x), 4)
                rot_str = (f" indIC={_r3(_w.get('within_rank_ic'))}"
                           f" expR2={_r3(_e.get('exposure_r2'))}"
                           f" disp={_r3(_d.get('dispersion_std'))}")
            print(f"  [{i:2d}/{len(factors)}] {name:34s} {'PASS' if not fails else 'fail':4s} "
                  f"ic={got.get('ic')} icir={got.get('icir')} cov={got.get('coverage')} "
                  f"ac={got.get('cs_autocorr')} turn={got.get('turnover_rebalance')} "
                  f"{rows[-1]['ms']}ms{'; '.join([''] + fails) if fails else ''} {err or ''}"
                  f"{ortho_str}{rot_str}", flush=True)
            _flush()

        if args.val:
            # 送 val 按全门（含正交）：统计预筛 + 正交都过的才送 val
            survivors = [r for r in rows if not r.get("fails")]
            print(f"\n[mine] 送 val 的因子 {len(survivors)} 个", flush=True)
            for r in survivors:
                try:
                    v = _extract(svc.eval_val(EvalValRequest(
                        session_id=sid, multi_line_expr=r["expr"], factor_name=r["name"],
                        include_detail_tables=False, label_quantile_n=10,
                    )))
                    r["val"] = v
                    r["val_ok"] = (v.get("ic") is not None
                                   and abs(v["ic"]) >= c.candidate.min_val_abs_ic)
                    print(f"  val {r['name']:34s} ic={v.get('ic')} icir={v.get('icir')} "
                          f"cov={v.get('coverage')} ok={r['val_ok']}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    r["val_err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    print(f"  val {r['name']:34s} ERROR {r['val_err']}", flush=True)
            _flush()
    finally:
        svc.release_session(sid)

    # 增量轨（2026-10-20）：评估完再统一算一次「对现有组合的增量」——
    # 放在这里而不是每轮，是为了只加载一次 panel、且不干扰单因子评估耗时。
    if getattr(args, "increment", False) and rows:
        try:
            _attach_increment(rows, factors, args)
            _flush()
        except Exception as exc:  # noqa: BLE001 — 增量是加分项，失败不能毁掉整批结果
            print(f"[mine] 增量轨失败（不影响 raw 评估结果）: {type(exc).__name__}: {exc}",
                  flush=True)

    n_stat_pass = sum(1 for r in rows if r.get("pass_stage_one_pre"))
    n_full_pass = sum(1 for r in rows if not r.get("fails"))
    print(f"\n[mine] 完成：{len(rows)} 次评估，stage_one 统计预检通过 {n_stat_pass} 个，"
          f"含正交门全通过 {n_full_pass} 个，用时 {time.perf_counter() - t0:.0f}s", flush=True)
    for r in rows:
        if r.get("cold_track_passed"):
            print(f"  ★ [cold] {r['name']}: resid_ic={r.get('resid_ic')} "
                  f"train={r.get('train_resid_ic')} proxy={r.get('crowd_proxy_corr')} "
                  f"root={r.get('root_op')}", flush=True)
        elif not r.get("fails"):
            print(f"  ★ {r['name']}: ic={r['ic']} icir={r['icir']} cov={r['coverage']} "
                  f"ac={r['cs_autocorr']} turn={r['turnover_rebalance']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

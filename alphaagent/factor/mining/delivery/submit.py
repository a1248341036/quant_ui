"""挖掘会话内因子交付入库。"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.ingest import (
    align_values_to_rows,
    clip_values,
    compute_ingest_metrics,
    ingest_factor,
    materialize_factor,
)
from alphaagent.factor.metrics import annualized_long_group_excess_return
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.zoo import DEFAULT_FACTORLIB_ROOT, FactorZoo, SimilarityMatrix
from alphaagent.factor.zoo.similarity import (
    SIMILARITY_BASIS_SAMPLED,
    SIMILARITY_KIND,
    cross_sectional_pearson_mean,
)
from alphaagent.factor.mining.registry_io import (
    load_mining_registry,
    set_candidate_review,
    set_candidate_promotion,
    upsert_mining_registry,
    write_candidate_registry,
)
from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria
from alphaagent.factor.mining.delivery_checker import DeliveryChecker
from alphaagent.factor.mining.runlog import log_step
from core import factor_categories


def _candidate_registry_similarity(
    cand_values: np.ndarray,
    panel: pd.DataFrame,
    candidate_registry_path: Path,
    *,
    exclude_factor_id: str | None = None,
    min_pairs: int = 30,
    top_k: int = 3,
    cache=None,
) -> dict[str, Any] | None:
    """候选因子与候选 registry 中已有因子的截面 Pearson 相似度。

    候选 registry 中的因子只有 DSL 表达式、无 dense values，需要在会话域
    panel 上重新求值。返回与 SimilarityMatrix.cross_sectional_neighbor_report
    同结构的 dict，或 None（registry 为空/无可比因子时）。
    """
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel

    registry = load_mining_registry(candidate_registry_path)
    if not registry:
        return None

    new_series = pd.Series(np.asarray(cand_values, dtype=np.float64), index=panel.index)
    corrs: list[tuple[str, float, str]] = []  # (factor_id, corr, name)

    # 逐日截面 Pearson 相关均值（在 panel 的 MultiIndex 上直接计算，
    # 不依赖 cross_sectional_pearson_mean，因为后者需要 RowIndex 而非 MultiIndex）。
    def _panel_cs_pearson_mean(a: pd.Series, b: pd.Series, *, min_pairs: int = 30) -> float:
        vals: list[float] = []
        for ts, a_sub in a.groupby(level="datetime", sort=False):
            b_sub = b.xs(ts, level="datetime")
            av = a_sub.to_numpy(dtype=np.float64)
            bv = b_sub.to_numpy(dtype=np.float64)
            mask = np.isfinite(av) & np.isfinite(bv)
            if mask.sum() < min_pairs:
                continue
            av, bv = av[mask], bv[mask]
            av = av - av.mean()
            bv = bv - bv.mean()
            denom = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
            if denom <= 0.0:
                continue
            vals.append(float((av * bv).sum() / denom))
        if not vals:
            return float("nan")
        return float(np.mean(vals))

    for fid, entry in sorted(registry.items()):
        if fid == exclude_factor_id or not isinstance(entry, dict):
            continue
        expr_text = str(entry.get("expr") or "").strip()
        if not expr_text:
            continue
        try:
            if cache is not None:
                old_raw = cache.evaluate(expr_text, panel, lambda: eval_factor(expr_text, panel))
            else:
                old_raw = eval_factor(expr_text, panel)
            if not isinstance(old_raw, pd.Series):
                continue
            old_aligned = align_series_to_panel(old_raw, panel)
        except Exception:
            continue
        old_series = pd.Series(np.asarray(old_aligned, dtype=np.float64), index=panel.index)
        corr = _panel_cs_pearson_mean(new_series, old_series, min_pairs=min_pairs)
        if np.isfinite(corr):
            name = str(entry.get("name") or fid)
            corrs.append((fid, float(corr), name))

    if not corrs:
        return None

    corrs.sort(key=lambda p: abs(p[1]), reverse=True)
    max_abs = max(abs(c) for _, c, _ in corrs)
    top_slice = corrs[:top_k] if top_k > 0 else []

    return {
        "kind": SIMILARITY_KIND,
        "basis": "candidate_registry_reeval",
        "max_abs_corr": max_abs,
        "top_neighbors": [
            {"factor_id": fid, "name": name, "cs_corr": corr}
            for fid, corr, name in top_slice
        ],
    }


def slug_factor_id(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "factor"


def _detect_replacement(
    similarity_report: dict[str, Any],
    new_metrics: dict[str, Any],
    zoo: "FactorZoo",
) -> dict[str, Any] | None:
    """检测新因子是否显著优于高相关旧因子，标记替换候选。

    触发条件：与正式库因子相关性 >= 0.4 且新因子 IC/ICIR 均比旧因子高 50% 以上。
    仅标记不执行——返回 replacement_candidate dict 供人工确认。
    """
    max_corr = float(similarity_report.get("max_abs_corr", 0))
    if max_corr < 0.4:
        return None

    top_neighbors = similarity_report.get("top_neighbors") or []
    if not top_neighbors:
        return None

    old_neighbor = top_neighbors[0]
    old_factor_id = str(old_neighbor.get("factor_id") or "")
    if not old_factor_id:
        return None

    # 从 catalog 读取旧因子元信息，指标存在 extra["metrics"] 中
    old_meta = zoo.catalog.get(old_factor_id)
    if old_meta is None:
        return None

    old_metrics = old_meta.extra.get("metrics", {}) if isinstance(old_meta.extra, dict) else {}
    old_ic = abs(float(old_metrics.get("ic", 0) or 0))
    old_icir = abs(float(old_metrics.get("icir", 0) or 0))
    if old_ic < 0.001:
        return None

    new_ic = abs(float(new_metrics.get("ic", 0) or 0))
    new_icir = abs(float(new_metrics.get("icir", 0) or 0))

    # 替换条件：新因子 IC 和 ICIR 均比旧因子高 50% 以上，且新因子 IC >= 0.03
    if new_ic > old_ic * 1.5 and new_icir > old_icir * 1.5 and new_ic >= 0.03:
        return {
            "old_factor_id": old_factor_id,
            "old_factor_name": old_neighbor.get("name"),
            "old_cs_corr": abs(float(old_neighbor.get("cs_corr", 0))),
            "old_ic": round(old_ic, 6),
            "old_icir": round(old_icir, 6),
            "new_ic": round(new_ic, 6),
            "new_icir": round(new_icir, 6),
            "improvement_ratio": round(new_ic / old_ic, 2) if old_ic else None,
            "action": "replace_after_approval",
            "note": "检测到新因子显著优于旧因子，建议人工确认后 deprecate 旧因子",
        }
    return None



_INGEST_METRICS_CACHE_MAX = 96


def _ingest_metrics_fingerprint(policy: IngestPolicy) -> str:
    """冻结 dataclass 的标量字段指纹：窗口/口径任一不同 → 缓存键不同。"""
    fields = {f.name: getattr(policy, f.name) for f in dataclasses.fields(policy)}
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _cached_ingest_metrics(
    cache: dict,
    key: tuple[str, str],
    compute: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """submit 内 train/val/全窗 ingest 指标的会话级缓存。

    同一因子值数组 + 同一窗口口径在单次 run 内必复现，直接复用免去 20~30s
    的重复指标计算；命中返回拷贝，调用方原地修改（如补 val_long_excess）
    不污染缓存；FIFO 上限防长 run 膨胀。
    """
    hit = cache.get(key)
    if hit is not None:
        return dict(hit)
    metrics = compute()
    cache[key] = dict(metrics)
    while len(cache) > _INGEST_METRICS_CACHE_MAX:
        cache.pop(next(iter(cache)))
    return metrics


class FactorSubmitService:
    """两阶段提交：海选候选池，再精筛进入正式 FactorZoo。"""

    def __init__(
        self,
        service: StockEvalService,
        *,
        factorlib_path: Path | None = None,
        registry_path: Path | None = None,
        expr_dir: Path | None = None,
        repo_root: Path,
        research_mode: str = "technical",
        max_cs_corr: float = 0.8,
        delivery_policy: dict[str, Any] | None = None,
        similar_top_k: int = 3,
        overwrite: bool = False,
        auto_realign_panel: bool = True,
    ) -> None:
        self.service = service
        self.research_mode = research_mode
        self.factorlib_path = (
            Path(factorlib_path).expanduser().resolve()
            if factorlib_path is not None
            else factor_categories.production_dir(research_mode)
        )
        self.registry_path = (
            Path(registry_path).expanduser().resolve()
            if registry_path is not None
            else factor_categories.production_registry_path(research_mode)
        )
        self.expr_dir = (
            Path(expr_dir).expanduser().resolve()
            if expr_dir is not None
            else factor_categories.production_expr_dir(research_mode)
        )
        self.repo_root = Path(repo_root).resolve()
        self.max_cs_corr = max_cs_corr
        self.delivery_policy = delivery_policy or {}
        # 两阶段门槛单一来源：由 research_spec 注入的 delivery_policy 构造，
        # 缺失键回落 canonical 默认（与 DEFAULT_RESEARCH_SPEC 一致）。
        # 下游判定与 prompt 渲染统一读 self.criteria，不再散落硬编码默认值。
        self.criteria = DeliveryCriteria.from_spec(self.delivery_policy)
        self.checker = DeliveryChecker(self.criteria)
        self.similar_top_k = similar_top_k
        self.overwrite = overwrite
        self.auto_realign_panel = auto_realign_panel

    def _ensure_factorzoo(self, session) -> "FactorZoo":
        """打开生产因子库；首次使用未初始化时用会话域 panel 自动初始化。

        原 submit() 中两处相同的 FileNotFoundError→init_library 块收敛于此，
        避免 LLM 对未初始化 production 目录反复碰壁。
        """
        try:
            return FactorZoo.open(self.factorlib_path)
        except FileNotFoundError:
            # 因子库未初始化（首次使用新 research_mode 的 production 目录）。
            # 用会话域 panel 自动初始化，避免 LLM 反复碰壁。
            from alphaagent.factor.zoo.index import init_library

            panel = session.panel
            if panel is None or len(panel) == 0:
                raise ValueError("session_panel_empty")
            from alphaagent.data.adapters.cnequity import is_cne_source

            panel_path_str = ""
            ctx = session.ctx
            if ctx.panel_path and is_cne_source(ctx.panel_path):
                panel_path_str = "D:\\Quant\\quant_ui\\cne:"
            elif ctx.panel_path:
                panel_path_str = str(Path(ctx.panel_path).resolve())
            init_library(
                self.factorlib_path,
                panel=panel,
                panel_path=Path(panel_path_str) if panel_path_str else None,
                n_sample_rows=200_000,
                max_factors=2048,
            )
            return FactorZoo.open(self.factorlib_path)

    @property
    def candidate_factorlib_path(self) -> Path:
        return factor_categories.candidate_dir(self.research_mode)

    @property
    def candidate_registry_path(self) -> Path:
        return factor_categories.candidate_registry_path(self.research_mode)

    @property
    def candidate_expr_dir(self) -> Path:
        return factor_categories.candidate_expr_dir(self.research_mode)

    def submit(
        self,
        session_id: str,
        *,
        multi_line_expr: str,
        factor_name: str,
        comment: str,
        evaluation_evidence: dict[str, Any] | None = None,
        review_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        orthogonality_hook: Callable[[], dict[str, Any]] | None = None,
        interaction: dict[str, Any] | None = None,
        rebalance_freq: str | None = None,
    ) -> dict[str, Any]:
        expr = multi_line_expr.strip()
        if not expr:
            return {
                "ok": False,
                "stored": False,
                "error": "multi_line_expr_required_non_empty_string",
                "error_type": "ToolArgumentsError",
            }
        if not str(comment).strip():
            return {
                "ok": False,
                "stored": False,
                "error": "comment_required_non_empty_string",
                "error_type": "ToolArgumentsError",
            }

        factor_id = slug_factor_id(factor_name)
        name = str(factor_name).strip() or factor_id
        log_step("submit.start", name)

        engine_cfg = self.delivery_policy.get("production", {}).get("engine_gate") or {}
        allowed_freqs = [str(f).lower() for f in (engine_cfg.get("allowed_freqs") or ["daily"])]
        chosen_freq = str(rebalance_freq or engine_cfg.get("freq") or "daily").lower()
        if chosen_freq not in allowed_freqs:
            return {
                "ok": False,
                "stored": False,
                "error": f"invalid_rebalance_freq: {chosen_freq} (allowed={allowed_freqs})",
                "error_type": "ToolArgumentsError",
            }

        try:
            session = self.service.sessions.get(session_id)
        except KeyError:
            return {
                "ok": False,
                "stored": False,
                "error": f"session_not_found: {session_id}",
                "error_type": "SessionError",
            }

        ctx = session.ctx
        try:
            zoo = self._ensure_factorzoo(session)
        except ValueError as e:
            if str(e) == "session_panel_empty":
                return {
                    "ok": False,
                    "stored": False,
                    "error": "session_panel_empty",
                    "error_type": "SessionError",
                }
            raise

        # ③ 会话域复用：submit 全程在挖掘会话驻内存 panel 上评估，
        #    不再全量重载因子库域 panel（两者行集本就不同——库含全部历史；
        #    旧行为的二次构建曾触发 float64 合并的 8.69GiB OOM）。
        #    相似度抽样与最终入库通过键映射 (datetime, instrument) 对齐。
        panel = session.panel
        if panel is None or len(panel) == 0:
            return {
                "ok": False,
                "stored": False,
                "error": "session_panel_empty",
                "error_type": "SessionError",
            }

        stage_one_policy = IngestPolicy.from_context(
            ctx, max_cs_corr=self.criteria.candidate.max_abs_corr, similar_top_k=self.similar_top_k
        )
        # ①+③ 会话域物化：长度恒等于 panel 行数，指标直接可算。
        materialized = materialize_factor(expr, panel, cache=getattr(session, "factor_cache", None))
        cand_values = materialized.values
        if stage_one_policy.clip_pct is not None:
            lo_pct, hi_pct = float(stage_one_policy.clip_pct[0]), float(stage_one_policy.clip_pct[1])
            cand_values, _ = clip_values(cand_values, lower_pct=lo_pct, upper_pct=hi_pct)
        values_by_key = pd.Series(cand_values, index=panel.index)
        values_fp = hashlib.sha1(
            np.ascontiguousarray(cand_values, dtype=np.float32).tobytes()
        ).hexdigest()
        ingest_cache: dict = getattr(session, "_ingest_metrics_cache", None)
        if ingest_cache is None:
            ingest_cache = session._ingest_metrics_cache = {}

        def _ingest_metrics_cached(policy: IngestPolicy) -> dict[str, Any]:
            key = (values_fp, _ingest_metrics_fingerprint(policy))
            return _cached_ingest_metrics(ingest_cache, key, lambda: compute_ingest_metrics(cand_values, panel, policy))

        # ① 统计门槛前置（train-only 准入口径 + 换手可行性）：
        #    - 准入看 train 窗口，防止 val 衰减被混合窗口稀释；
        #    - cs_autocorr 硬门淘汰排名日度剧变的不可交付因子；
        #    - 任一不达标直接拒绝，跳过最贵的相似度对比。
        metrics_train = _ingest_metrics_cached(
            dataclasses.replace(stage_one_policy, val_end=ctx.train_end)
        )
        # 组合换手预检：quantile_portfolio 提前到门槛前算一次（全窗口径——
        # 换手是因子结构性属性，与窗口无关），供 stage_one 换手硬门与最终报告复用，
        # 避免高换手因子走完全流程才在 engine_gate 被拒。
        from alphaagent.factor.metrics import quantile_portfolio_metrics
        # holding_days 对齐 label 名义持有期（label_20d → 20）：避免 20 日收益被
        # 逐日重叠计入 20 次，组合年化/回撤/夏普全部失真（曾致回撤虚标 99%）。
        label_digits = "".join(ch for ch in str(ctx.label_col) if ch.isdigit())
        qp_holding_days = max(1, int(label_digits) if label_digits else 1)
        qp_metrics = quantile_portfolio_metrics(
            pd.Series(cand_values, index=panel.index), panel[ctx.label_col],
            n_groups=10, cost_bps=0.0, holding_days=qp_holding_days,
        )
        metrics_train["quantile_portfolio"] = qp_metrics

        # ── 盲测终审（stage_one 之前，不通过不进候选池）──────────────────
        # test 段从未参与 train/val/engine_gate，是最干净的样本外验证。
        # IC 保留比 ≥ 0.50 + 方向一致性 → 不通过直接拒绝，不消耗后续算力。
        test_start = ctx.test_start
        test_end = ctx.resolved_test_end()
        test_report: dict[str, Any] | None = None
        try:
            test_policy = dataclasses.replace(
                stage_one_policy,
                train_start=test_start,
                val_end=test_end,
            )
            test_metrics = _ingest_metrics_cached(test_policy)
            test_report = {
                "range": {"start": test_start, "end": test_end},
                "ic": test_metrics.get("ic"),
                "icir": test_metrics.get("icir"),
                "rank_ic": test_metrics.get("rank_ic"),
                "coverage": test_metrics.get("factor_coverage"),
            }
            t_ic = metrics_train.get("ic")
            s_ic = test_metrics.get("ic")
            if t_ic is not None and s_ic is not None:
                t_sign = 1 if float(t_ic) >= 0 else -1
                s_sign = 1 if float(s_ic) >= 0 else -1
                test_report["sign_consistent"] = (t_sign == s_sign)
                if abs(float(t_ic)) > 1e-12:
                    test_report["ic_retention"] = round(abs(float(s_ic) / float(t_ic)), 4)
        except Exception:
            test_report = {"error": "test_evaluation_failed"}
            test_metrics = {}

        # 盲测终审门禁（enabled=False 时跳过）
        blind_result = self.checker.blind_test(metrics_train, test_metrics)
        log_step(
            "submit.blind_test",
            name,
            test_ic=test_report.get("ic"),
            retention=test_report.get("ic_retention"),
            sign=test_report.get("sign_consistent"),
            fail=(blind_result.fail_reasons or None) if not blind_result.passed else None,
        )
        if not blind_result.passed:
            payload = {
                "ok": False,
                "stored": False,
                "factor_id": factor_id,
                "factor_name": name,
                "comment": comment.strip(),
                "interaction": interaction,
                "eval_range": {"start": ctx.train_start, "end": ctx.val_end},
                "metrics": dict(metrics_train),
                "test_holdout": test_report,
                "candidate_stored": False,
                "rebalance_freq": chosen_freq,
                "delivery_check": {
                    "blind_test": {"passed": False, "fail_reasons": blind_result.fail_reasons},
                    "stage_one": {"passed": False, "fail_reasons": []},
                    "stage_two": {"passed": False, "fail_reasons": []},
                },
                "skipped_reason": f"blind_test_failed:{','.join(blind_result.fail_reasons)}",
                "error_type": "BlindTestError",
                "error": f"blind_test_failed:{','.join(blind_result.fail_reasons)}",
            }
            return payload

        gate_reasons = self.checker.stage_one_stats(metrics_train).fail_reasons
        log_step(
            "submit.stage_one_stats",
            name,
            ic=metrics_train.get("ic"),
            icir=metrics_train.get("icir"),
            coverage=metrics_train.get("factor_coverage"),
            autocorr=metrics_train.get("cs_pearson_autocorr"),
            turnover=(metrics_train.get("quantile_portfolio") or {}).get("avg_daily_side_turnover"),
            fail=gate_reasons or None,
        )

        val_metrics: dict[str, Any] = {}
        if not gate_reasons and ctx.val_start > ctx.train_end:
            # 样本外保留比：|val_ic|/|train_ic| ≥ 阈值且方向不反转
            val_metrics = _ingest_metrics_cached(
                dataclasses.replace(stage_one_policy, train_start=ctx.val_start)
            )
            retention_reasons = self.checker.stage_one_val_retention(
                metrics_train, val_metrics
            ).fail_reasons
            gate_reasons += retention_reasons
            log_step(
                "submit.val_retention",
                name,
                val_ic=val_metrics.get("ic"),
                val_icir=val_metrics.get("icir"),
                retention=round(abs(float(val_metrics.get("ic") or 0) / float(metrics_train.get("ic") or 1e-12)), 4),
                fail=retention_reasons or None,
            )
            # val 多头端毛值超额（方向自适应）：IC 为正不代表多头组合为正，
            # alpha 可能全在空头端/中段排名——纯多头可交易口径必须单独为正。
            label_col = stage_one_policy.label_col
            if label_col in panel.columns:
                dt_level = panel.index.get_level_values("datetime")
                val_rows = (dt_level >= pd.Timestamp(ctx.val_start)) & (dt_level <= pd.Timestamp(ctx.val_end))
                f_val = pd.Series(cand_values, index=panel.index)[val_rows]
                l_val = panel[label_col][val_rows]
                if len(f_val) > 0:
                    dir_sign = 1 if float(metrics_train.get("ic") or 0.0) >= 0 else -1
                    val_metrics["val_long_excess"] = annualized_long_group_excess_return(
                        f_val, l_val, direction=dir_sign, holding_days=qp_holding_days
                    )

        similarity_report: dict[str, Any] | None = None
        candidate_similarity: dict[str, Any] | None = None
        production_similarity: dict[str, Any] | None = None
        if not gate_reasons:
            # ── 正式库相似度（抽样行快速口径）──
            # 正式库准入只查正式库：候选池内部冗余不卡正式库晋升
            # （候选池冗余由 dedup_candidate_factors.py 主动去重管理，
            #   而非让候选池因子互相挡死正式库准入——历史死锁根源）。
            zoo_report: dict[str, Any] | None = None
            if zoo.n_factors > 0:
                cand_sample = align_values_to_rows(
                    values_by_key, zoo.index.rows.iloc[zoo.index.sample_row_ids]
                )
                sim_matrix = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
                zoo_report = sim_matrix.cross_sectional_neighbor_report(
                    zoo, None, top_k=stage_one_policy.similar_top_k,
                    candidate_sample=cand_sample,
                )
            production_similarity = zoo_report if zoo_report is not None else {
                "kind": SIMILARITY_KIND,
                "basis": SIMILARITY_BASIS_SAMPLED,
                "max_abs_corr": 0.0,
                "top_neighbors": [],
            }
            similarity_report = production_similarity

            # ── 候选 registry 相似度（在会话域 panel 上对已有候选 DSL 重新求值）──
            # 仅作诊断/报告，不参与正式库准入拦截。
            candidate_similarity = _candidate_registry_similarity(
                cand_values, panel, self.candidate_registry_path,
                exclude_factor_id=factor_id,
                top_k=stage_one_policy.similar_top_k,
                cache=getattr(session, "factor_cache", None),
            )
        # 准入判定：统计/换手/保留比任一不过即拒；全过再叠加相似度 corr 检查
        # （只查正式库，见上）。
        stage_one_ok = not gate_reasons
        stage_one_reasons = list(gate_reasons)
        if stage_one_ok:
            corr_result = self.checker.stage_one_correlation(production_similarity)
            stage_one_ok = corr_result.passed
            stage_one_reasons = corr_result.fail_reasons
        log_step(
            "submit.stage_one",
            name,
            passed=stage_one_ok,
            max_corr=(production_similarity or {}).get("max_abs_corr"),
            fail=stage_one_reasons or None,
        )

        # 上报/存档指标仍用全窗口（与历史 registry 口径一致），附 train/val 分解。
        metrics = _ingest_metrics_cached(stage_one_policy)
        reported = dict(metrics)
        for src, prefix in ((metrics_train, "train"), (val_metrics, "val")):
            for key in ("ic", "icir", "rank_ic"):
                v = src.get(key)
                if v is not None and np.isfinite(float(v)):
                    reported[f"{prefix}_{key}"] = round(float(v), 6)
        t_ic, v_ic = metrics_train.get("ic"), val_metrics.get("ic")
        if t_ic is not None and v_ic is not None and abs(float(t_ic)) > 1e-12:
            reported["val_ic_retention"] = round(abs(float(v_ic) / float(t_ic)), 4)
        vle = val_metrics.get("val_long_excess")
        if vle is not None and np.isfinite(float(vle)):
            reported["val_long_excess"] = round(float(vle), 6)

        # 组合层收益指标（多头年化/夏普等）：复用门槛前的 qp_metrics，不重算。
        if stage_one_ok:
            reported["quantile_portfolio"] = {
                k: (round(float(v), 6) if isinstance(v, (int, float)) and np.isfinite(float(v)) else v)
                for k, v in qp_metrics.items()
                if k not in {"group_means"}
            }
        # test 段指标写入 reported（供 registry 检索）
        if test_report:
            for key in ("ic", "icir", "rank_ic"):
                v = test_report.get(key)
                if v is not None and np.isfinite(float(v)):
                    reported[f"test_{key}"] = round(float(v), 6)
            ret = test_report.get("ic_retention")
            if ret is not None:
                reported["test_ic_retention"] = ret
            sc = test_report.get("sign_consistent")
            if sc is not None:
                reported["test_sign_consistent"] = sc
        metrics = reported

        payload: dict[str, Any] = {
            "ok": False,
            "stored": False,
            "factor_id": factor_id,
            "factor_name": name,
            "comment": comment.strip(),
            "interaction": interaction,
            "eval_range": {"start": ctx.train_start, "end": ctx.val_end},
            "metrics": metrics,
            "similarity": similarity_report,
            "candidate_similarity": candidate_similarity,
            "candidate_stored": False,
            "rebalance_freq": chosen_freq,
            "test_holdout": test_report,
            "delivery_check": {
                "blind_test": {"passed": True, "fail_reasons": []},
                "stage_one": {"passed": stage_one_ok, "fail_reasons": stage_one_reasons},
                "stage_two": {"passed": False, "fail_reasons": []},
            },
        }

        if not stage_one_ok:
            payload["skipped_reason"] = f"stage_one_failed:{','.join(stage_one_reasons)}"
            payload["error_type"] = "StageOneDeliveryCheckError"
            payload["error"] = payload["skipped_reason"]
            return payload

        # ── 正交性检查（stage_one 统一查，不再只在 approve 后触发）──
        # 所有通过统计门槛的因子（无论后续 verdict 是 approve 还是 revise）
        # 都需经过正交性检查，避免与正式库/候选池已有因子高度重复。
        orthogonality: dict[str, Any] = {
            "passed": True,
            "skipped_reason": "hook_not_configured",
        }
        if orthogonality_hook is not None:
            orthogonality = orthogonality_hook()
        payload["offline_orthogonality"] = orthogonality
        if not bool(orthogonality.get("passed")):
            reason = str(orthogonality.get("error") or orthogonality.get("skipped_reason") or "correlation_threshold")
            payload["promotion_status"] = "offline_orthogonality_blocked"
            payload["skipped_reason"] = f"offline_orthogonality_failed:{reason}"
            payload["error_type"] = "OfflineOrthogonalityError"
            payload["error"] = payload["skipped_reason"]
            return payload

        review: dict[str, Any] | None = None
        if review_hook is not None:
            review = review_hook({
                "multi_line_expr": expr,
                "factor_name": name,
                "comment": comment.strip(),
                "interaction": interaction,
                "candidate_stored": True,
                "metrics": metrics,
            })
        review_verdict = str((review or {}).get("verdict") or "").lower()
        review_status = (
            {"approve": "approved", "revise": "revise", "reject": "rejected"}.get(review_verdict, "pending_review")
        )
        log_step(
            "submit.review",
            name,
            verdict=review_verdict or "none",
            source=(review or {}).get("source"),
            canonical=str((review or {}).get("canonical_form") or "")[:80] or None,
            novelty=(review or {}).get("novelty"),
        )
        payload.update(
            review_status=review_status,
        )

        if review_verdict == "reject":
            # 仅 reject 硬拦（抄袭/经典变换等垃圾候选不进任何库）；
            # revise 只是"暂缓转正式"，候选池仍按统计门槛照常收纳。
            payload["delivery_check"]["stage_two"] = {
                "passed": False,
                "fail_reasons": ["factor_review"],
            }
            payload["review"] = review
            payload["promotion_status"] = "review_rejected"
            payload["skipped_reason"] = f"factor_review:{review_verdict}"
            payload["error_type"] = "FactorReviewRejected"
            payload["error"] = f"factor_review_{review_verdict}_blocked"
            return payload

        # 统计达标 → 先入候选池（registry_only）；Reviewer 意见只影响是否继续冲正式库。
        candidate_reg, candidate_dsl = write_candidate_registry(
            self.candidate_registry_path,
            factor_id=factor_id,
            name=name,
            comment=comment.strip(),
            expr=expr,
            expr_dir=self.candidate_expr_dir,
            repo_root=self.repo_root,
            policy=stage_one_policy,
            metrics=metrics,
            similarity=similarity_report,
            source="submit_stage_one",
            evaluation_evidence=evaluation_evidence,
            interaction=interaction,
            eval_label=str(ctx.label_col),
            data_fingerprint={
                "panel_path": str(ctx.panel_path),
                "index_hash": zoo.manifest.index_hash,
                "n_rows": int(zoo.manifest.n_rows),
            },
        )
        if review is not None:
            set_candidate_review(
                self.candidate_registry_path,
                factor_id=factor_id,
                review=review,
                promotion_status="pending",
            )
        payload.update(
            candidate_stored=True,
            candidate_storage="registry_only",
            candidate_registry_path=candidate_reg,
            candidate_dsl_path=candidate_dsl,
        )
        log_step("submit.candidate_stored", name, path=str(candidate_reg))

        # review 意见已记录：reject 在上方硬拦（抄袭/经典暴露不进任何库）。
        # revise / pending_review 不阻断晋升——stage_two 统计门槛 + engine_gate
        # 才是正式库准入的最终裁决，与系统提示词"Reviewer 意见仅供参考改进方向，
        # 不阻断提交"保持一致（历史行为把 revise 当硬拦，导致达标候选堆积候选池）。
        payload["review"] = review

        # stage_two 相似度只看正式库（similarity_report 已是正式库口径）：
        # 候选池内部冗余不卡正式库准入，正式库准入只关心与正式库已有因子是否重复。
        stage_two_result = self.checker.stage_two(
            metrics_train, val_metrics, similarity_report
        )
        stage_two_ok = stage_two_result.passed
        stage_two_reasons = stage_two_result.fail_reasons
        payload["delivery_check"]["stage_two"] = {
            "passed": stage_two_ok,
            "fail_reasons": stage_two_reasons,
        }
        log_step(
            "submit.stage_two",
            name,
            passed=stage_two_ok,
            train_ic=metrics_train.get("ic"),
            val_ic=val_metrics.get("ic"),
            fail=stage_two_reasons or None,
        )
        if not stage_two_ok:
            set_candidate_promotion(
                self.candidate_registry_path,
                factor_id=factor_id,
                promotion_status="stage_two_failed",
            )
            payload["skipped_reason"] = f"stage_two_failed:{','.join(stage_two_reasons)}"
            payload["error_type"] = "StageTwoDeliveryCheckError"
            payload["error"] = payload["skipped_reason"]
            return payload

        # ── 替换检测：新因子质量显著优于高相关旧因子时标记替换候选 ──
        # 仅检测和标记，不自动删除旧因子；需人工确认后再执行 deprecate。
        if similarity_report and similarity_report.get("top_neighbors"):
            _replacement = _detect_replacement(
                similarity_report, metrics_train, zoo
            )
            if _replacement is not None:
                payload["replacement_candidate"] = _replacement

        # engine_gate 配置从原始 delivery_policy 读取（缺失时跳过回测门禁，
        # 与重构前语义一致）；criteria 的 engine_gate 仅用于默认值兜底与 prompt 渲染。
        engine_gate_cfg = (self.delivery_policy.get("production") or {}).get("engine_gate")
        if isinstance(engine_gate_cfg, dict) and engine_gate_cfg.get("enabled"):
            from alphaagent.factor.mining.engine_gate import run_engine_gate
            ic_sign = 1 if float(metrics.get("ic") or 0.0) >= 0 else -1
            # 会话域回测：cand_values 与 session panel 行序一一对应。
            gate = run_engine_gate(
                panel,
                cand_values,
                val_start=ctx.val_start,
                val_end=ctx.val_end,
                direction=ic_sign,
                policy={**engine_gate_cfg, "freq": chosen_freq},
                asset_type=getattr(ctx, "asset_type", "stock"),
            )
            payload["engine_backtest"] = gate
            log_step(
                "submit.engine_gate",
                name,
                passed=bool(gate.get("passed")),
                freq=chosen_freq,
                fail=gate.get("fail_reasons") or None,
            )
            if not gate.get("passed"):
                set_candidate_promotion(
                    self.candidate_registry_path,
                    factor_id=factor_id,
                    promotion_status="engine_gate_failed",
                )
                payload["skipped_reason"] = (
                    f"engine_gate_failed:{','.join(gate.get('fail_reasons') or [])}"
                )
                payload["error_type"] = "EngineGateError"
                payload["error"] = payload["skipped_reason"]
                return payload

        # ── test 段 engine_gate 回测（补充到 test_report，供报告展示）─────
        # 盲测终审门禁已在 stage_one 之前执行（IC 保留比 + 方向一致性）。
        # 此处补算 test 段 engine_gate 回测，仅作为诊断写入 payload/registry，
        # 不再卡准入（硬门已由盲测终审守住）。
        if test_report and "error" not in test_report:
            try:
                if isinstance(engine_gate_cfg, dict) and engine_gate_cfg.get("enabled"):
                    from alphaagent.factor.mining.engine_gate import run_engine_gate
                    ic_sign = 1 if float(metrics.get("ic") or 0.0) >= 0 else -1
                    test_gate = run_engine_gate(
                        panel,
                        cand_values,
                        val_start=test_start,
                        val_end=test_end,
                        direction=ic_sign,
                        policy={**engine_gate_cfg, "freq": chosen_freq},
                        asset_type=getattr(ctx, "asset_type", "stock"),
                    )
                    test_report["engine_gate"] = {
                        "passed": test_gate.get("passed"),
                        "fail_reasons": test_gate.get("fail_reasons"),
                        "metrics": test_gate.get("metrics"),
                    }
            except Exception:
                pass

        if test_report:
            payload["test_holdout"] = test_report

        # 真正入库才做一次 canonical 对齐（会话域 → 库行序），指标复用免重算。
        canonical_values = align_values_to_rows(values_by_key, zoo.index.rows)
        result = ingest_factor(
            zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=None,
            policy=IngestPolicy.from_context(ctx, max_cs_corr=self.criteria.production.max_abs_corr, similar_top_k=self.similar_top_k),
            stored_values=canonical_values,
            metrics_override=metrics,
            overwrite=self.overwrite,
        )
        if not result.stored:
            set_candidate_promotion(
                self.candidate_registry_path,
                factor_id=factor_id,
                promotion_status="production_ingest_failed",
            )
            payload["skipped_reason"] = result.skipped_reason
            payload["error_type"] = "ProductionIngestError"
            payload["error"] = result.skipped_reason or "production_ingest_failed"
            return payload

        policy = IngestPolicy.from_context(ctx, max_cs_corr=self.criteria.production.max_abs_corr, similar_top_k=self.similar_top_k)
        reg_path, dsl_path = upsert_mining_registry(
            self.registry_path,
            factor_id=factor_id,
            name=name,
            comment=comment.strip(),
            expr=expr,
            expr_dir=self.expr_dir,
            repo_root=self.repo_root,
            policy=policy,
            metrics=result.metrics,
            similarity=result.similarity,
            ingest_status="production",
            source="submit_stage_two",
            interaction=interaction,
            eval_label=str(ctx.label_col),
        )
        set_candidate_promotion(
            self.candidate_registry_path,
            factor_id=factor_id,
            promotion_status="promoted",
        )
        payload.update(
            ok=True,
            stored=True,
            promotion_status="promoted",
            registry_path=reg_path,
            dsl_path=dsl_path,
            factorlib_path=str(self.factorlib_path),
            skipped_reason=None,
        )
        log_step(
            "submit.promoted",
            name,
            train_ic=metrics_train.get("ic"),
            val_ic=val_metrics.get("ic"),
            registry=str(reg_path),
        )

        return payload


def default_factorlib_path(repo_root: Path) -> Path:
    _ = repo_root  # 兼容旧签名；AlphaAgent 使用固定 artifacts 路径
    return DEFAULT_FACTORLIB_ROOT

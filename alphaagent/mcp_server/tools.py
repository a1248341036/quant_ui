# -*- coding: utf-8 -*-
"""AlphaAgent MCP server · 工具实现层。

设计原则（详见 `docs/review/agent_tool_mcp_inventory_20261007.md`）：

1. **门槛单一真源**：所有判定读 `effective_research_spec` + `DeliveryCriteria`，不硬编码数值；
2. **读/写分离**：读工具随时可用；写工具（`submit_factor` / `memory_record`）必须 `confirm=true`
   且落审计日志；
3. **评估口径与真实 run 一致**：同一 `StockEvalService`、同一 label（按档位）；
4. **会话串行复用**：单会话 panel 6–8GB，本进程内按 (mode, label, fundamentals) 缓存一个，
   切换时释放旧的；不做并发多会话；
5. **批量优先**：`eval_batch` 一次可评数十个表达式（这是 harness 直驱相对逐次工具调用的核心优势）。

写工具刻意走 `FactorSubmitService.submit`（真实交付链路，含 stage_one→盲测→stage_two→engine_gate）
与 `memory.record_tool_result`（记忆写回），从而不让"直驱"绕过交付判定与跨 run 学习。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# 档位 → 评估 label / 交付调仓频率（与 core/research_modes.py 一致）
MODE_LABEL = {
    "technical": "label_1d_open_to_open",
    "technical_weekly": "label_5d_close_to_close",
    "technical_monthly": "label_20d_close_to_close",
    "fundamental": "label_20d_close_to_close",
}
MODE_FREQ = {
    "technical": "daily",
    "technical_weekly": "weekly",
    "technical_monthly": "monthly",
    "fundamental": "monthly",
}
MAX_BATCH = 60


class ToolError(Exception):
    """工具级错误：以 MCP result(isError=true) 返回，供 agent 自行修正后重试。"""


def _norm_universe(universe: str) -> str:
    """规范化池名（strip+lower）并早校验；非法名转 ToolError（附可用清单）。

    所有带 ``universe`` 的工具在入口统一调用：响应/元数据回显规范化名，
    与会话缓存键、面板口径严格一致（OCR review D2，2026-10-09）。
    """
    from alphaagent.factor.mining.eval.universe import UNIVERSE_ALL, parse_universe
    try:
        return parse_universe(str(universe or UNIVERSE_ALL))["name"]  # 非字符串入参（如数字）也统一走 ValueError→ToolError（OCR R2-3）
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


class SessionLock:
    """跨进程会话锁（OS 文件锁，进程退出自动释放，无陈旧锁问题）。

    为什么需要：每个 MCP 客户端（DSH / Codex / Cursor）会各起一个 stdio server 进程，
    而每个进程的评估会话要占 6–8GB panel。多客户端同时挖矿会直接把内存打爆，
    故同一时刻只允许一个进程持有"重会话"；后来者拿到明确错误（而不是 OOM）。
    只读/轻量工具（门槛/字段/算子/预检/台账/记忆查询）不受影响。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: Any = None

    def _try_lock(self, fh: Any) -> None:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover — 本仓运行在 Windows
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def acquire(self, info: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            self._try_lock(fh)
        except OSError as exc:
            # Windows 上被锁的字节区间会拒绝其它句柄读取 → 读取持有者信息需容错
            owner = ""
            try:
                fh.seek(0)
                owner = (fh.read() or "").strip().replace("\n", " ")[:200]
            except OSError:
                owner = "(被锁，读取受限；见上方路径)"
            fh.close()
            raise ToolError(
                f"另一个 MCP 会话正持有评估会话（锁：{self.path}；当前持有者：{owner or '未知'}）。"
                "请稍后重试，或结束另一个客户端后重连；确实需要并行时用 --no-session-lock 启动（内存自负）。"
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(info + "\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:  # noqa: BLE001 — 解锁失败随进程退出自然释放
                logger.debug("unlock failed", exc_info=True)
            try:
                self._fh.close()
            finally:
                self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None


# ── 路径 ──────────────────────────────────────────────────────────────


def _memory_db(ctx: "ServerContext") -> Path:
    return ctx.root / "artifacts" / "alphaagent" / "research_memory.db"


def _runs_dir(ctx: "ServerContext") -> Path:
    return ctx.root / "logs" / "factor_mining" / "ui"


def _panel_parquet(ctx: "ServerContext") -> Path | None:
    files = sorted((ctx.root / "artifacts" / "panel" / "cache").glob("panel_v9_*_all.parquet"))
    return files[-1] if files else None


# ── 会话管理 ──────────────────────────────────────────────────────────


class ServerContext:
    """服务器上下文：仓库根 + 单会话缓存（panel 很重，串行复用）+ 跨进程会话锁。"""

    def __init__(self, root: Path | str | None = None, *, session_lock: bool = True) -> None:
        self.root = Path(root).resolve() if root else ROOT
        self._lock = threading.Lock()
        self._key: tuple[str, str, bool] | None = None
        self._svc: Any = None
        self._sid: str | None = None
        self._load_ms: float = 0.0
        self._session_lock: SessionLock | None = (
            SessionLock(self.root / "artifacts" / "alphaagent" / ".mcp_session.lock") if session_lock else None
        )

    def session(self, mode: str, *, fundamentals: bool = False, universe: str = "all") -> tuple[Any, str]:
        label = MODE_LABEL.get(mode)
        if label is None:
            raise ToolError(f"unknown mode {mode!r}; expected one of {sorted(MODE_LABEL)}")
        universe = _norm_universe(universe)  # 早失败：非法池名直接报错（附可用清单）
        # 缓存键必须含 universe：池子不同的面板不能复用（否则拿全市场面板冒充池内）
        key = (mode, label, bool(fundamentals), universe)
        with self._lock:
            if self._svc is not None and self._key == key:
                return self._svc, self._sid
            self._release_locked()
            if self._session_lock is not None and not self._session_lock.held:
                self._session_lock.acquire(
                    f"pid={os.getpid()} mode={mode} label={label} universe={universe} "
                    f"at={time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            from alphaagent.data.adapters.cnequity import CNE_SOURCE
            from alphaagent.factor.mining.eval.schemas import SessionCreateRequest
            from alphaagent.factor.mining.service import StockEvalService

            t0 = time.perf_counter()
            svc = StockEvalService(max_parallel_eval=1)
            resp = svc.create_session(SessionCreateRequest(
                panel_path=str(CNE_SOURCE), label_col=label, include_fundamentals=fundamentals,
                universe=universe,
            ))
            self._svc, self._sid, self._key = svc, resp.session_id, key
            self._load_ms = round((time.perf_counter() - t0) * 1000)
            logger.info("session opened mode=%s label=%s universe=%s rows=%s load_ms=%s",
                        mode, label, universe, resp.panel_rows, self._load_ms)
            return svc, self._sid

    def session_info(self) -> dict[str, Any]:
        k = self._key or (None, None, None, None)
        return {"mode": k[0], "session_id": self._sid,
                "universe": k[3],
                "load_ms": self._load_ms, "open": self._svc is not None,
                "lock_held": bool(self._session_lock and self._session_lock.held)}

    def _release_locked(self) -> None:
        if self._svc is not None and self._sid:
            try:
                self._svc.release_session(self._sid)
            except Exception:  # noqa: BLE001 — 释放失败不应影响后续会话
                logger.warning("release_session failed", exc_info=True)
        self._svc, self._sid, self._key = None, None, None
        if self._session_lock is not None and self._session_lock.held:
            self._session_lock.release()

    def release_session(self) -> dict[str, Any]:
        """显式释放评估会话与跨进程锁（多客户端共享机器时的礼貌动作）。"""
        with self._lock:
            was_open = self._svc is not None
            self._release_locked()
            return {"released": was_open}

    def close(self) -> None:
        with self._lock:
            self._release_locked()


# ── 门槛 ──────────────────────────────────────────────────────────────


def _crit(mode: str):
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
    from alphaagent.factor.mining.research_spec import effective_research_spec

    return DeliveryCriteria.from_spec(effective_research_spec(mode))


def release_session(ctx: ServerContext) -> dict:
    """显式释放评估会话与跨进程会话锁（多客户端共用机器时用完即放，让别的客户端能接管 panel）。"""
    before = ctx.session_info()
    out = ctx.release_session()
    return {**out, "before": before,
            "note": "已释放 panel 会话与重会话锁；下次调用评估类工具会自动重新开（约 5–15s）。"}


def get_thresholds(ctx: ServerContext, mode: str = "technical") -> dict:
    """返回该档位的**生效门槛**（候选 / 正式库 / engine_gate / 换手门）。"""
    if mode not in MODE_LABEL:
        raise ToolError(f"unknown mode {mode!r}; expected one of {sorted(MODE_LABEL)}")
    c = _crit(mode)
    cand, prod, gate = c.candidate, c.production, c.engine_gate
    return {
        "mode": mode,
        "label_col": MODE_LABEL[mode],
        "rebalance_freq": MODE_FREQ[mode],
        "candidate": {
            "min_abs_ic": cand.min_abs_ic, "min_icir": cand.min_icir,
            "min_coverage": cand.min_coverage, "min_cs_autocorr": cand.min_cs_autocorr,
            "min_val_abs_ic": cand.min_val_abs_ic, "min_val_ic_retention": cand.min_val_ic_retention,
            "max_abs_corr": cand.max_abs_corr,
            "turnover_gate": c.turnover_gate_limit,
            "turnover_thresholds_by_freq": dict(cand.turnover_thresholds_by_freq or {}),
        },
        "production": {
            "min_train_abs_ic": prod.min_train_abs_ic, "min_train_icir": prod.min_train_icir,
            "min_val_abs_ic": prod.min_val_abs_ic, "min_val_ic_retention": prod.min_val_ic_retention,
            "max_winsorized_abs_ic_decay": getattr(prod, "max_winsorized_abs_ic_decay", None),
        },
        "engine_gate": {
            "freq": gate.freq, "allowed_freqs": list(gate.allowed_freqs or []),
            "min_excess_annual": gate.min_excess_annual, "min_excess_sharpe": gate.min_excess_sharpe,
            "max_drawdown": getattr(gate, "max_drawdown", None),
            "max_avg_daily_turnover": getattr(gate, "max_avg_daily_turnover", None),
        },
        "note": "turnover_gate 是当前 freq 档的换手硬门（与 delivery_checker 同源）；"
                "换手指标按交付调仓频率重算，勿跨档比较。",
    }


# ── 字段 / 算子 ───────────────────────────────────────────────────────


def list_fields(ctx: ServerContext, prefixes: str = "", limit: int = 0) -> dict:
    """列出面板可用列（写 DSL 前必需）。`prefixes` 逗号分隔可过滤，如 "funda_,holder_"."""
    import pyarrow.parquet as pq

    path = _panel_parquet(ctx)
    if path is None:
        raise ToolError("panel cache not found: artifacts/panel/cache/panel_v9_*_all.parquet")
    cols = [c.name for c in pq.ParquetFile(path).schema_arrow]
    want = tuple(p.strip() for p in prefixes.split(",") if p.strip())
    picked = [c for c in cols if not want or c.startswith(want)]
    if limit:
        picked = picked[: int(limit)]
    families: dict[str, int] = {}
    for c in cols:
        key = "label" if c.startswith("label") else (c.split("_")[0] if "_" in c else "base")
        families[key] = families.get(key, 0) + 1
    return {"panel": path.name, "total_columns": len(cols), "returned": len(picked),
            "columns": picked, "family_counts": dict(sorted(families.items(), key=lambda kv: -kv[1])),
            "note": "字段语义见 alphaagent/factor/mining/prompt/modules/data_fields.py；"
                    "列名前缀 $ 在 DSL 中引用（如 $adj_close）。"}


def describe_operator(ctx: ServerContext, name: str = "") -> dict:
    """算子目录（无 name 返回全量 Markdown；有 name 只回相关行）。"""
    from alphaagent.dsl.catalog import operator_catalog_markdown

    md = operator_catalog_markdown(tier="full")
    if not name:
        return {"catalog": md}
    key = name.strip().upper()
    if not key.endswith("("):
        key = f"{key}("
    lines = [ln for ln in md.splitlines() if key in ln.upper()]
    if not lines:
        raise ToolError(f"operator {name!r} not found in catalog")
    return {"operator": name.upper(), "signature_lines": lines}


# ── 表达式预检 ────────────────────────────────────────────────────────


def precheck_expression(ctx: ServerContext, multi_line_expr: str) -> dict:
    """DSL 结构风险静态预检（纯 AST，不触发评估、不占评估预算）。"""
    from alphaagent.factor.mining.tools._precheck import precheck_expression as _precheck

    if not str(multi_line_expr or "").strip():
        raise ToolError("multi_line_expr is required")
    return dict(_precheck(str(multi_line_expr)))


# ── run 台账 ──────────────────────────────────────────────────────────


def list_runs(ctx: ServerContext, limit: int = 10) -> dict:
    """最近 run 列表（按目录 mtime 倒序）。"""
    base = _runs_dir(ctx)
    if not base.is_dir():
        return {"runs": [], "note": f"{base} 不存在"}
    dirs = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    rows = []
    for d in dirs[: max(1, int(limit))]:
        summary = _read_summary(d)
        rows.append({
            "run_id": d.name,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.stat().st_mtime)),
            "outcome": summary.get("outcome"), "termination_reason": summary.get("termination_reason"),
            "mode": summary.get("mode") or (summary.get("params") or {}).get("research_mode"),
            "candidate_stored": (summary.get("funnel") or {}).get("candidate_stored"),
            "evaluated": (summary.get("funnel") or {}).get("unique_train_evaluated"),
        })
    return {"runs": rows, "count": len(rows)}


def _read_summary(run_dir: Path) -> dict:
    for fname in ("run_summary.json", "run_meta.json"):
        p = run_dir / fname
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:  # noqa: BLE001
                logger.warning("bad %s", p, exc_info=True)
    return {}


def run_summary(ctx: ServerContext, run_id: str = "") -> dict:
    """单个 run 的汇总（run_id 为空取最新）。"""
    base = _runs_dir(ctx)
    if not base.is_dir():
        raise ToolError(f"{base} 不存在")
    if run_id:
        d = base / run_id
        if not d.is_dir():
            raise ToolError(f"run {run_id!r} not found")
    else:
        dirs = [x for x in base.iterdir() if x.is_dir()]
        if not dirs:
            raise ToolError("no runs found")
        d = max(dirs, key=lambda x: x.stat().st_mtime)
    summary = _read_summary(d)
    steps = d / "steps.log"
    tail = ""
    if steps.is_file():
        lines = steps.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-15:])
    return {"run_id": d.name, "summary": summary, "steps_tail": tail,
            "paths": {"dir": str(d), "summary": str(d / "run_summary.json"), "steps": str(steps)}}


# ── 记忆（读） ────────────────────────────────────────────────────────


def _memory_store(ctx: ServerContext):
    from alphaagent.factor.mining.memory.store import ResearchMemoryStore

    db = _memory_db(ctx)
    if not db.is_file():
        raise ToolError(f"research memory db not found: {db}")
    return ResearchMemoryStore(db, max_inject_chars=2400)


def memory_search(ctx: ServerContext, research_goal: str, limit: int = 12) -> dict:
    """检索跨 run 研究记忆（经验 / 死路证据 / 饱和度 / 多样性注入块）。"""
    if not str(research_goal or "").strip():
        raise ToolError("research_goal is required")
    store = _memory_store(ctx)
    text = store.context_for(str(research_goal), limit=int(limit))
    return {"research_goal": research_goal, "block_chars": len(text or ""), "block": text or "",
            "note": "block 即真实 run 每轮注入的记忆块（同源 context_for）。"}


def memory_stats(ctx: ServerContext) -> dict:
    """记忆层统计：面/算子饱和度和漏斗。"""
    store = _memory_store(ctx)
    out: dict[str, Any] = {}
    try:
        out["saturation"] = store.compute_saturation()
    except Exception as exc:  # noqa: BLE001
        out["saturation_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from alphaagent.factor.mining.memory.analytics import research_funnel

        out["funnel"] = research_funnel(_memory_db(ctx))
    except Exception as exc:  # noqa: BLE001
        out["funnel_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ── 评估（会话域） ────────────────────────────────────────────────────


def _extract(res: dict) -> dict:
    """评估返回 → 指标（同时保留门槛判定所需的**原始键名**）。"""
    s = res.get("summary") if isinstance(res, dict) else None
    s = s if isinstance(s, dict) else {}
    return {
        "ic": s.get("ic"), "icir": s.get("icir"), "rank_ic": s.get("rank_ic"),
        "coverage": s.get("factor_coverage"), "factor_coverage": s.get("factor_coverage"),
        "cs_pearson_autocorr": s.get("cs_pearson_autocorr"),
        "avg_daily_side_turnover": res.get("avg_daily_side_turnover"),
        "avg_rebalance_side_turnover": res.get("avg_rebalance_side_turnover"),
        "n_days": s.get("n_days"), "n_instruments": s.get("n_instruments"),
        # 展示用别名
        "cs_autocorr": s.get("cs_pearson_autocorr"),
        "turnover_rebalance": res.get("avg_rebalance_side_turnover"),
    }


def _evaluate(svc: Any, sid: str, expr: str, name: str, split: str, quantile_n: int) -> tuple[dict, dict]:
    from alphaagent.factor.mining.eval.schemas import EvalTrainRequest, EvalValRequest

    if split == "val":
        raw = svc.eval_val(EvalValRequest(session_id=sid, multi_line_expr=expr, factor_name=name,
                                          include_detail_tables=False, label_quantile_n=quantile_n))
    else:
        raw = svc.eval_train(EvalTrainRequest(session_id=sid, multi_line_expr=expr, factor_name=name,
                                              include_detail_tables=False, label_quantile_n=quantile_n))
    return _extract(raw), raw


def _gate_fails(mode: str, metrics: dict) -> list[str]:
    c = _crit(mode)
    cand = c.candidate
    ic, icir, cov = metrics.get("ic"), metrics.get("icir"), metrics.get("coverage")
    ac, turn = metrics.get("cs_pearson_autocorr"), metrics.get("avg_rebalance_side_turnover")
    fails: list[str] = []
    if ic is None or abs(ic) < cand.min_abs_ic:
        fails.append(f"ic<{cand.min_abs_ic}")
    if icir is None or abs(icir) < cand.min_icir:
        fails.append(f"icir<{cand.min_icir}")
    if cov is None or cov <= cand.min_coverage:
        fails.append(f"coverage<={cand.min_coverage}")
    if ac is None or ac < cand.min_cs_autocorr:
        fails.append(f"cs_autocorr<{cand.min_cs_autocorr}")
    if turn is None or turn > c.turnover_gate_limit:
        fails.append(f"turnover>{c.turnover_gate_limit}")
    return fails


def eval_batch(
    ctx: ServerContext,
    exprs: list[dict] | list[str],
    mode: str = "technical",
    split: str = "train",
    fundamentals: bool = False,
    quantile_n: int = 10,
    universe: str = "all",
) -> dict:
    """★ 批量评估（一次调用评多个因子）+ 逐门 PASS/FAIL（真源门槛）。

    ``universe``：股票池（评估口径，非因子属性）—— ``all`` / ``top300cap`` /
    ``mid301_800cap`` / ``szcomp`` / ``chinext``。非成分股票整行剔除。
    研报复现建议同一批因子跨池各评一次（结果按池记账）。
    """
    if mode not in MODE_LABEL:
        raise ToolError(f"unknown mode {mode!r}; expected one of {sorted(MODE_LABEL)}")
    if split not in ("train", "val"):
        raise ToolError("split must be 'train' or 'val'")
    universe = _norm_universe(universe)  # 入口规范化+早校验（响应回显与键一致，OCR D2）
    items: list[dict] = []
    for i, e in enumerate(exprs or []):
        items.append({"name": f"expr_{i + 1}", "expr": e} if isinstance(e, str) else dict(e))
    if not items:
        raise ToolError("exprs is required (non-empty)")
    if len(items) > MAX_BATCH:
        raise ToolError(f"too many expressions: {len(items)} > {MAX_BATCH}（分多次调用）")
    for it in items:
        if not str(it.get("expr") or "").strip():
            raise ToolError(f"expression #{items.index(it) + 1} is empty")

    svc, sid = ctx.session(mode, fundamentals=fundamentals, universe=universe)
    rows: list[dict] = []
    for it in items:
        t0 = time.perf_counter()
        try:
            metrics, _raw = _evaluate(svc, sid, str(it["expr"]), str(it.get("name") or "expr"),
                                      split, int(quantile_n))
            err = None
        except Exception as exc:  # noqa: BLE001 — 单因子失败不中断批量
            metrics, err = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
        fails = _gate_fails(mode, metrics) if metrics else ["eval_error"]
        rows.append({"name": it.get("name"), "why": it.get("why", ""), "expr": it["expr"],
                     "ms": round((time.perf_counter() - t0) * 1000), "error": err,
                     "ic": metrics.get("ic"), "icir": metrics.get("icir"), "rank_ic": metrics.get("rank_ic"),
                     "coverage": metrics.get("coverage"), "cs_autocorr": metrics.get("cs_pearson_autocorr"),
                     "turnover_rebalance": metrics.get("avg_rebalance_side_turnover"),
                     "turnover_daily": metrics.get("avg_daily_side_turnover"),
                     "fails": fails, "pass_stage_one_pre": not fails})
    n_pass = sum(1 for r in rows if r["pass_stage_one_pre"])
    return {"mode": mode, "label_col": MODE_LABEL[mode], "split": split, "universe": universe,
            "n": len(rows), "n_pass_stage_one_pre": n_pass,
            "thresholds": get_thresholds(ctx, mode)["candidate"], "rows": rows,
            "note": "pass_stage_one_pre 是按真源门槛的**预检**（未含库内相关性与 engine_gate）；"
                    "过门后建议先 dry_run_delivery 再 submit_factor。"}


def eval_val(ctx: ServerContext, multi_line_expr: str, mode: str = "technical",
             fundamentals: bool = False, quantile_n: int = 10, universe: str = "all") -> dict:
    """样本外（val 段）评估 + 与 train 的保留比（``universe`` 同 ``eval_batch``）。"""
    if not str(multi_line_expr or "").strip():
        raise ToolError("multi_line_expr is required")
    universe = _norm_universe(universe)  # 入口规范化+早校验（响应回显与键一致，OCR D2）
    svc, sid = ctx.session(mode, fundamentals=fundamentals, universe=universe)
    train_m, _ = _evaluate(svc, sid, str(multi_line_expr), "expr", "train", int(quantile_n))
    val_m, _ = _evaluate(svc, sid, str(multi_line_expr), "expr", "val", int(quantile_n))
    retention = None
    if train_m.get("ic") not in (None, 0) and val_m.get("ic") is not None:
        retention = abs(val_m["ic"]) / abs(train_m["ic"]) if abs(train_m["ic"]) > 1e-12 else None
    sign_consistent = (None if train_m.get("ic") in (None, 0) or val_m.get("ic") is None
                       else (train_m["ic"] * val_m["ic"] > 0))
    return {"mode": mode, "label_col": MODE_LABEL[mode], "universe": universe,
            "train": {k: train_m.get(k) for k in ("ic", "icir", "coverage")},
            "val": {k: val_m.get(k) for k in ("ic", "icir", "coverage")},
            "val_ic_retention": retention, "sign_consistent": sign_consistent,
            "thresholds": {"min_val_abs_ic": _crit(mode).candidate.min_val_abs_ic,
                           "min_val_ic_retention": _crit(mode).candidate.min_val_ic_retention}}


def library_similarity(ctx: ServerContext, multi_line_expr: str, mode: str = "technical",
                       top_k: int = 3, universe: str = "all",
                       fundamentals: bool = False) -> dict:
    """★ 与**候选池已有因子**的截面相关性（stage_two 相关性墙的提前预警）。

    ``universe``/``fundamentals`` 同 ``eval_batch``：相关性必须在**同一会话口径**内比较
    （同池 + 同基本面列装载）——任一维度不一致都会另开 6-8GB 会话且特征体系错位（OCR R2-1）。
    """
    from alphaagent.factor.mining.delivery.submit import (
        _candidate_registry_similarity,
        _materialize_split_aware,
        _visible_range_of,
    )
    from alphaagent.factor.metrics import materialize_factor
    from core import factor_categories

    universe = _norm_universe(universe)  # 入口规范化+早校验（OCR D2）
    svc, sid = ctx.session(mode, fundamentals=fundamentals, universe=universe)
    session = svc.sessions.get(sid)
    panel = session.panel
    expr = str(multi_line_expr)
    materialized = _materialize_split_aware(expr, session, panel, name="mcp_similarity")
    if materialized is None:
        materialized = materialize_factor(expr, panel, cache=getattr(session, "factor_cache", None))
    visible = _visible_range_of(session)
    sim = _candidate_registry_similarity(
        materialized.values, panel, factor_categories.candidate_registry_path(mode),
        top_k=int(top_k), cache=getattr(session, "factor_cache", None), visible_range=visible,
    )
    if sim is None:
        return {"similarity": None, "universe": universe, "note": "候选 registry 为空或不可比"}
    max_corr = sim.get("max_abs_corr")
    limit = _crit(mode).candidate.max_abs_corr
    return {"similarity": sim, "max_abs_corr": max_corr, "limit": limit, "universe": universe,
            "passes_stage_two_correlation": (max_corr is not None and max_corr <= limit),
            "note": "> limit 时 stage_two（正式库晋升）会被拒；入库候选池不受此门限制。"}


def dry_run_delivery(ctx: ServerContext, multi_line_expr: str, mode: str = "technical",
                     fundamentals: bool = False, quantile_n: int = 10,
                     universe: str = "all") -> dict:
    """★ 只读交付预检：train+val 评估 + 候选相关性 + 各 stage 判定（不写任何库）。"""
    from alphaagent.factor.mining.delivery.delivery_checker import DeliveryChecker

    expr = str(multi_line_expr or "").strip()
    if not expr:
        raise ToolError("multi_line_expr is required")
    universe = _norm_universe(universe)  # 入口规范化+早校验（OCR D2）
    svc, sid = ctx.session(mode, fundamentals=fundamentals, universe=universe)
    train_m, _ = _evaluate(svc, sid, expr, "mcp_dryrun", "train", int(quantile_n))
    val_m, _ = _evaluate(svc, sid, expr, "mcp_dryrun", "val", int(quantile_n))
    sim: dict | None = None
    sim_err: str | None = None
    try:
        sim = library_similarity(ctx, expr, mode, universe=universe,
                                 fundamentals=fundamentals).get("similarity")  # OCR D1/R2-1：相关墙与 train/val 同会话同面板
    except Exception as exc:  # noqa: BLE001 — 相似度为诊断项，失败不阻断
        sim_err = f"{type(exc).__name__}: {str(exc)[:200]}"
    checker = DeliveryChecker(_crit(mode))
    freq = MODE_FREQ[mode]

    def _pack(stage) -> dict:
        return {"passed": getattr(stage, "passed", None), "fail_reasons": list(getattr(stage, "fail_reasons", []) or [])}

    s1c = _pack(checker.stage_one_correlation(sim))
    s2 = _pack(checker.stage_two(train_m, val_m, sim))
    if sim is None and sim_err:
        # OCR R2-1：StageOneCorrelation 对空证据按 max_abs_corr=0 fail-open（delivery_checker.py:164-172）
        # ——相似度失败时相关门判定不可信，显式降级为 passed=None + 原因，防止预检静默放行
        for st in (s1c, s2):
            st["passed"] = None
            st["fail_reasons"] = sorted(set(st["fail_reasons"]) | {"similarity_unavailable"})
    return {
        "mode": mode, "label_col": MODE_LABEL[mode], "rebalance_freq": freq,
        "universe": universe,
        "train": {k: train_m.get(k) for k in ("ic", "icir", "coverage", "cs_pearson_autocorr", "avg_rebalance_side_turnover")},
        "val": {k: val_m.get(k) for k in ("ic", "icir", "coverage")},
        "gate_precheck_fails": _gate_fails(mode, train_m),
        "stage_one": _pack(checker.stage_one_stats(train_m, rebalance_freq=freq)),
        "stage_one_val_retention": _pack(checker.stage_one_val_retention(train_m, val_m)),
        "stage_one_correlation": s1c,
        "stage_two": s2,
        "candidate_similarity_max_abs_corr": (sim or {}).get("max_abs_corr"),
        "similarity_error": sim_err,
        "note": "只读预检，未写候选池；engine_gate（净值回测）仅在 submit 时执行。",
    }


def submit_factor(
    ctx: ServerContext,
    multi_line_expr: str,
    factor_name: str,
    comment: str,
    mode: str = "technical",
    confirm: bool = False,
    fundamentals: bool = False,
    universe: str = "all",
) -> dict:
    """**写**：走真实交付链路（stage_one → 盲测 → stage_two → engine_gate）入库。需 confirm=true。

    ``universe``：股票池（OCR D3①）——交付门在池内算，与 ``dry_run_delivery`` 同池口径；
    缺省 ``all`` = 全市场（与历史行为一致）。响应回显规范化池名；候选元数据按池记账
    （evaluated_universe 进 factorzoo）仍属 P2。
    """
    if not confirm:
        raise ToolError("submit_factor 是写操作：确认后请传 confirm=true（建议先 dry_run_delivery）")
    expr, name = str(multi_line_expr or "").strip(), str(factor_name or "").strip()
    if not expr or not name:
        raise ToolError("multi_line_expr 与 factor_name 均必填")
    if not str(comment or "").strip():
        raise ToolError("comment 必填：需说明机制与经济直觉（与真实 run 的提交要求一致）")
    universe = _norm_universe(universe)
    from alphaagent.factor.mining.delivery.submit import FactorSubmitService
    from alphaagent.factor.mining.research_spec import effective_research_spec

    svc, sid = ctx.session(mode, fundamentals=fundamentals, universe=universe)
    submitter = FactorSubmitService(
        svc, repo_root=ctx.root, research_mode=mode, max_cs_corr=0.8,
        delivery_policy=(effective_research_spec(mode).get("delivery_policy") or {}),
        similar_top_k=3, overwrite=False,
    )
    res = submitter.submit(sid, multi_line_expr=expr, factor_name=name, comment=str(comment),
                           rebalance_freq=MODE_FREQ[mode])
    logger.info("submit_factor name=%s universe=%s candidate_stored=%s stored=%s skipped=%s",
                name, universe, res.get("candidate_stored"), res.get("stored"), res.get("skipped_reason"))
    keep = ("ok", "stored", "candidate_stored", "candidate_storage", "skipped_reason", "error",
            "error_type", "promotion_status", "review_status", "factor_id", "rebalance_freq",
            "delivery_check", "test_holdout", "candidate_similarity", "engine_backtest", "metrics",
            "candidate_registry_path", "candidate_dsl_path")
    out = {k: res.get(k) for k in keep if k in res}
    out["universe"] = universe
    return out


def memory_record(
    ctx: ServerContext,
    multi_line_expr: str,
    factor_name: str = "expr",
    mode: str = "technical",
    split: str = "train",
    run_id: str = "mcp-session",
    confirm: bool = False,
    universe: str = "all",
) -> dict:
    """**写**：把一次评估写回 `research_memory`（跨 run 学习；harness 直驱的最大缺口）。"""
    if not confirm:
        raise ToolError("memory_record 是写操作：确认后请传 confirm=true")
    expr = str(multi_line_expr or "").strip()
    if not expr:
        raise ToolError("multi_line_expr is required")
    universe = _norm_universe(universe)  # OCR D3①：记忆记录与评估同池口径
    svc, sid = ctx.session(mode, universe=universe)
    metrics, raw = _evaluate(svc, sid, expr, str(factor_name or "expr"), split, 10)
    from alphaagent.factor.mining.memory.store import ResearchMemoryStore

    store = ResearchMemoryStore(_memory_db(ctx))
    row = {"name": "evaluate_factor" if split == "train" else "eval_on_val_set",
           "arguments_raw": json.dumps({"multi_line_expr": expr, "factor_name": factor_name,
                                        "universe": universe}, ensure_ascii=False),
           "result": raw}
    wrote = store.record_tool_result(
        run_id=str(run_id or "mcp-session"), row=row,
        # OCR R2-4：universe 随 observation.metrics 落库，跨池评估可按池归因（条目签名仍为
        # 纯表达式——同因子跨池共存于一个条目，逐次 observation 各自带池名）
        run_freq_context={"rebalance_freq": MODE_FREQ[mode], "research_mode": mode,
                          "freq_source": "mcp", "universe": universe},
    )
    return {"recorded": bool(wrote), "run_id": run_id, "universe": universe,
            "metrics": {k: metrics.get(k) for k in ("ic", "icir", "coverage")},
            "entry": wrote}

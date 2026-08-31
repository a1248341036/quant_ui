"""挖掘启动自检（preflight）：Agent 开工前验证整条管道，快速失败而非中途踩雷。

背景
----
历史问题：生产库未初始化（``factorlib_not_initialized``）、FactorZoo 索引契约
不匹配（``'MultiIndex' object has no attribute 'rows'``）等基础设施错误只在
run 中途的 submit 才暴露。LLM 会把这类错误当临时故障反复重试（同一错误最多
重试 7+ 次、每次 140-170 秒），一次 run 一半时间烧在重复踩雷上，且"修不完"。

方案
----
在 session 创建 + submit_service 构建之后、Agent 启动之前执行 ``run_preflight``：

1. ``check_factor_library``：确保生产库已初始化（未初始化则用 session panel
   自动建），验证 ``zoo.index`` 是 RowIndex（有 ``.rows``）且 manifest 一致。
2. ``check_eval_smoke``：跑一个最小因子走 evaluate 全链路（DSL 求值 + 评估
   引擎），验证管道可算。
3. ``check_submit_smoke``：跑一个必然不达标的 submit（验证 submit 管道能正常
   返回判定而非抛异常），不真写库。

任一项失败立即抛 ``PreflightError`` 并带明确修复方向；调用方捕获后终止 run，
不进入 LLM 挖掘。修一次自检通过后，整个 run 不再中途踩雷。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.submit import FactorSubmitService
from alphaagent.factor.zoo import FactorZoo

# 冒烟因子：最小算子组合，只验证管道可算，不追求任何 IC。
# 用 TS_MEAN + DELTA 两个基础算子，避免引入 CHIP_* / 交互等重算子干扰自检。
_SMOKE_EVAL_EXPR = "ma = TS_MEAN($ret, 5)\nDELTA(ma, 1)"
# 冒烟 submit：必然不达标（低 IC/无时序结构），验证 submit 返回正常判定而非抛异常。
_SMOKE_SUBMIT_EXPR = "CS_ZSCORE($ret)"
_SMOKE_NAME = "__preflight_smoke__"


class PreflightError(RuntimeError):
    """自检失败：携带具体环节与修复方向。"""


@dataclass
class PreflightReport:
    checks: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.get("passed") for c in self.checks)

    def add(self, name: str, passed: bool, detail: str, duration_s: float) -> None:
        self.checks.append({
            "check": name,
            "passed": passed,
            "detail": detail,
            "duration_s": round(duration_s, 3),
        })


def _ensure_zoo(
    submit_service: FactorSubmitService,
    session,
) -> FactorZoo:
    """打开生产库；未初始化则用会话 panel 自动建（复用 submit 的 _ensure_factorzoo）。"""
    return submit_service._ensure_factorzoo(session)


def check_factor_library(
    submit_service: FactorSubmitService,
    session,
    report: PreflightReport,
) -> FactorZoo:
    """生产库初始化 + 索引契约检查。

    失败修复方向：
    - ``FileNotFoundError``/未初始化 → 应已由 _ensure_factorzoo 自动初始化；
    - ``zoo.index`` 无 ``.rows`` → 索引文件损坏/格式不兼容，删除该 production
      目录后重跑自检（会自动重建），或检查 realign 流程是否产出旧格式。
    """
    import time

    t0 = time.perf_counter()
    try:
        zoo = _ensure_zoo(submit_service, session)
    except Exception as exc:  # noqa: BLE001
        report.add("factor_library", False, f"生产库初始化失败: {type(exc).__name__}: {exc}", time.perf_counter() - t0)
        raise PreflightError(
            f"生产库初始化失败: {type(exc).__name__}: {exc}\n"
            "修复方向: 删除该 research_mode 的 production 目录后重跑自检（会自动重建），"
            "或检查 panel 能否正常加载。"
        ) from exc

    if not hasattr(zoo.index, "rows"):
        report.add("factor_library", False, f"索引契约异常: zoo.index 类型={type(zoo.index).__name__}，缺少 .rows", time.perf_counter() - t0)
        raise PreflightError(
            f"FactorZoo 索引契约异常: zoo.index 是 {type(zoo.index).__name__}（应为 RowIndex，含 .rows）。\n"
            "修复方向: 该 production 目录索引损坏或格式不兼容，删除后重跑自检会自动重建。"
        )

    if zoo.manifest.n_rows != len(zoo.index.rows):
        report.add("factor_library", False, f"index/manifest 行数不一致: index={len(zoo.index.rows)} manifest={zoo.manifest.n_rows}", time.perf_counter() - t0)
        raise PreflightError(
            f"FactorZoo index/manifest 行数不一致: index={len(zoo.index.rows)} "
            f"manifest={zoo.manifest.n_rows}。\n修复方向: 删除该 production 目录后重跑自检会自动重建。"
        )

    report.add(
        "factor_library",
        True,
        f"生产库就绪: {zoo.paths.root.name}, n_factors={zoo.n_factors}, n_rows={zoo.manifest.n_rows}",
        time.perf_counter() - t0,
    )
    return zoo


def check_eval_smoke(
    service: StockEvalService,
    session_id: str,
    report: PreflightReport,
) -> None:
    """evaluate 全链路冒烟：DSL 求值 + 评估引擎，验证管道可算。"""
    import time

    t0 = time.perf_counter()
    try:
        result = service.eval_train(
            __import__("alphaagent.factor.mining.schemas", fromlist=["EvalTrainRequest"]).EvalTrainRequest(
                session_id=session_id,
                multi_line_expr=_SMOKE_EVAL_EXPR,
                factor_name=_SMOKE_NAME,
            )
        )
        ok = bool(result.get("ok"))
        timing = (result.get("timing_ms") or {})
        total = timing.get("total_ms")
        detail = f"evaluate ok={ok}, total_ms={total}"
        if not ok:
            detail += f", error={result.get('error')}"
        report.add("eval_smoke", ok, detail, time.perf_counter() - t0)
        if not ok:
            raise PreflightError(
                f"evaluate 冒烟失败: {result.get('error') or result.get('error_type')}\n"
                f"表达式: {_SMOKE_EVAL_EXPR}\n"
                "修复方向: 检查 DSL 求值后端（Numba JIT / C++ 加速）与评估引擎 plugins。"
            )
    except PreflightError:
        raise
    except Exception as exc:  # noqa: BLE001
        report.add("eval_smoke", False, f"evaluate 冒烟异常: {type(exc).__name__}: {exc}", time.perf_counter() - t0)
        raise PreflightError(
            f"evaluate 冒烟异常: {type(exc).__name__}: {exc}\n"
            f"表达式: {_SMOKE_EVAL_EXPR}\n"
            "修复方向: 检查 DSL 求值后端（Numba JIT / C++ 加速）与评估引擎 plugins。"
        ) from exc


def check_submit_smoke(
    submit_service: FactorSubmitService,
    session_id: str,
    report: PreflightReport,
) -> None:
    """submit 管道冒烟：验证 submit 正常返回判定而非抛异常（暴露索引契约 bug）。

    用低信息表达式（``CS_ZSCORE($ret)``）走 submit 全链路。核心价值是暴露
    管道级错误（``'MultiIndex' object has no attribute 'rows'`` 等）而非让
    表达式通过。若意外 candidate_stored=True（极低概率），立即从候选 registry
    删除冒烟记录，避免污染真实候选池。
    """
    import time

    t0 = time.perf_counter()
    try:
        result = submit_service.submit(
            session_id,
            multi_line_expr=_SMOKE_SUBMIT_EXPR,
            factor_name=_SMOKE_NAME,
            comment="preflight smoke submit (not a real factor)",
        )
        # 若意外写入候选池，立即清理（冒烟因子绝不留库）
        if bool(result.get("candidate_stored")):
            from alphaagent.factor.mining.registry_io import load_mining_registry, save_mining_registry
            cand_path = submit_service.candidate_registry_path
            reg = load_mining_registry(cand_path)
            if _SMOKE_NAME in reg:
                reg.pop(_SMOKE_NAME, None)
                save_mining_registry(cand_path, reg)
            # 顺带清理表达式文件
            dsl_path = submit_service.candidate_expr_dir / f"{_SMOKE_NAME}.dsl"
            if dsl_path.is_file():
                try:
                    dsl_path.unlink()
                except OSError:
                    pass

        st_one = result.get("delivery_check", {}).get("stage_one") or {}
        passed = True  # 核心价值是"不抛异常"，是否 stage_one 失败不影响自检通过
        report.add(
            "submit_smoke",
            passed,
            f"submit 返回正常判定 stage_one_passed={st_one.get('passed')}, "
            f"reasons={st_one.get('fail_reasons')}, candidate_stored={result.get('candidate_stored')}",
            time.perf_counter() - t0,
        )
    except PreflightError:
        raise
    except Exception as exc:  # noqa: BLE001
        report.add("submit_smoke", False, f"submit 冒烟异常: {type(exc).__name__}: {exc}", time.perf_counter() - t0)
        raise PreflightError(
            f"submit 冒烟异常: {type(exc).__name__}: {exc}\n"
            f"表达式: {_SMOKE_SUBMIT_EXPR}\n"
            "修复方向: 检查 FactorSubmitService / FactorZoo / similarity 管道的索引契约"
            "（常见 'MultiIndex' object has no attribute 'rows' 即在此暴露）。"
        ) from exc


def run_preflight(
    *,
    service: StockEvalService,
    session_id: str,
    submit_service: FactorSubmitService,
    session=None,
) -> PreflightReport:
    """执行全部启动自检；任一项失败抛 PreflightError。返回报告。"""
    import time

    t0 = time.perf_counter()
    report = PreflightReport()
    if session is None:
        session = service.sessions.get(session_id)

    # 1) 生产库初始化 + 索引契约（最可能中途踩雷的环节，最先查）
    zoo = check_factor_library(submit_service, session, report)
    del zoo  # 自检用；真实挖掘仍由 submit 各自打开

    # 2) evaluate 全链路冒烟
    check_eval_smoke(service, session_id, report)

    # 3) submit 管道冒烟（不真写库）
    check_submit_smoke(submit_service, session_id, report)

    report.duration_s = round(time.perf_counter() - t0, 3)
    return report


def preflight_summary(report: PreflightReport) -> str:
    # 用 ASCII 标记而非 emoji：Windows 控制台默认 GBK，emoji 打印会崩（同 ensure_utf8_stream）。
    lines = ["=== 挖掘启动自检 ==="]
    for c in report.checks:
        mark = "[OK]" if c["passed"] else "[FAIL]"
        lines.append(f"  {mark} {c['check']}: {c['detail']} ({c['duration_s']}s)")
    lines.append(f"总耗时 {report.duration_s}s")
    return "\n".join(lines)

"""ML 组合（stacking）训练运行管理。

与挖掘 run 同模式：子进程执行 scripts/train_ml_composite.py，stdout 逐行写入
进度日志（progress.log），完成后 report.json 落盘到确定性输出目录。
同一时间只允许一个训练（panel 全量物化内存重，并发只会互相拖慢）。
"""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STACKING_ROOT = ROOT / "artifacts" / "alphaagent" / "stacking"
PYTHON_EXECUTABLE = ROOT / ".venv" / "Scripts" / "python.exe"

_lock = threading.Lock()
_current: dict[str, Any] | None = None  # {train_id, proc, out_dir, log_path, params}


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _progress_log(train_id: str) -> Path:
    d = ROOT / "artifacts" / "alphaagent" / "stacking_ui" / train_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "progress.log"


def start_training(params: dict[str, Any]) -> dict[str, Any]:
    """启动一次 ML/简单加权组合运行。返回 {train_id, status} 或 {error}。

    scheme：ml=学习加权（现状）；equal/icir/hrp=不拟合模型，按选定规则
    滚动加权（--scheme 传给脚本，其余报告/gate/pred 链路一致）。
    """
    global _current
    with _lock:
        if _current is not None and _current["proc"].poll() is None:
            return {"error": "training_already_running",
                    "train_id": _current["train_id"]}

        scheme = str(params.get("scheme") or "ml")
        if scheme not in ("ml", "equal", "icir", "hrp"):
            return {"error": f"invalid_scheme（支持 ml/equal/icir/hrp，收到 {scheme!r}）"}

        train_id = _now_id()
        modes = params.get("modes") or ["technical", "fundamental"]
        if isinstance(modes, str):
            modes = [modes]
        modes = [str(m) for m in modes]
        out_dir = STACKING_ROOT / train_id
        log_path = _progress_log(train_id)

        eval_mode = str(params.get("eval_mode") or "tuning")
        command = [
            str(PYTHON_EXECUTABLE), str(ROOT / "scripts" / "train_ml_composite.py"),
            "--eval-mode", eval_mode,
            "--modes", *modes,
            "--scheme", scheme,
            "--model", str(params.get("model") or "both"),
            "--label-days", str(int(params.get("label_days") or 5)),
            "--train-months", str(int(params.get("train_months") or 18)),
            "--step-months", str(int(params.get("step_months") or 6)),
            "--purge-days", str(int(params.get("purge_days") or 5)),
            "--warmup-days", str(int(params.get("warmup_days") or 250)),
            "--max-corr", str(float(params.get("max_corr") or 0.6)),
            "--out-dir", str(out_dir),
        ]
        if params.get("mining_end"):
            command += ["--mining-end", str(params["mining_end"])]
        if params.get("no_candidate"):
            command += ["--no-candidate"]
        if params.get("no_gate"):
            command += ["--no-gate"]
        command += ["--isolation", str(params.get("isolation") or "holdout")]
        if params.get("size_neutral") is False:
            command += ["--no-size-neutral"]
        if params.get("subset_curve"):
            command += ["--subset-curve"]
        include_factors = params.get("include_factors") or None
        if include_factors:
            command += ["--include-factors", *[str(n) for n in include_factors]]
        score_smooth = int(params.get("score_smooth") or 0)
        if score_smooth:
            command += ["--score-smooth", str(score_smooth)]
        if params.get("multi_path"):
            command += ["--multi-path-shifts", "0,2,4"]
        if params.get("llm_assist"):
            command += ["--llm-assist"]

        # 参数快照落盘：历史列表/详情页展示训练配置用
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        log_handle = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()  # Popen 已继承句柄

        _current = {
            "train_id": train_id,
            "proc": proc,
            "out_dir": out_dir,
            "log_path": log_path,
            "params": params,
        }
        return {"train_id": train_id, "status": "running", "out_dir": str(out_dir)}


def _proc_status() -> tuple[str, dict[str, Any] | None]:
    """返回 (status, _current)。running 时等待子进程结束则转为完成/失败。"""
    global _current
    with _lock:
        cur = _current
        if cur is None:
            return "idle", None
        code = cur["proc"].poll()
        if code is None:
            return "running", cur
        status = "completed" if code == 0 else "failed"
        cur["exit_code"] = code
        return status, cur


def _read_report_json(report_path: Path) -> dict[str, Any] | None:
    """读 stacking report.json，兼容历史 GBK 落盘（train_ml_composite 旧版
    write_text 未指定 encoding，Windows 默认 GBK——UnicodeDecodeError 曾把
    trainings 列表接口整个 500）。utf-8 失败回退 gbk，再失败返回 None。"""
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            return json.loads(report_path.read_text(encoding="gbk"))
        except (json.JSONDecodeError, OSError):
            return None
    except (json.JSONDecodeError, OSError):
        return None


def list_trainings(limit: int = 30) -> list[dict[str, Any]]:
    """列出历史训练：磁盘上所有 stacking 输出目录 + 当前运行状态。"""
    status, cur = _proc_status()
    running_id = cur["train_id"] if cur and status == "running" else None
    out: list[dict[str, Any]] = []
    if STACKING_ROOT.is_dir():
        for d in sorted(STACKING_ROOT.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            report_path = d / "report.json"
            item: dict[str, Any] = {
                "train_id": d.name,
                "status": "running" if d.name == running_id else
                          ("completed" if report_path.is_file() else "unknown"),
                "out_dir": str(d),
            }
            if d.name == running_id:
                item["status"] = "running"
            elif cur is not None and d.name == cur["train_id"] and status == "failed":
                item["status"] = "failed"
            if report_path.is_file():
                report = _read_report_json(report_path)
                if report is not None:
                    blended = report.get("oos_ic_blended") or {}
                    gate = report.get("gate") or {}
                    gm = gate.get("metrics") or {}
                    gd = gate.get("diagnostics") or {}
                    item["oos_ic_mean"] = blended.get("ic_mean")
                    item["oos_ic_ir"] = blended.get("ic_ir")
                    item["n_folds"] = report.get("folds")
                    item["n_features"] = len(report.get("feature_names") or [])
                    item["gate_passed"] = gate.get("passed")
                    item["time_isolation"] = report.get("time_isolation")
                    item["label_days"] = report.get("label_days")
                    item["mining_end"] = report.get("mining_end")
                    item["model"] = "/".join((report.get("fold_metrics") or {}).keys()) or None
                    # ── 详细历史指标（跨训练对比）：gate 四项 + 分模型 IC + 最差折 + 衰减保留 ──
                    item["gate_excess_annual"] = gm.get("excess_annual")
                    item["gate_excess_sharpe"] = gm.get("excess_sharpe")
                    item["gate_max_drawdown"] = gm.get("max_drawdown")
                    item["gate_daily_overlap"] = gm.get("daily_overlap")
                    item["gate_daily_turnover"] = gd.get("avg_daily_turnover")
                    model_ics: dict[str, float] = {}
                    worst: float | None = None
                    for kind, rows in (report.get("fold_metrics") or {}).items():
                        ics = [float(r["ic_mean"]) for r in rows if isinstance(r.get("ic_mean"), (int, float))]
                        if ics:
                            model_ics[kind] = round(sum(ics) / len(ics), 5)
                            worst = min(ics) if worst is None else min(worst, min(ics))
                    item["model_ic"] = model_ics
                    item["worst_fold_ic"] = worst
                    ratios = [
                        float(r["decay_ratio"]) for r in (report.get("decay_table") or [])
                        if isinstance(r.get("decay_ratio"), (int, float))
                    ]
                    item["decay_retention"] = round(sum(ratios) / len(ratios), 4) if ratios else None
                    item["n_dropped"] = len(report.get("dropped") or [])
                    item["scheme"] = report.get("scheme") or "ml"
                    item["scheme_label"] = report.get("scheme_label")
            # 参数快照（start_training 落盘）：历史行展示完整训练配置
            params_path = d / "params.json"
            if params_path.is_file():
                try:
                    item["params"] = json.loads(params_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            out.append(item)
        # running 状态但目录尚未创建（panel 加载阶段）也补一条
        if running_id and not any(x["train_id"] == running_id for x in out):
            out.insert(0, {"train_id": running_id, "status": "running"})
    return out[: max(1, limit)]


def get_training(train_id: str, *, tail_lines: int = 40) -> dict[str, Any]:
    """单个训练详情：状态 + 进度日志尾部 + 完成时的 report。"""
    status, cur = _proc_status()
    running_id = cur["train_id"] if cur else None
    log_path = _progress_log(train_id)
    tail: list[str] = []
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-tail_lines:]

    out: dict[str, Any] = {"train_id": train_id, "status": "unknown", "progress_tail": tail}
    if train_id == running_id:
        out["status"] = "running"
    out_dir = STACKING_ROOT / train_id
    report_path = out_dir / "report.json"
    if train_id != running_id:
        out["status"] = "completed" if report_path.is_file() else "unknown"
    if report_path.is_file():
        report = _read_report_json(report_path)
        if report is not None:
            out["report"] = report
        else:
            out["report_error"] = "report.json 读取失败（编码/格式损坏）"
    elif train_id == running_id:
        pass
    else:
        out["progress_tail"] = tail or [f"未找到训练 {train_id} 的输出"]
    out["events"] = get_training_events(train_id)
    return out


def get_training_events(train_id: str) -> list[dict[str, Any]]:
    """读取训练的结构化事件列表；若无 events.jsonl 则从 report.json 合成兜底回放事件。"""
    out_dir = STACKING_ROOT / train_id
    events_file = out_dir / "events.jsonl"
    events: list[dict[str, Any]] = []
    if events_file.is_file():
        try:
            for line in events_file.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    if events:
        return events

    # 若没有 events.jsonl（历史老训练），从 report.json 与 progress_log 合成回放事件
    report_path = out_dir / "report.json"
    if report_path.is_file():
        report = _read_report_json(report_path)
        if report:
            events.append({
                "ts": report.get("run_id") or "0",
                "event": "session_start",
                "run_id": train_id,
                "scheme": report.get("scheme"),
                "scheme_label": report.get("scheme_label"),
                "label_days": report.get("label_days"),
                "isolation": report.get("time_isolation"),
                "eval_mode": report.get("eval_mode", "tuning"),
                "blind_test_isolated": report.get("blind_test_isolated", True),
            })
            if report.get("llm_recommendation"):
                rec = report["llm_recommendation"]
                events.append({
                    "ts": report.get("run_id") or "0",
                    "event": "ml_pool_screened",
                    "title": "因子语义研判推荐",
                    "recommended": rec.get("recommended", []),
                    "rationale": rec.get("rationale", ""),
                    "n_recommended": rec.get("n_recommended", 0),
                    "reused": rec.get("reused", False),
                })
            if report.get("feature_names"):
                events.append({
                    "ts": report.get("run_id") or "0",
                    "event": "ml_features_filtered",
                    "title": "特征筛选完成",
                    "feature_names": report.get("feature_names", []),
                    "n_features": len(report.get("feature_names", [])),
                    "n_dropped": len(report.get("dropped", [])),
                    "dropped": report.get("dropped", []),
                })
            fold_metrics = report.get("fold_metrics") or {}
            for kind, rows in fold_metrics.items():
                if isinstance(rows, list):
                    events.append({
                        "ts": report.get("run_id") or "0",
                        "event": "ml_fold_progress",
                        "title": f"{kind.upper()} 拟合完成",
                        "kind": kind,
                        "fold_reports": rows,
                        "feature_weights": (report.get("feature_weights", {}).get(kind) or [])[:15],
                    })
            if report.get("gate"):
                g = report["gate"]
                events.append({
                    "ts": report.get("run_id") or "0",
                    "event": "ml_gate_evaluated",
                    "title": "engine_gate 门禁回测完成",
                    "passed": g.get("passed"),
                    "metrics": g.get("metrics") or {},
                    "fail_reasons": g.get("fail_reasons") or [],
                })
            if report.get("llm_summary"):
                events.append({
                    "ts": report.get("run_id") or "0",
                    "event": "ml_summary_generated",
                    "title": "组合说明书",
                    "summary": report["llm_summary"],
                })
            events.append({
                "ts": report.get("run_id") or "0",
                "event": "session_end",
                "title": "训练完成",
                "status": "completed",
                "run_id": train_id,
                "oos_ic": (report.get("oos_ic_blended") or {}).get("ic_mean"),
                "oos_ir": (report.get("oos_ic_blended") or {}).get("ic_ir"),
            })
    return events


async def stream_training_events(train_id: str):
    """异步生成 SSE 事件流：先回放存量事件，若运行中则持续异步 tail 新行直至结束。"""
    import asyncio

    out_dir = STACKING_ROOT / train_id
    events_file = out_dir / "events.jsonl"

    # 1. 存量事件回放
    initial_events = get_training_events(train_id)
    for ev in initial_events:
        yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

    # 2. 如果当前没有处于 running 状态，回放完即结束
    status, cur = _proc_status()
    running_id = cur["train_id"] if cur else None
    if train_id != running_id:
        return

    # 3. 运行中：持续监听 events.jsonl 文件追加
    pos = 0
    if events_file.is_file():
        pos = events_file.stat().st_size

    while True:
        await asyncio.sleep(1.0)
        if events_file.is_file():
            try:
                with events_file.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    lines = f.readlines()
                    pos = f.tell()
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                ev = json.loads(line)
                                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                                if ev.get("event") == "session_end":
                                    return
                            except Exception:
                                pass
            except Exception:
                pass

        status, cur = _proc_status()
        if cur is None or cur.get("train_id") != train_id or status != "running":
            # 子进程已退出，再次检查一次剩余文件内容
            if events_file.is_file():
                try:
                    with events_file.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                try:
                                    ev = json.loads(line)
                                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                                except Exception:
                                    pass
                except Exception:
                    pass
            break


def stop_training(train_id: str) -> dict[str, Any]:
    global _current
    with _lock:
        cur = _current
        if cur is None or cur["train_id"] != train_id:
            return {"ok": False, "error": "training_not_running"}
        if cur["proc"].poll() is None:
            cur["proc"].kill()
        return {"ok": True, "train_id": train_id}


# ── B 族：mRMR 因子推荐（同步子进程：物化因子面板 + 两两相关 → 推荐清单）──

RECOMMEND_ROOT = ROOT / "artifacts" / "alphaagent" / "recommend"
_recommend_busy = threading.Lock()


def run_mrmr_recommendation(params: dict[str, Any]) -> dict[str, Any]:
    """进程内快速计算 mRMR 因子推荐（Tool 执行核心）：
    复用磁盘因子值缓存与当前已加载的 CNE 宽表，免去重新 fork 进程的巨大延迟。
    """
    import pandas as pd
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.factor.cache import FactorValueCache
    from alphaagent.factor.stacking import build_stacking_dataset, collect_factor_entries
    from alphaagent.factor.stacking.model import mrmr_rank_features
    from alphaagent.factor.window_config import DEFAULT_TRAIN_END, DEFAULT_VAL_END

    k = int(params.get("k") or 8)
    k = max(1, min(k, 30))
    max_corr = float(params.get("max_corr") or 0.6)
    no_candidate = bool(params.get("no_candidate", False))
    include = params.get("include_factors")

    entries = collect_factor_entries(
        include_candidate=not no_candidate, include_production=True
    )
    if include and isinstance(include, list):
        wanted = {str(n).strip() for n in include if str(n).strip()}
        if wanted:
            entries = [e for e in entries if e.name in wanted]

    if len(entries) < 2:
        return {"ok": False, "error": "insufficient_factors", "message": "可用因子不足 2 个"}

    mining_end = pd.Timestamp(DEFAULT_TRAIN_END)
    panel_start = mining_end - pd.DateOffset(months=12) - pd.DateOffset(days=250)
    end = pd.Timestamp(DEFAULT_VAL_END)
    panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)

    cache = FactorValueCache()
    dataset = build_stacking_dataset(
        panel,
        entries,
        label_days=5,
        mining_end=mining_end,
        size_neutral=True,
        max_corr=max_corr,
        cache=cache,
        decay_months=12,
    )

    if len(dataset.feature_names) < 2:
        return {"ok": False, "error": "insufficient_valid_features", "message": "有效特征不足 2 个"}

    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    rec_start = mining_end - pd.DateOffset(months=12)
    ranking = mrmr_rank_features(
        dataset.feature_matrix,
        dataset.label,
        pd.Series(dts),
        dataset.feature_names,
        window_start=rec_start,
        window_end=mining_end,
        k=k,
        beta=0.7,
    )

    name_lib_map = {e.name: e.library for e in entries}
    for item in ranking:
        item["library"] = "正式" if "production" in name_lib_map.get(item["name"], "") else "候选"

    return {
        "ok": True,
        "k": k,
        "ranking": ranking,
        "recommended_names": [r["name"] for r in ranking],
        "window": f"[{rec_start.date()} ~ {mining_end.date()}]",
        "summary": f"已通过 mRMR 选出 {len(ranking)} 个互补因子",
    }


def start_recommend(params: dict[str, Any]) -> dict[str, Any]:
    """跑一次 mRMR 推荐：优先走进程内快速通道，失败回退子进程。"""
    try:
        res = run_mrmr_recommendation(params)
        if res.get("ok"):
            return res
    except Exception as exc:
        pass

    with _lock:
        if _current is not None and _current["proc"].poll() is None:
            return {"error": "training_already_running"}
    if not _recommend_busy.acquire(blocking=False):
        return {"error": "recommend_already_running"}

    train_id = _now_id()
    out_dir = RECOMMEND_ROOT / train_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "progress.log"
    try:
        modes = params.get("modes") or ["technical", "fundamental"]
        if isinstance(modes, str):
            modes = [modes]
        command = [
            str(PYTHON_EXECUTABLE), str(ROOT / "scripts" / "train_ml_composite.py"),
            "--modes", *[str(m) for m in modes],
            "--label-days", str(int(params.get("label_days") or 5)),
            "--max-corr", str(float(params.get("max_corr") or 0.6)),
            "--out-dir", str(out_dir),
            "--recommend-k", str(max(1, min(int(params.get("k") or 8), 30))),
            "--no-write-pred",
        ]
        if params.get("mining_end"):
            command += ["--mining-end", str(params["mining_end"])]
        if params.get("no_candidate"):
            command += ["--no-candidate"]
        if params.get("size_neutral") is False:
            command += ["--no-size-neutral"]
        include = params.get("include_factors") or None
        if include:
            command += ["--include-factors", *[str(n) for n in include]]
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
            )
        finally:
            log_handle.close()
        exit_code = proc.wait()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"spawn_failed: {exc}"}
    finally:
        _recommend_busy.release()

    rec_path = out_dir / "recommend.json"
    if exit_code != 0 or not rec_path.is_file():
        tail: list[str] = []
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        return {"error": f"recommend_failed(exit={exit_code})", "progress_tail": tail}
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        try:
            rec = json.loads(rec_path.read_text(encoding="gbk"))
        except (json.JSONDecodeError, OSError):
            return {"error": "recommend.json 读取失败"}
    rec["train_id"] = train_id
    rec["progress_log"] = str(log_path)
    return {"ok": True, **rec}

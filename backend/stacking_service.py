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
    """启动一次 ML 组合训练。返回 {train_id, status} 或 {error}。"""
    global _current
    with _lock:
        if _current is not None and _current["proc"].poll() is None:
            return {"error": "training_already_running",
                    "train_id": _current["train_id"]}

        train_id = _now_id()
        modes = params.get("modes") or ["technical", "fundamental"]
        if isinstance(modes, str):
            modes = [modes]
        modes = [str(m) for m in modes]
        out_dir = STACKING_ROOT / train_id
        log_path = _progress_log(train_id)

        command = [
            str(PYTHON_EXECUTABLE), str(ROOT / "scripts" / "train_ml_composite.py"),
            "--modes", *modes,
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
    return out


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


def start_recommend(params: dict[str, Any]) -> dict[str, Any]:
    """同步跑一次 mRMR 推荐：构建数据集（复用因子值磁盘缓存）→ recommend.json。

    与训练互斥（panel 全量物化内存重）；训练中或已有推荐在跑时直接返回 error。
    """
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

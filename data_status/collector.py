"""Collector: aggregates data status from parquet, services and resources.

All checks are aggregation-only (parquet footer metadata / DuckDB max date)
so a full run stays light on the 3.6G machine.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

from .config import (
    LOGS_DIR,
    PARQUET_TABLES,
    PG_PARQUET_DIR,
    QUANT_UI_ROOT,
    SCRIPTS_DIR,
    TABLE_DATE_COLUMN,
)
from .state import load_state, save_state


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parquet_stats(name: str) -> dict | None:
    path = PG_PARQUET_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    stat = path.stat()
    base = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }
    if pq is None:
        return base
    try:
        md = pq.ParquetFile(path).metadata
        base["rows"] = md.num_rows
        date_col = TABLE_DATE_COLUMN.get(name)
        if date_col:
            col_idx = {md.schema.column(i).name: i for i in range(md.num_columns)}
            if date_col in col_idx:
                mx = None
                ci = col_idx[date_col]
                for rg in range(md.num_row_groups):
                    s = md.row_group(rg).column(ci).statistics
                    if s is None:
                        continue
                    mx = s.max if mx is None else max(mx, s.max)
                base["max_date"] = str(mx) if mx is not None else None
    except Exception as exc:
        base["error"] = str(exc)
    return base


def collect_parquet() -> dict:
    out = {"status": "ok", "files": {}}
    missing = [t for t in PARQUET_TABLES
               if not (PG_PARQUET_DIR / f"{t}.parquet").exists()]
    if missing:
        out["status"] = "warn"
        out["missing"] = missing
    for name in PARQUET_TABLES:
        stats = _parquet_stats(name)
        if stats is not None:
            out["files"][name] = stats
    return out


def _parquet_max_date(path: Path, date_col: str, where: str = "") -> str | None:
    """用 DuckDB 读 parquet 最新日期（轻量，不整表进 pandas）。"""
    if not path.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect()
        try:
            where_sql = f" WHERE {where}" if where else ""
            row = con.execute(
                f"SELECT max(CAST({date_col} AS VARCHAR)) FROM read_parquet(?){where_sql}",
                [str(path)],
            ).fetchone()
            return str(row[0]) if row and row[0] is not None else None
        finally:
            con.close()
    except Exception:
        return None


def collect_sources(pq_data: dict) -> dict:
    """Compare calendar max vs stock_daily max to estimate Tushare lag."""
    out = {"status": "ok", "lag_days": None}
    try:
        cal = _parquet_max_date(
            PG_PARQUET_DIR / "trade_cal.parquet", "cal_date",
            where=f"{'cal_date'} <= CURRENT_DATE",
        )
        daily = pq_data.get("files", {}).get("stock_daily", {}).get("max_date")
        if cal and daily:
            d0 = datetime.strptime(cal, "%Y-%m-%d").date()
            d1 = datetime.strptime(daily, "%Y-%m-%d").date()
            lag = (d0 - d1).days
            out["calendar_max"] = cal
            out["stock_daily_max"] = daily
            out["lag_days"] = lag
            if lag > 5:
                out["status"] = "critical"
            elif lag > 1:
                out["status"] = "warn"
        else:
            out["errors"] = ["calendar or stock_daily max_date missing"]
    except Exception as exc:
        out["status"] = "error"
        out["errors"] = [str(exc)]
    return out


def collect_services() -> dict:
    out = {"status": "ok", "checks": {}}

    def _exec(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            return p.returncode, (p.stdout + p.stderr).strip()
        except Exception as exc:
            return -1, str(exc)

    # quant-api health endpoint.
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/api/health", timeout=5) as r:
            out["checks"]["quant-api"] = {"ok": r.status == 200, "detail": f"HTTP {r.status}"}
    except Exception as exc:
        out["checks"]["quant-api"] = {"ok": False, "detail": str(exc)}

    rc, out_s = _exec(["pgrep", "-f", "/home/ubuntu/qqbot/main.py"])
    out["checks"]["qqbot"] = {"ok": rc == 0, "detail": f"pid={out_s.strip()}" if rc == 0 else "not running"}

    # systemd timer last results.
    timers = [
        "quant-data-refresh.timer",
        "quant-healthcheck.timer",
        "quant-paper.timer",
        "quant-tencent-weekly.timer",
        "quant-tushare-expiry.timer",
        "qqbot-daily-push-noon.timer",
        "qqbot-daily-push-close.timer",
        "qqbot-daily-push-us.timer",
    ]
    for t in timers:
        rc, info = _exec(
            ["systemctl", "show", t, "--property=ActiveState,Result,LastTriggerUSec"],
            timeout=5,
        )
        if rc == 0 and info:
            kv = {}
            for line in info.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    kv[k] = v
            out["checks"][t] = {
                "ok": kv.get("ActiveState") in ("active", "activating") and kv.get("Result") in ("success", "skip"),
                "active": kv.get("ActiveState"),
                "last_result": kv.get("Result"),
                "last_trigger": kv.get("LastTriggerUSec"),
            }
    return out


def collect_resources() -> dict:
    out = {"status": "ok"}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            mem = {}
            for line in f:
                k, _, v = line.partition(":")
                mem[k.strip()] = int(v.strip().split()[0]) * 1024
        total = mem.get("MemTotal", 0)
        avail = mem.get("MemAvailable", 0)
        out["memory"] = {
            "total_bytes": total,
            "available_bytes": avail,
            "used_pct": round((1 - avail / total) * 100, 1) if total else None,
        }
    except Exception as exc:
        out["errors"] = [f"memory: {exc}"]
    try:
        du = shutil.disk_usage(QUANT_UI_ROOT)
        out["disk"] = {
            "total_bytes": du.total,
            "used_bytes": du.used,
            "free_bytes": du.free,
            "free_pct": round(du.free / du.total * 100, 1),
        }
        if du.free / du.total < 0.1:
            out["status"] = "critical"
    except Exception as exc:
        out.setdefault("errors", []).append(f"disk: {exc}")
    return out


def _overall(groups: dict[str, dict]) -> str:
    if any(g.get("status") == "critical" or g.get("status") == "error" for g in groups.values()):
        return "critical"
    if any(g.get("status") == "warn" for g in groups.values()):
        return "warn"
    return "ok"


def _alert(state: dict, groups: dict[str, dict]) -> None:
    prev = load_state().get("overall")
    cur = state["overall"]
    if prev == cur and prev != "ok":
        return
    if cur == "ok":
        if prev != "ok":
            text = "[data-status] 数据状态已恢复: ok"
        else:
            return
    else:
        problems = []
        for name, g in groups.items():
            if g.get("status") in ("critical", "error"):
                problems.append(f"- {name}: {g.get('errors', g.get('status'))}")
        text = f"[data-status] 数据状态异常: {cur}\n" + "\n".join(problems[:20])
    try:
        sys.path.insert(0, str(SCRIPTS_DIR))
        from healthcheck import send_qq_alert  # type: ignore
        send_qq_alert(text)
    except Exception:
        pass


def collect_all(alert: bool = True) -> dict:
    started = time.time()
    groups = {
        "parquet": collect_parquet(),
    }
    groups["sources"] = collect_sources(groups["parquet"])
    groups["services"] = collect_services()
    groups["resources"] = collect_resources()
    state = {
        "updated_at": _utcnow(),
        "elapsed_s": round(time.time() - started, 2),
        "overall": _overall(groups),
        "groups": groups,
    }
    save_state(state)
    if alert:
        _alert(state, groups)
    return state


def ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""与腾讯云服务器双向同步 data/ 数据文件。

用法:
    python scripts/sync_server.py                     # dry-run（默认，只打印动作）
    python scripts/sync_server.py --apply             # 执行同步
    python scripts/sync_server.py --apply --group merge
    python scripts/sync_server.py --apply --path report_rc
    python scripts/sync_server.py --apply --rebuild-panel-remote

规则由 scripts/sync_manifest.json 定义:
    server_to_local : 服务器权威，覆盖本机
    local_to_server : 本机权威，覆盖服务器
    merge           : 双向取并集（如 report_rc 按 ts_code+report_date）

安全机制:
    - 默认 dry-run，只打印计划动作
    - 覆盖前备份: 本机 -> data/sync_backup/<时间戳>/，服务器 -> ~/quant_sync_backup/<时间戳>/
    - 传输后 SHA256 校验，不一致时报错并保留备份
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
LOCAL_BASE = ROOT / "data"
MANIFEST_PATH = ROOT / "scripts" / "sync_manifest.json"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
TS_FORMAT = "%Y%m%d_%H%M%S"

manifest: dict = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def ssh(server: str, remote_cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    # 服务器输出是 UTF-8 中文，Windows 默认 GBK 解码会崩，显式指定编码
    return subprocess.run(
        ["ssh", *SSH_OPTS, server, remote_cmd],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def shlex_quote(s: str) -> str:
    # 路径不含引号时直接加单引号即可；兼容 Windows subprocess 传参。
    return "'" + s.replace("'", "'\\''") + "'"


def remote_stat(server: str, path: str) -> tuple[int, int] | None:
    """返回远程文件 (size, mtime)；不存在返回 None。"""
    proc = ssh(server, f"stat -c '%s %Y' -- {shlex_quote(path)}", check=False)
    if proc.returncode != 0:
        return None
    parts = proc.stdout.strip().split()
    if len(parts) != 2:
        return None
    return int(parts[0]), int(parts[1])


def remote_hash(server: str, path: str) -> str | None:
    proc = ssh(server, f"sha256sum -- {shlex_quote(path)}", check=False)
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out.split()[0] if out else None


def local_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_stat(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return st.st_size, int(st.st_mtime)
    except FileNotFoundError:
        return None


def local_abs(rel: str) -> Path:
    return LOCAL_BASE.joinpath(*rel.split("/"))


def remote_abs(rel: str, base: str) -> str:
    return base.rstrip("/") + "/" + rel


def backup_local(rel: str, ts: str) -> Path:
    src = local_abs(rel)
    if not src.exists():
        return src
    dst = LOCAL_BASE / "sync_backup" / ts / rel.replace("/", "\\")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log(f"  [bak] 本机备份 {rel} -> {dst}")
    return dst


def backup_remote(server: str, rel: str, base: str, backup_root: str, ts: str) -> None:
    src = remote_abs(rel, base)
    rel_dir = "/".join(rel.split("/")[:-1]) if "/" in rel else "."
    dst_dir = backup_root.rstrip("/") + "/" + ts + "/" + rel_dir
    dst = backup_root.rstrip("/") + "/" + ts + "/" + rel
    proc = ssh(
        server,
        f"mkdir -p {shlex_quote(dst_dir)} && cp -p {shlex_quote(src)} {shlex_quote(dst)}",
        check=False,
    )
    if proc.returncode != 0:
        log(f"  [warn] 远程备份失败 {rel}: {proc.stderr.strip()}")
    else:
        log(f"  [bak] 远程备份 {rel} -> {dst}")


def scp_download(server: str, remote_path: str, local_tmp: Path) -> None:
    # Windows OpenSSH scp 大文件会卡在 0 字节，改用 ssh cat 流式传输
    with local_tmp.open("wb") as f:
        proc = subprocess.run(
            ["ssh", *SSH_OPTS, server, f"cat {shlex_quote(remote_path)}"],
            stdout=f,
            stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"下载失败 {remote_path}: {proc.stderr.decode('utf-8', 'replace').strip()}")


def scp_upload(server: str, local_tmp: Path, remote_path: str) -> None:
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, server, f"cat > {shlex_quote(remote_path)}"],
        stdin=local_tmp.open("rb"),
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"上传失败 {remote_path}: {proc.stderr.decode('utf-8', 'replace').strip()}")


REMOTE_FRESH = r"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

def _pick(names):
    for c in ("trade_date", "date", "report_date", "cal_date", "nav_date", "surv_date", "end_date", "ann_date", "start_date", "list_date"):
        if c in names:
            return c
    return next((c for c in names if c.endswith("_date")), None)

def _sig(path: Path):
    if path.suffix.lower() == ".csv":
        try:
            n = sum(1 for _ in open(path, encoding="utf-8", errors="ignore"))
        except Exception:
            n = -1
        return {"rows": n, "max_date": None}
    pf = pq.ParquetFile(str(path))
    rows = pf.metadata.num_rows
    names = pf.schema_arrow.names
    col = _pick(names)
    mx = None
    if col:
        try:
            s = pd.to_datetime(pq.read_table(str(path), columns=[col]).column(col).to_pandas(), errors="coerce").dropna()
            if len(s):
                mx = str(s.max())[:10]
        except Exception:
            mx = None
    return {"rows": rows, "max_date": mx}

p = Path(sys.argv[1])
print(json.dumps({"missing": not p.exists()} | ({} if not p.exists() else _sig(p))))
"""


REMOTE_SLICE = r"""
import sys, json
import duckdb

src, out, col, gt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
con = duckdb.connect()
sql = "SELECT * FROM read_parquet(?) WHERE " + col + " > CAST(? AS DATE)"
n = con.execute("SELECT count(*) FROM (" + sql + ")", [src, gt]).fetchone()[0]
con.execute("COPY (" + sql + ") TO '" + out + "' (FORMAT PARQUET, COMPRESSION ZSTD)", [src, gt])
print(json.dumps({"rows": int(n)}))
"""


def incremental_download(server: str, base: str, rel: str, item: dict, ts: str) -> bool:
    """只拉本机缺的日期切片并本地追加；成功返回 True，否则 False（回退全量）。"""
    rp = remote_abs(rel, base)
    lp = local_abs(rel)
    date_col = item["date_col"]
    key_cols = item.get("key_cols") or []
    lf = freshness_local(lp)
    rf = freshness_remote(server, rp)
    if not lf or not rf or not lf.get("max_date") or not rf.get("max_date"):
        return False
    if rf["max_date"] <= lf["max_date"]:
        return False
    days = (pd.Timestamp(rf["max_date"]) - pd.Timestamp(lf["max_date"])).days
    if days > 10:
        log(f"  [info] {rel} 增量区间 {days} 天过大，回退全量")
        return False

    remote_tmp = "/tmp/quant_delta_" + rel.replace("/", "_")
    script_args = " ".join(
        shlex_quote(a) for a in (rp, remote_tmp, date_col, lf["max_date"])
    )
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, server, manifest["remote_python"] + " - " + script_args],
        input=REMOTE_SLICE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        log(f"  [warn] 远程切片失败: {proc.stderr.strip()}")
        return False
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    try:
        remote_rows = int(json.loads(lines[-1]).get("rows", 0))
    except Exception:
        remote_rows = 0
    if remote_rows <= 0:
        ssh(server, f"rm -f {shlex_quote(remote_tmp)}", check=False)
        return False

    tmp_dir = Path(tempfile.mkdtemp(prefix="quant_delta_"))
    local_tmp = tmp_dir / Path(rel).name
    scp_download(server, remote_tmp, local_tmp)
    ssh(server, f"rm -f {shlex_quote(remote_tmp)}", check=False)

    schema_local = pq.read_schema(lp)
    schema_delta = pq.read_schema(local_tmp)
    if [n for n in schema_delta.names] != [n for n in schema_local.names]:
        log(f"  [warn] {rel} 增量 schema 不一致，回退全量")
        return False

    df_new = pd.concat([pd.read_parquet(lp), pd.read_parquet(local_tmp)], ignore_index=True)
    before = len(df_new)
    if key_cols:
        df_new = df_new.drop_duplicates(subset=key_cols, keep="last")
    backup_local(rel, ts)
    table = pa.Table.from_pandas(df_new, schema=schema_local, preserve_index=False)
    pq.write_table(table, lp, compression="zstd")
    log(
        f"  [ok] 增量 {rel}: 拉取 {remote_rows:,} 行（> {lf['max_date']}），"
        f"追加后 {len(df_new):,} 行"
    )
    return True


def freshness_local(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        try:
            n = sum(1 for _ in open(path, encoding="utf-8", errors="ignore"))
        except Exception:
            n = -1
        return {"rows": n, "max_date": None}
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    rows = pf.metadata.num_rows
    names = pf.schema_arrow.names
    col = next((c for c in ("trade_date", "date", "report_date", "cal_date", "nav_date", "surv_date", "end_date", "ann_date", "start_date", "list_date") if c in names), None)
    if col is None:
        col = next((c for c in names if c.endswith("_date")), None)
    mx = None
    if col:
        try:
            s = pd.to_datetime(pq.read_table(str(path), columns=[col]).column(col).to_pandas(), errors="coerce").dropna()
            if len(s):
                mx = str(s.max())[:10]
        except Exception:
            mx = None
    return {"rows": rows, "max_date": mx}


def freshness_remote(server: str, path: str) -> dict | None:
    proc = subprocess.run(
        ["ssh", *SSH_OPTS, server, manifest["remote_python"] + " - " + shlex_quote(path)],
        input=REMOTE_FRESH,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return None
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        sig = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    if sig.get("missing"):
        return None
    return {"rows": sig.get("rows"), "max_date": sig.get("max_date")}


def smart_decision(ls: dict | None, rs: dict | None) -> str:
    """数据源规则：服务器优先；状态不确定/有异常时一律拉服务器。

    - 本机缺失或读取异常 -> download（拉服务器）
    - 服务器读取异常但文件在 -> download（服务器权威）
    - 只有明确本机 max_date 更新/行数更全时 -> upload（补服务器）
    """
    if ls is None and rs is None:
        return "skip"
    if ls is None:
        return "download"
    if rs is None:
        return "download"
    lm, rm = ls.get("max_date"), rs.get("max_date")
    lr, rr = ls.get("rows"), rs.get("rows")
    if lm and rm:
        if rm > lm:
            return "download"
        if lm > rm:
            return "upload"
        if rr >= lr:
            return "download" if rr > lr else "skip"
        return "upload"
    # 日期缺失：服务器优先；本机行数更多才补服务器
    if rr >= lr:
        return "download" if rr > lr else "skip"
    return "upload"


def expand_glob(items: list[dict]) -> list[dict]:
    """展开带 ** 的目录通配（如 results/**），按本机现有文件生成逐文件条目。"""
    out: list[dict] = []
    for item in items:
        path = item["path"]
        if "**" not in path:
            out.append(item)
            continue
        prefix = path.split("**")[0].rstrip("/")
        base = LOCAL_BASE.joinpath(*prefix.split("/"))
        if base.exists():
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    rel = prefix + "/" + f.relative_to(base).as_posix()
                    out.append({"path": rel, "direction": item["direction"]})
    return out


def plan_item(server: str, base: str, item: dict) -> dict:
    rel = item["path"]
    direction = item["direction"]
    lp = local_abs(rel)
    rp = remote_abs(rel, base)
    ls = local_stat(lp)
    rs = remote_stat(server, rp)
    action = {
        "rel": rel,
        "direction": direction,
        "action": None,
        "detail": "",
        "local": ls,
        "remote": rs,
    }
    if item.get("smart"):
        ls = local_stat(lp)
        rs = remote_stat(server, rp)
        if ls is None and rs is None:
            action["action"] = "skip"
            action["detail"] = "两侧都缺失"
        elif ls is None:
            action["action"] = "download"
            action["detail"] = "本机缺失，拉服务器"
        elif rs is None:
            action["action"] = "upload"
            action["detail"] = "服务器缺失，推本机"
        else:
            lf = freshness_local(lp)
            rf = freshness_remote(server, rp)
            action["action"] = smart_decision(lf, rf)
            action["detail"] = (
                f"本机 rows={lf and lf['rows']},max={lf and lf['max_date']} / "
                f"服务器 rows={rf and rf['rows']},max={rf and rf['max_date']}"
            )
        return action
    if direction == "merge":
        if rs is None:
            action["action"] = "skip"
            action["detail"] = "远程缺失，无法合并"
        elif ls is None:
            action["action"] = "skip"
            action["detail"] = "本机缺失，无法合并"
        elif remote_hash(server, rp) == local_hash(lp):
            action["action"] = "skip"
            action["detail"] = "两侧已一致"
        else:
            action["action"] = "merge"
            action["detail"] = f"本机{ls[0]:,}B / 远程{rs[0]:,}B 不一致"
        return action
    if direction in ("server_to_local", "local_to_server"):
        src_stat = rs if direction == "server_to_local" else ls
        dst_stat = ls if direction == "server_to_local" else rs
        if src_stat is None:
            action["action"] = "skip"
            action["detail"] = "源缺失"
            return action
        if dst_stat is None:
            action["action"] = "download" if direction == "server_to_local" else "upload"
            action["detail"] = "目标缺失"
            return action
        if src_stat[0] == dst_stat[0]:
            action["action"] = "skip"
            action["detail"] = "大小一致"
            return action
        action["action"] = "download" if direction == "server_to_local" else "upload"
        action["detail"] = f"本机{ls[0]:,}B / 远程{rs[0]:,}B 不一致"
        return action
    action["action"] = "skip"
    action["detail"] = f"未知方向 {direction}"
    return action


def do_download(server: str, base: str, rel: str, ts: str, item: dict | None = None) -> None:
    if item and item.get("incremental"):
        try:
            if incremental_download(server, base, rel, item, ts):
                return
        except Exception as exc:
            log(f"  [warn] 增量下载失败({exc})，回退全量")
    rp = remote_abs(rel, base)
    lp = local_abs(rel)
    lp.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="quant_sync_")) / Path(rel).name
    backup_local(rel, ts)
    scp_download(server, rp, tmp)
    remote_sha = remote_hash(server, rp)
    if remote_sha is None:
        raise RuntimeError(f"远程校验失败: {rel}")
    if local_hash(tmp) != remote_sha:
        raise RuntimeError(f"SHA256 不一致，已保留备份: {rel}")
    shutil.move(str(tmp), str(lp))
    log(f"  [ok] 下载 {rel} ({lp.stat().st_size:,} B)")


def do_upload(server: str, base: str, rel: str, ts: str) -> None:
    rp = remote_abs(rel, base)
    lp = local_abs(rel)
    backup_remote(server, rel, base, manifest["remote_backup"], ts)
    ssh(server, f"mkdir -p {shlex_quote('/'.join(rp.split('/')[:-1]))}", check=False)
    scp_upload(server, lp, rp)
    remote_sha = remote_hash(server, rp)
    if remote_sha is None or remote_sha != local_hash(lp):
        raise RuntimeError(f"SHA256 不一致，已保留远程备份: {rel}")
    log(f"  [ok] 上传 {rel}")


def do_merge(server: str, base: str, rel: str, ts: str, keys: list[str]) -> None:
    rp = remote_abs(rel, base)
    lp = local_abs(rel)
    remote_sha = remote_hash(server, rp)
    # 两侧文件已一致则跳过
    if (
        remote_sha is not None
        and local_stat(lp) is not None
        and remote_stat(server, rp) == local_stat(lp)
        and remote_sha == local_hash(lp)
    ):
        log(f"  [skip] 合并 {rel}: 两侧已一致")
        return
    remote_tmp = Path(tempfile.mkdtemp(prefix="quant_sync_")) / Path(rel).name
    scp_download(server, rp, remote_tmp)
    df_local = pd.read_parquet(lp)
    df_remote = pd.read_parquet(remote_tmp)
    schema = pq.read_schema(lp)
    before = len(df_local) + len(df_remote)
    df = pd.concat([df_local, df_remote], ignore_index=True)
    exact_dup = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first")
    backup_local(rel, ts)
    backup_remote(server, rel, base, manifest["remote_backup"], ts)
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, lp, compression="zstd")
    scp_upload(server, lp, rp)
    remote_sha = remote_hash(server, rp)
    if remote_sha is None or remote_sha != local_hash(lp):
        raise RuntimeError(f"SHA256 不一致，已保留备份: {rel}")
    log(
        f"  [ok] 合并 {rel}: 本机 {len(df_local):,} 行 + 远程 {len(df_remote):,} 行 "
        f"- 完全重复 {exact_dup:,} 行 = 并集 {len(df):,} 行（未按 key 去重，保留两侧全部数据）"
    )


def update_local_meta() -> None:
    """panel 替换后刷新本机 meta.json，避免与文件内容不一致。"""
    panel = LOCAL_BASE / "panel.parquet"
    if not panel.exists():
        return
    try:
        t = pq.read_table(panel, columns=["date", "code"])
        df = t.to_pandas()
        meta = {
            "last_update": datetime.now().isoformat(timespec="seconds"),
            "mode": "pg_rebuild",
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "n_codes": int(df["code"].nunique()),
            "n_rows": int(len(df)),
        }
        (LOCAL_BASE / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"  [ok] 刷新本机 meta.json: {meta}")
    except Exception as exc:
        log(f"  [warn] meta.json 刷新失败: {exc}")


def main() -> int:
    global manifest
    ap = argparse.ArgumentParser(description="与腾讯云服务器双向同步 data/ 数据")
    ap.add_argument("--apply", action="store_true", help="执行同步（默认 dry-run）")
    ap.add_argument("--group", help="只处理指定方向: server_to_local / local_to_server / merge")
    ap.add_argument("--path", help="只处理路径包含该子串的文件")
    ap.add_argument("--rebuild-panel-remote", action="store_true", help="同步前先在服务器重建 panel")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    server = manifest["server"]
    base = manifest["remote_base"]

    if args.rebuild_panel_remote:
        log("== 服务器重建 panel ==")
        proc = ssh(
            server,
            "/home/ubuntu/stock-analyzer/local_venv/bin/python "
            "/home/ubuntu/quant/quant_ui/scripts/rebuild_stock_panel_from_pg.py --start 2015-01-01",
            check=False,
        )
        log(proc.stdout)
        if proc.returncode != 0:
            log(proc.stderr)
            return 1

    items = expand_glob(manifest["items"])
    if args.group:
        items = [i for i in items if i["direction"] == args.group]
    if args.path:
        items = [i for i in items if args.path in i["path"]]
    if not items:
        log("没有匹配的条目")
        return 0

    log("== 计划 ==")
    plans = [plan_item(server, base, item) for item in items]
    for p in plans:
        act = p["action"] or "?"
        log(f"  [{act:8}] {p['direction']:16} {p['rel']}  ({p['detail']})")

    if not args.apply:
        log("\n(dry-run，未执行任何操作；加 --apply 执行)")
        return 0

    ts = datetime.now().strftime(TS_FORMAT)
    log("\n== 执行 ==")
    errors = 0
    panel_replaced = False
    for p in plans:
        try:
            if p["action"] == "download":
                item = next((i for i in items if i["path"] == p["rel"]), None)
                do_download(server, base, p["rel"], ts, item)
                if p["rel"] == "panel.parquet":
                    panel_replaced = True
            elif p["action"] == "upload":
                do_upload(server, base, p["rel"], ts)
            elif p["action"] == "merge":
                item = next(i for i in items if i["path"] == p["rel"])
                do_merge(server, base, p["rel"], ts, item.get("merge_keys", []))
        except Exception as exc:
            errors += 1
            log(f"  [FAIL] {p['rel']}: {exc}")
    if panel_replaced:
        update_local_meta()
    log(f"\n完成: {len(plans)} 项，错误 {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MinerU 全量解析驱动（GPU basic 档）：断点续跑 + 并发 + 计时 + 失败重试。

背景（2026-09-29）：本地已装 MinerU 4.0.9（`D:\\Quant\\MinerU`，独立 venv，torch 2.14.0+cu132）、
managed parse-server 已开（`parse_server.local.mode=managed` + `managed_tier=basic`，模型走 ModelScope）。
本脚本把 `data/research_reports/**/*.pdf` 逐篇交给 MinerU，产出 `parsed_mineru/<相对路径>.md`。

要点：
- **必须传 `--pages all`**：MinerU 默认只解析前 10 页。
- `-o` 是**文件路径**（传目录会报 PermissionError）。
- **断点续跑**：已存在且非空的 md 直接跳过，可安全重跑。
- **图片引用后处理**：MinerU 输出 `![Chart block](doc:<id>/tier:basic/page:N/block:M)` 这类
  doclib 定位符（不是文件，离开本地 doclib 即失效）→ 统一转成
  ``<!-- image: page:N block:M -->`` 注释，保留页码/块号溯源但不再产生死链。
- 并发：`--workers N`（默认 2）。显存 8GB 下 basic 档开 2~3 路较稳。

2026-09-30 review 修：
- 超时改为**杀进程树**（MinerU 会拉起 parse-server 子进程，只杀父进程会留下占显存的孤儿）；
- **非零退出码不再采信产物**（错误页/半成品只要 >1000 字节就会被永久 skip，属数据污染）；
- md 后处理改**原子写**（write_text 中断会留下截断文件且原内容已丢）；
- MinerU 路径支持 `--mineru` / `MINERU_EXE`（原为硬编码 Windows 路径）；
- 扫描 PDF 时按输出根目录排除（原判断 `"parsed" not in p.parts` 与 `parsed_mineru` 永不相等）；
- 元数据每行带 `run` 批次号，便于区分多次运行。

用法::

    # 全量（后台推荐）
    python scripts/batch_mineru_parse.py --workers 2

    # 冒烟 20 篇
    python scripts/batch_mineru_parse.py --limit 20

    # 只跑某一类
    python scripts/batch_mineru_parse.py --only "因子周报"

    # 指定 MinerU CLI
    python scripts/batch_mineru_parse.py --mineru "D:\\MinerU\\.venv\\Scripts\\mineru.exe"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MINERU_DEFAULT = Path(r"D:\Quant\MinerU\.venv\Scripts\mineru.exe")
MINERU = Path(os.environ.get("MINERU_EXE") or MINERU_DEFAULT)
RE_DOC_IMG = re.compile(r"!\[([^\]]*)\]\(doc:[^)]*?/page:(\d+)/block:(\d+)\)")
# Windows：让子进程不再各弹一个控制台黑框（本机控制台托管给 Windows Terminal，
# 父进程的 -WindowStyle Hidden 对子进程无效，必须显式 CREATE_NO_WINDOW）
CREATE_NO_WINDOW = 0x08000000
# md 有效判定阈值（字节）：小于此值视为空/半成品，会重新解析
MIN_MD_BYTES = 1000


def default_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return Path(out).parent / "data" / "research_reports"
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parents[1] / "data" / "research_reports"


RE_YEAR = re.compile(r"(20\d{2})")


def year_of(p: Path) -> int:
    """从文件名提取年份（如 2019-05-21_华泰… / 20240924_5775175_…）；无则 0。"""
    m = RE_YEAR.search(p.name)
    return int(m.group(1)) if m else 0


def _md_state(p: Path) -> tuple[bool, int]:
    """(存在, 字节数)；不存在或 stat 失败返回 (False, 0)。"""
    try:
        return True, p.stat().st_size
    except OSError:
        return False, 0


def _kill_tree(proc: subprocess.Popen) -> None:
    """杀掉整棵进程树（Windows 用 taskkill /T；POSIX 退化为 kill）。"""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=CREATE_NO_WINDOW)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    """先写临时文件再 replace：中断时不会留下截断的正式产物。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def postprocess(md_path: Path) -> int:
    """把 doclib 图片定位符转成 HTML 注释；返回替换数。"""
    try:
        txt = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    new, n = RE_DOC_IMG.subn(lambda m: f"<!-- image: page:{m.group(2)} block:{m.group(3)} {m.group(1)} -->", txt)
    if n:
        _atomic_write_text(md_path, new)
    return n


def parse_one(args: tuple[int, str], root: Path, out_root: Path, mineru: Path,
              timeout: int, retries: int, run_id: str) -> dict:
    idx, pdf_str = args
    pdf = Path(pdf_str)
    rel = pdf.relative_to(root)
    out_md = out_root / rel.with_suffix(".md")
    rec = {"run": run_id, "index": idx, "rel_path": str(rel), "out": str(out_md)}
    _exists, _size = _md_state(out_md)
    if _exists and _size > MIN_MD_BYTES:
        rec["status"] = "skipped"
        return rec
    out_md.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    last_err = ""
    for attempt in range(retries + 1):
        t0 = time.time()
        cmd = [str(mineru), "parse", str(pdf), "--pages", "all", "--tier", "basic",
               "-o", str(out_md), "--wait", str(timeout)]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", env=env,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                out, err = proc.communicate(timeout=timeout + 120)
            except subprocess.TimeoutExpired:
                # 必须杀**进程树**：MinerU 会拉起 parse-server 子进程，只杀父进程会在
                # GPU 上留下占显存的孤儿，后续重试/其它 worker 直接争不到显存。
                _kill_tree(proc)
                try:
                    out, err = proc.communicate(timeout=30)
                except Exception:  # noqa: BLE001
                    out, err = "", ""
                last_err = f"timeout>{timeout}s（已杀进程树）"
                time.sleep(3 * (attempt + 1))
                continue
            exists, size = _md_state(out_md)
            if proc.returncode != 0:
                # 非零退出码不采信产物：MinerU 失败时也可能留下 >1000 字节的错误页/
                # 半成品，若判 ok 会被后续断点续跑永久 skip（污染知识库）。
                last_err = (f"returncode={proc.returncode} "
                            + (out or "")[-200:] + (err or "")[-300:])
                time.sleep(3 * (attempt + 1))
                continue
            if exists and size > MIN_MD_BYTES:
                rec.update(status="ok", bytes=size, image_refs=postprocess(out_md),
                           time_s=round(time.time() - t0, 1), attempt=attempt + 1)
                return rec
            last_err = f"no_valid_output size={size} " + (out or "")[-200:] + (err or "")[-300:]
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
        time.sleep(3 * (attempt + 1))
    rec.update(status="error", error=last_err[:400])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="MinerU 全量解析驱动（GPU basic 档，断点续跑）")
    ap.add_argument("--root", default=None, help="PDF 根目录（默认 data/research_reports）")
    ap.add_argument("--out", default=None, help="输出根目录（默认 <root>/parsed_mineru）")
    ap.add_argument("--mineru", default=None,
                    help=f"MinerU CLI 路径（默认 $MINERU_EXE 或 {MINERU_DEFAULT}）")
    ap.add_argument("--workers", type=int, default=2, help="并发篇数（8GB 显存建议 2~3）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇")
    ap.add_argument("--only", default="", help="只处理路径含该子串的报告")
    ap.add_argument("--min-year", type=int, default=0,
                    help="跳过文件名年份早于该值的报告（0=不限）；老研报对当前预测窗口价值低")
    ap.add_argument("--order", choices=["recent", "path"], default="recent",
                    help="解析顺序：recent=按文件名年份倒序（优先新研报），path=路径顺序")
    ap.add_argument("--timeout", type=int, default=1800, help="单篇 --wait 上限（秒）")
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else default_root()
    out_root = Path(args.out).resolve() if args.out else root / "parsed_mineru"
    out_root.mkdir(parents=True, exist_ok=True)
    mineru = Path(args.mineru).resolve() if args.mineru else MINERU
    if not mineru.exists():
        print(f"找不到 MinerU CLI: {mineru}（可用 --mineru 或 MINERU_EXE 指定）", file=sys.stderr)
        return 2

    # 排除输出目录内的 PDF：原写法 `"parsed" not in p.parts` 与实际目录名
    # `parsed_mineru` 永不相等（等于没过滤）。
    pdfs = [p for p in root.rglob("*.pdf") if out_root not in p.parents]
    if args.only:
        pdfs = [p for p in pdfs if args.only in str(p)]
    if args.min_year:
        before = len(pdfs)
        pdfs = [p for p in pdfs if year_of(p) >= args.min_year]
        print(f"年份过滤 <{args.min_year}: 跳过 {before - len(pdfs)} 篇，保留 {len(pdfs)} 篇", flush=True)
    if args.order == "recent":
        pdfs.sort(key=lambda p: (-year_of(p), str(p)))
    else:
        pdfs.sort()
    if args.limit:
        pdfs = pdfs[: args.limit]
    total = len(pdfs)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"PDF: {total}  workers={args.workers}  out={out_root}  mineru={mineru}  run={run_id}", flush=True)
    if not total:
        return 0

    meta_path = out_root / "batch_metadata.jsonl"
    done = ok = err = skip = 0
    t_start = time.time()
    with meta_path.open("a", encoding="utf-8") as mf, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_one, (i, str(p)), root, out_root, mineru,
                          args.timeout, args.retries, run_id): i
                for i, p in enumerate(pdfs)}
        for fut in as_completed(futs):
            r = fut.result()
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            mf.flush()
            done += 1
            ok += r["status"] == "ok"
            err += r["status"] == "error"
            skip += r["status"] == "skipped"
            if done % 10 == 0 or done == total:
                el = time.time() - t_start
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                print(f"[{done}/{total}] ok={ok} skip={skip} err={err} "
                      f"{el/60:.1f}min  ETA {eta/60:.0f}min", flush=True)
    print(f"\n完成: total={total} ok={ok} skipped={skip} errors={err} "
          f"耗时 {(time.time() - t_start)/60:.1f} 分钟\n元数据: {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

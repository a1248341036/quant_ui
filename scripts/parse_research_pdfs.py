#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报 PDF 批量解析：PyMuPDF 正文（span 过滤）+ pdfplumber 表格（结构过滤）。

口径（与 2026-09-24 的 batch_parse_v3 一致，落库以便复现）：
- 正文：PyMuPDF ``get_text("dict")``，只保留 span ``size >= --min-size``（默认 9pt）
  的行；丢弃单 CJK 字符行（图表刻度/轴标签噪声）。
- 表格：pdfplumber ``extract_tables()``，丢弃 ≤1 行/≤1 列的表，以及非空单元格
  占比 < ``--min-table-ratio`` 的表；保留的表转 markdown 追加到正文之后
  （``## 表格数据`` 段）。
- **不提取图片**：金工研报的因子公式/因子表现图/IC 表大量以图片形式存在。
  下游不得从图注或上下文"补全"公式（幻觉源）——见
  ``docs/specs/alphaagent_report_knowledge_integration_spec.md`` §7.3。

输出：镜像源目录结构到 ``<out>/<相对源路径>.md``（``parsed/`` 自身被排除），
结束（含被中断）时写 ``<out>/batch_metadata.json``。

断点续传：已存在的 .md 直接跳过，可安全重跑。

用法::

    # 默认：解析主工作区 data/research_reports 下全部 PDF（排除 parsed/）
    python scripts/parse_research_pdfs.py

    # 冒烟：只跑 5 篇
    python scripts/parse_research_pdfs.py --limit 5

    # 指定源与输出（数据目录在 worktree 中不存在，故显式指定）
    python scripts/parse_research_pdfs.py --root D:/Quant/quant_ui/data/research_reports

依赖：``pymupdf``（正文）+ ``pdfplumber``（表格），需在当前 venv 内可用。
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RE_SINGLE_CJK = re.compile(r"^[\u4e00-\u9fff]$")

# worker 进程配置。Windows 下 ProcessPoolExecutor 用 spawn：子进程重新 import 本模块，
# 父进程运行时写入的全局量**不会**继承，故经 initializer 显式下发。
_CFG: dict = {}


def _init_worker(base: Path, out_base: Path, min_size: float, min_table_ratio: float) -> None:
    _CFG.update(base=base, out_base=out_base, min_size=min_size, min_table_ratio=min_table_ratio)


def _default_root() -> Path:
    """定位主工作区的 ``data/research_reports``。

    独立 worktree 里 ``data/`` 不存在（被 .gitignore 排除、不进版本控制），
    因此用 git common dir 反查主仓库，避免误写到 worktree 内的空目录。
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return Path(out).parent / "data" / "research_reports"
    except Exception:  # noqa: BLE001 - 非 git 环境回落
        pass
    return Path(__file__).resolve().parents[1] / "data" / "research_reports"


def extract_text_pymupdf(pdf_path: str, min_size: float) -> str:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        pages = []
        for page in doc:
            d = page.get_text("dict")
            lines_out = []
            for block in d["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    parts = [s["text"] for s in line["spans"] if s["size"] >= min_size]
                    text = "".join(parts).strip()
                    if not text or RE_SINGLE_CJK.match(text):
                        continue
                    lines_out.append(text)
            pages.append("\n".join(lines_out))
        return "\n\n".join(pages)
    finally:
        doc.close()


def extract_tables_pdfplumber(pdf_path: str, min_table_ratio: float) -> str:
    import pdfplumber

    tables_md = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            for t in page.extract_tables():
                if not t or len(t) <= 1 or len(t[0]) <= 1:
                    continue
                total = sum(len(r) for r in t)
                nonempty = sum(1 for r in t for c in r if c and str(c).strip())
                if total == 0 or nonempty / total < min_table_ratio:
                    continue
                rows = []
                for ri, row in enumerate(t):
                    cells = [str(c).replace("\n", " ").strip() if c is not None else "" for c in row]
                    rows.append("| " + " | ".join(cells) + " |")
                    if ri == 0:
                        rows.append("|" + "|".join(["---"] * len(row)) + "|")
                tables_md.append(f"<!-- table p{pno + 1} -->\n" + "\n".join(rows))
    return "\n\n".join(tables_md)


def parse_single(args: tuple[int, str]) -> dict:
    idx, pdf_path_str = args
    base: Path = _CFG["base"]
    out_base: Path = _CFG["out_base"]
    min_size: float = _CFG["min_size"]
    min_table_ratio: float = _CFG["min_table_ratio"]

    pdf_path = Path(pdf_path_str)
    rel = str(pdf_path.relative_to(base))
    out_file = out_base / Path(rel).with_suffix(".md")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    result = {"index": idx, "rel_path": rel, "name": pdf_path.stem, "out": str(out_file)}

    if out_file.exists():
        result["status"] = "skipped"
        return result

    try:
        t0 = time.time()
        text = extract_text_pymupdf(str(pdf_path), min_size)
        tables = extract_tables_pdfplumber(str(pdf_path), min_table_ratio)
        md = text + ("\n\n## 表格数据\n\n" + tables if tables else "")
        out_file.write_text(md, encoding="utf-8")
        result.update(
            status="ok",
            chars=len(md),
            text_chars=len(text),
            table_chars=len(tables),
            time_s=round(time.time() - t0, 2),
        )
    except Exception as e:  # noqa: BLE001 - 单篇失败不阻断整批
        result["status"] = "error"
        result["error"] = str(e)[:300]
    finally:
        gc.collect()
    return result


def _write_metadata(out_base: Path, results: list[dict]) -> Path:
    meta_file = out_base / "batch_metadata.json"
    ordered = sorted((r for r in results if r), key=lambda r: r["index"])
    meta_file.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_file


def main() -> int:
    ap = argparse.ArgumentParser(description="研报 PDF 批量解析（PyMuPDF 正文 + pdfplumber 表格）")
    ap.add_argument("--root", default=None, help="源根目录（默认：主工作区 data/research_reports）")
    ap.add_argument("--out", default=None, help="输出目录（默认：<root>/parsed）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-size", type=float, default=9.0, help="正文 span 最小字号（pt）")
    ap.add_argument("--min-table-ratio", type=float, default=0.3, help="表格非空单元格占比下限")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（冒烟用；0 = 全部）")
    args = ap.parse_args()

    base = Path(args.root).resolve() if args.root else _default_root()
    out_base = Path(args.out).resolve() if args.out else base / "parsed"
    if not base.is_dir():
        print(f"源目录不存在: {base}", file=sys.stderr)
        return 2

    pdfs = sorted(p for p in base.rglob("*.pdf") if "parsed" not in p.parts)
    if args.limit:
        pdfs = pdfs[: args.limit]
    total = len(pdfs)
    print(f"源: {base}\n输出: {out_base}\nPDF: {total}  workers={args.workers} "
          f"min_size={args.min_size} min_table_ratio={args.min_table_ratio}", flush=True)
    if not total:
        return 0

    out_base.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results: list[dict | None] = [None] * total
    done = errors = skipped = 0

    # 中断（Ctrl-C / 被 kill）也落 metadata——batch_parse_v3 只在末尾写，
    # 上次中断导致 batch_metadata.json 永久缺失，此处用 finally 兜住。
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(base, out_base, args.min_size, args.min_table_ratio),
        ) as ex:
            futures = {ex.submit(parse_single, (i, str(p))): i for i, p in enumerate(pdfs)}
            for fut in as_completed(futures):
                r = fut.result()
                results[r["index"]] = r
                done += 1
                if r["status"] == "error":
                    errors += 1
                elif r["status"] == "skipped":
                    skipped += 1
                if done % 200 == 0 or done == total:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / rate if rate > 0 else 0.0
                    print(f"[{done}/{total}] {elapsed:.0f}s, ETA {eta:.0f}s, "
                          f"errors={errors}, skipped={skipped}", flush=True)
    finally:
        meta_file = _write_metadata(out_base, [r for r in results if r])
        ok = sum(1 for r in results if r and r["status"] == "ok")
        sk = sum(1 for r in results if r and r["status"] == "skipped")
        er = sum(1 for r in results if r and r["status"] == "error")
        print(f"\n{'=' * 60}\nBATCH SUMMARY\n{'=' * 60}", flush=True)
        print(f"Total: {total}  OK: {ok}  Skipped: {sk}  Errors: {er}", flush=True)
        print(f"Total time: {time.time() - t_start:.0f}s", flush=True)
        for r in results:
            if r and r["status"] == "error":
                print(f"  ERR {r['rel_path']}: {r.get('error', '')[:120]}", flush=True)
        print(f"Metadata: {meta_file}  (records={sum(1 for r in results if r)})", flush=True)

    return 1 if errors and errors == total else 0


if __name__ == "__main__":
    sys.exit(main())

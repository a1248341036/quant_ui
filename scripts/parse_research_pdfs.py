#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研报 PDF 解析 v2：PyMuPDF 正文 + pdfplumber 表格 **+ 图片提取 + VLM 转写**。

相对 v1（`parsed/`，只做正文与表格）的增量：

- **小字层（2026-09-29 实测新增，零 API 成本）**：v1 用 ``size >= 9pt`` 过滤，抽样
  40 篇实测**丢掉近一半文本**（<9pt 去噪后仍有 1,052,389 字符 ≈ 正文的 95%），
  其中包含图表参数表、图注与**公式**（华泰《海量技术因子》101 个 Alpha 公式全在
  8pt 层，v1 一个没留）。v2 改为**分层输出**：正文层（≥ ``--min-size``）+ 小字层
  （去页眉页脚/免责声明/页码噪声）。
- **图片提取**：PyMuPDF 逐页取图，按 md5 去重，落盘到
  ``<out>/media/<报告名>/page{N}_img{M}.png``；跳过装饰性小图。
- **VLM 转写**：每张有效图调用本地 **EasyCLI（8317）Qwen3.8-27B** 生成结构化 JSON
  （类型 / 标题 / 公式转写 / 维度 / 关键数据 / 图义 / 置信度），渲染为自包含
  ``<!-- chart:N -->`` 块，供向量召回与「照机制/照公式复现」使用。
- **矢量图页渲染（``--page-render``）**：金工研报的走势图/公式表多为**矢量绘制**，
  ``get_drawings()`` 上千对象而 ``get_images()`` 常常为 0 —— 因此对"绘制对象多且有
  ``图表N：`` 标题"的页整页渲染成 PNG 再交 VLM 转写。
- **元数据头**：文首写机构 / 日期 / 标题（由路径推断），便于 L0 摘要抽取。

纪律（沿用既有约定，不得放宽）：

1. **精确数值不从图表像素反推**：VLM 只做"文字转写 + 图义描述"，输出里标注
   ``来源=VLM读图`` 与 ``置信度``；正文/表格仍走文本层（来源=正文，高置信）。
2. **保真优先**：图里没有公式就留空（``formula_text: null``），**禁止按上下文补全**。
3. v1 产物 ``parsed/`` 不动，v2 默认落 ``parsed_v2/``，两者可并存对比与回滚。

用法::

    # 冒烟：3 篇样本，含图片 + VLM 转写
    python scripts/parse_research_pdfs.py --limit 3 --images --vlm --workers 2

    # 只要正文+表格（v1 等价），输出到 parsed/
    python scripts/parse_research_pdfs.py --no-images --out <root>/parsed

    # 指定单篇（调试）
    python scripts/parse_research_pdfs.py --only "华泰多因子系列之十一"

依赖：``pymupdf`` + ``pdfplumber`` + ``Pillow``（均已在 venv）；
VLM 依赖本机 EasyCLI ``http://127.0.0.1:8317`` 可用。
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RE_SINGLE_CJK = re.compile(r"^[\u4e00-\u9fff]$")
RE_CAPTION = re.compile(r"^\s*((?:图表|图|表)\s*\d+\s*[:：].{0,120})$")
RE_FIG_TITLE = re.compile(r"(?:图表|图|表)\s*\d+\s*[:：]")  # 整页文本用（非锚定）
# 小字层噪声：页眉页脚、免责声明、页码、公众号/评级说明等（不含公式与参数表）
RE_SMALL_NOISE = re.compile(
    r"谨请参阅|免责声明|证券研究报告|请务必阅读|分析师声明|投资评级说明|法律声明|"
    r"风险提示及免责|特别声明|微信公众号|扫码关注|^\s*\d{1,3}\s*$|"
    r"金工研究/|金融工程/|金融工程研究|证券研究报告|未经许可|版权所有"
)
RE_INST = re.compile(r"([\u4e00-\u9fff]{2,6}(?:证券|基金|期货|银行|资管))")
RE_DATE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")

# VLM 默认走 EasyCLI（CodeFree 通道的 Qwen3.8-27B）。可用 CLI 覆盖。
VLM_DEFAULT_BASE = "http://127.0.0.1:8317/v1/chat/completions"
VLM_DEFAULT_KEY = "123456"
VLM_DEFAULT_MODEL = "Qwen3.8-27B"
VLM_PROMPT = (
    "你是研报图表结构化助手。请只输出 JSON（不要解释、不要 markdown 代码块），字段：\n"
    '{"type": "公式|图表|表格|示意图|装饰|其他",\n'
    ' "title": "该图的标题或最接近的小标题（没有就空串）",\n'
    ' "formula_text": "若图中有因子/指标公式，逐字符转写为单行文本；没有公式或无法辨认则 null",\n'
    ' "axes_or_dims": "坐标轴/维度/分组说明（没有就空串）",\n'
    ' "key_data": "图中可读的关键结论或数值（只写图上确实可读的，不要推算）",\n'
    ' "meaning": "这张图在讲什么、对因子研究的用处（一句话）",\n'
    ' "confidence": "high|medium|low"}\n'
    "硬约束：① 不得根据上下文补全图中不存在的内容；② 不得从曲线像素反推数值；"
    "③ 公式转写必须逐字符忠实，模糊不清就置 null 并把 confidence 降为 low。"
)

# worker 进程配置。Windows 下 ProcessPoolExecutor 用 spawn：子进程重新 import 本模块，
# 父进程运行时写入的全局量**不会**继承，故经 initializer 显式下发。
_CFG: dict = {}


def _init_worker(base: Path, out_base: Path, cfg: dict) -> None:
    _CFG.update(base=base, out_base=out_base, **cfg)


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


# ─────────────────────────── 正文 / 表格（v1 口径不变） ───────────────────────────

def extract_text_layers(pdf_path: str, min_size: float, small_min: float) -> tuple[list[str], list[str]]:
    """按页返回 (正文层 pages, 小字层 pages)。

    正文层：``size >= min_size``（v1 口径，保持可比）。
    小字层：``small_min <= size < min_size``——图表参数表、图注与公式大量落在这一层
    （实测占全文近一半；v1 整层丢弃是"研报没有公式"假象的根因）。
    两层都保留**逐页**结构，供渲染时加 ``<!-- Page N -->`` 锚点。
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    body_pages: list[str] = []
    small_pages: list[str] = []
    try:
        for page in doc:
            d = page.get_text("dict")
            body_lines, small_lines = [], []
            for block in d["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    raw = "".join(sp["text"] for sp in line["spans"]).strip()
                    if not raw or RE_SINGLE_CJK.match(raw):
                        continue
                    size = max((sp["size"] for sp in line["spans"]), default=0.0)
                    if size >= min_size:
                        body_lines.append(raw)
                    elif size >= small_min:
                        small_lines.append(raw)
            body_pages.append("\n".join(body_lines))
            small_pages.append("\n".join(small_lines))
        return body_pages, small_pages
    finally:
        doc.close()


def _clean_layer(pages: list[str], *, dup_threshold: int = 3, drop_short: bool = True) -> list[str]:
    """层内清洗：剔除版式噪声，保留信息行。

    规则（按有效性排序，实测小字层近一半是噪声）：
    1. **跨页重复行只留首次**：出现 >= ``dup_threshold`` 次且长度 < 45 的行——
       页眉/页脚/免责声明/封面"往期报告列表"；这是最大噪声源（实测 46%）。
       含算符/等号/公式特征的行**不参与**去重（避免误删表格重复表头里的公式行）。
    2. **超短碎片**：<= 3 字符且不含字母/数字/算符的行（坐标轴单字刻度）删除。
    3. 小字层额外套用 ``RE_SMALL_NOISE`` 关键词黑名单。
    """
    from collections import Counter

    FORM_HINT = re.compile(r"[=+*/()<>]|[A-Za-z]{3,}|→|[0-9]\.[0-9]")
    freq: Counter = Counter()
    for page in pages:
        for line in page.splitlines():
            line = line.strip()
            if line:
                freq[line] += 1

    out_pages: list[str] = []
    kept_once: set[str] = set()
    for page in pages:
        kept: list[str] = []
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                continue
            if drop_short and len(line) <= 3 and not FORM_HINT.search(line):
                continue
            if RE_SMALL_NOISE.search(line):
                continue
            repeated = freq[line] >= dup_threshold and len(line) < 45
            if repeated and not FORM_HINT.search(line):
                if line in kept_once:
                    continue
                kept_once.add(line)
            kept.append(line)
        out_pages.append("\n".join(kept))
    return out_pages


def _render_paged(pages: list[str]) -> str:
    """按页输出并加 ``<!-- Page N -->`` 锚点（保留阅读顺序，便于引用与区域归位）。"""
    blocks = []
    for i, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        blocks.append(f"<!-- Page {i} -->\n{page}")
    return "\n\n".join(blocks)


def render_pages(pdf_path: str, media_dir: Path, cfg: dict) -> list[dict]:
    """对"矢量绘制多 + 有图表标题"的页整页渲染，供 VLM 读图（走势图/公式表多为矢量）。"""
    import pymupdf

    dpi = int(cfg.get("page_render_dpi", 150))
    min_draw = int(cfg.get("page_render_min_drawings", 60))
    max_pages = int(cfg.get("max_page_renders", 20))
    media_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    out: list[dict] = []
    try:
        for pno, page in enumerate(doc, start=1):
            if len(out) >= max_pages:
                break
            txt = page.get_text("text")
            if not RE_FIG_TITLE.search(txt):
                continue
            try:
                draws = len(page.get_drawings())
            except Exception:  # noqa: BLE001
                draws = 0
            if draws < min_draw:
                continue
            m_cap = RE_CAPTION.search(txt)
            m_fig = RE_FIG_TITLE.search(txt)
            cap_txt = (m_cap.group(1) if m_cap else (m_fig.group(0) if m_fig else "")).strip()
            name = f"page{pno}_render.png"
            try:
                page.get_pixmap(dpi=dpi).save(media_dir / name)
            except Exception:  # noqa: BLE001
                continue
            out.append({"name": name, "page": pno, "kind": "page_render",
                        "caption": cap_txt, "width": 0, "height": 0,
                        "bytes": (media_dir / name).stat().st_size})
    finally:
        doc.close()
    return out


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


# ─────────────────────────────── 图片提取（v2 新增） ───────────────────────────────

def _page_captions(page_text: str) -> list[str]:
    out = []
    for line in page_text.splitlines():
        m = RE_CAPTION.match(line)
        if m:
            out.append(m.group(1).strip())
    return out


def extract_images(pdf_path: str, media_dir: Path, cfg: dict) -> list[dict]:
    """逐页提取图片，md5 去重，落盘 PNG；返回元数据列表（含所在页与图注）。"""
    import pymupdf
    from PIL import Image

    min_px = int(cfg.get("min_image_px", 120))
    min_bytes = int(cfg.get("min_image_bytes", 6000))
    max_images = int(cfg.get("max_images", 40))
    media_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    seen: set[str] = set()
    out: list[dict] = []
    try:
        for pno, page in enumerate(doc, start=1):
            if len(out) >= max_images:
                break
            captions = _page_captions(page.get_text("text"))
            img_no = 0
            for info in page.get_images(full=True):
                if len(out) >= max_images:
                    break
                xref = info[0]
                try:
                    raw = doc.extract_image(xref)
                except Exception:  # noqa: BLE001
                    continue
                data = raw.get("image") or b""
                if len(data) < min_bytes:
                    continue
                digest = hashlib.md5(data).hexdigest()
                if digest in seen:
                    continue
                try:
                    im = Image.open(io.BytesIO(data))
                    w, h = im.size
                except Exception:  # noqa: BLE001
                    continue
                if w < min_px or h < min_px:
                    continue
                seen.add(digest)
                img_no += 1
                name = f"page{pno}_img{img_no}.png"
                try:
                    im.convert("RGB").save(media_dir / name, format="PNG")
                except Exception:  # noqa: BLE001
                    continue
                caption = captions[min(img_no - 1, len(captions) - 1)] if captions else ""
                out.append(
                    {
                        "name": name,
                        "page": pno,
                        "width": w,
                        "height": h,
                        "bytes": len(data),
                        "md5": digest,
                        "caption": caption,
                    }
                )
    finally:
        doc.close()
    return out


# ────────────────────────────── VLM 转写（v2 新增） ──────────────────────────────

def _encode_image(path: Path, max_side: int = 1280) -> tuple[str, str]:
    """读图 → 等比缩到 max_side → JPEG(base64)。返回 (data_uri, mime)。"""
    from PIL import Image

    im = Image.open(path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", "image/jpeg"


def _parse_vlm_json(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except Exception:  # noqa: BLE001
                return None
    return None


def vlm_transcribe(img_path: Path, cfg: dict, caption: str = "") -> dict:
    """调用 EasyCLI Qwen3.8-27B 转写单张图；失败返回 confidence=low 的兜底结构。"""
    base = cfg.get("vlm_base") or VLM_DEFAULT_BASE
    key = cfg.get("vlm_key") or VLM_DEFAULT_KEY
    model = cfg.get("vlm_model") or VLM_DEFAULT_MODEL
    timeout = float(cfg.get("vlm_timeout", 180))
    retries = int(cfg.get("vlm_retries", 2))
    prompt = VLM_PROMPT + (f"\n已知图注（可能为空，仅供消歧）：{caption}" if caption else "")

    last_err = ""
    for attempt in range(retries + 1):
        try:
            data_uri, _ = _encode_image(img_path)
            body = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                "max_tokens": 900,
                "temperature": 0,
            }
            req = urllib.request.Request(
                base,
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = _parse_vlm_json(content)
            if parsed:
                parsed.setdefault("confidence", "low")
                parsed["_raw"] = content[:600]
                return parsed
            last_err = "unparsable_response"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
    return {"type": "其他", "title": "", "formula_text": None, "axes_or_dims": "",
            "key_data": "", "meaning": "", "confidence": "low", "_error": last_err}


# ────────────────────────────── 渲染与元数据（v2 新增） ──────────────────────────────

def _report_meta(rel_path: str) -> dict:
    name = Path(rel_path).stem
    m = RE_DATE.search(name)
    date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    inst = RE_INST.search(name)
    title = name
    parts = re.split(r"[_：:]", name)
    if len(parts) >= 3:
        title = parts[-1] if len(parts[-1]) > 4 else name
    return {"institution": inst.group(1) if inst else "", "date": date, "title": title, "file": rel_path}


def render_chart_blocks(images: list[dict], transcriptions: dict[str, dict]) -> str:
    blocks = []
    for i, img in enumerate(images, start=1):
        tr = transcriptions.get(img["name"], {})
        if tr.get("type") == "装饰" or tr.get("_skip"):
            continue
        title = (tr.get("title") or img.get("caption") or "").strip()
        lines = [
            f"<!-- chart:{i} -->",
            f"#### [图{i}] {title or '(无标题)'}",
            f"- 类型：{tr.get('type') or '其他'}　页：p{img['page']}　"
            f"来源：VLM读图／`media/{img['name']}`　置信度：{tr.get('confidence') or 'low'}",
        ]
        formula = tr.get("formula_text")
        if formula:
            lines.append("- 公式（图像转写，供复现参考；**不得据此反推精确数值**）：")
            lines.append("  ```")
            lines.append(f"  {str(formula).strip()}")
            lines.append("  ```")
        if tr.get("axes_or_dims"):
            lines.append(f"- 维度/坐标轴：{tr['axes_or_dims']}")
        if tr.get("key_data"):
            lines.append(f"- 关键内容：{tr['key_data']}")
        if tr.get("meaning"):
            lines.append(f"- 图义：{tr['meaning']}")
        if tr.get("_error"):
            lines.append(f"- ⚠ 转写失败：{tr['_error']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ─────────────────────────────────── 单篇处理 ───────────────────────────────────

def parse_single(args: tuple[int, str]) -> dict:
    idx, pdf_path_str = args
    base: Path = _CFG["base"]
    out_base: Path = _CFG["out_base"]

    pdf_path = Path(pdf_path_str)
    rel = str(pdf_path.relative_to(base))
    out_file = out_base / Path(rel).with_suffix(".md")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    result = {"index": idx, "rel_path": rel, "name": pdf_path.stem, "out": str(out_file)}
    if out_file.exists():
        result["status"] = "skipped"
        return result

    with_images = bool(_CFG.get("images"))
    with_vlm = bool(_CFG.get("vlm"))
    try:
        t0 = time.time()
        body_pages, small_pages = extract_text_layers(
            str(pdf_path), float(_CFG.get("min_size", 9.0)), float(_CFG.get("small_font_min", 5.0))
        )
        body_pages = _clean_layer(body_pages, drop_short=False)
        small_pages = _clean_layer(small_pages)
        text = _render_paged(body_pages)
        small_text = _render_paged(small_pages)
        tables = extract_tables_pdfplumber(str(pdf_path), float(_CFG.get("min_table_ratio", 0.3)))

        images: list[dict] = []
        transcriptions: dict[str, dict] = {}
        vlm_ok = vlm_fail = 0
        media_dir = out_base / "media" / pdf_path.stem
        if with_images:
            images = extract_images(str(pdf_path), media_dir, _CFG)
        if _CFG.get("page_render"):
            images = images + render_pages(str(pdf_path), media_dir, _CFG)
        if with_images and with_vlm:
            for img in images:
                tr = vlm_transcribe(media_dir / img["name"], _CFG, caption=img.get("caption", ""))
                transcriptions[img["name"]] = tr
                if tr.get("_error"):
                    vlm_fail += 1
                else:
                    vlm_ok += 1

        meta = _report_meta(rel)
        parts = [f"<!-- meta: institution={meta['institution']} date={meta['date']} -->",
                 f"# {meta['title']}", "",
                 f"- 机构：{meta['institution'] or '未识别'}　日期：{meta['date'] or '未识别'}"
                 f"　源文件：`{rel}`", "", text]
        if small_text:
            parts += ["", "## 小字层（图表文字／公式／参数表，字号 < %.0fpt）" % float(_CFG.get("min_size", 9.0)),
                      "", small_text]
        if tables:
            parts += ["", "## 表格数据", "", tables]
        if images:
            charts = render_chart_blocks(images, transcriptions)
            if charts:
                parts += ["", "## 图表转写", "", charts]
        md = "\n".join(parts)
        out_file.write_text(md, encoding="utf-8")

        result.update(
            status="ok",
            chars=len(md),
            text_chars=len(text),
            small_chars=len(small_text),
            table_chars=len(tables),
            images=len(images),
            page_renders=sum(1 for i in images if i.get("kind") == "page_render"),
            charts=sum(1 for k, v in transcriptions.items() if v.get("type") != "装饰"),
            vlm_ok=vlm_ok,
            vlm_fail=vlm_fail,
            formulas=sum(1 for v in transcriptions.values() if v.get("formula_text")),
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
    ap = argparse.ArgumentParser(description="研报 PDF 解析 v2（正文 + 表格 + 图片提取 + Qwen3.8-27B 转写）")
    ap.add_argument("--root", default=None, help="源根目录（默认：主工作区 data/research_reports）")
    ap.add_argument("--out", default=None, help="输出目录（默认：有图片→<root>/parsed_v2，无图片→<root>/parsed）")
    ap.add_argument("--workers", type=int, default=2, help="并行篇数（开 VLM 时建议 ≤2，避免打爆中继）")
    ap.add_argument("--min-size", type=float, default=9.0, help="正文 span 最小字号（pt）")
    ap.add_argument("--small-font-min", type=float, default=5.0,
                    help="小字层字号下限（pt）；< min-size 且 ≥ 此值的行进小字层（公式/参数表多在此层）")
    ap.add_argument("--min-table-ratio", type=float, default=0.3, help="表格非空单元格占比下限")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（冒烟用；0 = 全部）")
    ap.add_argument("--only", default="", help="只处理路径含该子串的报告（调试单篇用）")
    img = ap.add_mutually_exclusive_group()
    img.add_argument("--images", dest="images", action="store_true", default=True, help="提取图片（默认开）")
    img.add_argument("--no-images", dest="images", action="store_false", help="不提取图片（v1 等价）")
    vlm = ap.add_mutually_exclusive_group()
    vlm.add_argument("--vlm", dest="vlm", action="store_true", default=True, help="VLM 转写图片（默认开，需 --images）")
    vlm.add_argument("--no-vlm", dest="vlm", action="store_false", help="只存图不转写")
    ap.add_argument("--min-image-px", type=int, default=120, help="图片最小边长（像素），低于此判为装饰")
    ap.add_argument("--min-image-bytes", type=int, default=6000, help="图片最小字节数")
    ap.add_argument("--max-images", type=int, default=40, help="单篇最多提取图片数")
    ap.add_argument("--vlm-base", default=VLM_DEFAULT_BASE, help="VLM 接口（默认 EasyCLI 8317）")
    ap.add_argument("--vlm-key", default=VLM_DEFAULT_KEY)
    ap.add_argument("--vlm-model", default=VLM_DEFAULT_MODEL)
    ap.add_argument("--page-render", dest="page_render", action="store_true", default=False,
                    help="对矢量绘制多且有图表标题的页整页渲染后交 VLM（走势图/公式表多为矢量）")
    ap.add_argument("--page-render-dpi", type=int, default=150)
    ap.add_argument("--page-render-min-drawings", type=int, default=60)
    ap.add_argument("--max-page-renders", type=int, default=20)
    ap.add_argument("--vlm-timeout", type=float, default=180.0)
    ap.add_argument("--vlm-retries", type=int, default=2)
    args = ap.parse_args()

    base = Path(args.root).resolve() if args.root else _default_root()
    default_name = "parsed_v2" if args.images else "parsed"
    out_base = Path(args.out).resolve() if args.out else base / default_name
    if not base.is_dir():
        print(f"源目录不存在: {base}", file=sys.stderr)
        return 2

    pdfs = sorted(p for p in base.rglob("*.pdf") if "parsed" not in p.parts)
    if args.only:
        pdfs = [p for p in pdfs if args.only in p.name]
    if args.limit:
        pdfs = pdfs[: args.limit]
    total = len(pdfs)
    print(f"源: {base}\n输出: {out_base}\nPDF: {total}  workers={args.workers} "
          f"images={args.images} vlm={args.vlm and args.images} model={args.vlm_model}", flush=True)
    if not total:
        return 0

    out_base.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        min_size=args.min_size,
        small_font_min=args.small_font_min,
        page_render=args.page_render,
        page_render_dpi=args.page_render_dpi,
        page_render_min_drawings=args.page_render_min_drawings,
        max_page_renders=args.max_page_renders,
        min_table_ratio=args.min_table_ratio,
        images=args.images,
        vlm=args.vlm and args.images,
        min_image_px=args.min_image_px,
        min_image_bytes=args.min_image_bytes,
        max_images=args.max_images,
        vlm_base=args.vlm_base,
        vlm_key=args.vlm_key,
        vlm_model=args.vlm_model,
        vlm_timeout=args.vlm_timeout,
        vlm_retries=args.vlm_retries,
    )
    t_start = time.time()
    results: list[dict | None] = [None] * total
    done = errors = skipped = 0
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(base, out_base, cfg),
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
                if done % 5 == 0 or done == total:
                    elapsed = time.time() - t_start
                    print(f"[{done}/{total}] {elapsed:.0f}s errors={errors} skipped={skipped}", flush=True)
    finally:
        meta_file = _write_metadata(out_base, [r for r in results if r])
        oks = [r for r in results if r and r["status"] == "ok"]
        print(f"\n{'=' * 60}\nBATCH SUMMARY\n{'=' * 60}", flush=True)
        print(f"Total: {total}  OK: {len(oks)}  Skipped: {sum(1 for r in results if r and r['status'] == 'skipped')}"
              f"  Errors: {sum(1 for r in results if r and r['status'] == 'error')}", flush=True)
        if oks:
            imgs = sum(r.get("images", 0) for r in oks)
            charts = sum(r.get("charts", 0) for r in oks)
            vok = sum(r.get("vlm_ok", 0) for r in oks)
            vfail = sum(r.get("vlm_fail", 0) for r in oks)
            forms = sum(r.get("formulas", 0) for r in oks)
            small = sum(r.get("small_chars", 0) for r in oks)
            big = sum(r.get("text_chars", 0) for r in oks)
            pren = sum(r.get("page_renders", 0) for r in oks)
            print(f"正文: {big:,} 字符  小字层: {small:,} 字符（+{100*small/max(1,big):.0f}%）"
                  f"  图片: {imgs}（页渲染 {pren}）  图表块: {charts}  转写成功: {vok}  失败: {vfail}  含公式: {forms}",
                  flush=True)
        print(f"Total time: {time.time() - t_start:.0f}s", flush=True)
        for r in results:
            if r and r["status"] == "error":
                print(f"  ERR {r['rel_path']}: {r.get('error', '')[:120]}", flush=True)
        print(f"Metadata: {meta_file}", flush=True)

    return 1 if errors and errors == total else 0


if __name__ == "__main__":
    sys.exit(main())

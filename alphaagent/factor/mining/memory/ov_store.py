"""OpenViking 长期记忆接入层（冷路径跨 session 策略记忆）。

定位：与 SQLite research_memory.db 互补——SQLite 做热路径表达式级精确去重（毫秒级、
结构指纹匹配），OpenViking 做冷路径语义模糊检索（自然语言策略问题，如"vwap 族为什么
不该再挖"、"1d label 配 weekly 调仓换手率必爆"）。

隔离：OpenViking 无 per-agent ACL，所有调用硬编码 SCOPE="viking://resources/alphaagent/"，
不暴露 target_uri 参数，防止误用泄漏到 viking://user/default/memories/（个人记忆）或
其他项目资源。所有方法失败静默——OpenViking 不可用时挖掘照常跑（纯 SQLite 降级）。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

log = logging.getLogger(__name__)

# 专属库 scope：所有 OpenViking 调用的硬编码边界，禁止外部传入
SCOPE = "viking://resources/alphaagent/"
REPORT_SCOPE = "viking://resources/research_reports/"


def _research_reports_roots() -> list:
    """定位 data/research_reports（兼容独立 worktree：用 git common dir 反查主仓库）。"""
    import subprocess
    from pathlib import Path

    roots = []
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            roots.append(Path(out).parent / "data" / "research_reports")
    except Exception:  # noqa: BLE001
        pass
    roots.append(Path(__file__).resolve().parents[4] / "data" / "research_reports")
    roots.append(Path(r"D:\Quant\quant_ui\data\research_reports"))
    return roots


def _report_local_corpus():
    """本地研报语料目录。

    ``ALPHA_REPORT_CORPUS`` 可取值 ``parsed``（v1）或 ``parsed_mineru``
    （MinerU 4.0.9 重抽版：页锚点 + 真表格 + 公式逐字符），也可给绝对路径。
    未显式指定时**优先 MinerU 重抽版**：实测同题下 ``parsed_mineru`` 命中的是
    真因子机制文献（东吴「估值异常 EPA 因子」、海通「动量 beta 构建」），
    v1 ``parsed`` 只有文件头/目次级内容；缺失时回落 v1，保证不空转。
    """
    import os
    from pathlib import Path

    raw = (os.environ.get("ALPHA_REPORT_CORPUS") or "").strip()
    if raw:
        cand = Path(raw)
        if cand.is_absolute() and cand.is_dir():
            return cand
        for base in _research_reports_roots():
            d = base / raw
            if d.is_dir():
                return d
        return None
    for name in ("parsed_mineru", "parsed"):
        for base in _research_reports_roots():
            d = base / name
            if d.is_dir():
                return d
    return None


def _report_rag_source() -> str:
    """研报 RAG 数据源：``ov``（OpenViking，默认）| ``local``（本地语料目录直检）。"""
    import os

    return (os.environ.get("ALPHA_REPORT_RAG_SOURCE") or "ov").strip().lower()



class OVStore:
    """OpenViking 长期记忆接入层。所有调用硬编码 viking://resources/alphaagent/ scope。"""

    def __init__(self, endpoint: str = "http://127.0.0.1:1933", *, inject_max_chars: int = 2400) -> None:
        self.endpoint = endpoint
        self.inject_max_chars = inject_max_chars
        self._client = None

    # ── 懒初始化：无 OpenViking 环境返回 None，失败静默 ──
    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openviking_sdk.client import SyncHTTPClient

            client = SyncHTTPClient(url=self.endpoint, account="default", user="default")
            client.initialize()
            self._client = client
            return client
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenViking 初始化失败（长期记忆降级为纯 SQLite）: %s", exc)
            return None

    # ── run 启动：检索跨 session 策略记忆，返回注入用文本块 ──
    def retrieve_lessons(
        self,
        focus_facets: Iterable[str] | None = None,
        research_mode: str = "technical",
        *,
        limit: int = 5,
        max_chars: int | None = None,
    ) -> str:
        """检索跨 session 策略记忆，返回注入 system prompt 的文本块。失败/无 OpenViking 返回空串。

        focus_facets 参与 query 构造（如"价量面 量能面 technical 饱和度 教训"），
        数字类事实（如具体族统计）由调用方实时查 SQLite 生成，不存 OpenViking 快照。
        """
        client = self._get_client()
        if client is None:
            return ""
        facets = tuple(focus_facets or ())
        query = " ".join([*facets, research_mode or "technical", "饱和度 教训 死路 换手率"])
        try:
            results = client.search(query=query, target_uri=SCOPE, limit=limit)
            hits = self._flatten_hits(results, limit)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenViking 检索失败: %s", exc)
            return ""
        if not hits:
            return ""
        budget = max_chars or self.inject_max_chars
        lines: list[str] = ["### 跨 session 策略记忆（来自 OpenViking 长期记忆库）", ""]
        used = 0
        for h in hits:
            uri = str(h.get("uri") or "")
            abstract = str(h.get("abstract") or "")
            if not abstract:
                continue
            if len(abstract) > 220:  # 单条上限，防长文档撑爆注入预算
                abstract = abstract[:220] + "…"
            block = f"- **{self._short_uri(uri)}**：{abstract}"
            if used + len(block) > budget:
                break
            lines.append(block)
            used += len(block)
        if len(lines) <= 2:
            return ""
        return "\n".join(lines)

    # ── R3 方案：研报原文 RAG 检索 ──
    def retrieve_report_knowledge(
        self,
        focus_facets: Iterable[str] | None = None,
        research_mode: str = "technical",
        query_text: str | None = None,
        *,
        limit: int = 4,
        max_chars: int | None = None,
        snippet_chars: int = 700,
    ) -> str:
        """检索研报原文文本段落（R3 方案，支持 R4 课题动态对齐）。

        优先走 OpenViking REPORT_SCOPE，若不可用或未命中则自动平滑降级到
        本地 parsed 研报库的高性能相关性检索（带缓存与超时控制，失败静默）。
        """
        budget = max_chars or 1600
        facets = tuple(focus_facets or ())

        if query_text:
            # 课题对齐模式：以课题主题和核心假说为主，辅以数据面
            import re
            cleaned_words = [w for w in re.split(r"[^\w\u4e00-\u9fa5]+", query_text) if len(w) >= 2][:10]
            query_terms = cleaned_words + list(facets) + ["选股", "因子"]
        else:
            # 泛化模式：按数据面与研究模式检索
            query_terms = list(facets) + [research_mode or "technical", "多因子 选股 机制 异象"]

        query = " ".join(query_terms)

        client = self._get_client() if _report_rag_source() != "local" else None
        hits = []
        if client is not None:
            try:
                # 多取候选：下面会剔除研报 PDF 的图片语义索引条目（pageN_imgM.png），
                # 若只按 limit 取，过滤后可能不足额甚至归零、无谓触发本地降级。
                results = client.search(query=query, target_uri=REPORT_SCOPE, limit=max(limit * 3, 8))
                hits = self._flatten_hits(results, limit, scope=REPORT_SCOPE)
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenViking 研报检索异常，降级到本地检索: %s", exc)

        # 本地降级检索：OV 命中为空，或命中的全是行情复盘类文档（对机制/公式无价值）。
        if hits:
            kept = [h for h in hits if not _is_recap_doc(str(h.get("title") or h.get("uri") or ""))]
            hits = kept if kept else []
        if not hits:
            hits = _local_report_search(query_terms, limit=limit, corpus_dir=_report_local_corpus())

        if not hits:
            return ""

        lines: list[str] = [
            "### 研报原文先验【外部研报·非本平台实测】",
            "",
            "以下为从券商金工研报库中实时检索出的相关文献摘录（供机制与算子设计参考）：",
            "",
        ]
        used = 0
        for h in hits:
            title = h.get("title") or h.get("uri") or "研报摘录"
            snippet = h.get("snippet") or h.get("abstract") or ""
            if not snippet:
                continue
            if len(snippet) > snippet_chars:
                snippet = snippet[:snippet_chars].rsplit("\n", 1)[0] + "…"
            block = f"- **《{title}》**：\n  > {snippet.strip()}"
            if used + len(block) > budget:
                break
            lines.append(block)
            lines.append("")
            used += len(block)

        if len(lines) <= 3:
            return ""
        return "\n".join(lines).strip()

    def _flatten_hits(self, results: dict[str, Any], limit: int, scope: str = SCOPE) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for cat in ("memories", "resources", "skills"):
            for item in results.get(cat) or []:
                if not isinstance(item, dict):
                    continue
                uri = str(item.get("uri") or "")
                if not uri.startswith(scope):  # 防御：只接受指定 scope 内命中
                    continue
                # 排除隐藏文件（OpenViking 自动生成的 .overview.md 等目录概览）
                if "/." in uri or uri.rstrip("/").rsplit("/", 1)[-1].startswith("."):
                    continue
                # 排除研报 PDF 的“页面图片语义索引”条目（pageN_imgM.png 的图注描述）：
                # 它们检索得分不低，但对因子机制设计没有价值——实测 2026-09-28 注入块
                # 几乎被图注占满（"Image Description ..."），正文段落反而被挤掉。
                leaf = uri.rstrip("/").rsplit("/", 1)[-1].lower()
                if leaf.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")):
                    continue
                if "_img" in leaf or "_page" in leaf:
                    continue
                hits.append(item)
        hits.sort(key=lambda h: -float(h.get("score") or 0.0))
        return hits[:limit]

    @staticmethod
    def _short_uri(uri: str) -> str:
        return uri[len(SCOPE):] if uri.startswith(SCOPE) else uri

    @staticmethod
    def _safe_name(name: str) -> str:
        import re

        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(name or "model")).strip("-") or "model"

    # ── run 结束：写回摘要到独立文件（文件名含 run_id，多 run 并行无冲突） ──
    def write_run_summary(self, run_id: str, model_name: str, summary: dict[str, Any] | None) -> bool:
        """run 结束时写回摘要到 run_summaries/<date>-<model>-<run_id>.md。失败静默。

        summary 由调用方传 run 的 summary dict（含 candidate_funnel / failure_counts /
        submitted_factors / unsubmitted_promising 等）。独立文件 + 定期 compaction 合并到
        run_summaries.md 主文件（见 spec §3.4/§3.5.2）。
        """
        client = self._get_client()
        if client is None:
            return False
        try:
            content = self._build_summary_markdown(run_id, model_name, summary or {})
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            uri = f"{SCOPE}run_summaries/{date_str}-{self._safe_name(model_name)}-{run_id}.md"
            client.write(uri=uri, content=content, mode="create")
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenViking run 摘要写回失败: %s", exc)
            return False

    def _build_summary_markdown(self, run_id: str, model_name: str, summary: dict[str, Any]) -> str:
        funnel = summary.get("candidate_funnel") or {}
        usage = summary.get("usage") or {}
        lines = [
            f"# Run 摘要 {run_id}（{model_name}）",
            "",
            "## 关键统计",
        ]
        for key, label in (
            ("unique_train_evaluated", "train 去重评估数"),
            ("unique_val_evaluated", "val 去重评估数"),
            ("unique_auto_val_verified", "系统自动 val 验证数"),
            ("unique_val_effective", "有效 val 覆盖（显式∪自动）"),
            ("candidate_stored", "候选入库"),
            ("production_stored", "正式入库"),
            ("train_to_val_rate", "train→val 保留比"),
            ("train_to_val_rate_incl_auto", "train→val 保留比（含自动 val）"),
            ("val_to_production_rate", "val→入库率"),
        ):
            v = funnel.get(key)
            if v is not None:
                lines.append(f"- {label}：{v}")
        if failure_counts := summary.get("failure_counts"):
            lines.append("")
            lines.append("## 失败分布")
            for code, n in sorted(failure_counts.items(), key=lambda kv: -kv[1]):
                if code in ("ok", "passed", "promising"):
                    continue
                lines.append(f"- {code}：{n}")
        if submitted := summary.get("submitted_factors"):
            lines.append("")
            lines.append("## 提交因子")
            for s in submitted[:10]:
                if isinstance(s, dict):
                    lines.append(f"- {s.get('factor_name') or s.get('name') or s.get('factor_uid')}")
        if usage:
            lines.append("")
            lines.append("## 用量")
            lines.append(f"- calls：{usage.get('calls')}")
        return "\n".join(lines) + "\n"

    # ── dead_families.md 维护：read-modify-write 追加新饱和族 ──
    def update_dead_families(self, families: Iterable[str], *, details: dict[str, str] | None = None) -> bool:
        """追加新饱和族到 dead_families.md。已存在的族跳过。失败静默。

        结构化主文件由本方法维护，不随 run 摘要滚动（spec §3.4）。read-modify-write，
        并发冲突时后写者覆盖（族条目幂等，重复追加同族无实质损失，接受最终一致性）。
        """
        client = self._get_client()
        if client is None:
            return False
        fams = [f for f in (families or []) if f]
        if not fams:
            return False
        uri = f"{SCOPE}dead_families.md"
        try:
            try:
                existing = str(client.read(uri=uri) or "")
            except Exception:
                existing = ""
            new_blocks: list[str] = []
            details = details or {}
            for fam in fams:
                if fam in existing:
                    continue
                detail = str(details.get(fam) or "")
                new_blocks.append(f"- `{fam}`：{detail}" if detail else f"- `{fam}`")
            if not new_blocks:
                return True
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            addition = "\n" + "\n".join(new_blocks) + f"\n\n<!-- updated {today} -->\n"
            client.write(uri=uri, content=existing + addition)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenViking dead_families 更新失败: %s", exc)
            return False


_PARSED_INDEX_CACHE: dict[str, list[tuple[str, Path]]] = {}


def _local_report_search(query_terms: list[str], limit: int = 4, corpus_dir=None) -> list[dict[str, Any]]:
    """在本地研报语料目录下快速检索最相关段落（带按目录缓存，耗时<50ms）。

    ``corpus_dir`` 为空时回落到默认 ``parsed/``；由 ``ALPHA_REPORT_CORPUS`` 决定用哪套语料。
    """
    import re
    from pathlib import Path

    parsed_dir = corpus_dir or _report_local_corpus()
    if not parsed_dir or not parsed_dir.is_dir():
        return []

    key = str(parsed_dir)
    if key not in _PARSED_INDEX_CACHE:
        idx = []
        for p in parsed_dir.rglob("*.md"):
            idx.append((p.stem, p))
        _PARSED_INDEX_CACHE[key] = idx
    index = _PARSED_INDEX_CACHE[key]

    keywords = [q for q in query_terms if len(q) >= 2]
    scored = []
    for stem, path in index:
        s = 0.0
        for kw in keywords:
            if kw in stem:
                s += 5.0
        if "多因子" in stem or "选股" in stem or "因子" in stem or "策略" in stem:
            s += 1.0
        if s > 0:
            scored.append((s, stem, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = []
    for _, title, path in scored[:limit]:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            snippet = _best_paragraph(content, keywords)
            clean_title = re.sub(r"^\d{4}-?\d{2}-?\d{2}_?", "", title).replace("_", " ")
            hits.append({"title": clean_title, "snippet": snippet, "uri": str(path)})
        except Exception:
            continue
    return hits


# 版式/合规噪声：这些段落对因子机制无价值（MinerU 语料里同样存在）
_PARA_NOISE_SRC = (
    r"免责|版权|法律声明|投资评级说明|分析师声明|执业证书|请务必阅读|风险提示|"
    r"微信公众号|扫码|联系方式|销售团队|地址[:：]|邮编|本公司具有中国证监会|"
    r"评级标准|买入.*增持.*中性.*减持"
)

# 目录/目次行：`- 图 1: xxx .... 5 - 表 2: yyy ...`。对公式与机制零价值，
# 但含"分组/多空/策略"+数字，原打分下会压过正文段落胜出（2026-10-01 实测 RQ_030 注入的
# 三条全是图表目次）→ 直接跳过，并把公式/机制段落提权。正则见 _best_paragraph（模块内
# `re` 是函数级导入，保持既有风格，不在模块层编译）。


def _is_recap_doc(text: str) -> bool:
    """行情复盘/市场评述类文档判定：对"因子机制/公式"检索无价值。

    实测（2026-10-01，RQ_030）：OV 命中三条全是渤海证券「公募基金周报 / 上周市场回顾 /
    权益市场主要指数震荡修复房地产领涨」——得分不低，但会把真因子机制文献挤掉。
    命中此类且无其它命中时，返回空以触发本地因子语料降级。
    """
    import re

    return bool(re.search(
        r"市场回顾|市场表现|市场评述|复盘|盘面|晨报|盘后|收盘|领涨|领跌|震荡|"
        r"涨跌|资金流向|基金周报|周报回顾",
        text or "",
    ))


def _best_paragraph(content: str, keywords: list[str], max_chars: int = 700) -> str:
    """在整篇 md 里挑与查询关键词最相关的段落（替代"取文件头 3 行"）。

    MinerU 重抽版（``parsed_mineru``）带页锚点/表格/公式，头部常是标题与目录，
    直接取头会注入无信息量的样本；这里按段落打分：关键词命中 + 含数字/算符加分，
    合规噪声、目录目次行剔除，含公式/构建方法的段落提权，取最高分段并截断。
    """
    import re

    toc_line = re.compile(r"\.{4,}|(?:图|表|图表)\s*\d+\s*[:：][^\n]{0,80}(?:图|表|图表)\s*\d+\s*[:：]")
    toc_label = re.compile(r"(?:图|表|图表)\s*\d+")
    formula_hint = re.compile(
        r"公式|因子构建|构建方法|构建方式|计算方法|计算方式|因子定义|指标定义|"
        r"\\frac|_\{|\^\{|算子|表达式"
    )
    noise = re.compile(_PARA_NOISE_SRC)
    paras: list[str] = []
    for block in re.split(r"\n\s*\n", content):
        t = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if len(t) < 40:
            continue
        if t.startswith("<!--") or t.startswith("| ---") or t.count("|") > 8:
            continue          # 页锚点注释 / 纯表格分隔行
        if noise.search(t):
            continue
        if toc_line.search(t) or len(toc_label.findall(t)) >= 3:
            continue          # 图表目次（点线目录 or 连续 图N/表N 列表）
        paras.append(t)
    if not paras:
        # 全篇都像图表目次时不返回任何东西（宁缺勿注入目次噪声）
        if len(toc_label.findall(content)) >= 3 or toc_line.search(content):
            return ""
        return content[:max_chars]

    best, best_score = paras[0], -1.0
    for t in paras:
        score = 0.0
        for kw in keywords:
            if kw and kw in t:
                score += 2.0 + 0.5 * t.count(kw)
        if re.search(r"[0-9]{2,}|IC|Rank ?IC|ICIR|夏普|换手|超额|分组|多空", t):
            score += 1.5
        if re.search(r"[a-z_]+\(|[A-Z]{2,}_[A-Z_]+", t):
            score += 1.0          # 公式/算子痕迹
        if formula_hint.search(t):
            score += 3.0          # 公式/构建方法段落优先（"公式明确"目标）
        if score > best_score:
            best, best_score = t, score
    if len(best) > max_chars:
        cut = best[:max_chars].rfind("。")
        best = best[: cut + 1] if cut > 80 else best[:max_chars] + "…"
    return best


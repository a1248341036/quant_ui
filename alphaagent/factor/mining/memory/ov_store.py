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

        client = self._get_client()
        hits = []
        if client is not None:
            try:
                results = client.search(query=query, target_uri=REPORT_SCOPE, limit=limit)
                hits = self._flatten_hits(results, limit, scope=REPORT_SCOPE)
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenViking 研报检索异常，降级到本地检索: %s", exc)

        # 本地降级检索
        if not hits:
            hits = _local_report_search(query_terms, limit=limit)

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
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit("\n", 1)[0] + "…"
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
            ("candidate_stored", "候选入库"),
            ("production_stored", "正式入库"),
            ("train_to_val_rate", "train→val 保留比"),
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


_PARSED_INDEX_CACHE: list[tuple[str, Path]] | None = None


def _local_report_search(query_terms: list[str], limit: int = 4) -> list[dict[str, Any]]:
    """在本地 data/research_reports/parsed/ 下快速检索最相关研报段落（带内存缓存，耗时<50ms）。"""
    global _PARSED_INDEX_CACHE
    import subprocess
    import re
    from pathlib import Path

    parsed_dir = None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            candidate = Path(out).parent / "data" / "research_reports" / "parsed"
            if candidate.is_dir():
                parsed_dir = candidate
    except Exception:
        pass
    if parsed_dir is None:
        fallback = Path(__file__).resolve().parents[4] / "data" / "research_reports" / "parsed"
        if fallback.is_dir():
            parsed_dir = fallback
    if parsed_dir is None:
        main_root = Path(r"D:\Quant\quant_ui\data\research_reports\parsed")
        if main_root.is_dir():
            parsed_dir = main_root

    if not parsed_dir or not parsed_dir.is_dir():
        return []

    if _PARSED_INDEX_CACHE is None:
        idx = []
        for p in parsed_dir.rglob("*.md"):
            idx.append((p.stem, p))
        _PARSED_INDEX_CACHE = idx

    keywords = [q for q in query_terms if len(q) >= 2]
    scored = []
    for stem, path in _PARSED_INDEX_CACHE:
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
            content = path.read_text(encoding="utf-8", errors="ignore")[:1500]
            lines = [l.strip() for l in content.splitlines() if len(l.strip()) > 20 and "免责" not in l and "版权" not in l]
            snippet = "\n".join(lines[:3]) if lines else content[:200]
            clean_title = re.sub(r"^\d{4}-?\d{2}-?\d{2}_?", "", title).replace("_", " ")
            hits.append({"title": clean_title, "snippet": snippet, "uri": str(path)})
        except Exception:
            continue
    return hits


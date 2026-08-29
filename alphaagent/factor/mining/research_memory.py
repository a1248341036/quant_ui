"""Persistent, evidence-backed memory for AlphaAgent factor research."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOPWORDS = {"add", "subtract", "multiply", "divide", "ts", "cs", "mean", "std", "rank", "expr", "factor"}

# 肯定（正向证据）verdict：已有明确有效性证据，应鼓励模型在其相似邻域继续探索
_POSITIVE_VERDICTS = {"production_approved", "validated", "candidate_approved", "promising"}
# 否定 verdict：已证明无效的路径，不应机械重复
_NEGATIVE_VERDICTS = {"rejected", "revise_required", "weak"}

# ── 表达式结构解析（Phase 1: 轻量级 regex 解析器，非完整 AST）────────────

# 已知算子集合（从 DSL core 中提取的高频算子，用于结构指纹和编辑类型识别）
_KNOWN_OPERATORS = frozenset({
    # 时序算子
    "ts_mean", "ts_sum", "ts_std", "ts_var", "ts_median", "ts_max", "ts_min",
    "ts_rank", "ts_quantile", "ts_delta", "ts_delay", "ts_advance",
    "ts_corr", "ts_cov", "ts_regression", "ts_decay_linear", "ts_skewness",
    "ts_kurtosis", "ts_arg_max", "ts_arg_min", "ts_pct_change",
    # 截面算子
    "cs_rank", "cs_zscore", "cs_winsorize", "cs_demean", "cs_quantile",
    "cs_residualize", "cs_neutralize",
    # 算术
    "add", "subtract", "multiply", "divide", "abs", "log", "sign",
    "max", "min", "power", "sqrt", "reciprocal", "negate",
    # 其他
    "rank", "zscore", "winsorize", "normalize", "demean", "quantile",
    "residualize", "neutralize", "if_else", "cond",
})

# 已知原始变量（含 $ 前缀的形式和无前缀形式）
_KNOWN_VARIABLES = frozenset({
    "close", "open", "high", "low", "volume", "amount", "vwap", "adj_open",
    "adj_close", "adj_high", "adj_low", "ret", "returns", "turnover",
    "market_cap", "market_value", "float_share", "total_share",
    "amt", "adv20", "adv60", "adv120",
    # 财务字段
    "funda_roe", "funda_roa", "funda_eps", "funda_bps", "funda_revenue",
    "funda_net_profit", "funda_gross_margin", "funda_net_margin",
    # 资金流/事件
    "ff_net_flow", "ff_main_flow", "ff_large_order",
    "pred_net_profit", "pred_eps",
    "holder_count", "holder_change",
    "dt_big_order", "bt_block_trade",
})

# 匹配函数调用: func_name(args)
_FUNC_CALL_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*")
# 匹配 $variable 或 bare variable
_VAR_RE = re.compile(r"\$?([a-zA-Z_][a-zA-Z0-9_]*)")
# 匹配数字常量
_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _parse_expression_structure(expression: str) -> dict[str, Any]:
    """轻量级 regex 解析器，提取表达式的算子/变量/窗口结构。

    返回::
        {
            "operators": ["rank", "subtract", "ts_mean", "ts_mean"],
            "variables": ["close"],
            "window_params": {"ts_mean": [5, 20]},
            "constants": [5, 20],
            "fingerprint": "hash(...)",
        }
    """
    if not expression:
        return {"operators": [], "variables": [], "window_params": {}, "constants": [], "fingerprint": ""}

    text = str(expression).strip()

    # 提取算子（函数名）
    operators: list[str] = []
    for match in _FUNC_CALL_RE.finditer(text):
        name = match.group(1).lower()
        if name in _KNOWN_OPERATORS:
            operators.append(name)

    # 提取变量（$close → close）
    variables: list[str] = []
    seen_vars: set[str] = set()
    for match in _VAR_RE.finditer(text):
        name = match.group(1).lower()
        # 排除算子名和纯数字
        if name in _KNOWN_OPERATORS:
            continue
        if name in {"true", "false", "null", "none", "nan", "inf"}:
            continue
        if name in _KNOWN_VARIABLES or name.startswith("funda_") or name.startswith("ff_") \
                or name.startswith("pred_") or name.startswith("holder_") \
                or name.startswith("dt_") or name.startswith("bt_"):
            if name not in seen_vars:
                variables.append(name)
                seen_vars.add(name)

    # 提取数字常量
    constants: list[int | float] = []
    for match in _NUM_RE.finditer(text):
        val = match.group(1)
        constants.append(int(val) if "." not in val else float(val))

    # 窗口参数：将算子与其后续常量关联
    window_params: dict[str, list[int | float]] = {}
    # 简单策略：按算子出现顺序，将紧跟其后的常量关联为窗口参数
    # 用 token 流扫描
    tokens = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)|(\d+(?:\.\d+)?)", text)
    op_queue: list[str] = []
    for name_tok, num_tok in tokens:
        if name_tok:
            lower = name_tok.lower()
            if lower in _KNOWN_OPERATORS:
                op_queue.append(lower)
        elif num_tok and op_queue:
            val = int(num_tok) if "." not in num_tok else float(num_tok)
            # 最后出现的算子取这个常量
            last_op = op_queue[-1]
            window_params.setdefault(last_op, []).append(val)

    # 结构指纹：变量→VAR，常量→N，算子保留，拼接后哈希
    fingerprint = _structure_fingerprint(text)

    return {
        "operators": operators,
        "variables": variables,
        "window_params": window_params,
        "constants": constants,
        "fingerprint": fingerprint,
    }


def _structure_fingerprint(expression: str) -> str:
    """计算表达式结构指纹：变量替换为 VAR，数字替换为 N，算子保留。

    示例::
        rank(subtract(ts_mean($close, 5), ts_mean($close, 20)))
        → hash("rank(subtract(ts_mean(VAR,N),ts_mean(VAR,N)))")
    """
    if not expression:
        return ""
    text = str(expression).strip()
    # 替换 $variable → VAR
    text = re.sub(r"\$[a-zA-Z_][a-zA-Z0-9_]*", "VAR", text)
    # 替换 bare variables → VAR (只替换已知变量，避免误替换算子)
    for var in sorted(_KNOWN_VARIABLES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(var)}\b", "VAR", text)
    # 替换 funda_*/ff_*/pred_*/holder_*/dt_*/bt_* 前缀变量
    text = re.sub(r"\b(?:funda_|ff_|pred_|holder_|dt_|bt_)[a-zA-Z0-9_]+", "VAR", text)
    # 替换数字 → N
    text = re.sub(r"\d+(?:\.\d+)?", "N", text)
    # 规范化空格和换行
    text = re.sub(r"\s+", "", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _identify_edit_type(
    parent_struct: dict[str, Any],
    child_struct: dict[str, Any],
) -> dict[str, Any] | None:
    """对比两个因子结构，识别编辑类型和详情。

    返回 ``None`` 表示无法识别有意义的编辑（结构完全不同或完全相同）。

    返回示例::
        {"edit_type": "window_extend", "detail": {"operator": "ts_mean", "from": 5, "to": 20}}
    """
    if not parent_struct or not child_struct:
        return None
    if parent_struct.get("fingerprint") == child_struct.get("fingerprint"):
        return None  # 结构完全相同，无编辑

    p_ops = parent_struct.get("operators", [])
    c_ops = child_struct.get("operators", [])
    p_vars = set(parent_struct.get("variables", []))
    c_vars = set(child_struct.get("variables", []))
    p_wins = parent_struct.get("window_params", {})
    c_wins = child_struct.get("window_params", {})

    # 算子集合差异
    p_ops_set = set(p_ops)
    c_ops_set = set(c_ops)
    ops_added = c_ops_set - p_ops_set
    ops_removed = p_ops_set - c_ops_set

    # 变量差异
    vars_added = c_vars - p_vars
    vars_removed = p_vars - c_vars

    # 窗口参数差异
    win_changes: list[dict[str, Any]] = []
    common_op_wins = set(p_wins.keys()) & set(c_wins.keys())
    for op in sorted(common_op_wins):
        p_vals = p_wins.get(op, [])
        c_vals = c_wins.get(op, [])
        if p_vals != c_vals:
            # 找出变化的具体窗口
            for i, (pv, cv) in enumerate(zip(p_vals, c_vals)):
                if pv != cv:
                    if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
                        if cv > pv:
                            win_changes.append({
                                "operator": op, "position": i,
                                "from": pv, "to": cv, "direction": "extend",
                            })
                        else:
                            win_changes.append({
                                "operator": op, "position": i,
                                "from": pv, "to": cv, "direction": "shrink",
                            })

    # 判定编辑类型（优先级从高到低）

    # 1. interaction_add: 从单信号变为双信号（变量数量增加）
    if len(p_vars) < len(c_vars) and len(ops_added) > 0:
        return {
            "edit_type": "interaction_add",
            "detail": {
                "new_variables": sorted(vars_added),
                "new_operators": sorted(ops_added),
            },
        }

    # 2. composition_add: 增加了算子层级（算子数量增加但变量不变）
    if len(c_ops) > len(p_ops) and vars_added == set() and vars_removed == set():
        return {
            "edit_type": "composition_add",
            "detail": {"new_operators": sorted(ops_added)},
        }

    # 3. variable_replace: 变量替换，算子和窗口不变
    if vars_added and vars_removed and ops_added == set() and ops_removed == set() and not win_changes:
        return {
            "edit_type": "variable_replace",
            "detail": {
                "from": sorted(vars_removed),
                "to": sorted(vars_added),
            },
        }

    # 4. window_extend / window_shrink: 窗口参数变化
    if win_changes and not vars_added and not vars_removed and ops_added == set() and ops_removed == set():
        all_extend = all(c["direction"] == "extend" for c in win_changes)
        all_shrink = all(c["direction"] == "shrink" for c in win_changes)
        if all_extend:
            return {
                "edit_type": "window_extend",
                "detail": {"changes": win_changes},
            }
        if all_shrink:
            return {
                "edit_type": "window_shrink",
                "detail": {"changes": win_changes},
            }
        # 混合窗口变化
        return {
            "edit_type": "window_change",
            "detail": {"changes": win_changes},
        }

    # 5. operator_swap: 算子替换，变量和窗口不变
    if ops_added and ops_removed and not vars_added and not vars_removed and not win_changes:
        return {
            "edit_type": "operator_swap",
            "detail": {
                "from": sorted(ops_removed),
                "to": sorted(ops_added),
            },
        }

    # 6. normalize_change: 归一化方式改变（末尾算子变化）
    if ops_added and ops_removed:
        # 如果差异只在外层归一化算子
        norm_ops = {"rank", "zscore", "winsorize", "normalize", "demean", "quantile",
                    "cs_rank", "cs_zscore", "cs_winsorize", "cs_demean", "cs_quantile"}
        if ops_added.issubset(norm_ops) and ops_removed.issubset(norm_ops):
            return {
                "edit_type": "normalize_change",
                "detail": {
                    "from": sorted(ops_removed),
                    "to": sorted(ops_added),
                },
            }

    # 7. decorrelation_add: 新增去相关处理
    decorr_ops = {"residualize", "neutralize", "cs_residualize", "cs_neutralize"}
    if ops_added & decorr_ops:
        return {
            "edit_type": "decorrelation_add",
            "detail": {"new_operators": sorted(ops_added & decorr_ops)},
        }

    # 8. 复合编辑：多种变化同时发生
    changes: list[str] = []
    if win_changes:
        changes.append("window_change")
    if vars_added or vars_removed:
        changes.append("variable_change")
    if ops_added or ops_removed:
        changes.append("operator_change")
    if changes:
        return {
            "edit_type": "compound_edit",
            "detail": {
                "changes": changes,
                "vars_added": sorted(vars_added),
                "vars_removed": sorted(vars_removed),
                "ops_added": sorted(ops_added),
                "ops_removed": sorted(ops_removed),
                "win_changes": win_changes,
            },
        }

    return None


def _neg_str(s: Any) -> str:
    """Return a string that sorts *before* all normal strings, so that when
    used as a secondary sort key in ascending order the *most recent*
    (lexicographically largest) timestamp comes first."""
    # Invert by prefixing with a high char — simple but effective for
    # ISO-8601 timestamps which are all ASCII.
    return "\uffff" + str(s)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None

_jieba_cut = None
try:  # 可选依赖：装 jieba 时用标准中文分词，否则回退 bigram
    import jieba  # type: ignore
    _jieba_cut = jieba.lcut_for_search
except Exception:  # pragma: no cover - 环境无 jieba 时走 bigram
    pass


def _cjk_tokens(text: str) -> set[str]:
    """抽取中文字段为可检索 token。

    优先使用 jieba（安装后生效），否则按连续中文串生成 bigram。
    """
    if not text:
        return set()
    if _jieba_cut is not None:
        words = {w.strip().lower() for w in _jieba_cut(text) if len(w.strip()) >= 2 and _CJK_RE.fullmatch(w.strip())}
        if words:
            return words
    out: set[str] = set()
    for seq in _CJK_RE.findall(text):
        if len(seq) == 2:
            out.add(seq)
            continue
        out.update(seq[i:i + 2] for i in range(len(seq) - 1))
    return out


def _tokens(*values: Any) -> list[str]:
    """从英文表达式/字段与中文结论中抽取去重 token，供 BM25 检索。"""
    found: set[str] = set()
    for value in values:
        text = str(value or "")
        for token in _TOKEN_RE.findall(text):
            token = token.lower()
            found.add(token)
            # 下划线连接的复合词拆开（momentum_reversal -> momentum + reversal），
            # 否则查询侧 "momentum reversal"（空格分词）与条目侧对不上
            found.update(part for part in token.split("_") if len(part) >= 2)
        found.update(_cjk_tokens(text))
    return sorted(found - _STOPWORDS)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class ResearchMemoryStore:
    """Stores compact research conclusions, never raw model thinking or prompts."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS memory_entries (
        id TEXT PRIMARY KEY,
        factor_name TEXT NOT NULL,
        expression TEXT NOT NULL,
        conclusion TEXT,
        verdict TEXT NOT NULL,
        stage TEXT,
        profile_id TEXT,
        profile_hash TEXT,
        candidate_id TEXT,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        interaction_json TEXT,
        error TEXT,
        failure_code TEXT,
        last_run_id TEXT,
        attempts INTEGER NOT NULL DEFAULT 1,
        tokens_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT,
        updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS store_meta (
        k TEXT PRIMARY KEY,
        v TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_memory_entries_verdict
        ON memory_entries(verdict);
    CREATE INDEX IF NOT EXISTS idx_memory_entries_stage
        ON memory_entries(stage);
    CREATE INDEX IF NOT EXISTS idx_memory_entries_updated_at
        ON memory_entries(updated_at);
    CREATE INDEX IF NOT EXISTS idx_memory_entries_failure_code
        ON memory_entries(failure_code);

    CREATE TABLE IF NOT EXISTS memory_observations (
        entry_id TEXT NOT NULL,
        run_id TEXT,
        observed_at TEXT,
        stage TEXT,
        verdict TEXT,
        failure_code TEXT,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (entry_id, observed_at, run_id),
        FOREIGN KEY (entry_id) REFERENCES memory_entries(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_memory_observations_entry
        ON memory_observations(entry_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        entry_id UNINDEXED,
        factor_name,
        expression,
        conclusion,
        failure_code,
        search_tokens,
        tokenize='unicode61'
    );

    CREATE TRIGGER IF NOT EXISTS memory_entries_after_delete
    AFTER DELETE ON memory_entries
    BEGIN
        DELETE FROM memory_fts WHERE entry_id = old.id;
    END;

    CREATE TABLE IF NOT EXISTS memory_patterns (
        id TEXT PRIMARY KEY,
        layer TEXT NOT NULL,
        category TEXT,
        content TEXT NOT NULL,
        evidence_json TEXT,
        success_rate REAL,
        total_attempts INTEGER NOT NULL DEFAULT 1,
        success_count INTEGER NOT NULL DEFAULT 0,
        saturation_score REAL NOT NULL DEFAULT 0,
        confidence REAL NOT NULL DEFAULT 0.5,
        created_at TEXT,
        updated_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_memory_patterns_layer
        ON memory_patterns(layer);
    CREATE INDEX IF NOT EXISTS idx_memory_patterns_category
        ON memory_patterns(category);
    CREATE INDEX IF NOT EXISTS idx_memory_patterns_confidence
        ON memory_patterns(confidence);

    -- ── Phase 1: 编辑模式表（AlphaMemo SSPM 对标）─────────────────────
    -- 记录父因子→编辑操作→子因子→结果 的完整链条
    CREATE TABLE IF NOT EXISTS edit_patterns (
        id TEXT PRIMARY KEY,
        parent_expression TEXT,
        child_expression TEXT NOT NULL,
        parent_fingerprint TEXT,
        child_fingerprint TEXT NOT NULL,
        edit_type TEXT NOT NULL,          -- window_extend/shrink, operator_swap, variable_replace, interaction_add, ...
        edit_detail_json TEXT,            -- JSON: 具体编辑详情（哪些算子/变量/窗口变了）
        parent_ic REAL,
        child_ic REAL,
        delta_ic REAL,                    -- IC 变化 (child - parent)
        parent_icir REAL,
        child_icir REAL,
        delta_icir REAL,
        verdict TEXT,                     -- success / failure / neutral
        confidence REAL NOT NULL DEFAULT 0.5,
        total_uses INTEGER NOT NULL DEFAULT 0,
        success_count INTEGER NOT NULL DEFAULT 0,
        vetoed INTEGER NOT NULL DEFAULT 0,  -- 不对称否决标记
        family TEXT,
        parent_entry_id TEXT,             -- 关联 memory_entries.id
        child_entry_id TEXT,              -- 关联 memory_entries.id
        created_at TEXT,
        updated_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_edit_patterns_type
        ON edit_patterns(edit_type);
    CREATE INDEX IF NOT EXISTS idx_edit_patterns_verdict
        ON edit_patterns(verdict);
    CREATE INDEX IF NOT EXISTS idx_edit_patterns_family
        ON edit_patterns(family);
    CREATE INDEX IF NOT EXISTS idx_edit_patterns_vetoed
        ON edit_patterns(vetoed);
    CREATE INDEX IF NOT EXISTS idx_edit_patterns_confidence
        ON edit_patterns(confidence);
    """

    # 因子族分类规则：用于蒸馏算子和饱和度跟踪
    _FAMILY_RULES: dict[str, tuple[str, ...]] = {
        "gap_overnight": ("gap", "overnight", "open_close"),
        "vwap": ("vwap",),
        "chip": ("chip", "peak", "entropy"),
        "momentum": ("momentum", "pctchange", "ma_w", "ma20", "ma_dev"),
        "reversal": ("reversal", "neg_ts"),
        "volume": ("volume", "amount", "turnover"),
        "volatility": ("std", "var", "vol_"),
        "fundamental": ("funda_", "roe", "roa", "growth", "quality", "value"),
        "correlation": ("corr", "cov", "rankcorr"),
        "liquidity": ("liquidity", "float", "amihud"),
        "breadth": ("breadth", "advance", "decline"),
    }

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() == ".json":
            # 存储已迁至 SQLite；容忍调用方传旧 JSON 路径（迁移逻辑仍读该 JSON）。
            self.path = self.path.with_suffix(".db")
        self._schema_ready = False

    @contextmanager
    def _open(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_schema(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.executescript(self._SCHEMA)
        # memory_entries 列迁移
        entry_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_entries)").fetchall()
        }
        if "interaction_json" not in entry_columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN interaction_json TEXT")
        # Phase 1: 新增结构指纹/算子列表/窗口参数/父因子列
        if "structure_fingerprint" not in entry_columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN structure_fingerprint TEXT")
        if "operator_list_json" not in entry_columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN operator_list_json TEXT")
        if "window_params_json" not in entry_columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN window_params_json TEXT")
        if "parent_id" not in entry_columns:
            conn.execute("ALTER TABLE memory_entries ADD COLUMN parent_id TEXT")

        # memory_patterns 列迁移（Phase 1/2）
        pattern_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_patterns)").fetchall()
        }
        if "structure_fingerprint" not in pattern_columns:
            conn.execute("ALTER TABLE memory_patterns ADD COLUMN structure_fingerprint TEXT")
        if "edit_types_json" not in pattern_columns:
            conn.execute("ALTER TABLE memory_patterns ADD COLUMN edit_types_json TEXT")
        if "operator_combo_json" not in pattern_columns:
            conn.execute("ALTER TABLE memory_patterns ADD COLUMN operator_combo_json TEXT")
        if "parent_context_json" not in pattern_columns:
            conn.execute("ALTER TABLE memory_patterns ADD COLUMN parent_context_json TEXT")

        self._migrate_legacy_json(conn)
        conn.commit()
        self._schema_ready = True

    def _migrate_legacy_json(self, conn: sqlite3.Connection) -> None:
        """One-time import from the original JSON store when switching to SQLite."""
        legacy_path = self.path.with_suffix(".json")
        if legacy_path == self.path or not legacy_path.is_file():
            return
        try:
            raw = json.loads(legacy_path.read_text(encoding="utf-8"))
            entries = raw.get("entries", []) if isinstance(raw, dict) else []
        except (OSError, json.JSONDecodeError):
            return
        if not any(isinstance(item, dict) and item.get("id") for item in entries):
            return

        count = conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0]
        if count:
            return
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id"):
                self._write_entry(conn, self._normalize_entry(entry))

    @staticmethod
    def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(entry)
        observations = normalized.get("observations", [])
        if isinstance(observations, int):
            normalized["observations"] = [{
                "at": normalized.get("updated_at"),
                "verdict": normalized.get("verdict"),
            }]
        elif not isinstance(observations, list):
            normalized["observations"] = []
        return normalized

    def _load(self) -> list[dict[str, Any]]:
        with self._open() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_entries ORDER BY updated_at DESC"
            ).fetchall()
            return self._hydrate_entries(conn, rows)

    @staticmethod
    def _metrics_json(metrics: dict[str, Any]) -> str:
        return json.dumps(metrics, ensure_ascii=False, separators=(",", ":"))

    def _write_entry(self, conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT INTO memory_entries (
                id, factor_name, expression, conclusion, verdict, stage,
                profile_id, profile_hash, candidate_id, metrics_json, interaction_json, error,
                failure_code, last_run_id, attempts, tokens_json,
                created_at, updated_at,
                structure_fingerprint, operator_list_json, window_params_json, parent_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                factor_name=excluded.factor_name,
                expression=excluded.expression,
                conclusion=excluded.conclusion,
                verdict=excluded.verdict,
                stage=excluded.stage,
                profile_id=excluded.profile_id,
                profile_hash=excluded.profile_hash,
                candidate_id=excluded.candidate_id,
                metrics_json=excluded.metrics_json,
                interaction_json=excluded.interaction_json,
                error=excluded.error,
                failure_code=excluded.failure_code,
                last_run_id=excluded.last_run_id,
                attempts=excluded.attempts,
                tokens_json=excluded.tokens_json,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                structure_fingerprint=excluded.structure_fingerprint,
                operator_list_json=excluded.operator_list_json,
                window_params_json=excluded.window_params_json,
                parent_id=excluded.parent_id
            """,
            (
                entry["id"], entry.get("factor_name"), entry.get("expression"),
                entry.get("conclusion"), entry.get("verdict"), entry.get("stage"),
                entry.get("profile_id"), entry.get("profile_hash"),
                entry.get("candidate_id"), self._metrics_json(entry.get("metrics", {})),
                json.dumps(entry.get("interaction"), ensure_ascii=False) if entry.get("interaction") is not None else None,
                entry.get("error"), entry.get("failure_code"),
                entry.get("last_run_id"), int(entry.get("attempts", 1)),
                json.dumps(entry.get("tokens", []), ensure_ascii=False),
                entry.get("created_at"), entry.get("updated_at"),
                entry.get("structure_fingerprint"),
                entry.get("operator_list_json"),
                entry.get("window_params_json"),
                entry.get("parent_id"),
            ),
        )
        conn.execute("DELETE FROM memory_observations WHERE entry_id = ?", (entry["id"],))
        conn.executemany(
            """
            INSERT OR IGNORE INTO memory_observations (
                entry_id, run_id, observed_at, stage, verdict, failure_code, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry["id"], obs.get("run_id"), obs.get("at"), obs.get("stage"),
                    obs.get("verdict"), obs.get("failure_code"),
                    self._metrics_json(obs.get("metrics", {})),
                )
                for obs in entry.get("observations", [])
                if isinstance(obs, dict)
            ],
        )
        conn.execute("DELETE FROM memory_fts WHERE entry_id = ?", (entry["id"],))
        conn.execute(
            """
            INSERT INTO memory_fts (
                entry_id, factor_name, expression, conclusion, failure_code, search_tokens
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry["id"], entry.get("factor_name") or "", entry.get("expression") or "",
                entry.get("conclusion") or "", entry.get("failure_code") or "",
                " ".join(entry.get("tokens", [])),
            ),
        )

    def _hydrate_entries(
        self,
        conn: sqlite3.Connection,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        obs_rows = conn.execute(
            f"""
            SELECT * FROM memory_observations
            WHERE entry_id IN ({placeholders})
            ORDER BY observed_at
            """,
            ids,
        ).fetchall()
        observations: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in rows}
        for row in obs_rows:
            observations.setdefault(row["entry_id"], []).append({
                "run_id": row["run_id"],
                "at": row["observed_at"],
                "stage": row["stage"],
                "verdict": row["verdict"],
                "failure_code": row["failure_code"],
                "metrics": json.loads(row["metrics_json"] or "{}"),
            })

        output: list[dict[str, Any]] = []
        for row in rows:
            item = {
                "id": row["id"],
                "factor_name": row["factor_name"],
                "expression": row["expression"],
                "conclusion": row["conclusion"],
                "verdict": row["verdict"],
                "stage": row["stage"],
                "profile_id": row["profile_id"],
                "profile_hash": row["profile_hash"],
                "candidate_id": row["candidate_id"],
                "metrics": json.loads(row["metrics_json"] or "{}"),
                "error": row["error"],
                "failure_code": row["failure_code"],
                "last_run_id": row["last_run_id"],
                "attempts": row["attempts"],
                "tokens": json.loads(row["tokens_json"] or "[]"),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "observations": observations.get(row["id"], []),
            }
            output.append(item)
        return output

    def record_tool_result(self, *, run_id: str, row: dict[str, Any]) -> dict[str, Any] | None:
        name = str(row.get("name") or "")
        if name not in {"evaluate_factor", "eval_on_train_set", "eval_on_val_set", "submit_factor"}:
            return None
        args = _parse_args(row.get("arguments_raw"))
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        expression = str(args.get("multi_line_expr") or "").strip()
        factor_name = str(args.get("factor_name") or result.get("factor_name") or "expr").strip()
        if not expression:
            return None

        profile_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else profile_metrics.get("cross_sectional_core", {})
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else summary
        if name == "evaluate_factor":
            metrics = summary
        # 提取嵌套回测指标到扁平 metrics，供 _compact_metrics 捕获
        metrics = self._flatten_backtest_metrics(metrics, profile_metrics)
        error = str(result.get("error") or result.get("skipped_reason") or "")
        verdict, conclusion = self._classify(name, result, metrics, error)
        canonical_expression = "\n".join(line.strip() for line in expression.splitlines() if line.strip())
        signature = hashlib.sha256(canonical_expression.encode("utf-8")).hexdigest()[:20]
        failure_code = self._failure_code(name, result, error, verdict)

        # Phase 1: 解析表达式结构
        struct = _parse_expression_structure(expression)

        with self._open() as conn:
            previous_row = conn.execute(
                "SELECT attempts, created_at FROM memory_entries WHERE id = ?",
                (signature,),
            ).fetchone()
            # Phase 2: 隐式父子链接——找最相似的历史因子作为隐式父因子
            parent_id = None
            if struct.get("fingerprint"):
                parent_id = self._find_implicit_parent(conn, struct, exclude_id=signature)

        previous_attempts = int(previous_row["attempts"]) if previous_row else 0
        observation = {
            "run_id": run_id,
            "at": _now(),
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "verdict": verdict,
            "failure_code": failure_code,
            "metrics": self._compact_metrics(metrics),
            "interaction": _parse_args(args.get("interaction")),
        }
        entry = {
            "id": signature,
            "factor_name": factor_name,
            "expression": expression,
            "tokens": _tokens(factor_name, expression, conclusion),
            "verdict": verdict,
            "conclusion": conclusion,
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "profile_id": result.get("profile", {}).get("profile_id") if isinstance(result.get("profile"), dict) else None,
            "profile_hash": result.get("profile_hash"),
            "candidate_id": result.get("candidate", {}).get("candidate_id") if isinstance(result.get("candidate"), dict) else None,
            "metrics": self._compact_metrics(metrics),
            "error": error[:500],
            "failure_code": failure_code,
            "last_run_id": run_id,
            "attempts": previous_attempts + 1,
            "updated_at": _now(),
            "observations": [observation],
            # Phase 1: 结构信息
            "structure_fingerprint": struct.get("fingerprint"),
            "operator_list_json": json.dumps(struct.get("operators", []), ensure_ascii=False),
            "window_params_json": json.dumps(struct.get("window_params", {}), ensure_ascii=False),
            "parent_id": parent_id,
        }

        if previous_row:
            entry["created_at"] = previous_row["created_at"] or entry["updated_at"]
            with self._open() as conn:
                old_rows = conn.execute(
                    """
                    SELECT * FROM memory_observations
                    WHERE entry_id = ?
                    ORDER BY observed_at
                    """,
                    (signature,),
                ).fetchall()
            old_observations = [
                {
                    "run_id": row["run_id"],
                    "at": row["observed_at"],
                    "stage": row["stage"],
                    "verdict": row["verdict"],
                    "failure_code": row["failure_code"],
                    "metrics": json.loads(row["metrics_json"] or "{}"),
                }
                for row in old_rows
            ]
            entry["observations"] = [*old_observations[-98:], observation]
        else:
            entry["created_at"] = entry["updated_at"]

        with self._open() as conn:
            self._write_entry(conn, entry)

        # Phase 2/4: 如果有隐式父因子，记录编辑模式
        if parent_id and struct.get("fingerprint"):
            try:
                self._record_edit_pattern_from_parent(
                    parent_id=parent_id,
                    child_id=signature,
                    child_expression=expression,
                    child_struct=struct,
                    child_metrics=entry["metrics"],
                )
            except Exception:
                pass  # 编辑模式记录失败不影响主流程

        return entry

    @staticmethod
    def _flatten_backtest_metrics(metrics: dict[str, Any], profile_metrics: dict[str, Any]) -> dict[str, Any]:
        """把 quantile_portfolio / topn_portfolio / engine_backtest 的嵌套回测指标
        提取到扁平 metrics，供 _compact_metrics 统一捕获。

        evaluate_factor（eval_profile 路径）的完整结果在 profile_metrics 里；
        submit_factor 的结果在 metrics["quantile_portfolio"] 里。
        evaluate_factor 走 summary 路径时 profile_metrics 才有嵌套结构。
        """
        flat = dict(metrics)  # 不修改原 dict

        # quantile_portfolio：submit_factor metrics 和 evaluate_factor profile_metrics 都有
        qp = profile_metrics.get("quantile_portfolio") if isinstance(profile_metrics, dict) else None
        if isinstance(qp, dict):
            flat.setdefault("annualized_return", qp.get("top_group_annualized_return"))
            flat.setdefault("annualized_excess_return", qp.get("top_group_annualized_excess_return"))
            flat.setdefault("sharpe", qp.get("top_group_sharpe"))
            flat.setdefault("excess_sharpe", qp.get("top_group_excess_sharpe"))
            flat.setdefault("max_drawdown", qp.get("top_group_max_drawdown"))
            flat.setdefault("annual_turnover", qp.get("avg_daily_side_turnover"))
            flat.setdefault("monotonicity", qp.get("monotonicity"))

        # topn_portfolio：evaluate_factor（eval_profile）路径的 profile_metrics
        tp = profile_metrics.get("topn_portfolio") if isinstance(profile_metrics, dict) else None
        if isinstance(tp, dict):
            flat.setdefault("annualized_return", tp.get("annualized_return"))
            flat.setdefault("annualized_excess_return", tp.get("annualized_excess_return"))
            flat.setdefault("sharpe", tp.get("sharpe"))
            flat.setdefault("max_drawdown", tp.get("max_drawdown"))
            flat.setdefault("daily_overlap", tp.get("daily_overlap"))
            flat.setdefault("excess_sharpe", tp.get("excess_sharpe"))

        # engine_backtest：submit_factor 路径（submit 结果里也可能嵌在 metrics 中）
        eb = profile_metrics.get("engine_backtest") if isinstance(profile_metrics, dict) else None
        if isinstance(eb, dict):
            m = eb.get("metrics") if isinstance(eb.get("metrics"), dict) else eb
            flat.setdefault("annualized_return", m.get("annual_return"))
            flat.setdefault("annualized_excess_return", m.get("excess_annual"))
            flat.setdefault("sharpe", m.get("sharpe"))
            flat.setdefault("max_drawdown", m.get("max_drawdown"))
            flat.setdefault("daily_overlap", m.get("daily_overlap"))

        return flat

    @staticmethod
    def _compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
        keys = (
            "ic", "icir", "rank_ic", "factor_coverage", "coverage",
            "long_group_annual_excess_return", "winsorized_abs_ic_decay",
            "annualized_return", "annualized_excess_return", "sharpe",
            "max_drawdown", "annual_turnover", "daily_overlap",
            "excess_sharpe", "monotonicity",
        )
        return {key: value for key in keys if (value := _safe_float(metrics.get(key))) is not None}

    @staticmethod
    def _failure_code(name: str, result: dict[str, Any], error: str, verdict: str) -> str | None:
        if not error and verdict not in {"weak", "rejected", "revise_required"}:
            return None
        text = error.lower()
        if "timeout" in text:
            return "model_timeout" if "model" in text else "eval_timeout"
        if "corr" in text or "similar" in text or "duplicate" in text:
            return "correlation_duplicate"
        if "sign" in text:
            return "sign_flip"
        if name == "submit_factor":
            if "engine_gate" in text:
                return "backtest_failed"
            return "stage_one_failed" if not result.get("candidate_stored") else "stage_two_failed"
        if name == "eval_on_train_set":
            return "train_threshold"
        if name == "eval_on_val_set":
            return "val_threshold"
        return "tool_error" if error else "metric_threshold"

    @staticmethod
    def _classify(name: str, result: dict[str, Any], metrics: dict[str, Any], error: str) -> tuple[str, str]:
        review = result.get("factor_review") if isinstance(result.get("factor_review"), dict) else {}
        review_verdict = review.get("verdict")
        canonical = str(review.get("canonical_form") or "因子结构审核")
        reasons = review.get("reasons") if isinstance(review.get("reasons"), list) else []
        if review_verdict == "reject":
            return "rejected", f"{canonical}：Reviewer 拒绝；{str(reasons[0]) if reasons else '不得重复同构表达式。'}"
        if review_verdict == "revise":
            return "revise_required", f"{canonical}：Reviewer 要求结构性改造后再评估。"
        if error:
            # error 可能携带巨型内部状态（如 pandas 报错的完整日期列表），
            # 必须截断后再写入 conclusion，避免记忆库膨胀。
            snippet = error if len(error) <= 500 else error[:497] + "..."
            return "rejected", f"{name} 被否定：{snippet}"
        if name == "submit_factor":
            if result.get("stored"):
                return "production_approved", "已通过精筛并正式入库，应保留其经济机制并避免重复。"
            if result.get("candidate_stored"):
                return "candidate_approved", "通过海选进入候选池，尚未满足精筛条件，应针对失败项改进。"
            return "rejected", "提交未通过，避免在未改变机制或拒绝原因的情况下重复提交。"
        ic = _safe_float(metrics.get("ic"))
        icir = _safe_float(metrics.get("icir"))
        coverage = _safe_float(metrics.get("factor_coverage", metrics.get("coverage")))
        is_val = name == "eval_on_val_set" or result.get("split") == "val"

        # Phase 2: 生成包含结构信息的 conclusion
        ic_str = f"IC={ic:+.4f}" if ic is not None else "IC=N/A"
        icir_str = f"ICIR={icir:+.3f}" if icir is not None else "ICIR=N/A"
        cov_str = f"Coverage={coverage:.2f}" if coverage is not None else ""

        if is_val and (result.get("sign_check", {}).get("matches_expected_sign") is not False) and abs(ic or 0) >= 0.015:
            return "validated", f"训练外验证通过：{ic_str} {icir_str} {cov_str}。方向一致且有可用相关性，可在相邻但不重复的机制上扩展。"
        if abs(ic or 0) >= 0.015 and (icir or 0) > 0.2 and (coverage or 0) > 0.85:
            return "promising", f"训练阶段有潜力：{ic_str} {icir_str} {cov_str}。优先进行训练外验证或独立性改造。"
        # weak: 包含具体指标，便于后续编辑模式分析
        return "weak", f"指标不足：{ic_str} {icir_str} {cov_str}。除非改变变量、经济机制或处理方式，否则不要机械重试。"

    # Verdict 分类
    _POSITIVE_VERDICTS = frozenset({"production_approved", "validated", "candidate_approved", "promising"})
    _NEGATIVE_VERDICTS = frozenset({"rejected", "revise_required", "weak"})

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: 隐式父子链接 + 编辑模式记录
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_implicit_parent(
        conn: sqlite3.Connection,
        child_struct: dict[str, Any],
        *,
        exclude_id: str | None = None,
        limit: int = 100,
    ) -> str | None:
        """用结构指纹相似度找到最接近的历史因子作为隐式父因子。

        相似度 = 算子列表 Jaccard + 变量列表 Jaccard + 窗口参数编辑距离。
        相似度 > 0.5 时建立隐式链接；完全相同的指纹不视为父因子（同一因子的重评估）。
        """
        child_fp = child_struct.get("fingerprint")
        if not child_fp:
            return None

        child_ops = set(child_struct.get("operators", []))
        child_vars = set(child_struct.get("variables", []))

        # 查询候选父因子（最近 limit 条，排除自身和相同指纹）
        params: list[Any] = []
        where_parts = ["structure_fingerprint IS NOT NULL", "structure_fingerprint != ?"]
        params.append(child_fp)
        if exclude_id:
            where_parts.append("id != ?")
            params.append(exclude_id)
        where = " AND ".join(where_parts)

        rows = conn.execute(
            f"""
            SELECT id, structure_fingerprint, operator_list_json, window_params_json
            FROM memory_entries
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (*params, int(limit)),
        ).fetchall()

        if not rows:
            return None

        best_id: str | None = None
        best_score: float = 0.0

        for r in rows:
            parent_ops = set(json.loads(r["operator_list_json"] or "[]"))
            parent_vars: set[str] = set()  # 从表达式中提取
            # 简单从 operator_list_json 之外提取变量（需要读 expression，这里用 tokens_json 近似）
            # 用 expression 重新解析更准确，但性能考虑用已有的 operator set
            parent_wins = json.loads(r["window_params_json"] or "{}")

            # 算子 Jaccard
            ops_union = child_ops | parent_ops
            ops_inter = child_ops & parent_ops
            ops_sim = len(ops_inter) / len(ops_union) if ops_union else 0.0

            # 变量 Jaccard（从 parent fingerprint 提取不准确，用 expression 重新解析）
            # 简化：用窗口参数编辑距离
            child_wins = child_struct.get("window_params", {})
            common_ops = set(child_wins.keys()) & set(parent_wins.keys())
            if common_ops:
                win_diffs = []
                for op in common_ops:
                    cv = child_wins.get(op, [])
                    pv = parent_wins.get(op, [])
                    if cv and pv:
                        # 对同位置的窗口值做归一化差异
                        for i in range(min(len(cv), len(pv))):
                            if pv[i] != cv[i]:
                                try:
                                    max_val = max(float(pv[i]), float(cv[i]))
                                    if max_val > 0:
                                        win_diffs.append(abs(float(cv[i]) - float(pv[i])) / max_val)
                                except (TypeError, ValueError):
                                    pass
                win_sim = 1.0 - (sum(win_diffs) / len(win_diffs)) if win_diffs else 1.0
            else:
                win_sim = 0.5  # 无共同窗口算子，给中等分

            # 综合相似度
            score = 0.5 * ops_sim + 0.5 * win_sim
            if score > best_score:
                best_score = score
                best_id = r["id"]

        return best_id if best_score > 0.5 else None

    def _record_edit_pattern_from_parent(
        self,
        *,
        parent_id: str,
        child_id: str,
        child_expression: str,
        child_struct: dict[str, Any],
        child_metrics: dict[str, float],
    ) -> str | None:
        """当有隐式或显式父因子时，记录编辑模式。返回 edit_pattern id。"""
        # 读取父因子信息
        with self._open() as conn:
            parent_row = conn.execute(
                "SELECT expression, structure_fingerprint, operator_list_json, window_params_json, metrics_json FROM memory_entries WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if not parent_row:
                return None

            parent_expression = parent_row["expression"]
            parent_struct = {
                "fingerprint": parent_row["structure_fingerprint"],
                "operators": json.loads(parent_row["operator_list_json"] or "[]"),
                "variables": [],  # 简化：从 expression 重新解析
                "window_params": json.loads(parent_row["window_params_json"] or "{}"),
                "constants": [],
            }
            # 补全父因子结构
            parent_struct = _parse_expression_structure(parent_expression)
            parent_metrics = json.loads(parent_row["metrics_json"] or "{}")

        # 识别编辑类型
        edit = _identify_edit_type(parent_struct, child_struct)
        if edit is None:
            return None

        edit_type = edit["edit_type"]
        edit_detail = edit["detail"]

        # 计算 delta_ic, delta_icir
        parent_ic = _safe_float(parent_metrics.get("ic"))
        child_ic = _safe_float(child_metrics.get("ic"))
        parent_icir = _safe_float(parent_metrics.get("icir"))
        child_icir = _safe_float(child_metrics.get("icir"))

        delta_ic = None
        if parent_ic is not None and child_ic is not None:
            delta_ic = child_ic - parent_ic

        delta_icir = None
        if parent_icir is not None and child_icir is not None:
            delta_icir = child_icir - parent_icir

        # 判定 verdict
        IC_SUCCESS_THRESHOLD = 0.003
        if delta_ic is not None:
            if delta_ic > IC_SUCCESS_THRESHOLD:
                verdict = "success"
            elif delta_ic < -IC_SUCCESS_THRESHOLD:
                verdict = "failure"
            else:
                verdict = "neutral"
        else:
            verdict = "neutral"

        # 信号族
        family = self._classify_family(
            str(child_struct.get("variables", [""])),
            child_expression,
        )

        # 写入 edit_patterns 表
        pattern_id = hashlib.sha256(
            f"{edit_type}|{parent_struct.get('fingerprint', '')}|{child_struct.get('fingerprint', '')}".encode("utf-8")
        ).hexdigest()[:20]

        now = _now()
        edit_detail_json = json.dumps(edit_detail, ensure_ascii=False, separators=(",", ":"))

        with self._open() as conn:
            existing = conn.execute(
                "SELECT total_uses, success_count, confidence, vetoed FROM edit_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()

            if existing:
                # 更新统计（不对称否决）
                total = int(existing["total_uses"]) + 1
                succ = int(existing["success_count"]) + (1 if verdict == "success" else 0)
                # 不对称置信度：失败增长更快
                if verdict == "success":
                    confidence = min(0.95, succ / (total ** 0.8)) if total else 0.5
                else:
                    failures = total - succ
                    confidence = min(0.98, failures / (total ** 0.5)) if total else 0.5
                # 不对称否决：失败置信度 > 0.85 时标记 vetoed
                vetoed = 1 if (verdict != "success" and confidence > 0.85) else int(existing["vetoed"])
                conn.execute(
                    """
                    UPDATE edit_patterns
                    SET total_uses = ?, success_count = ?, confidence = ?, vetoed = ?,
                        child_ic = ?, delta_ic = ?, child_icir = ?, delta_icir = ?,
                        verdict = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (total, succ, round(confidence, 4), vetoed,
                     child_ic, delta_ic, child_icir, delta_icir,
                     verdict, now, pattern_id),
                )
            else:
                # 新建
                confidence = 0.5
                vetoed = 0
                total = 1
                succ = 1 if verdict == "success" else 0
                conn.execute(
                    """
                    INSERT INTO edit_patterns
                        (id, parent_expression, child_expression, parent_fingerprint, child_fingerprint,
                         edit_type, edit_detail_json,
                         parent_ic, child_ic, delta_ic, parent_icir, child_icir, delta_icir,
                         verdict, confidence, total_uses, success_count, vetoed, family,
                         parent_entry_id, child_entry_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pattern_id, parent_expression, child_expression,
                     parent_struct.get("fingerprint"), child_struct.get("fingerprint"),
                     edit_type, edit_detail_json,
                     parent_ic, child_ic, delta_ic, parent_icir, child_icir, delta_icir,
                     verdict, round(confidence, 4), total, succ, vetoed, family,
                     parent_id, child_id, now, now),
                )

        return pattern_id

    def query_edit_patterns(
        self,
        *,
        edit_type: str | None = None,
        verdict: str | None = None,
        family: str | None = None,
        min_confidence: float = 0.3,
        exclude_vetoed: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """检索编辑模式，按置信度降序。"""
        clauses = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if edit_type:
            clauses.append("edit_type = ?")
            params.append(edit_type)
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        if family:
            clauses.append("family = ?")
            params.append(family)
        if exclude_vetoed:
            clauses.append("vetoed = 0")
        where = " AND ".join(clauses)
        with self._open() as conn:
            rows = conn.execute(
                f"SELECT * FROM edit_patterns WHERE {where} "
                f"ORDER BY confidence DESC, total_uses DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "edit_type": r["edit_type"],
                "edit_detail": json.loads(r["edit_detail_json"] or "{}"),
                "parent_fingerprint": r["parent_fingerprint"],
                "child_fingerprint": r["child_fingerprint"],
                "parent_ic": r["parent_ic"],
                "child_ic": r["child_ic"],
                "delta_ic": r["delta_ic"],
                "parent_icir": r["parent_icir"],
                "child_icir": r["child_icir"],
                "delta_icir": r["delta_icir"],
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "total_uses": r["total_uses"],
                "success_count": r["success_count"],
                "vetoed": r["vetoed"],
                "family": r["family"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def delete_entry(self, entry_id: str) -> bool:
        with self._open() as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    @staticmethod
    def entry_signature(expr: str) -> str:
        """与 record_tool_result 一致的表达式签名（规范换行后 sha256 前 20 位）。"""
        canonical = "\n".join(line.strip() for line in (expr or "").splitlines() if line.strip())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    def purge_factor(
        self,
        *,
        factor_names: list[str] | tuple[str, ...] = (),
        expressions: list[str] | tuple[str, ...] = (),
    ) -> int:
        """删除与指定因子相关的全部记忆条目（含 FTS 与观察记录，经级联/触发器）。

        匹配规则：表达式签名精确命中，或 factor_name 精确命中。返回删除条数。
        """
        ids = {self.entry_signature(e) for e in expressions if e}
        names = [n for n in factor_names if n]
        if not ids and not names:
            return 0
        deleted = 0
        with self._open() as conn:
            for eid in ids:
                cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (eid,))
                deleted += cursor.rowcount
            for name in names:
                cursor = conn.execute("DELETE FROM memory_entries WHERE factor_name = ?", (name,))
                deleted += cursor.rowcount
        return deleted

    def context_for(
        self,
        research_goal: str,
        *,
        limit: int = 12,
        include_rejected: bool = True,
        prefer_orthogonal: bool = True,
        include_expression: bool = True,
        max_expression_chars: int | None = None,
        enable_factor_retrieval: bool = False,
        enable_edit_patterns: bool = False,
    ) -> str:
        """Build a compact, retrieval-based research memory context block.

        四层记忆架构（渐进式模块化）：
        0. **编辑模式层**（``enable_edit_patterns=True`` 时启用）
           — AlphaMemo SSPM 对标：从历史编辑模式中提取"哪种编辑操作在什么上下文
           里好/不好"，按置信门控注入（硬推荐/硬否决/软推荐/软否决/残差不注入）。
        1. **单因子检索层**（``enable_factor_retrieval=True`` 时启用）
           — BM25 + recency 混合检索，分 positive/negative 两池展示。
           默认关闭，因逐因子记录中 46% 是同一句话，信息密度低；
           需要时可经 ``memory_policy.enable_factor_retrieval`` 开启。
        2. **模式层**（始终启用）
           — 跨因子经验提炼，全量按置信度注入，不走 BM25。
        3. **饱和度层**（始终启用）
           — 因子族拥挤度，> 0.6 时注入警告。
        """
        lines: list[str] = []
        header_added = False

        def _ensure_header() -> None:
            nonlocal header_added
            if not header_added:
                lines.append("# 长期研究记忆（来自真实评估与提交结果）")
                lines.append("以下结论必须作为实验先验。")
                header_added = True

        # ── ⓪ 编辑模式层（AlphaMemo SSPM，置信门控残差记忆）──
        if enable_edit_patterns:
            edit_block = self._edit_pattern_block()
            if edit_block:
                _ensure_header()
                lines.append(edit_block)

        # ── ① 单因子检索层（可开关，默认关闭）──
        if enable_factor_retrieval:
            factor_lines = self._factor_retrieval_block(
                research_goal,
                limit=limit,
                include_rejected=include_rejected,
                prefer_orthogonal=prefer_orthogonal,
                include_expression=include_expression,
                max_expression_chars=max_expression_chars,
            )
            if factor_lines:
                _ensure_header()
                lines.append(factor_lines)

        # ── ② 模式层（跨因子经验提炼，全量按置信度注入，不走 BM25）──
        patterns = self.query_patterns(min_confidence=0.3, limit=10)
        if patterns:
            _ensure_header()
            lines.append("")
            lines.append("## 研究模式记忆（跨因子经验提炼）")
            lines.append("以下模式来自历史多轮挖掘的统计提炼，优先级高于单因子记忆。")

            recommends = [p for p in patterns if p["layer"] == "recommend" and (p.get("success_rate") or 0) >= 0]
            forbids = [p for p in patterns if p["layer"] == "forbid"]
            insights = [p for p in patterns if p["layer"] == "insight"]

            if recommends:
                lines.append("")
                lines.append("### 推荐方向（成功率 > 0，在其邻近空间继续探索）")
                for p in recommends:
                    rate = p.get("success_rate") or 0
                    conf = p.get("confidence") or 0
                    lines.append(
                        f"- [{p['category']}] {p['content']} "
                        f"(成功率 {rate:.0%}，置信度 {conf:.0%})"
                    )

            if forbids:
                lines.append("")
                lines.append("### 禁止方向（已验证无效，除非改变核心机制否则不要重复）")
                for p in forbids:
                    conf = p.get("confidence") or 0
                    n = p.get("total_attempts") or 0
                    lines.append(
                        f"- [{p['category']}] {p['content']} "
                        f"(已尝试 {n} 次，置信度 {conf:.0%})"
                    )

            if insights:
                lines.append("")
                lines.append("### 战略洞察")
                for p in insights:
                    lines.append(f"- {p['content']}")

        # ── ③ 饱和度层 ──
        saturation = self.compute_saturation()
        crowded = {
            f: d for f, d in saturation.items()
            if d.get("saturation_score", 0) > 0.6
        }
        if crowded:
            _ensure_header()
            lines.append("")
            lines.append("### 饱和度警告")
            lines.append("以下因子族已拥挤（多个相似因子入库），继续微调边际收益低：")
            for family, data in sorted(crowded.items(), key=lambda x: -x[1].get("saturation_score", 0)):
                lines.append(
                    f"- {family}: {int(data['n_promising'])} 个有潜力 + "
                    f"{int(data['n_validated'])} 个已验证，"
                    f"饱和度 {data['saturation_score']:.0%}"
                )
            lines.append("建议切换到饱和度 < 0.3 的未探索族。")

        return "\n".join(lines)

    # ── 编辑模式注入（AlphaMemo SSPM 对标）──

    def _edit_pattern_block(self) -> str:
        """构建编辑模式注入块，按置信门控分四级注入。

        置信门控残差记忆：
        - > 0.85  : 硬推荐 / 硬否决（vetoed）
        - 0.5~0.85: 软推荐 / 软否决
        - 0.3~0.5 : 弱提示
        - < 0.3   : 不注入（残差记忆，保留在库中继续积累统计）
        """
        # 检索成功模式（非 vetoed）
        successes = self.query_edit_patterns(
            verdict="success",
            min_confidence=0.3,
            exclude_vetoed=True,
            limit=10,
        )
        # 检索失败模式（含 vetoed）
        failures = self.query_edit_patterns(
            verdict="failure",
            min_confidence=0.3,
            exclude_vetoed=False,
            limit=10,
        )

        if not successes and not failures:
            return ""

        block_lines: list[str] = []
        block_lines.append("")
        block_lines.append("## 编辑模式记忆（基于历史编辑操作统计）")
        block_lines.append("以下模式记录了"在特定信号上下文中，哪种编辑操作有效或失败"。")

        # ── 成功模式（硬推荐 + 软推荐 + 弱提示）──
        if successes:
            hard_rec = [p for p in successes if p["confidence"] > 0.85]
            soft_rec = [p for p in successes if 0.5 < p["confidence"] <= 0.85]
            weak_rec = [p for p in successes if 0.3 < p["confidence"] <= 0.5]

            if hard_rec:
                block_lines.append("")
                block_lines.append("### 优先尝试（高置信度有效编辑，>85%）")
                for p in hard_rec:
                    block_lines.append(self._format_edit_pattern(p, prefix="优先尝试"))

            if soft_rec:
                block_lines.append("")
                block_lines.append("### 可以尝试（中等置信度有效编辑，50%~85%）")
                for p in soft_rec:
                    block_lines.append(self._format_edit_pattern(p, prefix="可以尝试"))

            if weak_rec:
                block_lines.append("")
                block_lines.append("### 可参考（低置信度有效编辑，30%~50%）")
                for p in weak_rec:
                    block_lines.append(self._format_edit_pattern(p, prefix="可参考"))

        # ── 失败模式（硬否决 + 软否决 + 弱提示）──
        if failures:
            hard_veto = [p for p in failures if p["confidence"] > 0.85 or p.get("vetoed")]
            soft_veto = [p for p in failures if 0.5 < p["confidence"] <= 0.85 and not p.get("vetoed")]
            weak_veto = [p for p in failures if 0.3 < p["confidence"] <= 0.5 and not p.get("vetoed")]

            if hard_veto:
                block_lines.append("")
                block_lines.append("### 不要尝试（高置信度无效编辑，>85%，强否决）")
                for p in hard_veto:
                    block_lines.append(self._format_edit_pattern(p, prefix="不要尝试"))

            if soft_veto:
                block_lines.append("")
                block_lines.append("### 谨慎尝试（中等置信度无效编辑，50%~85%）")
                for p in soft_veto:
                    block_lines.append(self._format_edit_pattern(p, prefix="谨慎尝试"))

            if weak_veto:
                block_lines.append("")
                block_lines.append("### 注意（低置信度无效编辑，30%~50%）")
                for p in weak_veto:
                    block_lines.append(self._format_edit_pattern(p, prefix="注意"))

        return "\n".join(block_lines)

    @staticmethod
    def _format_edit_pattern(p: dict[str, Any], *, prefix: str) -> str:
        """格式化单条编辑模式为注入文本。"""
        detail = p.get("edit_detail") or {}
        detail_str = ""
        if isinstance(detail, dict):
            parts = []
            for k, v in detail.items():
                if v is not None:
                    parts.append(f"{k}={v}")
            detail_str = ", ".join(parts) if parts else ""
        elif isinstance(detail, str) and detail:
            detail_str = detail

        delta_ic = p.get("delta_ic")
        delta_str = f"ΔIC={delta_ic:+.4f}" if delta_ic is not None else ""

        total = p.get("total_uses") or 0
        succ = p.get("success_count") or 0
        conf = p.get("confidence") or 0
        family = p.get("family") or "unknown"

        line = (
            f"- [{p['edit_type']}] {prefix}："
            f"在 {family} 信号上"
        )
        if detail_str:
            line += f"（{detail_str}）"
        line += f"。{delta_str}"
        if total > 0:
            line += f"，{succ}/{total} 次有效"
        line += f"，置信度 {conf:.0%}"
        return line

    def _factor_retrieval_block(
        self,
        research_goal: str,
        *,
        limit: int = 12,
        include_rejected: bool = True,
        prefer_orthogonal: bool = True,
        include_expression: bool = True,
        max_expression_chars: int | None = None,
    ) -> str:
        """单因子 BM25 检索段（模块化抽取，可经 enable_factor_retrieval 开关）。

        Retrieval uses a local BM25 scoring over each entry's token set (zero
        API cost).  Entries are split into **positive** (validated, promising…)
        and **negative** (rejected, weak…) pools, scored independently, then
        merged with a guaranteed minimum quota for positive entries — so that
        known-good factor families always surface even when negatives vastly
        outnumber them (which is the common case after many dead-end attempts).
        """
        entries = self._retrieval_candidates(research_goal, include_rejected)
        if not include_rejected:
            entries = [entry for entry in entries if entry.get("verdict") not in self._NEGATIVE_VERDICTS]
        if not entries:
            return ""

        verdict_rank = {
            "production_approved": 5, "validated": 4, "candidate_approved": 3,
            "promising": 2, "revise_required": 1, "rejected": 1, "weak": 0,
        }

        def _key(entry: dict[str, Any], bm: float) -> tuple[float, int, int, str]:
            observations = entry.get("observations", [])
            n = observations if isinstance(observations, int) else len(observations)
            return (bm, verdict_rank.get(str(entry.get("verdict")), 0), n, str(entry.get("updated_at", "")))

        positive_pool = [(e, float(e.pop("_bm25", 0.0))) for e in entries if e.get("verdict") in self._POSITIVE_VERDICTS]
        negative_pool = [(e, float(e.pop("_bm25", 0.0))) for e in entries if e.get("verdict") in self._NEGATIVE_VERDICTS]

        positive_pool.sort(key=lambda pair: _key(pair[0], pair[1]), reverse=True)
        negative_pool.sort(key=lambda pair: _key(pair[0], pair[1]), reverse=True)

        positive_quota = max(1, int(limit * 0.4)) if positive_pool else 0
        positive_relevant = [(e, b) for e, b in positive_pool if b > 0]
        if not positive_relevant and positive_pool:
            positive_relevant = positive_pool[:positive_quota]
        positive_selected = positive_relevant[:positive_quota]
        n_pos = len(positive_selected)
        n_neg = limit - n_pos
        negative_selected = negative_pool[:n_neg]
        if n_pos < positive_quota:
            negative_selected = negative_pool[:limit - n_pos]

        positive = positive_selected
        negative = negative_selected

        block_lines: list[str] = []

        # ── 肯定段 ──
        if positive:
            block_lines.append("")
            block_lines.append("## 已验证 / 有潜力的因子（优先在其邻近空间继续挖掘相似机制）")
            block_lines.append(
                "这些因子在训练集或验证集上表现可用。**鼓励**基于它们的经济逻辑，"
                "通过更换窗口、算子族或原始字段，在相似但不重复的方向上继续探索。"
            )
            if prefer_orthogonal:
                block_lines.append("扩展时优先引入正交变量，避免仅改窗口长度的同质微调。")
            for entry, _ in positive:
                block_lines.append(self._format_entry(entry, include_expression, max_expression_chars))

        # ── 否定段 ──
        if negative:
            block_lines.append("")
            block_lines.append("## 已否定 / 不足的因子（避免机械重复同一死路）")
            block_lines.append(
                "以下路径已被评估否定。除非改变了核心变量、经济机制或处理方式，"
                "否则不要重复尝试相同结构。"
            )
            for entry, _ in negative:
                block_lines.append(self._format_entry(entry, include_expression, max_expression_chars))

        return "\n".join(block_lines)

    @staticmethod
    def _fts_match_query(tokens: set[str]) -> str:
        return " OR ".join(
            '"' + str(token).replace('"', '""') + '"'
            for token in sorted(tokens)
        )

    def _retrieval_candidates(
        self,
        research_goal: str,
        include_rejected: bool,
        *,
        scan_limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Fetch a small FTS-ranked plus recency candidate set, never all history."""
        filters = "" if include_rejected else (
            f"AND e.verdict NOT IN ({','.join('?' for _ in self._NEGATIVE_VERDICTS)})"
        )
        filter_args = tuple() if include_rejected else tuple(sorted(self._NEGATIVE_VERDICTS))
        candidates: dict[str, dict[str, Any]] = {}

        with self._open() as conn:
            recent_rows = conn.execute(
                f"""
                SELECT e.*, 0.0 AS relevance
                FROM memory_entries e
                WHERE 1=1 {filters}
                ORDER BY e.updated_at DESC
                LIMIT ?
                """,
                (*filter_args, int(scan_limit)),
            ).fetchall()
            hydrated_recent = self._hydrate_entries(conn, recent_rows)
            for row, entry in zip(recent_rows, hydrated_recent):
                entry["_bm25"] = float(row["relevance"])
                candidates[entry["id"]] = entry

            tokens = set(_tokens(research_goal))
            match_query = self._fts_match_query(tokens)
            if match_query:
                matched_rows = conn.execute(
                    f"""
                    SELECT e.*, -bm25(memory_fts) AS relevance
                    FROM memory_fts
                    JOIN memory_entries e ON e.id = memory_fts.entry_id
                    WHERE memory_fts MATCH ? {filters}
                    ORDER BY relevance DESC
                    LIMIT ?
                    """,
                    (match_query, *filter_args, int(scan_limit)),
                ).fetchall()
                hydrated_matched = self._hydrate_entries(conn, matched_rows)
                for row, entry in zip(matched_rows, hydrated_matched):
                    existing = candidates.get(entry["id"])
                    entry["_bm25"] = float(row["relevance"])
                    if existing is None:
                        candidates[entry["id"]] = entry
                    else:
                        existing["_bm25"] = max(
                            float(existing.get("_bm25", 0.0)),
                            float(row["relevance"]),
                        )

        return list(candidates.values())

    def query_for_attempts(
        self,
        research_goal: str,
        recent_rows: list[dict[str, Any]],
        *,
        max_recent_attempts: int = 8,
        max_expression_chars: int = 1200,
    ) -> str:
        """Build a retrieval query from the goal plus the current run's latest work."""
        parts = [str(research_goal or "A股日频因子挖掘")]
        for row in list(recent_rows)[-max(0, int(max_recent_attempts)):]:
            args = _parse_args(row.get("arguments_raw"))
            factor_name = str(args.get("factor_name") or "").strip()
            expression = str(args.get("multi_line_expr") or "").strip()
            error = str(row.get("error") or "").strip()
            if factor_name:
                parts.append(factor_name)
            if expression:
                parts.append(expression[:max_expression_chars])
            if error:
                parts.append(error[:300])
            interaction = args.get("interaction")
            if isinstance(interaction, dict):
                parts.append(str(interaction.get("interaction_type") or ""))
                parts.append(str(interaction.get("base_signal") or ""))
                parts.append(str(interaction.get("condition_signal") or ""))
                parts.append(str(interaction.get("economic_mechanism") or ""))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _format_entry(
        entry: dict[str, Any],
        include_expression: bool,
        max_expression_chars: int | None = None,
    ) -> str:
        m = entry.get("metrics", {})
        metric_text = " ".join(
            f"{key}={value:.4g}" for key, value in m.items() if isinstance(value, (int, float))
        )
        interaction = entry.get("interaction") if isinstance(entry.get("interaction"), dict) else {}
        interaction_text = ""
        if interaction:
            interaction_text = f"；交互类型={interaction.get('interaction_type')}；条件={interaction.get('condition_signal')}"
        expr = entry.get("expression")
        if include_expression and expr and max_expression_chars is not None:
            text = str(expr)
            if int(max_expression_chars) <= 0:
                expr = None
            elif len(text) > int(max_expression_chars):
                text = text[:max(int(max_expression_chars) - 3, 0)] + "..."
            expr = text
        expr_tail = f"；表达式：{expr}" if (include_expression and expr) else ""
        return (
            f"- [{entry.get('verdict')}] {entry.get('factor_name')}: {entry.get('conclusion')} "
            f"指标({metric_text or '无'}){interaction_text}{expr_tail}"
        )

    # Verdict display order — positive first, then negative.
    _VERDICT_ORDER = {
        "production_approved": 0,
        "validated": 1,
        "candidate_approved": 2,
        "promising": 3,
        "revise_required": 4,
        "rejected": 5,
        "weak": 6,
    }

    def recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return entries sorted by verdict priority (positive first) then
        recency, so the frontend always shows validated/promising factors
        even when recent runs produced mostly rejections.
        """
        case_parts = " ".join(
            f"WHEN '{verdict}' THEN {rank}"
            for verdict, rank in self._VERDICT_ORDER.items()
        )
        with self._open() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_entries
                ORDER BY CASE verdict {case_parts} ELSE 99 END,
                         updated_at DESC
                LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()
            return self._hydrate_entries(conn, rows)

    def statistics(self) -> dict[str, Any]:
        with self._open() as conn:
            entry_count = int(conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0])
            obs_rows = conn.execute(
                """
                SELECT e.verdict AS verdict, COUNT(*) AS n
                FROM memory_observations o
                JOIN memory_entries e ON e.id = o.entry_id
                GROUP BY e.verdict
                """
            ).fetchall()
        counts = {str(row["verdict"] or "unknown"): int(row["n"]) for row in obs_rows}
        attempts = sum(counts.values())
        return {
            "entries": entry_count,
            "observations": attempts,
            "verdict_counts": counts,
            "train_to_validated_rate": round(counts.get("validated", 0) / attempts, 4) if attempts else None,
            "production_rate": round(counts.get("production_approved", 0) / attempts, 4) if attempts else None,
        }

    def backfill_from_logs(self, log_root: Path) -> int:
        """Populate an empty memory store from prior UI JSONL event logs once.

        以 store_meta.backfill_done 标记防止「删空记忆后重启又被全量复活」：
        只要标记存在（无论库内是否还有条目），就不再回放历史日志。
        """
        with self._open() as conn:
            existing = conn.execute("SELECT 1 FROM memory_entries LIMIT 1").fetchone()
            done = conn.execute("SELECT v FROM store_meta WHERE k = 'backfill_done'").fetchone()
        if existing or done:
            if not done:
                with self._open() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done', '1')"
                    )
            return 0
        count = 0
        for log_path in sorted(Path(log_root).glob("*/run_*.jsonl")):
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "tool_results":
                    continue
                for row in event.get("results") or []:
                    if isinstance(row, dict) and self.record_tool_result(run_id=log_path.parent.name, row=row):
                        count += 1
        with self._open() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done', '1')"
            )
        return count

    # ──────────────────────────────────────────────────────────────────────
    # 改进一：模式层记忆 CRUD
    # ──────────────────────────────────────────────────────────────────────

    def record_pattern(
        self,
        *,
        layer: str,
        category: str,
        content: str,
        evidence: dict[str, Any] | None = None,
        total_attempts: int = 1,
        success_count: int = 0,
    ) -> str:
        """写入或更新一条模式记忆。

        ``layer`` ∈ {"recommend", "forbid", "insight"}。
        同 ``layer|category|content`` 签名去重；已存在则 total_attempts/success_count 累加。
        """
        if layer not in ("recommend", "forbid", "insight"):
            raise ValueError(f"invalid layer: {layer}")
        pattern_id = hashlib.sha256(
            f"{layer}|{category}|{content}".encode("utf-8")
        ).hexdigest()[:20]
        now = _now()
        evidence_json = json.dumps(evidence or {}, ensure_ascii=False, separators=(",", ":"))
        with self._open() as conn:
            existing = conn.execute(
                "SELECT total_attempts, success_count FROM memory_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            if existing:
                new_total = int(existing["total_attempts"]) + total_attempts
                new_succ = int(existing["success_count"]) + success_count
                rate = new_succ / new_total if new_total else 0.0
                conf = max(0.1, min(0.95, 1.0 - 1.0 / (new_total ** 0.5)))
                conn.execute(
                    """
                    UPDATE memory_patterns
                    SET evidence_json = ?, total_attempts = ?, success_count = ?,
                        success_rate = ?, confidence = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (evidence_json, new_total, new_succ, round(rate, 4), round(conf, 4), now, pattern_id),
                )
            else:
                rate = success_count / total_attempts if total_attempts else 0.0
                conf = max(0.1, min(0.95, 1.0 - 1.0 / (total_attempts ** 0.5))) if total_attempts else 0.5
                conn.execute(
                    """
                    INSERT INTO memory_patterns
                        (id, layer, category, content, evidence_json,
                         total_attempts, success_count, success_rate, saturation_score,
                         confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (pattern_id, layer, category, content, evidence_json,
                     total_attempts, success_count, round(rate, 4), round(conf, 4), now, now),
                )
        return pattern_id

    def update_pattern_stats(self, pattern_id: str, *, success: bool) -> None:
        """评估结果返回后更新模式的成功/失败计数和置信度。"""
        with self._open() as conn:
            row = conn.execute(
                "SELECT total_attempts, success_count FROM memory_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            if not row:
                return
            total = int(row["total_attempts"]) + 1
            succ = int(row["success_count"]) + (1 if success else 0)
            rate = succ / total if total else 0.0
            conf = max(0.1, min(0.95, 1.0 - 1.0 / (total ** 0.5)))
            conn.execute(
                """
                UPDATE memory_patterns
                SET total_attempts = ?, success_count = ?, success_rate = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (total, succ, round(rate, 4), round(conf, 4), _now(), pattern_id),
            )

    def query_patterns(
        self,
        *,
        layer: str | None = None,
        category: str | None = None,
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """检索模式记忆，按置信度降序。"""
        clauses = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if layer:
            clauses.append("layer = ?")
            params.append(layer)
        if category:
            clauses.append("category = ?")
            params.append(category)
        where = " AND ".join(clauses)
        with self._open() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_patterns WHERE {where} "
                f"ORDER BY confidence DESC, success_rate DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "layer": r["layer"],
                "category": r["category"],
                "content": r["content"],
                "evidence": json.loads(r["evidence_json"] or "{}"),
                "success_rate": r["success_rate"],
                "total_attempts": r["total_attempts"],
                "success_count": r["success_count"],
                "saturation_score": r["saturation_score"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    # ──────────────────────────────────────────────────────────────────────
    # 改进二：每批蒸馏算子（规则式，零 LLM 成本）
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _classify_family(cls, factor_name: str, expression: str) -> str:
        """根据因子名和表达式启发式分类到信号族。"""
        text = (str(factor_name or "") + " " + str(expression or "")).lower()
        for family, keywords in cls._FAMILY_RULES.items():
            if any(kw in text for kw in keywords):
                return family
        return "other"

    def _group_by_family(self, results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        families: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            family = self._classify_family(
                str(r.get("factor_name", "")),
                str(r.get("expression", "")),
            )
            families.setdefault(family, []).append(r)
        return families

    def distill_batch_patterns(
        self,
        *,
        run_id: str,
        turn: int,
        batch_results: list[dict[str, Any]],
    ) -> list[str]:
        """从一批评估结果中蒸馏模式记忆，返回写入的 pattern_ids。

        Phase 2 改进：使用表达式结构信息生成结构化模式，
        而非仅看 IC 绝对值套模板。
        1. 解析每个因子的结构（算子/变量/窗口参数/指纹）
        2. 按信号族 + 结构相似度分组
        3. 生成包含结构信息的 recommend/forbid 模式
        4. 全局洞察保留但补充结构统计
        """
        pattern_ids: list[str] = []
        if not batch_results:
            return pattern_ids

        # 解析每个因子的结构
        enriched: list[dict[str, Any]] = []
        for r in batch_results:
            expr = str(r.get("expression", ""))
            struct = _parse_expression_structure(expr)
            enriched.append({
                **r,
                "_struct": struct,
                "_family": self._classify_family(str(r.get("factor_name", "")), expr),
            })

        families = self._group_by_family(enriched)

        for family_name, members in families.items():
            ics = [
                abs(float(m.get("metrics", {}).get("ic", 0) or 0))
                for m in members
            ]
            n = len(members)
            if n < 2:
                continue

            all_weak = all(ic < 0.01 for ic in ics)
            any_promising = any(ic >= 0.02 for ic in ics)

            # 收集结构统计
            op_counts: dict[str, int] = {}
            var_counts: dict[str, int] = {}
            window_set: set[str] = set()
            for m in members:
                struct = m.get("_struct", {})
                for op in struct.get("operators", []):
                    op_counts[op] = op_counts.get(op, 0) + 1
                for var in struct.get("variables", []):
                    var_counts[var] = var_counts.get(var, 0) + 1
                for op, wins in struct.get("window_params", {}).items():
                    for w in wins:
                        window_set.add(f"{op}:{w}")

            if all_weak and n >= 3:
                # 结构化 forbid 模式：包含具体算子和窗口信息
                top_ops = sorted(op_counts.items(), key=lambda x: -x[1])[:3]
                top_vars = sorted(var_counts.items(), key=lambda x: -x[1])[:3]
                ops_str = ", ".join(f"{op}({cnt})" for op, cnt in top_ops) if top_ops else "无"
                vars_str = ", ".join(f"{var}({cnt})" for var, cnt in top_vars) if top_vars else "无"
                content = (
                    f"{family_name} 信号族在 {n} 次尝试中 IC 均低于 0.01"
                    f"（最高 {max(ics):.4f}）。"
                    f"常用算子：{ops_str}；常用变量：{vars_str}；"
                    f"窗口：{', '.join(sorted(window_set)[:5]) if window_set else '无'}。"
                    f"该族在当前数据和 label 下可能已饱和，"
                    f"除非引入新变量或交互机制，否则不建议机械重复。"
                )
                pid = self.record_pattern(
                    layer="forbid",
                    category=family_name,
                    content=content,
                    total_attempts=n,
                    success_count=n,  # forbid 模式：全部 IC<0.01 验证了禁止方向正确
                    evidence={
                        "factor_names": [m.get("factor_name") for m in members],
                        "ic_range": [round(min(ics), 6), round(max(ics), 6)],
                        "n_attempts": n,
                        "top_operators": top_ops,
                        "top_variables": top_vars,
                        "windows": sorted(window_set)[:10],
                        "run_id": run_id,
                        "turn": turn,
                    },
                )
                pattern_ids.append(pid)

            if any_promising:
                best = max(
                    members,
                    key=lambda m: abs(float(m.get("metrics", {}).get("ic", 0) or 0)),
                )
                best_ic = float(best.get("metrics", {}).get("ic", 0) or 0)
                best_struct = best.get("_struct", {})
                best_ops = best_struct.get("operators", [])
                best_vars = best_struct.get("variables", [])
                best_wins = best_struct.get("window_params", {})

                # 结构化 recommend 模式：包含最佳因子的结构信息
                ops_str = ", ".join(best_ops[:5]) if best_ops else "无"
                vars_str = ", ".join(best_vars[:3]) if best_vars else "无"
                wins_str = "; ".join(
                    f"{op}: {vals}" for op, vals in best_wins.items()
                ) if best_wins else "无"

                content = (
                    f"{family_name} 信号族中有因子 IC 达 {best_ic:+.4f}。"
                    f"有效结构：算子[{ops_str}]，变量[{vars_str}]，窗口[{wins_str}]。"
                    f"建议在邻近空间变异：扩展/缩减窗口、替换算子、引入正交变量。"
                )
                # 统计本批次有效因子数（IC >= 0.02）
                n_success = sum(1 for ic in ics if ic >= 0.02)
                pid = self.record_pattern(
                    layer="recommend",
                    category=family_name,
                    content=content,
                    total_attempts=n,
                    success_count=n_success,
                    evidence={
                        "best_factor": best.get("factor_name"),
                        "best_ic": round(best_ic, 6),
                        "best_operators": best_ops,
                        "best_variables": best_vars,
                        "best_windows": best_wins,
                        "n_attempts": n,
                        "run_id": run_id,
                        "turn": turn,
                    },
                )
                pattern_ids.append(pid)

        # 全局洞察（补充结构统计）
        all_ics = [
            abs(float(r.get("metrics", {}).get("ic", 0) or 0))
            for r in batch_results
        ]
        if len(all_ics) >= 5 and max(all_ics) < 0.005:
            # 统计本批次的结构多样性
            all_fingerprints = {r.get("_struct", {}).get("fingerprint", "") for r in enriched}
            all_ops: set[str] = set()
            all_vars: set[str] = set()
            for r in enriched:
                struct = r.get("_struct", {})
                all_ops.update(struct.get("operators", []))
                all_vars.update(struct.get("variables", []))

            content = (
                f"连续 {len(all_ics)} 个因子 IC 均低于 0.005"
                f"（{len(all_fingerprints)} 种不同结构，"
                f"涉及 {len(all_ops)} 种算子、{len(all_vars)} 种变量）。"
                f"当前数据/label 组合下 alpha 可能极度稀薄。"
                f"建议：切换 label 列、扩展数据源、或尝试交互信号。"
            )
            pid = self.record_pattern(
                layer="insight",
                category="global",
                content=content,
                evidence={
                    "n_consecutive": len(all_ics),
                    "max_ic": round(max(all_ics), 6),
                    "n_distinct_structures": len(all_fingerprints),
                    "n_operators": len(all_ops),
                    "n_variables": len(all_vars),
                    "run_id": run_id,
                    "turn": turn,
                },
            )
            pattern_ids.append(pid)

        return pattern_ids

    # ──────────────────────────────────────────────────────────────────────
    # 改进四：饱和度跟踪
    # ──────────────────────────────────────────────────────────────────────

    def compute_saturation(self) -> dict[str, dict[str, float]]:
        """计算各因子族的饱和度。

        saturation_score = min(1.0, n_promising / 5 + n_validated / 3)
        > 0.6 时标记为"拥挤"，建议切换方向。
        """
        families: dict[str, dict[str, float]] = {}
        with self._open() as conn:
            rows = conn.execute(
                "SELECT factor_name, expression, verdict FROM memory_entries"
            ).fetchall()

        for row in rows:
            family = self._classify_family(row["factor_name"], row["expression"])
            fam = families.setdefault(
                family, {"n_entries": 0, "n_promising": 0, "n_validated": 0}
            )
            fam["n_entries"] += 1
            if row["verdict"] in ("promising", "candidate_approved"):
                fam["n_promising"] += 1
            if row["verdict"] in ("validated", "production_approved"):
                fam["n_validated"] += 1

        for fam_data in families.values():
            n_p = fam_data["n_promising"]
            n_v = fam_data["n_validated"]
            fam_data["saturation_score"] = round(min(1.0, n_p / 5.0 + n_v / 3.0), 3)

        return families

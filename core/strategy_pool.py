"""策略池：全量池（注册表 + 回测归档）与配置池（用户精选）。

全量池 = strategies.registry.STRATEGIES ∪ backtest_runs 里出现过的策略。
配置池 = strategy_pool 表，回测/模拟盘/信号页的策略下拉框以配置池优先。
PG 不可用时配置池自动回退到注册表全量，不影响原有功能。
"""
from __future__ import annotations

import json

import pandas as pd

from . import sqldb as pg
from .strategy_types import StrategyDefinition, from_legacy_dict
from strategies.registry import STRATEGIES, list_strategies


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_pool (
    id          BIGSERIAL PRIMARY KEY,
    name        VARCHAR(64) NOT NULL UNIQUE,
    factor      VARCHAR(64) NOT NULL,
    ascending   BOOLEAN NOT NULL,
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    group_name  VARCHAR(32),
    description TEXT,
    source      VARCHAR(16) NOT NULL DEFAULT 'registry',
    sort_order  INT NOT NULL DEFAULT 0,
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at  TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS strategy_trash (
    name        VARCHAR(64) PRIMARY KEY,
    factor      VARCHAR(64),
    ascending   BOOLEAN,
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    group_name  VARCHAR(32),
    description TEXT,
    source      VARCHAR(16),
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# 老库迁移（幂等）
_ensure_schema_sql = SCHEMA_SQL + """
ALTER TABLE strategy_pool ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
"""


def _ensure_schema() -> None:
    if pg.configured():
        try:
            pg.exec_sql(_ensure_schema_sql)
        except Exception:
            pass


def _trashed_names() -> set[str]:
    """已从全量池删除（回收站）的策略名。"""
    if not pg.configured():
        return set()
    try:
        df = pg.query_df("SELECT name FROM strategy_trash")
        return set(df["name"].tolist()) if not df.empty else set()
    except Exception:
        return set()


def _loads(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return v or {}


def _fmt_metric(row, key: str) -> float | None:
    m = _loads(row.get("last_metrics"))
    if not isinstance(m, dict):
        return None
    aliases = {
        "年化收益": ("年化收益", "年化"),
        "总收益": ("总收益", "收益"),
        "夏普": ("夏普",),
        "最大回撤": ("最大回撤",),
    }
    v = None
    for cand in aliases.get(key, (key,)):
        if m.get(cand) is not None:
            v = m[cand]
            break
    if v is None:
        return None
    try:
        fv = float(v)
        return None if pd.isna(fv) else fv
    except (TypeError, ValueError):
        return None


def _sweep_label(p: dict) -> str | None:
    """参数扫描记录没有 strategy 名时，从参数生成一个可读名称。"""
    factor = p.get("factor", "")
    fast, slow = p.get("fast"), p.get("slow")
    if fast is None or slow is None:
        return None
    base = f"MA{fast}/{slow}"
    adx = p.get("adx_filter")
    if adx:
        base += f"+ADX{adx}"
    uni = p.get("universe") or ""
    if uni:
        base = f"{uni}·{base}"
    return base


def archive_pool() -> pd.DataFrame:
    """回测归档里出现过的策略（按策略名聚合，取最近一次参数与指标）。"""
    if not pg.configured():
        return pd.DataFrame(columns=[
            "name", "factor", "ascending", "universe", "n_runs", "last_run_at",
            "last_metrics", "params_summary",
        ])
    sql = """
        SELECT
            COALESCE(NULLIF(params->>'strategy', ''), '') AS strat_name,
            params->>'factor' AS factor,
            params->>'ascending' AS ascending_str,
            params->>'universe' AS universe,
            params->>'fast' AS fast,
            params->>'slow' AS slow,
            params->>'adx_filter' AS adx_filter,
            params->>'freq' AS freq,
            params->>'top_n' AS top_n,
            params->>'capital' AS capital,
            params->>'min_history' AS min_history,
            created_at, metrics
        FROM backtest_runs
        ORDER BY created_at DESC
    """
    df = pg.query_df(sql)
    if df.empty:
        return df

    groups: dict[str, dict] = {}
    for _, r in df.iterrows():
        p = {k: r[k] for k in (
            "fast", "slow", "adx_filter", "freq", "top_n", "capital",
            "min_history", "universe",
        ) if r[k] not in (None, "")}
        name = r["strat_name"] or _sweep_label(p)
        if not name:
            continue
        g = groups.setdefault(name, {
            "name": name,
            "factor": None,
            "ascending": None,
            "universe": None,
            "n_runs": 0,
            "last_run_at": None,
            "last_metrics": None,
            "params_summary": {},
        })
        g["n_runs"] += 1
        if g["last_run_at"] is None:
            g["factor"] = r["factor"]
            g["ascending"] = None if r["ascending_str"] is None \
                else str(r["ascending_str"]).lower() in ("true", "1")
            g["universe"] = r["universe"]
            g["last_run_at"] = r["created_at"]
            g["last_metrics"] = r["metrics"]
            g["params_summary"] = p
    return pd.DataFrame(list(groups.values()))


def full_pool() -> pd.DataFrame:
    """全量池：注册表策略 + 回测归档策略，附最近一次回测指标。"""
    arch = archive_pool()
    arch_by_name = {}
    if not arch.empty:
        arch_by_name = {r["name"]: r for _, r in arch.iterrows()}
    trashed = _trashed_names()

    rows = []
    for name in list_strategies():
        if name in trashed:
            continue
        s = STRATEGIES[name]
        a = arch_by_name.get(name)
        rows.append({
            "name": name,
            "factor": s.get("factor"),
            "ascending": s.get("ascending"),
            "group": s.get("group", "其他"),
            "desc": s.get("desc", ""),
            "source": "registry",
            "n_runs": int(a["n_runs"]) if a is not None else 0,
            "last_run_at": a["last_run_at"] if a is not None else None,
            "sharpe": _fmt_metric(a, "夏普") if a is not None else None,
            "annual": _fmt_metric(a, "年化收益") if a is not None else None,
            "total_return": _fmt_metric(a, "总收益") if a is not None else None,
            "mdd": _fmt_metric(a, "最大回撤") if a is not None else None,
            "universe": a["universe"] if a is not None else None,
            "params_summary": (a["params_summary"] if a is not None else {}),
        })
    if not arch.empty:
        for _, a in arch.iterrows():
            if a["name"] in STRATEGIES or a["name"] in trashed:
                continue
            rows.append({
                "name": a["name"],
                "factor": a["factor"],
                "ascending": a["ascending"],
                "group": "回测历史",
                "desc": f"回测归档 {int(a['n_runs'])} 次",
                "source": "archive",
                "n_runs": int(a["n_runs"]),
                "last_run_at": a["last_run_at"],
                "sharpe": _fmt_metric(a, "夏普"),
                "annual": _fmt_metric(a, "年化收益"),
                "total_return": _fmt_metric(a, "总收益"),
                "mdd": _fmt_metric(a, "最大回撤"),
                "universe": a["universe"],
                "params_summary": a["params_summary"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["source", "group", "name"]).reset_index(drop=True)
    return df


def pool_names() -> list[str]:
    """配置池策略名（按 sort_order, id）。"""
    _ensure_schema()
    if not pg.configured():
        return []
    df = pg.query_df(
        "SELECT name FROM strategy_pool WHERE deleted_at IS NULL "
        "ORDER BY sort_order, id")
    return [] if df.empty else df["name"].tolist()


def pool_rows() -> pd.DataFrame:
    """配置池详情，含最近回测指标。"""
    names = pool_names()
    if not names:
        return pd.DataFrame()
    fp = full_pool()
    fp_by_name = {r["name"]: r for _, r in fp.iterrows()} if not fp.empty else {}
    rows = []
    for i, name in enumerate(names):
        f = fp_by_name.get(name)
        rows.append({
            "sort": i + 1,
            "name": name,
            "factor": f["factor"] if f is not None else None,
            "ascending": f["ascending"] if f is not None else None,
            "group": (f["group"] if f is not None else "配置池"),
            "desc": (f["desc"] if f is not None else ""),
            "source": (f["source"] if f is not None else "custom"),
            "n_runs": int(f["n_runs"]) if f is not None else 0,
            "sharpe": f["sharpe"] if f is not None else None,
            "annual": f["annual"] if f is not None else None,
            "total_return": f["total_return"] if f is not None else None,
            "mdd": f["mdd"] if f is not None else None,
        })
    return pd.DataFrame(rows)


def pool_def(name: str) -> dict | None:
    """从配置池取策略定义（factor/ascending/params）。"""
    _ensure_schema()
    if not pg.configured():
        return None
    df = pg.query_df(
        "SELECT name, factor, ascending, params, group_name, description, source "
        "FROM strategy_pool WHERE name = %s AND deleted_at IS NULL", (name,))
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "name": r["name"],
        "factor": r["factor"],
        "ascending": bool(r["ascending"]),
        "params": _loads(r["params"]),
        "group": r["group_name"],
        "desc": r["description"],
        "source": r["source"],
    }


def add_from_full(name: str) -> bool:
    """把全量池策略加入配置池；已存在返回 False。"""
    _ensure_schema()
    fp = full_pool()
    hit = fp[fp["name"] == name]
    if hit.empty:
        return False
    r = hit.iloc[0]
    params = {}
    if isinstance(r.get("params_summary"), dict):
        params = {k: v for k, v in r["params_summary"].items() if v not in (None, "")}
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO strategy_pool
                (name, factor, ascending, params, group_name, description, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                deleted_at = NULL,
                sort_order = 0
            WHERE strategy_pool.deleted_at IS NOT NULL
            """,
            (name,
             str(r["factor"]) if r.get("factor") is not None else "",
             bool(r.get("ascending")) if r.get("ascending") is not None else False,
             json.dumps(params, ensure_ascii=False),
             str(r.get("group") or "其他"),
             str(r.get("desc") or ""),
             str(r.get("source") or "registry"),
            ),
        )
        return cur.rowcount > 0


def remove_from_pool(name: str) -> bool:
    """从配置池移除：只影响其他页面的策略下拉，策略仍在全量池。"""
    _ensure_schema()
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM strategy_pool WHERE name = %s", (name,))
        return cur.rowcount > 0


def delete_from_full(name: str) -> bool:
    """把全量池策略删除进回收站；若已在配置池，同步移除。"""
    _ensure_schema()
    fp = full_pool()
    hit = fp[fp["name"] == name]
    if hit.empty:
        return False
    r = hit.iloc[0]
    params = {}
    if isinstance(r.get("params_summary"), dict):
        params = {k: v for k, v in r["params_summary"].items()
                  if v not in (None, "")}
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO strategy_trash
                (name, factor, ascending, params, group_name, description, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                factor = EXCLUDED.factor,
                ascending = EXCLUDED.ascending,
                params = EXCLUDED.params,
                group_name = EXCLUDED.group_name,
                description = EXCLUDED.description,
                source = EXCLUDED.source,
                deleted_at = now()
            """,
            (name,
             str(r["factor"]) if r.get("factor") is not None else None,
             bool(r.get("ascending")) if r.get("ascending") is not None else None,
             json.dumps(params, ensure_ascii=False),
             str(r.get("group") or "其他"),
             str(r.get("desc") or ""),
             str(r.get("source") or "archive"),
            ),
        )
        cur.execute("DELETE FROM strategy_pool WHERE name = %s", (name,))
    return True


def trash_rows() -> pd.DataFrame:
    """回收站：从全量池删除的策略。"""
    _ensure_schema()
    if not pg.configured():
        return pd.DataFrame(columns=[
            "name", "factor", "group_name", "source", "deleted_at",
        ])
    df = pg.query_df(
        "SELECT name, factor, ascending, group_name, description, source, "
        "deleted_at FROM strategy_trash "
        "ORDER BY deleted_at DESC")
    if not df.empty:
        df = df.rename(columns={"group_name": "group", "description": "desc"})
    return df


def restore_from_trash(name: str) -> bool:
    """从回收站恢复回全量池。"""
    _ensure_schema()
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM strategy_trash WHERE name = %s", (name,))
        return cur.rowcount > 0


def purge_from_trash(name: str) -> bool:
    """回收站彻底删除。"""
    _ensure_schema()
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM strategy_trash WHERE name = %s", (name,))
        return cur.rowcount > 0


def empty_trash() -> int:
    """清空回收站，返回删除条数。"""
    _ensure_schema()
    with pg.get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM strategy_trash")
        return cur.rowcount


def reorder_pool(names: list[str]) -> None:
    """按给定顺序重写 sort_order（调用方传完整配置池顺序）。"""
    _ensure_schema()
    if not pg.configured():
        return
    with pg.get_conn() as conn, conn.cursor() as cur:
        for i, name in enumerate(names):
            cur.execute(
                "UPDATE strategy_pool SET sort_order = %s WHERE name = %s",
                (int(i), name))


def resolve_strategy(name: str) -> dict:
    """策略定义解析：注册表 → 配置池 → 全量池（归档）；找不到抛 KeyError。

    兼容旧调用方（返回 dict）。新代码请用 resolve_strategy_def 拿到
    结构化 StrategyDefinition（含 kind/source/fingerprint）。
    """
    return resolve_strategy_def(name).to_dict()


def resolve_strategy_def(name: str) -> StrategyDefinition:
    """统一策略解析：返回 StrategyDefinition（三源 + source 溯源）。

    解析顺序（与旧 resolve_strategy 一致）：
      1. 注册表 strategies.registry.STRATEGIES
      2. 配置池 strategy_pool 表
      3. 回测归档 backtest_runs（full_pool）

    同名多源时按上述优先级取源（不静默合并），source 字段标识实际来源，
    便于审计/去重。
    """
    if name in STRATEGIES:
        return from_legacy_dict(name, STRATEGIES[name], source="registry")
    d = pool_def(name)
    if d is not None:
        return from_legacy_dict(name, d, source="pool", default_group="配置池")
    fp = full_pool()
    if not fp.empty:
        hit = fp[fp["name"] == name]
        if not hit.empty:
            r = hit.iloc[0]
            p = r.get("params_summary") or {}
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except json.JSONDecodeError:
                    p = {}
            merged = {
                "factor": r["factor"],
                "ascending": bool(r["ascending"])
                if r.get("ascending") is not None else False,
                "group": r.get("group") or "回测历史",
                "desc": r.get("desc") or "",
                **({k: v for k, v in p.items() if v is not None}),
            }
            return from_legacy_dict(name, merged, source="archive",
                                    default_group="回测历史")
    raise KeyError(name)


def pool_strategy_options(scope: str = "配置池") -> list[str]:
    """策略下拉选项：配置池优先，空则回退全量注册表。"""
    if scope == "全部策略":
        fp = full_pool()
        return fp["name"].tolist() if not fp.empty else list_strategies()
    names = pool_names()
    return names or list_strategies()


def scope_label(scope: str) -> str:
    return "配置池" if scope == "配置池" else "全部策略"

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from core import trading_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABS_DIR = PROJECT_ROOT / "labs"
LABS_TMP_DIR = LABS_DIR / ".tmp"
RUN_TIMEOUT = 180

router = APIRouter(prefix="/api/code", tags=["code"])


class RunRequest(BaseModel):
    # 单模块模式：code 直接是完整策略代码（推荐）
    code: str = ""
    # 兼容旧双文件模式
    registry: str = ""
    factors: str = ""
    strategy: str = ""
    universe: str = "科技TMT"
    top_n: int = 3
    capital: float = trading_config.CAPITAL
    freq: str = "monthly"
    start: str = ""
    end: str = ""
    exclude_kechuang: bool = True
    affordable: bool = True
    amount_q: float = trading_config.AMOUNT_Q
    warmup_days: int | None = trading_config.WARMUP_DAYS
    industry_cap: int | None = None
    slippage_bps: float = trading_config.SLIPPAGE_BPS
    max_participation: float = trading_config.MAX_PARTICIPATION
    buy_cost: float = trading_config.BUY_COST
    sell_cost: float = trading_config.SELL_COST


class SaveRequest(BaseModel):
    name: str
    code: str = ""
    registry: str = ""
    factors: str = ""
    engine: str = "legacy"


class QweaveRunRequest(BaseModel):
    code: str = ""
    universe: str = "沪深300+中证500+中证1000"
    start: str = "2022-01-01"
    end: str = ""
    alpha_set: str = "alpha158"
    alpha_limit: int | None = 30
    horizons: list[int] = [1, 5, 10, 20]
    quantiles: int = 10
    min_cs_count: int = 30
    cost_bps: float = 8.0
    exclude_kechuang: bool = True
    run_backtest: bool = False
    score_factor: str = ""
    top_n: int = 10
    selection_mode: str = "top_n"
    selection_pct: float = 0.10
    min_positions: int = 1
    max_positions: int | None = None
    capital: float = trading_config.CAPITAL
    freq: str = "weekly"
    affordable: bool = True
    amount_q: float = trading_config.AMOUNT_Q
    warmup_days: int | None = trading_config.WARMUP_DAYS
    slippage_bps: float = trading_config.SLIPPAGE_BPS
    max_participation: float = trading_config.MAX_PARTICIPATION
    max_weight: float | None = None
    buy_cost: float = trading_config.BUY_COST
    sell_cost: float = trading_config.SELL_COST
    industry_cap: int | None = None


def _default_registry() -> str:
    return (PROJECT_ROOT / "strategies" / "registry.py").read_text(encoding="utf-8")


def _default_factors() -> str:
    src = (PROJECT_ROOT / "core" / "engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_factor_frames":
            seg = ast.get_source_segment(src, node)
            if seg:
                return seg
    raise RuntimeError("core/engine.py 中找不到 build_factor_frames")


def _build_module(registry: str, factors: str) -> str:
    return (
        "# ===================== 策略注册表（可编辑） =====================\n"
        + registry.rstrip()
        + "\n\n\n"
        + "# ===================== 因子构建（可编辑） =====================\n"
        + factors.rstrip()
        + "\n"
    )


def _module_from_req(req) -> str:
    """优先用完整 code，其次用 registry+factors 拼装。"""
    if req.code and req.code.strip():
        return req.code
    return _build_module(req.registry or "", req.factors or "")


_FACTOR_SNIPPETS = {
    "turn20": "scores = turn20.copy()  # 20 日平均换手率",
    "am20": "scores = am20.copy()  # 20 日平均成交额",
    "mom20": "scores = close.pct_change(20, fill_method=None)  # 20 日涨幅",
    "mom60": "scores = close.pct_change(60, fill_method=None)  # 60 日涨幅",
    "vol20": ("scores = close.pct_change(fill_method=None).rolling(20).std()"
              ".reindex_like(am20)  # 20 日波动率"),
    "composite": ("scores = (am20.rank(axis=1) + close.pct_change(fill_method=None)"
                  ".rolling(20).std().reindex_like(am20).rank(axis=1))"
                  "  # 低成交 + 低波动复合"),
    "ma_cross5_10": ("scores = (close.rolling(5).mean() / close.rolling(10).mean() - 1)"
                     "  # MA5 相对 MA10 乖离：正=超短线多头，负=超短线超跌"),
    "ma_cross5_20": ("scores = (close.rolling(5).mean() / close.rolling(20).mean() - 1)"
                     "  # MA5 相对 MA20 乖离：正=多头/金叉上方，负=空头/死叉下方"),
    "ma_cross10_30": ("scores = (close.rolling(10).mean() / close.rolling(30).mean() - 1)"
                      "  # MA10 相对 MA30 乖离：正=波段多头，负=波段超跌"),
    "ma_cross20_60": ("scores = (close.rolling(20).mean() / close.rolling(60).mean() - 1)"
                      "  # MA20 相对 MA60 乖离：正=中期多头，负=中期空头"),
}


def _template_for(strategy: str) -> str:
    from strategies.registry import STRATEGIES
    if strategy not in STRATEGIES:
        raise ValueError(f"策略不存在: {strategy}")
    s = STRATEGIES[strategy]
    asc = bool(s["ascending"])
    direction = "分数从小到大（买低分）" if asc else "分数从大到小（买高分）"
    factor = s["factor"]

    if s.get("long_short"):
        params = "    LONG_N = 3      # 多头只数\n    SHORT_N = 3     # 空头只数"
        if factor == "mom20":
            params += "\n    LOOKBACK = 20   # 动量回看天数"
            logic = ("    # 多头取动量最强 LONG_N 只，空头取动量最弱 SHORT_N 只\n"
                     "    scores = close.pct_change(LOOKBACK)\n"
                     "    scores = _zscore(scores)")
        elif factor == "turn20":
            logic = ("    # 多头取低换手 LONG_N 只，空头取高换手 SHORT_N 只\n"
                     "    scores = turn20\n"
                     "    scores = _zscore(scores)")
        else:
            logic = ("    # 多头取因子最强 LONG_N 只，空头取最弱 SHORT_N 只（引擎自动配对）\n"
                     "    scores = am20.copy()\n"
                     "    scores = _zscore(scores)")
    elif factor.startswith("ma_cross"):
        m = re.match(r"ma_cross(\d+)_(\d+)", factor)
        sh, lg = (m.group(1), m.group(2)) if m else ("5", "20")
        params = f"    SHORT = {sh}      # 短期均线天数\n    LONG = {lg}       # 长期均线天数"
        logic = ("    # 均线乖离：短期均线相对长期均线的偏离度\n"
                 "    scores = close.rolling(SHORT).mean() / close.rolling(LONG).mean() - 1\n"
                 "    scores = _zscore(scores)")
    elif factor == "composite":
        params = "    W_AM = 1.0     # 低成交权重\n    W_VOL = 1.0    # 低波动权重"
        logic = ("    # 多因子复合：先截面百分位再加权\n"
                 "    scores = _rank_pct(am20) * (-W_AM) + _rank_pct(close.pct_change().rolling(20).std()) * (-W_VOL)")
    elif factor in ("mom20", "mom60"):
        lookback = 60 if factor == "mom60" else 20
        params = f"    LOOKBACK = {lookback}   # 动量回看天数"
        logic = ("    # 动量：N 日累计涨幅（不包含未来数据）\n"
                 "    scores = close.pct_change(LOOKBACK)\n"
                 "    scores = _zscore(scores)")
    elif factor == "vol20":
        params = "    WINDOW = 20     # 波动率窗口"
        logic = ("    # 低波动：滚动标准差越小越符合（ascending=True 自动选低分）\n"
                 "    scores = close.pct_change().rolling(WINDOW).std()\n"
                 "    scores = _rank_pct(scores)")
    elif factor in ("turn20", "am20"):
        src = "turn20" if factor == "turn20" else "am20"
        tip = ("换手率越低越符合（ascending=True 自动选低分）"
               if factor == "turn20" else "成交额越高越符合（ascending=False 自动选高分）")
        params = "    WINDOW = 20     # 平均窗口（由引擎预计算）"
        logic = f"    # {tip}\n    scores = {src}\n    scores = _rank_pct(scores)"
    else:
        params = "    # 自定义参数"
        logic = "    scores = am20.copy()\n    scores = _zscore(scores)"

    header = f'''# ============================================================
# 策略模板：{strategy}
# 组：{s.get("group", "其他")} ｜ 说明：{s.get("desc", "")}
# 方向：{direction}
#
# 使用方式：
#   1. 只改下方「参数」区的值（LOOKBACK / SHORT / WINDOW 等）
#   2. 在 build_factor_frames 的「打分逻辑」区改因子
#   3. 点「解析策略」再「跑代码」
# ============================================================
# ===================== FactorKit 数据/日期封装 =====================
# close / am20 / turn20 都是 DataFrame：
#   行 = 交易日（升序），列 = 股票代码
# 引擎已处理：因子预热(warmup_days)、T+1 成交、涨跌停/停牌过滤、一手过滤
# ⚠ 只允许使用当前行及之前的数据；
#   禁止 pct_change(-n) / shift(-n) / iloc 未来行（会造成前视偏差）
def _zscore(df):
    import numpy as np
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)

def _rank_pct(df):
    # 截面百分位：每行（每日）内按列排名归一化到 0-1
    return df.rank(axis=1) / df.count(axis=1)

def _ts_zscore(df, n):
    # 时序 Z-Score：按每只股票自身滚动均值/标准差标准化
    roll = df.rolling(n, min_periods=5)
    return (df - roll.mean()) / roll.std()
# ================================================================
STRATEGIES = {{
    "{strategy}": {{"factor": "score", "ascending": {asc},
                   "group": "{s.get("group", "其他")}", "desc": "{s.get("desc", "")}"}},
}}

def build_factor_frames(close, am20, turn20):
    import numpy as np
    import pandas as pd

    # ---- 参数 ----
{params}

    # ---- 打分逻辑：给每只股票打一个分数 ----
    # 分数越大 = 越符合「买高分」，越小 = 越符合「买低分」
{logic}
    return {{"score": scores}}
'''
    return header


def _default_event_code() -> str:
    return '''# ============================================================
# 事件驱动策略模板
#
# 结构：
#   STRATEGIES      前端下拉展示（factor 字段事件策略可留空）
#   MyEventStrategy 策略类：每个交易日 on_bar(ctx, bar) 被调用一次
#   EVENT_STRATEGIES 注册策略类，跑代码时按名称选择
#
# ctx 常用接口：
#   ctx.order_target_pct(code, pct)   调整到目标市值占比（0=清仓）
#   ctx.order_target_shares(code, n)  调整到目标股数
#   ctx.order_shares(code, delta)     增减 delta 股
#   ctx.position(code) / ctx.positions    当前持仓股数 / {代码: 股数}
#   ctx.portfolio_value / ctx.cash    组合总市值 / 现金
#   ctx.close_series(code, n)         信号日往前 n 个有效收盘价
#   ctx.history(code, fields, n)      多字段历史 DataFrame：行=交易日(旧→新，
#                                     末行=信号日)，fields 如 ["close","volume"]；
#                                     停牌/缺失为 NaN；未知字段抛 ValueError
#   ctx.available_fields              可用字段列表（面板全部数值列）
#   ctx.is_tradable(code)             执行日能否交易（开盘有效且昨日有成交）
#   ctx.can_buy(code) / ctx.can_sell(code)
#                                     执行日能否买入/卖出（含涨跌停限制）
#
# bar 常用字段：
#   bar.date / bar.exec_date          信号日 / 执行日（T+1）
#   bar.tradable                      可交易股票集合
# ============================================================
from core.event_engine import EventStrategy

STRATEGIES = {
    "双均线金叉事件": {"factor": "", "ascending": False,
                       "group": "事件驱动", "desc": "MA5/MA20 金叉买入、死叉清仓（示例）"},
}


class MyEventStrategy(EventStrategy):
    short = 5      # 短期均线
    long = 20      # 长期均线
    top_n = 3      # 最大持仓只数
    max_weight = 0.5  # 单票目标权重上限

    def on_bar(self, ctx, bar):
        # 1) 清仓：持仓中当前出现死叉的
        for code in list(ctx.positions):
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            short_prev = sum(closes[-self.short - 1:-1]) / self.short
            long_prev = sum(closes[-self.long - 1:-1]) / self.long
            short_now = sum(closes[-self.short:]) / self.short
            long_now = sum(closes[-self.long:]) / self.long
            if short_prev >= long_prev and short_now < long_now:
                ctx.order_target_pct(code, 0.0)

        # 2) 买入：金叉且未持仓，按强度取 top_n
        held = [c for c, sh in ctx.positions.items() if sh > 0]
        slots = max(0, self.top_n - len(held))
        scores = []
        for code in bar.tradable:
            closes = ctx.close_series(code, self.long + 2)
            if len(closes) < self.long + 2:
                continue
            short_prev = sum(closes[-self.short - 1:-1]) / self.short
            long_prev = sum(closes[-self.long - 1:-1]) / self.long
            short_now = sum(closes[-self.short:]) / self.short
            long_now = sum(closes[-self.long:]) / self.long
            if short_prev <= long_prev and short_now > long_now:
                scores.append((short_now - long_now, code))
        scores.sort(reverse=True)
        w = min(self.max_weight, 1.0 / self.top_n)
        for _, code in scores:
            if slots <= 0:
                break
            if ctx.position(code) > 0:
                continue
            ctx.order_target_pct(code, w)
            slots -= 1


EVENT_STRATEGIES = {
    "双均线金叉事件": MyEventStrategy,
}
'''


def _save_tmp(suffix: str, text: str) -> Path:
    LABS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    p = LABS_TMP_DIR / f"lab_{uuid.uuid4().hex}{suffix}"
    p.write_text(text, encoding="utf-8")
    return p


def _run_runner(cfg: dict) -> tuple[int, str, dict]:
    cfg_path = _save_tmp(".json", json.dumps(cfg, ensure_ascii=False))
    out_path = LABS_TMP_DIR / f"out_{uuid.uuid4().hex}.json"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "backend.lab_runner",
             "--config", str(cfg_path), "--out", str(out_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        cfg_path.unlink(missing_ok=True)
        return 1, f"代码运行超时（>{RUN_TIMEOUT}s）", {}
    finally:
        cfg_path.unlink(missing_ok=True)

    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            out_path.unlink(missing_ok=True)
            return proc.returncode, f"运行结果解析失败: {exc}", {}
        out_path.unlink(missing_ok=True)
        return proc.returncode, proc.stderr or "", payload
    return proc.returncode, proc.stderr or "子进程未产出结果文件", {}


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", name).strip("_")
    if not cleaned:
        raise ValueError("保存名称不能为空")
    return cleaned[:60]


@router.get("/default")
def get_default():
    return {
        "code": _default_event_code(),
        "event_code": _default_event_code(),
        "registry": _default_registry(),
        "factors": _default_factors(),
    }


@router.get("/qweave/default")
def get_qweave_default():
    from backend.qweave_runner import _default_code
    return {"engine": "qweave", "code": _default_code()}


@router.get("/qweave/templates")
def list_qweave_templates():
    from backend.qweave_runner import QWEAVE_TEMPLATES
    return {"items": [{"name": name, **item} for name, item in QWEAVE_TEMPLATES.items()]}


@router.get("/qweave/template")
def get_qweave_template(name: str):
    try:
        from backend.qweave_runner import QWEAVE_TEMPLATES, template_code
        item = QWEAVE_TEMPLATES[name]
        return {"ok": True, "engine": "qweave", "name": name,
                "label": item["label"], "code": template_code(name)}
    except (KeyError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/qweave/parse")
def parse_qweave(req: QweaveRunRequest):
    try:
        from backend.qweave_runner import parse_code as parse_qweave_code
        return parse_qweave_code(req.code)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@router.post("/qweave/run")
def run_qweave(req: QweaveRunRequest):
    from backend.qweave_runner import execute
    return execute(req.model_dump())


@router.get("/template")
def get_template(strategy: str):
    try:
        return {"name": strategy, "code": _template_for(strategy)}
    except ValueError as exc:
        return {"error": str(exc)}


@router.get("/saved")
def list_saved():
    if not LABS_DIR.exists():
        return {"items": []}
    items = []
    for p in sorted(LABS_DIR.glob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "name": meta.get("name", p.stem),
                "engine": meta.get("engine", "legacy"),
                "saved_at": meta.get("saved_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return {"items": items}


@router.get("/saved/{name}")
def get_saved(name: str):
    safe = _safe_name(name)
    p = LABS_DIR / f"{safe}.json"
    if not p.exists():
        return {"error": f"没有找到已保存代码: {safe}"}
    meta = json.loads(p.read_text(encoding="utf-8"))
    if meta.get("code"):
        return {
            "name": safe,
            "code": meta["code"],
            "saved_at": meta.get("saved_at", ""),
            "engine": meta.get("engine", "legacy"),
        }
    return {
        "name": safe,
        "registry": meta.get("registry", ""),
        "factors": meta.get("factors", ""),
        "saved_at": meta.get("saved_at", ""),
    }


@router.post("/save")
def save_code(req: SaveRequest):
    safe = _safe_name(req.name)
    LABS_DIR.mkdir(parents=True, exist_ok=True)
    code = _module_from_req(req)
    meta = {
        "name": safe,
        "code": code,
        "registry": req.registry,
        "factors": req.factors,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "engine": req.engine,
    }
    (LABS_DIR / f"{safe}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (LABS_DIR / f"{safe}.py").write_text(code, encoding="utf-8")
    return {"ok": True, "name": safe}


@router.post("/parse")
def parse_code(req: RunRequest):
    module_path = _save_tmp(".py", _module_from_req(req))
    cfg = {
        "root": str(PROJECT_ROOT),
        "module": str(module_path),
        "parse_only": True,
    }
    code, stderr, payload = _run_runner(cfg)
    module_path.unlink(missing_ok=True)
    if code != 0 or not payload.get("ok"):
        return {"ok": False, "error": payload.get("error") or stderr}
    return payload


@router.post("/run")
def run_code(req: RunRequest):
    module_path = _save_tmp(".py", _module_from_req(req))
    cfg = {
        "root": str(PROJECT_ROOT),
        "module": str(module_path),
        "parse_only": False,
        "strategy": req.strategy,
        "universe": req.universe,
        "top_n": req.top_n,
        "capital": req.capital,
        "freq": req.freq,
        "start": req.start,
        "end": req.end,
        "exclude_kechuang": req.exclude_kechuang,
        "affordable": req.affordable,
        "amount_q": req.amount_q,
        "warmup_days": req.warmup_days,
        "industry_cap": req.industry_cap,
        "slippage_bps": req.slippage_bps,
        "max_participation": req.max_participation,
        "buy_cost": req.buy_cost,
        "sell_cost": req.sell_cost,
    }
    code, stderr, payload = _run_runner(cfg)
    module_path.unlink(missing_ok=True)
    if code != 0 or not payload.get("ok"):
        return {
            "ok": False,
            "error": payload.get("error") or stderr,
            "traceback": payload.get("traceback", ""),
        }
    return payload

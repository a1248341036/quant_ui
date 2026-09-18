"""ST（风险警示板）横截面剔除：把 ST 票从因子评估横截面中剔除。

**做法**：在被 ST 的 (datetime, instrument) 行把**因子值**置 NaN，而不是给 label
打 NaN、也不改动 panel —— 所有 metric 的 ``np.isfinite`` 有限值掩码会自动剔除
这些样本（IC/ICIR/十分位/组合回测/月度稳健性/覆盖率/引擎回测一致生效），一处
生效、全链路一致；panel 与 label 保持原样，label 身份缓存（``metrics/_label_cache``）
与 ``artifacts/factor_value_cache`` 的因子值缓存不被污染。

**数据源**：CNE curated ``stock_st`` 数据集（一行 = 某交易日某票处于风险警示板；
未出现 = 当日非 ST），已与 baostock 逐日 ``isST`` 真值全量对账（precision/recall = 1.0）。
**不用 panel 自带的 ``is_st`` 列**：CNE ``stock_daily_wide`` 的 ``is_st`` 在
2020~2026 面板上对上述真值 precision 0.9988 但 **recall 仅 0.2942**
（漏报 13.3 万个 ST 票日），拿它做剔除会系统性漏掉约 70% 的 ST 样本。

**接入点**（同一份掩码，口径一致）：
- ``EvaluationEngine.evaluate``：train_screen / train_screen_lite / validation /
  size_neutral_validation / production_delivery 全 profile；
- ``ingest.compute_ingest_metrics``：submit 的 train / val / test(盲测) / 全窗指标；
- ``submit`` 直接消费因子值的组合回测与 engine_gate；
- ``eval.service._maybe_engine_preview``（val 引擎预演）、``population.screen_expr``（群体筛）。

**开关**：默认开启；``ALPHA_ST_MASK=0``（或 false/no/off）关闭。数据集缺失/不可读时
记 warning 并跳过（fail-open，绝不中断挖掘）。ETF（``asset_type != "stock"`` 或无
ST 代码命中）不受影响——ST 成员集合只含股票代码，按 symbol 精确 join。

**成本**：成员表进程级缓存（读一次 parquet，~0.15s）；行掩码按 panel 索引身份
缓存（weakref + ``is`` 校验，范式同 ``metrics/tradable.py``），8.2M 行面板首建
~0.35s（候选票过滤 + 复合键 join），同 session 后续评估摊销为 0。
"""
from __future__ import annotations

import logging
import os
import threading
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASET = "stock_st"
_PART_GLOB = "trade_date=*/part-merged.parquet"
# 复合键：品种序号 * 2^21 + 日期序号（距 1970 的天数 < 2^21 ≈ 5700 年）
_KEY_SHIFT = 1 << 21
_NS_PER_DAY = 86_400_000_000_000
_CACHE_MAX = 8  # 行掩码缓存条数（每条约 len(panel) 字节）
_FALSY = {"0", "false", "no", "off", "none", ""}

# 行掩码缓存：键 = id(panel.index)；值 = (索引弱引用, bool ndarray, 诊断 dict)
_MASK_CACHE: "OrderedDict[int, tuple[weakref.ref, np.ndarray, dict[str, Any]]]" = OrderedDict()
# ST 成员表缓存：键 = 数据集目录字符串；值 = (membership, 诊断) 或 None（不可用）
_MEMBERSHIP: dict[str, "tuple[_Membership, dict[str, Any]] | None"] = {}
_LOCK = threading.RLock()
_WARNED: set[str] = set()  # 同一原因只告警一次（fail-open 路径可能被调用数千次）


def _warn_once(key: str, message: str, *args: Any) -> None:
    with _LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    logger.warning(message, *args)


class _Membership:
    """ST 成员表：symbols（唯一代码 Index）+ keys（复合键哈希 Index，查成员 O(1)）。"""

    __slots__ = ("symbols", "keys")

    def __init__(self, symbols: pd.Index, keys: pd.Index) -> None:
        self.symbols = symbols
        self.keys = keys


def _enabled() -> bool:
    """``ALPHA_ST_MASK`` 逃生舱：未设置/空 → 开启；显式 0/false/no/off → 关闭。"""
    raw = os.environ.get("ALPHA_ST_MASK")
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSY


def _repo_root() -> Path:
    # alphaagent/factor/metrics/st_mask.py → 上溯 3 层 = 仓库根
    return Path(__file__).resolve().parents[3]


def dataset_dir() -> Path | None:
    """ST 数据集目录：``ALPHA_ST_MASK_PATH`` 优先，否则仓库内 CNE curated 默认路径。

    默认路径不存在时尝试从 CNE 运行时配置 ``[data].root`` 推导（数据湖可被
    配置迁移到别处）；都不可用返回 ``None``（调用方 fail-open）。
    """
    env = os.environ.get("ALPHA_ST_MASK_PATH")
    if env and str(env).strip():
        return Path(str(env).strip())

    root = _repo_root() / "CNEquity"
    default = root / "data" / "quant_dataset" / "_cnequity" / "curated" / DATASET
    if default.is_dir():
        return default

    cfg = root / "configs" / "cnequity.quant_dataset.toml"
    if cfg.is_file():
        try:
            import tomllib

            data = tomllib.loads(cfg.read_text(encoding="utf-8"))
            rel = str((data.get("data") or {}).get("root") or "").strip()
            if rel:
                # 配置内 root 相对配置文件所在目录（CNE 以 cwd=CNEquity 加载）
                cand = (cfg.parent / rel).resolve() / "curated" / DATASET
                if cand.is_dir():
                    return cand
        except Exception as exc:  # noqa: BLE001 — 配置解析失败只降级不抛
            logger.warning("ST 掩码：CNE 配置解析失败（%s）", exc)
    return None


def _to_days(values: Any) -> np.ndarray:
    """datetime 序列 → 距 1970-01-01 的天数（int64，与时区/单位无关）。"""
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize().to_numpy(dtype="datetime64[ns]").astype(np.int64) // _NS_PER_DAY


def _load_membership(root: Path) -> tuple[_Membership, dict[str, Any]] | None:
    """读全部月度分区 → (symbol, trade_date) 成员表；失败返回 None（fail-open）。"""
    cache_key = str(root)
    with _LOCK:
        if cache_key in _MEMBERSHIP:
            return _MEMBERSHIP[cache_key]

    result: tuple[_Membership, dict[str, Any]] | None = None
    warn: str | None = None
    try:
        import pyarrow.parquet as pq

        files = sorted(root.glob(_PART_GLOB))
        if not files:
            warn = f"数据集目录无分区文件: {root}"
        else:
            frames = []
            for path in files:
                # ParquetFile.read 而非 read_table：分区目录名 trade_date=YYYY-MM
                # 会被 pyarrow 当 hive 分区推断成字符串列，与文件内 date32 列冲突。
                frames.append(
                    pq.ParquetFile(path).read(columns=["symbol", "trade_date"]).to_pandas()
                )
            raw = pd.concat(frames, ignore_index=True).drop_duplicates()
            symbols = pd.Index(raw["symbol"].astype(str).unique())
            codes = symbols.get_indexer(raw["symbol"].astype(str).to_numpy())
            days = _to_days(raw["trade_date"])
            keys = pd.Index(codes.astype(np.int64) * _KEY_SHIFT + days)
            diag = {
                "source": str(root),
                "n_files": len(files),
                "n_rows": int(len(raw)),
                "n_symbols": int(len(symbols)),
                "start": str(pd.Timestamp(int(days.min()), unit="D").date()),
                "end": str(pd.Timestamp(int(days.max()), unit="D").date()),
            }
            result = (_Membership(symbols, keys), diag)
            logger.info(
                "ST 掩码：成员表就绪 %d 行 / %d 代码 / %s~%s（%s）",
                diag["n_rows"], diag["n_symbols"], diag["start"], diag["end"], root,
            )
    except Exception as exc:  # noqa: BLE001 — 数据源异常一律降级为"不剔除"
        warn = f"{type(exc).__name__}: {exc}"

    if result is None:
        _warn_once(
            f"membership:{root}",
            "ST 掩码数据源不可用（%s）：%s；本次评估跳过 ST 剔除（fail-open）",
            root, warn,
        )
    with _LOCK:
        _MEMBERSHIP[cache_key] = result
        return _MEMBERSHIP[cache_key]


def _build_mask(index: pd.MultiIndex, membership: _Membership) -> np.ndarray:
    """panel 索引 → ST 行掩码（True = ST，应剔除）。"""
    inst = pd.Index(index.get_level_values("instrument").astype(str))
    day = _to_days(index.get_level_values("datetime"))
    mask = np.zeros(len(index), dtype=bool)
    # 先按代码粗筛（ST 代码全市场仅 ~800 个），只对候选行做日期 join
    cand = np.asarray(inst.isin(membership.symbols.to_numpy()), dtype=bool)
    if not cand.any():
        return mask
    codes = membership.symbols.get_indexer(inst[cand].to_numpy())
    key = codes.astype(np.int64) * _KEY_SHIFT + day[cand]
    mask[cand] = membership.keys.get_indexer(key) >= 0
    return mask


def st_row_mask(panel: pd.DataFrame, *, asset_type: str | None = None) -> np.ndarray | None:
    """ST 行掩码（True = 该行是 ST）；关闭/不可用/非股票时返回 ``None``。

    按 panel **索引身份**缓存（``id(index)`` + weakref ``is`` 校验，防 id 复用误命中），
    8.2M 行面板首建一次，同 session 后续评估直接命中。
    """
    if not _enabled():
        return None
    if asset_type is not None and str(asset_type) != "stock":
        return None
    if not isinstance(panel, pd.DataFrame) or len(panel) == 0:
        return None
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
        return None

    index = panel.index
    key = id(index)
    with _LOCK:
        hit = _MASK_CACHE.get(key)
        if hit is not None:
            ref, cached, diag = hit
            if ref() is index:
                _MASK_CACHE.move_to_end(key)
                return cached
            _MASK_CACHE.pop(key, None)

        root = dataset_dir()
        if root is None:
            _warn_once(
                "dataset_missing",
                "ST 掩码数据源缺失（%s 下无 %s，且 ALPHA_ST_MASK_PATH 未设置）；跳过 ST 剔除（fail-open）",
                _repo_root() / "CNEquity", DATASET,
            )
            return None
        loaded = _load_membership(root)
        if loaded is None:
            return None

        try:
            membership, mem_diag = loaded
            mask = _build_mask(index, membership)
        except Exception as exc:  # noqa: BLE001 — 掩码构建失败不得中断评估
            logger.warning("ST 掩码构建失败（%s: %s）；跳过 ST 剔除（fail-open）", type(exc).__name__, exc)
            return None
        mask.setflags(write=False)
        diag = {
            "available": True,
            "n_rows": int(len(mask)),
            "n_st_rows": int(mask.sum()),
            "st_row_share": round(float(mask.mean()), 6),
            "membership": mem_diag,
        }
        _MASK_CACHE[key] = (weakref.ref(index), mask, diag)
        _MASK_CACHE.move_to_end(key)
        while len(_MASK_CACHE) > _CACHE_MAX:
            _MASK_CACHE.popitem(last=False)
        return mask


def _nan_at(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按掩码置 NaN（保持输入 float dtype；非浮点输入升为 float32）。"""
    if arr.dtype.kind != "f" or arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    arr[mask] = np.nan
    return arr


def mask_values(
    values: Any,
    panel: pd.DataFrame,
    *,
    asset_type: str | None = None,
) -> Any:
    """因子值在 ST 行置 NaN（返回新对象，不改动入参）；不可用/关闭时原样返回。"""
    mask = st_row_mask(panel, asset_type=asset_type)
    if mask is None or not bool(mask.any()):
        return values
    if isinstance(values, pd.Series):
        if len(values) != len(mask):
            return values
        out = _nan_at(values.to_numpy(copy=True), mask)
        return pd.Series(out, index=values.index, name=values.name)
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != len(mask):
        return values
    return _nan_at(arr.copy(), mask)


def st_mask_info(panel: pd.DataFrame, *, asset_type: str | None = None) -> dict[str, Any]:
    """诊断信息（不含掩码数组）：可用性 / 行数 / ST 行数与占比 / 成员表元信息。"""
    if not _enabled():
        return {"available": False, "reason": "disabled_by_env", "st_row_share": 0.0}
    mask = st_row_mask(panel, asset_type=asset_type)
    if mask is None:
        root = dataset_dir()
        return {
            "available": False,
            "reason": "dataset_unavailable" if root is None else "unavailable",
            "dataset_dir": str(root) if root is not None else None,
            "st_row_share": 0.0,
        }
    with _LOCK:
        hit = _MASK_CACHE.get(id(panel.index))
        diag = dict(hit[2]) if hit is not None and hit[0]() is panel.index else {}
    diag.setdefault("n_rows", int(len(mask)))
    diag.setdefault("n_st_rows", int(mask.sum()))
    diag.setdefault("st_row_share", round(float(mask.mean()), 6))
    return diag


def clear_cache() -> None:
    """清空成员表与行掩码缓存（测试用）。"""
    with _LOCK:
        _MASK_CACHE.clear()
        _MEMBERSHIP.clear()

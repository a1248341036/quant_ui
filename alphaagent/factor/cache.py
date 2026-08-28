"""因子值缓存：``eval_factor(expr, panel)`` 确定性结果的会话级复用。

背景
----
挖掘循环中同一因子表达式会被多处重复求值，每次 DSL 求值 30-48s：
- 训练集评估（train_screen）与验证集评估（validation）分别对 train/val 切片求值；
- ``submit`` 阶段在完整会话面板上再求值一次（materialize_factor）；
- ``_candidate_registry_similarity`` 对候选池每个 DSL 在完整面板上重新求值，
  每次提交都会重复执行整批。

设计
----
- **key** = sha256(expr + panel_fingerprint + schema_version)。
  panel_fingerprint 由面板结构（行数/日期范围/列/类型）+ 内容抽样哈希构成，
  保证同一历史面板跨会话稳定命中，数据变更时自动失效。
- **值**：只存对齐面板行序的 float32 数组（不含 MultiIndex），命中时用当前
  ``panel.index`` 重建 Series —— 面板指纹一致 ⇒ 行序一致，重建无损。
- **内存 LRU（跨会话共享）**：进程级单一 ``_SHARED_MEM`` OrderedDict + 共享锁，
  所有 ``FactorValueCache`` 实例（每个 StockEvalSession 一个）读写同一份热缓存。
  内容寻址 + 只存值数组 ⇒ 跨会话安全（无对象引用/地址复用问题），
  多开评估会话时内存不随会话数线性膨胀（上限为进程级总内存上限）。
- **磁盘**：``artifacts/factor_value_cache/`` 下 ``.npy`` 值 + ``.json`` 元数据。
  - 空间控制：总字节上限 ``_FV_MAX_BYTES``（默认 2GB）+ 文件数上限
    ``_FV_MAX_FILES``（默认 1500），可经环境变量覆盖。
  - 淘汰机制：写入后若超上限，按 ``last_access`` 淘汰最久未用条目（LRU），
    锁死长期磁盘占用；孤儿/损坏文件在启动对账时清理。
- **线程安全**：共享锁保护内存与各实例 manifest；磁盘写入原子化（tmp+os.replace），
  多进程并发时对同 key 内容一致、后写覆盖，淘汰为尽力而为。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from alphaagent.core.paths import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# 磁盘缓存根目录
_FV_DIR = ARTIFACTS_DIR / "factor_value_cache"
# 缓存格式/语义版本：DSL 算子语义变化影响结果时 +1 强制全部失效
_FV_SCHEMA_VERSION = 1
# 磁盘总字节上限（默认 2GB，可经 ALPHA_FACTOR_CACHE_MAX_BYTES 覆盖）
_FV_MAX_BYTES = int(os.environ.get("ALPHA_FACTOR_CACHE_MAX_BYTES", str(2 * 1024**3)))
# 磁盘文件数上限
_FV_MAX_FILES = int(os.environ.get("ALPHA_FACTOR_CACHE_MAX_FILES", "1500"))
# 进程内 LRU 条目上限（每条约 n_rows×4B；3.8M 行 ≈ 15MB）。
# 内存 LRU 为**跨会话/跨实例共享**（模块级），上限即进程级总内存上限，
# 多开评估会话不会随会话数线性膨胀（磁盘缓存本就共享）。
_FV_MEM_MAX_ENTRIES = 16
# 面板指纹内容抽样：最多抽样 ~2048 行参与哈希
_FV_FP_SAMPLE_ROWS = 2048

_MANIFEST_NAME = "manifest.json"
_NPY_SUFFIX = ".npy"
_META_SUFFIX = ".json"

# 共享内存 LRU：所有 FactorValueCache 实例读写同一份热缓存（跨会话复用）。
# key = _cache_key（expr + panel 内容指纹 + schema），内容寻址 ⇒ 跨会话安全；
# 值只存 float32 值数组，不持有 panel/Series 引用 ⇒ 无对象生命周期/地址复用风险。
_SHARED_MEM: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
_SHARED_MEM_LOCK = threading.RLock()
_SHARED_MEM_MAX_ENTRIES = int(os.environ.get("ALPHA_FACTOR_CACHE_MEM_MAX_ENTRIES", str(_FV_MEM_MAX_ENTRIES)))


def _env_bytes(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _panel_fingerprint(panel: pd.DataFrame) -> str:
    """面板确定性指纹：结构 + 内容抽样哈希。

    结构部分（行数/日期范围/列/类型）保证同参数的历史面板稳定命中；
    内容抽样部分捕获数据刷新（如基本面重述），变更时使旧缓存失效。
    """
    dts = panel.index.get_level_values("datetime")
    idx_min = dts.min() if len(dts) else None
    idx_max = dts.max() if len(dts) else None
    cols = tuple(str(c) for c in panel.columns)
    dtypes = tuple(str(panel[c].dtype) for c in panel.columns)
    h = hashlib.sha256()
    h.update(
        repr(
            (
                len(panel),
                str(idx_min),
                str(idx_max),
                cols,
                dtypes,
                _FV_SCHEMA_VERSION,
            )
        ).encode("utf-8")
    )
    n = len(panel)
    if n > 0:
        step = max(1, n // _FV_FP_SAMPLE_ROWS)
        sample = panel.iloc[::step]
        # 数值列统一转 float32 再哈希；非数值列以字符串表示参与
        num = sample.select_dtypes(include=[np.number])
        if not num.empty:
            h.update(num.to_numpy(dtype=np.float32, copy=False).tobytes())
        else:
            h.update(b"__NO_NUMERIC__")
        for c in sample.columns:
            if not np.issubdtype(sample[c].dtype, np.number):
                h.update(repr(tuple(str(v) for v in sample[c].iloc[:100])).encode("utf-8"))
    return h.hexdigest()[:24]


def _cache_key(expr: str, panel_fp: str) -> str:
    h = hashlib.sha256(f"{expr.strip()}|{panel_fp}|v{_FV_SCHEMA_VERSION}".encode("utf-8"))
    return h.hexdigest()[:24]


class FactorValueCache:
    """因子值缓存（内存 LRU + 磁盘持久化 + 磁盘空间上限/淘汰）。"""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        max_bytes: int | None = None,
        max_files: int | None = None,
        mem_max_entries: int | None = None,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _FV_DIR
        self.max_bytes = max_bytes if max_bytes is not None else _FV_MAX_BYTES
        self.max_files = max_files if max_files is not None else _FV_MAX_FILES
        # 内存 LRU 为进程级共享：所有实例读写同一份热缓存（跨会话复用），
        # 用同一把锁保护；条目上限取全局值（共享 LRU 是单一全局资源，
        # 忽略实例参数，避免多实例对同一资源各设各的上限互相打架）。
        self.mem_max_entries = _SHARED_MEM_MAX_ENTRIES
        self.enabled = enabled
        self._lock = _SHARED_MEM_LOCK
        self._mem: OrderedDict[str, tuple[np.ndarray, float]] = _SHARED_MEM
        self._manifest: dict[str, dict[str, Any]] = {}
        self._fp_memo: dict[int, str] = {}
        self._last_manifest_flush = 0.0
        self._manifest_flush_interval = 30.0
        self._load_manifest()

    # ------------------------------------------------------------------
    # 元数据 / manifest
    # ------------------------------------------------------------------
    def _manifest_path(self) -> Path:
        return self.cache_dir / _MANIFEST_NAME

    def _load_manifest(self) -> None:
        """启动时读 manifest；损坏/缺失时按目录 mtime 重建，并清理孤儿文件。"""
        try:
            if self._manifest_path().is_file():
                self._manifest = json.loads(self._manifest_path().read_text(encoding="utf-8"))
                if not isinstance(self._manifest, dict):
                    self._manifest = {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("因子值缓存 manifest 读取失败（将重建）: %s", exc)
            self._manifest = {}
        self._reconcile_disk()

    def _reconcile_disk(self) -> None:
        """把磁盘上的实际文件与 manifest 对齐：补缺、剔除孤儿、清理临时残留、重算总量。"""
        try:
            if not self.cache_dir.is_dir():
                return
            keys: set[str] = set()
            for p in self.cache_dir.glob(f"*{_NPY_SUFFIX}"):
                key = p.name[: -len(_NPY_SUFFIX)]
                if key.startswith(".fv."):
                    # 崩溃遗留的原子写临时文件（mkstemp 前缀 .fv.），直接清理
                    try:
                        p.unlink()
                    except OSError:
                        pass
                    continue
                keys.add(key)
                meta_p = self.cache_dir / f"{key}{_META_SUFFIX}"
                if key not in self._manifest or not meta_p.is_file():
                    self._index_entry_from_disk(key, p, meta_p)
            # 清理 manifest 临时文件与无主 .json（非 manifest）
            for p in self.cache_dir.glob(".manifest.*.json.tmp"):
                try:
                    p.unlink()
                except OSError:
                    pass
            # 剔除 manifest 里磁盘上已不存在的条目
            for key in list(self._manifest.keys()):
                if key not in keys:
                    self._manifest.pop(key, None)
            self._flush_manifest()
        except Exception as exc:  # noqa: BLE001
            logger.warning("因子值缓存磁盘对账异常（忽略）: %s", exc)

    def _index_entry_from_disk(self, key: str, npy_path: Path, meta_path: Path) -> None:
        try:
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = {"key": key}
            meta.setdefault("last_access", npy_path.stat().st_mtime)
            meta.setdefault("bytes", npy_path.stat().st_size)
            meta.setdefault("n_rows", 0)
            self._manifest[key] = meta
        except Exception as exc:  # noqa: BLE001
            logger.warning("因子值缓存索引条目失败 %s: %s", key, exc)

    def _flush_manifest(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.cache_dir), prefix=".manifest.", suffix=".json.tmp")
            os.close(fd)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._manifest, f, ensure_ascii=False, sort_keys=True)
                os.replace(tmp, self._manifest_path())
            except BaseException:  # noqa: BLE001
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("因子值缓存 manifest 写入失败（不影响使用）: %s", exc)

    # ------------------------------------------------------------------
    # 面板指纹（按对象 id 记忆化，同一面板对象只算一次）
    # ------------------------------------------------------------------
    def panel_fingerprint(self, panel: pd.DataFrame) -> str:
        oid = id(panel)
        fp = self._fp_memo.get(oid)
        if fp is None:
            fp = _panel_fingerprint(panel)
            if len(self._fp_memo) > 4096:
                self._fp_memo.clear()
            self._fp_memo[oid] = fp
        return fp

    def _npy_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}{_NPY_SUFFIX}"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}{_META_SUFFIX}"

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------
    def get(self, expr: str, panel: pd.DataFrame) -> pd.Series | None:
        """按 (expr, panel 指纹) 取缓存因子值；命中返回重建 Series，未命中返回 None。"""
        if not self.enabled:
            return None
        if panel is None or len(panel) == 0:
            return None
        fp = self.panel_fingerprint(panel)
        key = _cache_key(expr, fp)
        with self._lock:
            mem = self._mem.get(key)
            if mem is not None:
                values, last_access = mem
                self._mem.move_to_end(key)
                self._manifest.setdefault(key, {"last_access": last_access})
                return pd.Series(values, index=panel.index, name="expr", dtype=np.float32)
            entry = self._manifest.get(key)
            if entry is None:
                return None
            npy = self._npy_path(key)
            if not npy.is_file():
                self._manifest.pop(key, None)
                return None
            try:
                values = np.load(npy, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("因子值缓存读取失败（剔除 %s）: %s", key, exc)
                self._manifest.pop(key, None)
                try:
                    npy.unlink()
                except OSError:
                    pass
                return None
            if values.shape[0] != len(panel):
                self._manifest.pop(key, None)
                return None
            now = time.time()
            entry["last_access"] = now
            self._put_mem(key, values, now)
            self._mark_dirty()
            return pd.Series(values, index=panel.index, name="expr", dtype=np.float32)

    def put(self, expr: str, panel: pd.DataFrame, series: pd.Series) -> None:
        """缓存求值结果（对齐 panel 行序的 float32 数组）。"""
        if not self.enabled:
            return
        if panel is None or len(panel) == 0 or series is None:
            return
        values = np.asarray(series.reindex(panel.index).to_numpy(dtype=np.float32, copy=False))
        if values.shape[0] != len(panel):
            return
        fp = self.panel_fingerprint(panel)
        key = _cache_key(expr, fp)
        now = time.time()
        with self._lock:
            self._put_mem(key, values, now)
            entry = self._manifest.get(key)
            if entry is None:
                entry = {
                    "key": key,
                    "expr": expr.strip()[:200],
                    "panel_fp": fp,
                    "n_rows": int(len(panel)),
                    "bytes": int(values.nbytes),
                    "created": now,
                    "last_access": now,
                }
                self._manifest[key] = entry
            else:
                entry["last_access"] = now
                entry["n_rows"] = int(len(panel))
                entry["bytes"] = int(values.nbytes)
            self._write_entry_disk(key, values, entry)

    def evaluate(
        self,
        expr: str,
        panel: pd.DataFrame,
        evaluator: Callable[[], pd.Series],
    ) -> pd.Series:
        """get-or-compute：命中直接返回，未命中调用 evaluator 后缓存。"""
        cached = self.get(expr, panel)
        if cached is not None:
            return cached
        result = evaluator()
        if isinstance(result, pd.Series):
            self.put(expr, panel, result)
        return result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _put_mem(self, key: str, values: np.ndarray, last_access: float) -> None:
        self._mem[key] = (values, last_access)
        self._mem.move_to_end(key)
        while len(self._mem) > self.mem_max_entries:
            self._mem.popitem(last=False)

    def _write_entry_disk(self, key: str, values: np.ndarray, entry: dict[str, Any]) -> None:
        """原子写 npy + 元数据，然后按上限淘汰最久未用条目。"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # np.save 对不以 .npy 结尾的文件名会追加 .npy，因此 tmp 后缀必须为 .npy
            fd, tmp = tempfile.mkstemp(dir=str(self.cache_dir), prefix=".fv.", suffix=".npy")
            os.close(fd)
            try:
                np.save(tmp, values, allow_pickle=False)
                os.replace(tmp, self._npy_path(key))
            except BaseException:  # noqa: BLE001
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            fd2, tmp2 = tempfile.mkstemp(dir=str(self.cache_dir), prefix=".fv.", suffix=".json.tmp")
            os.close(fd2)
            try:
                with open(tmp2, "w", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False, sort_keys=True)
                os.replace(tmp2, self._meta_path(key))
            except BaseException:  # noqa: BLE001
                try:
                    os.unlink(tmp2)
                except OSError:
                    pass
                raise
            self._evict_disk()
            self._flush_manifest()
        except Exception as exc:  # noqa: BLE001
            logger.warning("因子值缓存写入失败（不影响使用）: %s", exc)

    def _evict_disk(self) -> None:
        """磁盘空间控制：总量超 max_bytes 或文件数超 max_files 时，LRU 淘汰。"""
        try:
            if not self.cache_dir.is_dir():
                return
            total = sum(int(e.get("bytes", 0) or 0) for e in self._manifest.values())
            n_files = len(self._manifest)
            if total <= self.max_bytes and n_files <= self.max_files:
                return
            victims = sorted(
                self._manifest.items(),
                key=lambda kv: float(kv[1].get("last_access", 0)),
            )
            for key, _ in victims:
                if total <= self.max_bytes and len(self._manifest) <= self.max_files:
                    break
                npy = self._npy_path(key)
                meta = self._meta_path(key)
                try:
                    size = int(self._manifest.get(key, {}).get("bytes", 0) or 0)
                    if npy.is_file():
                        npy.unlink()
                    if meta.is_file():
                        meta.unlink()
                    self._manifest.pop(key, None)
                    self._mem.pop(key, None)
                    total -= size
                    logger.info("因子值缓存淘汰旧条目: %s (%.0fMB)", key, size / 1e6)
                except OSError as exc:
                    logger.warning("因子值缓存淘汰失败 %s: %s", key, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("因子值缓存淘汰异常（忽略）: %s", exc)

    def _mark_dirty(self) -> None:
        """命中时更新 last_access 已改内存 manifest，节流落盘保持 LRU 顺序跨重启准确。"""
        now = time.time()
        if now - self._last_manifest_flush >= self._manifest_flush_interval:
            self._last_manifest_flush = now
            self._flush_manifest()

    # ------------------------------------------------------------------
    # 观测 / 清理
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = sum(int(e.get("bytes", 0) or 0) for e in self._manifest.values())
            return {
                "enabled": self.enabled,
                "memory_entries": len(self._mem),
                "disk_entries": len(self._manifest),
                "disk_bytes": total,
                "disk_max_bytes": self.max_bytes,
                "disk_max_files": self.max_files,
                "cache_dir": str(self.cache_dir),
            }

    def clear(self) -> None:
        """清空内存 + 磁盘缓存（保留目录）。"""
        with self._lock:
            self._mem.clear()
            self._manifest.clear()
            if self.cache_dir.is_dir():
                for p in self.cache_dir.glob("*"):
                    try:
                        if p.is_file():
                            p.unlink()
                    except OSError:
                        pass
            self._flush_manifest()


# 模块级默认实例（进程内共享；测试可自行 new）
_default_cache: FactorValueCache | None = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> FactorValueCache:
    global _default_cache
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                _default_cache = FactorValueCache()
    return _default_cache

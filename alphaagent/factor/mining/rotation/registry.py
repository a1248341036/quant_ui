# -*- coding: utf-8 -*-
"""旋转策略库轻量 registry（Phase 1 骨架，spec §3.9）。

与因子库平级、独立落盘：registry.json 即真源（不新建数据库）。职责：
- register / get / list / unregister / hash_for；
- mode 绑定：同一设计换 mode = 不同条目（编译校验按 mode 因子族）；
- active 单激活（per mode）；消费侧按 active 解析；
- 近重复查重：状态序列一致率 > 0.95 判近重复（拒收或标 variant_of），
  范围含内置 canonical 基线（内置 = 公共基线，不许换皮立项）。

Phase 1 只落 registry 元数据 + design/compiled JSON；状态序列可随条目存
`state_series`（查重用），组合权重表 weights.json 由 Phase 2 组合层消费。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROTATION_STRATEGIES_DIR = Path("artifacts/alphaagent/rotation_strategies")
REGISTRY_FILE = "registry.json"
NEAR_DUPLICATE_THRESHOLD = 0.95
CANONICAL_BASELINE_ID = "system_canonical"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"registry 无法序列化 {type(obj)}")


class RotationRegistryError(ValueError):
    """registry 操作非法（hash 冲突 / 近重复 / mode 冲突等）。"""


@dataclass(init=False)
class RotationRegistry:
    root: Path

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else ROTATION_STRATEGIES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    # ── 底层读写 ──────────────────────────────────────────────────────
    def _path(self) -> Path:
        return self.root / REGISTRY_FILE

    def _load(self) -> dict[str, Any]:
        p = self._path()
        if not p.exists():
            return {"entries": {}, "active": {}}
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("entries", {})
        data.setdefault("active", {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        # OCR(low,20261013)：registry.json 是真源，改原子写防截断（临时文件 + replace）
        p = self._path()
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ── 查询 ──────────────────────────────────────────────────────────
    def list(self, *, mode: str | None = None) -> list[dict[str, Any]]:
        entries = list(self._load()["entries"].values())
        if mode is not None:
            entries = [e for e in entries if e.get("mode") == mode]
        return sorted(entries, key=lambda e: str(e.get("strategy_id")))

    def get(self, strategy_id: str, *, mode: str | None = None) -> dict[str, Any] | None:
        entry = self._load()["entries"].get(strategy_id)
        if entry is None:
            return None
        if mode is not None and entry.get("mode") != mode:
            return None
        return entry

    def hash_for(self, strategy_id: str) -> dict[str, str] | None:
        entry = self._load()["entries"].get(strategy_id)
        if entry is None:
            return None
        return {
            "design": entry.get("design_fp"),
            "compile": entry.get("compile_fp"),
        }

    # ── 写操作 ────────────────────────────────────────────────────────
    def register(
        self,
        *,
        strategy_id: str,
        design_fp: str,
        compile_fp: str,
        mode: str,
        meta: dict[str, Any] | None = None,
        state_series: list[str] | None = None,
        allow_variant_of: bool = False,
    ) -> dict[str, Any]:
        """收录一个已编译策略。

        - 同 hash 不同 design = 冲突拒写（同 design 永远同 id，不应出现）；
        - mode 唯一：同 id 同 mode 重复收录幂等；同 id 不同 mode 视为不同条目
          （strategy_id 由 design(+mode) 派生，天然不同）；
        - 近重复（状态序列一致率 > 0.95，含 canonical）默认拒收；显式传
          allow_variant_of=True 时标 variant_of 收录。
        """

        data = self._load()
        cur = data["entries"].get(strategy_id)
        if cur is not None:
            if cur.get("design_fp") != design_fp or cur.get("compile_fp") != compile_fp:
                raise RotationRegistryError(
                    "hash 冲突：registry 已有同 id 但不同指纹的条目，拒绝覆盖"
                )
            if cur.get("mode") != mode:
                # OCR(medium,20261013)：同 id 不同 mode = 不同设计，拒绝静默错配
                raise RotationRegistryError(
                    f"mode 冲突：registry 已有同 id 但 mode={cur.get('mode')!r} 的条目，"
                    "同一设计换 mode 应派生不同 strategy_id，拒绝混录"
                )
            # 幂等：同 id 同 mode 同指纹重复收录为无操作
            return cur

        duplicate, variant_of = self.find_near_duplicate(
            state_series=state_series,
            exclude_ids={strategy_id},
        )
        if duplicate:
            if not allow_variant_of:
                raise RotationRegistryError(
                    f"近重复拒收：与 {variant_of} 状态序列一致率 > "
                    f"{NEAR_DUPLICATE_THRESHOLD}（防刷库；内置 canonical 为公共基线，"
                    "不被换皮立项）。确为有意变体请显式允许并标注 variant_of"
                )

        entry: dict[str, Any] = {
            "strategy_id": strategy_id,
            "design_fp": design_fp,
            "compile_fp": compile_fp,
            "mode": mode,
            "meta": meta or {},
            "status": "candidate",
        }
        if state_series is not None:
            entry["state_series"] = list(state_series)
        if duplicate:
            entry["variant_of"] = variant_of
        data["entries"][strategy_id] = entry
        self._save(data)
        return entry

    def register_canonical_baseline(self, state_series: list[str]) -> None:
        """收录内置 canonical 状态序列作为查重基线（公共基线，不允许换皮立项）。"""

        data = self._load()
        data["entries"][CANONICAL_BASELINE_ID] = {
            "strategy_id": CANONICAL_BASELINE_ID,
            "design_fp": "canonical",
            "compile_fp": "canonical",
            "mode": "system",
            "meta": {"note": "内置 canonical size/value_growth 基线"},
            "status": "system",
            "state_series": list(state_series),
        }
        self._save(data)

    def unregister(self, strategy_id: str) -> bool:
        data = self._load()
        if strategy_id in data["entries"]:
            del data["entries"][strategy_id]
            data["active"].pop(strategy_id, None)
            # OCR(low,20261013)：清掉指向被删条目的 variant_of 悬空引用
            for e in data["entries"].values():
                if e.get("variant_of") == strategy_id:
                    e.pop("variant_of", None)
            self._save(data)
            return True
        return False

    # ── active 单激活（per mode）──────────────────────────────────────
    def set_active(self, strategy_id: str, *, mode: str | None = None) -> None:
        data = self._load()
        entry = data["entries"].get(strategy_id)
        if entry is None:
            raise RotationRegistryError(f"strategy_id {strategy_id} 不存在")
        mode = mode or entry.get("mode")
        if mode is None:
            raise RotationRegistryError("无法解析 mode")
        if entry.get("mode") != mode:
            # OCR(medium,20261013)：mode 绑定契约——策略按自身 mode 登记，禁止跨 mode 顶替
            raise RotationRegistryError(
                f"strategy_id {strategy_id} 的 mode={entry.get('mode')!r} 与 "
                f"传入 mode={mode!r} 不一致，拒绝跨 mode 登记"
            )
        # 同 mode 单激活：其余条目撤销 active
        for sid, e in data["entries"].items():
            if e.get("mode") == mode and sid != strategy_id:
                data["active"].pop(sid, None)
        data["active"][strategy_id] = mode
        self._save(data)

    def active(self, mode: str) -> str | None:
        data = self._load()
        for sid, m in data["active"].items():
            if m == mode and sid in data["entries"]:
                return sid
        return None

    # ── 近重复查重（含 canonical）────────────────────────────────────
    def find_near_duplicate(
        self,
        *,
        state_series: list[str] | None,
        exclude_ids: set[str] | None = None,
    ) -> tuple[bool, str | None]:
        """返回 (is_duplicate, match_id)。一致率 = 同位置同状态占比（长度须一致）。"""

        if not state_series:
            return False, None
        exclude = set(exclude_ids or set())
        for sid, entry in self._load()["entries"].items():
            if sid in exclude:
                continue
            other = entry.get("state_series")
            if not isinstance(other, list) or not other:
                continue
            if len(other) != len(state_series):
                continue
            same = sum(1 for a, b in zip(other, state_series) if a == b)
            ratio = same / len(other)
            if ratio > NEAR_DUPLICATE_THRESHOLD:
                return True, sid
        return False, None


def save_design_json(
    root: str | Path,
    strategy_id: str,
    *,
    design: dict[str, Any],
    compiled: dict[str, Any] | None = None,
) -> Path:
    """落 strategy 目录：design.json + compiled.json（Phase 1 复现留档入口）。"""

    root = Path(root)
    strategy_dir = root / strategy_id
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if compiled is not None:
        (strategy_dir / "compiled.json").write_text(
            json.dumps(compiled, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    return strategy_dir

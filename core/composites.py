from __future__ import annotations

"""多因子自由组合配置的持久化与因子选项。

组合 = {因子名: 权重} + {因子名: 方向}。权重可为负（反向暴露），
方向 True 表示该因子买低（rank 翻转），False 表示买高。

因子清单统一来自 core.factor_registry（引擎因子注册表），
这里不再维护第二份因子列表。
"""

import json
import os
import tempfile

from .factor_registry import factor_options
from .store import DATA_DIR, STOCK_DIR


COMPOSITES_FILE = STOCK_DIR / "composites.json"


# 组合编辑器可用的因子选项（由引擎因子注册表派生，见 core.factor_registry）
FACTOR_OPTIONS: list[dict] = factor_options()


def _atomic_write(data: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), prefix=".composites.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, COMPOSITES_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_composites() -> dict[str, dict]:
    if not COMPOSITES_FILE.exists():
        return {}
    try:
        data = json.loads(COMPOSITES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_composite(name: str, weights: dict, directions: dict | None = None) -> dict:
    if not name.strip():
        raise ValueError("组合名称不能为空")
    if not weights:
        raise ValueError("组合至少需要一个因子")
    composites = load_composites()
    item = {
        "name": name.strip(),
        "weights": {str(k): float(v) for k, v in weights.items()},
        "directions": {str(k): bool(v) for k, v in (directions or {}).items()},
    }
    composites[item["name"]] = item
    _atomic_write(json.dumps(composites, ensure_ascii=False, indent=2))
    return item


def delete_composite(name: str) -> bool:
    composites = load_composites()
    if name not in composites:
        return False
    del composites[name]
    _atomic_write(json.dumps(composites, ensure_ascii=False, indent=2))
    return True

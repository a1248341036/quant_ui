"""因子库与数据面板的板块口径一致性检测。

因子库 canonical index 是建库那一刻的股票全集快照，因子值 memmap 按行号
绑死。当数据面板的板块过滤开关（QUANT_EXCLUDE_BSE / QUANT_EXCLUDE_KECHUANG /
QUANT_EXCLUDE_CHINEXT）与建库时不一致时，面板股票集合与因子库 index 错位，
因子值会串到错误的股票/日期上。

本模块提供轻量检测：读 manifest 里记录的 board_filter，与当前环境变量对比。
不一致时返回差异说明，由调用方决定警告或阻断。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alphaagent.data.adapters.cnequity import _board_filter_flag

_BOARD_LABELS: dict[str, str] = {
    "bse": "北交所",
    "kc": "科创板",
    "cyb": "创业板",
}


def _parse_board_filter(flag: str) -> dict[str, bool]:
    """把 board_filter 摘要（如 'bse1|kc0|cyb0'）解析成 {板块: 是否排除}。"""
    out: dict[str, bool] = {}
    for part in str(flag).split("|"):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("1")
        if key and val == "":
            out[key] = True
        else:
            key2, _, val2 = part.partition("0")
            if key2 and val2 == "":
                out[key2] = False
    return out


def check_board_consistency(root: Path) -> tuple[bool, str]:
    """检测因子库建库口径与当前板块开关是否一致。

    返回 (ok, detail)：ok=False 表示不一致，detail 说明差异。
    因子库未记录 board_filter（旧库）时视为一致（无法判断，不误报）。
    """
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return True, f"因子库未初始化: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return True, f"manifest 读取失败（跳过检测）: {exc}"
    stored = str((manifest.get("extra") or {}).get("board_filter") or "")
    if not stored:
        return True, "因子库未记录板块口径（旧库），跳过一致性检测"
    current = _board_filter_flag()
    if stored == current:
        return True, f"板块口径一致（{current}）"
    # 解析差异，给出可读说明
    stored_map = _parse_board_filter(stored)
    current_map = _parse_board_filter(current)
    diffs: list[str] = []
    for key, label in _BOARD_LABELS.items():
        s = stored_map.get(key)
        c = current_map.get(key)
        if s is None or c is None:
            continue
        if s != c:
            action = "排除" if c else "包含"
            diffs.append(f"{label}当前{action}，但因子库建库时{'排除' if s else '包含'}")
    detail = f"因子库口径 {stored} ≠ 当前 {current}"
    if diffs:
        detail += "；" + "；".join(diffs)
    detail += "。因子库 canonical index 与面板股票集合错位，因子值会串位，需重建因子库（scripts/migrate_factorzoo_drop_bse.py 或 init_library）"
    return False, detail


def warn_if_inconsistent(root: Path, *, logger: Any | None = None) -> bool:
    """检测并警告（不阻断）。返回是否一致。"""
    ok, detail = check_board_consistency(root)
    if not ok:
        msg = f"[board-consistency] {detail}"
        if logger is not None:
            logger.warning(msg)
        else:
            import logging

            logging.getLogger(__name__).warning(msg)
    return ok
# -*- coding: utf-8 -*-
"""产物落盘统一 writer（B-04，2026-10-26）。

问题：夜战与批处理脚本各自 ``json.dump`` 落盘产物，异常分支返回空 dict
也被静默写成 ``{}``（实测 rounds/b32_to16_jk_daily.json 仅 2 字节），空洞
产物污染归档面、误导下游对账（详见 docs/specs/alphaagent_known_bugs_spec.md
B-04）。

约定：产物落盘统一走 :func:`write_artifact_json`——
- 有内容：原样写 JSON，返回 True；
- 空内容且给了 ``empty_reason``：写 ``{"empty": true, "reason": ...}``，
  返回 False（合法的空，可审计）；
- 空内容且未给 reason：抛 :class:`EmptyArtifactError`（拒绝裸 ``{}``）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EmptyArtifactError(ValueError):
    """空产物且未声明 empty_reason——拒绝静默落盘裸 {}。"""


def write_artifact_json(
    path: str | Path,
    data: Any,
    *,
    empty_reason: str | None = None,
) -> bool:
    """带最小校验的 JSON 产物落盘。

    Parameters
    ----------
    path:
        目标文件路径（父目录自动创建）。
    data:
        待写对象（dict / list / 标量均可；``bool(data)`` 判定是否有内容）。
    empty_reason:
        内容为空时的声明性原因（如 "无满足条件样本"）。给出则写
        ``{"empty": true, "reason": ...}`` 并返回 False；缺失则抛
        :class:`EmptyArtifactError`。

    Returns
    -------
    bool
        True = 写入了有内容的产物；False = 写入了显式 empty 占位。
    """
    has_content = bool(data)
    if not has_content and not empty_reason:
        raise EmptyArtifactError(
            f"拒绝将空产物写入 {path}（B-04）：如确为合法空结果，"
            "请传 empty_reason 说明原因，产物将以 {empty:true, reason:...} 落盘"
        )
    payload: Any = data if has_content else {"empty": True, "reason": str(empty_reason)}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return has_content

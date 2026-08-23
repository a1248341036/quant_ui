"""因子元数据 catalog（factors.parquet）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.factor.zoo.types import FactorMeta, FactorStatus


def _empty_factors_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "factor_id",
            "name",
            "expr",
            "col_idx",
            "status",
            "finite_count",
            "created_at",
            "extra",
        ]
    )


class FactorCatalog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self._df = pd.read_parquet(self.path)
        else:
            self._df = _empty_factors_df()

    @property
    def n_factors(self) -> int:
        return len(self._df)

    def list_factor_ids(self) -> list[str]:
        if self._df.empty:
            return []
        return self._df["factor_id"].astype(str).tolist()

    def get(self, factor_id: str) -> FactorMeta | None:
        if self._df.empty:
            return None
        hit = self._df[self._df["factor_id"] == factor_id]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return FactorMeta(
            factor_id=str(row["factor_id"]),
            name=str(row["name"]),
            expr=str(row["expr"]),
            col_idx=int(row["col_idx"]),
            status=FactorStatus(str(row["status"])),
            finite_count=int(row.get("finite_count", 0)),
            created_at=str(row.get("created_at", "")),
            extra=_parse_extra(row.get("extra")),
        )

    def col_idx_for(self, factor_id: str) -> int | None:
        meta = self.get(factor_id)
        return None if meta is None else meta.col_idx

    def append(
        self,
        *,
        factor_id: str,
        name: str,
        expr: str,
        col_idx: int,
        status: FactorStatus,
        finite_count: int,
        extra: dict[str, Any] | None = None,
    ) -> FactorMeta:
        if not self._df.empty and factor_id in set(self._df["factor_id"].astype(str)):
            raise ValueError(f"factor_id 已存在: {factor_id}")
        created_at = datetime.now(timezone.utc).isoformat()
        row = {
            "factor_id": factor_id,
            "name": name,
            "expr": expr,
            "col_idx": int(col_idx),
            "status": status.value,
            "finite_count": int(finite_count),
            "created_at": created_at,
            "extra": json.dumps(extra, ensure_ascii=False) if extra else None,
        }
        self._df = pd.concat([self._df, pd.DataFrame([row])], ignore_index=True)
        self.save()
        return FactorMeta(
            factor_id=factor_id,
            name=name,
            expr=expr,
            col_idx=col_idx,
            status=status,
            finite_count=finite_count,
            created_at=created_at,
            extra=extra or {},
        )

    def remove(self, factor_id: str) -> None:
        if self._df.empty:
            raise KeyError(f"factor_id 不存在: {factor_id}")
        ids = set(self._df["factor_id"].astype(str))
        if factor_id not in ids:
            raise KeyError(f"factor_id 不存在: {factor_id}")
        self._df = self._df[self._df["factor_id"] != factor_id].reset_index(drop=True)
        self.save()

    def update(
        self,
        factor_id: str,
        *,
        name: str,
        expr: str,
        finite_count: int,
        status: FactorStatus,
        extra: dict[str, Any] | None = None,
    ) -> FactorMeta:
        if self._df.empty or factor_id not in set(self._df["factor_id"].astype(str)):
            raise KeyError(f"factor_id 不存在: {factor_id}")
        idx = self._df.index[self._df["factor_id"] == factor_id][0]
        row = self._df.loc[idx]
        col_idx = int(row["col_idx"])
        created_at = str(row.get("created_at", ""))
        self._df.at[idx, "name"] = name
        self._df.at[idx, "expr"] = expr
        self._df.at[idx, "finite_count"] = int(finite_count)
        self._df.at[idx, "status"] = status.value
        self._df.at[idx, "extra"] = json.dumps(extra, ensure_ascii=False) if extra else None
        self.save()
        return FactorMeta(
            factor_id=factor_id,
            name=name,
            expr=expr,
            col_idx=col_idx,
            status=status,
            finite_count=int(finite_count),
            created_at=created_at,
            extra=extra or {},
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(self.path, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


def _parse_extra(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return {}

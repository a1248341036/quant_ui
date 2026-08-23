"""因子入库类型与默认策略。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from alphaagent.factor.mining.context import StockEvalContext

DEFAULT_TRAIN_START = "2019-01-01"
DEFAULT_VAL_END = "2024-12-31"
DEFAULT_LABEL_COL = "label_1d_open_to_open"
DEFAULT_MAX_CS_CORR = 0.8
DEFAULT_SIMILAR_TOP_K = 3


@dataclass(frozen=True)
class IngestPolicy:
    train_start: str = DEFAULT_TRAIN_START
    val_end: str = DEFAULT_VAL_END
    label_col: str = DEFAULT_LABEL_COL
    max_cs_corr: float = DEFAULT_MAX_CS_CORR
    similar_top_k: int = DEFAULT_SIMILAR_TOP_K
    clip_pct: tuple[float, float] | None = None

    @property
    def mask_before_start(self) -> str:
        return self.train_start

    @property
    def eval_start(self) -> str:
        return self.train_start

    @property
    def eval_end(self) -> str:
        return self.val_end

    def to_ingest_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "label_col": self.label_col,
            "clip_pct": self.clip_pct,
            "mask_before_start": self.mask_before_start,
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
            "max_cs_corr": self.max_cs_corr,
            "similar_top_k": self.similar_top_k,
        }
        kw.update(overrides)
        return kw

    def ingest_config_dict(self) -> dict[str, str]:
        return {
            "train_start": self.train_start,
            "ingest_start": self.eval_start,
            "ingest_end": self.eval_end,
            "label_col": self.label_col,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("clip_pct") is not None:
            d["clip_pct"] = list(d["clip_pct"])
        return d

    @classmethod
    def from_context(cls, ctx: StockEvalContext, **overrides: Any) -> IngestPolicy:
        return cls(
            train_start=ctx.train_start,
            val_end=ctx.val_end,
            label_col=ctx.label_col,
            **overrides,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestPolicy:
        clip = data.get("clip_pct")
        if isinstance(clip, list) and len(clip) == 2:
            clip_pct: tuple[float, float] | None = (float(clip[0]), float(clip[1]))
        else:
            clip_pct = None
        return cls(
            train_start=str(data.get("train_start", DEFAULT_TRAIN_START)),
            val_end=str(data.get("val_end", DEFAULT_VAL_END)),
            label_col=str(data.get("label_col", DEFAULT_LABEL_COL)),
            max_cs_corr=float(data.get("max_cs_corr", DEFAULT_MAX_CS_CORR)),
            similar_top_k=int(data.get("similar_top_k", DEFAULT_SIMILAR_TOP_K)),
            clip_pct=clip_pct,
        )


DEFAULT_INGEST_POLICY = IngestPolicy()


@dataclass
class MaterializeResult:
    values: np.ndarray
    expr: str
    aux_tags: list[str]


@dataclass
class IngestResult:
    factor_id: str
    col_idx: int | None
    stored: bool
    skipped_reason: str | None
    metrics: dict[str, Any]
    similarity: dict[str, Any] | None
    extra: dict[str, Any]

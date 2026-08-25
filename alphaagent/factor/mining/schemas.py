"""挖掘评估 API 请求/响应 schema。"""

from __future__ import annotations

from dataclasses import dataclass

from alphaagent.factor.types import DEFAULT_LABEL_COL


@dataclass
class SessionCreateRequest:
    panel_path: str
    train_start: str = "2019-01-01"
    train_end: str = "2021-12-31"
    val_start: str = "2022-01-01"
    val_end: str = "2023-12-31"
    label_col: str = DEFAULT_LABEL_COL
    include_fundamentals: bool = True


@dataclass
class SessionCreateResponse:
    session_id: str
    panel_rows: int
    load_ms: float
    columns_sample: list[str]
    # 完整列清单（panel 加载后的真实字段），供提示词按可用性裁剪可选字段族文档。
    available_columns: list[str] | None = None


@dataclass
class EvalTrainRequest:
    session_id: str
    multi_line_expr: str
    factor_name: str = "expr"
    include_detail_tables: bool = False
    label_quantile_n: int = 10


@dataclass
class EvalValRequest:
    session_id: str
    multi_line_expr: str
    factor_name: str = "expr"
    include_detail_tables: bool = False
    label_quantile_n: int = 10
    expected_sign: int | None = None


@dataclass
class EvalProfileRequest:
    session_id: str
    profile_id: str
    multi_line_expr: str
    factor_name: str = "expr"

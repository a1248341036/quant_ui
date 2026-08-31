"""挖掘评估 API 请求/响应 schema。"""

from __future__ import annotations

from dataclasses import dataclass

from alphaagent.factor.types import (
    DEFAULT_LABEL_COL,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)


def _test_end_default(asset_type: str = "stock") -> str:
    """测试段右端默认：未显式传值时动态解析数据源最新交易日。"""
    from alphaagent.factor.window_config import resolve_test_end

    return resolve_test_end(asset_type=asset_type)


@dataclass
class SessionCreateRequest:
    panel_path: str
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    val_start: str = DEFAULT_VAL_START
    val_end: str = DEFAULT_VAL_END
    test_start: str = DEFAULT_TEST_START
    test_end: str | None = DEFAULT_TEST_END
    """测试段右端；None = 动态解析数据源最新交易日。"""
    label_col: str = DEFAULT_LABEL_COL
    include_fundamentals: bool = True
    asset_type: str = "stock"
    """资产类型：'stock'（默认）/ 'etf'。决定数据源与评估口径。"""

    def resolved_test_end(self) -> str:
        """解析后的测试段右端（None → 动态值）。"""
        return self.test_end or _test_end_default(self.asset_type)


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

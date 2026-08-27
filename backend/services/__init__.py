"""后端服务模块（包含通用服务和 AlphaAgent 服务）。"""

# ── 通用服务（从 backend.services_old 导入） ─────────────────────────────
from backend.services_old import (
    # 核心函数
    build_codes,
    normalize_universe,
    load_data,
    load_tech,
    load_etf,
    load_fund,
    load_universe,
    # 映射函数
    get_name_map,
    get_fund_name_map,
    get_industry_map,
    # 工具函数
    series_to_points,
    clean_records,
    _to_float,
    # 更新相关
    run_update_background,
    configured_update_tasks,
    run_configured_update_background,
    invalidate_data,
)

# ── AlphaAgent 服务（新模块化） ────────────────────────────────────────
from backend.services.session_manager import SessionManager, LRUSessionCache
from backend.services.factor_evaluator import FactorEvaluator
from backend.services.factor_repository import FactorRepository
from backend.services.factor_submitter import FactorSubmitter

__all__ = [
    # 通用服务 - 核心函数
    "build_codes",
    "normalize_universe",
    "load_data",
    "load_tech",
    "load_etf",
    "load_fund",
    "load_universe",
    # 通用服务 - 映射函数
    "get_name_map",
    "get_fund_name_map",
    "get_industry_map",
    # 通用服务 - 工具函数
    "series_to_points",
    "clean_records",
    "_to_float",
    # 通用服务 - 更新相关
    "run_update_background",
    "configured_update_tasks",
    "run_configured_update_background",
    "invalidate_data",
    # AlphaAgent 服务
    "SessionManager",
    "LRUSessionCache",
    "FactorEvaluator",
    "FactorRepository",
    "FactorSubmitter",
]

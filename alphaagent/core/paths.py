"""仓库路径常量。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
MARKET_DIR = ARTIFACTS_DIR / "market"
MARKET_HQ_PATH = MARKET_DIR / "daily_hq.parquet"
PANEL_PATH = ARTIFACTS_DIR / "panel" / "panel_1d.parquet"
FUNDAMENTAL_DIR = ARTIFACTS_DIR / "fundamental"
FUNDAMENTAL_QUARTERLY_PATH = FUNDAMENTAL_DIR / "quarterly.parquet"
DISCLOSURE_CALENDAR_PATH = FUNDAMENTAL_DIR / "disclosure_calendar.parquet"
INDUSTRY_DIR = ARTIFACTS_DIR / "industry"
INDUSTRY_SW_PATH = INDUSTRY_DIR / "sw_l1_membership.parquet"
INDEX_DIR = ARTIFACTS_DIR / "index"
FACTORZOO_DIR = ARTIFACTS_DIR / "factorzoo" / "stock_1d"
FACTOR_EXPR_DIR = FACTORZOO_DIR / "expressions"
CONFIGS_DIR = ROOT / "configs"
FACTOR_REGISTRY_EXAMPLE = CONFIGS_DIR / "factors" / "registry.example.json"
MLS_FMB_PERCENTILES_PATH = FACTORZOO_DIR / "mls_fmb_percentiles.json"
MINING_REGISTRY_PATH = FACTORZOO_DIR / "mining_delivered_registry.json"
MINING_EXPR_DIR = FACTOR_EXPR_DIR

"""backend.services_old compatibility shim.

All functions have migrated to backend.services.common.
"""
import backend.services.common as _c

load_tech = _c.load_tech
load_universe = _c.load_universe
load_etf = _c.load_etf
load_fund = _c.load_fund
load_panel = _c.load_panel
load_index = _c.load_index
load_etf_panel = _c.load_etf_panel
load_fund_nav = _c.load_fund_nav
load_fund_panel = _c.load_fund_panel
normalize_universe = _c.normalize_universe
load_data = _c.load_data
get_name_map = _c.get_name_map
get_fund_name_map = _c.get_fund_name_map
get_industry_map = _c.get_industry_map
series_to_points = _c.series_to_points
clean_records = _c.clean_records
_to_float = _c._to_float
run_update_background = _c.run_update_background
configured_update_tasks = _c.configured_update_tasks
run_configured_update_background = _c.run_configured_update_background
invalidate_data = _c.invalidate_data
DATA_CACHE = _c.DATA_CACHE
UPDATE_STATE = _c.UPDATE_STATE
CONFIG_UPDATE_STATES = _c.CONFIG_UPDATE_STATES

def build_codes(universe: str, exclude_kechuang: bool, panel=None):
    # Delegate to common, but if load_tech/load_universe/load_etf/load_fund were mocked on this module,
    # temporarily reflect those mocks onto common
    orig_tech = _c.load_tech
    orig_uni = _c.load_universe
    orig_etf = _c.load_etf
    orig_fund = _c.load_fund
    try:
        _c.load_tech = load_tech
        _c.load_universe = load_universe
        _c.load_etf = load_etf
        _c.load_fund = load_fund
        return _c.build_codes(universe, exclude_kechuang, panel=panel)
    finally:
        _c.load_tech = orig_tech
        _c.load_universe = orig_uni
        _c.load_etf = orig_etf
        _c.load_fund = orig_fund

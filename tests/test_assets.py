from core.assets import (ETF_PROFILE, FUND_NAV_PROFILE, STOCK_PROFILE,
                          get_execution_profile, panel_kind)


def test_execution_profiles_are_explicit():
    assert STOCK_PROFILE.lot_size == 100
    assert ETF_PROFILE.asset_type == "etf"
    assert FUND_NAV_PROFILE.lot_size == 1
    assert not FUND_NAV_PROFILE.uses_intraday_execution


def test_execution_profile_aliases():
    assert get_execution_profile("ETF") is ETF_PROFILE
    assert get_execution_profile("场外基金") is FUND_NAV_PROFILE
    assert panel_kind(FUND_NAV_PROFILE) == "nav"
    assert panel_kind(STOCK_PROFILE) == "ohlcv"

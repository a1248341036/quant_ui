"""测试后端通用服务函数（build_codes, normalize_universe 等）。"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


def test_normalize_universe():
    """测试 universe 名称标准化。"""
    from core.store import normalize_universe
    
    # 测试默认值
    assert normalize_universe("科技 TMT") == "科技 TMT"
    assert normalize_universe("中证 800") == "中证 800"
    assert normalize_universe("ETF") == "ETF"
    assert normalize_universe("场外基金") == "场外基金"


@patch("backend.services_old.load_tech")
@patch("core.data.load_panel_codes")
def test_build_codes_tech_universe(mock_panel_codes, mock_load_tech):
    """测试科技 TMT 股票池构建。"""
    from backend.services_old import build_codes
    
    # 模拟科技池数据
    mock_load_tech.return_value = pd.DataFrame({
        "code": ["000001", "000002", "688001", "300001"]  # 包含科创板和创业板
    })
    
    # 模拟 panel 代码
    mock_panel_codes.return_value = {"000001", "000002", "688001", "300001", "600000"}
    
    # 测试排除创业板和科创板（300, 301, 688, 689）
    codes = build_codes("科技 TMT", exclude_kechuang=True)
    assert "688001" not in codes  # 科创板被排除
    assert "300001" not in codes  # 创业板被排除
    assert "000001" in codes
    assert "000002" in codes
    
    # 测试不排除创业板和科创板
    codes_with_kechuang = build_codes("科技 TMT", exclude_kechuang=False)
    assert "688001" in codes_with_kechuang
    assert "300001" in codes_with_kechuang


@patch("backend.services_old.load_universe")
@patch("core.data.load_panel_codes")
def test_build_codes_default_universe(mock_panel_codes, mock_load_universe):
    """测试默认（中证 800）股票池构建。"""
    from backend.services_old import build_codes
    
    # 模拟中证 800 数据
    mock_load_universe.return_value = pd.DataFrame({
        "code": ["000001", "600000", "688001", "300001"]
    })
    
    # 模拟 panel 代码
    mock_panel_codes.return_value = {"000001", "600000", "688001", "300001"}
    
    # 测试排除创业板和科创板
    codes = build_codes("中证 800", exclude_kechuang=True)
    assert "688001" not in codes
    assert "300001" not in codes
    assert "000001" in codes
    assert "600000" in codes


@patch("backend.services_old.load_etf")
@patch("core.data.load_etf_panel_codes")
def test_build_codes_etf_universe(mock_etf_panel_codes, mock_load_etf):
    """测试 ETF 股票池构建。"""
    from backend.services_old import build_codes
    
    # 模拟 ETF 数据
    mock_load_etf.return_value = pd.DataFrame({
        "code": ["510300", "510500", "159915"]
    })
    
    # 模拟 ETF panel 代码
    mock_etf_panel_codes.return_value = {"510300", "510500", "159915", "518880"}
    
    # ETF 不受 exclude_kechuang 影响
    codes = build_codes("ETF", exclude_kechuang=True)
    assert codes == ["159915", "510300", "510500"]  # 已排序


@patch("backend.services_old.load_fund")
@patch("core.data.load_fund_nav_codes")
def test_build_codes_fund_universe(mock_fund_panel_codes, mock_load_fund):
    """测试场外基金池构建。"""
    from backend.services_old import build_codes
    
    # 模拟场外基金数据
    mock_load_fund.return_value = pd.DataFrame({
        "code": ["000001", "000002", "110001"]
    })
    
    # 模拟基金 panel 代码
    mock_fund_panel_codes.return_value = {"000001", "000002", "110001", "001234"}
    
    codes = build_codes("场外基金", exclude_kechuang=True)
    assert codes == ["000001", "000002", "110001"]  # 已排序


def test_build_codes_empty_panel_intersection():
    """测试当股票池与 panel 无交集时的处理。"""
    from backend.services_old import build_codes
    
    with patch("backend.services_old.load_tech") as mock_load_tech, \
         patch("core.data.load_panel_codes") as mock_panel_codes:
        
        # 科技池有数据
        mock_load_tech.return_value = pd.DataFrame({
            "code": ["000001", "000002"]
        })
        
        # panel 中没有这些股票
        mock_panel_codes.return_value = {"600000", "600001"}
        
        codes = build_codes("科技 TMT", exclude_kechuang=True)
        # 科技池代码与 panel 无交集，返回空列表
        assert codes == []


def test_build_codes_empty_universe():
    """测试当股票池为空时的处理。"""
    from backend.services_old import build_codes
    
    with patch("backend.services_old.load_tech") as mock_load_tech, \
         patch("core.data.load_panel_codes") as mock_panel_codes:
        
        # 科技池为空 DataFrame
        mock_load_tech.return_value = pd.DataFrame(columns=["code"])
        mock_panel_codes.return_value = {"000001", "000002"}
        
        codes = build_codes("科技 TMT", exclude_kechuang=True)
        # 空科技池与 panel 取交集，返回空列表
        assert codes == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

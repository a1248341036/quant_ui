"""测试会话 LRU 缓存功能。"""

import pytest
from unittest.mock import Mock, MagicMock
from backend.services.session_manager import SessionManager, LRUSessionCache


def test_lru_cache_basic():
    """测试 LRU 缓存的基本功能。"""
    cache = LRUSessionCache(max_size=3)
    
    # 创建模拟会话
    session1 = Mock(session_id="session1")
    session2 = Mock(session_id="session2")
    session3 = Mock(session_id="session3")
    
    # 加入缓存
    cache.put("key1", session1)
    cache.put("key2", session2)
    cache.put("key3", session3)
    
    # 验证缓存大小
    assert len(cache._cache) == 3
    
    # 验证获取
    assert cache.get("key1") == session1
    assert cache.get("key2") == session2
    assert cache.get("key3") == session3
    
    # 验证未命中
    assert cache.get("key4") is None


def test_lru_cache_eviction():
    """测试 LRU 缓存的淘汰机制。"""
    cache = LRUSessionCache(max_size=3)
    
    # 创建模拟会话
    session1 = Mock(session_id="session1")
    session2 = Mock(session_id="session2")
    session3 = Mock(session_id="session3")
    session4 = Mock(session_id="session4")
    
    # 加入 3 个会话
    cache.put("key1", session1)
    cache.put("key2", session2)
    cache.put("key3", session3)
    
    # 访问 key1，使其成为最近使用的
    cache.get("key1")
    
    # 加入第 4 个会话，应该淘汰最久未使用的 key2
    cache.put("key4", session4)
    
    # 验证 key2 被淘汰
    assert cache.get("key2") is None
    
    # 验证其他会话仍在缓存中
    assert cache.get("key1") == session1
    assert cache.get("key3") == session3
    assert cache.get("key4") == session4
    
    # 验证缓存大小仍为 3
    assert len(cache._cache) == 3


def test_lru_cache_update():
    """测试 LRU 缓存的更新机制。"""
    cache = LRUSessionCache(max_size=3)
    
    session1 = Mock(session_id="session1")
    session1_updated = Mock(session_id="session1_updated")
    
    # 加入会话
    cache.put("key1", session1)
    
    # 更新会话
    cache.put("key1", session1_updated)
    
    # 验证更新后的会话
    assert cache.get("key1") == session1_updated
    
    # 验证缓存大小不变
    assert len(cache._cache) == 1


def test_session_manager_hash_params():
    """测试会话参数哈希生成。"""
    manager = SessionManager(max_cached_sessions=3)
    
    from alphaagent.factor.mining.schemas import SessionCreateRequest
    
    req1 = SessionCreateRequest(
        panel_path="cne://",
        train_start="2020-01-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2024-12-31",
        label_col="label_1d_open_to_open",
        include_fundamentals=False,
    )
    
    req2 = SessionCreateRequest(
        panel_path="cne://",
        train_start="2020-01-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2024-12-31",
        label_col="label_1d_open_to_open",
        include_fundamentals=False,
    )
    
    req3 = SessionCreateRequest(
        panel_path="cne://",
        train_start="2019-01-01",  # 不同
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2024-12-31",
        label_col="label_1d_open_to_open",
        include_fundamentals=False,
    )
    
    # 相同参数应该生成相同哈希
    hash1 = manager._hash_params(req1)
    hash2 = manager._hash_params(req2)
    assert hash1 == hash2
    
    # 不同参数应该生成不同哈希
    hash3 = manager._hash_params(req3)
    assert hash1 != hash3


def test_session_manager_cache_stats():
    """测试会话缓存统计信息。"""
    manager = SessionManager(max_cached_sessions=3)
    
    stats = manager.get_cache_stats()
    
    assert "current_size" in stats
    assert "max_size" in stats
    assert "keys" in stats
    assert stats["max_size"] == 3
    assert stats["current_size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

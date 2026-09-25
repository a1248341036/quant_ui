"""访问门禁（backend.main.access_gate）回归测试。

覆盖两个真实发生过的问题：
1. 本机来源（127.0.0.1）提交登录表单时，POST /_gate/login 落到业务路由表 → 405 Method Not Allowed；
2. 门禁登录页没有 Cache-Control，被浏览器缓存后继续提交旧页面。

测试直接使用真实 app，但 monkeypatch 门禁模块级状态，不依赖 .env；
TestClient 不进 lifespan，因此不会建库、不会预启算力工作池。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.main as main

KEY = "test-key"
LOGIN = "/_gate/login"
LOOPBACK = ("127.0.0.1", 51234)
REMOTE = ("203.0.113.7", 51234)


def _client(addr: tuple[str, int]) -> TestClient:
    return TestClient(main.app, client=addr, follow_redirects=False)


@pytest.fixture()
def gate(monkeypatch):
    """门禁开启 + 本机来源免密放行（默认配置）。"""
    monkeypatch.setattr(main, "_ACCESS_KEY", KEY)
    monkeypatch.setattr(main, "_TRUST_LOOPBACK", True)


def test_loopback_login_post_is_not_405(gate):
    """回归：本机来源提交登录表单必须拿到 303 + Cookie，而不是 405。"""
    resp = _client(LOOPBACK).post(LOGIN, data={"password": KEY, "next": "/"})

    assert resp.status_code == 303, f"got {resp.status_code}: {resp.text[:200]}"
    assert resp.headers["location"] == "/"
    assert resp.cookies.get(main._ACCESS_COOKIE) == KEY


def test_loopback_login_wrong_password_renders_error(gate):
    resp = _client(LOOPBACK).post(LOGIN, data={"password": "wrong", "next": "/"})

    assert resp.status_code == 200
    assert "密码错误" in resp.text
    assert "访问验证" in resp.text


def test_remote_login_post_sets_cookie(gate):
    resp = _client(REMOTE).post(LOGIN, data={"password": KEY, "next": "/dashboard"})

    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard"
    assert resp.cookies.get(main._ACCESS_COOKIE) == KEY


def test_login_post_rejects_open_redirect(gate):
    """next 只接受站内相对路径。"""
    resp = _client(REMOTE).post(LOGIN, data={"password": KEY, "next": "//evil.example"})

    assert resp.status_code == 303
    assert resp.headers["location"] == "/"


def test_login_page_is_not_cacheable(gate):
    resp = _client(REMOTE).get("/", headers={"Accept": "text/html"})

    assert resp.status_code == 200
    assert "访问验证" in resp.text
    assert resp.headers.get("cache-control") == "no-store"


def test_get_gate_login_path_returns_page(gate):
    """直接敲 /_gate/login 也应给登录页，不再落到静态挂载。"""
    resp = _client(REMOTE).get(LOGIN, headers={"Accept": "text/html"})

    assert resp.status_code == 200
    assert "访问验证" in resp.text


def test_loopback_bypass_when_trusted(gate):
    resp = _client(LOOPBACK).get("/", headers={"Accept": "text/html"})

    assert "访问验证" not in resp.text


def test_loopback_is_gated_when_trust_disabled(gate, monkeypatch):
    monkeypatch.setattr(main, "_TRUST_LOOPBACK", False)

    resp = _client(LOOPBACK).get("/", headers={"Accept": "text/html"})

    assert resp.status_code == 200
    assert "访问验证" in resp.text


def test_valid_cookie_passes_gate(gate):
    client = _client(REMOTE)
    client.cookies.set(main._ACCESS_COOKIE, KEY)

    resp = client.get("/", headers={"Accept": "text/html"})

    assert "访问验证" not in resp.text


def test_remote_api_without_cookie_is_401(gate):
    resp = _client(REMOTE).get("/api/data/panel", headers={"Accept": "application/json"})

    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_health_is_never_gated(gate):
    resp = _client(REMOTE).get("/api/health")

    assert resp.status_code == 200


def test_gate_disabled_without_key(gate, monkeypatch):
    monkeypatch.setattr(main, "_ACCESS_KEY", "")

    assert "访问验证" not in _client(REMOTE).get("/", headers={"Accept": "text/html"}).text
    assert "访问验证" not in _client(LOOPBACK).get("/", headers={"Accept": "text/html"}).text


def test_loggable_predicate():
    """原判据恒为 False，api.log 长期零写入；这里锁死正确语义。"""
    assert main._loggable("/api/data/panel") is True
    assert main._loggable("/_gate/login") is True
    assert main._loggable("/unknown/route") is True
    assert main._loggable("/api/health") is False
    assert main._loggable("/assets/index-abc123.js") is False
    assert main._loggable("/favicon.ico") is False

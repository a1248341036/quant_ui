from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

# --- Load .env so QUANT_UI_PASSWORD can be configured there ---
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUTH_FILE = DATA_DIR / "db" / ".auth.json"
SECRET_FILE = DATA_DIR / "db" / ".secret"

COOKIE = "quant_ui_session"
TTL = 60 * 60 * 24 * 7  # 7 天
DEFAULT_USER = "root"
DEFAULT_PASSWORD = os.environ.get("QUANT_UI_PASSWORD", "")
if not DEFAULT_PASSWORD:
    raise RuntimeError(
        "QUANT_UI_PASSWORD 未设置。请在 .env 中配置 QUANT_UI_PASSWORD=你的密码"
    )

router = APIRouter()


def _get_secret() -> bytes:
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    DATA_DIR.mkdir(exist_ok=True)
    secret = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(secret)
    return secret


def _hash_password(password: str, salt: bytes, iterations: int = 100_000) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations).hex()


def ensure_auth() -> dict:
    """首次启动自动创建 root 账户（密码哈希持久化，不存明文）。"""
    if AUTH_FILE.exists():
        return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    DATA_DIR.mkdir(exist_ok=True)
    salt = secrets.token_bytes(16)
    users = {
        DEFAULT_USER: {
            "salt": salt.hex(),
            "hash": _hash_password(DEFAULT_PASSWORD, salt),
            "iterations": 100_000,
        }
    }
    AUTH_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    return users


def issue_token(username: str) -> str:
    payload = f"{username}|{int(time.time()) + TTL}"
    sig = hmac.new(_get_secret(), payload.encode(), hashlib.sha256).hexdigest()
    body = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{body}.{sig}"


def verify_token(token: str) -> str | None:
    if not token:
        return None
    try:
        body, sig = token.rsplit(".", 1)
        payload = base64.urlsafe_b64decode(body.encode()).decode()
        expect = hmac.new(_get_secret(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return None
        username, exp = payload.rsplit("|", 1)
        if int(exp) < time.time():
            return None
        return username
    except Exception:
        return None


def require_auth(request: Request) -> str:
    username = verify_token(request.cookies.get(COOKIE, ""))
    if not username:
        raise HTTPException(401, "未登录")
    return username


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(req: LoginRequest, response: Response):
    users = ensure_auth()
    user = users.get(req.username)
    if not user:
        raise HTTPException(401, "用户名或密码错误")
    salt = bytes.fromhex(user["salt"])
    got = _hash_password(req.password, salt, int(user.get("iterations", 100_000)))
    if not hmac.compare_digest(got, user["hash"]):
        raise HTTPException(401, "用户名或密码错误")
    token = issue_token(req.username)
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=TTL)
    return {"username": req.username}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    username = verify_token(request.cookies.get(COOKIE, ""))
    if not username:
        raise HTTPException(401, "未登录")
    return {"username": username}

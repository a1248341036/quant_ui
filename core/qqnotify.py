#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QQ 官方机器人私聊推送（HTTP 直连，不依赖 botpy）。

凭据从两个 .env 读取（项目 .env 之后加载 QQ 机器人目录 .env）：
    QQ_APP_ID
    QQ_APP_SECRET
    QQBOT_PUSH_OPENID（可选，缺省用 daily_market_push.py 的默认 openid）

用法:
    from core.qqnotify import send_qq_text
    send_qq_text("hello")
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from core.store import QQBOT_DIR  # noqa: E402

DEFAULT_OPENID = "22E35659FD854014125ED50661EDF28F"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(QQBOT_DIR / ".env")


def _access_token(appid: str, secret: str) -> str:
    body = json.dumps({"appId": appid, "clientSecret": secret}).encode("utf-8")
    req = urllib.request.Request(
        "https://bots.qq.com/app/getAppAccessToken",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"QQ 获取 access_token 失败: {data}")
    return token


def send_qq_text(text: str) -> bool:
    """推一条 QQ 私聊；未配置或失败只打印日志，不抛异常。返回是否成功。"""
    _load_env()
    appid = os.getenv("QQ_APP_ID", "").strip()
    secret = os.getenv("QQ_APP_SECRET", "").strip()
    openid = os.getenv("QQBOT_PUSH_OPENID", DEFAULT_OPENID).strip()
    if not (appid and secret and openid):
        print("[qqnotify] QQ 推送未配置，跳过（QQ_APP_ID / QQ_APP_SECRET / QQBOT_PUSH_OPENID）",
              flush=True)
        return False
    try:
        token = _access_token(appid, secret)
        body = json.dumps(
            {"content": text, "msg_type": 0, "msg_seq": 1},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.sgroup.qq.com/v2/users/{openid}/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"QQBot {token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
        print("[qqnotify] QQ 推送成功", flush=True)
        return True
    except Exception as exc:
        print(f"[qqnotify] QQ 推送失败: {exc}", flush=True)
        return False

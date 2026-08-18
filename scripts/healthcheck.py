#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quant_ui 健康检查：API + PostgreSQL + 数据新鲜度 + 磁盘。

失败时（可选）推送企业微信机器人 webhook。配在 .env：
  ALERT_WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx
失败时也可推送 QQ（官方 QQ 机器人，凭据在 $QUANT_UI_QQBOT_DIR/.env，默认 ~/qqbot）：
  QQ_APP_ID / QQ_APP_SECRET / QQBOT_PUSH_OPENID
由 systemd timer quant-healthcheck.timer 每 5 分钟执行一次。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.store import QQBOT_DIR  # noqa: E402

QQBOT_ROOT = QQBOT_DIR
STATE_FILE = ROOT / ".healthcheck_state.json"
if load_dotenv is not None:
    load_dotenv(ROOT / ".env")
    load_dotenv(QQBOT_ROOT / ".env")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def check_api() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/api/health",
                                    timeout=10) as r:
            return r.status == 200, f"API HTTP {r.status}"
    except Exception as exc:
        return False, f"API 不可达: {exc}"


def check_pg() -> tuple[bool, str]:
    dsn = os.getenv("PG_DSN", "").strip()
    if not dsn:
        return False, "PG_DSN 未配置"
    if psycopg is None:
        return False, "psycopg 未安装"
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.execute("SELECT max(trade_date) FROM stock_daily")
            latest = cur.fetchone()[0]
        if latest is not None:
            age = (datetime.now(timezone.utc).date() - latest).days
            if age > 5:
                return False, f"stock_daily 最新 {latest}，距今 {age} 天"
        return True, f"PG OK，最新数据 {latest}"
    except Exception as exc:
        return False, f"PG: {exc}"


def check_disk() -> tuple[bool, str]:
    st = os.statvfs(ROOT)
    pct = st.f_bavail / st.f_blocks * 100
    return pct >= 10, f"磁盘可用 {pct:.1f}%"


def send_alert(text: str) -> None:
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return
    payload = json.dumps(
        {"msgtype": "text", "text": {"content": text}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"[healthcheck] 通知失败: {exc}", flush=True)


def _qq_access_token(appid: str, secret: str) -> str:
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


def _qq_send_text(openid: str, token: str, text: str) -> None:
    body = json.dumps({"content": text, "msg_type": 0, "msg_seq": 1}).encode("utf-8")
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


def send_qq_alert(text: str) -> None:
    appid = os.getenv("QQ_APP_ID", "").strip()
    secret = os.getenv("QQ_APP_SECRET", "").strip()
    openid = os.getenv("QQBOT_PUSH_OPENID",
                       "22E35659FD854014125ED50661EDF28F").strip()
    if not (appid and secret and openid):
        print("[healthcheck] QQ 推送未配置，跳过（QQ_APP_ID / QQ_APP_SECRET / QQBOT_PUSH_OPENID）",
              flush=True)
        return
    try:
        token = _qq_access_token(appid, secret)
        _qq_send_text(openid, token, text)
        print("[healthcheck] QQ 推送成功", flush=True)
    except Exception as exc:
        print(f"[healthcheck] QQ 推送失败: {exc}", flush=True)


def main() -> int:
    checks = [
        ("quant-api", *check_api()),
        ("postgres", *check_pg()),
        ("disk", *check_disk()),
    ]
    failed = [f"{name}: {msg}" for name, ok, msg in checks if not ok]
    state = load_state()
    was_failed = bool(state.get("failed"))
    for name, ok, msg in checks:
        print(f"[{name}] {'OK' if ok else 'FAIL'} - {msg}", flush=True)
    if failed:
        text = f"[quant_ui] 健康检查失败\n" + "\n".join(failed)
        print(text, flush=True)
        if not was_failed:
            send_alert(text)
            send_qq_alert(text)
            save_state({"failed": True})
            print("[healthcheck] 状态变化: 健康→异常，已推送", flush=True)
        else:
            print("[healthcheck] 仍异常，跳过重复推送", flush=True)
        return 1
    if was_failed:
        save_state({})
        print("[healthcheck] 状态恢复: 异常→健康，已重置推送状态", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

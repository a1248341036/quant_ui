#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tushare token 到期 QQ 提醒。

在 quant_ui/.env 配置 TUSHARE_EXPIRE_DATE=YYYY-MM-DD 后，配合 systemd timer
每天检查一次，在到期前 7/3/1 天、到期当天、过期 3 天分别推送一次 QQ 私聊提醒。

用法:
    python scripts/tushare_expiry_notify.py            # 检查并推送（timer 调用）
    python scripts/tushare_expiry_notify.py --dry-run  # 只打印不发送
    python scripts/tushare_expiry_notify.py --force    # 忽略去重状态强制发送
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv


QUANT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUANT_ROOT))
from core.store import QQBOT_DIR  # noqa: E402

QQBOT_ROOT = QQBOT_DIR
STATE_FILE = QUANT_ROOT / ".tushare_expiry_state.json"

# 提醒节点：剩余天数阈值 -> 阶段 key（该阶段只提醒一次）
PHASES = [
    (7, "d7"),
    (3, "d3"),
    (1, "d1"),
    (0, "d0"),
    (-3, "overdue3"),
]


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


def phase_label(key: str, days_left: int) -> str:
    if key == "d7":
        return f"Tushare token 还有 {days_left} 天到期"
    if key == "d3":
        return f"Tushare token 还有 {days_left} 天到期"
    if key == "d1":
        return "Tushare token 明天到期"
    if key == "d0":
        return "Tushare token 今天到期"
    if key == "overdue3":
        return "Tushare token 已过期，请尽快续费"
    return "Tushare token 即将到期"


def pick_phase(days_left: int) -> tuple[int, str] | None:
    # 只在「离最近节点不远」的窗口内提醒，剩余太多天时不打扰
    if days_left > 7:
        return None
    if days_left > 3:
        return PHASES[0]
    if days_left > 1:
        return PHASES[1]
    if days_left > 0:
        return PHASES[2]
    if days_left == 0:
        return PHASES[3]
    if days_left >= -3:
        return PHASES[4]
    return None


def build_message(days_left: int, label: str) -> str:
    today = date.today().isoformat()
    if days_left >= 0:
        state = f"剩余 {days_left} 天"
        tip = "到期前记得在 Tushare 官网续费/续积分，否则数据接口会失效。"
    else:
        state = f"已过期 {-days_left} 天"
        tip = "请尽快续费，否则量化回测的数据拉取会报错。"
    return (
        f"⚠️ {label}\n\n"
        f"- 日期：{today}\n"
        f"- 状态：{state}\n"
        f"- 配置：quant_ui/.env → TUSHARE_EXPIRE_DATE\n\n"
        f"{tip}"
    )


async def send_qq(text: str) -> None:
    from botpy.api import BotAPI
    from botpy.http import BotHttp
    from botpy.robot import Token

    appid = os.getenv("QQ_APP_ID")
    secret = os.getenv("QQ_APP_SECRET")
    openid = os.getenv("QQBOT_PUSH_OPENID", "")
    if not appid or not secret or not openid:
        raise RuntimeError("缺少 QQ_APP_ID / QQ_APP_SECRET / QQBOT_PUSH_OPENID")
    http = BotHttp(timeout=30, app_id=appid, secret=secret)
    try:
        await http.login(Token(app_id=appid, secret=secret))
        api = BotAPI(http)
        await api.post_c2c_message(openid=openid, msg_type=0, content=text)
        print("PUSH_TEXT_OK")
    finally:
        await http.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tushare token 到期 QQ 提醒")
    parser.add_argument("--dry-run", action="store_true", help="只打印不发送")
    parser.add_argument("--force", action="store_true", help="忽略去重状态强制发送")
    args = parser.parse_args()

    load_dotenv(QUANT_ROOT / ".env")
    load_dotenv(QQBOT_ROOT / ".env")

    expire_raw = os.getenv("TUSHARE_EXPIRE_DATE", "").strip()
    if not expire_raw:
        print("[skip] 未配置 TUSHARE_EXPIRE_DATE（quant_ui/.env），无法提醒", file=sys.stderr)
        return 1
    try:
        expire = datetime.strptime(expire_raw, "%Y-%m-%d").date()
    except ValueError:
        print(f"[error] TUSHARE_EXPIRE_DATE 格式错误: {expire_raw!r}，应为 YYYY-MM-DD",
              file=sys.stderr)
        return 1

    today = date.today()
    days_left = (expire - today).days
    phase = pick_phase(days_left)
    if phase is None:
        print(f"[skip] 距离到期 {days_left} 天，不在提醒窗口内")
        return 0

    threshold, key = phase
    label = phase_label(key, days_left)
    state = load_state()
    today_key = today.isoformat()
    if state.get(key) == today_key and not args.force:
        print(f"[skip] {key} 今天已提醒过（{today_key}）")
        return 0

    msg = build_message(days_left, label)
    print(msg)
    if args.dry_run:
        print("[dry-run] 跳过发送")
        return 0

    try:
        asyncio.run(send_qq(msg))
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    state[key] = today_key
    save_state(state)
    print(f"[ok] 已推送并记录 {key} @ {today_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

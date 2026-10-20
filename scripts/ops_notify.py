#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一通知通道（quant-ops）：console + 落盘日志 + 可插拔推送（QQ 邮箱 / OneBot）。

设计约束：**通知永远不能弄挂调用方**——send() 对任何单通道失败吞异常、只记录状态；
未配置任何推送通道时自动退化为 console + 文件日志。

配置文件：~/.config/quant_ops/notify.json（缺省不存在 = 只 console + 落盘）::

    {
      "qqmail": {                       # QQ 邮箱 SMTP → 手机 QQ「邮箱提醒」弹通知
        "user": "you@qq.com",
        "auth_code": "QQ邮箱设置-账户-SMTP授权码",
        "to": ["you@qq.com"]
      },
      "onebot": {                       # NapCat/LLOneBot 等协议端的 OneBot 11 HTTP
        "http_url": "http://127.0.0.1:3000",
        "access_token": "",
        "msg_type": "private",          # private | group
        "target_id": 10001
      },
      "qqbot": true                     # QQ 官方机器人私聊推送（复用 healthcheck 的
                                        # ~/qqbot/.env 凭据，零额外配置，推荐）
    }

用法（模块 / CLI 等价）::

    from ops_notify import send
    send("整夜收尾完成", "run=4 候选+2 正式+1", level="info")

    python scripts/ops_notify.py --title "后端失联" --body "17891 5 分钟未就绪" --level error
"""
from __future__ import annotations

import argparse
import json
import smtplib
import sys
import urllib.request
from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path.home() / ".config" / "quant_ops" / "notify.json"
LOG_PATH = ROOT / "logs" / "ops_notify.log"

LEVELS = ("info", "warn", "error")

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _load_config(config_path: Path | str | None = None) -> dict:
    """读配置；文件缺失/损坏/类型不对（合法 JSON 但非 dict，如数组）一律返回 {}。"""
    p = Path(config_path) if config_path else CONFIG_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fmt(title: str, body: str, level: str) -> str:
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    return f"[{ts}][{level}] {title}" + (f"\n{body}" if body else "")


def _send_qqmail(cfg: dict, subject: str, text: str) -> str:
    user = cfg["user"]
    to = cfg.get("to") or [user]
    if isinstance(to, str):  # 配置误写成单个字符串时归一化，避免 join 按字符拆分
        to = [to]
    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = user
    msg["To"] = ", ".join(to)
    with smtplib.SMTP_SSL(cfg.get("host", "smtp.qq.com"),
                          int(cfg.get("port", 465)), timeout=20) as smtp:
        smtp.login(user, cfg["auth_code"])
        smtp.sendmail(user, to, msg.as_string())
    return "ok"


def _send_onebot(cfg: dict, text: str) -> str:
    url = cfg["http_url"].rstrip("/")
    action = "/send_private_msg" if cfg.get("msg_type", "private") == "private" else "/send_group_msg"
    key = "user_id" if action.endswith("private_msg") else "group_id"
    payload = {"action": action.strip("/"), "params": {key: cfg["target_id"], "message": text}}
    headers = {"Content-Type": "application/json"}
    if cfg.get("access_token"):
        headers["Authorization"] = f"Bearer {cfg['access_token']}"
    req = urllib.request.Request(url + action, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        # OneBot 11 HTTP：业务失败（target 不存在/风控/限频）也返回 HTTP 200，
        # 只有响应体 retcode != 0 才是失败——只看状态码会把失败误报为 ok
        try:
            data = json.loads(resp.read().decode("utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return f"error: bad resp body: {exc}"
        if int(data.get("retcode", -1)) != 0 or data.get("status") == "failed":
            return (f"error: retcode={data.get('retcode')} "
                    f"status={data.get('status')} {str(data.get('wording') or '')[:120]}").strip()
        return "ok"


def send(title: str, body: str = "", level: str = "info",
         config_path: Path | str | None = None) -> dict[str, str]:
    """发一条通知。返回 {通道: 状态}；单通道失败不抛异常。"""
    if level not in LEVELS:
        level = "info"
    text = _fmt(title, body, level)
    status: dict[str, str] = {"console": "ok"}

    print(text, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(text.replace("\n", " | ") + "\n")
        status["file"] = "ok"
    except Exception as exc:  # noqa: BLE001
        status["file"] = f"error: {exc}"

    cfg = _load_config(config_path)
    if cfg.get("qqmail"):
        try:
            status["qqmail"] = _send_qqmail(cfg["qqmail"], f"【quant-ops·{level}】{title}", text)
        except Exception as exc:  # noqa: BLE001
            status["qqmail"] = f"error: {type(exc).__name__}: {exc}"[:200]
    if cfg.get("onebot"):
        try:
            status["onebot"] = _send_onebot(cfg["onebot"], text)
        except Exception as exc:  # noqa: BLE001
            status["onebot"] = f"error: {type(exc).__name__}: {exc}"[:200]
    if cfg.get("qqbot"):
        try:
            # 任意入口可用：不依赖调用方是否已把 scripts/ 插入 sys.path
            _here = Path(__file__).resolve().parent
            if str(_here) not in sys.path:
                sys.path.insert(0, str(_here))
            import healthcheck  # 复用 QQ 官方机器人推送（凭据 ~/qqbot/.env）

            # send_qq_alert 内部失败只打印不抛出（fire-and-forget）——"ok" 表示
            # 已调用成功，送达与否以 healthcheck 控制台输出为准
            healthcheck.send_qq_alert(text)
            status["qqbot"] = "ok (fire-and-forget)"
        except Exception as exc:  # noqa: BLE001
            status["qqbot"] = f"error: {type(exc).__name__}: {exc}"[:200]
    if not (cfg.get("qqmail") or cfg.get("onebot") or cfg.get("qqbot")):
        status["push"] = "未配置推送通道（console+落盘已写）；模板见 scripts/ops_notify.py 头注释"
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="quant-ops 统一通知（详见模块头注释）")
    ap.add_argument("--title", required=True)
    ap.add_argument("--body", default="")
    ap.add_argument("--level", default="info", choices=LEVELS)
    args = ap.parse_args()
    for ch, st in send(args.title, args.body, args.level).items():
        print(f"  {ch}: {st}")
    return 0  # 通知失败也返回 0：不阻塞调用方


if __name__ == "__main__":
    raise SystemExit(main())

"""LLM 公共通道：load_codex_provider 上移 + chat_json 同步调用。

- load_codex_provider 从 scripts/run_alphaagent.py 上移，原脚本 re-export 保持行为不变
- chat_json 供训练脚本（llm_assist）单次补全调用，不引入 agentscope 框架
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from time import sleep


def load_codex_provider() -> None:
    """读 .env + ~/.codex/config.toml，注入 OPENAI_API_KEY / OPENAI_API_BASE / MODEL。

    与原 scripts/run_alphaagent.py 逻辑完全一致，仅位置上移。
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:  # noqa: BLE001
        pass
    if (os.getenv("ALPHA_LLM_PROVIDER") or "").lower() != "codex":
        return
    path = Path(os.getenv("CODEX_CONFIG", Path.home() / ".codex" / "config.toml"))
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    provider_name = config["model_provider"]
    provider = config["model_providers"][provider_name]
    token = provider.get("experimental_bearer_token") or config.get("experimental_bearer_token")
    if not token:
        raise RuntimeError(
            f"Codex provider '{provider_name}' has no bearer token in provider or top-level config"
        )
    os.environ["OPENAI_API_KEY"] = str(token)
    os.environ["OPENAI_API_BASE"] = str(provider["base_url"]).rstrip("/")
    os.environ["MODEL"] = str(config["model"])


def chat_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    retries: int = 2,
    timeout: float | None = None,
) -> dict | None:
    """同步调用 LLM 返回 JSON dict；失败返回 None（调用方兜底）。

    - max_tokens 显式传参（吸取挖掘链 max_tokens 断裂教训：多层默认值覆盖 config）
    - timeout 显式传参：None=用 OpenAI 默认（600s）；C 族说明书建议传 40（含重试外的单次超时）
    - 先试 response_format=json_object，中转不支持（400/422）时降级普通调用
    - 降级后用正则提取首个 JSON 对象，解析失败同样返回 None
    - 429/超时按 retries 退避重试（2 次，间隔 3s/9s）
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("MODEL", "")
    if not api_key or not model:
        return None

    timeout = timeout if timeout and timeout > 0 else None
    client = (
        OpenAI(api_key=api_key, base_url=base_url, timeout=timeout) if base_url
        else OpenAI(api_key=api_key, timeout=timeout)
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(retries + 1):
        try:
            # 先试 JSON mode
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            # 中转不支持 JSON mode → 降级普通调用
            if any(k in err for k in ("400", "422", "json_object", "response_format")):
                return _chat_json_fallback(client, model, messages, max_tokens, temperature)
            # 429/超时 → 退避重试
            if attempt < retries and any(k in err for k in ("429", "Timeout", "timed out")):
                sleep(3 * (attempt + 1))
                continue
            # 其他错误 → 尝试一次降级
            if attempt < retries:
                sleep(3 * (attempt + 1))
                continue
            return _chat_json_fallback(client, model, messages, max_tokens, temperature)

    return None


def _chat_json_fallback(
    client, model: str, messages: list, max_tokens: int, temperature: float,
) -> dict | None:
    """降级：普通补全 + 正则提取首个 JSON 对象（继承 client 的 timeout 配置）。"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        # 去掉 markdown 代码块包裹
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip())
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 正则提取首个 JSON 对象
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except Exception:  # noqa: BLE001
        return None

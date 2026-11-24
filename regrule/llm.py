"""大模型接入。用 LangChain 的 ChatOpenAI 连接任何兼容 OpenAI 格式的接口（DeepSeek、通义、Kimi……）。

DeepSeek 注意：思考模式默认开启，而思考模式下带 tools 的请求必须回传 reasoning_content，
ChatOpenAI 不会回传，会导致 HTTP 400。所以默认通过 extra_body 关闭思考模式（见 .env.example）。
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import llm_settings


def provider_status() -> dict:
    s = llm_settings()
    return {
        "configured": s.configured,
        "model": s.model,
        "base_url": s.base_url,
        "issue": "" if s.configured else "未配置大模型：请在项目根目录的 .env 里填写 LLM_API_KEY 等（参考 .env.example）。",
    }


def build_chat_model() -> BaseChatModel:
    s = llm_settings()
    if not s.configured:
        raise RuntimeError(provider_status()["issue"])
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.model,
        base_url=s.base_url,
        api_key=s.api_key,
        temperature=0,
        timeout=s.timeout,
        max_retries=1,
        extra_body=s.extra_body or None,
    )

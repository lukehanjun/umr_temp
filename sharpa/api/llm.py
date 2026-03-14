"""LLM wrapper — all language model calls go through call_llm()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    from sharpa.config import Settings

log = structlog.get_logger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_openrouter_model(model: str) -> bool:
    """Models with a provider prefix (e.g. 'google/gemini-2.5-pro') go via OpenRouter."""
    return "/" in model


async def call_llm(
    messages: list[dict],
    model: str,
    system: str | None = None,
    response_format: Literal["text", "json"] = "text",
    max_tokens: int = 2048,
    settings: Settings | None = None,
) -> str:
    """Call the LLM and return the response text.

    Routing:
    - Models with a provider prefix ('google/...', 'anthropic/...', etc.) are sent
      to OpenRouter using openrouter_api_key.
    - Plain model names ('gpt-4o', 'gpt-4-turbo', etc.) use OpenAI directly.

    messages may contain multimodal content (image_url entries) for vision models.
    Agents must not call any LLM SDK directly; use this wrapper.
    """
    from openai import AsyncOpenAI

    from sharpa.config import Settings as _Settings

    if settings is None:
        settings = _Settings()

    if _is_openrouter_model(model):
        client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )
        log.debug("call_llm.route", backend="openrouter", model=model)
    else:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        log.debug("call_llm.route", backend="openai", model=model)

    all_messages: list[dict] = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    log.debug("call_llm", model=model, n_messages=len(all_messages))

    kwargs: dict = {
        "model": model,
        "messages": all_messages,
        "max_tokens": max_tokens,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    log.debug("call_llm_done", model=model, chars=len(content))
    return content

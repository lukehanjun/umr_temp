"""Image generation wrapper — all image synthesis calls go through generate_image()."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from sharpa.config import Settings

log = structlog.get_logger(__name__)


def _extract_image_from_message(message: dict) -> bytes:
    """Pull base64 image bytes from an OpenRouter Gemini message dict.

    OpenRouter returns Gemini-generated images in a non-standard top-level
    `images` field on the message object (not inside `content`):

        message = {
            "role": "assistant",
            "content": "Here's your image: ",   # caption text — ignored
            "images": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,<DATA>"},
                 "index": 0}
            ]
        }
    """
    images = message.get("images") or []
    for item in images:
        if not isinstance(item, dict):
            continue
        url = (item.get("image_url") or {}).get("url", "")
        if url.startswith("data:"):
            _, encoded = url.split(",", 1)
            return base64.b64decode(encoded)

    # Fallback: some routing may embed images inside content parts
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    _, encoded = url.split(",", 1)
                    return base64.b64decode(encoded)

    content_preview = content[:200] if isinstance(content, str) else repr(content)[:200]
    raise ValueError(
        f"No image data found in Gemini message. "
        f"images={images!r:.200} content={content_preview!r}"
    )


def _pil_to_b64_jpeg(image: PILImage.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


async def generate_image(
    prompt: str,
    model: str,
    size: tuple[int, int] = (512, 512),
    reference_image: PILImage.Image | None = None,
    settings: Settings | None = None,
) -> PILImage.Image:
    """Generate or edit an image.

    When reference_image is provided the model receives the real photo as input
    and edits it — this is the primary mode for goal frame synthesis.
    When reference_image is None the model generates from scratch (fallback).

    Routes to the correct backend based on model name:
    - Gemini models (google/*): OpenRouter multimodal chat completions.
    - DALL-E models (dall-e-*): OpenAI images.generate (no editing support).

    Agents must not call any image generation SDK directly; use this wrapper.
    """
    import io as _io

    from PIL import Image

    from sharpa.config import Settings as _Settings

    if settings is None:
        settings = _Settings()

    mode = "edit" if reference_image is not None else "generate"
    log.info("generate_image", model=model, mode=mode, prompt_chars=len(prompt))

    if model.startswith("google/") or "gemini" in model:
        img_bytes = await _gemini_via_openrouter(prompt, model, settings, reference_image)
    else:
        img_bytes = await _generate_via_dalle(prompt, model, settings)

    img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
    log.debug("generate_image_done", final_size=img.size)
    return img


_GEMINI_MAX_RETRIES = 3


async def _gemini_via_openrouter(
    prompt: str,
    model: str,
    settings: Settings,
    reference_image: PILImage.Image | None = None,
) -> bytes:
    """Call OpenRouter with a Gemini image model.

    Edit mode (reference_image provided):
      Sends the real photo + a precise edit instruction. Gemini modifies only
      what the instruction specifies and preserves everything else.

    Generate mode (no reference_image):
      Pure text-to-image fallback.

    Gemini occasionally returns images=[] even on success — retried up to
    _GEMINI_MAX_RETRIES times with exponential backoff.
    """
    import asyncio

    import httpx

    if reference_image is not None:
        # Image editing: send the real photo and describe only what changes.
        b64 = _pil_to_b64_jpeg(reference_image)
        message_content: str | list = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
            {
                "type": "text",
                "text": (
                    "Edit this image precisely. "
                    "Apply ONLY the following change and keep everything else "
                    "exactly as it appears in the photo:\n\n"
                    f"{prompt}"
                ),
            },
        ]
        log.info("gemini_edit_request", model=model, prompt_chars=len(prompt))
    else:
        # Text-to-image fallback — force image output mode.
        message_content = f"Generate a photorealistic image showing: {prompt}"
        log.info("gemini_generate_request", model=model, prompt_chars=len(message_content))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
    }

    last_exc: Exception = RuntimeError("unreachable")
    async with httpx.AsyncClient(timeout=120.0) as http:
        for attempt in range(1, _GEMINI_MAX_RETRIES + 1):
            r = await http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if not r.is_success:
                log.error(
                    "gemini_http_error",
                    attempt=attempt,
                    status=r.status_code,
                    body=r.text[:600],
                )
                r.raise_for_status()

            data = r.json()
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"Unexpected OpenRouter response shape: {data!r:.400}"
                ) from exc

            try:
                return _extract_image_from_message(message)
            except ValueError as exc:
                last_exc = exc
                log.warning(
                    "gemini_empty_image_retry",
                    attempt=attempt,
                    max=_GEMINI_MAX_RETRIES,
                    reason=str(exc)[:120],
                )
                if attempt < _GEMINI_MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)

    raise last_exc


async def _generate_via_dalle(
    prompt: str,
    model: str,
    settings: Settings,
) -> bytes:
    """Call OpenAI images.generate (DALL-E 2/3) and return raw image bytes."""
    import httpx
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    log.info("dalle_image_url", url=(image_url or "")[:80])

    async with httpx.AsyncClient(timeout=60.0) as http:
        r = await http.get(image_url)  # type: ignore[arg-type]
        r.raise_for_status()
        return r.content

"""VLM wrapper - all vision-language model calls go through score_image_pair()."""

from __future__ import annotations

import json

from PIL import Image

from sharpa.api.llm import call_llm


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_score(raw: object, key: str) -> float:
    try:
        return _clamp_01(float(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid or missing score '{key}' in VLM response") from exc


async def score_image_pair(
    goal_image: Image.Image,
    actual_image: Image.Image,
    task_description: str,
    model: str,
) -> dict[str, float | str]:
    """Compare actual_image against goal_image and return 3 structured scores.

    Returns dict keys:
      - semantic: float in [0, 1]
      - pose: float in [0, 1]
      - contact: float in [0, 1]
      - reasoning: short explanation of the judgement

    Agents must not call any VLM SDK directly; use this wrapper.
    """
    import base64
    import io

    def _pil_to_b64_jpeg(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    goal_b64 = _pil_to_b64_jpeg(goal_image)
    actual_b64 = _pil_to_b64_jpeg(actual_image)

    system_prompt = (
        "You are a robot goal-frame verifier. Compare the intended goal image and the actual "
        "observation image. Score each dimension from 0.0 to 1.0 where 1.0 is a perfect match. "
        "Return strict JSON only with keys: semantic, pose, contact, reasoning."
    )

    user_content = [
        {"type": "text", "text": f"Task/subgoal: {task_description}"},
        {"type": "text", "text": "Goal state image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{goal_b64}"}},
        {"type": "text", "text": "Actual observation image:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{actual_b64}"}},
        {
            "type": "text",
            "text": (
                "Scoring rubric: semantic = object/state correctness; "
                "pose = geometric/positional alignment; "
                "contact = grasp/contact quality."
            ),
        },
    ]

    response_text = await call_llm(
        messages=[{"role": "user", "content": user_content}],
        model=model,
        system=system_prompt,
        response_format="json",
    )

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse VLM response as JSON: {response_text}") from exc

    semantic = _to_score(parsed.get("semantic"), "semantic")
    pose = _to_score(parsed.get("pose"), "pose")
    contact = _to_score(parsed.get("contact"), "contact")
    reasoning = str(parsed.get("reasoning", ""))

    return {
        "semantic": semantic,
        "pose": pose,
        "contact": contact,
        "reasoning": reasoning,
    }
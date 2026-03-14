"""GoalPlannerAgent — decomposes a task into a sequence of visual subgoals (GoalFrames)."""

from __future__ import annotations

import base64
import io
import json
import re

import structlog
from PIL import Image

from sharpa.api.imagegen import generate_image
from sharpa.api.llm import call_llm
from sharpa.config import Settings
from sharpa.models.episode import GoalFrame

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a robot manipulation planning assistant with expertise in spatial reasoning.

You will receive a real photo of a scene and a task description. Your job is to
decompose the task into the minimum sequence of discrete, observable intermediate
states required to accomplish it — no more, no fewer.

Each state will be rendered by an image-editing model that takes the PREVIOUS photo
and applies exactly one described change. Therefore:

- "description" must describe ONLY WHAT IS DIFFERENT from the previous state —
  not the whole scene. Be precise about the single physical change that occurs.
- Describe changes in terms of physical state (position, orientation, contact,
  support) — not robot actions ("pick up", "move").
- "key_visual_features" lists 2–4 things a verifier should check in the result.

Stopping rule — ask yourself after each step:
  "Does the scene now satisfy the task goal, with no further changes needed?"
  If yes, that is your LAST step. Do not add steps that are unnecessary or that
  merely rearrange things without making progress toward the goal.

Respond in EXACTLY this format, with no text before or after:

<thinking>
  [Reason about: what the current scene looks like, what physical changes are
  required to achieve the goal, what the scene must look like after each step,
  stability/contact constraints, and — critically — when you have reached the
  goal state and no more steps are needed.]
</thinking>
<goals>
  [
    {
      "step": 1,
      "description": "...",
      "key_visual_features": ["...", "..."]
    }
  ]
</goals>
"""


def _encode_image(image: Image.Image) -> str:
    """Return a base64-encoded JPEG string (no data-URL prefix) for the image."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_response(raw: str) -> tuple[str, list[dict]]:
    """Extract (thinking, goals_list) from the LLM XML response."""
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    goals_match = re.search(r"<goals>(.*?)</goals>", raw, re.DOTALL)
    if not goals_match:
        raise ValueError(f"No <goals> block found in LLM response:\n{raw[:500]}")

    goals_raw = goals_match.group(1).strip()
    try:
        goals = json.loads(goals_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse goals JSON: {exc}\nRaw:\n{goals_raw[:300]}") from exc

    if not isinstance(goals, list):
        raise ValueError(f"Expected a JSON array of goals, got: {type(goals)}")

    return thinking, goals


class GoalPlannerAgent:
    """Decomposes a high-level task description into a sequence of GoalFrames.

    Each frame is produced by editing the previous frame (or the initial image
    for step 1) — not by generating from scratch. This keeps the scene grounded
    in reality: lighting, table texture, surrounding objects and their positions
    are all preserved; only the manipulated objects change state.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def plan(
        self,
        initial_image: Image.Image,
        task_description: str,
        current_state_image: Image.Image | None = None,
        failed_frames: list[GoalFrame] | None = None,
    ) -> list[GoalFrame]:
        """Return a sequence of GoalFrames for the given task.

        Goal frame images are produced as a chain of edits:
          initial_image → frame_1 → frame_2 → ... → frame_N

        When current_state_image is provided (replan), planning starts from that
        image instead of initial_image.  failed_frames are injected as negative
        examples so the planner avoids repeating them.
        """
        is_replan = current_state_image is not None
        start_image = current_state_image if is_replan else initial_image

        log.info(
            "planner.plan",
            task=task_description[:60],
            n_goals=self.settings.n_goals,
            is_replan=is_replan,
        )

        # --- build LLM prompt -------------------------------------------
        user_content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(initial_image)}"},
            },
            {
                "type": "text",
                "text": "This is the initial scene image." if not is_replan
                        else "This is the initial scene image (for reference).",
            },
        ]

        if is_replan and current_state_image is not None:
            user_content += [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_encode_image(current_state_image)}"
                    },
                },
                {
                    "type": "text",
                    "text": "This is the CURRENT scene state. Plan from here.",
                },
            ]

        if failed_frames:
            failed_desc = "\n".join(
                f"  - Step {f.step}: {f.description}" for f in failed_frames
            )
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "The following subgoals were attempted and FAILED — "
                        f"do not repeat them:\n{failed_desc}"
                    ),
                }
            )

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"Task: {task_description}\n\n"
                    "Generate the minimum number of sequential subgoals needed to "
                    "accomplish this task. Stop as soon as the goal state is reached."
                ),
            }
        )

        raw = await call_llm(
            messages=[{"role": "user", "content": user_content}],
            model=self.settings.planner_llm,
            system=_SYSTEM_PROMPT,
            max_tokens=4096,
            settings=self.settings,
        )

        log.debug("planner.raw_response", chars=len(raw))
        thinking, goals = _parse_response(raw)
        log.info("planner.parsed", n_subgoals=len(goals), thinking_chars=len(thinking))

        # Safety ceiling — the model self-determines step count, but cap runaway outputs.
        if len(goals) > self.settings.n_goals:
            log.warning(
                "planner.truncated",
                original=len(goals),
                cap=self.settings.n_goals,
            )
            goals = goals[: self.settings.n_goals]

        # --- generate frames as a chain of image edits -------------------
        frames: list[GoalFrame] = []
        prev_image: Image.Image = start_image  # edit chain starts from real photo

        for g in goals:
            log.info(
                "planner.editing_frame",
                step=g["step"],
                change=g["description"][:70],
            )
            img = await generate_image(
                prompt=g["description"],
                model=self.settings.imagegen_model,
                reference_image=prev_image,   # ← always edit from current state
                settings=self.settings,
            )
            frame = GoalFrame(
                step=g["step"],
                description=g["description"],
                key_visual_features=g.get("key_visual_features", []),
                reasoning=thinking,
                image=img
            )
            frames.append(frame)
            prev_image = img  # next edit builds on this frame
            log.info("planner.frame_ready", step=g["step"])

        log.info("planner.done", total_frames=len(frames))
        return frames

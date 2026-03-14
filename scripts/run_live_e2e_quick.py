"""Live end-to-end SHARPA check against ManiSkill with real API calls.

This script intentionally uses real planner/verifier/image generation APIs.
It is configured to keep costs low:
- low `n_goals`
- low `max_retries`
- short simulation horizon
- explorer disabled

Usage:
  uv run scripts/run_live_e2e_quick.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from sharpa.config import Settings
from sharpa.interface.policy_bridge import SimPolicyBridge
from sharpa.loop import run_episode

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
LIVE_RUNS_DIR = DATA_DIR / "live_e2e_runs"

DEFAULT_TASK = (
    "Stack the movable cube on top of the target cube and leave a stable stack "
    "near the center of the workspace."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="High-level task description.")
    parser.add_argument(
        "--env-id",
        default="StackCube-v1",
        help="ManiSkill environment id (moderately complex default).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Hard timeout so live tests do not run indefinitely.",
    )
    parser.add_argument(
        "--n-goals",
        type=int,
        default=2,
        help="Planner goal cap (kept low to reduce API usage).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Verifier retries per frame (kept low to reduce API usage).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Policy rollout horizon per goal step.",
    )
    parser.add_argument(
        "--advance-threshold",
        type=float,
        default=0.0,
        help=(
            "Advance threshold for this quick live check. "
            "Default 0.0 guarantees finite progression and bounded API spend."
        ),
    )
    return parser.parse_args()


def episode_summary(episode: Any, bridge: SimPolicyBridge) -> dict[str, Any]:
    return {
        "episode_id": episode.id,
        "task_description": episode.task_description,
        "success": episode.success,
        "num_goal_frames": len(episode.goal_frames),
        "num_verifications": len(episode.verifications),
        "verifications": [
            {
                "step": v.goal_frame.step,
                "decision": v.decision,
                "semantic_score": v.semantic_score,
                "pose_score": v.pose_score,
                "contact_score": v.contact_score,
                "composite_score": v.composite_score,
                "retry_count": v.retry_count,
                "reasoning": v.reasoning,
                "num_action_frames": len(bridge.get_action_frames_for_goal(v.goal_frame.step)),
            }
            for v in episode.verifications
        ],
    }


async def main() -> None:
    args = parse_args()
    settings = Settings()

    settings.sim_env_id = args.env_id
    settings.sim_policy_mode = "random"
    settings.sim_obs_mode = "state"
    settings.sim_render_mode = "rgb_array"
    settings.sim_policy_step_horizon = args.horizon

    settings.n_goals = args.n_goals
    settings.max_retries = args.max_retries
    settings.advance_threshold = args.advance_threshold
    settings.epsilon_explore = 0.0  # Explorer disabled (not implemented yet)

    log.info(
        "live_e2e.start",
        task=args.task,
        env_id=settings.sim_env_id,
        n_goals=settings.n_goals,
        max_retries=settings.max_retries,
        horizon=settings.sim_policy_step_horizon,
        epsilon_explore=settings.epsilon_explore,
        advance_threshold=settings.advance_threshold,
        planner_llm=settings.planner_llm,
        verifier_vlm=settings.verifier_vlm,
        imagegen_model=settings.imagegen_model,
    )

    bridge = SimPolicyBridge(settings=settings)
    try:
        episode = await asyncio.wait_for(
            run_episode(task_description=args.task, settings=settings, bridge=bridge),
            timeout=args.timeout_sec,
        )
    finally:
        if hasattr(bridge, "_env"):
            bridge._env.close()  # noqa: SLF001 - explicit simulation cleanup path

    run_dir = LIVE_RUNS_DIR / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    goal_meta: list[dict[str, Any]] = []
    for frame in episode.goal_frames:
        frame_dir = run_dir / f"goal_step_{frame.step:02d}"
        action_dir = frame_dir / "actions"
        action_dir.mkdir(parents=True, exist_ok=True)

        frame.image.save(frame_dir / "goal.jpg", "JPEG", quality=92)

        goal_payload = {
            "step": frame.step,
            "description": frame.description,
            "key_visual_features": frame.key_visual_features,
            "reasoning": frame.reasoning,
        }
        (frame_dir / "goal_metadata.json").write_text(
            json.dumps(goal_payload, indent=2),
            encoding="utf-8",
        )
        goal_meta.append(goal_payload)

        action_frames = bridge.get_action_frames_for_goal(frame.step)
        for idx, action_img in enumerate(action_frames, start=1):
            action_img.save(action_dir / f"action_{idx:03d}.jpg", "JPEG", quality=92)

    for idx, verification in enumerate(episode.verifications, start=1):
        verification.actual_image.save(
            run_dir / f"actual_{idx:02d}_step_{verification.goal_frame.step:02d}.jpg",
            "JPEG",
            quality=92,
        )

    reasoning_trace = ""
    if episode.goal_frames:
        reasoning_trace = episode.goal_frames[0].reasoning

    (run_dir / "planner_reasoning.txt").write_text(reasoning_trace, encoding="utf-8")
    (run_dir / "goal_frames_metadata.json").write_text(
        json.dumps(goal_meta, indent=2),
        encoding="utf-8",
    )

    summary = episode_summary(episode, bridge)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nLive E2E artifacts written to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
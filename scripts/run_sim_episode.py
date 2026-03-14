"""Run a full simulator-backed SHARPA episode (planner + verifier + loop + SimPolicyBridge)."""

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
DEFAULT_TEXT = DATA_DIR / "input_text.txt"
SIM_RUNS_DIR = DATA_DIR / "sim_runs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task description. If omitted, --task-file is used.",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        default=DEFAULT_TEXT,
        help="Path to task description text file.",
    )
    parser.add_argument("--env-id", default=None, help="Override settings.sim_env_id")
    parser.add_argument(
        "--policy-mode",
        choices=["random", "ppo"],
        default=None,
        help="Override settings.sim_policy_mode",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for PPO mode.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override settings.sim_policy_step_horizon",
    )
    return parser.parse_args()


def _load_task(args: argparse.Namespace) -> str:
    if args.task:
        return str(args.task).strip()
    if not args.task_file.exists():
        raise FileNotFoundError(f"Task file not found: {args.task_file}")
    return args.task_file.read_text(encoding="utf-8").strip()


def _episode_summary(episode: Any, bridge: SimPolicyBridge) -> dict[str, Any]:
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
    args = _parse_args()
    settings = Settings()

    if args.env_id:
        settings.sim_env_id = args.env_id
    if args.policy_mode:
        settings.sim_policy_mode = args.policy_mode
    if args.ppo_checkpoint:
        settings.sim_ppo_checkpoint = str(args.ppo_checkpoint)
    if args.horizon is not None:
        settings.sim_policy_step_horizon = args.horizon

    task_description = _load_task(args)
    log.info(
        "sim_episode.start",
        task=task_description,
        env_id=settings.sim_env_id,
        policy_mode=settings.sim_policy_mode,
    )

    bridge = SimPolicyBridge(settings=settings)
    try:
        episode = await run_episode(
            task_description=task_description,
            settings=settings,
            bridge=bridge,
        )
    finally:
        if hasattr(bridge, "_env"):
            bridge._env.close()  # noqa: SLF001 - explicit simulation cleanup

    run_dir = SIM_RUNS_DIR / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for frame in episode.goal_frames:
        frame_dir = run_dir / f"goal_step_{frame.step:02d}"
        action_dir = frame_dir / "actions"
        action_dir.mkdir(parents=True, exist_ok=True)

        frame.image.save(frame_dir / "goal.jpg", "JPEG", quality=92)

        action_frames = bridge.get_action_frames_for_goal(frame.step)
        for idx, action_img in enumerate(action_frames, start=1):
            action_img.save(action_dir / f"action_{idx:03d}.jpg", "JPEG", quality=92)

    for idx, verification in enumerate(episode.verifications, start=1):
        verification.actual_image.save(
            run_dir / f"actual_{idx:02d}_step_{verification.goal_frame.step:02d}.jpg",
            "JPEG",
            quality=92,
        )

    summary = _episode_summary(episode, bridge)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
"""Run a full simulator-backed SHARPA episode (planner + verifier + loop + SimPolicyBridge)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

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


def _normalize_gpu_ids(gpu_ids: str) -> str:
    parts = [p.strip() for p in gpu_ids.split(",") if p.strip()]
    if not parts:
        raise ValueError("GPU list is empty.")
    normalized: list[str] = []
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"GPU id must be an integer, got: {p!r}")
        idx = int(p)
        if idx < 0 or idx > 7:
            raise ValueError(f"GPU id out of range [0,7]: {idx}")
        normalized.append(str(idx))
    return ",".join(normalized)


def _apply_gpu_selection(gpu_ids: str | None) -> str | None:
    if gpu_ids is None or not str(gpu_ids).strip():
        return None
    normalized = _normalize_gpu_ids(str(gpu_ids))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = normalized
    return normalized


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
        choices=["random", "ppo", "pi05"],
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
    parser.add_argument(
        "--gpus",
        default=None,
        help='GPU ids to expose for this run (e.g. "0" or "0,1").',
    )
    parser.add_argument(
        "--openpi-config",
        default=None,
        help="OpenPI config name (e.g. pi05_droid). Used when --policy-mode=pi05.",
    )
    parser.add_argument(
        "--openpi-checkpoint",
        default=None,
        help="OpenPI checkpoint dir/URI. Used when --policy-mode=pi05.",
    )
    parser.add_argument(
        "--openpi-prompt",
        default=None,
        help="Default language prompt passed to PI0.5.",
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
        "termination_reason": episode.termination_reason,
        "replan_count": episode.replan_count,
        "planning_rounds": episode.planning_rounds,
        "total_attempts": episode.total_attempts,
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


def _save_demo_gif(frames: list[Image.Image], output_path: Path, fps: int = 20) -> None:
    if not frames:
        return
    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _run_demo_rollout(bridge: SimPolicyBridge, steps: int = 200) -> list[Image.Image]:
    demo_frames: list[Image.Image] = []
    if not hasattr(bridge, "_env"):
        return demo_frames

    obs, _ = bridge._env.reset()  # noqa: SLF001
    bridge._latest_obs = obs  # noqa: SLF001
    bridge._capture_frame()  # noqa: SLF001
    demo_frames.append(bridge._latest_frame.copy())  # noqa: SLF001

    for _ in range(max(1, steps)):
        action = bridge._select_action(bridge._latest_obs)  # noqa: SLF001
        bridge._latest_obs, _, terminated, truncated, _ = bridge._env.step(action)  # noqa: SLF001
        bridge._capture_frame()  # noqa: SLF001
        demo_frames.append(bridge._latest_frame.copy())  # noqa: SLF001
        if terminated or truncated:
            break
    return demo_frames


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
    if args.gpus:
        settings.sim_gpu_ids = args.gpus
    if args.openpi_config:
        settings.sim_openpi_config_name = args.openpi_config
    if args.openpi_checkpoint:
        settings.sim_openpi_checkpoint = args.openpi_checkpoint
    if args.openpi_prompt:
        settings.sim_openpi_prompt = args.openpi_prompt
    if settings.sim_policy_mode.lower().strip() == "pi05" and settings.sim_obs_mode == "state":
        settings.sim_obs_mode = "rgb+depth"
        log.info("sim_episode.obs_mode_auto_set", sim_obs_mode=settings.sim_obs_mode)
    selected_gpus = _apply_gpu_selection(settings.sim_gpu_ids)
    if selected_gpus is not None:
        log.info("sim_episode.gpu_selection", cuda_visible_devices=selected_gpus)

    task_description = _load_task(args)
    run_dir = SIM_RUNS_DIR / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "sim_episode.start",
        task=task_description,
        env_id=settings.sim_env_id,
        policy_mode=settings.sim_policy_mode,
        gpus=selected_gpus,
    )

    bridge = SimPolicyBridge(settings=settings)
    try:
        episode = await run_episode(
            task_description=task_description,
            settings=settings,
            bridge=bridge,
            artifact_dir=run_dir,
        )
        demo_frames = _run_demo_rollout(bridge, steps=200)
        _save_demo_gif(demo_frames, run_dir / "demo_policy.gif", fps=20)
    finally:
        if hasattr(bridge, "_env"):
            bridge._env.close()  # noqa: SLF001 - explicit simulation cleanup

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

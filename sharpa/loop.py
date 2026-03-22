"""Orchestration loop.

Ties GoalPlannerAgent, VerifierAgent, and ExplorerAgent together per episode.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import structlog
from PIL import Image
from PIL import ImageChops

from sharpa.agents.explorer import ExplorerAgent
from sharpa.agents.planner import GoalPlannerAgent
from sharpa.agents.verifier import VerifierAgent
from sharpa.config import Settings
from sharpa.interface.policy_bridge import PolicyBridgeBase, StubPolicyBridge
from sharpa.models.episode import Episode, GoalFrame

log = structlog.get_logger(__name__)


async def run_episode(
    task_description: str,
    settings: Settings,
    initial_image: Image.Image | None = None,
    bridge: PolicyBridgeBase | None = None,
    artifact_dir: Path | None = None,
) -> Episode:
    """Run a full episode from task description to completion or failure."""
    planner = GoalPlannerAgent(settings)
    verifier = VerifierAgent(settings)
    explorer = ExplorerAgent(settings)
    bridge = bridge or StubPolicyBridge()

    if initial_image is None:
        initial_image = await bridge.get_observation()

    frames = await planner.plan(initial_image, task_description)

    episode = Episode(
        task_description=task_description,
        initial_image=initial_image,
        goal_frames=frames,
    )

    current_index = 0
    retry_count = 0
    replan_count = 0
    attempt_index = 0
    planning_round = 0

    planner_round_meta: dict[int, dict] = {}
    source_img_by_round_step: dict[tuple[int, int], Image.Image] = {}

    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_planner_round(
        round_idx: int,
        planning_start_image: Image.Image,
        planned_frames: list[GoalFrame],
        *,
        is_replan: bool,
        failed_frames: list[GoalFrame] | None,
    ) -> None:
        if artifact_dir is None:
            return

        plan_dir = artifact_dir / "planner" / f"round_{round_idx:03d}"
        plan_dir.mkdir(parents=True, exist_ok=True)
        planning_start_image.save(plan_dir / "start_image.jpg", "JPEG", quality=92)

        input_payload = {
            "planning_round": round_idx,
            "is_replan": is_replan,
            "task_description": task_description,
            "failed_frames": [
                {
                    "step": f.step,
                    "description": f.description,
                    "key_visual_features": f.key_visual_features,
                }
                for f in (failed_frames or [])
            ],
            "settings": {
                "planner_llm": settings.planner_llm,
                "imagegen_model": settings.imagegen_model,
                "n_goals_cap": settings.n_goals,
            },
        }
        _write_json(plan_dir / "planner_input.json", input_payload)

        goals_payload: list[dict] = []
        prev_image = planning_start_image
        for frame in planned_frames:
            frame_dir = plan_dir / f"goal_{frame.step:02d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            source_path = frame_dir / "old_image.jpg"
            goal_path = frame_dir / "new_image.jpg"
            diff_path = frame_dir / "image_diff.jpg"
            prev_image.save(source_path, "JPEG", quality=92)
            frame.image.save(goal_path, "JPEG", quality=92)
            diff = ImageChops.difference(prev_image.convert("RGB"), frame.image.convert("RGB"))
            diff.save(diff_path, "JPEG", quality=92)
            source_img_by_round_step[(round_idx, frame.step)] = prev_image.copy()
            goals_payload.append(
                {
                    "step": frame.step,
                    "description": frame.description,
                    "key_visual_features": frame.key_visual_features,
                    "reasoning": frame.reasoning,
                    "artifacts": {
                        "old_image": str(source_path.relative_to(artifact_dir)),
                        "new_image": str(goal_path.relative_to(artifact_dir)),
                        "image_diff": str(diff_path.relative_to(artifact_dir)),
                    },
                }
            )
            prev_image = frame.image

        output_payload = {
            "planning_round": round_idx,
            "num_goals": len(planned_frames),
            "goals": goals_payload,
        }
        _write_json(plan_dir / "planner_output.json", output_payload)
        planner_round_meta[round_idx] = {
            "is_replan": is_replan,
            "input_file": str((plan_dir / "planner_input.json").relative_to(artifact_dir)),
            "output_file": str((plan_dir / "planner_output.json").relative_to(artifact_dir)),
        }

    _save_planner_round(
        planning_round,
        initial_image,
        frames,
        is_replan=False,
        failed_frames=None,
    )

    while current_index < len(frames):
        frame: GoalFrame = frames[current_index]
        attempt_index += 1
        episode.total_attempts = attempt_index

        step_dir: Path | None = None
        if artifact_dir is not None:
            step_dir = (
                artifact_dir
                / "steps"
                / f"attempt_{attempt_index:03d}_round_{planning_round:03d}_goal_{frame.step:02d}"
            )
            step_dir.mkdir(parents=True, exist_ok=True)

        await bridge.send_goal_frame(frame)
        while not await bridge.is_step_complete():
            await asyncio.sleep(0.05)

        actual_image = await bridge.get_observation()
        result = await verifier.score(frame, actual_image, retry_count)
        episode.verifications.append(result)

        if step_dir is not None:
            source_img = source_img_by_round_step.get((planning_round, frame.step), initial_image)
            source_img.save(step_dir / "planner_old_image.jpg", "JPEG", quality=92)
            frame.image.save(step_dir / "planner_goal_image.jpg", "JPEG", quality=92)
            ImageChops.difference(
                source_img.convert("RGB"), frame.image.convert("RGB")
            ).save(step_dir / "planner_image_diff.jpg", "JPEG", quality=92)
            actual_image.save(step_dir / "robot_state.jpg", "JPEG", quality=92)

            verifier_input = {
                "goal_step": frame.step,
                "retry_count": retry_count,
                "goal_description": frame.description,
                "goal_key_visual_features": frame.key_visual_features,
            }
            verifier_output = {
                "decision": result.decision,
                "semantic_score": result.semantic_score,
                "pose_score": result.pose_score,
                "contact_score": result.contact_score,
                "composite_score": result.composite_score,
                "reasoning": result.reasoning,
                "retry_count": result.retry_count,
            }
            _write_json(step_dir / "verifier_input.json", verifier_input)
            _write_json(step_dir / "verifier_output.json", verifier_output)

            planner_input_payload = {
                "planning_round": planning_round,
                "is_replan": planner_round_meta.get(planning_round, {}).get("is_replan", False),
                "task_description": task_description,
            }
            planner_output_payload = {
                "step": frame.step,
                "description": frame.description,
                "key_visual_features": frame.key_visual_features,
                "reasoning": frame.reasoning,
            }
            _write_json(step_dir / "planner_input.json", planner_input_payload)
            _write_json(step_dir / "planner_output.json", planner_output_payload)

            if hasattr(bridge, "get_action_frames_for_goal"):
                action_dir = step_dir / "actions"
                action_dir.mkdir(parents=True, exist_ok=True)
                for idx, action_img in enumerate(bridge.get_action_frames_for_goal(frame.step), start=1):  # type: ignore[attr-defined]
                    action_img.save(action_dir / f"action_{idx:03d}.jpg", "JPEG", quality=92)

        log.info(
            "loop.verification",
            step=frame.step,
            semantic=result.semantic_score,
            pose=result.pose_score,
            contact=result.contact_score,
            composite=result.composite_score,
            decision=result.decision,
            retry_count=retry_count,
        )

        if result.decision == "advance":
            current_index += 1
            retry_count = 0
        elif result.decision == "retry":
            retry_count += 1
        else:  # replan
            replan_count += 1
            episode.replan_count = replan_count
            if replan_count >= settings.max_replans:
                log.warning(
                    "loop.max_replans_reached",
                    replan_count=replan_count,
                    max_replans=settings.max_replans,
                )
                episode.success = False
                episode.termination_reason = "max_replans_reached"
                break

            failed = [frame]
            planning_round += 1
            frames = await planner.plan(
                initial_image,
                task_description,
                current_state_image=actual_image,
                failed_frames=failed,
            )
            episode.goal_frames = frames
            _save_planner_round(
                planning_round,
                actual_image,
                frames,
                is_replan=True,
                failed_frames=failed,
            )
            current_index = 0
            retry_count = 0

        if (
            settings.enable_explorer
            and settings.epsilon_explore > 0
            and current_index < len(frames)
            and random.random() < settings.epsilon_explore
        ):
            try:
                variants = await explorer.generate_variants(frame, episode)
                if variants:
                    log.info("loop.variants", step=frame.step, count=len(variants))
                    # Should select some variants and use those as our goal frames here
            except NotImplementedError:
                log.debug("loop.variants_unavailable")

    if episode.success is None:
        episode.success = True
        episode.termination_reason = "completed"
    episode.planning_rounds = planning_round + 1
    return episode

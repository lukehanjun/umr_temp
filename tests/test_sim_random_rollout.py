"""Simulation smoke test: random policy rollout with no external model API calls.

This test patches planner/verifier to avoid LLM/VLM/imagegen calls, runs the loop with
SimPolicyBridge, and saves artifacts (goal frames + verifier outputs) for inspection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from sharpa.config import Settings
from sharpa.interface.policy_bridge import SimPolicyBridge
from sharpa.loop import run_episode
from sharpa.models.episode import GoalFrame, VerificationResult

ARTIFACT_ROOT = Path("data") / "sim_smoke"


@pytest.mark.asyncio
async def test_sim_random_rollout_saves_artifacts(mocker: MockerFixture) -> None:
    try:
        import gymnasium  # noqa: F401
        import mani_skill.envs  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent skip path
        pytest.skip(f"ManiSkill not available in this environment: {exc}")

    settings = Settings(openrouter_api_key="test", openai_api_key="test")
    settings.sim_policy_mode = "random"
    settings.sim_env_id = "PickCube-v1"
    settings.sim_obs_mode = "state"
    settings.sim_render_mode = "rgb_array"
    settings.sim_policy_step_horizon = 6
    settings.epsilon_explore = 0.0

    n_goals = 3

    async def fake_plan(
        self,
        initial_image: Image.Image,
        task_description: str,
        current_state_image: Image.Image | None = None,
        failed_frames: list[GoalFrame] | None = None,
    ) -> list[GoalFrame]:
        _ = (self, task_description, failed_frames)
        seed_img = current_state_image if current_state_image is not None else initial_image
        return [
            GoalFrame(
                step=i,
                description=f"Synthetic goal step {i}",
                key_visual_features=["scene visible", "object present"],
                reasoning="Local test stub: no external model call",
                image=seed_img.copy(),
            )
            for i in range(1, n_goals + 1)
        ]

    async def fake_score(
        self,
        goal_frame: GoalFrame,
        actual_image: Image.Image,
        retry_count: int = 0,
    ) -> VerificationResult:
        _ = self
        return VerificationResult(
            goal_frame=goal_frame,
            actual_image=actual_image,
            semantic_score=0.9,
            pose_score=0.85,
            contact_score=0.8,
            composite_score=0.865,
            decision="advance",
            reasoning="Synthetic verifier pass",
            retry_count=retry_count,
        )

    mocker.patch("sharpa.loop.GoalPlannerAgent.plan", new=fake_plan)
    mocker.patch("sharpa.loop.VerifierAgent.score", new=fake_score)

    # Safety guard: if any API wrapper is accidentally called, fail immediately.
    mocker.patch(
        "sharpa.api.llm.call_llm",
        side_effect=AssertionError("LLM API should not be called"),
    )
    mocker.patch(
        "sharpa.api.imagegen.generate_image",
        side_effect=AssertionError("ImageGen API should not be called"),
    )
    mocker.patch(
        "sharpa.api.vlm.score_image_pair",
        side_effect=AssertionError("VLM API should not be called"),
    )

    bridge = SimPolicyBridge(settings=settings)
    try:
        episode = await run_episode(
            task_description="sim smoke test task",
            settings=settings,
            bridge=bridge,
        )
    finally:
        if hasattr(bridge, "_env"):
            bridge._env.close()  # noqa: SLF001 - test cleanup path

    assert episode.success is True
    assert len(episode.goal_frames) == n_goals
    assert len(episode.verifications) == n_goals

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACT_ROOT / f"pytest_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for frame in episode.goal_frames:
        frame_dir = run_dir / f"goal_step_{frame.step:02d}"
        action_dir = frame_dir / "actions"
        action_dir.mkdir(parents=True, exist_ok=True)

        frame.image.save(frame_dir / "goal.jpg", "JPEG", quality=92)

        action_frames = bridge.get_action_frames_for_goal(frame.step)
        for idx, action_img in enumerate(action_frames, start=1):
            action_img.save(action_dir / f"action_{idx:03d}.jpg", "JPEG", quality=92)

    results: list[dict[str, object]] = []
    for idx, result in enumerate(episode.verifications, start=1):
        result.actual_image.save(
            run_dir / f"actual_{idx:02d}_step_{result.goal_frame.step:02d}.jpg",
            "JPEG",
            quality=92,
        )
        results.append(
            {
                "step": result.goal_frame.step,
                "decision": result.decision,
                "semantic_score": result.semantic_score,
                "pose_score": result.pose_score,
                "contact_score": result.contact_score,
                "composite_score": result.composite_score,
                "retry_count": result.retry_count,
                "reasoning": result.reasoning,
                "num_action_frames": len(bridge.get_action_frames_for_goal(result.goal_frame.step)),
            }
        )

    (run_dir / "verifier_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    assert (run_dir / "verifier_results.json").exists()
    for step in range(1, n_goals + 1):
        frame_dir = run_dir / f"goal_step_{step:02d}"
        action_dir = frame_dir / "actions"
        assert (frame_dir / "goal.jpg").exists()
        assert len(list(action_dir.glob("action_*.jpg"))) >= 1
    assert len(list(run_dir.glob("actual_*.jpg"))) == n_goals
"""Tests for the orchestration loop."""

from __future__ import annotations

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from sharpa.config import Settings
from sharpa.interface.policy_bridge import StubPolicyBridge
from sharpa.loop import run_episode
from sharpa.models.episode import GoalFrame, VerificationResult


@pytest.fixture
def settings() -> Settings:
    return Settings(openrouter_api_key="test", openai_api_key="test")


@pytest.mark.asyncio
async def test_run_episode_with_stub_bridge(
    settings: Settings,
    mocker: MockerFixture,
) -> None:
    frame = GoalFrame(
        step=1,
        description="Red cube is stacked on blue cylinder",
        key_visual_features=["cube on top", "objects touching"],
        reasoning="single step",
        image=Image.new("RGB", (512, 512)),
    )

    mocker.patch("sharpa.loop.GoalPlannerAgent.plan", return_value=[frame])
    mocker.patch("sharpa.loop.ExplorerAgent.generate_variants", return_value=[])
    mocker.patch(
        "sharpa.loop.VerifierAgent.score",
        return_value=VerificationResult(
            goal_frame=frame,
            actual_image=Image.new("RGB", (512, 512)),
            semantic_score=0.95,
            pose_score=0.9,
            contact_score=0.9,
            composite_score=0.925,
            decision="advance",
            reasoning="done",
            retry_count=0,
        ),
    )

    episode = await run_episode(
        "stack the red cube on the blue cylinder",
        settings,
        bridge=StubPolicyBridge(),
    )

    assert episode.success is True
    assert len(episode.goal_frames) == 1
    assert len(episode.verifications) == 1
    assert episode.verifications[0].decision == "advance"
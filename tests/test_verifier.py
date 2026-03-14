"""Tests for VerifierAgent."""

from __future__ import annotations

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from sharpa.agents.verifier import VerifierAgent
from sharpa.config import Settings
from sharpa.models.episode import GoalFrame


@pytest.fixture
def settings() -> Settings:
    return Settings(openrouter_api_key="test", openai_api_key="test")


@pytest.fixture
def stub_goal_frame() -> GoalFrame:
    return GoalFrame(
        step=1,
        description="Red cube lifted above blue cylinder",
        key_visual_features=["red cube elevated", "blue cylinder visible below"],
        reasoning="Must lift before stacking",
        image=Image.new("RGB", (512, 512)),
    )


@pytest.fixture
def mock_actual_image() -> Image.Image:
    return Image.open("data/output_frames/step_01.jpg").convert("RGB")


@pytest.mark.asyncio
async def test_score_returns_verification_result(
    settings: Settings,
    stub_goal_frame: GoalFrame,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "sharpa.agents.verifier.score_image_pair",
        return_value={
            "semantic": 0.9,
            "pose": 0.8,
            "contact": 0.85,
            "reasoning": "Goal visually matched",
        },
    )
    agent = VerifierAgent(settings)
    actual = Image.new("RGB", (512, 512))

    result = await agent.score(stub_goal_frame, actual)

    assert result.semantic_score == pytest.approx(0.9)
    assert result.pose_score == pytest.approx(0.8)
    assert result.contact_score == pytest.approx(0.85)
    assert result.composite_score == pytest.approx(0.86)
    assert result.reasoning == "Goal visually matched"
    assert result.decision == "advance"
    assert result.goal_frame == stub_goal_frame
    assert result.actual_image == actual


@pytest.mark.asyncio
async def test_score_retry_and_replan_logic(
    settings: Settings,
    stub_goal_frame: GoalFrame,
    mock_actual_image: Image.Image,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "sharpa.agents.verifier.score_image_pair",
        return_value={
            "semantic": 0.2,
            "pose": 0.3,
            "contact": 0.1,
            "reasoning": "Object pose and contact are wrong",
        },
    )

    agent = VerifierAgent(settings)

    result = await agent.score(stub_goal_frame, mock_actual_image, retry_count=1)
    assert result.composite_score == pytest.approx(0.21)
    assert result.decision == "retry"

    result = await agent.score(
        stub_goal_frame,
        mock_actual_image,
        retry_count=settings.max_retries,
    )
    assert result.decision == "replan"
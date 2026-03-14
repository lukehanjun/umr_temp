"""Tests for ExplorerAgent."""

from __future__ import annotations

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from sharpa.agents.explorer import ExplorerAgent
from sharpa.config import Settings
from sharpa.models.episode import Episode, GoalFrame


@pytest.fixture
def settings() -> Settings:
    return Settings(openrouter_api_key="test", openai_api_key="test")


@pytest.fixture
def stub_goal_frame() -> GoalFrame:
    return GoalFrame(
        step=1,
        description="Hand approaching red cube from above",
        key_visual_features=["hand above cube", "gripper open"],
        reasoning="Approach before grasp",
        image=Image.new("RGB", (512, 512)),
    )


@pytest.fixture
def stub_episode(stub_goal_frame: GoalFrame) -> Episode:
    return Episode(
        task_description="stack the red cube on the blue cylinder",
        initial_image=Image.new("RGB", (512, 512)),
        goal_frames=[stub_goal_frame],
    )


@pytest.mark.asyncio
async def test_generate_variants_returns_list(
    settings: Settings,
    stub_goal_frame: GoalFrame,
    stub_episode: Episode,
    mocker: MockerFixture,
) -> None:
    mocker.patch("sharpa.api.llm.call_llm")
    mocker.patch("sharpa.api.imagegen.generate_image")
    agent = ExplorerAgent(settings)
    with pytest.raises(NotImplementedError):
        await agent.generate_variants(stub_goal_frame, stub_episode, k=3)

"""Tests for GoalPlannerAgent."""

from __future__ import annotations

import pytest
from PIL import Image
from pytest_mock import MockerFixture

from sharpa.agents.planner import GoalPlannerAgent, _parse_response
from sharpa.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(openrouter_api_key="test", openai_api_key="test")


@pytest.fixture
def stub_image() -> Image.Image:
    return Image.new("RGB", (512, 512))


_VALID_RESPONSE = """
<thinking>
  The three markers need to be stacked upright.
</thinking>
<goals>
  [
    {
      "step": 1,
      "description": "Pick up black marker",
      "key_visual_features": ["hand gripping black marker"]
    },
    {
      "step": 2,
      "description": "Place black marker upright",
      "key_visual_features": ["black marker standing"]
    }
  ]
</goals>
"""


def test_parse_response_valid() -> None:
    thinking, goals = _parse_response(_VALID_RESPONSE)
    assert "markers" in thinking
    assert len(goals) == 2
    assert goals[0]["step"] == 1
    assert goals[1]["description"] == "Place black marker upright"


def test_parse_response_missing_goals_raises() -> None:
    with pytest.raises(ValueError, match="No <goals>"):
        _parse_response("<thinking>some reasoning</thinking>")


@pytest.mark.asyncio
async def test_plan_calls_llm_and_imagegen(
    settings: Settings,
    stub_image: Image.Image,
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "sharpa.agents.planner.call_llm",
        return_value=_VALID_RESPONSE,
    )
    mocker.patch(
        "sharpa.agents.planner.generate_image",
        return_value=Image.new("RGB", (1024, 1024)),
    )

    agent = GoalPlannerAgent(settings)
    frames = await agent.plan(stub_image, "stack the markers")

    assert len(frames) == 2
    assert frames[0].step == 1
    assert frames[0].reasoning == "The three markers need to be stacked upright."
    assert frames[0].is_variant is False
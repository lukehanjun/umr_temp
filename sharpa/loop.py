"""Orchestration loop.

Ties GoalPlannerAgent, VerifierAgent, and ExplorerAgent together per episode.
"""

from __future__ import annotations

import asyncio
import random

import structlog
from PIL import Image

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

    while current_index < len(frames):
        frame: GoalFrame = frames[current_index]

        await bridge.send_goal_frame(frame)
        while not await bridge.is_step_complete():
            await asyncio.sleep(0.05)

        actual_image = await bridge.get_observation()
        result = await verifier.score(frame, actual_image, retry_count)
        episode.verifications.append(result)

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
            failed = [frame]
            frames = await planner.plan(
                initial_image,
                task_description,
                current_state_image=actual_image,
                failed_frames=failed,
            )
            episode.goal_frames = frames
            current_index = 0
            retry_count = 0

        if current_index < len(frames) and random.random() < settings.epsilon_explore:
            try:
                variants = await explorer.generate_variants(frame, episode)
                if variants:
                    log.info("loop.variants", step=frame.step, count=len(variants))
                    # Should select some variants and use those as our goal frames here
            except NotImplementedError:
                log.debug("loop.variants_unavailable")

    episode.success = True
    return episode
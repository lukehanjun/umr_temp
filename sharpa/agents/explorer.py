"""ExplorerAgent — generates alternative goal frame variants for under-explored subgoal states."""

from __future__ import annotations

import structlog

from sharpa.config import Settings
from sharpa.models.episode import Episode, GoalFrame

log = structlog.get_logger(__name__)


class ExplorerAgent:
    """Generates K alternative GoalFrame variants for a given subgoal.

    Explorer only generates variants — it does not execute or log them.
    loop.py decides when to invoke Explorer and what to do with variants.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_variants(
        self,
        goal_frame: GoalFrame,
        episode: Episode,
        k: int = 3,
    ) -> list[GoalFrame]:
        """Return k alternative GoalFrames for the same subgoal via different approaches.

        # TODO (Task 3): Implement variant generation.
        #
        # Steps:
        #   1. Build a prompt for the LLM that includes:
        #        - goal_frame.description (the subgoal to find alternatives for)
        #        - episode.task_description (the overall task for context)
        #        - goal_frame.key_visual_features (what the verifier checks)
        #        - Instruction: "Generate {k} alternative visual descriptions that achieve
        #          the same intermediate state via a different physical approach."
        #        - Ask for a JSON array of objects with description/features.
        #   2. Call api/llm.call_llm() with response_format="json".
        #      Use self.settings.planner_llm (same model as the planner).
        #   3. Parse the JSON response to get k descriptions.
        #   4. For each description, call api/imagegen.generate_image() with:
        #        - prompt = the variant description
        #        - reference_image = goal_frame.image  (edit from same base state)
        #        - model = self.settings.imagegen_model
        #   5. Pack each result into a GoalFrame with:
        #        - step = goal_frame.step (same step, alternative approach)
        #        - is_variant = True
        #        - reasoning = the LLM's rationale (you can include it in the prompt response)
        #   6. Log how many variants were generated at INFO level.
        #   7. Return the list of variant GoalFrames.
        #
        # Explorer does NOT execute variants or decide when to use them — that is loop.py's job.
        """
        raise NotImplementedError


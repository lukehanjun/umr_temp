"""VerifierAgent - VLM-as-judge comparing actual observation against intended goal frame."""

from __future__ import annotations

import structlog
from PIL import Image

from sharpa.api.vlm import score_image_pair
from sharpa.config import Settings
from sharpa.models.episode import GoalFrame, VerificationResult

log = structlog.get_logger(__name__)


class VerifierAgent:
    """Checks whether the actual robot observation satisfied a goal frame.

    Uses VLM scores for semantic, pose, and contact match.
    Decision logic follows:
      composite = 0.5 * semantic + 0.3 * pose + 0.2 * contact
      if composite >= settings.advance_threshold -> advance
      elif retry_count < settings.max_retries -> retry
      else -> replan
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def score(
        self,
        goal_frame: GoalFrame,
        actual_image: Image.Image,
        retry_count: int = 0,
    ) -> VerificationResult:
        """Compare actual_image against goal_frame and return a VerificationResult."""

        judge_input = (
            f"Goal description: {goal_frame.description}\n"
            f"Key visual features: {', '.join(goal_frame.key_visual_features)}"
        )
        result = await score_image_pair(
            goal_image=goal_frame.image,
            actual_image=actual_image,
            task_description=judge_input,
            model=self.settings.verifier_vlm,
        )

        semantic = float(result["semantic"])
        pose = float(result["pose"])
        contact = float(result["contact"])
        reasoning = str(result.get("reasoning", ""))
        composite = 0.5 * semantic + 0.3 * pose + 0.2 * contact

        if composite >= self.settings.advance_threshold:
            decision = "advance"
        elif retry_count < self.settings.max_retries:
            decision = "retry"
        else:
            decision = "replan"

        log.info(
            "verifier.decision",
            step=goal_frame.step,
            semantic=semantic,
            pose=pose,
            contact=contact,
            composite=composite,
            threshold=self.settings.advance_threshold,
            decision=decision,
        )

        return VerificationResult(
            goal_frame=goal_frame,
            actual_image=actual_image,
            semantic_score=semantic,
            pose_score=pose,
            contact_score=contact,
            composite_score=composite,
            decision=decision,
            reasoning=reasoning,
            retry_count=retry_count,
        )
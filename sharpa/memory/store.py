"""SQLite-backed store for Episode and VerificationResult persistence.

Images are stored as base64 strings and decoded to PIL.Image on read.
No image files are written to disk.
"""

from __future__ import annotations

import structlog

from sharpa.config import Settings
from sharpa.models.episode import Episode, VerificationResult

log = structlog.get_logger(__name__)


class EpisodeStore:
    """Persist and retrieve episodes and verification results from SQLite."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def save_episode(self, episode: Episode) -> None:
        """Persist an episode (upsert by id).

        # TODO (Task 5): Implement SQLite persistence.
        #
        # Use aiosqlite for async access (`uv add aiosqlite`).
        # Create the DB and tables on first use (CREATE TABLE IF NOT EXISTS).
        #
        # Suggested schema:
        #   episodes(id TEXT PK, task_description TEXT, success INT, created_at TEXT,
        #            initial_image_b64 TEXT)
        #   goal_frames(id INTEGER PK AUTOINCREMENT, episode_id TEXT, step INT,
        #               description TEXT, key_visual_features TEXT,  -- JSON array
        #               reasoning TEXT, image_b64 TEXT, is_variant INT)
        #
        # To encode a PIL image: base64.b64encode(buf.getvalue()).decode()
        # To decode back:        Image.open(io.BytesIO(base64.b64decode(b64_str)))
        # See api/imagegen.py for a working encode/decode example.
        #
        # Upsert episodes by id using INSERT OR REPLACE.
        # Delete + re-insert goal_frames for the episode on upsert.
        """
        raise NotImplementedError

    async def load_episode(self, episode_id: str) -> Episode:
        """Load an episode by id.

        # TODO (Task 5): Load and reconstruct a full Episode from SQLite.
        #
        # Steps:
        #   1. SELECT the episodes row by id — raise KeyError if not found.
        #   2. SELECT all goal_frames rows for this episode, ordered by step.
        #   3. Decode each image_b64 back to a PIL.Image.
        #   4. Reconstruct GoalFrame objects from the rows.
        #   5. Load verifications via save_verification's table (design that schema too).
        #   6. Return a complete Episode object.
        """
        raise NotImplementedError

    async def save_verification(self, result: VerificationResult) -> None:
        """Append a verification result to the store.

        # TODO (Task 5): Persist a VerificationResult row.
        #
        # Suggested schema:
        #   verifications(id INTEGER PK AUTOINCREMENT, episode_id TEXT,
        #                 goal_step INT, semantic REAL, pose REAL, contact REAL,
        #                 composite REAL, decision TEXT, reasoning TEXT,
        #                 retry_count INT, actual_image_b64 TEXT)
        #
        # The episode_id can be found via result.goal_frame — you may want to pass
        # episode_id explicitly if it's not on the result object.
        """
        raise NotImplementedError

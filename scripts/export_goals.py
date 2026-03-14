"""Export goal frames from a completed episode for policy team consumption."""

from __future__ import annotations

import asyncio
import json
import sys

from sharpa.config import Settings
from sharpa.memory.store import EpisodeStore


async def main(episode_id: str) -> None:
    settings = Settings()
    store = EpisodeStore(settings)
    episode = await store.load_episode(episode_id)

    frames = []
    for gf in episode.goal_frames:
        frames.append(
            {
                "step": gf.step,
                "description": gf.description,
                "key_visual_features": gf.key_visual_features,
                "is_variant": gf.is_variant,
            }
        )

    print(json.dumps(frames, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: export_goals.py <episode_id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))

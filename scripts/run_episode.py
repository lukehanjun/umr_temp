"""Run the GoalPlannerAgent on a real image + task description and save goal frames."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import structlog
from PIL import Image

# ---------------------------------------------------------------------------
# Logging setup — human-readable console output
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Repo root so relative paths resolve regardless of cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_IMAGE = DATA_DIR / "input.jpg"
DEFAULT_TEXT = DATA_DIR / "input_text.txt"
OUTPUT_DIR = DATA_DIR / "output_frames"


async def main(
    image_path: Path = DEFAULT_IMAGE,
    text_path: Path = DEFAULT_TEXT,
) -> None:
    from sharpa.agents.planner import GoalPlannerAgent
    from sharpa.config import Settings

    # --- load inputs --------------------------------------------------------
    if not image_path.exists():
        log.error("input_image_not_found", path=str(image_path))
        sys.exit(1)
    if not text_path.exists():
        log.error("input_text_not_found", path=str(text_path))
        sys.exit(1)

    initial_image = Image.open(image_path).convert("RGB")
    task_description = text_path.read_text().strip()

    log.info("inputs_loaded", image=str(image_path), task=task_description)

    # --- run planner --------------------------------------------------------
    settings = Settings()  # reads from .env or environment variables
    agent = GoalPlannerAgent(settings)

    log.info("planning_start", model=settings.planner_llm, n_goals=settings.n_goals)
    frames = await agent.plan(initial_image=initial_image, task_description=task_description)

    # --- print reasoning trace ----------------------------------------------
    if frames:
        print("\n" + "=" * 70)
        print("CHAIN-OF-THOUGHT REASONING")
        print("=" * 70)
        print(frames[0].reasoning)

    print("\n" + "=" * 70)
    print(f"GENERATED {len(frames)} GOAL FRAMES")
    print("=" * 70)
    for f in frames:
        print(f"\n[Step {f.step}]")
        print(f"  Description : {f.description}")
        print(f"  Key features: {', '.join(f.key_visual_features)}")

    # --- save output images + reasoning trace --------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in frames:
        out_path = OUTPUT_DIR / f"step_{f.step:02d}.jpg"
        f.image.save(out_path, "JPEG", quality=92)
        log.info("frame_saved", step=f.step, path=str(out_path))

    trace_path = OUTPUT_DIR / "reasoning_trace.txt"
    with trace_path.open("w") as fh:
        fh.write(f"Task: {task_description}\n")
        fh.write(f"Model: {settings.planner_llm}  |  Steps: {len(frames)}\n")
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("CHAIN-OF-THOUGHT\n")
        fh.write("=" * 70 + "\n")
        fh.write((frames[0].reasoning if frames else "") + "\n")
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("GOAL FRAMES\n")
        fh.write("=" * 70 + "\n")
        for f in frames:
            fh.write(f"\n[Step {f.step}]\n")
            fh.write(f"  Description : {f.description}\n")
            fh.write(f"  Key features: {', '.join(f.key_visual_features)}\n")
    log.info("trace_saved", path=str(trace_path))

    print(f"\nFrames + reasoning trace saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    text_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TEXT
    asyncio.run(main(image_path, text_path))

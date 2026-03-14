"""Pydantic settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter key - used for all LLM calls (Claude, etc.)
    openrouter_api_key: str

    # OpenAI key - used for DALL-E image generation
    openai_api_key: str

    # Anthropic key - optional, only needed if calling Anthropic directly (not via OpenRouter)
    anthropic_api_key: str | None = None

    planner_llm: str = "anthropic/claude-opus-4-6"
    verifier_vlm: str = "gpt-4o"
    imagegen_model: str = "google/gemini-2.5-flash-image"

    advance_threshold: float = 0.80
    max_retries: int = 3
    # Safety ceiling: planner self-determines how many steps are needed.
    # This value is never passed to the LLM - it only truncates runaway outputs.
    n_goals: int = 12
    epsilon_explore: float = 0.20

    # Simulation bridge settings (used by SimPolicyBridge)
    sim_env_id: str = "PickCube-v1"
    sim_obs_mode: str = "state"
    sim_render_mode: str = "rgb_array"
    sim_policy_mode: str = "random"  # random | ppo
    sim_ppo_checkpoint: str | None = None
    sim_policy_deterministic: bool = True
    sim_policy_step_horizon: int = 32

    db_path: str = "sharpa_memory.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
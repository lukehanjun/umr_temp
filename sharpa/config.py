"""Pydantic settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenRouter key - used for all LLM calls (Claude, etc.)
    openrouter_api_key: str

    # OpenAI key - used for DALL-E image generation
    openai_api_key: str

    # Anthropic key - optional, only needed if calling Anthropic directly (not via OpenRouter)
    anthropic_api_key: str | None = None

    planner_llm: str = "qwen/qwen3.5-122b-a10b" # "anthropic/claude-opus-4-6"
    verifier_vlm: str = "google/gemini-3-flash-preview"
    imagegen_model: str = "google/gemini-2.5-flash-image"

    advance_threshold: float = 0.80
    max_retries: int = 3
    # Safety ceiling: planner self-determines how many steps are needed.
    # This value is never passed to the LLM - it only truncates runaway outputs.
    n_goals: int = 12
    enable_explorer: bool = True
    epsilon_explore: float = 0.20
    max_replans: int = 10

    # Simulation bridge settings (used by SimPolicyBridge)
    sim_env_id: str = "PickCube-v1"
    sim_obs_mode: str = "state"
    sim_render_mode: str = "rgb_array"
    sim_policy_mode: str = "random"  # random | ppo | pi05
    sim_ppo_checkpoint: str | None = None
    sim_policy_deterministic: bool = True
    sim_policy_step_horizon: int = 32
    sim_gpu_ids: str | None = None  # e.g. "0" or "0,1"
    sim_openpi_repo_dir: str = "openpi"
    sim_openpi_config_name: str = "pi05_droid"
    sim_openpi_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_droid"
    sim_openpi_prompt: str = "move the object to the target"
    sim_openpi_image_size: int = 224
    sim_openpi_state_dim: int = 8

    db_path: str = "sharpa_memory.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

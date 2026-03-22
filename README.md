# UMR

Reasoning-and-execution prototype for robotics:
- Planner (LLM + image generation)
- Verifier (VLM judge)
- Optional Explorer (not complete yet)
- Simulator bridge with switchable base policy (`random`, `ppo`, `pi05`)

This repo focuses on orchestration + policy bridge experiments in ManiSkill.

## 1) Setup (uv)

From repo root:

```bash
cd /home/hjyu/umr_temp
```

Install core deps:

```bash
uv sync
```

Install simulator deps (ManiSkill / SB3):

```bash
uv sync --extra sim
```

Install OpenPI (`pi05`) deps:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra sim --extra pi05
```

If Sapien complains about `pkg_resources`, ensure compatible setuptools:

```bash
uv pip install "setuptools<82"
```

## 2) Environment Variables (`.env`)

Create `.env`:

```bash
cp .env.example .env
```

Required keys:
- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY`

Main behavior knobs:
- `PLANNER_LLM`
- `VERIFIER_VLM`
- `IMAGEGEN_MODEL`
- `ADVANCE_THRESHOLD`
- `MAX_RETRIES`
- `N_GOALS`
- `ENABLE_EXPLORER`
- `EPSILON_EXPLORE`
- `MAX_REPLANS`

Simulator/policy knobs:
- `SIM_ENV_ID`
- `SIM_OBS_MODE`
- `SIM_RENDER_MODE`
- `SIM_POLICY_MODE` (`random` | `ppo` | `pi05`)
- `SIM_PPO_CHECKPOINT`
- `SIM_POLICY_DETERMINISTIC`
- `SIM_POLICY_STEP_HORIZON`
- `SIM_GPU_IDS` (e.g. `0` or `0,1`)

OpenPI knobs (`SIM_POLICY_MODE=pi05`):
- `SIM_OPENPI_REPO_DIR` (default `openpi`)
- `SIM_OPENPI_CONFIG_NAME` (default `pi05_droid`)
- `SIM_OPENPI_CHECKPOINT` (default `gs://openpi-assets/checkpoints/pi05_droid`)
- `SIM_OPENPI_PROMPT`
- `SIM_OPENPI_IMAGE_SIZE`
- `SIM_OPENPI_STATE_DIM`

## 3) CLI Options (`scripts/run_sim_episode.py`)

```bash
uv run scripts/run_sim_episode.py --help
```

Useful runtime overrides:
- `--task-file`
- `--env-id`
- `--policy-mode random|ppo|pi05`
- `--ppo-checkpoint`
- `--horizon`
- `--gpus 0` or `--gpus 0,1`
- `--openpi-config`
- `--openpi-checkpoint`
- `--openpi-prompt`

Notes:
- `--gpus` sets `CUDA_VISIBLE_DEVICES` for that run.
- If `--policy-mode pi05` and `SIM_OBS_MODE=state`, script auto-switches to `rgb+depth`.

## 4) How to Run Policies

### A) Random baseline

```bash
uv run scripts/run_sim_episode.py \
  --policy-mode random \
  --env-id PickCube-v1 \
  --task-file data/pickup.txt
```

### B) PPO baseline

Train PPO checkpoint first:

```bash
uv run scripts/train_eval_ppo.py \
  --env-id PickCube-v1 \
  --timesteps 100000 \
  --eval-episodes 20
```

Then run SHARPA loop with that checkpoint:

```bash
uv run scripts/run_sim_episode.py \
  --policy-mode ppo \
  --ppo-checkpoint data/policies/<your_run>/<your_checkpoint>.zip \
  --env-id PickCube-v1 \
  --task-file data/pickup.txt
```

### C) OpenPI PI0.5 baseline

```bash
uv run scripts/run_sim_episode.py \
  --policy-mode pi05 \
  --env-id PickCube-v1 \
  --gpus 0 \
  --openpi-config pi05_droid \
  --openpi-checkpoint gs://openpi-assets/checkpoints/pi05_droid \
  --openpi-prompt "pick up the cube" \
  --task-file data/pickup.txt
```

## 5) Run Artifacts

Each run gets its own folder:

`data/sim_runs/<timestamp>/`

Includes:
- `summary.json`
- `demo_policy.gif`
- planner round logs/images under `planner/round_###/`
- per-attempt logs/images under `steps/attempt_###_round_###_goal_##/`
- goal/action frame dumps

This makes it easy to compare runs across policy modes.

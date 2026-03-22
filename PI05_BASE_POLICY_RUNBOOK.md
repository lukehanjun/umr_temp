# PI0.5 Base Policy Runbook (SHARPA + OpenPI)

This project now supports `SIM_POLICY_MODE=pi05` in `SimPolicyBridge`.

## 1) Install dependencies

```bash
cd /home/hjyu/umr_temp

# SHARPA + simulator + local OpenPI deps
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra sim --extra pi05
```

## 2) Configure environment

From `/home/hjyu/umr_temp`, create/update `.env`:

```bash
cp .env.example .env
```

Set:

```dotenv
SIM_POLICY_MODE=pi05
SIM_OBS_MODE=rgb+depth
SIM_GPU_IDS=0
SIM_OPENPI_REPO_DIR=openpi
SIM_OPENPI_CONFIG_NAME=pi05_droid
SIM_OPENPI_CHECKPOINT=gs://openpi-assets/checkpoints/pi05_droid
SIM_OPENPI_PROMPT=pick and place the cube
SIM_POLICY_STEP_HORIZON=4
```

## 3) Run a simulator episode with PI0.5

```bash
cd /home/hjyu/umr_temp
uv run scripts/run_sim_episode.py \
  --policy-mode pi05 \
  --env-id PickCube-v1 \
  --gpus 0 \
  --openpi-config pi05_droid \
  --openpi-checkpoint gs://openpi-assets/checkpoints/pi05_droid \
  --openpi-prompt "pick up the cube"
```

Artifacts are saved under `data/sim_runs/<timestamp>/`.

## 4) Optional fallback

If you want to quickly validate simulator wiring without OpenPI inference:

```bash
uv run scripts/run_sim_episode.py --policy-mode random
```

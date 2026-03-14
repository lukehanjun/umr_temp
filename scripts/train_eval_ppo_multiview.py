"""Train/evaluate a PPO baseline on ManiSkill with explicit success-rate and artifacts.

Outputs per run:
- checkpoint (.zip)
- train_monitor.csv (episode-level train logs)
- train_summary.json
- eval_episodes.jsonl
- eval_summary.json
- demo.gif (first eval episode)

Example:
  uv run scripts/train_eval_ppo.py --env-id StackCube-v1 --timesteps 50000 --eval-episodes 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from PIL import Image

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "policies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=None, help="ManiSkill env id, e.g. StackCube-v1")
    parser.add_argument("--timesteps", type=int, default=10_000, help="PPO training timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episode count.")
    parser.add_argument(
        "--obs-mode",
        default="state",
        choices=["state", "rgb+depth"],
        help="Observation mode. Use rgb+depth for multi-view visual policy input.",
    )
    parser.add_argument(
        "--control-mode",
        default=None,
        help="Optional ManiSkill controller mode, e.g. pd_ee_delta_pose.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=200,
        help="Episode length cap for both training and evaluation environments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory for run artifacts.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Checkpoint filename. Defaults to ppo_<env>.zip",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/eval envs.")
    parser.add_argument("--progress", action="store_true", help="Show SB3 progress bar.")
    parser.add_argument(
        "--demo-max-frames",
        type=int,
        default=300,
        help="Max frames saved for demo GIF from first eval episode.",
    )
    parser.add_argument(
        "--demo-fps",
        type=int,
        default=20,
        help="GIF frame rate for demo output.",
    )
    return parser.parse_args()


def _to_float_scalar(value: Any) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _to_bool_scalar(value: Any) -> bool:
    arr = np.asarray(value)
    if arr.size == 0:
        return False
    return bool(arr.reshape(-1)[0])


def _extract_success(info: dict[str, Any]) -> bool | None:
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return _to_bool_scalar(info[key])
    return None


def _to_pil_frame(frame: Any) -> Image.Image | None:
    obj = frame
    if isinstance(obj, dict):
        for key in ("rgb", "color", "image", "render"):
            if key in obj:
                obj = obj[key]
                break
        else:
            if obj:
                obj = next(iter(obj.values()))

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().numpy()
    except ImportError:
        pass

    arr = np.asarray(obj)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        return None

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 4:
        arr = arr[:, :, :3]

    if arr.dtype != np.uint8:
        maxv = float(np.nanmax(arr)) if arr.size else 0.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def _read_monitor_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if row[0] == "r":
                continue
            if len(row) < 3:
                continue
            rows.append({"reward": float(row[0]), "length": float(row[1]), "time": float(row[2])})

    return rows


def _make_env_kwargs(
    obs_mode: str,
    max_episode_steps: int,
    control_mode: str | None,
    render_mode: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "obs_mode": obs_mode,
        "max_episode_steps": max_episode_steps,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if control_mode:
        kwargs["control_mode"] = control_mode
    return kwargs


def _tree_to_numpy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_tree_to_numpy(v) for v in obj)
    if isinstance(obj, list):
        return [_tree_to_numpy(v) for v in obj]

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except ImportError:
        pass

    return obj


def _wrap_for_sb3(env: Any, obs_mode: str) -> Any:
    import gymnasium as gym

    class _CPUObsWrapper(gym.ObservationWrapper):
        def observation(self, observation: Any) -> Any:
            return _tree_to_numpy(observation)

    wrapped_env = env
    if obs_mode == "rgb+depth":
        try:
            from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        except ImportError as exc:
            raise RuntimeError(
                "rgb+depth mode requires FlattenRGBDObservationWrapper from ManiSkill."
            ) from exc
        wrapped_env = FlattenRGBDObservationWrapper(wrapped_env)

    # Ensure SB3 always receives CPU numpy observations (not CUDA torch tensors).
    wrapped_env = _CPUObsWrapper(wrapped_env)
    return wrapped_env


def _infer_visual_info(obs: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if not isinstance(obs, dict):
        return info

    rgb = obs.get("rgb")
    if rgb is None:
        return info

    arr = np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            channels = int(arr.shape[0])
        else:
            channels = int(arr.shape[-1])
        info["rgb_channels"] = channels
        if channels % 3 == 0:
            info["estimated_num_rgb_views"] = channels // 3

    return info


def main() -> None:
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "This script requires simulator deps. Install with: uv sync --extra sim"
        ) from exc

    args = parse_args()
    env_id = args.env_id or os.getenv("SIM_ENV_ID", "PickCube-v1")

    run_dir = args.output_dir / f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{env_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = args.checkpoint_name or f"ppo_{env_id.replace('-', '_').lower()}.zip"
    checkpoint_path = run_dir / ckpt_name
    monitor_path = run_dir / "train_monitor.csv"

    train_cfg = {
        "env_id": env_id,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
    }

    train_kwargs = _make_env_kwargs(
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
        control_mode=args.control_mode,
    )
    raw_train_env = gym.make(env_id, **train_kwargs)
    train_env = _wrap_for_sb3(raw_train_env, args.obs_mode)
    train_env = Monitor(train_env, filename=str(monitor_path))

    policy_name = "MlpPolicy" if args.obs_mode == "state" else "MultiInputPolicy"
    train_cfg["policy"] = policy_name

    obs0, _ = train_env.reset(seed=args.seed)
    train_cfg.update(_infer_visual_info(obs0))
    (run_dir / "train_config.json").write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")

    log.info("ppo.train.start", **train_cfg)

    model = PPO(policy_name, train_env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress)
    model.save(str(checkpoint_path))

    train_rows = _read_monitor_csv(monitor_path)
    train_summary = {
        "num_episodes": len(train_rows),
        "mean_reward": float(np.mean([r["reward"] for r in train_rows])) if train_rows else None,
        "std_reward": float(np.std([r["reward"] for r in train_rows])) if train_rows else None,
        "mean_length": float(np.mean([r["length"] for r in train_rows])) if train_rows else None,
        "last_10_mean_reward": (
            float(np.mean([r["reward"] for r in train_rows[-10:]])) if train_rows else None
        ),
        "monitor_csv": str(monitor_path),
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(train_summary, indent=2),
        encoding="utf-8",
    )

    log.info("ppo.train.done", checkpoint=str(checkpoint_path), episodes=len(train_rows))

    eval_kwargs = _make_env_kwargs(
        obs_mode=args.obs_mode,
        max_episode_steps=args.max_episode_steps,
        control_mode=args.control_mode,
        render_mode="rgb_array",
    )
    raw_eval_env = gym.make(env_id, **eval_kwargs)
    eval_env = _wrap_for_sb3(raw_eval_env, args.obs_mode)

    eval_episode_path = run_dir / "eval_episodes.jsonl"

    episode_logs: list[dict[str, Any]] = []
    demo_frames: list[Image.Image] = []

    for ep in range(1, args.eval_episodes + 1):
        obs, info = eval_env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_success = False
        success_signal_seen = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += _to_float_scalar(reward)
            ep_len += 1

            succ = _extract_success(info)
            if succ is not None:
                success_signal_seen = True
                ep_success = ep_success or succ

            if ep == 1 and len(demo_frames) < args.demo_max_frames:
                frame = _to_pil_frame(raw_eval_env.render())
                if frame is not None:
                    demo_frames.append(frame)

            done = bool(terminated or truncated)

        episode_log = {
            "episode": ep,
            "reward": ep_reward,
            "length": ep_len,
            "success": ep_success,
            "success_signal_seen": success_signal_seen,
        }
        episode_logs.append(episode_log)

    with eval_episode_path.open("w", encoding="utf-8") as f:
        for row in episode_logs:
            f.write(json.dumps(row) + "\n")

    rewards = [e["reward"] for e in episode_logs]
    successes = [e["success"] for e in episode_logs]

    eval_summary = {
        "env_id": env_id,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "policy": policy_name,
        "max_episode_steps": args.max_episode_steps,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "std_reward": float(np.std(rewards)) if rewards else None,
        "success_rate": float(np.mean(successes)) if successes else None,
        "num_success": int(np.sum(successes)) if successes else 0,
        "success_signal_seen_in_any_episode": any(
            e["success_signal_seen"] for e in episode_logs
        ),
        "checkpoint": str(checkpoint_path),
        "eval_episodes_log": str(eval_episode_path),
    }
    eval_summary.update(_infer_visual_info(obs))

    eval_summary_path = run_dir / "eval_summary.json"
    eval_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    if demo_frames:
        demo_path = run_dir / "demo.gif"
        duration_ms = max(1, int(1000 / max(1, args.demo_fps)))
        demo_frames[0].save(
            demo_path,
            save_all=True,
            append_images=demo_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        eval_summary["demo_gif"] = str(demo_path)
        eval_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    print(json.dumps(eval_summary, indent=2))
    print(f"\nArtifacts written to: {run_dir}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
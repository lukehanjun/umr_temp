"""THE BOUNDARY: how we hand off goal frames to the policy team.

Keep this file minimal - it is a seam, not a feature.
Policy team concerns must never leak past this file into agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from PIL import Image

from sharpa.models.episode import GoalFrame

if TYPE_CHECKING:
    from sharpa.config import Settings

log = structlog.get_logger(__name__)


class PolicyBridgeBase(ABC):
    """Abstract interface. Two implementations: StubPolicyBridge and RealPolicyBridge."""

    @abstractmethod
    async def send_goal_frame(self, goal_frame: GoalFrame) -> None:
        """Send a goal frame to the policy execution system."""
        ...

    @abstractmethod
    async def get_observation(self) -> Image.Image:
        """Get the current camera observation after the last action."""
        ...

    @abstractmethod
    async def is_step_complete(self) -> bool:
        """Check if the policy has finished attempting the current goal."""
        ...


class StubPolicyBridge(PolicyBridgeBase):
    """Returns synthetic images. Used in all tests and dev runs."""

    async def send_goal_frame(self, goal_frame: GoalFrame) -> None:
        pass  # no-op in stub mode

    async def get_observation(self) -> Image.Image:
        return Image.new("RGB", (512, 512), color=(128, 128, 128))

    async def is_step_complete(self) -> bool:
        return True


class PPOPolicyHook:
    """Callable policy adapter wrapping an SB3 PPO checkpoint."""

    def __init__(self, checkpoint_path: str, deterministic: bool = True) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "PPO mode requires stable-baselines3. Install with: uv add stable-baselines3"
            ) from exc

        self.model = PPO.load(checkpoint_path)
        self.deterministic = deterministic
        log.info(
            "sim_bridge.ppo_loaded",
            checkpoint=checkpoint_path,
            deterministic=deterministic,
        )

    def __call__(self, obs: Any) -> Any:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


def build_policy_fn(settings: Settings) -> Callable[[Any], Any] | None:
    """Build a policy callable based on settings.sim_policy_mode."""
    mode = settings.sim_policy_mode.lower().strip()
    if mode == "random":
        return None

    if mode == "ppo":
        checkpoint = settings.sim_ppo_checkpoint
        if not checkpoint:
            raise RuntimeError("sim_policy_mode='ppo' requires settings.sim_ppo_checkpoint")
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise RuntimeError(f"PPO checkpoint not found: {checkpoint}")
        return PPOPolicyHook(
            checkpoint_path=str(ckpt_path),
            deterministic=settings.sim_policy_deterministic,
        )

    raise RuntimeError(f"Unsupported sim_policy_mode: {settings.sim_policy_mode}")


class SimPolicyBridge(PolicyBridgeBase):
    """Simulation-backed bridge for running a baseline policy in ManiSkill."""

    def __init__(
        self,
        settings: Settings,
        policy_fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.policy_fn = policy_fn if policy_fn is not None else build_policy_fn(settings)
        self._goal_frame: GoalFrame | None = None
        self._current_goal_step: int | None = None
        self._action_frames_by_goal_step: dict[int, list[Image.Image]] = {}
        self._latest_obs: Any = None
        self._latest_frame: Image.Image = Image.new("RGB", (512, 512), color=(80, 80, 80))

        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name == "pkg_resources":
                raise RuntimeError(
                    "Missing 'pkg_resources' (from setuptools), required by sapien/mani-skill. "
                    "Run: uv add setuptools --optional sim"
                ) from exc
            raise RuntimeError(
                "SimPolicyBridge requires gymnasium + mani_skill. "
                "Install with: uv sync --extra sim"
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "SimPolicyBridge requires gymnasium + mani_skill. "
                "Install with: uv sync --extra sim"
            ) from exc

        self._env = gym.make(
            self.settings.sim_env_id,
            obs_mode=self.settings.sim_obs_mode,
            render_mode=self.settings.sim_render_mode,
        )
        self._latest_obs, _ = self._env.reset()
        self._capture_frame()

        log.info(
            "sim_bridge.initialized",
            env_id=self.settings.sim_env_id,
            obs_mode=self.settings.sim_obs_mode,
            render_mode=self.settings.sim_render_mode,
            horizon=self.settings.sim_policy_step_horizon,
            policy_mode=self.settings.sim_policy_mode,
        )

    def _to_pil_frame(self, frame: Any) -> Image.Image | None:
        """Normalize render outputs (tensor/ndarray/dict) to PIL RGB."""
        import numpy as np

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
            log.warning("sim_bridge.render_unsupported", ndim=int(arr.ndim))
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

    def _capture_frame(self) -> None:
        frame = self._env.render()
        if frame is None:
            return
        pil = self._to_pil_frame(frame)
        if pil is not None:
            self._latest_frame = pil

    def _select_action(self, obs: Any) -> Any:
        if self.policy_fn is not None:
            return self.policy_fn(obs)
        return self._env.action_space.sample()

    def get_action_frames_for_goal(self, goal_step: int) -> list[Image.Image]:
        """Return per-action rendered frames captured for a specific goal step."""
        frames = self._action_frames_by_goal_step.get(goal_step, [])
        return [f.copy() for f in frames]

    async def send_goal_frame(self, goal_frame: GoalFrame) -> None:
        self._goal_frame = goal_frame
        self._current_goal_step = goal_frame.step
        self._action_frames_by_goal_step[self._current_goal_step] = []
        log.info("sim_bridge.goal_received", step=goal_frame.step)

    async def get_observation(self) -> Image.Image:
        return self._latest_frame

    async def is_step_complete(self) -> bool:
        horizon = max(1, self.settings.sim_policy_step_horizon)

        for _ in range(horizon):
            action = self._select_action(self._latest_obs)
            self._latest_obs, _, terminated, truncated, _ = self._env.step(action)
            self._capture_frame()

            if self._current_goal_step is not None:
                self._action_frames_by_goal_step[self._current_goal_step].append(
                    self._latest_frame.copy()
                )

            if terminated or truncated:
                self._latest_obs, _ = self._env.reset()  # initial obs, info
                self._capture_frame()
                break

        return True


class RealPolicyBridge(PolicyBridgeBase):
    """Filled in during hardware integration. Communicates with the policy team's system."""

    async def send_goal_frame(self, goal_frame: GoalFrame) -> None:
        raise NotImplementedError

    async def get_observation(self) -> Image.Image:
        raise NotImplementedError

    async def is_step_complete(self) -> bool:
        raise NotImplementedError
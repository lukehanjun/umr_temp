"""THE BOUNDARY: how we hand off goal frames to the policy team.

Keep this file minimal - it is a seam, not a feature.
Policy team concerns must never leak past this file into agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
import sys
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


class OpenPIPolicyHook:
    """Callable policy adapter that runs OpenPI PI0.5 as a base policy."""

    def __init__(self, settings: Settings) -> None:
        self._image_size = max(32, int(settings.sim_openpi_image_size))
        self._state_dim = max(1, int(settings.sim_openpi_state_dim))
        self._prompt = settings.sim_openpi_prompt

        repo_dir = Path(settings.sim_openpi_repo_dir).expanduser()
        if not repo_dir.is_absolute():
            repo_dir = Path.cwd() / repo_dir

        src_dir = repo_dir / "src"
        client_src_dir = repo_dir / "packages" / "openpi-client" / "src"
        for path in (src_dir, client_src_dir):
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        try:
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config
        except ImportError as exc:
            raise RuntimeError(
                "PI0.5 mode requires OpenPI modules. Ensure dependencies are installed and "
                "SIM_OPENPI_REPO_DIR points to the local openpi clone."
            ) from exc

        train_cfg = _config.get_config(settings.sim_openpi_config_name)
        self._policy = _policy_config.create_trained_policy(
            train_cfg,
            settings.sim_openpi_checkpoint,
            default_prompt=self._prompt,
        )

        log.info(
            "sim_bridge.openpi_loaded",
            config=settings.sim_openpi_config_name,
            checkpoint=settings.sim_openpi_checkpoint,
            prompt=self._prompt,
            image_size=self._image_size,
            state_dim=self._state_dim,
        )

    def _extract_state(self, obs: Any) -> Any:
        import numpy as np

        values: list[float] = []

        def _collect(obj: Any) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)
                return
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    _collect(v)
                return

            try:
                import torch

                if isinstance(obj, torch.Tensor):
                    obj = obj.detach().cpu().numpy()
            except ImportError:
                pass

            arr = np.asarray(obj)
            if arr.size == 0:
                return

            if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
                values.append(float(arr))
            elif arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                values.extend(float(x) for x in arr.tolist())

        _collect(obs)
        if not values:
            return np.zeros((self._state_dim,), dtype=np.float32)

        state = np.asarray(values, dtype=np.float32)
        if state.size < self._state_dim:
            state = np.pad(state, (0, self._state_dim - state.size))
        elif state.size > self._state_dim:
            state = state[: self._state_dim]
        return state

    def _extract_rgb_image(self, obs: Any) -> Any:
        import numpy as np

        def _as_image(obj: Any) -> np.ndarray | None:
            try:
                import torch

                if isinstance(obj, torch.Tensor):
                    obj = obj.detach().cpu().numpy()
            except ImportError:
                pass

            arr = np.asarray(obj)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim != 3:
                return None

            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] >= 4:
                arr = arr[:, :, :3]
            elif arr.shape[-1] != 3:
                return None

            if min(arr.shape[0], arr.shape[1]) < 32:
                return None

            if arr.dtype != np.uint8:
                maxv = float(np.nanmax(arr)) if arr.size else 0.0
                if maxv <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

        def _search(obj: Any) -> np.ndarray | None:
            if isinstance(obj, dict):
                for v in obj.values():
                    found = _search(v)
                    if found is not None:
                        return found
                return None
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    found = _search(v)
                    if found is not None:
                        return found
                return None
            return _as_image(obj)

        image = _search(obs)
        if image is None:
            return np.full((self._image_size, self._image_size, 3), 127, dtype=np.uint8)
        return image

    def _build_openpi_example(self, obs: Any) -> dict[str, Any]:
        import numpy as np

        state = self._extract_state(obs)
        if state.size < 8:
            state = np.pad(state, (0, 8 - state.size))
        elif state.size > 8:
            state = state[:8]

        image = self._extract_rgb_image(obs)
        pil_image = Image.fromarray(image).convert("RGB").resize((self._image_size, self._image_size))
        image_uint8 = np.asarray(pil_image, dtype=np.uint8)

        return {
            "observation/exterior_image_1_left": image_uint8,
            "observation/wrist_image_left": image_uint8,
            "observation/joint_position": state[:7].astype(np.float32),
            "observation/gripper_position": state[7:8].astype(np.float32),
            "prompt": self._prompt,
        }

    def __call__(self, obs: Any) -> Any:
        import numpy as np

        example = self._build_openpi_example(obs)
        outputs = self._policy.infer(example)
        actions = np.asarray(outputs["actions"])
        if actions.ndim == 2:
            return actions[0]
        if actions.ndim == 1:
            return actions
        return actions.reshape(-1)


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

    if mode == "pi05":
        return OpenPIPolicyHook(settings)

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
                    "Run: uv pip install 'setuptools<82' (or sync with sim deps)"
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
        import numpy as np

        if self.policy_fn is not None:
            raw_action = self.policy_fn(obs)
        else:
            raw_action = self._env.action_space.sample()

        space = self._env.action_space
        try:
            import gymnasium.spaces as spaces

            if isinstance(space, spaces.Box):
                arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                target_size = int(np.prod(space.shape))
                if arr.size < target_size:
                    arr = np.pad(arr, (0, target_size - arr.size))
                elif arr.size > target_size:
                    arr = arr[:target_size]
                arr = arr.reshape(space.shape)
                return np.clip(arr, space.low, space.high)

            if isinstance(space, spaces.Discrete):
                arr = np.asarray(raw_action).reshape(-1)
                val = int(arr[0]) if arr.size else 0
                return val % int(space.n)
        except Exception:
            log.warning("sim_bridge.action_adapt_fallback")

        return raw_action

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

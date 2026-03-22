"""Tests for policy bridge policy selection helpers."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from sharpa.config import Settings
from sharpa.interface.policy_bridge import OpenPIPolicyHook, PPOPolicyHook, build_policy_fn


@pytest.fixture
def settings() -> Settings:
    return Settings(openrouter_api_key="test", openai_api_key="test")


def test_build_policy_fn_random_returns_none(settings: Settings) -> None:
    settings.sim_policy_mode = "random"
    assert build_policy_fn(settings) is None


def test_build_policy_fn_ppo_requires_checkpoint(settings: Settings) -> None:
    settings.sim_policy_mode = "ppo"
    settings.sim_ppo_checkpoint = None
    with pytest.raises(RuntimeError, match="sim_policy_mode='ppo'"):
        build_policy_fn(settings)


def test_ppo_policy_hook_predicts_with_mocked_sb3(
    settings: Settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyModel:
        def predict(self, obs, deterministic=True):
            _ = obs
            return [1, 2, 3], None

    class _DummyPPO:
        @staticmethod
        def load(path):
            assert path.endswith("policy.zip")
            return _DummyModel()

    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=_DummyPPO))

    ckpt = tmp_path / "policy.zip"
    ckpt.write_text("dummy", encoding="utf-8")

    settings.sim_policy_mode = "ppo"
    settings.sim_ppo_checkpoint = str(ckpt)

    policy_fn = build_policy_fn(settings)
    assert isinstance(policy_fn, PPOPolicyHook)
    assert policy_fn(obs={}) == [1, 2, 3]


def test_build_policy_fn_pi05_uses_mocked_openpi(
    settings: Settings,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyPolicy:
        def infer(self, obs):
            _ = obs
            return {"actions": [[0.1] * 8]}

    def _dummy_get_config(name):
        assert name == "pi05_droid"
        return object()

    def _dummy_create_trained_policy(cfg, checkpoint, default_prompt=None):
        _ = (cfg, default_prompt)
        assert checkpoint == "local/pi05"
        return _DummyPolicy()

    fake_openpi = types.ModuleType("openpi")
    fake_openpi_training = types.ModuleType("openpi.training")
    fake_openpi_training_config = types.ModuleType("openpi.training.config")
    fake_openpi_training_config.get_config = _dummy_get_config

    fake_openpi_policies = types.ModuleType("openpi.policies")
    fake_openpi_policy_config = types.ModuleType("openpi.policies.policy_config")
    fake_openpi_policy_config.create_trained_policy = _dummy_create_trained_policy

    monkeypatch.setitem(sys.modules, "openpi", fake_openpi)
    monkeypatch.setitem(sys.modules, "openpi.training", fake_openpi_training)
    monkeypatch.setitem(sys.modules, "openpi.training.config", fake_openpi_training_config)
    monkeypatch.setitem(sys.modules, "openpi.policies", fake_openpi_policies)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", fake_openpi_policy_config)

    settings.sim_policy_mode = "pi05"
    settings.sim_openpi_repo_dir = str(tmp_path)
    settings.sim_openpi_config_name = "pi05_droid"
    settings.sim_openpi_checkpoint = "local/pi05"
    settings.sim_openpi_prompt = "test prompt"

    policy_fn = build_policy_fn(settings)
    assert isinstance(policy_fn, OpenPIPolicyHook)
    action = policy_fn(obs={"state": [0.0] * 8})
    assert len(action) == 8

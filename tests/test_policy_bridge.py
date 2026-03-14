"""Tests for policy bridge policy selection helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from sharpa.config import Settings
from sharpa.interface.policy_bridge import PPOPolicyHook, build_policy_fn


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
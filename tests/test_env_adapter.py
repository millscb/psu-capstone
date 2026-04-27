"""Tests for EnvAdapter construction and FinGym integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psu_capstone.environment.env_adapter import EnvAdapter
from psu_capstone.environment.fin_gym import FinGym


def test_env_adapter_accepts_instantiated_fingym() -> None:
    """EnvAdapter should wrap a pre-built FinGym instance directly."""

    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "target": [10.0, 11.0, 12.0],
        }
    )
    env = FinGym(data_source=frame, target_column="target")

    adapter = EnvAdapter(env)
    reset_obs, reset_info = adapter.reset()

    assert isinstance(reset_obs, np.ndarray)
    assert isinstance(reset_info, dict)

    payload_adapter = EnvAdapter(FinGym(data_source=frame, target_column="target"))
    reset_bridge = payload_adapter.reset_bridge()
    assert isinstance(reset_bridge["obs"], np.ndarray)
    assert isinstance(reset_bridge["inputs"], dict)

    step_obs, reward, terminated, truncated, info = adapter.step(1)

    assert isinstance(step_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    payload_adapter.reset_bridge()
    step_bridge = payload_adapter.step_bridge(1)
    assert isinstance(step_bridge["reward"], float)
    assert "terminated" in step_bridge
    assert "truncated" in step_bridge


def test_env_adapter_accepts_make_kwargs() -> None:
    """EnvAdapter should forward kwargs when constructing Gym env from id."""

    adapter = EnvAdapter("CartPole-v1", render_mode="rgb_array")

    reset_bridge = adapter.reset_bridge()
    assert isinstance(reset_bridge["inputs"], dict)


def test_env_adapter_rejects_kwargs_with_env_instance() -> None:
    """Passing gym kwargs with an env instance should raise ValueError."""

    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "target": [10.0, 11.0, 12.0],
        }
    )
    env = FinGym(data_source=frame, target_column="target")

    try:
        _ = EnvAdapter(env, render_mode="human")
        assert False, "Expected ValueError when kwargs are passed with env instance"
    except ValueError as exc:
        assert "gym_kwargs" in str(exc)

    @pytest.mark.parametrize(
        "env_id,step_action",
        [
            ("CartPole-v1", 0),
            ("MountainCar-v0", 0),
            ("FrozenLake-v1", 0),
            ("Pendulum-v1", [0.0]),
        ],
    )
    def test_env_adapter_accepts_standard_gym_envs(env_id, step_action):
        """EnvAdapter should accept and run standard Gym environments."""
        adapter = EnvAdapter(env_id)
        reset_bridge = adapter.reset_bridge()
        assert isinstance(reset_bridge["inputs"], dict)
        step_bridge = adapter.step_bridge(step_action)
        assert "reward" in step_bridge
        assert "terminated" in step_bridge
        assert "truncated" in step_bridge

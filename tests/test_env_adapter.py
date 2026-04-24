"""Tests for EnvAdapter construction and FinGym integration."""

# TS-20: EnvAdapter wrapper unit tests (Team 20 SWENG 481)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psu_capstone.environment.env_adapter import EnvAdapter
from psu_capstone.environment.fin_gym import FinGym


# Test Type: unit test
def test_env_adapter_accepts_instantiated_fingym() -> None:
    # TC-170
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


# Test Type: unit test
def test_env_adapter_accepts_make_kwargs() -> None:
    # TC-171
    """EnvAdapter should forward kwargs when constructing Gym env from id."""

    adapter = EnvAdapter("CartPole-v1", render_mode="rgb_array")

    reset_bridge = adapter.reset_bridge()
    assert isinstance(reset_bridge["inputs"], dict)


# Test Type: unit test
def test_env_adapter_rejects_kwargs_with_env_instance() -> None:
    # TC-172
    """Passing gym kwargs with an env instance should raise ValueError."""

    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "target": [10.0, 11.0, 12.0],
        }
    )
    env = FinGym(data_source=frame, target_column="target")

    with pytest.raises(ValueError, match="gym_kwargs"):
        _ = EnvAdapter(env, render_mode="human")

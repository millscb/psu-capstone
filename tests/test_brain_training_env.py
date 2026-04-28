# Test Suite: TS 29 (Brain Policy Training Across Server Envs)
"""System smoke tests for brain-policy training across server environment types."""

from __future__ import annotations

import pytest
from htmrl.agent_layer.agent_runtime import FRONTEND_ENV_SPECS
from htmrl.agent_layer.brain_training_helper import (
    EnvTrainingConfig,
    train_env_policy,
)


# Test Type: system test : TS 29 TC 261
@pytest.mark.parametrize(
    "env_id",
    [pytest.param(name, id=name) for name in FRONTEND_ENV_SPECS.keys()],
)
def test_train_env_policy_returns_metrics_for_each_server_env(env_id: str) -> None:
    """Training helper should run and return structured metrics for each server env type."""

    # LunarLander requires optional Box2D dependencies in some environments.
    if env_id.startswith("LunarLander"):
        pytest.importorskip("Box2D")

    metrics = train_env_policy(
        EnvTrainingConfig(
            env_id=env_id,
            episodes=1,
            max_steps_per_episode=5,
            input_size=64,
            cells_per_column=4,
            resolution=0.05,
            seed=11,
            render_mode=None,
        )
    )

    assert metrics["episodes"] == 1
    assert len(metrics["episode_rewards"]) == 1
    assert len(metrics["episode_steps"]) == 1
    assert all(step <= 5 for step in metrics["episode_steps"])
    assert isinstance(metrics["mean_reward"], float)
    assert isinstance(metrics["best_reward"], float)

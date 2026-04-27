"""Smoke tests for CartPole brain-policy training loop."""

from __future__ import annotations

from psu_capstone.agent_layer.cartpole_brain_training import (
    CartPoleTrainingConfig,
    train_cartpole_brain_policy,
)


def test_train_cartpole_brain_policy_returns_metrics() -> None:
    """Training helper should run and return structured metrics."""

    metrics = train_cartpole_brain_policy(
        CartPoleTrainingConfig(
            episodes=2,
            max_steps_per_episode=10,
            input_size=64,
            cells_per_column=4,
            resolution=0.05,
            rdse_seed=11,
        )
    )

    assert metrics["episodes"] == 2
    assert len(metrics["episode_rewards"]) == 2
    assert len(metrics["episode_steps"]) == 2
    assert all(step <= 10 for step in metrics["episode_steps"])
    assert isinstance(metrics["mean_reward"], float)
    assert isinstance(metrics["best_reward"], float)

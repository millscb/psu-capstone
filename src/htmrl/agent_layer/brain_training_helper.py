"""Generic env training helper compatibility API.

This module preserves the older helper names used by tests and callers while
routing to the current runtime build flow and brain policy execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from htmrl.agent_layer.agent_runtime import AgentRuntimeConfig, build_runtime


@dataclass(frozen=True)
class EnvTrainingConfig:
    """Configuration for brain-policy training on Gym-backed environments."""

    env_id: str = "CartPole-v1"
    episodes: int = 200
    max_steps_per_episode: int = 150
    input_size: int = 128
    cells_per_column: int = 8
    resolution: float = 0.001
    seed: int = 5
    render_mode: str | None = None


def train_env_policy(config: EnvTrainingConfig) -> dict[str, Any]:
    """Train with the current runtime stack and return episode metrics."""

    runtime_config = AgentRuntimeConfig(
        env_id=config.env_id,
        policy_mode="brain",
        episodes=config.episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        input_size=config.input_size,
        cells_per_column=config.cells_per_column,
        resolution=config.resolution,
        seed=config.seed,
        render_mode=config.render_mode,
    )
    runtime = build_runtime(runtime_config, allow_frontend_env=False)
    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    try:
        for _episode in range(config.episodes):
            runtime.agent.reset_episode()
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < config.max_steps_per_episode:
                transition = runtime.agent.step(learn=True)
                total_reward += float(transition["reward"])
                done = bool(transition["terminated"] or transition["truncated"])
                steps += 1

            episode_rewards.append(total_reward)
            episode_steps.append(steps)

        return {
            "episodes": config.episodes,
            "max_steps_per_episode": config.max_steps_per_episode,
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "mean_reward": float(fmean(episode_rewards)) if episode_rewards else 0.0,
            "best_reward": float(max(episode_rewards)) if episode_rewards else 0.0,
        }
    finally:
        runtime.close()

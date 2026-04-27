"""CartPole training loop driven by Brain policy mode.

This module provides a practical end-to-end loop for:

1. building a Brain that matches CartPole observation inputs,
2. wrapping CartPole with EnvAdapter,
3. running Agent in ``policy_mode='brain'`` across episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from psu_capstone.agent_layer.agent import Agent
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.environment.env_adapter import EnvAdapter


@dataclass(frozen=True)
class CartPoleTrainingConfig:
    """Configuration for CartPole brain-policy training."""

    episodes: int = 200
    max_steps_per_episode: int = 150
    input_size: int = 128
    cells_per_column: int = 8
    resolution: float = 0.001
    rdse_seed: int = 5


def build_cartpole_brain(
    adapter: EnvAdapter,
    config: CartPoleTrainingConfig,
) -> Brain:
    """Build a Brain for CartPole using Trainer-managed field construction.

    Trainer currently enforces naming conventions for field creation. This
    builder uses trainer-compatible names first and then remaps input field keys
    back to adapter input names so Agent.step can pass adapter inputs directly.
    """

    reset_bridge = adapter.reset_bridge()
    input_names = list(reset_bridge["inputs"].keys())
    trainer = Trainer(Brain({}))

    for index, name in enumerate(input_names):
        params = RDSEParameters(
            size=config.input_size,
            active_bits=0,
            sparsity=0.02,
            resolution=config.resolution,
            category=False,
            seed=config.rdse_seed + index,
        )
        trainer.add_input_field(f"{name}_input", config.input_size, params)

    # Motor action candidates map to CartPole's discrete actions.
    action_encoder_params = RDSEParameters(
        size=4,
        active_bits=1,
        sparsity=0.0,
        resolution=0.01,
        category=False,
        seed=config.rdse_seed + 999,
    )
    trainer.add_output_field(
        "action_output",
        4,
        action_encoder_params,
        possible_actions=[0, 1],
    )

    # Non-spatial column setup keeps one-to-one alignment from input bits to columns.
    trainer.add_column_field(
        "column_column",
        num_columns=config.input_size,
        cells_per_column=config.cells_per_column,
    )

    trainer_brain = trainer.main_brain
    remapped_fields: dict[str, Any] = {
        name: trainer_brain.fields[f"{name}_input"] for name in input_names
    }
    remapped_fields["action_output"] = trainer_brain.fields["action_output"]
    remapped_fields["column"] = trainer_brain.fields["column_column"]

    return Brain(remapped_fields)


def train_cartpole_brain_policy(
    config: CartPoleTrainingConfig | None = None,
) -> dict[str, Any]:
    """Run CartPole episodes using Agent in brain policy mode.

    Args:
        config: Optional training configuration. Defaults to
            ``CartPoleTrainingConfig()`` if not provided.

    Returns:
        Dictionary of training metrics and per-episode summaries.
    """

    cfg = config or CartPoleTrainingConfig()
    adapter = EnvAdapter("CartPole-v1")

    try:
        brain = build_cartpole_brain(adapter, cfg)
        agent = Agent(
            brain=brain,
            adapter=adapter,
            episodes=cfg.episodes,
            policy_mode="brain",
        )

        episode_rewards: list[float] = []
        episode_steps: list[int] = []

        for _episode in range(cfg.episodes):
            print(f"Starting episode {_episode + 1}/{cfg.episodes}")
            agent.reset_episode()
            done = False
            total_reward = 0.0
            steps = 0

            while not done and steps < cfg.max_steps_per_episode:
                transition = agent.step(learn=True)
                total_reward += float(transition["reward"])
                done = bool(transition["terminated"] or transition["truncated"])
                steps += 1

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            print(f"Episode {_episode + 1} finished: reward={total_reward}, steps={steps}")

            brain.print_stats()

        return {
            "episodes": cfg.episodes,
            "max_steps_per_episode": cfg.max_steps_per_episode,
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "mean_reward": float(fmean(episode_rewards)) if episode_rewards else 0.0,
            "best_reward": float(max(episode_rewards)) if episode_rewards else 0.0,
        }
    finally:
        if hasattr(adapter.env, "close"):
            adapter.env.close()


def main() -> None:
    """Run a default CartPole brain-policy training session and print metrics."""

    metrics = train_cartpole_brain_policy()
    print("CartPole brain-policy training summary")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Mean reward: {metrics['mean_reward']:.3f}")
    print(f"Best reward: {metrics['best_reward']:.3f}")


if __name__ == "__main__":
    main()

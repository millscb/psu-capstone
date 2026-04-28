"""
Unified runtime flow for Agent, Brain, server, and local env execution.

This module is the main entry point for connecting your backend RL agent to a frontend web app.

It handles:
1. Picking the right environment adapter (Gym or frontend-driven)
2. Building the Brain (encoder, SDR, etc.) to match the environment
3. Constructing the Agent (policy, training, etc.)
4. Running the WebSocket server so your frontend can connect and control episodes
5. Running local Gym sessions for backend-only testing

CALL CHAIN SUMMARY (backend to frontend):
1. run_server(config): Starts the WebSocket server for frontend connections
2. build_runtime(config): Builds adapter, Brain, Agent for the requested env
3. build_adapter(config): Chooses Gym or frontend adapter, passes max_steps_per_episode
4. build_brain_for_adapter(adapter, config): Sets up Brain fields to match env inputs/actions
5. AgentWebSocketServer: Handles WebSocket messages, episode state, and sends info to frontend
6. Frontend connects via WebSocket, sends commands (start_episode, step, etc.), receives state/results

TROUBLESHOOTING TIPS:
- If frontend doesn't see max_steps_per_episode, check build_adapter passes it to EnvAdapter
- If frontend can't start episodes, verify AgentWebSocketServer is running and reachable
- If Gym env doesn't match frontend, check env_id and adapter selection logic
- For custom environments, update FRONTEND_ENV_SPECS and adapter logic

To get the backend and frontend working together:
* Always start the server with run_server(config)
* Make sure your frontend connects to the correct ws://host:port
* Use the documented WebSocket protocol (see AGENT_WEBSOCKET_SERVER.md)
* Pass max_steps_per_episode and other config values through AgentRuntimeConfig
* Debug with logs from agent_server and agent_runtime for connection/state issues
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal, TypedDict, cast

try:
    import gym_trading_env  # noqa: F401  # Ensure TradingEnv is registered with Gymnasium
except ModuleNotFoundError:
    gym_trading_env = None

from htmrl.agent_layer.agent import Agent
from htmrl.agent_layer.pullin.pullin_brain import Brain
from htmrl.agent_layer.train import Trainer
from htmrl.encoder_layer.category_encoder import CategoryParameters
from htmrl.encoder_layer.rdse import RDSEParameters
from htmrl.environment.env_adapter import EnvAdapter
from htmrl.environment.frontend_env_adapter import FrontendEnvAdapter
from htmrl.log import get_logger

PolicyMode = Literal["q_table", "brain", "ppo"]


@dataclass(frozen=True)
class AgentPolicyProfile:
    """Environment-specific tuning knobs for the shared Agent.update method."""

    q_learning_rate: float = 0.1
    q_discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.01
    reward_scale: float = 1.0


DEFAULT_POLICY_PROFILE = AgentPolicyProfile()

ENV_POLICY_PROFILES: dict[str, AgentPolicyProfile] = {
    # Sparse reward grid world: keep exploration high for longer.
    "FrozenLake-v1": AgentPolicyProfile(epsilon_start=1.0, epsilon_decay=0.002, reward_scale=5.0),
    # Dense low-magnitude positive rewards.
    "CartPole-v1": AgentPolicyProfile(q_learning_rate=0.1, reward_scale=1.0),
    # Mostly -1 step costs, slower and conservative updates.
    "MountainCar-v0": AgentPolicyProfile(q_learning_rate=0.05, q_discount_factor=0.995),
    # Dense negative rewards; scale for a stronger learning signal in shared update path.
    "Pendulum-v1": AgentPolicyProfile(q_learning_rate=0.05, reward_scale=0.1),
    # Higher variance rewards; use moderate learning rate.
    "LunarLander-v3": AgentPolicyProfile(q_learning_rate=0.07, q_discount_factor=0.99),
    # Trading rewards can vary substantially by setup.
    "TradingEnv": AgentPolicyProfile(
        q_learning_rate=0.03, q_discount_factor=0.995, reward_scale=0.5
    ),
}


class FrontendEnvSpec(TypedDict):
    """Frontend-managed environment schema used by the websocket runtime."""

    observation_labels: list[str]
    action_count: int
    initial_observation: dict[str, float]
    max_steps_per_episode: int


FRONTEND_ENV_SPECS: dict[str, FrontendEnvSpec] = {
    "gym_trading_env": {
        "observation_labels": ["open", "high", "low", "close", "volume"],
        "action_count": 3,
        "initial_observation": {
            "open": 1769.5,
            "high": 1773.5,
            "low": 1739.75,
            "close": 1749.25,
        },
        "max_steps_per_episode": 500,
    },
    "TradingEnv": {
        "observation_labels": ["open", "high", "low", "close", "volume"],
        "action_count": 3,
        "initial_observation": {
            "open": 1769.5,
            "high": 1773.5,
            "low": 1739.75,
            "close": 1749.25,
            "volume": 1623508.0,
        },
        "max_steps_per_episode": 500,
    },
    "CartPole-v1": {
        "observation_labels": [
            "Cart Position",
            "Cart Velocity",
            "Pole Angle",
            "Pole Angular Velocity",
        ],
        "action_count": 2,
        "initial_observation": {
            "Cart Position": 0.0,
            "Cart Velocity": 0.0,
            "Pole Angle": 0.0,
            "Pole Angular Velocity": 0.0,
        },
        "max_steps_per_episode": 200,
    },
    "FrozenLake-v1": {
        "observation_labels": ["Agent X", "Agent Y", "Goal X", "Goal Y"],
        "action_count": 4,
        "initial_observation": {
            "Agent X": 0.0,
            "Agent Y": 0.0,
            "Goal X": 4.0,
            "Goal Y": 4.0,
        },
        "max_steps_per_episode": 100,
    },
    "MountainCar-v0": {
        "observation_labels": ["Position", "Velocity"],
        "action_count": 3,
        "initial_observation": {
            "Position": -0.5,
            "Velocity": 0.0,
        },
        "max_steps_per_episode": 200,
    },
    "Pendulum-v1": {
        "observation_labels": ["cos(theta)", "sin(theta)", "theta_dot"],
        "action_count": 3,
        "initial_observation": {
            "cos(theta)": 1.0,
            "sin(theta)": 0.0,
            "theta_dot": 0.0,
        },
        "max_steps_per_episode": 200,
    },
    "LunarLander-v3": {
        "observation_labels": [
            "x_position",
            "y_position",
            "x_velocity",
            "y_velocity",
            "angle",
            "angular_velocity",
            "left_leg_contact",
            "right_leg_contact",
        ],
        "action_count": 4,
        "initial_observation": {
            "x_position": 0.0,
            "y_position": 0.0,
            "x_velocity": 0.0,
            "y_velocity": 0.0,
            "angle": 0.0,
            "angular_velocity": 0.0,
            "left_leg_contact": 0.0,
            "right_leg_contact": 0.0,
        },
        "max_steps_per_episode": 1000,
    },
}


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Configuration shared by server and local execution paths.

    Attributes:
        env_id: Requested environment identifier.
        policy_mode: Agent policy mode.
        episodes: Number of episodes to run in local/train loops.
        max_steps_per_episode: Hard cap on steps per episode.
        input_size: SDR width for each input field.
        cells_per_column: Number of HTM cells per column.
        resolution: RDSE resolution used for continuous inputs.
        seed: Base seed for encoder parameter generation.
        render_mode: Optional Gymnasium render mode for local runs.
        reward_output_file: Output JSON path for local run episode metrics.
        step_delay_seconds: Optional delay after each env step for readability.
        non_spatial: Whether spatial pooling should be disabled.
        non_temporal: Whether temporal memory should be disabled.
        host: WebSocket bind host for server mode.
        port: WebSocket bind port for server mode.
        ppo_pretrain_timesteps: PPO warm-up timesteps before serving/running.
    """

    env_id: str = "CartPole-v1"
    policy_mode: PolicyMode = "brain"
    episodes: int = 200
    max_steps_per_episode: int = 200
    input_size: int = 256
    cells_per_column: int = 8
    resolution: float = 0.01
    seed: int = 5
    render_mode: str | None = None
    reward_output_file: str = "episode_rewards.json"
    step_delay_seconds: float = 0.0
    non_spatial: bool = False
    non_temporal: bool = False
    host: str = "localhost"
    port: int = 8765
    ppo_pretrain_timesteps: int = 50_000


@dataclass
class AgentRuntime:
    """Bundle of built runtime objects used by server and local flows."""

    config: AgentRuntimeConfig
    adapter: Any
    brain: Brain
    agent: Agent
    is_frontend_env: bool

    def close(self) -> None:
        """Close the underlying environment if one exists."""

        env = getattr(self.adapter, "_env", None)
        if env is not None and hasattr(env, "close"):
            env.close()


def build_adapter(config: AgentRuntimeConfig, *, allow_frontend_env: bool) -> tuple[Any, bool]:
    """Build the correct adapter for the requested environment id."""

    # Decide which adapter to use:
    # - FrontendEnvAdapter: for frontend-driven envs (web app controls episode)
    # - EnvAdapter: for Gym environments (backend controls episode)
    use_frontend_adapter = (
        allow_frontend_env and config.env_id in FRONTEND_ENV_SPECS and config.policy_mode != "ppo"
    )

    if use_frontend_adapter:
        # Build frontend adapter for web-driven environments
        spec = cast(FrontendEnvSpec, FRONTEND_ENV_SPECS[config.env_id])
        adapter = FrontendEnvAdapter(
            env_name=config.env_id,
            observation_labels=list(spec["observation_labels"]),
            action_count=spec["action_count"],
            initial_observation=dict(spec["initial_observation"]),
        )
        return adapter, True

    # Build Gym adapter for backend-driven environments
    adapter_kwargs: dict[str, Any] = {}
    if config.render_mode is not None:
        adapter_kwargs["render_mode"] = config.render_mode
    # Special handling for TradingEnv: provide a default DataFrame
    if config.env_id == "TradingEnv":
        if gym_trading_env is None:
            raise ModuleNotFoundError(
                "TradingEnv requires optional dependency 'gym_trading_env'. "
                "Install it or remove TradingEnv from the run list."
            )
        import pandas as pd

        df = pd.read_csv("data/fin_test.csv", parse_dates=["timestamp"], index_col="timestamp")
        # Rename columns to feature_* as required by gym_trading_env
        feature_cols = ["open", "high", "low", "close", "volume"]
        for col in feature_cols:
            # Normalize volume to avoid struct errors in encoding
            if col == "volume":
                df[f"feature_{col}"] = df[col] / 1e6
            else:
                df[f"feature_{col}"] = df[col]
        # Drop open_interest if present
        if "open_interest" in df.columns:
            df = df.drop(columns=["open_interest"])
        adapter_kwargs.clear()  # Remove any default keys
        adapter_kwargs.update(
            {
                "name": "BTCUSD",
                "df": df,
                "positions": [-1, 0, 1],
                "trading_fees": 0.01 / 100,
                "borrow_interest_rate": 0.0003 / 100,
            }
        )

    return EnvAdapter(config.env_id, **adapter_kwargs), False


def build_runtime(config: AgentRuntimeConfig, *, allow_frontend_env: bool) -> AgentRuntime:
    """Build the adapter, Brain, and Agent for a requested environment."""

    # --- Set random seeds for reproducibility ---
    import random

    import numpy as np

    random.seed(config.seed)
    np.random.seed(config.seed)

    adapter, is_frontend_env = build_adapter(config, allow_frontend_env=allow_frontend_env)
    profile = ENV_POLICY_PROFILES.get(config.env_id, DEFAULT_POLICY_PROFILE)
    logger = get_logger(None)
    logger.info(
        "Using policy profile for %s: lr=%.4f gamma=%.4f eps_start=%.4f eps_decay=%.4f reward_scale=%.4f",
        config.env_id,
        profile.q_learning_rate,
        profile.q_discount_factor,
        profile.epsilon_start,
        profile.epsilon_decay,
        profile.reward_scale,
    )
    trainer = Trainer(Brain({}))
    brain = trainer.build_brain_for_env(adapter, config)
    agent = Agent(
        brain=brain,
        adapter=adapter,
        episodes=config.episodes,
        policy_mode=config.policy_mode,
        q_learning_rate=profile.q_learning_rate,
        q_discount_factor=profile.q_discount_factor,
        epsilon_start=profile.epsilon_start,
        epsilon_decay=profile.epsilon_decay,
        reward_scale=profile.reward_scale,
    )

    # Only pre-train PPO if policy_mode is 'ppo'
    if (
        hasattr(agent, "train_ppo")
        and hasattr(adapter, "_env")
        and adapter._env is not None
        and config.policy_mode == "ppo"
    ):
        logger = get_logger(None)
        logger.info(
            "PPO mode selected: pre-training PPO for %d timesteps before normal execution...",
            config.ppo_pretrain_timesteps,
        )
        agent.train_ppo(total_timesteps=config.ppo_pretrain_timesteps)
        logger.info("PPO pre-training complete.")

    return AgentRuntime(
        config=config,
        adapter=adapter,
        brain=brain,
        agent=agent,
        is_frontend_env=is_frontend_env,
    )


def build_server(runtime: AgentRuntime) -> Any:
    """Create the websocket server for a built runtime."""

    from htmrl.agent_layer.agent_server import AgentWebSocketServer

    return AgentWebSocketServer(
        runtime.agent,
        host=runtime.config.host,
        port=runtime.config.port,
    )


async def run_server(config: AgentRuntimeConfig) -> None:
    """Build a runtime and start the websocket server."""

    logger = get_logger(None)
    runtime = build_runtime(config, allow_frontend_env=True)
    logger.info("Starting WebSocket server on ws://%s:%s", config.host, config.port)
    logger.info(
        "Server is running. Connect your web client to ws://%s:%s", config.host, config.port
    )
    logger.info("Press Ctrl+C to stop the server.")
    try:
        await build_server(runtime).start()
    finally:
        runtime.close()


def run_local_session(config: AgentRuntimeConfig) -> dict[str, Any]:
    """Run local env stepping for Gym-based testing with the same build flow.

    This path is intended for local testing of the env you pass in. When the
    env supports ``render_mode='human'``, Gymnasium will usually drive the
    pygame/window rendering internally.
    """

    runtime = build_runtime(config, allow_frontend_env=False)
    logger = get_logger(None)
    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    try:
        env = getattr(runtime.adapter, "_env", None)

        for episode_index in range(config.episodes):
            logger.info(
                "Starting local episode %d/%d for env %s with policy %s",
                episode_index + 1,
                config.episodes,
                config.env_id,
                config.policy_mode,
            )
            runtime.agent.reset_episode()
            done = False
            total_reward = 0.0
            steps = 0

            if env is not None and config.render_mode is not None and hasattr(env, "render"):
                env.render()

            while not done and steps < config.max_steps_per_episode:
                transition = runtime.agent.step(learn=True)
                total_reward += float(transition["reward"])
                done = bool(transition["terminated"] or transition["truncated"])
                steps += 1

                if env is not None and config.render_mode is not None and hasattr(env, "render"):
                    env.render()

                if config.step_delay_seconds > 0:
                    time.sleep(config.step_delay_seconds)

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            logger.info(
                "Episode %d finished: reward=%s steps=%s",
                episode_index + 1,
                total_reward,
                steps,
            )

            # Save TradingEnv render logs after every episode so the Flask
            # renderer can replay any episode without re-running the agent.
            if (
                config.env_id == "TradingEnv"
                and env is not None
                and hasattr(env, "save_for_render")
            ):
                env.save_for_render(dir="render_logs")

        results = {
            "env_id": config.env_id,
            "policy_mode": config.policy_mode,
            "episodes": config.episodes,
            "max_steps_per_episode": config.max_steps_per_episode,
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "mean_reward": float(fmean(episode_rewards)) if episode_rewards else 0.0,
            "best_reward": float(max(episode_rewards)) if episode_rewards else 0.0,
        }
        # Save results to file for graphing
        import json
        from pathlib import Path

        reward_path = Path(config.reward_output_file)
        if reward_path.parent != Path("."):
            reward_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reward_path, "w") as f:
            json.dump(results, f, indent=2)
        return results
    finally:
        runtime.close()

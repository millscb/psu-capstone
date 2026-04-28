#!/usr/bin/env python3
"""Run the agent WebSocket server for real-time visualization.

Branch behavior summary:
- ``--policy`` flag chooses the active policy mode.
- ``brain`` is the default to match current experimentation workflow.
- If ``ppo`` is selected, a fixed warm-up training pass is run before serving.
- currently - No runtime policy switching is performed by the server.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal, cast

sys.path.insert(0, str(Path(__file__).parent / "src"))

from htmrl.agent_layer.agent_runtime import (  # noqa: E402
    AgentRuntimeConfig,
    run_local_session,
    run_server,
)
from htmrl.log import get_logger  # noqa: E402


def main_sync(args: argparse.Namespace) -> None:
    """Run either server mode or local mode from one runtime config."""

    logger = get_logger(None)
    policy_mode = cast(Literal["q_table", "brain", "ppo"], args.policy)
    render_mode = None if args.render_mode == "none" else args.render_mode
    step_delay = args.step_delay
    if step_delay is None and args.mode == "local" and render_mode == "human":
        # Default to a readable pace when a human-rendered window is enabled.
        step_delay = 0.08
    if step_delay is None:
        step_delay = 0.0

    reward_file = args.reward_file
    if reward_file is None:
        reward_file = f"episode_rewards_{args.env}.json"

    config = AgentRuntimeConfig(
        env_id=args.env,
        policy_mode=policy_mode,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        render_mode=render_mode,
        reward_output_file=reward_file,
        step_delay_seconds=max(0.0, step_delay),
        non_spatial=args.no_spatial,
        non_temporal=args.no_temporal,
        host=args.host,
        port=args.port,
        ppo_pretrain_timesteps=args.ppo_pretrain_timesteps,
    )

    try:
        logger.info(
            "Starting %s mode for env %s with policy %s",
            args.mode,
            args.env,
            policy_mode,
        )
        if policy_mode == "ppo":
            logger.info(
                "PPO mode selected: runtime will pre-train for %d timesteps before normal execution.",
                config.ppo_pretrain_timesteps,
            )
        if args.mode == "server":
            asyncio.run(run_server(config))
        else:
            metrics = run_local_session(config)
            logger.info(
                "Local session summary: episodes=%s mean_reward=%.3f best_reward=%.3f",
                metrics["episodes"],
                metrics["mean_reward"],
                metrics["best_reward"],
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        raise


if __name__ == "__main__":
    defaults = AgentRuntimeConfig()

    parser = argparse.ArgumentParser(
        description="Run the unified agent runtime in server or local mode."
    )
    parser.add_argument(
        "--env",
        type=str,
        default=defaults.env_id,
        help=f"Environment id or frontend env alias to use (default: {defaults.env_id})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="server",
        choices=["server", "local"],
        help="Run websocket server mode or local Gym stepping mode (default: server)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=defaults.host,
        help=f"WebSocket server bind address (default: {defaults.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=defaults.port,
        help=f"WebSocket server bind port (default: {defaults.port})",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=defaults.policy_mode,
        choices=["ppo", "brain", "q_table"],
        help=f"Starting policy mode (default: {defaults.policy_mode})",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=defaults.episodes,
        help=f"Number of episodes for local/train loops (default: {defaults.episodes})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults.max_steps_per_episode,
        help=(
            f"Maximum steps per episode for local/train loops "
            f"(default: {defaults.max_steps_per_episode})"
        ),
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        help="Gym render mode for local runs; use 'none' to disable (default: none, local defaults to human)",
    )
    parser.add_argument(
        "--reward-file",
        type=str,
        default=None,
        help=(
            "Output file for local-run metrics JSON "
            "(default: episode_rewards_<env>.json in the repo root)"
        ),
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help=(
            "Delay in seconds after each local env step for human-readable playback "
            "(default: 0.08 for --mode local with --render-mode human, otherwise 0.0)"
        ),
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        default=False,
        help="Disable Spatial Pooling in the HTM brain (default: spatial enabled)",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        default=False,
        help="Disable Temporal Memory in the HTM brain (default: temporal enabled)",
    )
    parser.add_argument(
        "--ppo-pretrain-timesteps",
        type=int,
        default=defaults.ppo_pretrain_timesteps,
        help=(
            "Warm-up timesteps used when --policy ppo is selected "
            f"(default: {defaults.ppo_pretrain_timesteps})"
        ),
    )

    args = parser.parse_args()
    main_sync(args)

#!/usr/bin/env python3
"""
run_all_envs.py

Easily run all supported RL environments locally and graph performance after training.

Usage:
    python run_all_envs.py [--episodes N] [--max-steps N] [--no-render] [--step-delay S] [--graph]

Options:
    --episodes N       Number of episodes to run (default: 5)
    --max-steps N      Maximum steps per episode (default: 1000)
    --no-render        Disable environment window (headless mode)
    --step-delay S     Sleep S seconds after each step (default: 0.08 when rendering, else 0)
    --graph            Plot episode rewards after all runs
    --no-trade-render  Skip launching the TradingEnv Flask renderer after the run

Environments tested:
    - CartPole-v1
    - FrozenLake-v1
    - MountainCar-v0
    - Pendulum-v1
    - LunarLander-v3
    - TradingEnv
    - (add more as needed)

Requires: matplotlib, pandas, gymnasium, numpy
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# List of environments to test
envs = [
    "CartPole-v1",
    "FrozenLake-v1",
    "MountainCar-v0",
    "Pendulum-v1",
    "LunarLander-v3",
    "TradingEnv",  # requires optional dependency 'gym_trading_env'
]

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT_SERVER = REPO_ROOT / "run_agent_server.py"
REWARD_FILE_TEMPLATE = "episode_rewards_{env}.json"


def reward_file_path(env):
    return REPO_ROOT / REWARD_FILE_TEMPLATE.format(env=env)


RENDER_TRADING_SCRIPT = REPO_ROOT / "scripts" / "render_trading.py"


def launch_trading_renderer() -> None:
    """Start the gym-trading-env Flask renderer in a subprocess and wait for Ctrl+C."""
    render_logs = REPO_ROOT / "render_logs"
    if not render_logs.exists() or not any(render_logs.iterdir()):
        print("[WARN] No render_logs found – skipping TradingEnv renderer.")
        return
    print("\n=== Launching TradingEnv renderer at http://127.0.0.1:5000 ===")
    print("Press Ctrl+C to stop the renderer and continue.")
    try:
        subprocess.run(
            [sys.executable, str(RENDER_TRADING_SCRIPT)],
            cwd=REPO_ROOT,
        )
    except KeyboardInterrupt:
        print("\n[INFO] TradingEnv renderer stopped.")


def run_env(
    env,
    episodes,
    max_steps,
    render,
    step_delay,
    no_spatial=False,
    no_temporal=False,
    policy="brain",
    log_level="DEBUG",
):
    print(f"\n=== Running {env} for {episodes} episodes | policy={policy} ===")
    cmd = [
        sys.executable,
        str(RUN_AGENT_SERVER),
        "--env",
        env,
        "--mode",
        "local",
        "--policy",
        policy,
        "--log-level",
        log_level,
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--reward-file",
        str(reward_file_path(env)),
        "--step-delay",
        str(step_delay),
    ]
    if no_spatial:
        cmd.append("--no-spatial")
    if no_temporal:
        cmd.append("--no-temporal")
    if render:
        cmd += ["--render-mode", "human"]
    else:
        cmd += ["--render-mode", "none"]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] {env} failed with exit code {result.returncode}")
    else:
        print(f"[SUCCESS] {env} completed.")


def plot_rewards(envs, episodes):
    plt.figure(figsize=(10, 6))
    for env in envs:
        reward_file = reward_file_path(env)
        if not reward_file.exists():
            print(f"[WARN] No reward file for {env}, skipping plot.")
            continue
        with reward_file.open("r") as f:
            payload = json.load(f)
        rewards = payload.get("episode_rewards", payload)
        policy = payload.get("policy_mode", "unknown") if isinstance(payload, dict) else "unknown"
        plt.plot(rewards, label=f"{env} ({policy})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Episode Rewards over {episodes} Episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run all RL environments and graph results.")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes per environment"
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps per episode")
    parser.add_argument(
        "--render",
        action="store_true",
        dest="render",
        help="Show environment window (human render mode)",
    )
    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="Disable environment window (headless mode)",
    )
    parser.set_defaults(render=True)
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Delay in seconds after each step (default: 0.08 when rendering, otherwise 0)",
    )
    parser.add_argument("--graph", action="store_true", help="Plot episode rewards after all runs")
    parser.add_argument(
        "--no-trade-render",
        action="store_true",
        default=False,
        help="Skip launching the TradingEnv Flask renderer after the TradingEnv run",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="brain",
        choices=["brain", "ppo", "q_table"],
        help="Agent policy mode for all environments (default: brain)",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        default=False,
        help="Disable Spatial Pooling in the HTM brain for all environments",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        default=False,
        help="Disable Temporal Memory in the HTM brain for all environments",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level forwarded to run_agent_server (default: DEBUG)",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        metavar="ENV",
        help=(
            "Subset of environments to run (default: all). "
            "Example: --envs Pendulum-v1 CartPole-v1"
        ),
    )
    args = parser.parse_args()

    if args.step_delay is not None:
        step_delay = args.step_delay
    elif args.render:
        step_delay = 0.08
    else:
        step_delay = 0.0

    target_envs = envs
    if args.envs is not None:
        unknown = [e for e in args.envs if e not in envs]
        if unknown:
            print(f"[WARN] Unknown env(s) ignored: {unknown}")
        target_envs = [e for e in args.envs if e in envs]

    for env in target_envs:
        run_env(
            env,
            args.episodes,
            args.max_steps,
            args.render,
            step_delay,
            args.no_spatial,
            args.no_temporal,
            args.policy,
            args.log_level,
        )
        if env == "TradingEnv" and not args.no_trade_render:
            launch_trading_renderer()

    if args.graph:
        plot_rewards(target_envs, args.episodes)


if __name__ == "__main__":
    main()

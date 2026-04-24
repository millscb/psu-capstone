#!/usr/bin/env python3
"""Run an end-to-end HTMRL research pipeline demo.

This script showcases HTMRL as a reproducible research workflow:
1. Encoder stage: run encoder demonstrations to verify SDR generation.
2. Training/evaluation stage: run multiple policy experiments.
3. Analysis stage: summarize metrics and generate comparison plots.

Outputs are written to reports/research_demo/ by default and overwritten each run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT_SERVER = REPO_ROOT / "run_agent_server.py"
ENCODER_DEMO = REPO_ROOT / "scripts" / "encoder_types.py"

DEFAULT_ENVS = [
    "CartPole-v1",
    "FrozenLake-v1",
    "MountainCar-v0",
    "Pendulum-v1",
    "LunarLander-v3",
]

DISCRETE_ACTION_ENVS = {
    "CartPole-v1",
    "FrozenLake-v1",
    "MountainCar-v0",
    "LunarLander-v3",
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    policy: str
    no_spatial: bool = False
    no_temporal: bool = False


EXPERIMENTS = [
    ExperimentConfig(name="brain_full", policy="brain"),
    ExperimentConfig(name="brain_no_spatial", policy="brain", no_spatial=True),
    ExperimentConfig(name="brain_no_temporal", policy="brain", no_temporal=True),
    ExperimentConfig(name="ppo", policy="ppo"),
    ExperimentConfig(name="q_table", policy="q_table"),
]


@dataclass
class RunResult:
    env: str
    experiment: str
    policy: str
    reward_file: Path
    succeeded: bool
    exit_code: int
    mean_reward: float | None = None
    best_reward: float | None = None
    episode_rewards: list[float] | None = None
    error: str | None = None


def _load_reward_payload(path: Path) -> tuple[float, float, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        rewards = [float(x) for x in payload.get("episode_rewards", [])]
        mean_reward = float(payload.get("mean_reward", 0.0))
        best_reward = float(payload.get("best_reward", 0.0))
    else:
        rewards = [float(x) for x in payload]
        if rewards:
            mean_reward = float(sum(rewards) / len(rewards))
            best_reward = float(max(rewards))
        else:
            mean_reward = 0.0
            best_reward = 0.0

    return mean_reward, best_reward, rewards


def _run_encoder_stage(skip: bool) -> int:
    if skip:
        print("[Stage 1/3] Encoder demo skipped.")
        return 0

    print("[Stage 1/3] Running encoder demo (no plot) to verify SDR pipeline...")
    cmd = [sys.executable, str(ENCODER_DEMO), "--no-plot"]
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        print(f"[WARN] Encoder demo failed with code {completed.returncode}.")
    else:
        print("[OK] Encoder demo completed.")
    return completed.returncode


def _run_single_experiment(
    env: str,
    exp: ExperimentConfig,
    episodes: int,
    max_steps: int,
    ppo_max_steps: int,
    render: bool,
    step_delay: float,
    log_level: str,
    ppo_pretrain_timesteps: int,
    run_dir: Path,
) -> RunResult:
    reward_file = run_dir / "rewards" / f"{env}__{exp.name}.json"
    reward_file.parent.mkdir(parents=True, exist_ok=True)

    if exp.policy == "q_table" and env not in DISCRETE_ACTION_ENVS:
        return RunResult(
            env=env,
            experiment=exp.name,
            policy=exp.policy,
            reward_file=reward_file,
            succeeded=False,
            exit_code=2,
            error="Skipped: q_table requires a discrete action space.",
        )

    effective_max_steps = ppo_max_steps if exp.policy == "ppo" else max_steps

    cmd = [
        sys.executable,
        str(RUN_AGENT_SERVER),
        "--mode",
        "local",
        "--env",
        env,
        "--policy",
        exp.policy,
        "--episodes",
        str(episodes),
        "--max-steps",
        str(effective_max_steps),
        "--render-mode",
        "human" if render else "none",
        "--step-delay",
        str(step_delay),
        "--log-level",
        log_level,
        "--reward-file",
        str(reward_file),
    ]

    if exp.no_spatial:
        cmd.append("--no-spatial")
    if exp.no_temporal:
        cmd.append("--no-temporal")
    if exp.policy == "ppo":
        cmd += ["--ppo-pretrain-timesteps", str(ppo_pretrain_timesteps)]

    completed = subprocess.run(cmd, cwd=REPO_ROOT)

    result = RunResult(
        env=env,
        experiment=exp.name,
        policy=exp.policy,
        reward_file=reward_file,
        succeeded=completed.returncode == 0,
        exit_code=completed.returncode,
    )

    if result.succeeded and reward_file.exists():
        mean_reward, best_reward, episode_rewards = _load_reward_payload(reward_file)
        result.mean_reward = mean_reward
        result.best_reward = best_reward
        result.episode_rewards = episode_rewards
    elif not result.succeeded:
        result.error = f"Process exited with code {completed.returncode}."
    else:
        result.error = "Run completed but reward file was not produced."

    return result


def _plot_by_env(results: list[RunResult], plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for old_plot in plots_dir.glob("*_comparison.png"):
        old_plot.unlink()

    envs = sorted({r.env for r in results})
    for env in envs:
        env_results = [r for r in results if r.env == env and r.episode_rewards]
        if not env_results:
            continue

        fig = plt.figure(figsize=(10, 6))
        for item in env_results:
            assert item.episode_rewards is not None
            x_values = list(range(1, len(item.episode_rewards) + 1))
            plt.plot(x_values, item.episode_rewards, label=item.experiment)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"HTMRL Research Pipeline - {env}")
        plt.legend()
        plt.tight_layout()

        out_file = plots_dir / f"{env}_comparison.png"
        fig.savefig(out_file)
        plt.close(fig)
        saved.append(out_file)

    return saved


def _write_summary(run_dir: Path, results: list[RunResult]) -> Path:
    summary_path = run_dir / "summary.json"

    by_env: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        entry = {
            "experiment": result.experiment,
            "policy": result.policy,
            "succeeded": result.succeeded,
            "exit_code": result.exit_code,
            "mean_reward": result.mean_reward,
            "best_reward": result.best_reward,
            "reward_file": str(result.reward_file.relative_to(REPO_ROOT)),
            "error": result.error,
        }
        by_env.setdefault(result.env, []).append(entry)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": ["encoder_demo", "experiment_matrix", "analysis_plots"],
        "experiments": [e.__dict__ for e in EXPERIMENTS],
        "results": by_env,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return summary_path


def _reset_run_dir(run_dir: Path) -> None:
    if run_dir == REPO_ROOT:
        raise ValueError("Refusing to clear the repository root as a report directory.")

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _print_console_summary(results: list[RunResult]) -> None:
    print("\nResearch Summary")
    print("=" * 80)
    print(f"{'ENV':<18} {'EXPERIMENT':<18} {'STATUS':<10} {'MEAN':>10} {'BEST':>10}")
    for item in results:
        status = "OK" if item.succeeded else "FAILED"
        mean_text = f"{item.mean_reward:.3f}" if item.mean_reward is not None else "-"
        best_text = f"{item.best_reward:.3f}" if item.best_reward is not None else "-"
        print(f"{item.env:<18} {item.experiment:<18} {status:<10} {mean_text:>10} {best_text:>10}")
        if item.error:
            print(f"  note: {item.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Showcase HTMRL as an end-to-end research pipeline."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per env/experiment combination (default: 10).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps per episode (default: 100).",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_ENVS,
        help="Environment ids to include in the experiment matrix.",
    )
    parser.add_argument(
        "--skip-encoder-demo",
        action="store_true",
        default=False,
        help="Skip stage 1 encoder showcase.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Use Gym human rendering for all runs.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Delay between local env steps in seconds.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level forwarded to run_agent_server.",
    )
    parser.add_argument(
        "--ppo-pretrain-timesteps",
        type=int,
        default=20_000,
        help="Warm-up timesteps for PPO runs inside the research demo (default: 20000).",
    )
    parser.add_argument(
        "--ppo-max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode used for PPO runs only (default: 200).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to reports/research_demo/ (overwritten each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.out_dir:
        run_dir = Path(args.out_dir).expanduser().resolve()
    else:
        run_dir = (REPO_ROOT / "reports" / "research_demo").resolve()

    _reset_run_dir(run_dir)

    _run_encoder_stage(skip=args.skip_encoder_demo)

    print("[Stage 2/3] Running experiment matrix (policy + HTM ablations)...")
    results: list[RunResult] = []
    for env in args.envs:
        for exp in EXPERIMENTS:
            print(f"  -> {env} | {exp.name}")
            result = _run_single_experiment(
                env=env,
                exp=exp,
                episodes=args.episodes,
                max_steps=args.max_steps,
                ppo_max_steps=args.ppo_max_steps,
                render=args.render,
                step_delay=max(0.0, args.step_delay),
                log_level=args.log_level,
                ppo_pretrain_timesteps=args.ppo_pretrain_timesteps,
                run_dir=run_dir,
            )
            results.append(result)

    print("[Stage 3/3] Building analysis artifacts...")
    plots = _plot_by_env(results, run_dir / "plots")
    summary_path = _write_summary(run_dir, results)
    _print_console_summary(results)

    print("\nArtifacts")
    print("=" * 80)
    print(f"Summary JSON: {summary_path}")
    if plots:
        print("Per-env plots:")
        for plot in plots:
            print(f"  - {plot}")
    else:
        print("No plots were generated (no successful reward series found).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end HTM demo: InputHandler -> TradingEnv -> Agent step loop.

This script demonstrates a full pipeline using the project's runtime pieces:
1. Load raw market data from an Excel file via InputHandler.
2. Normalize/validate it into TradingEnv-compatible OHLCV + DatetimeIndex.
3. Build EnvAdapter -> Trainer -> Brain -> Agent.
4. Run local agent steps directly inside the dataset.
5. Optionally save TradingEnv render logs for scripts/render_trading.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep this script runnable from the repo root without installing the package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from htmrl.agent_layer.agent import Agent  # noqa: E402
from htmrl.agent_layer.agent_runtime import AgentRuntimeConfig  # noqa: E402
from htmrl.agent_layer.pullin.pullin_brain import Brain  # noqa: E402
from htmrl.agent_layer.train import Trainer  # noqa: E402
from htmrl.environment.env_adapter import EnvAdapter  # noqa: E402
from htmrl.input_layer.input_handler import InputHandler  # noqa: E402


def _normalize_market_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary OHLCV-ish columns into TradingEnv-required schema."""
    # EncoderLayer prep stage: normalize tabular columns into the shape expected
    # before encoder parameterization and environment observation wiring.
    df = raw_df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    alias_map = {
        "date": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
        "volume_usd": "volume",
        "vol": "volume",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required_cols = ["timestamp", "open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required market columns: {missing}. Found: {list(df.columns)}")

    if "volume" not in df.columns:
        # concat_ESData.xlsx has no volume column. Keep the pipeline runnable by
        # deriving a simple proxy from absolute close deltas.
        df["volume"] = df["close"].diff().abs().fillna(1.0) + 1.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)

    # EncoderLayer feature stage: create feature_* columns consumed downstream
    # as encoded observation inputs.
    # TradingEnv emits columns containing "feature" as observation inputs.
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"] - 1.0
    df["feature_high"] = df["high"] / df["close"] - 1.0
    df["feature_low"] = df["low"] / df["close"] - 1.0
    rolling_max = df["volume"].rolling(100, min_periods=1).max().replace(0, 1.0)
    df["feature_volume"] = df["volume"] / rolling_max
    df = df.dropna()

    if df.empty:
        raise ValueError("DataFrame became empty after normalization and feature generation.")

    return df


def _build_agent(df: pd.DataFrame, args: argparse.Namespace) -> tuple[Agent, EnvAdapter]:
    """Build EnvAdapter + Trainer-built Brain + Agent for TradingEnv."""
    # AgentLayer stage: EnvAdapter boundary between Gym/TradingEnv and the agent loop.
    # EnvAdapter wraps TradingEnv and provides bridge payloads consumed by Agent.
    adapter = EnvAdapter(
        "TradingEnv",
        name=args.market_name,
        df=df,
        positions=[-1, 0, 1],
        trading_fees=args.trading_fees,
        borrow_interest_rate=args.borrow_interest_rate,
    )

    config = AgentRuntimeConfig(
        env_id="TradingEnv",
        policy_mode=args.policy,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        input_size=args.input_size,
        cells_per_column=args.cells_per_column,
        resolution=args.resolution,
        seed=args.seed,
        non_spatial=args.no_spatial,
        non_temporal=args.no_temporal,
    )

    # AgentLayer stage: Trainer builds Brain fields and binds encoders to match adapter specs.
    trainer = Trainer(Brain({}))
    brain = trainer.build_brain_for_env(adapter, config)

    # AgentLayer stage: Agent orchestrates Brain inference, action selection, and env stepping.
    agent = Agent(
        brain=brain,
        adapter=adapter,
        episodes=args.episodes,
        policy_mode=args.policy,
    )
    return agent, adapter


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full InputHandler -> TradingEnv -> Agent step loop."""
    # InputLayer stage: InputHandler ingests raw files and returns normalized records.
    ih = InputHandler()
    normalized_records = ih.input_data(args.dataset)
    raw_df = pd.DataFrame(normalized_records)

    # EncoderLayer stage: schema + feature conversion into TradingEnv-ready frame.
    df = _normalize_market_dataframe(raw_df)

    if args.max_steps <= 0:
        # Use the full normalized dataset when max_steps is not explicitly set.
        args.max_steps = len(df)

    agent, adapter = _build_agent(df, args)
    rewards: list[float] = []
    steps_per_episode: list[int] = []

    try:
        env = getattr(adapter, "_env", None)
        for episode in range(1, args.episodes + 1):
            # AgentLayer runtime: episode reset through Agent -> EnvAdapter bridge.
            agent.reset_episode()
            done = False
            total_reward = 0.0
            steps = 0

            # AgentLayer runtime loop:
            # Agent.step internally runs Brain.step -> action selection -> EnvAdapter.step_bridge.
            while not done and steps < args.max_steps:
                transition = agent.step(learn=True)
                total_reward += float(transition["reward"])
                # Gymnasium episodes end on either terminal condition or truncation.
                done = bool(transition["terminated"] or transition["truncated"])
                steps += 1

            rewards.append(total_reward)
            steps_per_episode.append(steps)
            print(f"Episode {episode:>4d}: reward={total_reward:.4f} steps={steps}")

            if args.save_render_logs and env is not None and hasattr(env, "save_for_render"):
                # Compatible with scripts/render_trading.py Flask replay workflow.
                env.save_for_render(dir=args.render_logs_dir)

    finally:
        adapter.close()

    return {
        # AgentLayer output stage: run-level metrics export for analysis/plotting.
        "dataset": str(Path(args.dataset).resolve()),
        "policy_mode": args.policy,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "rows_used": len(df),
        "episode_rewards": rewards,
        "episode_steps": steps_per_episode,
        "mean_reward": float(sum(rewards) / len(rewards)) if rewards else 0.0,
        "best_reward": float(max(rewards)) if rewards else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the my_htm pipeline demo."""
    parser = argparse.ArgumentParser(
        description="InputHandler-to-TradingEnv end-to-end HTM pipeline demo."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "data" / "concat_ESData.xlsx"),
        help="Path to source dataset (default: data/concat_ESData.xlsx)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode; 0 means use full dataset length",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="brain",
        choices=["brain", "q_table", "ppo"],
        help="Policy mode for Agent",
    )
    parser.add_argument("--input-size", type=int, default=256, help="Input SDR size")
    parser.add_argument("--cells-per-column", type=int, default=8, help="Cells per column")
    parser.add_argument("--resolution", type=float, default=0.01, help="RDSE resolution")
    parser.add_argument("--seed", type=int, default=5, help="Random seed")
    parser.add_argument("--market-name", type=str, default="ES", help="TradingEnv market name")
    parser.add_argument(
        "--trading-fees",
        type=float,
        default=0.01 / 100,
        help="Per-trade fee ratio",
    )
    parser.add_argument(
        "--borrow-interest-rate",
        type=float,
        default=0.0003 / 100,
        help="Borrow interest per step",
    )
    parser.add_argument(
        "--save-render-logs",
        action="store_true",
        help="Save TradingEnv logs for scripts/render_trading.py",
    )
    parser.add_argument(
        "--render-logs-dir",
        type=str,
        default="render_logs",
        help="Directory used with env.save_for_render(...)",
    )
    parser.add_argument("--no-spatial", action="store_true", help="Disable spatial pooling")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal memory")
    parser.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "reports" / "my_htm_results.json"),
        help="Output metrics JSON path",
    )
    return parser


def main() -> None:
    """Run the my_htm end-to-end demo."""
    parser = build_parser()
    args = parser.parse_args()

    result = run_pipeline(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()

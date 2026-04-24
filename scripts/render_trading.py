#!/usr/bin/env python3
"""Launch the gym-trading-env Flask renderer for saved TradingEnv episodes.

Usage:
    # 1. Run the agent to produce render logs:
    #    python run_agent_server.py --env TradingEnv --mode local --episodes 5

    # 2. Open the interactive chart in your browser:
    #    python scripts/render_trading.py

    # 3. Optional: point at a different log directory:
    #    python scripts/render_trading.py --dir path/to/render_logs

The renderer starts a Flask server (default http://127.0.0.1:5000).
Press Ctrl+C to stop it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    from gym_trading_env.renderer import Renderer
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "Install with: pip install gym-trading-env pandas pyecharts flask",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render saved TradingEnv episodes via a local Flask web app."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="render_logs",
        help="Directory containing render logs saved by env.save_for_render() (default: render_logs)",
    )
    parser.add_argument(
        "--sma",
        type=int,
        nargs="*",
        default=[10, 20],
        metavar="N",
        help="Simple-moving-average windows to overlay on the price chart (default: 10 20)",
    )
    args = parser.parse_args()

    log_dir = Path(args.dir)
    if not log_dir.exists():
        print(
            f"Render log directory not found: {log_dir}\n"
            "Run the agent first with:\n"
            "  python run_agent_server.py --env TradingEnv --mode local --episodes 5",
            file=sys.stderr,
        )
        sys.exit(1)

    renderer = Renderer(render_logs_dir=str(log_dir))

    # Add SMA overlays
    colors = ["purple", "blue", "orange", "green", "red"]
    for i, window in enumerate(args.sma or []):
        color = colors[i % len(colors)]
        renderer.add_line(
            name=f"sma{window}",
            function=lambda df, w=window: df["close"].rolling(w).mean(),
            line_options={"width": 1, "color": color},
        )

    # Add portfolio metrics
    renderer.add_metric(
        name="Total Return",
        function=lambda df: (
            f"{(df['portfolio_valuation'].iloc[-1] / df['portfolio_valuation'].iloc[0] - 1) * 100:.2f}%"
            if "portfolio_valuation" in df.columns and len(df) > 0
            else "N/A"
        ),
    )
    renderer.add_metric(
        name="Final Valuation",
        function=lambda df: (
            f"${df['portfolio_valuation'].iloc[-1]:,.2f}"
            if "portfolio_valuation" in df.columns and len(df) > 0
            else "N/A"
        ),
    )

    print("Starting TradingEnv renderer at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.\n")
    renderer.run()


if __name__ == "__main__":
    main()

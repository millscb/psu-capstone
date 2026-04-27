import argparse
import json
import os
import sys
from pathlib import Path

# Select backend before importing pyplot to prevent Qt/xcb crashes in non-interactive runs.
if "--show" in sys.argv:
    try:
        import tkinter  # noqa: F401

        os.environ.setdefault("MPLBACKEND", "TkAgg")
    except ModuleNotFoundError:
        # Browser-based interactive backend that works without local GUI libs.
        os.environ.setdefault("MPLBACKEND", "WebAgg")
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def _load_rewards(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rewards = data.get("episode_rewards", [])
    else:
        rewards = data
    return [float(value) for value in rewards]


def _load_policy(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        policy = data.get("policy_mode")
        if isinstance(policy, str) and policy:
            return policy
    return None


def _default_reward_files(root: Path) -> list[Path]:
    candidates = sorted(root.glob("episode_rewards*.json"))
    return [path for path in candidates if path.is_file()]


def _series_label(file_path: Path) -> str:
    label = file_path.stem.replace("episode_rewards_", "")
    if label == "episode_rewards":
        label = "all"
    policy = _load_policy(file_path)
    if policy:
        label = f"{label} ({policy})"
    return label


def plot_rewards(files: list[Path], *, output: Path | None = None, show: bool = False) -> None:
    if not files:
        raise FileNotFoundError("No reward files found. Expected episode_rewards*.json")

    plt.figure(figsize=(10, 6))
    for file_path in files:
        rewards = _load_rewards(file_path)
        if not rewards:
            continue
        label = _series_label(file_path)
        episodes = range(1, len(rewards) + 1)
        plt.plot(episodes, rewards, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")
    plt.legend()
    plt.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        print(f"Saved plot to: {output}")
        return

    # Only open an interactive window when explicitly requested.
    if show:
        backend = plt.get_backend().lower()
        if "agg" in backend and "webagg" not in backend:
            raise RuntimeError(
                "Interactive plotting requested (--show), but no interactive backend is available. "
                "Install tkinter GUI support or use --output."
            )
        print(f"Showing plot with Matplotlib backend: {plt.get_backend()}")
        plt.show()
        return

    # Default non-interactive behavior: write an image so VS Code Run never hard-crashes on GUI backend issues.
    fallback = Path("reward_plot.png")
    plt.savefig(fallback)
    print(f"Saved plot to: {fallback} (use --show for interactive window)")


def plot_rewards_separate(
    files: list[Path], *, output_dir: Path | None = None, show: bool = False
) -> None:
    if not files:
        raise FileNotFoundError("No reward files found. Expected episode_rewards*.json")

    generated = 0
    for file_path in files:
        rewards = _load_rewards(file_path)
        if not rewards:
            continue

        label = _series_label(file_path)
        safe_name = file_path.stem.replace(" ", "_")
        fig = plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Episode Rewards Over Time - {label}")
        plt.tight_layout()

        if show:
            backend = plt.get_backend().lower()
            if "agg" in backend and "webagg" not in backend:
                raise RuntimeError(
                    "Interactive plotting requested (--show), but no interactive backend is available. "
                    "Install tkinter GUI support or use --output/--output-dir."
                )
            print(f"Showing plot with Matplotlib backend: {plt.get_backend()} | {label}")
            plt.show()
        else:
            target_dir = output_dir or Path(".")
            target_dir.mkdir(parents=True, exist_ok=True)
            out_file = target_dir / f"reward_plot_{safe_name}.png"
            fig.savefig(out_file)
            print(f"Saved plot to: {out_file}")

        plt.close(fig)
        generated += 1

    if generated == 0:
        raise ValueError("No plottable reward series found in the provided files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one or more reward JSON files.")
    parser.add_argument(
        "files",
        nargs="*",
        help="Reward JSON files. If omitted, uses episode_rewards*.json in repo root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path (e.g., reward_plot.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help=(
            "Show an interactive Matplotlib view. Uses TkAgg when available, "
            "otherwise falls back to WebAgg."
        ),
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        default=False,
        help="Render one graph per reward file instead of combining all series.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for separate graph outputs when using --separate.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    files = [Path(item).expanduser().resolve() for item in args.files]
    if not files:
        files = _default_reward_files(repo_root)

    output = Path(args.output).expanduser().resolve() if args.output else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if args.separate:
        plot_rewards_separate(files, output_dir=output_dir, show=args.show)
    else:
        plot_rewards(files, output=output, show=args.show)


if __name__ == "__main__":
    main()

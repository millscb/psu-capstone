"""Logging configuration for the HTM Reinforcement Learning project.

This module sets up a centralized logger with formatted output to stdout.
The logging level can be controlled via the DEBUG environment variable.

Attributes:
    logger: The configured logger instance for the entire project. Use this
        logger throughout the codebase by importing it:
        `from htmrl.log import logger`.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("htmrl")
_handler = logging.StreamHandler(stream=sys.stdout)


class ColorFormatter(logging.Formatter):
    """Format logs with ANSI colors for terminal readability."""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[41m",  # red background
    }

    BRAIN_NAME_COLOR = "\033[95m"  # bright magenta

    def __init__(self, use_color: bool = False) -> None:
        super().__init__(fmt="%(levelname)s [%(name)s]: %(message)s")
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self._use_color:
            return super().format(record)

        level_text = record.levelname
        level_color = self.LEVEL_COLORS.get(record.levelno, "")
        if level_color:
            level_text = f"{self.BOLD}{level_color}{record.levelname}{self.RESET}"

        logger_name = record.name
        if logger_name.endswith(".Brain") or logger_name == "htmrl.Brain":
            logger_name = f"{self.BOLD}{self.BRAIN_NAME_COLOR}{record.name}{self.RESET}"

        message = record.getMessage()
        return f"{level_text} [{logger_name}]: {message}"


_force_color = os.environ.get("HTMRL_COLOR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
_disable_color = os.environ.get("NO_COLOR") is not None
_stream_supports_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
_use_color = (_stream_supports_color or _force_color) and not _disable_color

_handler.setFormatter(ColorFormatter(use_color=_use_color))

if not logger.handlers:
    logger.addHandler(_handler)

if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


# SF-F-1 (Good)


class LoggerManager:
    """Manage project logging and per-brain training/evaluation artifacts."""

    def __init__(self) -> None:
        self._report_artifact_path: Path | None = None

        self._validated_datasets: dict[str, dict[str, list[Any]]] = {}
        self._last_evaluation_parameters_by_brain: dict[str, dict[str, Any]] = {}
        self._latest_prediction_report_paths: dict[str, Path] = {}
        self._latest_prediction_report_texts: dict[str, str] = {}

    # SF-F-12
    def get_logger(self, source: Any | None = None) -> logging.Logger:
        """Return the shared logger or a child logger for a specific source.

        Creates child loggers with hierarchical names based on the source, which
        helps identify the origin of log messages in the output.

        Args:
            source: Source identifier - can be None (returns root logger), a string name,
                a class type, or a class instance. Class types and instances generate
                child loggers named after the class.

        Returns:
            The root logger or a child logger with an appropriate name suffix.
        """
        if source is None:
            return logger

        if isinstance(source, str):
            suffix = source
        elif isinstance(source, type):
            suffix = source.__name__
        else:
            suffix = source.__class__.__name__

        return logger.getChild(suffix)

    # SF-F-6
    def set_report_artifact_path(self, path: str | Path) -> None:
        """Configure and create the root directory used for report artifacts."""
        artifact_path = Path(path).expanduser().resolve()
        artifact_path.mkdir(parents=True, exist_ok=True)
        self._report_artifact_path = artifact_path
        logger.info("Report artifact path set to: %s", artifact_path)

    def get_report_artifact_path(self) -> Path | None:
        """Return the configured report artifact directory, if available."""
        if self._report_artifact_path is None:
            logger.warning("No report artifact path has been configured yet.")
            return None
        return self._report_artifact_path

    def build_report_artifact_path(self, filename: str) -> Path:
        """Build an artifact path under the configured report directory."""
        if self._report_artifact_path is None:
            raise ValueError("Report artifact path has not been configured.")
        return self._report_artifact_path / filename

    # SF-F-10
    def _get_brain_key(self, brain: Any) -> str:
        if brain is None:
            raise ValueError("brain cannot be None.")
        if not hasattr(brain, "brain_id"):
            raise ValueError("brain must have a brain_id attribute.")

        brain_key = str(brain.brain_id).strip()
        if not brain_key:
            raise ValueError("brain.brain_id cannot be empty.")

        return brain_key

    def _get_brain_dir(self, brain: Any) -> Path:
        brain_key = self._get_brain_key(brain)
        brain_dir = self.build_report_artifact_path(brain_key)
        brain_dir.mkdir(parents=True, exist_ok=True)
        return brain_dir

    def _build_brain_report_path(self, brain: Any, filename: str) -> Path:
        return self._get_brain_dir(brain) / filename

    # SF-F-7
    def save_validated_dataset(
        self,
        brain: Any,
        dataset: dict[str, list[Any]],
        filename: str = "validated_dataset.json",
    ) -> Path:
        """Persist a validated dataset for a brain and cache it in memory."""
        brain_key = self._get_brain_key(brain)
        self._validated_datasets[brain_key] = copy.deepcopy(dataset)

        path = self._build_brain_report_path(brain, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, default=str)

        logger.info("Validated dataset for %s saved to %s", brain_key, path)
        return path

    def get_validated_dataset(self, brain: Any) -> dict[str, list[Any]] | None:
        """Load a previously saved validated dataset for a brain."""
        brain_key = self._get_brain_key(brain)

        if brain_key in self._validated_datasets:
            return copy.deepcopy(self._validated_datasets[brain_key])

        path = self._build_brain_report_path(brain, "validated_dataset.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._validated_datasets[brain_key] = data
            return copy.deepcopy(data)

        logger.warning("No validated dataset has been saved for %s.", brain_key)
        return None

    # SF-F-8
    def save_evaluation_parameters(
        self,
        brain: Any,
        params: dict[str, Any],
        filename: str = "evaluation_parameters.txt",
    ) -> Path:
        """Persist evaluation parameters for a brain and cache the latest copy."""
        brain_key = self._get_brain_key(brain)
        self._last_evaluation_parameters_by_brain[brain_key] = copy.deepcopy(params)

        path = self._build_brain_report_path(brain, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4, default=str)

        logger.info("Trainer evaluation parameters for %s saved to %s", brain_key, path)
        return path

    def get_last_evaluation_parameters(self, brain: Any) -> dict[str, Any] | None:
        """Return the last recorded evaluation parameters for a brain."""
        brain_key = self._get_brain_key(brain)

        if brain_key in self._last_evaluation_parameters_by_brain:
            return copy.deepcopy(self._last_evaluation_parameters_by_brain[brain_key])

        path = self._build_brain_report_path(brain, "evaluation_parameters.json")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._last_evaluation_parameters_by_brain[brain_key] = data
            return copy.deepcopy(data)

        logger.warning("No trainer evaluation parameters recorded for %s.", brain_key)
        return None

    # SF-F-1
    def log_training_progress(
        self,
        brain: Any,
        step: int,
        total_steps: int,
        inputs: dict[str, Any],
    ) -> None:
        """Log one training progress line with step counters and input values."""
        brain_key = self._get_brain_key(brain)
        input_str = ", ".join(f"{k}={v}" for k, v in inputs.items())
        logger.info(
            "[%s] Training Step %d/%d | Inputs: %s", brain_key, step, total_steps, input_str
        )

    # SF-F-2
    def save_mean_squared_error(
        self,
        brain: Any,
        mse: float,
        filename: str = "mean_squared_error.txt",
        append: bool = True,
    ) -> Path:
        """Write a mean-squared-error value to a per-brain artifact file."""
        path = self._build_brain_report_path(brain, filename)
        mode = "a" if append else "w"

        with open(path, mode, encoding="utf-8") as f:
            f.write(f"MSE={mse:.6f}\n")

        logger.info("Mean squared error saved to %s", path)
        return path

    # SF-F-4
    def output_final_training_performance(
        self,
        brain: Any,
        training_steps: int,
        test_results: dict[str, Any],
    ) -> None:
        """Format and emit a final training performance report to the logger."""
        brain_key = self._get_brain_key(brain)

        mean_squared_error = test_results.get("mean_squared_error", 0.0)
        total_prediction_failures = test_results.get("total_prediction_failures", 0)
        avg_bursting_columns = test_results.get("avg_bursting_columns", 0.0)
        field_errors = test_results.get("errors", {})
        prediction_failures = test_results.get("prediction_failures", {})

        field_errors = field_errors if isinstance(field_errors, dict) else {}
        prediction_failures = prediction_failures if isinstance(prediction_failures, dict) else {}

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(
            f"Final Training Performance Report [{brain_key}] - {datetime.now().isoformat()}"
        )
        lines.append("=" * 80)
        lines.append(f"Training Steps: {training_steps}")
        lines.append(f"Global MSE: {mean_squared_error:.6f}")
        lines.append(f"Total Prediction Failures: {total_prediction_failures}")
        lines.append(f"Average Bursting Columns: {avg_bursting_columns:.3f}")
        lines.append("")
        lines.append("Per-field Metrics:")
        lines.append("+----------------------------+---------------+-----------+")
        lines.append("| Field                      |     MSE       | Failures |")
        lines.append("+----------------------------+---------------+-----------+")

        for field_name, errors in field_errors.items():
            error_list = errors if isinstance(errors, list) else []
            mse = sum(error_list) / len(error_list) if error_list else 0.0
            failures = prediction_failures.get(field_name, 0)
            lines.append(f"| {field_name:<26} | {mse:>11.6f} | {failures:>9} |")

        lines.append("+----------------------------+---------------+-----------+")

        report = "\n".join(lines)
        logger.info("\n%s", report)

    # SF-F-5
    def save_final_training_performance(
        self,
        brain: Any,
        training_steps: int,
        test_results: dict[str, Any],
        filename: str = "final_training_performance.txt",
    ) -> Path:
        """Append a final training performance report to disk for a brain."""
        path = self._build_brain_report_path(brain, filename)

        mean_squared_error = test_results.get("mean_squared_error", 0.0)
        total_prediction_failures = test_results.get("total_prediction_failures", 0)
        avg_bursting_columns = test_results.get("avg_bursting_columns", 0.0)
        field_errors = test_results.get("errors", {})
        prediction_failures = test_results.get("prediction_failures", {})

        field_errors = field_errors if isinstance(field_errors, dict) else {}
        prediction_failures = prediction_failures if isinstance(prediction_failures, dict) else {}

        with open(path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(
                f"Final Training Performance Report [{self._get_brain_key(brain)}] - {datetime.now().isoformat()}\n"
            )
            f.write("=" * 80 + "\n")
            f.write(f"Training Steps: {training_steps}\n")
            f.write(f"Global MSE: {mean_squared_error:.6f}\n")
            f.write(f"Total Prediction Failures: {total_prediction_failures}\n")
            f.write(f"Average Bursting Columns: {avg_bursting_columns:.3f}\n")
            f.write("\n")

            f.write("Per-field Metrics:\n")
            f.write("+----------------------------+---------------+-----------+\n")
            f.write("| Field                      |     MSE       | Failures |\n")
            f.write("+----------------------------+---------------+-----------+\n")

            for field_name, errors in field_errors.items():
                error_list = errors if isinstance(errors, list) else []
                mse = sum(error_list) / len(error_list) if error_list else 0.0
                failures = prediction_failures.get(field_name, 0)
                f.write(f"| {field_name:<26} | {mse:>11.6f} | {failures:>9} |\n")

            f.write("+----------------------------+---------------+-----------+\n\n")

        logger.info("Final training performance saved to %s", path)
        return path

    # SF-F-3
    def save_agent_brain_shape(
        self,
        brain: Any,
        shape_data: dict[str, Any],
        filename: str = "agent_brain_shape.json",
    ) -> Path:
        """Persist serialized brain-shape metadata for an agent."""
        path = self._build_brain_report_path(brain, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(shape_data, f, indent=4, default=str)

        logger.info("Agent-brain shape saved to %s", path)
        return path

    # SF-F-9
    def save_prediction_report(
        self,
        brain: Any,
        report_text: str,
        filename: str = "latest_prediction_report.txt",
    ) -> Path:
        """Save and cache a plain-text prediction report for a brain."""
        brain_key = self._get_brain_key(brain)
        path = self._build_brain_report_path(brain, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)

        self._latest_prediction_report_paths[brain_key] = path
        self._latest_prediction_report_texts[brain_key] = report_text

        logger.info("Latest prediction report for %s saved to %s", brain_key, path)
        return path

    def get_latest_prediction_report(self, brain: Any) -> str | None:
        """Return the latest prediction report text for a brain, if present."""
        brain_key = self._get_brain_key(brain)

        if brain_key in self._latest_prediction_report_texts:
            return self._latest_prediction_report_texts[brain_key]

        path = self._latest_prediction_report_paths.get(brain_key)
        if path is None:
            path = self._build_brain_report_path(brain, "latest_prediction_report.txt")

        if path.exists():
            text = path.read_text(encoding="utf-8")
            self._latest_prediction_report_texts[brain_key] = text
            self._latest_prediction_report_paths[brain_key] = path
            return text

        logger.warning("No prediction report has been saved for %s.", brain_key)
        return None

    def get_latest_prediction_report_path(self, brain: Any) -> Path | None:
        """Return the artifact path of the latest saved prediction report."""
        brain_key = self._get_brain_key(brain)

        if brain_key in self._latest_prediction_report_paths:
            return self._latest_prediction_report_paths[brain_key]

        path = self._build_brain_report_path(brain, "latest_prediction_report.txt")
        if path.exists():
            self._latest_prediction_report_paths[brain_key] = path
            return path

        logger.warning("No prediction report path has been saved for %s.", brain_key)
        return None

    # SF-F-11
    def save_average_reward_per_step(
        self,
        brain: Any,
        avg_reward: float,
        filename: str = "average_reward_per_step.txt",
    ) -> Path | None:
        """Append a per-brain average reward summary entry to disk."""
        brain_key = self._get_brain_key(brain)
        path = self._build_brain_report_path(brain, filename)

        with open(path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(
                f"Average Reward Per Step Report [{brain_key}] - {datetime.now().isoformat()}\n"
            )
            f.write("=" * 80 + "\n")
            f.write(f"Average Reward Per Step: {avg_reward:.6f}\n")
            f.write("\n")

        logger.info("Average reward per step for %s saved to %s", brain_key, path)
        return path


_log_manager = LoggerManager()


def get_logger(source: Any | None = None) -> logging.Logger:
    """Return the shared project logger or a child logger for a source."""
    return _log_manager.get_logger(source)


def set_report_artifact_path(path: str | Path) -> None:
    """Set the root directory where run artifacts are stored."""
    _log_manager.set_report_artifact_path(path)


def get_report_artifact_path() -> Path | None:
    """Return the configured artifact root directory, if any."""
    return _log_manager.get_report_artifact_path()


def build_report_artifact_path(filename: str) -> Path:
    """Build an artifact path under the configured report directory."""
    return _log_manager.build_report_artifact_path(filename)


def save_validated_dataset(
    brain: Any,
    dataset: dict[str, list[Any]],
    filename: str = "validated_dataset.json",
) -> Path:
    """Save a validated dataset artifact for a specific brain."""
    return _log_manager.save_validated_dataset(brain, dataset, filename)


def get_validated_dataset(brain: Any) -> dict[str, list[Any]] | None:
    """Load the validated dataset previously saved for a brain."""
    return _log_manager.get_validated_dataset(brain)


def save_evaluation_parameters(
    brain: Any,
    params: dict[str, Any],
    filename: str = "evaluation_parameters.json",
) -> Path:
    """Save evaluation parameter metadata for a specific brain."""
    return _log_manager.save_evaluation_parameters(brain, params, filename)


def get_last_evaluation_parameters(brain: Any) -> dict[str, Any] | None:
    """Return the last stored evaluation parameters for a brain."""
    return _log_manager.get_last_evaluation_parameters(brain)


def log_training_progress(
    brain: Any,
    step: int,
    total_steps: int,
    inputs: dict[str, Any],
) -> None:
    """Log formatted training progress details for a brain timestep."""
    _log_manager.log_training_progress(brain, step, total_steps, inputs)


def save_mean_squared_error(
    brain: Any,
    mse: float,
    filename: str = "mean_squared_error.txt",
    append: bool = True,
) -> Path:
    """Persist an MSE metric entry for a brain to an artifact file."""
    return _log_manager.save_mean_squared_error(brain, mse, filename, append)


def output_final_training_performance(
    brain: Any,
    training_steps: int,
    test_results: dict[str, Any],
) -> None:
    """Emit the final training performance summary for a brain to logs."""
    _log_manager.output_final_training_performance(brain, training_steps, test_results)


def save_final_training_performance(
    brain: Any,
    training_steps: int,
    test_results: dict[str, Any],
    filename: str = "final_training_performance.txt",
) -> Path:
    """Save the final training performance summary artifact for a brain."""
    return _log_manager.save_final_training_performance(
        brain,
        training_steps,
        test_results,
        filename,
    )


def save_agent_brain_shape(
    brain: Any,
    shape_data: dict[str, Any],
    filename: str = "agent_brain_shape.json",
) -> Path:
    """Save agent/brain shape metadata as a JSON artifact."""
    return _log_manager.save_agent_brain_shape(brain, shape_data, filename)


def save_prediction_report(
    brain: Any,
    report_text: str,
    filename: str = "latest_prediction_report.txt",
) -> Path:
    """Save the latest plain-text prediction report for a brain."""
    return _log_manager.save_prediction_report(brain, report_text, filename)


def get_latest_prediction_report(brain: Any) -> str | None:
    """Return the latest prediction report text for a brain."""
    return _log_manager.get_latest_prediction_report(brain)


def get_latest_prediction_report_path(brain: Any) -> Path | None:
    """Return the path to the latest saved prediction report for a brain."""
    return _log_manager.get_latest_prediction_report_path(brain)

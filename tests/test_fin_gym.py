"""Tests for FinGym tabular-data environment."""

# Test Suite: TS-24 (FinGym)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from htmrl.environment.fin_gym import FinGym


# Test Type: unit test
def test_fingym_builds_observation_from_dataframe() -> None:
    # TS-24 TC-205
    """Observations should be float32 vectors with one value per feature column."""

    frame = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "close": [10.2, 10.7, 10.9],
            "sector": ["tech", "tech", "finance"],
        }
    )

    env = FinGym(data_source=frame, target_column="close")
    obs, info = env.reset()

    assert obs.dtype == np.float32
    assert obs.shape == (2,)
    assert info["row_index"] == 0
    assert "close" not in info["feature_columns"]


# Test Type: unit test
def test_fingym_step_progression_and_reward() -> None:
    """Long/short rewards should track target deltas and terminate on final transition."""

    frame = pd.DataFrame(
        {
            "f1": [1.0, 1.0, 1.0],
            "price": [100.0, 101.5, 100.5],
        }
    )

    env = FinGym(data_source=frame, feature_columns=["f1"], target_column="price")
    env.reset()

    _obs1, reward1, terminated1, truncated1, info1 = env.step(1)  # long
    _obs2, reward2, terminated2, truncated2, info2 = env.step(2)  # short

    assert np.isclose(reward1, 1.5)
    assert np.isclose(reward2, 1.0)
    assert terminated1 is False
    assert terminated2 is True
    assert truncated1 is False
    assert truncated2 is False
    assert info1["row_index"] == 1
    assert info2["row_index"] == 1


# Test Type: unit test
def test_fingym_rejects_invalid_action() -> None:
    """Invalid actions should raise ValueError."""

    frame = pd.DataFrame({"a": [1.0, 2.0], "target": [1.0, 2.0]})
    env = FinGym(data_source=frame, target_column="target")
    env.reset()

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


# Test Type: unit test
def test_fingym_loads_csv(tmp_path: Path) -> None:
    """CSV data sources should load into a valid environment."""

    csv_path = tmp_path / "table.csv"
    pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [5.0, 6.0, 7.0],
        }
    ).to_csv(csv_path, index=False)

    env = FinGym(data_source=csv_path, target_column="y")
    obs, _info = env.reset()

    assert obs.shape == (1,)

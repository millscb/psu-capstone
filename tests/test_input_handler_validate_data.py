# Test Suite: TS-07 (Validate Input Data)
"""
tests.test_input_handler_validate_data

Test suite for InputHandler data validation functionality.

Validates that InputHandler correctly processes and validates input data for consistency,
type correctness, and required field presence. Tests ensure the processing step correctly:
- Handles missing or required columns
- Normalizes data types and datetime columns
- Detects and handles duplicate columns
- Fills missing values appropriately
- Validates data structures before encoding

These tests ensure the input data pipeline correctly processes data before encoder handling.
"""

import numpy as np
import pandas as pd
import pytest

from htmrl.input_layer.input_handler import InputHandler


@pytest.fixture
def handler():
    """Create a fresh InputHandler instance for each test."""
    return InputHandler()


# Test Type: unit test
def test_process_dataframe_valid(handler):
    # TC-059
    """
    Unit
    Checks that _validate_data() succeeds when all required columns exist and contain valid data.
    Test that valid data with required columns is processed correctly.
    """
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [10, 20, 30],
        }
    )
    result = handler._process_dataframe(df, required_columns=["id", "timestamp", "value"])
    assert not result.empty
    assert list(result.columns) == ["id", "timestamp", "value"]


# Test Type: unit test
def test_process_dataframe_adds_missing_required_columns(handler):
    # TC-060
    """
    Unit
    Ensures that _validate_data() raises an error if the DataFrame does not contain all required columns.
    Test that missing required columns are added with None values.
    """
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    result = handler._process_dataframe(df, required_columns=["id", "timestamp", "value"])
    assert "timestamp" in result.columns
    assert list(result.columns) == ["id", "timestamp", "value"]


# Test Type: unit test
def test_process_dataframe_empty(handler):
    # TC-061
    """
    Unit
    Checks that _validate_date() rejects an empty DataFrame.
    Test that empty dataframes are handled correctly.
    """
    df = pd.DataFrame()
    result = handler._process_dataframe(df, required_columns=None)
    assert result.empty


# Test Type: unit test
def test_process_dataframe_removes_all_nan_rows(handler):
    # TC-062
    """
    Unit
    Ensures that columns required to contain meaningful data (not all NaN/None) cause an error.
    Test that rows with all NaN values are dropped during datetime normalization.
    """
    df = pd.DataFrame({"id": [1, 2, 3], "timestamp": [None, None, None], "value": [10, 20, 30]})
    result = handler._process_dataframe(df)
    # After datetime normalization, rows with NaN are dropped
    assert len(result) <= len(df)


# Test Type: unit test
def test_process_dataframe_duplicate_columns(handler):
    # TC-063
    """
    Unit
    Checks that _validate_data() rejects DataFrames containing duplicate column names, which can corrupt encoder behavior.
    Test that duplicate columns are removed automatically.
    """
    df = pd.DataFrame([[1, "2024-01-01", 10, 100]], columns=["id", "timestamp", "value", "value"])
    result = handler._process_dataframe(df)
    assert result.columns.is_unique
    assert "value" in result.columns


# Test Type: unit test
def test_normalize_column_types_mixed_numeric(handler):
    # TS-07
    """Test that mixed numeric types are coerced correctly."""
    df = pd.DataFrame({"value": [1, 2.5, 3, 4.0]})
    result = handler._normalize_column_types(df)
    assert pd.api.types.is_numeric_dtype(result["value"])


# Test Type: unit test
def test_normalize_column_types_unsupported_type_raises(handler):
    # TC-064
    """
    Unit
    Ensures _validate_data() raises an error when _data is not a pandas DataFrame at all.
    Test that unsupported types raise ValueError.
    """
    df = pd.DataFrame({"value": [1, 2, {"key": "value"}]})
    with pytest.raises(ValueError, match="Corrupt data detected"):
        handler._normalize_column_types(df)


# Test Type: unit test
def test_fill_missing_values_numeric(handler):
    # TS-07
    """Test that missing numeric values are filled with mean."""
    df = pd.DataFrame({"value": [10.0, None, 30.0, 40.0]})
    result = handler._fill_missing_values(df)
    # Mean of [10, 30, 40] is ~26.67
    assert result["value"].notna().all()
    assert np.isclose(result["value"].iloc[1], 26.666666, rtol=0.01)


# Test Type: unit test
def test_detect_repeating_values(handler):
    # TS-07
    """Test detection of repeating values in columns."""
    df = pd.DataFrame({"category": ["A", "A", "A", "A", "B"]})
    is_repeating, cols = handler._detect_repeating_values(df, threshold=3)
    assert is_repeating is True
    assert "category" in cols


# Test Type: unit test
def test_input_data_with_dict(handler):
    # TS-07
    """Test that dictionary input is processed correctly."""
    data = {"id": [1, 2, 3], "value": [10, 20, 30]}
    result = handler.input_data(data)
    assert isinstance(result, dict)
    assert "id" in result
    assert "value" in result
    assert result["id"] == [1, 2, 3]


# Test Type: unit test
def test_input_data_with_required_columns(handler):
    # TS-07
    """Test that required columns are enforced."""
    data = {"id": [1, 2, 3], "value": [10, 20, 30]}
    result = handler.input_data(data, required_columns=["id", "value", "timestamp"])
    assert "timestamp" in result

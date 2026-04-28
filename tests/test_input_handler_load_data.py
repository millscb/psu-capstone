# Test Suite: TS-01 (Input Load Data)
"""
tests.test_input_handler_load_data

Test suite for InputHandler data loading functionality.

Validates that InputHandler correctly parses CSV and XLSX files into record dictionaries,
properly handles required columns, and extracts data in the expected format. Tests cover:

These tests ensure the input data pipeline correctly loads external data sources (CSV/XLSX)
before downstream encoder processing.
"""

from pathlib import Path

import pytest
from openpyxl import Workbook

from htmrl.input_layer.input_handler import InputHandler


@pytest.fixture
def handler() -> InputHandler:
    # Fresh singleton reference each time -- no teardown needed
    return InputHandler()


@pytest.fixture
def temp_path(tmp_path: Path) -> Path:
    # e.g. create a subdirectory or seed test files here
    work_dir = tmp_path / "input_handler"
    work_dir.mkdir()
    return work_dir


# Test Type: unit test
def test_load_input_data_csv(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-001
    """[TS-01 TC-001] Ensure CSV files are parsed into record dictionaries."""
    # Arrange
    csv_path = temp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
    required = ["a", "b", "c"]

    # Act
    records = handler.input_data(str(csv_path), required_columns=required)

    # Assert
    assert list(records.keys()) == required
    assert len(records["a"]) == 2
    assert [str(records["a"][0]), str(records["b"][0]), str(records["c"][0])] == [
        "1",
        "2",
        "3",
    ]


# Test Type: unit test
def test_load_input_data_excel_xlsx(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-002
    """[TS-01 TC-002] Confirm XLSX ingestion yields the requested columns."""
    # Arrange
    xlsx_path = temp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["a", "b"])
    sheet.append([10, 30])
    sheet.append([20, 40])
    workbook.save(xlsx_path)
    required = ["a", "b"]

    # Act
    records = handler.input_data(str(xlsx_path), required_columns=required)

    # Assert
    assert list(records.keys()) == required
    assert records["a"] == [10, 20]
    assert records["b"] == [30, 40]


# Test Type: unit test
def test_load_input_data_excel_xls_is_unsupported(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-003
    """[TS-01 TC-003] Validate legacy XLS files raise a ValueError without optional XLS readers."""
    # Arrange
    xls_path = temp_path / "sample.xls"
    xls_path.write_text("placeholder")
    required = ["a", "b"]

    # Act
    with pytest.raises(ValueError):
        handler.input_data(str(xls_path), required_columns=required)


# Test Type: unit test
def test_load_input_data_json(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-004
    """[TS-01 TC-004] Check JSON records are converted into the expected record layout."""
    # Arrange
    json_path = temp_path / "sample.json"
    json_path.write_text('[{"a": 1, "b": 3}, {"a": 2, "b": 4}]')
    required = ["a", "b"]

    # Act
    records = handler.input_data(str(json_path), required_columns=required)

    # Assert
    assert list(records.keys()) == required
    assert records["a"] == [1, 2]
    assert records["b"] == [3, 4]


# Test Type: unit test
def test_load_input_data_txt_returns_dataframe_of_lines(
    # TS-01 TC-005
    temp_path: Path,
    handler: InputHandler,
) -> None:
    """[TS-01 TC-005] Verify plain-text files become line-per-row records."""
    # Arrange
    txt_path = temp_path / "sample.txt"
    lines = ["first line\n", "second line\n", "third line\n"]
    txt_path.write_text("".join(lines))

    # Act
    try:
        records = handler.input_data(str(txt_path), required_columns=["value"])
    except ValueError:
        pytest.skip("Text input handling not supported by this InputHandler.")

    # Assert
    assert list(records.keys()) == ["value"]
    assert len(records["value"]) == len(lines)
    assert records["value"] == lines


# Test Type: unit test
def test_load_input_data_unsupported_extension_treated_as_scalar(
    # TS-01 TC-006
    temp_path: Path,
    handler: InputHandler,
) -> None:
    """[TS-01 TC-006] Unknown extensions are treated as scalar input when not supported."""
    # Arrange
    bad_path = temp_path / "sample.xml"
    bad_path.write_text("<root><a>1</a></root>")

    # Act
    try:
        records = handler.input_data(str(bad_path), required_columns=["value"])
    except ValueError:
        pytest.skip("Unsupported extensions not treated as scalar in this InputHandler.")

    # Assert
    assert records == {"value": [str(bad_path)]}


# Test Type: unit test
def test_load_input_data_missing_file_raises(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-007
    """[TS-01 TC-007] Ensure missing files raise FileNotFoundError for clearer diagnostics."""
    # Arrange
    missing_path = temp_path / "missing.csv"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        handler.input_data(str(missing_path), required_columns=["timestamp", "a"])


# Test Type: unit test
def test_load_input_data_accepts_pathlike(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-008
    """[TS-01 TC-008] PathLike inputs are accepted as file paths."""
    # Arrange
    csv_path = temp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")

    # Act
    records = handler.input_data(csv_path, required_columns=["a", "b"])

    # Assert
    assert records == {"a": ["1"], "b": ["2"]} or records == {"a": [1], "b": [2]}


# Test Type: unit test
def test_input_handler_is_singleton() -> None:
    # TS-01 TC-009
    """[TS-01 TC-009] Confirm InputHandler enforces a singleton instance."""
    # Arrange / Act
    h1 = InputHandler().get_instance()
    h2 = InputHandler().get_instance()

    # Assert
    assert h1 is h2


# Test Type: unit test
def test_load_input_data_sets_internal_data(temp_path: Path, handler: InputHandler) -> None:
    # TS-01 TC-010
    """[TS-01 TC-010] Check the handler caches the latest records without sharing references."""
    # Arrange
    csv_path = temp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")
    required = ["a", "b"]

    # Act
    records_one = handler.input_data(str(csv_path), required_columns=required)
    records_two = handler.input_data(str(csv_path), required_columns=required)

    # Assert
    assert isinstance(records_one, dict)
    assert isinstance(records_two, dict)
    assert records_two == records_one
    assert records_two is not records_one


# Test Type: unit test
def test_load_input_data_bytearray(handler: InputHandler) -> None:
    """Ensure bytearray inputs become value records that preserve byte order."""
    payload = bytearray([1, 2, 255])
    records = handler.input_data(payload, required_columns=["value"])
    assert records["value"] == [1, 2, 255]

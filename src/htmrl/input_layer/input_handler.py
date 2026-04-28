"""Simplified pandas-backed input handler.

The handler accepts Any input, detects file-path inputs first, loads/normalizes data
with pandas, validates core constraints, and exposes data as dict[Any, list[Any]].
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from htmrl.log import get_logger, logger


class InputHandler:
    """Load, clean, validate, and normalize raw input payloads for the encoding pipeline.

    Args:
        data: Optional in-memory data to initialize with. If provided, will be
            processed through the input_data method for validation and setup.
    """

    __instance: ClassVar[InputHandler]
    """Singleton instance of the InputHandler class."""

    _SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".csv",
        ".json",
        ".xlsx",
        ".xls",
        ".parquet",
        ".txt",
    }

    def __new__(cls, data: Any | None = None) -> InputHandler:
        if getattr(cls, "_InputHandler__instance", None) is None:
            cls.__instance = super(InputHandler, cls).__new__(cls)
        return cls.__instance

    def __init__(self, data: Any | None = None) -> None:
        self.logger = get_logger(self)

        self._data: dict[Any, list[Any]] = {}
        self._columns: list[str] = []
        self._repeating_columns: list[str] = []
        self._is_repeating: bool = False
        self._contains_multidimensional_data: bool = False
        self._fields: list[Any] = []

        if data is not None:
            self.input_data(data)

    @classmethod
    def get_instance(cls) -> InputHandler:
        """Return the singleton InputHandler instance, creating it if needed."""
        if getattr(cls, "_InputHandler__instance", None) is None:
            cls.__instance = InputHandler()
        return cls.__instance

    @property
    def data(self) -> dict[Any, list[Any]]:
        return self._data

    @property
    def fields(self) -> list[Any]:
        """Return input fields associated with this handler.

        InputHandler currently normalizes tabular payloads directly and does not
        build HTM Field objects here, so this defaults to an empty list.
        """
        return self._fields

    @data.setter
    def data(self, data: dict[Any, list[Any]]) -> None:

        if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
            raise ValueError("Data must be a dictionary of lists representing columns.")

        self._data = data

    def input_data(
        self, input_source: Any, required_columns: list[str] | None = None
    ) -> dict[Any, list[Any]]:
        """Ingest data from files or in-memory payloads and return normalized column lists.

        Args:
            input_source: The raw input data, which can be a file path or an in-memory data structure.
            required_columns: Optional list of column names that must be present in the output records.

        Returns:
            A dictionary of lists representing the normalized columns extracted from the input data.

            Example:

            data = {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "city": ["New York", "Chicago", "San Francisco"]
            }

        """
        required = required_columns if required_columns else None

        # check if input_source is a file path before attempting to coerce it to a DataFrame
        if self._is_file_path(input_source):
            df = self._load_file_to_dataframe(input_source)
        else:
            df = self._coerce_non_file_to_dataframe(input_source)

        # Process the DataFrame to handle required columns, normalize types, and detect temporal patterns
        df = self._process_dataframe(df, required)

        self._columns = list(df.columns)

        column_listing = "\n    ".join(self._columns)
        self.logger.info(
            "Validating data:\n  rows=%d\n  columns=%d\n    %s",
            len(df),
            len(self._columns),
            column_listing,
        )

        data = df.to_dict(orient="list")

        self._data = data

        return self._data

    def to_numpy(self, data: list[dict[Any, Any]]) -> np.ndarray:
        """Convert a list of records into a numeric ``numpy.ndarray``."""
        if not data:
            raise ValueError("Cannot convert empty record list to numpy array.")
        df = pd.DataFrame(data)
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        if numeric_df.isna().all(axis=None):
            raise ValueError("Data validation failed; cannot convert to numpy ndarray.")
        return numeric_df.to_numpy(dtype=np.float64)

    def get_column_data(self, column: str | None = None) -> list[Any]:
        """Return the from internal data a list of Any for the specified column."""

        if not self._data:
            raise ValueError("No records available to build encoder sequence.")

        dataframe = pd.DataFrame(self._data)

        if column is None and len(dataframe.columns) == 1:
            column = str(dataframe.columns[0])
        elif column is None and len(dataframe.columns) > 1:
            raise ValueError(
                "Column must be specified when multiple columns are present in the input."
            )

        if column not in dataframe.columns:
            raise ValueError(f"Requested column '{column}' not found in normalized data.")

        sequence = dataframe[column].tolist()
        filtered = [value for value in sequence if value is not None]
        if not filtered:
            raise ValueError("Encoder sequence contains no valid values after filtering.")

        return filtered

    def _is_file_path(self, input_source: Any) -> bool:
        """Determine if the input source is a valid file path with a supported extension."""
        if isinstance(input_source, (os.PathLike, str)):
            path = Path(input_source)
            suffix = path.suffix.lower()
            if suffix in self._SUPPORTED_EXTENSIONS:
                self.logger.info("Detected file path input: %s", path)
                return True
            self.logger.warning("File extension '%s' is not supported for input: %s", suffix, path)
        return False

    def _load_file_to_dataframe(self, input_source: str | os.PathLike[str]) -> pd.DataFrame:
        """Load data from a file path into a pandas DataFrame based on the file extension.

        Args:
            input_source: A string or os.PathLike object representing the file path to load.

        Returns:
            A pandas DataFrame containing the data loaded from the specified file.

        Raises:
            ValueError: If the file extension is not supported for loading.
            FileNotFoundError: If the specified file does not exist.

        """

        path = Path(input_source)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        self.logger.info("Loading data from %s", path)

        if ext == ".csv":
            # Try to read CSV and detect metadata rows
            df = pd.read_csv(path)

            # Check if first row contains type information (metadata)
            if len(df) > 0 and all(
                isinstance(val, str) and val in ["datetime", "float", "int", "str", "T", ""]
                for val in df.iloc[0].values
                if pd.notna(val)
            ):
                # Skip metadata rows and re-read
                self.logger.info("Detected metadata rows in CSV, skipping them")
                df = pd.read_csv(path, skiprows=[1, 2])

            return df
        if ext == ".json":
            return pd.read_json(path)
        if ext == ".xlsx":
            return pd.read_excel(
                path
            )  # use openpyxl engine for XLSX files, which is the default in
            # recent pandas versions and supports modern Excel formats. For legacy XLS files, xlrd may be required but is not guaranteed to be available in all environments.
        if ext == ".xls":
            return pd.read_excel(
                path
            )  # use xlrd engine for legacy XLS files if available, otherwise fallback
            # to openpyxl which may not support XLS
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".txt":
            return pd.DataFrame(
                {"value": path.read_text(encoding="utf-8").splitlines(keepends=True)}
            )  # build inline DataFrame for text files, treating each line as a separate record
        raise ValueError(f"Unsupported file type: {ext}")

    def _coerce_non_file_to_dataframe(self, input_source: Any) -> pd.DataFrame:
        """Convert in-memory data structures into a pandas DataFrame for normalization and validation.

        Args:
            input_source: The raw input data, which can be a variety of in-memory types

        Returns:
            A pandas DataFrame representing the input data, ready for processing and validation.

        Raises:
            TypeError: If the input data type is not supported for conversion to a DataFrame.

        """

        if isinstance(input_source, bytes):
            input_source = bytearray(input_source)

        if isinstance(input_source, pd.DataFrame):
            return input_source.copy()

        elif isinstance(input_source, pd.Series):
            return input_source.to_frame()

        elif isinstance(input_source, np.ndarray):
            self._contains_multidimensional_data = input_source.ndim > 1
            return pd.DataFrame(input_source)

        elif isinstance(input_source, Mapping):
            return pd.DataFrame(input_source)

        elif isinstance(input_source, bytearray):
            if b"," in input_source or b"\n" in input_source:
                try:
                    text_payload = input_source.decode("utf-8")
                except UnicodeDecodeError:
                    text_payload = input_source.decode("latin-1")
                if all(ch.isprintable() or ch.isspace() for ch in text_payload):
                    try:
                        csv_df = pd.read_csv(StringIO(text_payload), dtype=str)
                        if not csv_df.empty and csv_df.columns.size > 0:
                            return csv_df
                    except Exception:
                        pass
            return pd.DataFrame({"value": list(input_source)})

        elif isinstance(input_source, Sequence) and not isinstance(
            input_source, (str, bytes, bytearray)
        ):
            # Handle sequences of mappings (e.g., list of dicts) and sequences of sequences (e.g., list of lists)
            if not input_source:
                return pd.DataFrame()
            first = input_source[0]
            if isinstance(first, Mapping):
                return pd.DataFrame(list(input_source))
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
                self._contains_multidimensional_data = True
                return pd.DataFrame([list(row) for row in input_source])
            return pd.DataFrame({"value": list(input_source)})

        elif isinstance(input_source, (str, int, float)):
            return pd.DataFrame({"value": [input_source]})

        else:

            raise TypeError(f"Unsupported data type for conversion: {type(input_source)}")

    def _process_dataframe(
        self, df: pd.DataFrame, required_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Normalize datetime columns, enforce required columns, detect repeating temporal patterns, and handle missing values.

        Args:
            df: The input DataFrame to process and normalize.
            required_columns: A list of column names that must be present in the output DataFrame.

        Returns:
            A processed and normalized DataFrame that meets the specified requirements.
        """
        if df.empty:
            self._repeating_columns = []
            return df

        if not df.columns.is_unique:
            duplicates = df.columns[df.columns.duplicated()].tolist()
            logger.info("Duplicate columns detected: %s", duplicates)

            df = df.loc[:, ~df.columns.duplicated()]
            logger.info("Removed duplicate columns, remaining columns: %s", df.columns.tolist())

        if required_columns is not None:
            missing = [col for col in required_columns if col not in df.columns]
            for col in missing:
                df[col] = None
            df = df[required_columns]
            logger.info("Added missing required columns: %s", missing)

        # check for existing datetime
        df = self._normalize_datetime_columns(df)

        # check for consistent primitive types and coerce numeric types for downstream processing
        df = self._normalize_column_types(df)

        # fill missing values in numeric columns with the mean of the column
        df = self._fill_missing_values(df)

        self._isRepeating, self._repeating_columns = self._detect_repeating_values(df, threshold=3)

        return df

    def _normalize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and normalize datetime columns while preserving datetimelike objects."""

        dataframe = df.copy()

        for col in dataframe.columns:
            series = dataframe[col]
            normal = series.dropna()
            if normal.empty:
                continue

            sample = normal.iloc[0]
            parsed: pd.Series | None = None

            if isinstance(sample, (datetime, pd.Timestamp, time.struct_time, np.datetime64)):

                continue  # already a datetimelike type, no parsing needed

            elif isinstance(sample, str):
                maybe_time = any(token in sample for token in ("-", ":", "T", "/", " "))
                if maybe_time:
                    parsed = pd.to_datetime(series)

            if parsed is not None and parsed.notna().sum() == normal.shape[0]:
                dataframe[col] = parsed

        dataframe = dataframe.dropna()

        return dataframe

    def _normalize_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce consistent primitive types across columns and coerce numeric types for downstream processing.

        Args:
            df: The input DataFrame to analyze and normalize column types.

        Returns:
            A DataFrame with consistent primitive types across columns, where numeric columns are coerced to appropriate numeric types for downstream processing, while non-numeric columns are preserved as-is.

        Raises:
            ValueError: If the DataFrame contains unsupported types or mixed primitive types within a column that cannot be normalized.


        """

        dataframe = df.copy()

        for col in dataframe.columns:

            normal = dataframe[col].dropna()

            logger.info(
                "Analyzing column %-24s dtype=%-12s non-null=%5d",
                col,
                str(normal.dtype),
                normal.shape[0],
            )

            # store a set of types present in the column to detect unsupported or mixed types
            primitive_types = {type(value) for value in normal}

            # check for any non-primitive types that would indicate corrupt or unsupported data
            # against the inline set of allowed primitive types (str, bool, int, float, and their numpy equivalents)
            has_non_primitive = any(
                t
                not in {
                    str,
                    bool,
                    int,
                    float,
                    datetime,
                    pd.Timestamp,
                    np.int64,
                    np.float64,
                    np.int32,
                    np.float32,
                    np.datetime64,
                    np.ndarray,
                    tuple,
                    list,
                }
                for t in primitive_types
            )

            if normal.empty:  # go to next column if all values are None/NaN
                continue

            elif has_non_primitive:  # look for unsupported types in a mapping {key: type(value)}
                logger.error(
                    "Unsupported types detected in column '%s': %s",
                    col,
                    sorted(t.__name__ for t in primitive_types),
                )
                raise ValueError(f"Corrupt data detected in column '{col}': unsupported value type")

            # enforce consistent primitive types across the columns, all values are the same primitive type
            # check for more than one type in the col
            elif len(primitive_types) > 1:
                logger.warning(
                    "Column '%s' has multiple types: %s",
                    col,
                    sorted(t.__name__ for t in primitive_types),
                )

                # attempt to convert to numeric
                if primitive_types.issubset(
                    {int, float, np.int64, np.float64, np.int32, np.float32, np.ndarray}
                ):
                    dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
                    dataframe[col] = dataframe[col].fillna(dataframe[col].mean(skipna=True))

                    primitive_types = dataframe[col].dropna().map(type).unique().tolist()
                    primitive_types = set(primitive_types)

                    if len(primitive_types) > 1:
                        raise ValueError(
                            f"Mixed primitive types detected in column '{col}': {sorted(t.__name__ for t in primitive_types)}"
                        )

        return dataframe

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in numeric columns with the mean of the column to ensure downstream processing can handle
          numeric data without NaN values.

        Args:
            df: The input DataFrame to analyze and fill missing values in numeric columns.

        Returns:
            A DataFrame with missing values in numeric columns filled with the mean of the respective column,
            while non-numeric columns are left unchanged.
        """
        dataframe = df.copy()

        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            mean_value = dataframe[col].mean(skipna=True)
            if pd.notna(mean_value):
                dataframe[col] = dataframe[col].fillna(mean_value)
        return dataframe

    def _detect_repeating_values(self, df: pd.DataFrame, threshold: int) -> tuple[bool, list[str]]:
        """Identify columns that contain repeating temporal values, which may indicate time series data with regular intervals.
        Args:
            df: The input DataFrame to analyze for repeating temporal patterns.
            threshold: An integer threshold for determining when a column has enough repeating values to be considered temporal

        Returns:
            A tuple containing a boolean indicating whether any repeating temporal columns were detected, and a list of

            column names that exhibit repeating temporal patterns based on the specified threshold.


        """
        # detect repeating values for any mapping key value
        dataframe = df.copy()
        repeating_col: list[str] = []

        for col in dataframe.columns:
            value_counts = dataframe[col].value_counts()
            if (
                not value_counts.empty and value_counts.iloc[0] > threshold
            ):  # arbitrary threshold for "repeating"
                repeating_col.append(col)

        return (bool(repeating_col), repeating_col)

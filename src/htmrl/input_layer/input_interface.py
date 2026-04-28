"""Protocol definitions for input layer components.

This module defines the interface contract that all input handlers must
satisfy for data loading, validation, and normalization.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from htmrl.agent_layer.HTM import Field


@runtime_checkable
class InputInterface(Protocol):
    """Protocol defining the interface for input data handlers.

    Input handlers implementing this protocol load data from various sources
    (files, DataFrames, dictionaries) and normalize it into a standardized
    format for downstream encoding.
    """

    @property
    def data(self) -> dict[Any, list[Any]]:
        """Get the normalized column data currently held by the handler.

        Returns:
            Dictionary mapping column names to lists of values.
        """
        ...

    @property
    def fields(self) -> list[Field]:
        """Return the primary input fields used for processing."""
        ...

    def input_data(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
    ) -> dict[Any, list[Any]]:
        """Load and normalize data from various sources.

        Accepts file paths, DataFrames, dictionaries, or other structures
        and converts them to a standardized column-based format.

        Args:
            input_source: Data source (file path, DataFrame, dict, etc.).
            required_columns: Optional list of columns that must be present.

        Returns:
            Dictionary mapping column names to lists of values.
        """
        ...

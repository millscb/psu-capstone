"""Encoder Handler to build composite SDRs

This module provides the EncoderHandler class, which manages multiple encoder types
and dynamically selects the appropriate encoder for each column in a pandas DataFrame
based on its dtype. It builds composite Sparse Distributed Representations (SDRs)
from the encoded columns.

"""

from __future__ import annotations

import copy
from datetime import datetime

import numpy as np
import pandas as pd

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from legacy.sdr_layer.sdr import SDR


class EncoderHandler:
    """Handles multiple encoders to create composite SDRs.

    This class uses a singleton pattern to ensure only one instance exists.
    It dynamically selects the appropriate encoder for each DataFrame column
    based on its dtype and builds a composite SDR from the encoded columns.

    Args:
        input_data: Optional DataFrame containing input data for encoder initialization.
    """

    __instance: EncoderHandler | None = None

    def __new__(cls, input_data: pd.DataFrame | None = None) -> "EncoderHandler":
        if cls.__instance is None:
            cls.__instance = super(EncoderHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self, input_data: pd.DataFrame | None = None):
        self._data_frame = copy.deepcopy(input_data) if input_data is not None else pd.DataFrame()
        self._encoders: list[BaseEncoder] = []

    @classmethod
    def get_instance(cls) -> "EncoderHandler":
        """Returns the singleton instance of EncoderHandler.

        Returns:
            The singleton instance.
        """
        if cls.__instance is None:
            cls.__instance = EncoderHandler()
        return cls.__instance

    def build_composite_sdr(self, input_data: pd.DataFrame) -> list[SDR]:
        """Builds a composite SDR from multiple encoders based on the input data.

        For each column in the input DataFrame, selects an encoder based on the column's dtype,
        encodes the value, and concatenates the resulting SDRs into a single composite SDR.

        Args:
            input_data: DataFrame containing input values for each encoder.

        Returns:
            Composite SDRs built from all encoded columns.

        Raises:
            TypeError: If a column's value type is unsupported.
            ValueError: If no SDRs are created or an unexpected error occurs.
        """
        self._encoders = []

        row_sdrs: list[SDR] = []

        scalartrue = False

        for _, row in input_data.iterrows():
            sdrs: list[SDR] = []

            for col_name, value in row.items():
                if isinstance(value, float) or isinstance(value, np.floating):
                    encoder = RandomDistributedScalarEncoder(
                        RDSEParameters(
                            active_bits=40,
                            sparsity=0.0,
                            size=2048,
                            radius=0.0,
                            resolution=1.0,
                            category=False,
                            seed=1,
                        )
                    )
                    sdr = SDR([encoder.size])
                    encoder.encode(float(value), sdr)

                elif scalartrue or isinstance(value, int) or isinstance(value, np.integer):
                    encoder = ScalarEncoder(
                        ScalarEncoderParameters(
                            minimum=0,
                            maximum=100,
                            clip_input=True,
                            periodic=False,
                            active_bits=40,
                            sparsity=0.0,
                            size=2048,
                            radius=0.0,
                            category=False,
                            resolution=0.0,
                        )
                    )
                    sdr = SDR([encoder.size])
                    sdr = encoder.encode(int(value))

                elif isinstance(value, str):
                    category_list = input_data[col_name].unique().tolist()
                    encoder = CategoryEncoder(
                        CategoryParameters(
                            w=3,
                            category_list=category_list,
                        )
                    )
                    sdr = encoder.encode(value)
                    encoder.encode(value)

                elif isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                    encoder = DateEncoder(DateEncoderParameters())
                    sdr = SDR([encoder.size])
                    sdr = encoder.encode(value)

                else:
                    raise TypeError(f"Unsupported value type for encoder: {type(value)}")

                sdrs.append(copy.deepcopy(sdr))

            if not sdrs:
                raise ValueError("No SDRs were created from the input data.")

            if len(sdrs) >= 2:
                flat_sdrs = []
                for s in sdrs:
                    if len(s.dimensions) != 1:
                        flat = SDR([s.size])
                        flat.set_sparse(s.get_sparse())
                        flat_sdrs.append(flat)
                    else:
                        flat_sdrs.append(s)

                total_size = sum(s.size for s in flat_sdrs)
                row_union = SDR([total_size])
                row_union.concatenate(flat_sdrs, axis=0)
            elif len(sdrs) == 1:
                row_union = copy.deepcopy(sdrs[0])
            else:
                raise ValueError("Unexpected error in building composite SDR.")

            row_sdrs.append(row_union)

        if not row_sdrs:
            raise ValueError("No SDRs were created from the input data.")

        # Always return the list of SDRs, one per row
        return row_sdrs


if __name__ == "__main__":
    """Smoke test for EncoderHandler.

    Creates a sample DataFrame with various column types, initializes the EncoderHandler,
    builds a composite SDR, and prints its sparse representation and size.
    """

    df = pd.DataFrame(
        [
            {
                "float_col": float(3.14),
                "int_col": int(42),
                "str_col": str("B"),
                "date_col": datetime(2023, 12, 25),
            },
            {
                "float_col": float(2.71),
                "int_col": int(7),
                "str_col": str("A"),
                "date_col": datetime(2024, 1, 1),
            },
        ]
    )

    handler = EncoderHandler(df)
    composite_sdr = handler.build_composite_sdr(df)
    for idx, sdr in enumerate(composite_sdr):
        print(f"Composite SDR {idx} sparse representation:", sdr.get_sparse())
        print(f"Composite SDR {idx} size:", sdr.size)

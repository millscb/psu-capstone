from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

import numpy as np
from htmrl.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.sdr_layer.sdr import SDR


class BatchEncoderHandler:
    """Backwards-compatible batch encoder handler.

    Provides the API expected by legacy tests and callers:
    - singleton construction
    - parameter setter methods
    - `_build_dict_list_sdr` returning per-column SDR lists
    """

    __instance: BatchEncoderHandler | None = None

    def __new__(cls, input_data: list[dict[str, Any]] | None = None) -> "BatchEncoderHandler":
        if cls.__instance is None:
            cls.__instance = super(BatchEncoderHandler, cls).__new__(cls)
        return cls.__instance

    def __init__(self, input_data: list[dict[str, Any]] | None = None) -> None:
        self._data_frame: list[dict[str, Any]] = copy.deepcopy(input_data) if input_data else []
        self._rdse_params = RDSEParameters(
            size=2048,
            active_bits=40,
            sparsity=0.0,
            radius=1.0,
            resolution=0.0,
            category=False,
            seed=1,
        )
        self._scalar_params = ScalarEncoderParameters(
            minimum=0,
            maximum=100,
            clip_input=True,
            periodic=False,
            active_bits=40,
            sparsity=0.0,
            size=2048,
            radius=0.0,
            category=False,
            resolution=1.0,
        )
        self._date_params = DateEncoderParameters(
            season_active_bits=0,
            day_of_week_active_bits=7,
            weekend_active_bits=2,
            holiday_active_bits=4,
            holiday_dates=[[12, 25], [1, 1], [7, 4], [11, 11]],
            time_of_day_active_bits=24,
            custom_active_bits=0,
            custom_days=[],
            rdse_used=False,
        )
        self._category_params = CategoryParameters(w=3, category_list=[])
        self._custom_encoding: dict[str, str] = {}

    def _encode_value(
        self,
        value: Any,
        col_name: str,
        input_data: list[dict[str, Any]],
    ) -> SDR:
        encoder_type = self._custom_encoding.get(col_name, "").lower()
        if not encoder_type:
            if isinstance(value, datetime):
                encoder_type = "date"
            elif isinstance(value, str):
                encoder_type = "category"
            elif isinstance(value, int) and not isinstance(value, bool):
                encoder_type = "scalar"
            elif isinstance(value, (float, np.floating)):
                encoder_type = "rdse"
            else:
                encoder_type = "rdse"

        if encoder_type == "scalar":
            encoder = ScalarEncoder(self._scalar_params)
            if not isinstance(value, (int, float, np.integer, np.floating, bool)):
                value = 0
            dense = encoder.encode(int(value))
        elif encoder_type == "category":
            categories = sorted(
                {str(row.get(col_name)) for row in input_data if row.get(col_name) is not None}
            )
            params = self._category_params
            params.category_list = categories
            encoder = CategoryEncoder(params)
            dense = encoder.encode(str(value))
        elif encoder_type == "date":
            encoder = DateEncoder(self._date_params)
            dense = encoder.encode(value)
        else:
            encoder = RandomDistributedScalarEncoder(self._rdse_params)
            if not isinstance(value, (int, float, np.integer, np.floating, bool)):
                value = 0.0
            dense = encoder.encode(float(value))

        sdr = SDR([encoder.size])
        sdr.set_dense(dense)
        return sdr

    def _build_dict_list_sdr(
        self, input_data: list[dict[str, Any]], threads_per_column: int
    ) -> dict[str, list[SDR]]:
        del threads_per_column
        if not input_data:
            return {}

        column_names = list(input_data[0].keys())
        output: dict[str, list[SDR]] = {name: [] for name in column_names}

        for row in input_data:
            for col_name in column_names:
                output[col_name].append(self._encode_value(row.get(col_name), col_name, input_data))

        return output

    def set_category_encoder_parameters(self, params: CategoryParameters) -> None:
        """Set custom parameters for category encoding.

        Args:
            params: CategoryParameters instance with encoder configuration.
        """
        self._category_params = params

    def set_rdse_encoder_parameters(self, params: RDSEParameters) -> None:
        """Set custom parameters for RDSE (float) encoding.

        Args:
            params: RDSEParameters instance with encoder configuration.
        """
        self._rdse_params = params

    def set_scalar_encoder_parameters(self, params: ScalarEncoderParameters) -> None:
        """Set custom parameters for scalar (integer) encoding.

        Args:
            params: ScalarEncoderParameters instance with encoder configuration.
        """
        self._scalar_params = params

    def set_date_encoder_parameters(self, params: DateEncoderParameters) -> None:
        """Set custom parameters for date/datetime encoding.

        Args:
            params: DateEncoderParameters instance with encoder configuration.
        """
        self._date_params = params

    def choose_custom_column_encoding(self, custom_encoding: dict[str, str]) -> None:
        """Override automatic type detection with custom encoder choices per column.

        Args:
            custom_encoding: Dict mapping column names to encoder types
                ("rdse", "scalar", "category", "date").
        """
        self._custom_encoding = custom_encoding


# from __future__ import annotations

# import copy
# from datetime import datetime
# from multiprocessing import Manager, Process
# from threading import Thread
# from typing import cast

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsRegressor

# from htmrl.encoder_layer.base_encoder import BaseEncoder
# from htmrl.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
# from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
# from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
# from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
# from legacy.sdr_layer.sdr import SDR


# class RdseThread(Thread):
#     """
#     Custom thread class to run RDSE encoding.
#     This overrides the run to enable encoding as the thread work.
#     The offset is dynamically chosen based on the number of threads
#     in the main batch_encoder_handler class.
#     """

#     def __init__(
#         self, column_data: pd.Series, params: RDSEParameters, output: list[SDR], row_offset: int
#     ):
#         """
#         Initializes the RdseThread with a Series of input data.
#         Args:
#             column_data (pd.Series): The input data column to encode.
#             params (RDSEParameters): Parameters for RDSE encoding.
#             output (list[SDR]): A list to store the encoded SDRs.
#             row_offset (int): Offset to place encoded SDRs in the output list.
#         """
#         super().__init__()
#         self._column_data = column_data
#         self._encoder = RandomDistributedScalarEncoder(params)
#         self._output = output
#         self._row_offset = row_offset

#     def run(self):
#         """
#         Encodes the data column using RDSE and stores results in the output list.
#         """
#         for i in range(len(self._column_data)):
#             value = self._column_data.iloc[i]

#             # fresh sdr instance per row
#             sdr_instance = SDR([self._encoder.size])
#             self._encoder.encode(value, sdr_instance)

#             self._output[self._row_offset + i] = sdr_instance

#             """
#             #debug prints if you want
#             active_bits = sdr_instance.get_sum()
#             sparsity_fraction = sdr_instance.get_sparsity()
#             print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
#             """


# class ScalarThread(Thread):
#     """
#     Custom thread class to run Scalar encoding.
#     This overrides the run to enable encoding as the thread work.
#     The offset is dynamically chosen based on the number of threads
#     in the main batch_encoder_handler class.
#     """

#     def __init__(
#         self,
#         column_data: pd.Series,
#         params: ScalarEncoderParameters,
#         output: list[SDR],
#         row_offset: int,
#     ):
#         """
#         Initializes the ScalarThread with a Series of input data.
#         Args:
#             column_data (pd.Series): The input data column to encode.
#             params (ScalarEncoderParameters): Parameters for RDSE encoding.
#             output (list[SDR]): A list to store the encoded SDRs.
#             row_offset (int): Offset to place encoded SDRs in the output list.
#         """
#         super().__init__()
#         self._column_data = column_data
#         self._encoder = ScalarEncoder(params)
#         self._output = output
#         self._row_offset = row_offset

#     def run(self):
#         """
#         Encodes the data column using scalar encoder and stores results in the output list.
#         """
#         for i in range(len(self._column_data)):
#             value = self._column_data.iloc[i]

#             # fresh sdr instance per row
#             sdr_instance = SDR([self._encoder.size])
#             self._encoder.encode(value, sdr_instance)
#             self._output[self._row_offset + i] = sdr_instance

#             """
#             #debug prints if you want
#             active_bits = sdr_instance.get_sum()
#             sparsity_fraction = sdr_instance.get_sparsity()
#             print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
#             """


# class DateThread(Thread):
#     """
#     Custom thread class to run Date encoding.
#     This overrides the run to enable encoding as the thread work.
#     The offset is dynamically chosen based on the number of threads
#     in the main batch_encoder_handler class.
#     """

#     def __init__(
#         self,
#         column_data: pd.Series,
#         params: DateEncoderParameters,
#         output: list[SDR],
#         row_offset: int,
#     ):
#         """
#         Initializes the DateThread with a Series of input data.
#         Args:
#             column_data (pd.Series): The input data column to encode.
#             params (DateEncoderParameters): Parameters for RDSE encoding.
#             output (list[SDR]): A list to store the encoded SDRs.
#             row_offset (int): Offset to place encoded SDRs in the output list.
#         """
#         super().__init__()
#         self._column_data = column_data
#         self._encoder = DateEncoder(params)
#         self._output = output
#         self._row_offset = row_offset

#     def run(self):
#         """
#         Encodes the data column using date encoder and stores results in the output list.
#         """
#         for i in range(len(self._column_data)):

#             value = self._column_data.iloc[i]

#             # fresh sdr instance per row
#             sdr_instance = SDR([self._encoder.size])
#             self._encoder.encode(value, sdr_instance)
#             self._output[self._row_offset + i] = sdr_instance

#             """
#             #debug prints if you want
#             active_bits = sdr_instance.get_sum()
#             sparsity_fraction = sdr_instance.get_sparsity()
#             print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
#             """


# class CategoryThread(Thread):
#     """
#     Custom thread class to run Category encoding.
#     This overrides the run to enable encoding as the thread work.
#     The offset is dynamically chosen based on the number of threads
#     in the main batch_encoder_handler class.
#     """

#     def __init__(
#         self, column_data: pd.Series, params: CategoryParameters, output: list[SDR], row_offset: int
#     ):
#         """
#         Initializes the CategoryThread with a Series of input data.
#         Args:
#             column_data (pd.Series): The input data column to encode.
#             params (CategoryEncoder): Parameters for RDSE encoding.
#             output (list[SDR]): A list to store the encoded SDRs.
#             row_offset (int): Offset to place encoded SDRs in the output list.
#         """
#         super().__init__()
#         self._column_data = column_data
#         self._encoder = CategoryEncoder(params)
#         self._output = output
#         self._row_offset = row_offset

#     def run(self):
#         """
#         Encodes the data column using category encoder and stores results in the output list.
#         """
#         for i in range(len(self._column_data)):
#             value = self._column_data.iloc[i]

#             # fresh sdr instance per row
#             sdr_instance = SDR([self._encoder.size])
#             self._encoder.encode(value, sdr_instance)
#             self._output[self._row_offset + i] = sdr_instance

#             """
#             #debug prints if you want
#             active_bits = sdr_instance.get_sum()
#             sparsity_fraction = sdr_instance.get_sparsity()
#             print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
#             """


# class CompositeBatchThread(Thread):
#     """
#     Thread responsible for merging a contiguous batch of rows into
#     composite SDRs.

#     Each thread:
#       - Processes rows [start, end)
#       - Reads column-wise SDRs
#       - Writes composite SDRs directly into the shared output list

#     """

#     def __init__(self, batch: list[list[SDR]], out: list[SDR | None], offset: int):
#         super().__init__()
#         self._batch = batch
#         self._out = out
#         self._offset = offset

#     def run(self):
#         for i in range(len(self._batch)):
#             row_sdrs = self._batch[i]  # list[SDR]

#             flat_sdrs: list[SDR] = []
#             for s in row_sdrs:
#                 if len(s.dimensions) != 1:
#                     flat = SDR([s.size])
#                     flat.set_sparse(s.get_sparse())
#                     flat_sdrs.append(flat)
#                 else:
#                     flat_sdrs.append(s)

#             if len(flat_sdrs) == 1:
#                 self._out[self._offset + i] = copy.deepcopy(flat_sdrs[0])
#                 continue

#             total_size = sum(s.size for s in flat_sdrs)
#             composite = SDR([total_size])
#             composite.concatenate(flat_sdrs, axis=0)
#             self._out[self._offset + i] = composite


# class BatchEncoderHandler:
#     """Handles multiple encoders to create composite SDRs.
#     This class uses a singleton pattern to ensure only one instance exists.
#     It dynamically selects the appropriate encoder for each DataFrame column
#     based on its dtype and builds a composite SDR from the encoded columns.
#     On top of that, it also enables custom configuration of specific encoder
#     parameters and allows for a custom encoding for specified columns in the
#     data. This is done through simple setter methods on the parameters and
#     on the custom encoding dict[str, str] which has the first string as a
#     column name and the second string as the encoder name.
#     """

#     __instance: BatchEncoderHandler | None = None

#     def __new__(cls, input_data: pd.DataFrame | None = None) -> "BatchEncoderHandler":
#         """Implements the singleton pattern for EncoderHandler.
#         Args:
#             input_data (pd.DataFrame | None): Input data for encoder initialization.
#         Returns:
#             BatchEncoderHandler: The singleton instance.
#         """
#         if cls.__instance is None:
#             cls.__instance = super(BatchEncoderHandler, cls).__new__(cls)
#         return cls.__instance

#     def __init__(self, input_data: pd.DataFrame | None = None):
#         """Initializes the EncoderHandler with a DataFrame of input data.
#         Args:
#             input_data (pd.DataFrame | None): DataFrame containing input data.
#         """
#         self._data_frame = copy.deepcopy(input_data) if input_data is not None else pd.DataFrame()
#         self._rdse_params = RDSEParameters(
#             size=2048,
#             active_bits=40,
#             sparsity=0,
#             radius=1,
#             resolution=0,
#             category=False,
#             seed=1,
#         )
#         self._scalar_params = ScalarEncoderParameters(
#             minimum=0,
#             maximum=100,
#             clip_input=True,
#             periodic=False,
#             active_bits=40,
#             sparsity=0.0,
#             size=2048,
#             radius=0.0,
#             category=False,
#             resolution=0.0,
#         )
#         self._date_params = DateEncoderParameters(
#             season_active_bits=0,
#             season_radius=91.5,
#             day_of_week_active_bits=7,
#             day_of_week_radius=1.0,
#             weekend_active_bits=2,
#             holiday_active_bits=4,
#             holiday_dates=[[12, 25], [1, 1], [7, 4], [11, 11]],
#             time_of_day_active_bits=24,
#             time_of_day_radius=1.0,
#             custom_active_bits=0,
#             custom_days=[],
#             rdse_used=False,
#         )
#         self._custom_encoding: dict[str, str] = {}
#         assert input_data is not None, "Input data must be provided for initialization."
#         category_list = input_data.columns.unique().tolist()
#         self._category_params = CategoryParameters(w=3, category_list=category_list)

#     @classmethod
#     def get_instance(cls) -> "BatchEncoderHandler":
#         """Returns the singleton instance of BatchEncoderHandler.
#         Args:
#             input_data (pd.DataFrame): Input data for encoder initialization.
#         Returns:
#             BatchEncoderHandler: The singleton instance.
#         """
#         if cls.__instance is None:
#             cls.__instance = BatchEncoderHandler()
#         return cls.__instance

#     def _build_dict_list_sdr(
#         self, input_data: pd.DataFrame, threads_per_column: int
#     ) -> dict[str, list[SDR]]:
#         """
#         Builds a dictionary mapping each column name to a list of SDRs.
#         Each column in input_data will have a list of SDRs of length equal to the number of rows.
#         The SDRs are generated using an appropriate encoder based on the data type of each column
#         or a custom encoding specified in custom_encoding. The encoding process is parallelized
#         across batches using threads for efficiency.
#         Args:
#             input_data (pd.DataFrame): The input data whose columns will be converted into SDRs.
#             threads_per_column (int): The number of threads to use per column. The column is split
#                 into this many batches for parallel processing.
#         Returns:
#             dict[str, list[SDR]]: A dictionary where each key is a column name and the corresponding value
#                 is a list of SDR objects representing that column's data.
#         """
#         num_rows = len(input_data)
#         column_sdrs = {}
#         size = 0
#         # The sizes here may need to be dynamic based on parameter settings
#         for col in input_data.columns:
#             size = 0
#             series = input_data[col].reset_index(drop=True)

#             encoder_type = None
#             if self._custom_encoding and col in self._custom_encoding:
#                 encoder_type = self._custom_encoding[col].lower()

#             if encoder_type == "rdse":
#                 size = self._rdse_params.size
#             elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
#                 width = self._category_params.w
#                 unique_values = series.dropna().unique().tolist()
#                 size = (len(unique_values) + 1) * width
#             elif encoder_type == "date":
#                 size = DateEncoder(self._date_params)._size
#             elif encoder_type == "scalar":
#                 size = self._scalar_params.size
#             else:
#                 if pd.api.types.is_numeric_dtype(series):
#                     size = self._rdse_params.size
#                 elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
#                     width = self._category_params.w
#                     unique_values = series.dropna().unique().tolist()
#                     size = (len(unique_values) + 1) * width
#                 elif pd.api.types.is_datetime64_any_dtype(series):
#                     size = DateEncoder(self._date_params)._size
#                 else:
#                     print("No size")

#             column_sdrs[col] = [SDR([size]) for _ in range(num_rows)]

#         threads: list[Thread] = []

#         for col in input_data.columns:
#             series = input_data[col].reset_index(drop=True)
#             batches = np.array_split(series.to_numpy(), threads_per_column)
#             scalartrue = False
#             offset = 0
#             for batch in batches:
#                 if len(batch) == 0:
#                     continue

#                 batch_series = pd.Series(batch)

#                 encoder_type = None
#                 if self._custom_encoding is not None and col in self._custom_encoding:
#                     encoder_type = self._custom_encoding[col].lower()

#                 if encoder_type is None:
#                     if pd.api.types.is_numeric_dtype(series) or isinstance(
#                         series.dropna().iloc[0], (int, float)
#                     ):
#                         encoder_type = "rdse"
#                     elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
#                         series
#                     ):
#                         encoder_type = "category"
#                     elif pd.api.types.is_datetime64_any_dtype(series):
#                         encoder_type = "date"
#                     elif scalartrue:
#                         encoder_type = "scalar"

#                 thread = None
#                 if encoder_type == "rdse":
#                     params = self._rdse_params
#                     thread = RdseThread(batch_series, params, column_sdrs[col], offset)
#                 elif encoder_type == "scalar":
#                     params = self._scalar_params
#                     thread = ScalarThread(batch_series, params, column_sdrs[col], offset)
#                 elif encoder_type == "category":
#                     params = self._category_params
#                     thread = CategoryThread(batch_series, params, column_sdrs[col], offset)
#                 elif encoder_type == "date":
#                     params = self._date_params
#                     thread = DateThread(batch_series, self._date_params, column_sdrs[col], offset)
#                 else:
#                     print(f"Skipping column '{col}' (unknown encoder type '{encoder_type}')")

#                 if thread is not None:
#                     threads.append(thread)
#                     thread.start()
#                     offset += len(batch_series)

#         for t in threads:
#             t.join()

#         return column_sdrs

#     def build_composite_sdr(
#         self, input_data, threads_per_column: int, threads_per_rows: int = 4
#     ) -> list[SDR]:
#         """
#         Builds a list of composite SDRs by concatenating SDRs from each column for each row.
#         If input_data is a DataFrame, it first generates column-wise SDRs using
#         build_dict_list_sdr. Otherwise, it assumes input_data is already a dictionary
#         mapping column names to lists of SDRs. Each rows SDRs across columns are concatenated
#         into a single composite SDR. For multi-dimensional SDRs, they are flattened into 1D
#         before concatenation.
#         Args:
#             input_data (pd.DataFrame or dict[str, list[SDR]]): Input data to convert into SDRs.
#                 If a DataFrame, column-wise SDRs are generated internally. If a dictionary,
#                 it should already contain SDR lists per column.
#             threads_per_column (int): Number of threads to use per column when generating SDRs
#                 from a DataFrame. Ignored if input_data is already a dictionary.
#         Returns:
#             list[SDR]: A list of composite SDRs, one for each row in the input data. Each
#                 composite SDR represents the concatenation of all column SDRs for that row.
#         Future work: I believe we could add thread creation to merge row SDRs. This could
#         speed up the process significantly.
#         """
#         if isinstance(input_data, pd.DataFrame):
#             column_sdrs = self._build_dict_list_sdr(input_data, threads_per_column)
#         else:
#             column_sdrs = input_data

#         num_rows = len(next(iter(column_sdrs.values())))
#         composite_sdrs: list[SDR] = [None] * num_rows
#         threads_per_rows = max(1, min(threads_per_rows, num_rows))

#         rowsdrs = [[column_sdrs[col][i] for col in input_data.columns] for i in range(num_rows)]
#         batches = np.array_split(rowsdrs, threads_per_rows)

#         workers: list[Thread] = []
#         offset = 0

#         for batch_list in batches:
#             th = CompositeBatchThread(batch_list, composite_sdrs, offset)
#             workers.append(th)
#             th.start()
#             offset += len(batch_list)

#         for th in workers:
#             th.join()

#         assert all(s is not None for s in composite_sdrs)
#         return cast(list[SDR], composite_sdrs)

#     def sdrs_to_dense_matrix(self, sdrs):
#         return np.array([sdr.get_dense() for sdr in sdrs], dtype=np.uint8)

#     def build_knn_from_tm_predictions(
#         self, tm_prediction_masks, input_data, n_neighbors, weights, distance
#     ):
#         tm_prediction_masks_dense = [
#             np.asarray(mask, dtype=np.int8).ravel() for mask in tm_prediction_masks
#         ]

#         x_full = np.vstack(tm_prediction_masks_dense)

#         num_samples = min(len(x_full), len(input_data) - 1)

#         x = x_full[:num_samples]

#         target_col = input_data.columns[1]

#         row_labels = [input_data.index[i] for i in range(1, 1 + num_samples)]
#         y = input_data.loc[row_labels, target_col].to_numpy().reshape(-1, 1)

#         knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=distance)
#         knn.fit(x, y)

#         return knn

#     def set_category_encoder_parameters(self, params: CategoryParameters):
#         """
#         Set the parameters for the category encoder.
#         Args:
#             params (CategoryParameters): Configuration parameters for the category encoder.
#         """
#         self._category_params = params

#     def set_rdse_encoder_parameters(self, params: RDSEParameters):
#         """
#         Set the parameters for the RDSE (Random Distributed Scalar Encoder).
#         Args:
#             params (RDSEParameters): Configuration parameters for the RDSE encoder.
#         """
#         self._rdse_params = params

#     def set_scalar_encoder_parameters(self, params: ScalarEncoderParameters):
#         """
#         Set the parameters for the scalar encoder.
#         Args:
#             params (ScalarEncoderParameters): Configuration parameters for the scalar encoder.
#         """
#         self._scalar_params = params

#     def set_date_encoder_parameters(self, params: DateEncoderParameters):
#         """
#         Set the parameters for the date encoder.
#         Args:
#             params (DateEncoderParameters): Configuration parameters for the date encoder.
#         """
#         self._date_params = params

#     def choose_custom_column_encoding(self, custom_encoding: dict[str, str]):
#         """
#         Specify custom encoder types for specific columns.
#         Args:
#             custom_encoding (dict[str, str]): A dictionary mapping column names to
#                 encoder types. Valid encoder types include "rdse", "scalar", "category", and "date".
#                 This overrides automatic type detection for the specified columns.
#         """
#         self._custom_encoding = custom_encoding

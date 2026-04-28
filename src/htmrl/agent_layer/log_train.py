"""Train Model facade to pre-train the Brain on a dataset.

Manager of Brains to build and train them on datasets. Provides a high-level API for
training without needing to interact with the Brain's internal fields directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from matplotlib.colors import PowerNorm

import htmrl.grapher as grapher
from htmrl.agent_layer.brain import Brain
from htmrl.agent_layer.HTM import ColumnField, Field, InputField, OutputField
from htmrl.encoder_layer.base_encoder import ParameterMarker
from htmrl.encoder_layer.category_encoder import CategoryParameters
from htmrl.encoder_layer.date_encoder import DateEncoderParameters
from htmrl.encoder_layer.fourier_encoder import FourierEncoderParameters
from htmrl.encoder_layer.geospatial_encoder import GeospatialParameters
from htmrl.encoder_layer.rdse import RDSEParameters
from htmrl.log import (
    get_logger,
    log_training_progress,
    output_final_training_performance,
    save_agent_brain_shape,
    save_evaluation_parameters,
    save_final_training_performance,
    save_mean_squared_error,
    save_prediction_report,
    save_validated_dataset,
)


class Trainer:
    """Build a Trainer for training Brains on a dataset.

    Args:
        brain: Brain instance to be trained with this trainer.
    """

    _BRAIN_NOT_INITIALIZED_ERROR = "Main Brain is not initialized. Please build the Brain first."

    def __init__(self, brain: Brain) -> None:
        self.logger = get_logger(self)
        self._main_brain: Brain = brain
        self._brains: list[Brain] = []
        self._trainer_input_fields: list[Field] = []
        self._trainer_output_fields: list[Field] = []
        self._trainer_column_fields: list[Field] = []
        self._values: list[Any] = []

        if brain is not None and brain not in self._brains:
            self._brains.append(brain)

    @property
    def brains(self) -> list[Brain]:
        """Access the list of cached Brains."""
        return self._brains

    @property
    def main_brain(self) -> Brain:
        """Access the main Brain being trained."""
        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)
        return self._main_brain

    @main_brain.setter
    def main_brain(self, brain: Brain) -> None:
        """Set the main Brain being trained."""
        if brain is None:
            raise ValueError("Cannot set main Brain to None.")
        if not isinstance(brain, Brain):
            raise ValueError("main_brain must be an instance of Brain.")

        self._main_brain = brain
        if brain not in self._brains:
            self._brains.append(brain)

    def _ensure_brain_id(self, brain: Brain) -> None:
        if not hasattr(brain, "brain_id") or not str(brain.brain_id).strip():
            brain.brain_id = f"brain_{id(brain)}"

    def _reset_trainer_fields(self) -> None:
        """Clear cached field definitions before building a new brain."""
        self._trainer_input_fields = []
        self._trainer_output_fields = []
        self._trainer_column_fields = []
        self._values = []

    def _build_shape_data(
        self, brain: Brain, cells_per_column: int | None = None
    ) -> dict[str, Any]:
        """Build a serializable description of the current brain structure."""
        column_field = brain.column_fields[0] if brain.column_fields else None

        return {
            "brain_id": getattr(brain, "brain_id", "unknown"),
            "input_fields": [
                field.name for field in self._trainer_input_fields if field is not None
            ],
            "output_fields": [
                field.name for field in self._trainer_output_fields if field is not None
            ],
            "column_fields": [
                field.name for field in self._trainer_column_fields if field is not None
            ],
            "num_columns": getattr(column_field, "num_columns", None),
            "cells_per_column": (
                cells_per_column
                if cells_per_column is not None
                else getattr(column_field, "cells_per_column", None)
            ),
            "field_count": len(getattr(brain, "fields", {})),
            "field_names": list(getattr(brain, "fields", {}).keys()),
            "created_at": datetime.now().isoformat(),
        }

    def _setup_io_fields(self, fields: list[tuple[str, int, ParameterMarker]]) -> None:
        """Setup the input/output fields for the Brain.

        Args:
            fields: A list of tuples containing field name, field size, and encoder parameters.

        Raises:
            ValueError: If unsupported encoder parameter type is provided.
        """
        for name, size, param in fields:
            self.logger.info(
                "Setting up field %-24s size=%5d encoder=%-28s",
                name,
                size,
                type(param).__name__,
            )

            if isinstance(param, RDSEParameters):
                encoder_params = RDSEParameters(
                    size=param.size,
                    sparsity=param.sparsity,
                    resolution=param.resolution,
                    category=param.category,
                    seed=param.seed,
                )
            elif isinstance(param, DateEncoderParameters):
                encoder_params = DateEncoderParameters(
                    size=param.size,
                    season_active_bits=param.season_active_bits,
                    season_radius=param.season_radius,
                    day_of_week_active_bits=param.day_of_week_active_bits,
                    day_of_week_radius=param.day_of_week_radius,
                    weekend_active_bits=param.weekend_active_bits,
                    holiday_active_bits=param.holiday_active_bits,
                    holiday_dates=param.holiday_dates,
                    time_of_day_active_bits=param.time_of_day_active_bits,
                    time_of_day_radius=param.time_of_day_radius,
                    custom_active_bits=param.custom_active_bits,
                    custom_days=param.custom_days,
                    rdse_used=param.rdse_used,
                )
            elif isinstance(param, CategoryParameters):
                encoder_params = CategoryParameters(
                    size=param.size,
                    category_list=param.category_list,
                    rdse_used=param.rdse_used,
                )
            elif isinstance(param, FourierEncoderParameters):
                encoder_params = FourierEncoderParameters(
                    size=param.size,
                    frequency_ranges=param.frequency_ranges,
                    sparsity_in_ranges=param.sparsity_in_ranges,
                    resolutions_in_ranges=param.resolutions_in_ranges,
                    seed=param.seed,
                )
            elif isinstance(param, GeospatialParameters):
                encoder_params = GeospatialParameters(
                    size=param.size,
                    scale=param.scale,
                    timestep=param.timestep,
                    max_radius=param.max_radius,
                    use_altitude=param.use_altitude,
                )
            else:
                raise ValueError(f"Unsupported encoder parameters type: {type(param)}")

            if name.endswith("_input"):
                field = InputField(size=size, encoder_params=encoder_params)
                field.name = name
                self._trainer_input_fields.append(field)
            elif name.endswith("_output"):
                field = OutputField(size=size, motor_action=(None,))
                field.name = name
                self._trainer_output_fields.append(field)
            else:
                raise ValueError(f"Unsupported field type: {name}")

    def _setup_column_fields(self, num_columns: int, cells_per_column: int) -> None:
        """Setup the ColumnField for the Brain."""
        column_field = ColumnField(
            input_fields=self._trainer_input_fields,
            non_spatial=True,
            num_columns=num_columns,
            cells_per_column=cells_per_column,
        )
        column_field.name = "Column_Field"
        self._trainer_column_fields.append(column_field)

    def _create_brain(self) -> Brain:
        """Create a Brain instance with the configured fields."""
        if not self._trainer_input_fields:
            raise ValueError("No input fields defined for the Brain.")
        if not self._trainer_column_fields:
            raise ValueError("No column fields defined for the Brain.")

        brain = Brain(
            {
                **{field.name: field for field in self._trainer_input_fields if field is not None},
                **{field.name: field for field in self._trainer_output_fields if field is not None},
                **{field.name: field for field in self._trainer_column_fields if field is not None},
            }
        )

        self._ensure_brain_id(brain)

        self.logger.info("Brain created with fields: %s", list(brain.fields.keys()))
        return brain

    def _build_values_list(self, dataset: dict[str, list[Any]]) -> None:
        """Build a flat list of values from the dataset for training diagnostics."""
        self._values = []

        for column_name, column_data in dataset.items():
            self.logger.debug(
                "Processing column '%s' with %d data points.",
                column_name,
                len(column_data),
            )
            for value in column_data:
                self._values.append(value)

        self.logger.info(
            "Built values list with %d total data points from dataset.",
            len(self._values),
        )

    def print_train_stats(
        self,
        test_results: dict[str, Any] | None = None,
        training_steps: int | None = None,
    ) -> None:
        """Print statistics about the training dataset and optional test results."""
        self.logger.info("Training dataset statistics:")
        self._main_brain.print_stats()

        if test_results is not None and training_steps is not None:
            output_final_training_performance(
                self._main_brain,
                training_steps=training_steps,
                test_results=test_results,
            )

    def build_brain(self, fields: list[tuple[str, int, ParameterMarker]]) -> Brain:
        """Build the Brain for training using an explicit field list."""
        if not fields:
            raise ValueError("fields cannot be empty.")

        self._reset_trainer_fields()

        size = max(field[1] for field in fields)
        self._setup_io_fields(fields)
        self._setup_column_fields(num_columns=size, cells_per_column=16)

        brain = self._create_brain()
        self._main_brain = brain

        if brain not in self._brains:
            self._brains.append(brain)

        save_agent_brain_shape(
            brain,
            self._build_shape_data(brain, cells_per_column=16),
        )

        return brain

    def add_input_field(self, name: str, size: int, encoder_params: ParameterMarker) -> None:
        """Add an input field to the Brain."""
        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if not name.endswith("_input"):
            raise ValueError("Input field name must end with '_input'.")

        field = InputField(size=size, encoder_params=encoder_params)
        field.name = name
        self._trainer_input_fields.append(field)
        self._main_brain.fields[name] = field

        save_agent_brain_shape(self._main_brain, self._build_shape_data(self._main_brain))

    def add_output_field(self, name: str, size: int, motor_action: tuple[Any, ...]) -> None:
        """Add an output field to the Brain."""
        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if not name.endswith("_output"):
            raise ValueError("Output field name must end with '_output'.")

        field = OutputField(size=size, motor_action=motor_action)
        field.name = name
        self._trainer_output_fields.append(field)
        self._main_brain.fields[name] = field

        save_agent_brain_shape(self._main_brain, self._build_shape_data(self._main_brain))

    def add_column_field(self, name: str, num_columns: int, cells_per_column: int) -> None:
        """Add a column field to the Brain."""
        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if not name.endswith("_column"):
            raise ValueError("Column field name must end with '_column'.")

        field = ColumnField(
            input_fields=self._trainer_input_fields,
            non_spatial=True,
            num_columns=num_columns,
            cells_per_column=cells_per_column,
        )
        field.name = name
        self._trainer_column_fields.append(field)
        self._main_brain.fields[name] = field

        save_agent_brain_shape(
            self._main_brain,
            self._build_shape_data(self._main_brain, cells_per_column=cells_per_column),
        )

    def train_column(self, brain: Brain, column: dict[str, list[Any]], steps: int) -> None:
        """Train the Brain on a single-column dataset."""
        self.main_brain = brain

        if len(column.keys()) == 0:
            raise ValueError("Dataset is empty. Cannot train on an empty dataset.")
        if len(column.keys()) > 1:
            raise ValueError(
                "Dataset has more than one column. Use train_full_brain() to train on multiple columns."
            )

        save_validated_dataset(brain, column)

        self.logger.info("Training one column with %d data points.", steps)

        name = list(column.keys())[0]
        values = column[name]

        if not values:
            raise ValueError(f"Column '{name}' has no data.")

        for step in range(steps):
            value = values[step % len(values)]
            input_field_name = name

            matching_field = next(
                (field for field in self._trainer_input_fields if field.name == input_field_name),
                None,
            )
            if matching_field is None:
                raise ValueError(f"No matching input field found for column '{name}'.")

            input_dict = {matching_field.name: value}
            log_training_progress(brain, step + 1, steps, input_dict)
            brain.step(inputs=input_dict, learn=True)

    def train_full_brain(self, brain: Brain, dataset: dict[str, list[Any]], steps: int) -> None:
        """Train the Brain on the specified dataset."""
        self.main_brain = brain

        if not dataset:
            raise ValueError("Dataset is empty. Cannot train on an empty dataset.")

        save_validated_dataset(brain, dataset)

        self.logger.info("Training on dataset with columns: %s", list(dataset.keys()))

        for column_name, column_data in dataset.items():
            self.logger.debug("Column '%s' has %d data points.", column_name, len(column_data))
            if not column_data:
                raise ValueError(f"Column '{column_name}' has no data.")

        for step in range(steps):
            column_data = {key: dataset[key][step % len(dataset[key])] for key in dataset}

            input_dict: dict[str, Any] = {}
            for field in self._trainer_input_fields:
                if field.name in column_data:
                    input_dict[field.name] = column_data[field.name]
                else:
                    raise ValueError(f"Data point is missing required input field: {field.name}")

            log_training_progress(brain, step + 1, steps, input_dict)
            brain.step(inputs=input_dict, learn=True)

    def test(
        self,
        brain: Brain | None,
        dataset: dict[str, list[Any]],
        steps: int | None = None,
    ) -> dict[str, Any]:
        """Replay inputs without learning and report prediction accuracy metrics."""
        if brain is None:
            brain = self._main_brain
        if brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if not isinstance(dataset, dict) or not dataset:
            raise ValueError("Dataset must be a non-empty dict mapping field names to data.")

        for field_name, values in dataset.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Dataset field '{field_name}' must contain a non-empty list.")

        steps = steps or min(len(values) for values in dataset.values())

        save_evaluation_parameters(
            brain,
            {
                "steps": steps,
                "dataset_fields": list(dataset.keys()),
                "dataset_lengths": {k: len(v) for k, v in dataset.items()},
                "learn": False,
                "timestamp": datetime.now().isoformat(),
            },
        )

        field_errors: dict[str, list[float]] = {name: [] for name in dataset}
        prediction_failures: dict[str, int] = dict.fromkeys(dataset, 0)
        evaluation_bursts: list[int] = []
        missing_prediction_penalty = 1.0

        self.logger.info(
            "Testing predictions for %d steps on fields: %s",
            steps,
            list(dataset.keys()),
        )

        for step in range(steps):
            predictions = brain.prediction()

            for field_name, series in dataset.items():
                expected_value = series[step % len(series)]
                predicted_value = (
                    predictions.get(field_name) if isinstance(predictions, dict) else None
                )

                if predicted_value is None:
                    prediction_failures[field_name] += 1
                    field_errors[field_name].append(missing_prediction_penalty)
                    continue

                try:
                    diff = float(expected_value) - float(predicted_value)
                    field_errors[field_name].append(diff * diff)
                except (TypeError, ValueError):
                    field_errors[field_name].append(
                        0.0 if expected_value == predicted_value else missing_prediction_penalty
                    )

            inputs = {name: series[step % len(series)] for name, series in dataset.items()}
            brain.step(inputs=inputs, learn=False)

            bursts = sum(len(column_field.bursting_columns) for column_field in brain.column_fields)
            evaluation_bursts.append(bursts)

        flattened_errors = [err for errors in field_errors.values() for err in errors]
        mean_squared_error = (
            sum(flattened_errors) / len(flattened_errors) if flattened_errors else 0.0
        )
        avg_bursting_columns = (
            sum(evaluation_bursts) / len(evaluation_bursts) if evaluation_bursts else 0.0
        )
        total_failures = sum(prediction_failures.values())

        field_rows: list[tuple[str, float, int]] = []
        for field_name, errors in field_errors.items():
            mse = sum(errors) / len(errors) if errors else 0.0
            failures = prediction_failures[field_name]
            field_rows.append((field_name, mse, failures))

        table_lines = [
            "+----------------------------+---------------+-----------+",
            "| Field                      |     MSE       | Failures |",
            "+----------------------------+---------------+-----------+",
        ]
        for name, mse, failures in field_rows:
            table_lines.append(f"| {name:<26} | {mse:>11.6f} | {failures:>9} |")
        table_lines.append("+----------------------------+---------------+-----------+")

        summary_lines = [
            "Prediction Summary:",
            f"  Global MSE: {mean_squared_error:.6f}",
            f"  Total prediction failures: {total_failures}",
            f"  Avg bursting columns: {avg_bursting_columns:.3f}",
            "  Per-field metrics:",
        ]
        summary_lines.extend(f"    {line}" for line in table_lines)

        report_text = "\n".join(summary_lines)
        self.logger.info(report_text)

        results = {
            "mean_squared_error": mean_squared_error,
            "prediction_failures": prediction_failures,
            "total_prediction_failures": total_failures,
            "avg_bursting_columns": avg_bursting_columns,
            "evaluation_bursts": evaluation_bursts,
            "errors": field_errors,
        }

        save_mean_squared_error(brain, mean_squared_error)
        save_final_training_performance(brain, steps, results)
        save_prediction_report(brain, report_text)

        return results

    def build_full_brain(
        self,
        dataset: dict[Any, list[Any]],
        size: int = 2048,
        params: ParameterMarker | None = None,
    ) -> Brain:
        """Build a full Brain with all fields based on the dataset.

        This method automatically determines the appropriate encoder type for each field
        based on the data type of the values in the dataset.
        """
        if not dataset:
            raise ValueError("dataset cannot be empty.")

        self._reset_trainer_fields()

        for key, values in dataset.items():
            if not values:
                raise ValueError(f"Dataset field '{key}' is empty.")

            first_value = values[0]

            if isinstance(first_value, (int, float)):
                encoder_params = (
                    RDSEParameters(
                        size=params.size,
                        sparsity=params.sparsity,
                        resolution=params.resolution,
                        category=False,
                        seed=params.seed,
                    )
                    if isinstance(params, RDSEParameters)
                    else RDSEParameters(size=size)
                )

            elif isinstance(first_value, str):
                encoder_params = (
                    CategoryParameters(
                        size=params.size,
                        category_list=list(set(values)),
                        rdse_used=params.rdse_used,
                    )
                    if isinstance(params, CategoryParameters)
                    else CategoryParameters(
                        size=size,
                        category_list=list(set(values)),
                        rdse_used=False,
                    )
                )

            elif isinstance(first_value, (list, np.ndarray)):
                encoder_params = (
                    FourierEncoderParameters(
                        size=params.size,
                        frequency_ranges=params.frequency_ranges,
                        sparsity_in_ranges=params.sparsity_in_ranges,
                        resolutions_in_ranges=params.resolutions_in_ranges,
                        seed=params.seed,
                    )
                    if isinstance(params, FourierEncoderParameters)
                    else FourierEncoderParameters(size=size)
                )

            elif isinstance(first_value, datetime):
                encoder_params = (
                    DateEncoderParameters(
                        size=params.size,
                        season_active_bits=params.season_active_bits,
                        season_radius=params.season_radius,
                        day_of_week_active_bits=params.day_of_week_active_bits,
                        day_of_week_radius=params.day_of_week_radius,
                        weekend_active_bits=params.weekend_active_bits,
                        holiday_active_bits=params.holiday_active_bits,
                        holiday_dates=params.holiday_dates,
                        time_of_day_active_bits=params.time_of_day_active_bits,
                        time_of_day_radius=params.time_of_day_radius,
                        custom_active_bits=params.custom_active_bits,
                        custom_days=params.custom_days,
                        rdse_used=params.rdse_used,
                    )
                    if isinstance(params, DateEncoderParameters)
                    else DateEncoderParameters(size=size)
                )

            elif (
                isinstance(first_value, tuple)
                and len(first_value) >= 2
                and all(isinstance(v, (int, float)) for v in first_value)
            ):
                encoder_params = (
                    GeospatialParameters(
                        size=params.size,
                        max_radius=params.max_radius,
                        scale=params.scale,
                        timestep=params.timestep,
                        use_altitude=params.use_altitude,
                    )
                    if isinstance(params, GeospatialParameters)
                    else GeospatialParameters(size=size)
                )

            else:
                raise ValueError(f"Unsupported data type for field '{key}': {type(first_value)}")

            self._setup_io_fields([(f"{key}_input", size, encoder_params)])

        self._setup_column_fields(num_columns=size, cells_per_column=32)

        brain = self._create_brain()
        self._main_brain = brain

        if brain not in self._brains:
            self._brains.append(brain)

        save_agent_brain_shape(
            brain,
            self._build_shape_data(brain, cells_per_column=32),
        )

        return brain

    def show_active_columns(self, brain: Brain, dataset_name: str | None = None) -> None:
        """Show the active columns in the Brain."""
        for column_field in brain.column_fields:
            sdr = [
                1 if column in column_field.active_columns else 0 for column in column_field.columns
            ]

            num_active = sum(sdr)
            sparsity = (num_active / len(sdr)) * 100 if sdr else 0.0
            dataset_info = f" - {dataset_name}" if dataset_name else ""

            grapher.plot_sdr(
                sdr,
                title=(
                    f"Active Columns: {column_field.name}{dataset_info}\n"
                    f"({num_active}/{len(sdr)} active, {sparsity:.1f}% sparsity)"
                ),
            )

    def show_heat_map(self, brain: Brain, dataset_name: str | None = None) -> None:
        """Show a heat map of the Brain's column duty cycle activity."""
        if not brain.column_fields:
            raise ValueError("No column fields available to visualize.")

        column_field = brain.column_fields[0]
        duty_cycles = np.array([column.active_duty_cycle for column in column_field.columns])

        if duty_cycles.size == 0:
            raise ValueError("Column field has no columns to visualize.")

        side = int(np.ceil(np.sqrt(duty_cycles.size)))
        heat_map = np.zeros((side, side))
        heat_map.flat[: duty_cycles.size] = duty_cycles

        positive = duty_cycles[duty_cycles > 0]
        active_columns = len(positive)
        max_duty = float(positive.max()) if positive.size > 0 else 0.0
        norm = None

        dataset_info = f" - {dataset_name}" if dataset_name else ""
        title = (
            f"Column Duty Cycle Heat Map: {column_field.name}{dataset_info}\n"
            f"({active_columns}/{len(duty_cycles)} active columns, max duty={max_duty:.3f})"
        )

        if positive.size > 0:
            vmax = max(max_duty, 1e-6)
            norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)
            grapher.show_heat_map(
                heat_map,
                title=title,
                norm=norm,
            )
        else:
            grapher.show_heat_map(
                heat_map,
                title=title,
                vmin=0.0,
                vmax=1.0,
            )

    # TODO: Implement save_brain and load_brain methods for persistence of trained Brains
    def save_brain(self, brain: Brain, filename: str) -> None:
        """Save the Brain to a file."""
        # Implement saving logic, e.g., using joblib or pickle

    def load_brain(self, filename: str) -> Brain:
        """Load a Brain from a file."""
        # Implement loading logic, e.g., using joblib or pickle
        return Brain()  # Placeholder return

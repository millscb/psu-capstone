"""Train Model facade to pre-train the Brain on a dataset.

Manager of Brains to build and train them on datasets. Provides a high-level API for training
without needing to interact with the Brain's internal fields directly.

"""

from __future__ import annotations

import io
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

import htmrl.encoder_layer as el
from htmrl.agent_layer.pullin.field_base import Field
from htmrl.agent_layer.pullin.pullin_brain import Brain
from htmrl.agent_layer.pullin.pullin_htm import ColumnField, InputField, OutputField
from htmrl.agent_layer.pullin.sungur import ValueField
from htmrl.encoder_layer.base_encoder import ParameterMarker
from htmrl.encoder_layer.encoder_factory import EncoderFactory
from htmrl.log import get_logger

# Rebind encoder parameter types locally so callers can use short names without
# importing them individually from the encoder sub-layer.
CategoryParameters = el.CategoryParameters
CoordinateParameters = el.CoordinateParameters
DateEncoderParameters = el.DateEncoderParameters
FourierEncoderParameters = el.FourierEncoderParameters
GeospatialParameters = el.GeospatialParameters
RDSEParameters = el.RDSEParameters
ScalarEncoderParameters = el.ScalarEncoderParameters


class Trainer:
    def build_brain_for_env(
        self,
        adapter: Any,
        config: Any,
        pretrain_dataset: dict | None = None,
        pretrain_steps: int = 0,
    ) -> Brain:
        """Build a Brain that matches the adapter's observation/action schema, with optional pre-training."""
        logger = get_logger("Trainer.build_brain_for_env")
        reset_bridge = adapter.reset_bridge()
        input_names = list(reset_bridge["inputs"].keys())

        obs_spec = adapter.get_observation_spec()
        action_spec = adapter.get_action_spec()
        obs_space_type = obs_spec.get("space_type", "Box")

        logger.info(f"Building Brain for environment: {getattr(adapter, 'env_id', str(adapter))}")
        logger.info(f"Observation space type: {obs_space_type}, Action spec: {action_spec}")

        # Clear any previous fields
        self._trainer_input_fields.clear()
        self._trainer_output_fields.clear()
        self._trainer_column_fields.clear()
        self._main_brain.fields.clear()

        for index, name in enumerate(input_names):
            if obs_space_type == "Discrete":
                n_categories = int(obs_spec.get("n", 16))
                params = CategoryParameters(
                    size=config.input_size,
                    w=3,
                    category_list=[str(i) for i in range(n_categories)],
                )
                logger.info(
                    f"Adding discrete input field: {name}_input with {n_categories} categories."
                )
            else:
                params = RDSEParameters(
                    size=config.input_size,
                    active_bits=0,
                    sparsity=0.02,
                    resolution=config.resolution,
                    category=False,
                    seed=config.seed + index,
                )
                logger.info(
                    f"Adding continuous input field: {name}_input (RDSE, size={config.input_size})."
                )
            self.add_input_field(f"{name}_input", config.input_size, params)

        reward_params = RDSEParameters(
            size=config.input_size,
            active_bits=0,
            sparsity=0.02,
            resolution=0.01,
            category=False,
            seed=config.seed + len(input_names),
        )
        self.add_input_field("reward_input", config.input_size, reward_params)
        logger.info("Added reward_input field (RDSE).")

        action_space_type = action_spec.get("space_type", "Discrete")
        if action_space_type == "Discrete":
            n_actions = int(action_spec.get("n", 2))
            possible_actions = list(range(n_actions))
            logger.info(f"Action space: Discrete with {n_actions} actions.")
        else:
            n_actions = 2
            possible_actions = [0, 1]
            logger.info("Action space: Non-discrete, defaulting to 2 actions.")

        min_active_bits = max(1, int(round(n_actions * 0.02)))
        encoder_params = RDSEParameters(
            size=n_actions,
            active_bits=min_active_bits,
            sparsity=0.0,
            resolution=0.01,
            category=False,
            seed=config.seed + 9999,
        )
        self.add_output_field(
            "action_output", n_actions, encoder_params, possible_actions=possible_actions
        )
        logger.info(
            f"Added action_output field (RDSE, size={n_actions}, actions={possible_actions})."
        )
        self.add_column_field(
            "column_column",
            num_columns=config.input_size,
            cells_per_column=config.cells_per_column,
            non_spatial=getattr(config, "non_spatial", False),
            non_temporal=getattr(config, "non_temporal", False),
        )
        logger.info(
            f"Added column_column field (num_columns={config.input_size}, cells_per_column={config.cells_per_column})."
        )

        trainer_brain = self.main_brain
        column_field = cast(ColumnField, trainer_brain.fields["column_column"])
        go_field = ValueField(
            input_fields=[column_field],
            num_columns=config.input_size,
            non_spatial=getattr(config, "non_spatial", False),
            non_temporal=getattr(config, "non_temporal", False),
            cells_per_column=config.cells_per_column,
        )
        nogo_field = ValueField(
            input_fields=[column_field],
            num_columns=config.input_size,
            non_spatial=getattr(config, "non_spatial", False),
            non_temporal=getattr(config, "non_temporal", False),
            cells_per_column=config.cells_per_column,
        )

        column_field.go_field = go_field
        column_field.nogo_field = nogo_field

        remapped_fields: dict[str, Any] = {
            name: trainer_brain.fields[f"{name}_input"] for name in input_names
        }
        remapped_fields["reward"] = trainer_brain.fields["reward_input"]
        remapped_fields["action_output"] = trainer_brain.fields["action_output"]
        remapped_fields["column"] = column_field
        remapped_fields["go"] = go_field
        remapped_fields["nogo"] = nogo_field

        logger.info(f"Brain fields: {list(remapped_fields.keys())}")
        logger.info("Brain construction complete.")
        brain = Brain(remapped_fields)

        # Pre-train if dataset and steps are provided
        if pretrain_dataset is not None and pretrain_steps > 0:
            logger.info(f"Pre-training brain for {pretrain_steps} steps on provided dataset.")
            self.train_full_brain(brain, pretrain_dataset, pretrain_steps)
            logger.info("Pre-training complete.")

        return brain

    """Build a Trainer for training Brains on a dataset.

    Args:
        brain: Brain instance to be trained with this trainer.

    Attributes:
        brain_blueprint: Blueprint class used for creating Brain instances.
        input_field_blueprint: Blueprint class used for creating input fields.
        output_field_blueprint: Blueprint class used for creating output fields.
        column_field_blueprint: Blueprint class used for creating column fields.
    """

    _BRAIN_NOT_INITIALIZED_ERROR = "Main Brain is not initialized. Please build the Brain first."

    # class variables for blueprints
    brain_blueprint = Brain
    input_field_blueprint = InputField
    output_field_blueprint = OutputField
    column_field_blueprint = ColumnField

    def __init__(self, brain: Brain) -> None:
        self.logger = get_logger(self)
        self._encoder_factory = EncoderFactory()
        self._main_brain = brain
        self._brains = []
        self._trainer_input_fields = []
        self._trainer_output_fields = []
        self._trainer_column_fields = []
        self._values = []

    @staticmethod
    def _encoder_type_from_params(param: ParameterMarker) -> str:
        """Map a parameter dataclass to the factory encoder type string."""

        encoder_name = param.encoder_class.__name__.lower()
        encoder_map = {
            "randomdistributedscalarencoder": "rdse",
            "dateencoder": "date",
            "categoryencoder": "category",
            "fourierencoder": "fourier",
            "geospatialencoder": "geospatial",
            "coordinateencoder": "coordinate",
            "scalarencoder": "scalar",
        }
        if encoder_name not in encoder_map:
            raise ValueError(f"Unsupported encoder parameters type: {type(param)}")
        return encoder_map[encoder_name]

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

    @property
    def input_fields(self) -> list[Field]:
        """Access the input fields configured for training."""
        return self._trainer_input_fields

    @property
    def output_fields(self) -> list[Field]:
        """Access the output fields configured for training."""
        return self._trainer_output_fields

    @property
    def column_fields(self) -> list[Field]:
        """Access the column fields configured for training."""
        return self._trainer_column_fields

    @main_brain.setter
    def main_brain(self, brain: Brain) -> None:
        """Set the main Brain being trained."""
        if brain is None:
            raise ValueError("Cannot set main Brain to None.")
        elif not isinstance(brain, Brain):
            raise ValueError("main_brain must be an instance of Brain.")

        self._main_brain = brain
        if brain not in self._brains:
            self._brains.append(brain)

    def _sync_trainer_fields_from_brain(self, brain: Brain) -> None:
        """Align cached field lists with ``brain.fields`` (e.g. after :meth:`load_brain`)."""

        self._trainer_input_fields = [f for f in brain.fields.values() if isinstance(f, InputField)]
        self._trainer_output_fields = [
            f for f in brain.fields.values() if isinstance(f, OutputField)
        ]
        self._trainer_column_fields = [
            f for f in brain.fields.values() if isinstance(f, ColumnField)
        ]

    def _setup_io_fields(
        self,
        fields: list[tuple[str, int, ParameterMarker]],
        possible_actions: list[Any] | None = None,
    ) -> None:
        """Setup the fields for the Brain through the passed in tuple.

        Args:
            fields: A list of tuples containing field name, field size, and encoder parameters.
            possible_actions: Optional action values to pre-register on output encoders.

        Raises:
            ValueError: If unsupported encoder parameter type is provided.

        Example:
            fields = [
                ("consumption_input", 12288, RDSEParameters(size=12288, sparsity=0.02, resolution=0.001, category=False, seed=5)),
                ("date_input", 12288, DateEncoderParameters(size=12288, resolution=0.001, seed=5)),
            ]

        Note:
            Field names ending with '_input' will automatically be treated as InputFields.
        """
        for name, size, param in fields:
            self.logger.info(
                "Setting up field %-24s size=%5d encoder=%-28s",
                name,
                size,
                type(param).__name__,
            )
            encoder_type = self._encoder_type_from_params(param)
            # Only use asdict for known encoder parameter dataclasses
            encoder_param_types = (
                RDSEParameters,
                CategoryParameters,
                DateEncoderParameters,
                FourierEncoderParameters,
                GeospatialParameters,
                CoordinateParameters,
            )
            if isinstance(param, encoder_param_types):
                pass
            else:
                raise ValueError(
                    f"Encoder parameter {param} is not a supported encoder parameter dataclass instance."
                )
            created_encoder = cast(
                el.BaseEncoder, self._encoder_factory.create_encoder(encoder_type, asdict(param))
            )
            encoder_params = param

            if not isinstance(created_encoder, el.BaseEncoder):
                self.logger.error("Failed to create encoder for field '%s'", name)
                continue

            if name.endswith("_input"):
                field = InputField(size=size, encoder_params=encoder_params)
                field.name = name
            elif name.endswith("_output"):
                # Use the first input field as the input_field for OutputField by default
                if not self._trainer_input_fields:
                    raise ValueError(
                        "At least one input field must be defined before creating an OutputField."
                    )
                input_field = self._trainer_input_fields[0]
                field = OutputField(
                    input_field=input_field, encoder_params=encoder_params, size=size
                )
                field.name = name
                # Register all possible actions with the encoder for OutputField
                if possible_actions is not None:
                    encoder = getattr(field, "encoder", None)
                    if encoder is not None:
                        for action in possible_actions:
                            try:
                                encoder.register_encoding(action)
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to register action {action} with OutputField encoder: {e}"
                                )
            else:
                raise ValueError("Unsupported field type: {}".format(name))

            if isinstance(field, InputField):
                self._trainer_input_fields.append(field)
            elif isinstance(field, OutputField):
                self._trainer_output_fields.append(field)

    def _setup_column_fields(self, num_columns: int, cells_per_column: int) -> None:
        """Setup the ColumnField for the Brain."""
        column_field = ColumnField(
            input_fields=self._trainer_input_fields,
            non_spatial=True,
            num_columns=num_columns,
            cells_per_column=1,
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

        self.logger.info("Brain created with fields: %s", list(brain.fields.keys()))

        return brain

    def _build_values_list(self, dataset: dict[str, list[Any]]) -> None:
        """Build a list of values from the dataset for training."""

        for column_name, column_data in dataset.items():
            self.logger.debug(
                f"Processing column '{column_name}' with {len(column_data)} data points."
            )
            for value in column_data:
                self._values.append(value)
        self.logger.info(
            f"Built values list with {len(self._values)} total data points from dataset."
        )

    def print_train_stats(
        self,
        save_path: str | None = None,
        test_results: dict[str, Any] | None = None,
        training_steps: int | None = None,
    ) -> None:
        """Print statistics about the training dataset.

        Args:
            save_path: Optional file path to append the brain report to. If provided,
                      the report will be appended to the file.
            test_results: Optional test/prediction results dictionary to include in the report.
            training_steps: Optional number of training steps to include in the report.
        """
        self.logger.info("Training dataset statistics:")
        self._main_brain.print_stats()

        if save_path:
            # Create parent directories if they don't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "a") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Brain Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")

                # Include training steps if provided
                if training_steps is not None:
                    f.write(f"Training Steps: {training_steps}\n")
                    f.write("-" * 80 + "\n")

                # Capture the brain stats

                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()

                self._main_brain.print_stats()
                sys.stdout = old_stdout
                f.write(buffer.getvalue())
                f.write("\n")

                # Include test results if provided
                if test_results:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("PREDICTION TEST RESULTS\n")
                    f.write("-" * 80 + "\n\n")

                    # Write summary metrics
                    f.write("Prediction Summary:\n")
                    f.write(f"  Global MSE: {test_results.get('mean_squared_error', 0.0):.6f}\n")
                    f.write(
                        f"  Total prediction failures: {test_results.get('total_prediction_failures', 0)}\n"
                    )
                    f.write(
                        f"  Avg bursting columns: {test_results.get('avg_bursting_columns', 0.0):.3f}\n\n"
                    )

                    # Write per-field metrics table
                    if "errors" in test_results and "prediction_failures" in test_results:
                        f.write("  Per-field metrics:\n")
                        f.write("    +----------------------------+---------------+-----------+\n")
                        f.write("    | Field                      |     MSE       | Failures |\n")
                        f.write("    +----------------------------+---------------+-----------+\n")

                        field_errors = test_results["errors"]
                        failures = test_results["prediction_failures"]

                        for field_name, errors in field_errors.items():
                            mse = sum(errors) / len(errors) if errors else 0.0
                            failure_count = failures.get(field_name, 0)
                            f.write(
                                f"    | {field_name:<26} | {mse:>11.6f} | {failure_count:>9} |\n"
                            )

                        f.write("    +----------------------------+---------------+-----------+\n")

                    f.write("\n")

    def build_brain(
        self,
        fields: list[tuple[str, int, el.ParameterMarker]],
        possible_actions: list[Any] | None = None,
    ) -> Brain:
        """Build the Brain for training. Building the Brain this way allows for more direct control over the fields and their parameters, which can be crucial for effective training.

        Args:
            fields: A list of tuples containing field name, field size, and encoder parameters.
            possible_actions: Optional action values to pre-register on output encoders.

                Example:
                fields = [
                    ("consumption_input", 2048, RDSEParameters(size=2048, sparsity=0.02, resolution=1.0, category=False, seed=5)),
                    ("date_input", 2048, DateEncoderParameters(size=2048, resolution=1.0, seed=5)),
                ]
        Returns:
            The built Brain instance ready for training.

        """
        # build a brain based on the provided fields, using the maximum size from the fields to determine the number of columns in the column field and the size of the output SDR. This ensures that the column field can accommodate the largest input field

        size = max(field[1] for field in fields)  # Get the maximum size from the fields
        # make the input/output fields, passing possible_actions for output fields
        self._setup_io_fields(fields, possible_actions=possible_actions)
        # make column fields
        self._setup_column_fields(num_columns=size, cells_per_column=16)
        # create the Brain with fields
        brain = self._create_brain()
        self._main_brain = brain

        if brain not in self._brains:
            self._brains.append(brain)

        return brain

    def add_input_field(self, name: str, size: int, encoder_params: el.ParameterMarker) -> None:
        """Add an input field to the Brain."""

        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if name.endswith("_input"):
            field = InputField(size=size, encoder_params=encoder_params)
            field.name = name
            self._trainer_input_fields.append(field)
            self._main_brain.fields[name] = field
        else:
            raise ValueError("Input field name must end with '_input'.")

    def add_output_field(
        self, name: str, size: int, encoder_params: Any, possible_actions: list[Any] | None = None
    ) -> None:
        """Add an output field to the Brain using the new OutputField signature."""

        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if name.endswith("_output"):
            if not self._trainer_input_fields:
                raise ValueError(
                    "At least one input field must be defined before creating an OutputField."
                )
            input_field = self._trainer_input_fields[0]
            # Support both OutputField signatures used in this repository:
            # - New: OutputField(input_field=..., encoder_params=..., size=...)
            # - Legacy: OutputField(size=..., motor_action=...)
            field = OutputField(input_field=input_field, encoder_params=encoder_params, size=size)
            field.name = name
            # Register all possible actions with the encoder for OutputField
            if possible_actions is not None:
                encoder = getattr(field, "encoder", None)
                if encoder is not None:
                    for action in possible_actions:
                        try:
                            encoder.register_encoding(action)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to register action {action} with OutputField encoder: {e}"
                            )
            self._trainer_output_fields.append(field)
            self._main_brain.fields[name] = field
        else:
            raise ValueError("Output field name must end with '_output'.")

    def add_column_field(
        self,
        name: str,
        num_columns: int,
        cells_per_column: int,
        non_spatial: bool = False,
        non_temporal: bool = False,
    ) -> None:
        """Add a column field to the Brain."""

        if self._main_brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if name.endswith("_column"):
            field = ColumnField(
                input_fields=self._trainer_input_fields,
                non_spatial=non_spatial,
                non_temporal=non_temporal,
                num_columns=num_columns,
                cells_per_column=cells_per_column,
            )
            field.name = name
            self._trainer_column_fields.append(field)
            self._main_brain.fields[name] = field
        else:
            raise ValueError("Column field name must end with '_column'.")

    def train_column(
        self, column: dict[str, list[Any]], steps: int, brain: Brain = main_brain
    ) -> None:
        """Train the Brain on the specified dataset."""

        self._main_brain = brain if brain is not None else self._main_brain

        if len(column.keys()) == 0:
            raise ValueError("Dataset is empty. Cannot train on an empty dataset.")
        elif len(column.keys()) > 1:
            raise ValueError(
                "Dataset has more than one column. Use train_brain() to train on multiple columns."
            )
        else:

            self.logger.info(f"Training one column with {steps} data points.")

        for step in range(steps):
            # Get the current data point from the column
            name = list(column.keys())[0]
            value = column[name][step % len(column[name])]
            self.logger.info(f"Step {step + 1}/{steps}, Data Point: {name}={value}")

            for field in self._trainer_input_fields:
                if field.name == name:
                    input_dict = {field.name: value}
                    break
            else:
                raise ValueError(f"No matching input field found for column '{name}'.")

            # Step the Brain with the prepared inputs
            brain.step(inputs=input_dict, learn=True)

    def train_full_brain(self, brain: Brain, dataset: dict[str, list[Any]], steps: int) -> None:
        """Train the Brain on the specified dataset."""

        self._main_brain = brain if brain is not None else self._main_brain

        self.logger.info(f"Training on dataset with columns: {list(dataset.keys())}")

        for column_name, column_data in dataset.items():
            self.logger.debug(f"Column '{column_name}' has {len(column_data)} data points.")

        for step in range(steps):
            # Get the current data point from the dataset
            column_data = {key: dataset[key][step % len(dataset[key])] for key in dataset}
            # avoid index out of range by looping over dataset if steps exceed its size
            self.logger.debug(f"Step {step + 1}/{steps}, Data Point: {column_data}")

            # Prepare the input dictionary for the Brain
            input_dict = {}
            for field in self._trainer_input_fields:
                # Check if the field name exists in the data point and add it to the input dictionary
                if field.name in column_data:
                    input_dict[field.name] = column_data[field.name]
                else:
                    raise ValueError(f"Data point is missing required input field: {field.name}")

            # Step the Brain with the prepared inputs
            brain.step(inputs=input_dict, learn=True)

    def test(
        self, brain: Brain | None, dataset: dict[str, list[Any]], steps: int | None = None
    ) -> dict[str, Any]:
        """Replay inputs without learning and report prediction accuracy metrics."""

        if brain is None:
            brain = self._main_brain
        if brain is None:
            raise ValueError(self._BRAIN_NOT_INITIALIZED_ERROR)

        if not isinstance(dataset, dict) or not dataset:
            raise ValueError("Dataset must be a non-empty dict mapping field names to data.")

        steps = steps or min(len(values) for values in dataset.values())
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
            brain.step(inputs, learn=False)

            bursts = sum(
                len(column_field.bursting_columns) for column_field in brain._column_fields.values()
            )
            evaluation_bursts.append(bursts)

        flattened_errors = [err for errors in field_errors.values() for err in errors]
        mean_squared_error = (
            sum(flattened_errors) / len(flattened_errors) if flattened_errors else 0.0
        )
        avg_bursting_columns = (
            sum(evaluation_bursts) / len(evaluation_bursts) if evaluation_bursts else 0.0
        )
        total_failures = sum(prediction_failures.values())

        field_rows = []
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

        self.logger.info("\n".join(summary_lines))

        return {
            "mean_squared_error": mean_squared_error,
            "prediction_failures": prediction_failures,
            "total_prediction_failures": total_failures,
            "avg_bursting_columns": avg_bursting_columns,
            "evaluation_bursts": evaluation_bursts,
            "errors": field_errors,
        }

    def build_full_brain(
        self,
        dataset: dict[Any, list[Any]],
        size: int = 2048,
        params: el.ParameterMarker | None = None,
    ) -> Brain:
        """Build a full Brain with all fields based on the dataset.

        This method automatically determines the appropriate encoder type for each field
        based on the data type of the values in the dataset.

        Args:
            dataset: A dictionary mapping field names to lists of data points.
            size: The size of the input and column fields. Defaults to 2048.
            params: Optional encoder parameters to use for all fields. If not provided,
                default parameters will be used based on field type.

        Returns:
            A fully built Brain instance with configured input, column, and output fields.

        Raises:
            ValueError: If dataset contains unsupported data types.

        Example:
            brain = trainer.build_full_brain(
                dataset={
                    "temperature": [20.5, 21.0, 19.8, ...],
                    "humidity": [0.45, 0.50, 0.40, ...],
                    "date": [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3), ...],
                }, size=2048, params=RDSEParameters(size=2048, sparsity=0.02, resolution=0.01, category=False, seed=5)
            )

        This method automatically determines the appropriate encoder type for each field based
        on the data type of the values in the dataset. It then builds the Brain with input fields for each column,
        a column field that connects to all input fields, and output fields as needed.

        # TODO: add a dict of ParameterMarker-compatible configs to specify encoder parameters
        # for each field when building the brain, and use those parameters to build the brain with
        # the appropriate encoders for each field type. This allows for more flexible and customized brain building
        #
        # based on the dataset characteristics.

        Mapping of data types to encoder parameters:
        - Numerical (int, float): el.RDSEParameters
        - Categorical (str): el.CategoryParameters
        - Date/Time (datetime): el.DateEncoderParameters
        - Spatial (list, np.ndarray): el.FourierEncoderParameters
        - Tuple of (numerical, numerical): el.GeospatialParameters
        - Other types will raise a ValueError indicating unsupported data type.

        """

        for key, values in dataset.items():
            if isinstance(values[0], (int, float)):
                encoder_params = (
                    el.RDSEParameters(
                        size=params.size,
                        sparsity=params.sparsity,
                        resolution=params.resolution,
                        category=False,
                        seed=params.seed,
                    )
                    if isinstance(params, el.RDSEParameters)
                    else el.RDSEParameters(size=size)
                )
            elif isinstance(values[0], str):
                encoder_params = (
                    el.CategoryParameters(
                        size=params.size,
                        category_list=list(set(values)),
                        rdse_used=params.rdse_used,
                    )
                    if isinstance(params, el.CategoryParameters)
                    else el.CategoryParameters(
                        size=size, category_list=list(set(values)), rdse_used=False
                    )
                )
            elif isinstance(values[0], (list, np.ndarray)):
                encoder_params = (
                    el.FourierEncoderParameters(
                        size=params.size,
                        frequency_ranges=params.frequency_ranges,
                        sparsity_in_ranges=params.sparsity_in_ranges,
                        resolutions_in_ranges=params.resolutions_in_ranges,
                        seed=params.seed,
                    )
                    if isinstance(params, el.FourierEncoderParameters)
                    else el.FourierEncoderParameters(size=size)
                )
            elif isinstance(values[0], datetime):
                encoder_params = (
                    el.DateEncoderParameters(
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
                    if isinstance(params, el.DateEncoderParameters)
                    else el.DateEncoderParameters(size=size)
                )
            elif (
                isinstance(values[0], tuple)
                and len(values[0]) >= 2
                and all(isinstance(v, (int, float)) for v in values[0])
            ):
                encoder_params = (
                    el.GeospatialParameters(
                        size=params.size,
                        max_radius=params.max_radius,
                        scale=params.scale,
                        timestep=params.timestep,
                        use_altitude=params.use_altitude,
                    )
                    if isinstance(params, el.GeospatialParameters)
                    else el.GeospatialParameters(size=size)
                )
            else:
                raise ValueError(f"Unsupported data type for field '{key}': {type(values[0])}")

            self._setup_io_fields([(f"{key}_input", size, encoder_params)])

        self._setup_column_fields(num_columns=size, cells_per_column=32)

        brain = self._create_brain()

        self._main_brain = brain
        self._brains.append(brain)

        return brain

    def show_active_columns(self, brain: Brain, dataset_name: str | None = None) -> None:
        """Show the active columns in the Brain."""

        import htmrl.grapher as _grapher

        _grapher.show_active_columns(brain, dataset_name)

    def show_heat_map(self, brain: Brain, dataset_name: str | None = None) -> None:
        """Show a heat map of the Brain's column duty cycle activity."""

        import htmrl.grapher as _grapher

        _grapher.show_heat_map(brain, dataset_name)

    def save_brain(self, brain: Brain, filename: str | Path) -> None:
        """Persist a trained Brain (weights, encoders, column state) to disk.

        Data is written with :mod:`pickle`. Only load files you produced yourself;
        unpickling untrusted files can execute arbitrary code.

        Args:
            brain: Brain to save.
            filename: Destination path; parent directories are created if missing.
        """

        path = Path(filename).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(brain, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_brain(self, filename: str | Path, *, set_as_main: bool = True) -> Brain:
        """Load a Brain previously saved with :meth:`save_brain`.

        Args:
            filename: Path to the saved file.
            set_as_main: If True, set :attr:`main_brain` and refresh trainer field
                lists so :meth:`train_full_brain`, :meth:`train_column`, and
                :meth:`test` run against this Brain.

        Returns:
            The deserialized Brain.

        Raises:
            TypeError: If the file does not unpickle to a :class:`Brain` instance.
        """

        path = Path(filename).expanduser()
        with path.open("rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, Brain):
            raise TypeError(f"Expected a Brain in {path}, got {type(loaded).__name__}")
        if set_as_main:
            self.main_brain = loaded
            self._sync_trainer_fields_from_brain(loaded)
        return loaded

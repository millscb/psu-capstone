from htmrl.log import get_logger

"""Field manager for simplified HTM input handling.

Encapsulates InputFields and ColumnFields to provide a unified API
for encoding inputs and computing temporal memory in a single step.

    HTM implementation for column spatial pooling and temporal memory.
    inspired by Sungar Thesis: http://etd.lib.metu.edu.tr/upload/12621275/index.pdf

    Developed by: Dr. Pullin Agrawal Penn State University, Capstone Advisor

"""

from typing import Any
from uuid import uuid4

from htmrl.agent_layer.abstract_brain import AbstractBrain
from htmrl.agent_layer.HTM import ColumnField as LegacyColumnField
from htmrl.agent_layer.HTM import InputField as LegacyInputField
from htmrl.agent_layer.HTM import OutputField as LegacyOutputField
from htmrl.agent_layer.pullin.field_base import Field
from htmrl.agent_layer.pullin.pullin_htm import ColumnField, InputField, OutputField
from htmrl.log import LoggerManager


class Brain(AbstractBrain):
    CONF_SUFFIX = ".conf"
    """Manages HTM input fields and column fields with a unified API.

    Allows binding named inputs to InputFields and processing all inputs
    with a single `step()` call instead of manually calling encode/compute.

    Args:
        fields: Optional dictionary of named Field instances to initialize with.
            Can include InputField, OutputField, and ColumnField types.
        brain_id: To save the id of the brain.

    Example:
        manager = Trainer()
        manager.add_input_field("consumption", consumption_field)
        manager.add_input_field("date", date_field)
        manager.set_column_field(column_field)

        # Training loop - single call replaces manual encode/compute
        for row in data:
            result = manager.step({
                "consumption": row["value"],
                "date": row["timestamp"],
            })
    """

    def __init__(
        self,
        fields: dict[str, Field] | None = None,
        brain_id: str | None = None,
    ) -> None:

        if fields is None:
            fields = {}

        self.brain_id = brain_id or str(uuid4())

        # Separate fields into input, output, and column fields for easy access
        # ensure that values are instances of the correct type
        self._input_fields: dict[str, InputField] = {
            k: v  # type: ignore[assignment]
            for k, v in fields.items()
            if isinstance(v, (InputField, LegacyInputField))
        }
        self._output_fields: dict[str, OutputField] = {
            k: v  # type: ignore[assignment]
            for k, v in fields.items()
            if isinstance(v, (OutputField, LegacyOutputField))
        }
        self._column_fields: dict[str, ColumnField] = {
            k: v  # type: ignore[assignment]
            for k, v in fields.items()
            if isinstance(v, (ColumnField, LegacyColumnField))
        }
        self.fields = fields
        self.logger = get_logger(self)

        self.logger.info("Brain initialized with fields: %s", list(fields.keys()))

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def __getattr__(self, name: str) -> Field:
        # Use __getattribute__ for `fields` so unpickling and edge cases do not recurse
        # via attribute lookup on a half-restored instance.
        fields = object.__getattribute__(self, "fields")
        if name in fields:
            return fields[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def step(
        self,
        inputs: dict[str, Any],
        learn: bool = True,
    ):
        """Process one timestep: encode all inputs and compute column field.

        Args:
            inputs: Dict mapping field names to input values.
            learn: Whether to enable learning during this step.

        Returns:
            Dict mapping output field names to their decoded predictions.
        """
        if learn:
            self.logger.info("Processing step with inputs: %s", inputs)

        self.encode_only(inputs)
        self.compute_only(learn=learn)
        return {name: field.decode() for name, field in self._output_fields.items()}

    def prediction(self) -> tuple[Any, ...]:
        """Get the current prediction for a specific input field.

        Returns:
            Result from decoder (typically value, confidence tuple).
        """
        predictions = {}
        for input_name in self._input_fields:
            input_field = self._input_fields[input_name]
            if hasattr(input_field.encoder, "decode"):
                try:
                    predictions[input_name], predictions[input_name + self.CONF_SUFFIX] = input_field.decode(  # type: ignore
                        "predictive"
                    )
                    self.logger.info(
                        "Decoded SDR into value: %s, with confidence: %s",
                        predictions[input_name],
                        predictions[input_name + self.CONF_SUFFIX],
                    )
                except ValueError:
                    predictions[input_name] = None
                    predictions[input_name + self.CONF_SUFFIX] = 0.0

        return predictions  # type: ignore

    def rl_policy_update(self, reward: float | None = None) -> None:
        """RL policy prediction method that returns the next action based on reward fields."""
        # TODO: Brain needs fully functioning RL reward fields. (GO, NO-GO)
        pass

    def encode_only(self, inputs: dict[str, Any]) -> None:
        """Encode inputs without computing (useful for getting predictions first).

        Args:
            inputs: Dict mapping field names to input values.

        Raises:
            KeyError: If input field name is not registered.
        """
        for name, value in inputs.items():
            if name not in self._input_fields:
                raise KeyError(f"Unknown input field: '{name}'")
            self._input_fields[name].encode(value)

    def compute_only(self, learn: bool = True) -> None:
        """Compute column field without encoding (inputs already encoded).

        Args:
            learn: Whether to enable learning during this step.
        """
        # Compute temporal memory
        for column_field in self._column_fields.values():
            column_field.compute(learn=learn)

    def print_stats(self) -> None:
        """Print statistics from the column field."""
        for column_field in self._column_fields.values():
            self.logger.info("Statistics for ColumnField '%s':", column_field.name)
            column_field.print_stats()

    def reset(self) -> None:
        """Clear all states in the column field."""
        for field in self.fields.values():
            field.reset()  # type: ignore

    @property
    def input_fields(self) -> list[InputField]:
        """Return list of input fields."""
        return list(self._input_fields.values())

    @property
    def column_fields(self) -> list[ColumnField]:
        """Return list of column fields."""
        return list(self._column_fields.values())

    @property
    def output_fields(self) -> list[OutputField]:
        """Return list of output fields."""
        return list(self._output_fields.values())

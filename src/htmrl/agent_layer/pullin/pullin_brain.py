"""Field manager for simplified HTM input handling.

Encapsulates InputFields and ColumnFields to provide a unified API
for encoding inputs and computing temporal memory in a single step.
"""

from typing import Any

from htmrl.agent_layer.pullin.pullin_htm import ColumnField, Field, InputField, OutputField
from htmrl.agent_layer.pullin.sungur import ValueField
from htmrl.log import get_logger


class Brain:
    """Manages HTM input fields and column fields with a unified API.

    Allows binding named inputs to InputFields and processing all inputs
    with a single `step()` call instead of manually calling encode/compute.

    Example:
        manager = Brain(fields={})
        manager.add_input_field("consumption", consumption_field)
        manager.add_input_field("date", date_field)
        manager.set_column_field(column_field)

        # Training loop - single call replaces manual encode/compute
        for row in data:
            result = manager.step({
                "consumption": row["value"],
                "date": row["timestamp"],
            })

    Args:
        fields: Mapping of field names to field objects managed by the brain.
    """

    def __init__(self, fields: dict[str, Field] | None = None) -> None:
        fields = fields or {}
        self._output_fields: dict[str, OutputField] = {
            k: v for k, v in fields.items() if isinstance(v, OutputField)
        }
        self._input_fields: dict[str, InputField] = {
            k: v
            for k, v in fields.items()
            if isinstance(v, InputField) and not isinstance(v, OutputField)
        }
        self._value_fields: dict[str, ValueField] = {
            k: v for k, v in fields.items() if isinstance(v, ValueField)
        }
        self._column_fields: dict[str, ColumnField] = {
            k: v
            for k, v in fields.items()
            if isinstance(v, ColumnField) and not isinstance(v, ValueField)
        }
        self.fields = fields

        # Combined dicts for rendering — all fields that share the same visual shape
        self.all_column_fields: dict[str, ColumnField] = {
            k: v for k, v in fields.items() if isinstance(v, ColumnField)
        }
        self.all_input_fields: dict[str, InputField] = {
            k: v
            for k, v in fields.items()
            if isinstance(v, InputField) and not isinstance(v, ColumnField)
        }
        self._logger = get_logger(self)

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def __getattr__(self, name: str) -> Field:
        if name in self.fields:
            return self.fields[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def step(
        self,
        inputs: dict[str, Any],
        learn: bool = True,
        reward: float | None = None,
    ):
        """Process one timestep: encode all inputs and compute column field.

        Args:
            inputs: Dict mapping field names to input values.
            learn: Whether to enable learning during this step.
            reward: External reward signal. If None, reward is computed internally.

        Returns:
            Behavior mapping produced by the output fields for this timestep.
        """
        logged_reward = inputs.get("reward", reward)
        if logged_reward is None:
            logged_reward = "pending"
        self._logger.info(f"Brain.step inputs={inputs} learn={learn} reward={logged_reward}")
        self.encode_only(inputs)
        self.compute_only(learn=learn)
        if "reward" in inputs:
            reward = inputs["reward"]
        if reward is not None:
            self.estimate_value(reward)
        self.activate_apical_segments()
        behavior = self.generate_behavior()
        self._logger.debug(f"generate_behavior() output: {behavior}")
        return behavior

    def rl_policy_update(self, reward: float | None = None) -> None:
        """
        Update all ValueFields using TD error propagation.

        Args:
            reward: The reward signal to use for TD update. If None, attempts to use the most recent reward.
        """
        if reward is None:
            # Try to get reward from a known field, fallback to 0.0
            reward = getattr(self, "_last_reward", 0.0)
        for name, value_field in self._value_fields.items():
            value_field.update_values(reward)

    def estimate_value(self, reward: float | None = None) -> None:
        for name, field in self._value_fields.items():
            if reward is None:
                reward = self.compute_intrinsic_reward()
            field.update_values(reward)

    def activate_apical_segments(self) -> None:
        """Activate apical segments in Go/NoGo fields based on current column states."""
        for field in self.fields:
            if isinstance(self.fields[field], ColumnField):
                self.fields[field].apical_compute()

    def generate_behavior(self) -> dict[str, Any]:
        """Compute output fields and return action dicts for agent compatibility.

        Returns:
            Dict mapping output field names to dicts with 'action' and 'confidence' keys.
        """
        for field in self._output_fields.values():
            field.compute()
        behavior = {}
        for name, field in self._output_fields.items():
            result = field.decode()
            behavior[name] = {"action": result["value"], "confidence": 1.0}
            self._logger.debug(f"Output field '{name}': action={result['value']}")
        return behavior

    def encode_only(self, inputs: dict[str, Any]) -> None:
        """Encode inputs without computing (useful for getting predictions first).

        Args:
            inputs: Dict mapping field names to input values.
        """
        for name in self._input_fields:
            if name in inputs:
                self._input_fields[name].encode(inputs[name])

    def compute_only(self, learn: bool = True) -> None:
        """Compute column field without encoding (inputs already encoded).

        Args:
            learn: Whether to enable learning during this step.
        """
        for field in self._column_fields.values():
            field.compute(learn=learn)

    def compute_intrinsic_reward(self) -> float:
        """Compute an intrinsic reward signal"""
        raise NotImplementedError("Needs Implementation")

    def print_stats(self) -> None:
        """Print statistics from the column field."""
        for name, column_field in self._column_fields.items():
            print(f"Statistics for ColumnField '{name}':")
            column_field.print_stats()

    def reset(self) -> None:
        """Clear all states in the column field."""
        for field in self.fields:
            self.fields[field].reset()

    def prediction(self) -> tuple[Any, ...]:
        """Get the current prediction for all input and output fields.

        Returns:
            Dict with predictions for input fields and output fields (actions).
        """
        predictions = {}
        # Input field predictions (if any)
        for input_name, input_field in self._input_fields.items():
            if hasattr(input_field.encoder, "decode"):
                try:
                    predictions[input_name], predictions[input_name + ".conf"] = input_field.decode(
                        "predictive"
                    )
                except Exception:
                    pass
        # Output field predictions (actions)
        for output_name, output_field in self._output_fields.items():
            # Ensure key starts with 'action' for agent compatibility
            if output_name.startswith("action"):
                action_key = output_name
            else:
                action_key = f"action_{output_name}"
            try:
                result = output_field.decode()
                predictions[action_key] = result.get("value", None)
                predictions[action_key + ".conf"] = result.get("confidence", 1.0)
            except Exception:
                predictions[action_key] = None
                predictions[action_key + ".conf"] = 0.0
        return predictions

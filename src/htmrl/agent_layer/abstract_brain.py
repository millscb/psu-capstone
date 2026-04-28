from abc import ABC, abstractmethod
from typing import Any


class AbstractBrain(ABC):
    """Abstract base class for HTM brains."""

    def __init__(self):
        pass

    @abstractmethod
    def step(self, inputs: dict[str, Any]):
        """Perform a single step with the given inputs.

        Args:
            inputs: Dictionary of input values.
        """
        raise NotImplementedError("Subclasses must implement the step method")

    @abstractmethod
    def prediction(self) -> tuple[Any, ...]:
        """Get the current predictions.

        Returns:
            Tuple of predictions.
        """
        raise NotImplementedError("Subclasses must implement the prediction method")

    @abstractmethod
    def encode_only(self, inputs: dict[str, Any]):
        """Encode inputs without computing.

        Args:
            inputs: Dictionary of input values.
        """
        raise NotImplementedError("Subclasses must implement the encode only method")

    @abstractmethod
    def compute_only(self, learn: bool = True):
        """Perform computation without encoding.

        Args:
            learn: Whether to perform learning.
        """
        raise NotImplementedError("Subclasses must implement the compute only method")

    @abstractmethod
    def print_stats(self):
        """Print statistics about the brain."""
        raise NotImplementedError("Subclasses must implement the print states method")

    @abstractmethod
    def reset(self):
        """Reset the brain to initial state."""
        raise NotImplementedError("Subclasses must implement the reset method")

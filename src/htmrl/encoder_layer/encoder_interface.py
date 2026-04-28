"""Protocol definitions for encoder layer components.

This module defines the interface contract that all encoder implementations
must satisfy, using Python's Protocol for structural subtyping.
"""

from typing import Any, Iterable, Protocol, runtime_checkable


@runtime_checkable
class EncoderInterface(Protocol):
    """Protocol defining the interface for encoder layer components.

    Encoders implementing this protocol convert input values into sparse
    distributed representations. The protocol uses structural subtyping,
    so any class with matching method signatures automatically satisfies it.
    """

    def encode(self, input_value: Any) -> list[int]:
        """Encode a single input value into an SDR.

        Transforms the input value into a sparse distributed representation
        expressed as a list of active bit indices.

        Args:
            input_value: The value to encode.

        Returns:
            List of active bit indices representing the encoded value.
        """
        ...

    def decode(
        self, encoded: list[int], candidates: Iterable[float] | None = None
    ) -> tuple[float | None, float]:
        """Decode an SDR back into its original input value.

        Transforms a sparse distributed representation back into the original
        input value.

        Args:
            encoded: List of active bit indices representing the encoded value.
            candidates: Optional iterable of candidate values for decoding.

        Returns:
            A tuple containing the decoded value (or None if decoding fails) and a confidence score.
        """
        ...

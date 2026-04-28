"""Build and return an EncoderLayer based on the specified type."""

from __future__ import annotations

from typing import Any

import htmrl.encoder_layer as en


class EncoderFactory:
    """Factory class to create encoders based on type.

    Maps encoder types to their corresponding classes and parameter dataclasses.
    The factory constructs parameter objects from dictionaries before instantiating encoders.
    """

    def __init__(self) -> None:
        self._encoder = None

    # Mapping of encoder type to (EncoderClass, ParameterClass)
    ENCODER_MAP: dict[str, tuple[type, type]] = {
        "category": (en.CategoryEncoder, en.CategoryParameters),
        "new_category": (en.CategoryEncoderNew, en.CategoryParametersNew),
        "date": (en.DateEncoder, en.DateEncoderParameters),
        "rdse": (en.RandomDistributedScalarEncoder, en.RDSEParameters),
        "scalar": (en.ScalarEncoder, en.ScalarEncoderParameters),
        "fourier": (en.FourierEncoder, en.FourierEncoderParameters),
        "geospatial": (en.GeospatialEncoder, en.GeospatialParameters),
        "coordinate": (en.CoordinateEncoder, en.CoordinateParameters),
        "delta": (en.DeltaEncoder, en.DeltaEncoderParameters),
    }

    @staticmethod
    def create_encoder(encoder_type: str, parameters: dict[str, Any] | None = None) -> object:
        """Create an encoder based on the specified type and parameters.

        Args:
            encoder_type: The type of encoder (e.g., 'rdse', 'category', 'scalar', 'date', 'fourier').
            parameters: Dictionary of parameters to pass to the encoder's parameter dataclass.

        Returns:
            An instantiated encoder object.

        Raises:
            ValueError: If encoder_type is not supported.
        """

        encoder_type = encoder_type.lower()

        if encoder_type not in EncoderFactory.ENCODER_MAP:
            raise ValueError(
                f"Unsupported encoder type: '{encoder_type}'. "
                f"Supported types: {', '.join(EncoderFactory.ENCODER_MAP.keys())}"
            )

        encoder_class, param_class = EncoderFactory.ENCODER_MAP[encoder_type]

        if not parameters:
            param_obj = param_class()  # Use default parameters if none provided
        else:
            param_obj = param_class(**parameters)

        return encoder_class(param_obj)


if __name__ == "__main__":

    # Test the factory with a sample encoder creation
    f = EncoderFactory()

    rdse_encoder = f.create_encoder("rdse")
    assert isinstance(rdse_encoder, en.RandomDistributedScalarEncoder)
    print("RDSE encoder created successfully with default parameters.")

"""Tests for structural ParameterMarker compatibility across encoder parameter classes."""

# Test Suite: TS-23 (Encoder-ParameterMarker)

import pytest

from htmrl.encoder_layer.base_encoder import ParameterMarker
from htmrl.encoder_layer.category_encoder import CategoryParameters
from htmrl.encoder_layer.coordinate_encoder import CoordinateParameters
from htmrl.encoder_layer.date_encoder import DateEncoderParameters
from htmrl.encoder_layer.delta_encoder import DeltaEncoderParameters
from htmrl.encoder_layer.fourier_encoder import FourierEncoderParameters
from htmrl.encoder_layer.geospatial_encoder import GeospatialParameters
from htmrl.encoder_layer.rdse import RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoderParameters


@pytest.mark.parametrize(
    "params",
    [
        CategoryParameters(),
        CoordinateParameters(),
        DateEncoderParameters(),
        DeltaEncoderParameters(),
        FourierEncoderParameters(),
        GeospatialParameters(),
        RDSEParameters(),
        ScalarEncoderParameters(),
    ],
    ids=[
        "category",
        "coordinate",
        "date",
        "delta",
        "fourier",
        "geospatial",
        "rdse",
        "scalar",
    ],
)
# Test Type: unit test
def test_encoder_parameter_is_parameter_marker(params):
    # TS-23 TC-204
    """All encoder parameter classes should satisfy the ParameterMarker protocol."""
    assert isinstance(params, ParameterMarker)

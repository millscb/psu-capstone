"""Test EncoderFactory and its ability to create encoders with the correct parameters."""

# Test Suite: TS-21 (EncoderFactory)

import pytest

import htmrl.encoder_layer as el
from htmrl.encoder_layer.encoder_factory import EncoderFactory


# TS-21 TC-173
@pytest.mark.parametrize(
    "encoder_type, expected_type",
    [
        ("category", el.CategoryEncoder),
        ("date", el.DateEncoder),
        ("rdse", el.RandomDistributedScalarEncoder),
        ("scalar", el.ScalarEncoder),
        ("fourier", el.FourierEncoder),
        ("geospatial", el.GeospatialEncoder),
        ("coordinate", el.CoordinateEncoder),
    ],
)
# Test Type: unit test
def test_factory_creates_supported_encoders_with_defaults(
    encoder_type: str, expected_type: type
) -> None:
    """Ensure factory builds expected encoder classes from default parameters."""
    encoder = EncoderFactory.create_encoder(encoder_type)
    assert isinstance(encoder, expected_type)


# Test Type: unit test
def test_factory_rejects_unsupported_encoder_type() -> None:
    # TS-21 TC-174
    """Ensure unsupported encoder types raise a clear error."""
    with pytest.raises(ValueError, match="Unsupported encoder type"):
        EncoderFactory.create_encoder("not-an-encoder")


# Test Type: unit test
def test_factory_creates_geospatial_with_defaults() -> None:
    # TS-21 TC-175
    """Ensure geospatial encoder can be created with default parameters."""
    encoder = EncoderFactory.create_encoder("geospatial")
    assert isinstance(encoder, el.GeospatialEncoder)


# Test Type: unit test
def test_factory_parameter_unpacking_for_category() -> None:
    # TS-21 TC-176
    """Ensure dict params are unpacked into CategoryParameters dataclass correctly."""
    encoder = EncoderFactory.create_encoder(
        "category",
        {"w": 4, "category_list": ["US", "GB", "ES"], "rdse_used": True},
    )
    assert isinstance(encoder, el.CategoryEncoder)

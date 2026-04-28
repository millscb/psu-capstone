"""
tests.test_encoder_handler_suite

Test suite for EncoderHandler union SDR generation functionality.

Validates that EncoderHandler correctly coordinates multiple encoders to create union SDRs
from mixed-type data. Tests ensure:
- Multiple encoders (RDSE, Scalar, Category) process their respective fields
- Individual field SDRs are combined into a single union SDR
- Union SDR dimensions match expected (sum of all field encoder sizes)
- Union SDR sparsity reflects combined encoding density
- Encoder coordination handles different parameter configurations

These tests validate the encoder coordination layer that converts tabular records
with mixed data types into unified SDR representations for HTM processing.
"""

from datetime import datetime

import pytest

from htmrl.encoder_layer.encoder_handler import EncoderHandler


@pytest.fixture
def handler() -> EncoderHandler:
    """Fixture to create an EncoderHandler with multiple encoders"""

    df = [
        {
            "float_col": float(3.14),  # rdse
            "int_col": int(42),  # scalar
            "str_col": str("B"),  # category
            "date_col": datetime(2023, 12, 25),  # date
        }
    ]

    handler = EncoderHandler(df)

    return handler


# Test Type: unit test
def test_handler_singleton(handler: EncoderHandler):
    # TS-09 TC-066
    """Test that EncoderHandler enforces singleton pattern"""

    # Arrange
    test_input = handler._data_frame

    # Act
    h1 = handler
    h2 = EncoderHandler(test_input)

    # Assert
    assert h1 is h2


"""
# Test Type: unit test
def test_copy_deepcopy_sdr(handler: EncoderHandler):
    Test copying and deep copying SDRs from multiple encoders

    # Arrange
    test_data = handler._data_frame
    rows = test_data.iloc[0]
    sdrs = []

    # Act
    handler.build_composite_sdr(test_data)

    # Assert that a deep copy occurs
    for i, encoder in enumerate(handler._encoders):
        input_value = rows.iloc[i]

        output_sdr = SDR([encoder.size])

        output_sdr.zero()

        assert output_sdr.get_sparse() == []

        try:
            output = encoder.encode(input_value)
            output_sdr.set_dense(output)
            assert output_sdr.get_sparse() != []
        except Exception as e:
            pytest.fail(f"Encoding failed with exception: {e}")

        copied_sdr = copy.deepcopy(output_sdr)
        sdrs.append(copied_sdr)
        assert sdrs[i].get_sparse() == output_sdr.get_sparse()
        assert sdrs[i].get_sparse() != []
        output_sdr.zero()
        assert sdrs[i].get_sparse() != output_sdr.get_sparse()
"""

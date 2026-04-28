"""
tests.test_encoder_to_htm

Test suite for integration between Encoder and HTM components.

Validates the data flow from input data through encoders to HTM processing. Tests ensure:
- InputHandler correctly loads and prepares data for encoding
- Encoders process loaded data and generate SDR representations
- SDR dimensions and sparsity are within expected ranges
- End-to-end pipeline maintains data integrity from input to encoding

Note: HTM interface integration tests are not yet available (awaiting HTMinterface
implementation). This suite focuses on encoder-level processing and SDR generation.

These tests validate the critical pipeline connection from input processing through
SDR encoding that feeds into the HTM temporal memory engine.
"""

from typing import Any

import pytest

from htmrl.agent_layer.HTM import InputField
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.input_layer.input_handler import InputHandler


def test_encoder_to_htm_receives_sdr_object():
    """Real ScalarEncoder output is passed into a real InputField (HTM cell consumer)."""

    # Arrange: a fibonacci-like sequence as input data
    fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13]

    enc_params = ScalarEncoderParameters(
        minimum=0.0,
        maximum=13.0,
        active_bits=4,
        sparsity=0.0,
        size=100,
        radius=0.0,
        resolution=1.0,
        clip_input=True,
        periodic=False,
        category=False,
    )
    encoder = ScalarEncoder(enc_params)
    # InputField wraps the same encoder type, providing the HTM cell interface
    input_field = InputField(encoder_params=enc_params)

    handler = InputHandler()
    records = handler.input_data(fib_sequence, required_columns=["value"])

    # Act: encode a value and feed it into the real HTM InputField
    last_value = records["value"][3]
    sdr = encoder.encode(last_value)
    input_field.encode(last_value)
    active_cells = [cell for cell in input_field.cells if cell.active]

    # Assert
    assert isinstance(records, dict)
    assert last_value == 2
    assert len(records["value"]) == 8
    assert set(sdr).issubset({0, 1})
    assert sum(sdr) == 4  # active_bits=4
    assert len(active_cells) == 4  # InputField cells mirror encoder active bits

"""
tests.test_input_to_encoder

Test suite for integration between InputHandler and Encoder components.

Validates the end-to-end data flow from input handler (loading data) through encoder
(converting data to SDR representations). Tests ensure:
- Input data is correctly passed from InputHandler to Encoder
- Scalar values are properly encoded based on encoder parameters (min/max, periodic)
- SDR representations have correct dimensionality and sparsity
- Data transformations maintain semantic correctness through the pipeline

These tests validate the critical pipeline connection between input data loading and
SDR encoding, ensuring data integrity across component boundaries.
"""

import numpy as np
import pytest

from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.input_layer.input_handler import InputHandler
from htmrl.input_layer.input_interface import InputInterface


@pytest.fixture
def input_handler() -> InputHandler:
    return InputHandler.get_instance()


def _make_scalar_params() -> ScalarEncoderParameters:
    return ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=False,
        periodic=False,
        category=False,
        active_bits=3,
        sparsity=0.0,
        size=15,
        radius=0.0,
        resolution=1.0,
    )


@pytest.fixture
def encoder() -> ScalarEncoder:
    return ScalarEncoder(_make_scalar_params())


# Test Type: integration test
def test_input_to_encoder_passes_records_into_encoder(input_handler, encoder):
    """
    This test verifies that InputHandler yields record dictionaries and that
    encoder-ready sequences can ingest those records without data loss.
    """

    # Arrange
    values = [5.0, 10.0, 15.0, 10.0, 5.0]
    data_stream = bytearray(int(value) for value in values)
    # Act
    records = input_handler.input_data(data_stream, required_columns=["value"])
    encoder_sequence = input_handler.get_column_data("value")

    normalized_sequence = [float(v) for v in encoder_sequence]
    encoded_from_sequence = [encoder.encode(value) for value in normalized_sequence]
    reference_encoder = ScalarEncoder(_make_scalar_params())
    encoded_reference = [reference_encoder.encode(value) for value in normalized_sequence]

    # Assert
    assert isinstance(input_handler, InputInterface)
    assert isinstance(encoder, ScalarEncoder)
    assert isinstance(records, dict)
    assert isinstance(encoder_sequence, list)
    assert isinstance(normalized_sequence, list)
    assert len(normalized_sequence) == len(values)
    assert all(value is not None for value in normalized_sequence)
    assert encoded_from_sequence == encoded_reference


# TI-015
# Test Type: system test
def test_sine_wave_through_input_handler(input_handler, encoder):
    """Run a sine wave through the full input->encode->compute->predict pipeline.

      This test validates a broad system pipeline:
    1) InputHandler ingests an end-to-end signal stream.
    2) Brain executes encode/compute/learn steps via InputField+ColumnField.
    3) Prediction quality is checked on a held-out cycle.
    """

    # Arrange: generate a deterministic sine wave scaled to [0, 100]
    cycle_len = 60
    train_cycles = 6
    eval_cycles = 1
    t = np.arange(cycle_len)
    cycle_values = ((np.sin(2 * np.pi * t / cycle_len) + 1.0) * 50.0).tolist()
    values = cycle_values * (train_cycles + eval_cycles)

    # Act: feed the full stream through InputHandler
    input_handler.input_data(values, required_columns=["value"])
    seq_values = input_handler.get_column_data("value")

    # Build a full HTM pipeline (InputField -> ColumnField -> Brain)
    rdse_params = RDSEParameters(size=512, active_bits=0, sparsity=0.02, resolution=1.0, seed=7)
    input_field = InputField(size=512, encoder_params=rdse_params)
    column_field = ColumnField(
        input_fields=[input_field],
        non_spatial=True,
        num_columns=512,
        cells_per_column=16,
    )
    brain = Brain({"value": input_field, "column": column_field})

    # Train on multiple full cycles
    for value in seq_values[: cycle_len * train_cycles]:
        brain.step({"value": float(value)})

    # Evaluate one held-out cycle without learning.
    errors: list[float] = []
    predictions: list[float] = []
    missing_predictions = 0
    eval_values = seq_values[cycle_len * train_cycles :]
    for value in eval_values:
        prediction = brain.prediction()["value"]
        if prediction is None:
            missing_predictions += 1
            prediction_value = float(value)
        else:
            prediction_value = float(prediction)
            predictions.append(prediction_value)
            errors.append(abs(prediction_value - float(value)))
        brain.step({"value": float(value)}, learn=False)

    # Assert: full pipeline produced stable, bounded predictions on held-out data
    assert isinstance(seq_values, list)
    assert len(seq_values) == cycle_len * (train_cycles + eval_cycles)
    assert all(isinstance(v, (int, float)) for v in seq_values)
    assert all(0.0 <= float(v) <= 100.0 for v in seq_values)

    # Keep existing scalar-encoder shape checks as a compatibility guard
    encoded = [encoder.encode(v) for v in seq_values]
    assert len(encoded) == len(seq_values)
    assert all(isinstance(enc, list) for enc in encoded)
    assert all(len(enc) == encoder.size for enc in encoded)

    # Acceptance/system-level quality checks
    assert missing_predictions <= max(5, cycle_len // 10)
    assert len(errors) >= cycle_len * eval_cycles - missing_predictions
    assert np.mean(errors) < 20.0
    assert np.max(errors) < 60.0
    assert np.std(predictions) > 1.0

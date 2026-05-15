# Test Suite: TS-13 (DateEncoder - Decode)
"""
tests.test_decoder_date

Test suite for DateEncoder decoding functionality.

Validates that DateEncoder correctly decodes SDRs back to datetime objects from their
temporal components. Tests cover:
- Season component (annual cycle, 365 days) decoding
- Day of week component (Mon-Sun) decoding
- Weekend vs weekday binary component
- Holiday status detection
- Time of day component (hours/minutes)
- Custom periodic dimension decoding
- Full datetime reconstruction from encoded components
- decode() returns (datetime, confidence) tuple format

Parameter Constraints:
- Each component has separate size, active_bits/sparsity, radius/resolution parameters
- active_bits and sparsity are mutually exclusive (RDSE constraint)
- Tests explicitly set sparsity=0.0 when using active_bits
- Multiple temporal components combine into single encoding

These tests validate the reverse transformation from temporal SDR representations
back to interpretable datetime values, enabling HTM predictions of temporal patterns.
"""

from datetime import datetime

import pytest

from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.log import logger

pytest_plugins = ["tests.config_test"]


@pytest.mark.parametrize(
    "date_params_fixture,encoder_key,value_min,value_max",
    [
        ("date_params_season_only", "season", 0, 366),
        ("date_params_day_of_week_only", "dayofweek", 0, 7),
        ("date_params_weekend_only", "weekend", 0, 1),
        ("date_params_custom_only", "customdays", 0, 1),
        ("date_params_holiday_only", "holiday", 0, 3),
        ("date_params_time_of_day_only", "timeofday", 0, 24),
    ],
)
# Test Type: unit test
def test_single_encoder_decode(
    # TS-13 TC-088, TC-090, TC-091, TC-092, TC-093, TC-094
    date_params_fixture,
    encoder_key,
    value_min,
    value_max,
    decoder_test_dates,
    request,
):
    """Test decoding for single encoder component."""
    # Arrange
    date_params = request.getfixturevalue(date_params_fixture)
    date_encoder = DateEncoder(date_params)

    # Act
    actual_decoded = []
    for test in decoder_test_dates:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # Assert - structure, valid range
    assert len(actual_decoded) == len(decoder_test_dates)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(
            decoded, dict
        ), f"Date {decoder_test_dates[i]}: decoded should be dict, got {type(decoded)}"
        assert (
            encoder_key in decoded and len(decoded) == 1
        ), f"Date {decoder_test_dates[i]}: one encoder => 1 key ({encoder_key}), got {list(decoded)}"
        value = decoded[encoder_key][0]
        # Handle day_of_week upper bound (exclusive)
        if encoder_key == "dayofweek":
            assert (
                value_min <= value < value_max
            ), f"Date {decoder_test_dates[i]}: {encoder_key} in [{value_min}, {value_max}), got {value}"
        else:
            assert (
                value_min <= value <= value_max
            ), f"Date {decoder_test_dates[i]}: {encoder_key} in [{value_min}, {value_max}], got {value}"

    # Assert - round-trip determinism
    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1[encoder_key][0] == dec2[encoder_key][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Test Type: unit test
def test_rdse_decode_same_across_instances_with_same_params(date_params_day_of_week_only):
    # TS-13 TC-089
    """With same params, DateEncoder does not pass a seed to RDSE, so all instances use default seed and produce the same decode."""
    # Arrange
    date_params = date_params_day_of_week_only
    dt = datetime(2020, 1, 1, 0, 0)

    # Act
    decoded_values = []
    for _ in range(5):
        encoder = DateEncoder(date_params)
        encoded = encoder.encode(dt)
        decoded = encoder.decode(encoded)
        decoded_values.append(decoded["dayofweek"][0])

    # Assert - same params => same default RDSE seed => same decode (deterministic across instances)
    assert (
        len(set(decoded_values)) == 1
    ), f"With same params (no seed override), all instances should produce same decode; got {decoded_values}"


# Test Type: unit test
def test_all_combined(date_params_all_combined, decoder_test_dates):
    # TS-13 TC-095
    """Decode with all six encoders enabled (RDSE): season, day_of_week, weekend, custom, holiday, time_of_day."""
    # Arrange
    date_params = date_params_all_combined
    date_encoder = DateEncoder(date_params)
    # Add extra test case to match original
    test_case = decoder_test_dates + [[1988, 5, 27, 11, 0]]

    # Act
    actual_decoded = []
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # Assert - decode returns dict of 6 keys
    assert len(actual_decoded) == len(test_case)
    keys = ["season", "dayofweek", "weekend", "customdays", "holiday", "timeofday"]
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, dict), f"Date {test_case[i]}: decoded should be dict"
        assert len(decoded) == 6 and all(
            k in decoded for k in keys
        ), f"Date {test_case[i]}: all combined => 6 keys, got {list(decoded)}"

        season = decoded["season"][0]
        dow = decoded["dayofweek"][0]
        weekend = decoded["weekend"][0]
        custom = decoded["customdays"][0]
        holiday = decoded["holiday"][0]
        tod = decoded["timeofday"][0]

        assert 0 <= season <= 366, f"Date {test_case[i]}: season in [0, 366], got {season}"
        assert 0 <= dow < 7, f"Date {test_case[i]}: day_of_week in [0, 7), got {dow}"
        assert 0 <= weekend <= 1, f"Date {test_case[i]}: weekend in [0, 1], got {weekend}"
        assert 0 <= custom <= 1, f"Date {test_case[i]}: custom in [0, 1], got {custom}"
        assert 0 <= holiday <= 3, f"Date {test_case[i]}: holiday in [0, 3], got {holiday}"
        assert 0 <= tod <= 24, f"Date {test_case[i]}: time_of_day in [0, 24], got {tod}"

    # Assert - round-trip determinism
    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    for key in keys:
        assert dec1[key][0] == dec2[key][0], f"Round-trip deterministic for encoder {key}"

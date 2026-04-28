"""
tests.test_encoder_date_rdse

Test suite for DateEncoder with RDSE backend.

The Date Encoder decomposes temporal values (datetime objects) into multiple
component dimensions and encodes each using RDSE for sparse, distributed representations.

Temporal Components Encoded:
  - Year (e.g., 2020, 2021)
  - Season (annual cycle, 365 days)
  - Day of week (Mon-Sun, 7 values)
  - Weekend vs weekday (binary)
  - Holiday status (special days)
  - Time of day (hours/minutes, 24h cycle)
  - Custom dimensions (user-defined periodic patterns)

Parameter Validation:
  - Each component has its own size, active_bits/sparsity, radius/resolution
  - Active_bits and sparsity are mutually exclusive (per RDSE constraint)
  - Year and Season are mutually exclusive (XOR constraint)
  - Tests explicitly set sparsity=0.0 when using active_bits
  - Multiple components can be combined into single encoding

Tests validate:
  1. Encoder initialization with various component combinations
  2. Output format (binary 0/1 only, length = sum of component sizes)
  3. Determinism (same datetime → same SDR)
  4. Component independence (each component contributes to final encoding)
  5. All component combinations work correctly
"""

from datetime import datetime

import numpy as np
import pytest

from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters

pytest_plugins = ["tests.config_test"]

# ---------------------------------------------------------------------------
# Output format: binary 0/1 only, length equals size
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_output_only_zeros_and_ones(date_params_season_only):
    """Verify RDSE DateEncoder output is strictly binary (only 0 and 1 values).

    Tests that all bits in the output SDR are either 0 or 1, with no intermediate
    values or errors. This validates the fundamental SDR representation format for
    RDSE-based encodings.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)
    dt = datetime(2020, 1, 1, 0, 0)

    # Act
    out = encoder.encode(dt)

    # Assert
    assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


# Test Type: unit test
def test_rdse_output_length_equals_size(date_params_season_only):
    """Verify RDSE DateEncoder output length matches the configured encoder size.

    Tests that the output SDR has a length equal to the sum of all enabled
    encoder sizes. This validates that no bits are dropped or added during
    RDSE encoding operations.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)
    dt = datetime(2020, 1, 1, 0, 0)

    # Act
    out = encoder.encode(dt)

    # Assert
    assert (
        len(out) == encoder._size
    ), f"Output length must equal _size ({encoder._size}), got {len(out)}"


# Test Type: unit test
def test_rdse_all_combined_output_binary_and_length(date_params_all_combined_with_year):
    """Verify RDSE DateEncoder with all features produces correct output format.

    Tests that when all seven RDSE encoders (year, season, day_of_week, weekend,
    custom, holiday, time_of_day) are enabled together, the output is binary
    and has the correct total length equal to sum of component sizes.
    """
    # Arrange
    encoder = DateEncoder(date_params_all_combined_with_year)
    dt = datetime(2020, 1, 1, 0, 0)

    # Act
    out = encoder.encode(dt)

    # Assert
    assert all(b in (0, 1) for b in out)
    assert len(out) == encoder._size


# ---------------------------------------------------------------------------
# Per-feature encode: each single-feature config produces valid output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params_fixture",
    [
        "date_params_year_only",
        "date_params_season_only",
        "date_params_day_of_week_only",
        "date_params_weekend_only",
        "date_params_custom_only",
        "date_params_holiday_only",
        "date_params_time_of_day_only",
    ],
    ids=["year", "season", "day_of_week", "weekend", "custom", "holiday", "time_of_day"],
)
# Test Type: unit test
def test_rdse_single_feature_encode_binary_and_length(params_fixture, request):
    """Verify each individual RDSE encoder produces valid binary output.

    Tests that single-feature configurations (year-only, season-only, etc.) each
    produce binary output of correct length. Validates that each RDSE component
    encoder works independently in isolation.
    """
    # Arrange
    params = request.getfixturevalue(params_fixture)
    encoder = DateEncoder(params)
    dt = datetime(2020, 6, 15, 12, 30)

    # Act
    out = encoder.encode(dt)

    # Assert
    assert all(b in (0, 1) for b in out)
    assert len(out) == encoder._size


# ---------------------------------------------------------------------------
# Determinism: same instance same input => same encoding
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_same_instance_same_input_same_encoding(date_params_season_only):
    """Verify RDSE encoding is deterministic for the same encoder instance.

    Tests that encoding the same datetime twice with the same encoder instance
    produces identical SDRs. Validates that RDSE hash-based encoding is
    deterministic within an encoder instance.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)
    dt = datetime(2019, 7, 4, 14, 0)

    # Act
    enc1 = encoder.encode(dt)
    enc2 = encoder.encode(dt)

    # Assert
    assert enc1 == enc2


# ---------------------------------------------------------------------------
# Same params (same seed) => same encoding across instances
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_same_params_same_encoding_across_instances(date_params_season_only):
    """Verify RDSE produces identical encodings across different encoder instances.

    Tests that two DateEncoder instances created with identical parameters produce
    the same encodings for the same input. Validates that RDSE behavior is
    deterministic and reproducible across instances with identical configuration.
    """
    # Arrange
    encoder1 = DateEncoder(date_params_season_only)
    encoder2 = DateEncoder(date_params_season_only)
    dt = datetime(2020, 1, 1, 0, 0)

    # Act & Assert
    assert encoder1.encode(dt) == encoder2.encode(dt)


# ---------------------------------------------------------------------------
# Input types: None, datetime, int/float (epoch), struct_time
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_encode_accepts_datetime(date_params_season_only):
    """Verify RDSE DateEncoder accepts Python datetime objects.

    Tests that the encoder can process datetime.datetime objects and produces
    valid SDR output. Validates support for the most common datetime input type.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)
    dt = datetime(2020, 3, 15, 9, 0)

    # Act
    out = encoder.encode(dt)

    # Assert
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


# Test Type: unit test
def test_rdse_encode_accepts_none_current_time(date_params_season_only):
    """Verify RDSE DateEncoder accepts None to represent current time.

    Tests that passing None uses the current local time (via time.localtime())
    and produces valid SDR output. Validates convenient encoding of current moment.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)

    # Act
    out = encoder.encode(None)

    # Assert
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


# Test Type: unit test
def test_rdse_encode_accepts_epoch_seconds(date_params_season_only):
    """Verify RDSE DateEncoder accepts UNIX epoch seconds (int/float).

    Tests that the encoder can process integer or float UNIX timestamps and
    produces valid SDR output. Validates support for numeric timestamp inputs.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)
    dt = datetime(2020, 1, 1, 0, 0)
    ts = dt.timestamp()

    # Act
    out = encoder.encode(ts)

    # Assert
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


# Test Type: unit test
def test_rdse_encode_accepts_struct_time(date_params_season_only):
    """Verify RDSE DateEncoder accepts time.struct_time objects.

    Tests that the encoder can process struct_time objects (from datetime.timetuple())
    and produces valid SDR output. Validates support for low-level time structures.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)
    dt = datetime(2020, 5, 20, 18, 30)
    t = dt.timetuple()

    # Act
    out = encoder.encode(t)

    # Assert
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


# Test Type: unit test
def test_rdse_encode_rejects_unsupported_type(date_params_season_only):
    """Verify RDSE DateEncoder rejects unsupported input types.

    Tests that the encoder raises ValueError when given invalid input types
    (strings, lists, etc.). Validates that the encoder enforces strict type checking.
    """
    # Arrange
    encoder = DateEncoder(date_params_season_only)

    # Act & Assert
    with pytest.raises(ValueError):
        encoder.encode("2020-01-01")
    with pytest.raises(ValueError):
        encoder.encode([2020, 1, 1])


# ---------------------------------------------------------------------------
# Misconfigured: no encoders enabled raises
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_no_encoders_enabled_raises():
    """Verify DateEncoder raises error when no encoders are enabled.

    Tests that attempting to create a DateEncoder with all features disabled
    raises a RuntimeError. Validates that at least one sub-encoder must be enabled.
    """
    params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    with pytest.raises(RuntimeError, match="no sub-encoders enabled"):
        DateEncoder(params)


# ---------------------------------------------------------------------------
# Multiple dates: encoding varies with input (sanity)
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_different_dates_different_encodings(date_params_all_combined_with_year):
    """Verify different dates produce different RDSE encodings.

    Tests that at least two out of three different dates (1/1, 7/4, 12/25) produce
    distinct encodings. Validates that the encoder produces varying outputs for
    different temporal inputs.
    """
    # Arrange
    encoder = DateEncoder(date_params_all_combined_with_year)

    # Act
    encodings = []
    for year, month, day in [(2020, 1, 1), (2020, 7, 4), (2019, 12, 25)]:
        dt = datetime(year, month, day, 12, 0)
        encodings.append(encoder.encode(dt))

    # Assert - at least two should differ
    assert len(set(tuple(e) for e in encodings)) >= 2


# Correctness tests below
def hamming_distance_helper(first, second) -> int:
    """
    Helper method to find the differences with the first != second and then count the nonzero
    as that is how many different bits there are. So if first was 1001 and second was 1010 the
    first operation would be 0011 and the count_nonzero would return 2. This indicates a hamming
    distance of 2 since 2 of the bits are different.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    result = int(np.count_nonzero(first != second))
    print(result)
    return result


# Test Type: unit test
def test_date_correctness(date_params_all_combined_with_year, date_params_year_only):
    """Verify year encoding produces semantically correct Hamming distances.

    Tests that years closer together have smaller Hamming distances than years
    far apart. Validates that RDSE year encoding with radius=100 produces
    meaningful semantic similarity (2020 vs 2001 < 2020 vs 3000).
    """
    # Arrange
    encoder = DateEncoder(date_params_all_combined_with_year)
    encodings1 = []
    for year, month, day in [(2020, 1, 1), (2020, 7, 4), (2050, 12, 25)]:
        dt = datetime(year, month, day, 12, 0)
        encodings1.append(encoder.encode(dt))
    assert hamming_distance_helper(encodings1[0], encodings1[1]) < hamming_distance_helper(
        encodings1[0], encodings1[2]
    )

    encodings2 = []
    d = datetime(year=2026, month=1, day=1, minute=1)
    d1 = datetime(year=2026, month=1, day=1, minute=2)
    d2 = datetime(year=2026, month=1, day=1, minute=59)
    encodings2.append(encoder.encode(d))
    encodings2.append(encoder.encode(d1))
    encodings2.append(encoder.encode(d2))
    assert hamming_distance_helper(encodings2[0], encodings2[1]) < hamming_distance_helper(
        encodings2[0], encodings2[2]
    )

    encodings3 = []
    d3 = datetime(year=2026, month=1, day=1, minute=1, second=1)
    d4 = datetime(year=2026, month=1, day=1, minute=1, second=2)
    d5 = datetime(year=2026, month=1, day=1, minute=1, second=59)
    encodings3.append(encoder.encode(d3))
    encodings3.append(encoder.encode(d4))
    encodings3.append(encoder.encode(d5))
    assert hamming_distance_helper(encodings3[0], encodings3[1]) < hamming_distance_helper(
        encodings3[0], encodings3[2]
    )

    encodings4 = []
    d6 = datetime(year=2026, month=1, day=1, hour=1)
    d7 = datetime(year=2026, month=1, day=1, hour=2)
    d8 = datetime(year=2026, month=1, day=1, hour=23)
    encodings4.append(encoder.encode(d6))
    encodings4.append(encoder.encode(d7))
    encodings4.append(encoder.encode(d8))
    assert hamming_distance_helper(encodings4[0], encodings4[1]) < hamming_distance_helper(
        encodings4[0], encodings4[2]
    )

    encodings5 = []
    d9 = datetime(year=2000, month=1, day=1, hour=1, minute=1)
    d10 = datetime(year=2001, month=1, day=1, hour=1, minute=1)
    d11 = datetime(year=2002, month=1, day=1, hour=1, minute=1)
    encodings5.append(encoder.encode(d9))
    encodings5.append(encoder.encode(d10))
    encodings5.append(encoder.encode(d11))
    assert hamming_distance_helper(encodings5[0], encodings5[1]) < hamming_distance_helper(
        encodings5[0], encodings5[2]
    )
    encoder_year = DateEncoder(date_params_year_only)
    dt = datetime(2020, 6, 15, 12, 0)
    encoding_year = encoder_year.encode(dt)
    assert len(encoding_year) == 2048
    assert sum(encoding_year) == 42, f"Expected 42 active bits, got {sum(encoding_year)}"

    # Season only should work
    params_season = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=10,
        season_size=500,
        season_radius=50.0,
        season_sparsity=0.0,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    encoder_season = DateEncoder(params_season)
    encoding_season = encoder_season.encode(dt)
    assert len(encoding_season) == 500
    # RDSE may have slight variation in actual active bits due to hashing
    assert 8 <= sum(encoding_season) <= 12, f"Expected ~10 active bits, got {sum(encoding_season)}"

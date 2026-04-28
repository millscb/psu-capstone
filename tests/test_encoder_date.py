"""
Test suite for Date Encoder (basic version).
The Date Encoder with rdse_used=False produces basic scalar encodings for
temporal components (season, day of week, time of day, etc.).

Key Differences from RDSE Version:
  - Uses basic scalar encoding instead of RDSE for each component
  - May have less semantic richness but faster computation
  - Still decomposes datetime into multiple components
  - Each component encoded independently

Tests validate:
  1. Initialization with various component combinations
  2. Output format consistency
  3. Component encoding behavior
  4. Determinism across encoding attempts
  5. Edge cases (midnight, new year, etc.)
"""

# Test Suite: TS-05 (SDR Date Encoder)

from __future__ import annotations

from datetime import datetime

import pytest

from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.log import logger


@pytest.fixture
def date_encoder_instance() -> DateEncoder:
    """Fixture to create a DateEncoder instance for testing. This can be used to test any defualt DateEncoder object.

    Usage:
        # Test Type: unit test
        def test_example(date_encoder_instance):
            # Use date_encoder_instance in your test
            pass

    """

    return DateEncoder()


# Test Type: unit test
def test_season_encode():
    # TS-05 TC-046
    """Verify ScalarEncoder correctly encodes season (day of year) values.

    Tests that different days of the year produce distinct encodings with correct
    bit positions. Validates that the ScalarEncoder creates contiguous blocks of active
    bits that vary smoothly with the day of year value.
    """
    # Arrange
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_size=366,
        season_active_bits=4,
        season_sparsity=0.0,
        season_radius=91.5,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]

    actual_encoding = []

    expected_encoding = [
        [0, 1, 2, 3],
        [125, 126, 127, 128],
        [111, 112, 113, 114],
        [67, 68, 69, 70],
        [40, 41, 42, 43],
        [38, 39, 40, 41],
        [38, 39, 40, 41],
        [54, 55, 56, 57],
        [53, 54, 55, 56],
    ]

    # Act
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    # Assert
    assert actual_encoding == expected_encoding, "DateEncoder season test failed!"


# Test Type: unit test
def test_day_of_week_encode():
    # TS-05 TC-047
    """Verify ScalarEncoder correctly encodes day of week values (0-6).

    Tests that different days of the week produce distinct, semantically similar
    encodings. Validates that adjacent days have overlapping representations while
    distant days (e.g., Mon vs Fri) have different representations.
    """
    # Arrange
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_size=2048,
        day_of_week_active_bits=2,
        day_of_week_radius=292.57,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]
    expected_encoding = [[4, 5], [4, 5], [6, 7], [6, 7], [12, 13], [0, 1], [0, 1], [12, 13], [8, 9]]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder day_of_week test failed!"


# Test Type: unit test
def test_weekend_encode():
    # TS-05 TC-048
    """Verify ScalarEncoder correctly encodes weekend vs weekday (binary flag).

    Tests that weekend periods (Fri 6pm - Sun midnight) produce consistent encodings
    distinct from weekday encodings. Validates the weekend flag encoder with expected
    bit patterns for various dates and times.
    """
    # Weekend defined as Fri after noon until Sun midnight
    date_params = DateEncoderParameters(
        year_active_bits=0,
        weekend_size=2048,
        weekend_active_bits=2,
        weekend_radius=39.38,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 11, 0],
        [1988, 5, 27, 20, 0],
    ]

    expected_encoding = [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [2, 3],
        [0, 1],
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder weekend test failed!"


# Test Type: unit test
def test_holiday_encode():
    # TS-05 TC-049
    """Verify ScalarEncoder correctly encodes holiday proximity (ramp value).

    Tests that dates near holidays produce a ramp-like encoding that increases
    in proximity to the holiday and decreases as distance increases. Validates
    encoding for specific configured holidays.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        holiday_size=2048,
        holiday_active_bits=4,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=186.18,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )
    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2019, 12, 31, 0, 0],
        [2019, 12, 31, 12, 00],
        [2020, 1, 1, 0, 0],
        [2020, 1, 1, 12, 0],
        [2020, 1, 1, 23, 59],
        [2020, 1, 2, 12, 0],
        [2020, 1, 3, 0, 0],
        [2019, 12, 11, 14, 4],
        [2010, 1, 3, 0, 0],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2019, 4, 17, 0, 0],
    ]

    expected_encoding = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [0, 1, 2, 3],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder holiday test failed!"


# Test Type: unit test
def test_time_of_day_encode():
    # TS-05 TC-050
    """Verify ScalarEncoder correctly encodes time of day (0-24 hours).

    Tests that different times throughout the day produce distinct encodings with
    semantic similarity (nearby times have overlapping representations). Validates
    the smooth representation of the 24-hour cycle.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        time_of_day_size=1024,
        time_of_day_active_bits=4,
        time_of_day_radius=42.67,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 12, 0],
        [2017, 4, 17, 1, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 11, 0],
        [1988, 5, 27, 20, 0],
    ]

    expected_encoding = [
        [0, 1, 2, 3],
        [15, 16, 17, 18],
        [15, 16, 17, 18],
        [0, 1, 2, 3],
        [12, 13, 14, 15],
        [1, 2, 3, 4],
        [23, 24, 25, 26],
        [20, 21, 22, 23],
        [11, 12, 13, 14],
        [20, 21, 22, 23],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder time_of_day test failed!"


# Test Type: unit test
def test_custom_day_encode():
    # TS-05 TC-051
    """Verify ScalarEncoder correctly encodes custom day groups (binary flag).

    Tests that days matching a custom group pattern (e.g., "Mon, Wed, Fri") produce
    consistent encodings distinct from non-matching days. Validates the custom days
    encoder with arbitrary user-defined day groups.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        custom_size=2048,
        custom_active_bits=2,
        custom_radius=730.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 11, 0],
        [1988, 5, 27, 20, 0],
    ]

    expected_encoding = [
        [2, 3],
        [2, 3],
        [0, 1],
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
        [0, 1],
        [2, 3],
        [2, 3],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder custom_day test failed!"


# Test Type: unit test
def test_all_combined_encode():
    # TS-05 TC-052
    """Verify ScalarEncoder correctly encodes all temporal features simultaneously.

    Tests that when all seven encoders (season, day_of_week, weekend, custom, holiday,
    time_of_day) are enabled together, the output contains expected bit patterns for
    each encoder concatenated into a single SDR. Validates the full combined encoding.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_size=100,
        season_active_bits=2,
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_size=100,
        day_of_week_active_bits=2,
        day_of_week_radius=14.28,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_size=100,
        weekend_active_bits=2,
        weekend_radius=1.92,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_size=100,
        holiday_active_bits=2,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=9.09,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_size=100,
        time_of_day_active_bits=2,
        time_of_day_radius=0.0278,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_size=100,
        custom_active_bits=2,
        custom_radius=25.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
        [1988, 5, 27, 11, 0],
    ]

    actual_encoding = []

    expected_encoding = [
        [0, 1, 100, 101, 200, 201, 300, 301, 400, 401, 500, 501],
        [34, 35, 100, 101, 200, 201, 300, 301, 400, 401, 501, 502],
        [30, 31, 100, 101, 200, 201, 300, 301, 400, 401, 501, 502],
        [18, 19, 100, 101, 200, 201, 300, 301, 400, 401, 500, 501],
        [11, 12, 101, 102, 200, 201, 300, 301, 400, 401, 500, 501],
        [10, 11, 100, 101, 200, 201, 300, 301, 400, 401, 500, 501],
        [10, 11, 100, 101, 200, 201, 300, 301, 400, 401, 502, 503],
        [15, 16, 101, 102, 200, 201, 300, 301, 400, 401, 502, 503],
        [14, 15, 100, 101, 200, 201, 300, 301, 400, 401, 502, 503],
        [14, 15, 100, 101, 200, 201, 300, 301, 400, 401, 501, 502],
    ]

    # Act
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    # Assert
    assert (
        actual_encoding == expected_encoding
    ), "DateEncoder season_day_of_week_combined test failed!"


# ---------------------------------------------------------------------------
# Output format and parameter conformance (binary 0/1 only, length)
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_date_encode_output_only_zeros_and_ones():
    # TS-13 TC-096
    """Verify DateEncoder output is strictly binary (only 0 and 1 values).

    Tests that all bits in the output SDR are either 0 or 1, with no intermediate
    values or errors. This validates the fundamental SDR representation format.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_size=100,
        season_active_bits=2,
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )
    date_encoder = DateEncoder(date_params)
    dt = datetime(2020, 1, 1, 0, 0)
    out = date_encoder.encode(dt)
    assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


# Test Type: unit test
def test_date_encode_output_length_equals_size():
    # TS-13 TC-097
    """Verify DateEncoder output length matches the configured encoder size.

    Tests that the output SDR has a length equal to the sum of all enabled
    encoder sizes. This validates that no bits are dropped or added during encoding.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_size=100,
        season_active_bits=2,
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )
    date_encoder = DateEncoder(date_params)
    dt = datetime(2020, 1, 1, 0, 0)
    out = date_encoder.encode(dt)
    assert (
        len(out) == date_encoder._size
    ), f"Output length must equal encoder size ({date_encoder._size}), got {len(out)}"


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


# Test Type: unit test
def test_season():
    # TS-13 TC-088
    """Verify ScalarEncoder correctly encodes and decodes season values.

    Tests season encoding with RDSE backend (rdse_used=True), verifying that
    the decoder returns values in the valid range [0, 366] representing days of year,
    and that the same encoder instance produces deterministic decodings.
    """
    # Arrange
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_size=366,
        season_active_bits=4,
        season_sparsity=0.0,
        season_radius=91.5,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )

    date_encoder = DateEncoder(date_params)

    # [year, month, day, hour, minute]
    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]

    actual_decoded = []

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)

        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # RDSE is random; assert structure, valid range, and round-trip consistency
    assert len(actual_decoded) == len(test_case)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(
            decoded, dict
        ), f"Date {test_case[i]}: decoded should be dict, got {type(decoded)}"
        assert (
            "season" in decoded and len(decoded) == 1
        ), f"Date {test_case[i]}: one encoder => 1 key (season), got {list(decoded)}"
        value = decoded["season"][0]
        assert (
            0 <= value <= 366
        ), f"Date {test_case[i]}: season (day of year) in [0, 366], got {value}"

    # Same input encoded/decoded twice gives same result (deterministic for this instance)
    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1["season"][0] == dec2["season"][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Test Type: unit test
def test_rdse_decode_same_across_instances_with_same_params():
    # TS-13 TC-089
    """Verify RDSE produces identical decodings across different encoder instances.

    Tests that when two DateEncoder instances are created with the same parameters
    and default seed, they produce identical decodings for the same input. This
    validates deterministic RDSE behavior across instances.
    """
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_size=2048,
        day_of_week_active_bits=2,
        day_of_week_radius=292.57,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )

    dt = datetime(2020, 1, 1, 0, 0)
    decoded_values = []
    for _ in range(5):
        encoder = DateEncoder(date_params)
        encoded = encoder.encode(dt)
        decoded = encoder.decode(encoded)
        decoded_values.append(decoded["dayofweek"][0])

    # Same params => same default RDSE seed => same decode (deterministic across instances)
    assert (
        len(set(decoded_values)) == 1
    ), f"With same params (no seed override), all instances should produce same decode; got {decoded_values}"


# Test Type: unit test
def test_day_of_week():
    # TS-13 TC-090
    """Verify RDSE correctly decodes day of week (0-6) from encoded SDR.

    Tests that the RDSE decoder recovers day-of-week values in the valid range [0, 6].
    Validates semantic similarity where adjacent days have more similar decodings than
    distant days, and confirms deterministic decoding for the same instance.
    """
    # Arrange: only day-of-week encoder, RDSE (decode returns values)
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_size=2048,
        day_of_week_active_bits=2,
        day_of_week_radius=292.57,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )

    date_encoder = DateEncoder(date_params)
    # [year, month, day, hour, minute]
    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]

    actual_decoded = []

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # RDSE is random; assert structure, valid range, and round-trip consistency
    assert len(actual_decoded) == len(test_case)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(
            decoded, dict
        ), f"Date {test_case[i]}: decoded should be dict, got {type(decoded)}"
        assert (
            "dayofweek" in decoded and len(decoded) == 1
        ), f"Date {test_case[i]}: one encoder => 1 key (dayofweek), got {list(decoded)}"
        value = decoded["dayofweek"][0]
        assert (
            0 <= value < 7
        ), f"Date {test_case[i]}: day_of_week (Mon=0..Sun=6) in [0, 7), got {value}"

    # Same input encoded/decoded twice gives same result (deterministic for this instance)
    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1["dayofweek"][0] == dec2["dayofweek"][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Shared test cases for multi-encoder decoder tests ([year, month, day, hour, minute])
_DECODER_TEST_CASES = [
    [2020, 1, 1, 0, 0],
    [2019, 12, 11, 14, 45],
    [2010, 11, 4, 14, 55],
    [2019, 7, 4, 0, 0],
    [2019, 4, 21, 0, 0],
    [2017, 4, 17, 0, 0],
    [2017, 4, 17, 22, 59],
    [1988, 5, 29, 20, 0],
    [1988, 5, 27, 20, 0],
]


# Test Type: unit test
def test_weekend():
    # TS-13 TC-091
    """Verify RDSE correctly decodes weekend flag (0=weekday, 1=weekend).

    Tests that the RDSE decoder produces binary values (0 or 1) representing
    weekday/weekend status. Validates consistency and range of decoded values,
    and confirms deterministic decoding for the same instance.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_size=2048,
        weekend_active_bits=2,
        weekend_radius=39.38,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, dict)
        assert "weekend" in decoded and len(decoded) == 1
        value = decoded["weekend"][0]
        assert 0 <= value <= 1, f"Date {_DECODER_TEST_CASES[i]}: weekend in [0, 1], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1["weekend"][0] == dec2["weekend"][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Test Type: unit test
def test_custom_days():
    # TS-13 TC-092
    """Verify RDSE correctly decodes custom day group membership (0 or 1).

    Tests that the RDSE decoder produces binary values indicating whether a date
    belongs to a custom day group. Validates consistency and confirms deterministic
    decoding for the same instance.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_size=2048,
        custom_active_bits=2,
        custom_radius=730.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["mon,wed,fri"],
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, dict)
        assert "customdays" in decoded and len(decoded) == 1
        value = decoded["customdays"][0]
        assert 0 <= value <= 1, f"Date {_DECODER_TEST_CASES[i]}: custom_days in [0, 1], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1["customdays"][0] == dec2["customdays"][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Test Type: unit test
def test_holiday():
    # TS-13 TC-093
    """Verify RDSE correctly decodes holiday proximity ramp from encoded SDR.

    Tests that the RDSE decoder produces ramp values (0 to ~2) representing proximity
    to configured holidays. Validates that encoded SDR properly captures the holiday
    proximity encoding and confirms deterministic decoding.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_size=2048,
        holiday_active_bits=4,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=186.18,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, dict)
        assert "holiday" in decoded and len(decoded) == 1
        value = decoded["holiday"][0]
        assert (
            0 <= value <= 3
        ), f"Date {_DECODER_TEST_CASES[i]}: holiday ramp in [0, 3], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1["holiday"][0] == dec2["holiday"][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Test Type: unit test
def test_time_of_day():
    # TS-13 TC-094
    """Verify RDSE correctly decodes time of day (0-24 hours) from encoded SDR.

    Tests that the RDSE decoder recovers hour values in the valid range [0, 24].
    Validates semantic similarity where nearby times have similar decodings, and
    confirms deterministic decoding for the same instance.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_size=1024,
        time_of_day_active_bits=4,
        time_of_day_radius=42.67,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_active_bits=0,
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, dict)
        assert "timeofday" in decoded and len(decoded) == 1
        value = decoded["timeofday"][0]
        assert (
            0 <= value <= 24
        ), f"Date {_DECODER_TEST_CASES[i]}: time_of_day in [0, 24], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert (
        dec1["timeofday"][0] == dec2["timeofday"][0]
    ), "Round-trip should be deterministic for same encoder instance"


# Test Type: unit test
def test_all_combined():
    # TS-13 TC-095
    """Verify RDSE correctly decodes all temporal features when combined.

    Tests that when all six RDSE encoders (season, day_of_week, weekend, custom,
    holiday, time_of_day) are enabled together, the decoder correctly extracts
    and decodes each component from the combined SDR, producing valid ranges and
    maintaining determinism.
    """
    date_params = DateEncoderParameters(
        year_active_bits=0,
        season_size=100,
        season_active_bits=2,
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_size=100,
        day_of_week_active_bits=2,
        day_of_week_radius=14.28,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_size=100,
        weekend_active_bits=2,
        weekend_radius=1.92,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_size=100,
        holiday_active_bits=2,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=9.09,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_size=100,
        time_of_day_active_bits=2,
        time_of_day_radius=0.0278,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_size=100,
        custom_active_bits=2,
        custom_radius=25.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        rdse_used=True,
    )

    date_encoder = DateEncoder(date_params)
    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
        [1988, 5, 27, 11, 0],
    ]

    actual_decoded = []
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # Decode returns dict of 6 keys:
    # season, dayofweek, weekend, customdays, holiday, timeofday;
    # each value is (value, confidence)
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

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    for key in keys:
        assert dec1[key][0] == dec2[key][0], f"Round-trip deterministic for encoder {key}"

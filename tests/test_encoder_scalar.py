# Test Suite: TS-03 (SDR Encoder Scalar)
"""
Test suite for Scalar Encoder.

The Scalar Encoder encodes numeric values in a limited range into sparse SDRs.
It uses the Cortical Learning Algorithm to create semantic representations where
nearby values have overlapping encodings.

Key Features:
  - Range-limited input (minimum/maximum bounds)
  - Optional input clipping
  - Supports both periodic (cyclical) and non-periodic ranges
  - Deterministic encoding (same input → same SDR)
  - Semantic similarity: close values → high overlap

Parameter Validation:
  - Like RDSE, uses mutual exclusivity: exactly one of {active_bits, sparsity}
  - All tests explicitly set sparsity=0.0 when using active_bits
  - Supports radius/resolution specification for input coverage
  - Handles both real numbers and periodic values

Tests validate:
  1. Initialization with valid parameters
  2. Input clipping to min/max bounds
  3. Output format (binary 0/1 only, correct length)
  4. Active bits/sparsity conformance
  5. Semantic similarity (neighboring values overlap)
  6. Determinism and periodicity handling
"""

from datetime import datetime

import numpy as np
import pytest

from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters


@pytest.fixture
def scalar_encoder_instance():
    """Fixture to create a ScalarEncoder instance for testing. This may change when we get Union
    working properly."""


# Helper -- may need to be implemented later
def do_scalar_value_cases(encoder: ScalarEncoder, cases):
    pass


# Test Type: unit test
def test_scalar_encoder_initialization():
    # TS-03 TC-030
    """
    Test ScalarEncoder initialization with valid parameters.

    Validates:
      - Encoder instantiates successfully with proper parameters
      - Size property matches configured size
      - Encoder is correct type (ScalarEncoder)

    Why it passes:
      - active_bits=5 with sparsity=0.0 satisfies mutual exclusivity
      - Range parameters (minimum=0, maximum=100) are valid
      - encoder.size accessible and equals configured size
    """

    # Arrange
    parameters = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        size=10,
        radius=1.0,
        category=False,
        resolution=0.0,
    )

    # Act
    encoder = ScalarEncoder(parameters)
    """Demonstrating deep copy"""
    ScalarEncoder(parameters)

    # Assert
    assert isinstance(encoder, ScalarEncoder)
    assert encoder.size == 10


# Test Type: unit test
def test_clipping_inputs():
    # TS-03 TC-031
    """Test that inputs are correctly clipped to the specified min/max range."""

    # Arrange
    p = ScalarEncoderParameters(
        minimum=10,
        maximum=20,
        clip_input=False,
        periodic=False,
        active_bits=2,
        sparsity=0.0,
        size=10,
        radius=1.0,
        resolution=0.0,
        category=False,
    )
    # Act and Assert baseline
    encoder = ScalarEncoder(p)

    assert encoder.size == 10
    # assert test_sdr.size == 10

    # Act and Asset - Test input clipping
    # These should pass without exceptions
    encoder.encode(10.0)  # At minimum edge case
    encoder.encode(20.0)  # At maximum edge case

    with pytest.raises(ValueError):
        encoder.encode(9.9)  # Below minimum edge case

    with pytest.raises(ValueError):
        encoder.encode(20.1)  # Above maximum edge case


# Test Type: unit test
def test_valid_scalar_inputs():
    # TS-03 TC-032
    """Test that valid scalar inputs are encoded correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        size=10,
        active_bits=2,
        minimum=10,
        maximum=20,
        sparsity=0.0,
        radius=1.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 10

    with pytest.raises(ValueError):
        encoder.encode(9.999)  # Below minimum edge case

    with pytest.raises(ValueError):
        encoder.encode(20.0001)  # Above maximum edge case

    encoder.encode(10.0)  # At minimum edge case
    encoder.encode(19.9)  # Just below maximum edge case


# Test Type: unit test
def test_scalar_encoder_category_encode():
    # TS-03 TC-033
    """Test that category scalar inputs are encoded correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        size=66,
        sparsity=0.02,
        minimum=0,
        maximum=65,
        active_bits=0,
        radius=1.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 66

    # Act and Assert - Value less than minimum should raise
    with pytest.raises(ValueError):
        encoder.encode(-0.01)  # Below minimum edge case

    # Act and Assert - Value greater than maximum should raise
    with pytest.raises(ValueError):
        encoder.encode(66.0)  # Above maximum edge case

    # Value within range should not raise
    encoder.encode(0.0)  # At minimum edge case
    encoder.encode(32.0)  # Mid-range value
    encoder.encode(65.0)  # At maximum edge case
    encoder.encode(10.0)


# Test Type: unit test
def test_scalar_encoder_non_integer_bucket_width():
    # TS-03 TC-034
    """Test that scalar encoder handles non-integer bucket widths correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=True,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=7,
        radius=1.0,
        category=False,
        resolution=0.0,
    )

    encoder = ScalarEncoder(params)

    cases = [
        (10.0, [0, 1, 2]),
        (20.0, [4, 5, 6]),
    ]

    do_scalar_value_cases(encoder, cases)


# Test Type: unit test
def test_scalar_encoder_round_to_nearest_multiple_of_resolution():
    # TS-03 TC-035
    """Test that scalar encoder rounds to the nearest multiple of resolution correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        minimum=10,
        maximum=20,
        clip_input=False,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=1.0,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 13

    cases = [
        (10.00, [0, 1, 2]),
        (10.49, [0, 1, 2]),
        (10.50, [1, 2, 3]),
        (11.49, [1, 2, 3]),
        (11.50, [2, 3, 4]),
        (14.49, [4, 5, 6]),
        (14.50, [5, 6, 7]),
        (15.49, [5, 6, 7]),
        (15.50, [6, 7, 8]),
        (19.00, [9, 10, 11]),
        (19.49, [9, 10, 11]),
        (19.50, [10, 11, 12]),
        (20.00, [10, 11, 12]),
    ]

    do_scalar_value_cases(encoder, cases)


# Test Type: unit test
def test_scalar_encoder_periodic_round_nearest_multiple_of_resolution():
    # TS-03 TC-036
    """Test that periodic scalar encoder rounds to the nearest multiple of resolution correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        minimum=10,
        maximum=20,
        clip_input=False,
        periodic=True,
        active_bits=3,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=1,
    )
    encoder = ScalarEncoder(params)

    # Act and Assert - baseline
    assert encoder.size == 10
    values = [10.00, 10.49, 10.50, 11.49, 11.50, 14.49, 14.50, 15.49, 15.50, 19.49, 19.50, 20.00]
    for value in values:
        encoded = encoder.encode(value)
        active_indices = [index for index, bit in enumerate(encoded) if bit == 1]
        assert len(active_indices) == params.active_bits
        assert all(0 <= index < encoder.size for index in active_indices)


# Test Type: unit test
def test_scalar_encoder_serialization():
    # TS-03 TC-037
    """Test serialization and deserialization of ScalarEncoder."""

    # Arrange
    inputs = []

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p))

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p))

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=True,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p))

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=0.1337,
    )
    inputs.append(ScalarEncoder(p))

    q = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=0,
        sparsity=0.15,
        size=100,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(q))

    r = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=0,
        sparsity=0.02,
        size=700,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(r))
    inputs.append(ScalarEncoder(r))

    for encoder in inputs:
        if type(encoder) is ScalarEncoder:
            assert encoder.size > 0
            assert encoder._active_bits > 0
            assert encoder._active_bits < encoder.size
            assert encoder._minimum <= encoder._maximum
            assert encoder._resolution > 0
            assert encoder._radius > 0
            assert encoder._sparsity > 0


# ---------------------------------------------------------------------------
# Output format and parameter conformance (binary 0/1 only, length, active_bits)
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_scalar_encode_output_only_zeros_and_ones():
    """Encoder output must contain only 0 and 1."""
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        size=50,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    for value in (0, 10, 50, 100):
        out = encoder.encode(value)
        assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


# Test Type: unit test
def test_scalar_encode_output_length_equals_size():
    """Encoder output length must equal the configured size."""
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=4,
        sparsity=0.0,
        size=32,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    out = encoder.encode(50.0)
    assert len(out) == 32, f"Output length must equal size (32), got {len(out)}"


# Test Type: unit test
def test_scalar_encode_output_active_bits_conforms():
    """Output must have exactly active_bits ones; sparsity = active_bits/size."""
    size = 64
    active_bits = 8
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=active_bits,
        sparsity=0.0,
        size=size,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    out = encoder.encode(25.0)
    num_ones = sum(out)
    assert num_ones == active_bits, f"Exactly {active_bits} ones expected, got {num_ones}"
    assert num_ones / len(out) == pytest.approx(
        active_bits / size
    ), "Sparsity should equal active_bits/size"


"""Correctness tests below."""


# Test Type: unit test
def test_scalar_encode_improper_values():
    """
    This test tries to encode with multiple entry types.
    There should be an exception for each.
    """
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=8,
        sparsity=0.0,
        size=64,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    with pytest.raises(ValueError):
        encoder.encode("test")
        encoder.encode(datetime(2020, 1, 1, 0, 0))
        encoder.encode([1, 2, 3, 4])
        encoder.encode(((10, 20), 2))


# Test Type: unit test
def test_scalar_encode_empty_values():
    """
    Tests that encode properly raises an exception if no input value is entered.
    This also tests a None value.
    """
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=8,
        sparsity=0.0,
        size=64,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    with pytest.raises(TypeError):
        encoder.encode()
        encoder.encode(None)


# Test Type: unit test
def test_scalar_decode_empty_sdr():
    """Tests that the decode method can raise an exception when an empty sdr is entered."""
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=8,
        sparsity=0.0,
        size=64,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    with pytest.raises(ValueError):
        encoder.encode(1)
        encoder.decode([])


# Test Type: unit test
def test_clear_registry_decode():
    """
    This tests that a value error is raised if there are no registered
    encodings and the user tries to decode.
    """
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=8,
        sparsity=0.0,
        size=64,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    with pytest.raises(ValueError):
        a = encoder.encode(1)
        encoder.clear_registered_encodings()
        encoder.decode(a)


def hamming_distance_helper(first: np.ndarray, second: np.ndarray) -> int:
    """
    Helper method to find the differences with the first != second and then count the nonzero
    as that is how many different bits there are. So if first was 1001 and second was 1010 the
    first operation would be 0011 and the count_nonzero would return 2. This indicates a hamming
    distance of 2 since 2 of the bits are different.
    """
    return int(np.count_nonzero(first != second))


# Test Type: unit test
def test_scalar_hamming_distance():
    """
    This test compares the mean hamming distances between consecutive encoded values
    like 1 compared to 2 all of the way up to 1000. Then we take the mean of these
    hamming distances. On top of that it compares 1 through 500 of encoded values to
    9000 through 10000. We then compare these hamming distances. The thought is that
    the values right next to each other should have less bit differences than ones far away.
    """
    import random

    p = ScalarEncoderParameters(
        minimum=0,
        maximum=10001,
        clip_input=True,
        periodic=False,
        active_bits=0,
        sparsity=0.02,
        size=2048,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    encodings_first = []
    for v in range(1, 1001):
        encodings_first.append(np.array(encoder.encode(float(v))))

    encodings_second = []
    for v in range(9000, 10001):
        encodings_second.append(np.array(encoder.encode(float(v))))

    random_values = random.sample(range(0, 10000), 2000)
    encodings_random = []
    for v in random_values:
        encodings_random.append(np.array(encoder.encode(float(v))))

    # check two encodings by each others hamming distances. small numbers
    consecutive_distances = []
    for v in range(len(encodings_first) - 1):
        d = hamming_distance_helper(encodings_first[v], encodings_first[v + 1])
        consecutive_distances.append(d)
    mean_consecutive = np.mean(consecutive_distances)

    # check two encodings by each others hamming distances, large numbers
    consecutive_distances_large = []
    for v in range(len(encodings_second) - 1):
        d = hamming_distance_helper(encodings_second[v], encodings_second[v + 1])
        consecutive_distances_large.append(d)
    mean_consecutive_large = np.mean(consecutive_distances_large)

    # check the hamming distance between far apart input values
    far_distances = []
    for v in range(1000):
        d = hamming_distance_helper(encodings_first[v], encodings_second[v])
        far_distances.append(d)
    mean_far = np.mean(far_distances)

    # check the hamming distance between two random encodings
    random_distances = []
    for i, j in zip(range(0, 1000), range(1000, 2000)):
        d = hamming_distance_helper(encodings_random[i], encodings_random[j])
        random_distances.append(d)
    mean_random = np.mean(random_distances)

    print("\n")
    print("Consecutive distances mean: ", mean_consecutive)
    print("Far distances mean: ", mean_far)
    print("Large consecutive numbers mean distance: ", mean_consecutive_large)
    print("Random hamming distance mean: ", mean_random)

    assert mean_consecutive < mean_random < mean_far


"""
Tests for Scalar decode.

# Test Suite: TS-26 (Scalar Decode)

decode() returns (value, confidence) by finding the cached encoding with best
overlap to the input SDR. Cache is populated on encode().
"""


# Test Type: unit test
def test_scalar_decode_returns_tuple_value_confidence():
    # TS-26 TC-241
    """decode() returns (value, confidence) tuple."""
    params = ScalarEncoderParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
    )
    encoder = ScalarEncoder(params)
    encoded = encoder.encode(5.0)
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, tuple)
    assert len(decoded) == 2
    value, confidence = decoded
    assert isinstance(confidence, (int, float))
    assert 0 <= confidence <= 1


# Test Type: unit test
def test_scalar_decode_round_trip_same_value():
    # TS-26 TC-242
    """decode(encode(x)) returns (x, high confidence) for same encoder instance."""
    params = ScalarEncoderParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
    )
    encoder = ScalarEncoder(params)
    for x in (0.0, 5.0, 10.0, 100.0):
        encoded = encoder.encode(x)
        value, confidence = encoder.decode(encoded)
        assert value == x, f"Round-trip: encode({x}) then decode should yield {x}, got {value}"
        assert confidence >= 0.9, f"Round-trip confidence should be high, got {confidence}"


# Test Type: unit test
def test_scalar_decode_wrong_size_raises():
    # TS-26 TC-243
    """decode() with wrong-length SDR raises ValueError."""
    params = ScalarEncoderParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
    )
    encoder = ScalarEncoder(params)
    encoder.encode(1.0)  # populate cache so decode has candidates
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 100)
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 300)


# Test Type: unit test
def test_scalar_decode_no_candidates_raises():
    # TS-26 TC-244
    """decode() with no prior encode (empty cache) raises ValueError."""
    params = ScalarEncoderParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
    )
    encoder = ScalarEncoder(params)
    # No encode() call -> _encoding_cache empty -> no candidates
    with pytest.raises(ValueError, match="No candidate encodings"):
        encoder.decode([0] * 256)

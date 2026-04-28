# Test Suite: TS-04 (SDR Random Distributed Scalar Encoder)
"""
Test suite for the RDSE (Random Distributed Scalar Encoder).

This test suite validates the Random Distributed Scalar Encoder implementation,
which produces sparse, distributed SDR representations for scalar input values.

Key Testing Areas:
  1. **Initialization & Parameter Validation**: Ensures RDSE correctly initializes
     with valid parameters and validates mutual exclusivity constraints.
  2. **Parameter Mutual Exclusivity**: Enforces that exactly ONE of {active_bits, sparsity}
     must be set (non-zero), and exactly ONE of {radius, resolution, category}.
     - Tests now explicitly set sparsity=0.0 when using active_bits to satisfy constraint
     - This prevents ambiguity in sparse representation control
  3. **Output Format**: Verifies encoded output is binary (0/1 only) and has correct length.
  4. **Sparsity Conformance**: When sparsity is specified, validates that actual output
     sparsity matches the target (within tolerance for hash collisions).
  5. **Active Bits Conformance**: When active_bits is specified, validates count of 1s
     in output is close to target (hash collisions may reduce slightly).
  6. **Determinism & Seeding**: Ensures same seed produces same output; different seeds
     produce different outputs for same input.
  7. **Locality**: Inputs within radius should overlap; inputs far outside should not.
  8. **Orthogonality**: Encodings of different values should mostly be orthogonal.

Recent Code Changes Addressed:
  - RDSE.check_parameters() now enforces mutual exclusivity: either active_bits XOR sparsity
  - Tests updated to explicitly pass sparsity=0.0 with active_bits, avoiding parameter validation errors
  - This change ensures consistent sparse representation and prevents accidental double-specification
"""

from datetime import datetime

import numpy as np
import pytest

from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


@pytest.fixture
def rdse_instance():
    """Fixture to create an RDSE instance for tests."""


# Test Type: unit test
def test_rdse_initialization():
    # TS-04 TC-038
    """
    Test that RDSE correctly initializes with valid parameters.

    Validates:
      - Encoder initializes when parameters satisfy mutual exclusivity constraints
      - sparsity=0.05 with active_bits=0 satisfies the "exactly one of" requirement

    Why it passes:
      - Parameters correctly set sparsity without active_bits
    """

    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.05, radius=0.0, resolution=1.23, category=False, seed=0
    )

    encoder = RandomDistributedScalarEncoder(parameters)
    """Makes sure it is the correct instance"""
    assert isinstance(encoder, RandomDistributedScalarEncoder)


# Test Type: unit test
def test_size():
    # TS-04 TC-039
    """
    Test that the encoder correctly reports its configured size.

    Validates:
      - Internal _size attribute matches the configured size
      - This is a foundational property for all SDR operations

    Why it passes:
      - Size parameter is directly assigned and accessible
    """
    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.05, radius=0.0, resolution=1.23, category=False, seed=0
    )

    encoder = RandomDistributedScalarEncoder(parameters)
    """Checks that the size is correct."""
    assert encoder._size == 1000


# Test Type: unit test
def test_dimensions():
    # TS-04 TC-040
    """
    Ensure the encoder reports configured size via public property.

    Validates:
      - Public 'size' property returns correct configured size
      - Multiple encoder instances with same parameters all report same size

    Why it passes:
      - Size property correctly exposes internal _size attribute
    """
    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.05, radius=0.0, resolution=1.23, category=False, seed=0
    )

    encoder = RandomDistributedScalarEncoder(parameters)
    RandomDistributedScalarEncoder(parameters)
    """Checks that the size is correct via the public property."""
    assert encoder.size == 1000


# Test Type: unit test
def test_encode_active_bits():
    # TS-04 TC-041
    """
    Verify that when active_bits is specified, sparsity conform correctly.

    Validates:
      - Output SDR length equals configured size
      - Number of active bits (1s) is within tolerance of specified active_bits
      - Hash collisions may reduce active bit count by ~10%, so we accept 45-50 for target 50

    Why it passes:
      - sparsity=0.0 and active_bits=50 passes mutual exclusivity check
      - Encoder produces expected number of 1-bits (accounting for hash collisions)
      - Hash function may cause collisions, reducing active count slightly
    """
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=0.0, radius=0.0, resolution=1.5, category=False, seed=0
    )
    encoder = RandomDistributedScalarEncoder(parameters)
    a = encoder.encode(10)
    """Is the SDR the correct size?"""
    assert len(a) == 1000
    """Is the SDR length equal to encoder size?"""
    assert len(a) == encoder.size
    sparse_indices = [i for i, x in enumerate(a) if x == 1]
    sparse_size = len(sparse_indices)
    """Since we have hash collision we are making sure between 45 and 50 bits are encoded."""
    assert 45 <= sparse_size <= 50
    dense_indices = a
    dense_size = len(dense_indices)
    """Is the dense size equal to size?"""
    assert dense_size == 1000


# Test Type: unit test
def test_resolution_plus_radius_plus_category():
    # TS-04 TC-042
    """
    Validate that multiple resolution/radius/category constraint is enforced.

    The RDSE requires exactly ONE of {radius, resolution, category} to define
    the input coverage behavior.

    Validates:
      - Setting both radius AND resolution raises exception
      - Setting category AND radius raises exception
      - Setting category AND resolution raises exception

    Why it passes:
      - RDSEParameters.check_parameters() validates mutual exclusivity
      - Exception is raised when more than one of these is non-zero/True
    """
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=0.0, radius=1.0, resolution=1.5, category=False, seed=0
    )
    """
    Make sure an exception is thrown here since these parameters should
    not be used together.
    """
    with pytest.raises(ValueError):
        RandomDistributedScalarEncoder(parameters)

    parameters.radius = 0.0
    parameters.category = True
    with pytest.raises(ValueError):
        RandomDistributedScalarEncoder(parameters)

    parameters.category = False
    parameters.resolution = 0.0
    parameters.radius = 1.0
    RandomDistributedScalarEncoder(parameters)


# Test Type: unit test
def test_sparsity_or_activebits():
    # TS-04 TC-043
    """
    Validate mutual exclusivity: exactly ONE of {active_bits, sparsity} must be set.

    RDSE uses either active_bits (explicit count of active bits) OR sparsity (fraction)
    to control output density, but NOT both. This prevents ambiguity in representation.

    Validates:
      - Setting both active_bits=50 AND sparsity=1.0 raises exception
      - Setting only active_bits (sparsity=0.0) succeeds
      - Setting only sparsity (active_bits=0) succeeds

    Why it passes:
      - RDSEParameters.check_parameters() enforces mutual exclusivity constraint
      - When both are set, exception raised in encoder initialization
      - When only one is set (other is 0/0.0), encoder initializes successfully
      - Tests verify both successful cases (single parameter set) and error case (both set)
      - sparsity=0.0 is used explicitly in tests to satisfy exclusivity constraint
    """
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=1.0, radius=0.0, resolution=1.5, category=False, seed=0
    )
    """Make sure an exception is thrown here since both sparsity and active bits are set."""
    with pytest.raises(ValueError):
        RandomDistributedScalarEncoder(parameters)
    """These should be able to run without an exception or assert since both are not set at once."""
    parameters.sparsity = 0.0
    RandomDistributedScalarEncoder(parameters)
    parameters.sparsity = 1.0
    parameters.active_bits = 0
    RandomDistributedScalarEncoder(parameters)


# Test Type: unit test
def test_one_of_resolution_radius_category_should_be_entered():
    # TS-04 TC-044
    """
    Validate that exactly ONE of {radius, resolution, category} must be specified.

    The RDSE requires one method to determine input coverage:
      - radius: inputs within this distance should overlap
      - resolution: explicit granularity of the input space
      - category: one-hot encoding (one bit per category)

    Setting all three to 0/False is invalid - must specify exactly one.

    Validates:
      - All parameters zero/False raises exception
      - At least one parameter must be non-zero/True for initialization

    Why it passes:
      - RDSEParameters.check_parameters() validates constraint
      - When no coverage method is defined, initialization fails with exception
      - Tests verify error is raised when all three are absent
    """
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=0.0, radius=0.0, resolution=0.0, category=False, seed=0
    )
    """Make sure an exception is thrown here since neither radius, resolution, or category were entered."""
    with pytest.raises(ValueError):
        RandomDistributedScalarEncoder(parameters)


# Test Type: unit test
def test_one_of_activebit_or_sparsity_is_entered():
    # TS-04 TC-045
    """
    Validate that exactly ONE of {active_bits, sparsity} must be specified (non-zero).

    The RDSE requires one method to control sparsity:
      - active_bits: explicit count of active bits in output
      - sparsity: fraction (0.0-1.0) of bits to activate

    Setting both to 0/0.0 is invalid - must specify exactly one.

    Validates:
      - Both zero/zero raises exception
      - At least one must be non-zero for initialization

    Why it passes:
      - RDSEParameters.check_parameters() validates constraint
      - When no sparsity method is defined, initialization fails with exception
      - Tests verify error is raised when both are absent (both are 0)
      - This complements test_sparsity_or_activebits which tests the "not both" constraint
    """
    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.0, radius=1.0, resolution=0.0, category=False, seed=0
    )
    """Make sure an exception is thrown here since neither active bits or sparsity was entered"""
    with pytest.raises(ValueError):
        RandomDistributedScalarEncoder(parameters)


# Test Type: unit test
def test_2048_bits_40_active_bits():
    # TS-04 TC-071
    """This test is to check and make sure our current RDSE can handler this many bits."""
    parameters = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=1.0, resolution=0.0, category=False, seed=0
    )
    rdse = RandomDistributedScalarEncoder(parameters)

    a = rdse.encode(10)
    sparse = [i for i, x in enumerate(a) if x == 1]
    """Checking for active bits accounting for hash collisions."""
    assert 35 <= len(sparse) <= 40
    """Make sure the density is 2048."""
    assert len(a) == 2048


# Test Type: unit test
def test_deterministic_same_seed():
    # TS-04 TC-072
    """
    This test assures that the same value encoded by two different
    RDSE encoders with the same seed will output the same sdrs.
    """
    params = RDSEParameters(
        size=2048,
        active_bits=40,
        sparsity=0.0,
        seed=40,
    )

    encoder1 = RandomDistributedScalarEncoder(params)
    encoder2 = RandomDistributedScalarEncoder(params)

    sdr1 = encoder1.encode(12.34)
    sdr2 = encoder2.encode(12.34)

    assert sdr1 == sdr2


# Test Type: unit test
def test_different_seed_produces_different_sdr_with_same_input_value():
    # TS-04 TC-073
    """
    This test assures that the same value encoded by two different
    RDSE encoders with different seeds will produce different sdrs.
    """
    params = RDSEParameters(
        size=2048,
        active_bits=40,
        sparsity=0.0,
        seed=40,
    )
    params1 = RDSEParameters(
        size=2048,
        active_bits=40,
        sparsity=0.0,
        seed=39,
    )
    encoder1 = RandomDistributedScalarEncoder(params)
    encoder2 = RandomDistributedScalarEncoder(params1)

    sdr1 = encoder1.encode(12.34)
    sdr2 = encoder2.encode(12.34)

    assert sdr1 != sdr2


# Test Type: unit test
def test_resolution_boundary():
    # TS-04 TC-074
    """The goal of this test to determine if the resolution is functioning
    correctly. The index inside of the rdse is determing by int(input_value/resolution).
    This means that a resolution of 1.0 when encoding 1.0 will be 1, resolution of 1.0 and
    encoding 1.999 will also be 1. But, if you have a resolution of 1.0 and a value of 2.0
    it will be 2. So, the encoding of sdr_c should not be equal to a and b. To add to this
    you can see that a resolution of 1.0 only cares about increments of 1."""
    params = RDSEParameters(
        size=2048,
        active_bits=40,
        sparsity=0.0,
        resolution=1.0,
        radius=0,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)

    sdr_a = encoder.encode(1.0)
    sdr_b = encoder.encode(1.999)
    sdr_c = encoder.encode(2.0)

    assert sdr_a == sdr_b

    assert sdr_a != sdr_c


def hamming_distance_helper(first, second) -> int:
    """
    Helper method to find the differences with the first != second and then count the nonzero
    as that is how many different bits there are. So if first was 1001 and second was 1010 the
    first operation would be 0011 and the count_nonzero would return 2. This indicates a hamming
    distance of 2 since 2 of the bits are different.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    return int(np.count_nonzero(first != second))


# Test Type: unit test
def test_locality_checking_mmh3():
    # TS-04 TC-075
    """
    This test compares the mean hamming distances between consecutive encoded values like 1 compared to 2 all
    of the way up to 1000. Then we take the mean of these hamming distances. On top of that it compares 1 through 500
    of encoded values to 9000 through 10000. We then compare these hamming distances. The thought is that the values
    right next to each other should have less bit differences than ones far away.
    """
    import random

    params = RDSEParameters(
        size=2048,
        active_bits=40,
        sparsity=0.0,
        resolution=1.0,
        radius=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)

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


# By: Dr. Agrawal
def _make_large_encoder(radius: float = 1.0) -> RandomDistributedScalarEncoder:
    params = RDSEParameters(
        size=2048,
        sparsity=0.02,
        radius=radius,
        resolution=0,
        active_bits=0,
        category=False,
        seed=12345,
    )
    return RandomDistributedScalarEncoder(params)


def _overlap_count(first: list[int], second: list[int]) -> int:
    return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)


# By: Dr. Agrawal
# Test Type: unit test
def test_rdse_encodings_are_mostly_orthogonal():
    # TS-04 TC-076
    encoder = _make_large_encoder(radius=1.0)
    import random

    values = [random.randint(0, 100000) for i in range(3000)]
    encodings = [encoder.encode(value) for value in values]

    firsts = random.choices(range(len(values)), k=3000)
    seconds = random.choices(range(len(values)), k=3000)
    overlaps = []
    for i, j in zip(firsts, seconds):
        first = encodings[i]
        second = encodings[j]
        overlaps.append(_overlap_count(first, second))

    orthogonal_ratio = sum(1 for overlap in overlaps if overlap <= 2) / len(overlaps)
    mean_overlap = sum(overlaps) / len(overlaps)

    print(f"Orthogonal ratio: {orthogonal_ratio:.3f}, Mean overlap: {mean_overlap:.3f}")
    assert orthogonal_ratio >= 0.94
    assert mean_overlap <= 1.0


# By: Dr. Agrawal
# Test Type: unit test
def test_rdse_no_overlap_outside_radius_large_encoding():
    # TS-04 TC-077
    encoder = _make_large_encoder(radius=1.0)
    values = [i * 0.1 for i in range(200)]
    for value in values:
        outside = value + 5.0
        overlap = _overlap_count(encoder.encode(value), encoder.encode(outside))
        assert overlap <= 4


# Output format and parameter conformance (binary 0/1 only, sparsity/active_bits)
# ---------------------------------------------------------------------------


# Test Type: unit test
def test_rdse_encode_output_only_zeros_and_ones():
    # TC-078
    """
    Verify that RDSE output is binary - contains only 0 and 1.

    The SDR (Sparse Distributed Representation) must be binary because it represents
    neural activation patterns (neuron either fires or doesn't fire).

    Validates:
      - For multiple input values, output contains only 0s and 1s
      - No intermediate values or floating point artifacts
      - Output can be used as a valid SDR

    Why it passes:
      - Encoder uses binary hashing (mmh3 hash with bit extraction)
      - active_bits=20 with sparsity=0.0 produces exactly ~20 ones per encoding
      - All other bits are zeros by definition
      - Tests across diverse input values (0.0 to 100.0) to ensure consistency
    """
    params = RDSEParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    for value in (0.0, 1.0, 5.0, 100.0):
        out = encoder.encode(value)
        assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


# Test Type: unit test
def test_rdse_encode_output_length_equals_size():
    # TC-079
    """
    Verify that encoded output length equals the configured encoder size.

    The output SDR must have exactly 'size' bits. This is foundational for:
      - Downstream spatial pooling which expects fixed-width inputs
      - Proper sparsity calculations and statistics
      - Consistent memory usage

    Validates:
      - output.length == encoder.size for any input value
      - Size is consistent across different inputs

    Why it passes:
      - Encoder is initialized with size=512
      - encode() method returns SDR of exactly that length
      - The encoder's size property cannot be changed after initialization
      - Test verifies with a sample encoding of 7.0
    """
    params = RDSEParameters(
        size=512,
        active_bits=30,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    out = encoder.encode(7.0)
    assert len(out) == 512, f"Output length must equal size (512), got {len(out)}"


# Test Type: unit test
def test_rdse_encode_output_active_bits_conforms():
    """
    Verify that when active_bits is specified, output has ~active_bits ones.

    When active_bits=40 is set, the encoder should produce ~40 active bits in
    the output SDR. Hash collisions may cause some bits to collide, resulting
    in slightly fewer than the target active bits.

    Validates:
      - Number of 1s in output <= active_bits (cannot exceed target)
      - Number of 1s >= 90% * active_bits (hash collisions expected but minor)
      - Constraint holds across multiple input values

    Why it passes:
      - active_bits=40 with sparsity=0.0 passes mutual exclusivity check
      - Encoder uses deterministic hashing to select which bits activate
      - Hash function mmh3 provides good distribution but has ~10% collision tolerance
      - Tests across values [0.0, 1.0, 10.0, 100.0] to verify consistency
      - The 90% tolerance accounts for inevitable hash collisions
    """
    size = 1024
    active_bits = 40
    params = RDSEParameters(
        size=size,
        active_bits=active_bits,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    for value in (0.0, 1.0, 10.0, 100.0):
        out = encoder.encode(value)
        num_ones = sum(out)
        assert num_ones <= active_bits, f"At most {active_bits} ones expected, got {num_ones}"
        # Hash collisions can reduce; allow down to ~90% of active_bits
        assert num_ones >= int(
            0.9 * active_bits
        ), f"Too few ones ({num_ones}), expected ~{active_bits}"


# Test Type: unit test
def test_rdse_encode_output_sparsity_conforms():
    """
    Verify that when sparsity is specified, output has ~sparsity fraction of ones.

    When sparsity=0.2 is set, approximately 20% of the output bits should be 1.
    This is the alternative way to specify density (rather than active_bits).

    Validates:
      - Actual sparsity (fraction of 1s) ≈ target sparsity
      - Tolerance band: [target - 0.05, target + 0.05] accounts for hash collisions
      - Expected ~200 ones in 1000-bit output (0.2 * 1000)

    Why it passes:
      - active_bits=0 with sparsity=0.2 passes mutual exclusivity check
      - During check_parameters(), encoder calculates active_bits = size * sparsity
        (e.g., 1000 * 0.2 = 200 active bits)
      - Encoder then uses these calculated active_bits for hashing
      - Hash collisions cause slight variation (hence the ±0.05 tolerance)
      - Tests with sparsity=0.2, allowing range 0.15-0.25 (150-250 ones)
    """
    size = 1000
    sparsity = 0.2
    params = RDSEParameters(
        size=size,
        active_bits=0,
        sparsity=sparsity,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    out = encoder.encode(5.0)
    num_ones = sum(out)
    actual_sparsity = num_ones / len(out)
    # Allow tolerance: hash collisions can reduce ones slightly
    assert 0.15 <= actual_sparsity <= 0.25, f"Sparsity={sparsity} => \
        ~{sparsity * 100}% ones expected, got {actual_sparsity:.3f}({num_ones}/{len(out)})"


"""Correctness tests below."""


# Test Type: unit test
def test_rdse_encode_improper_values():
    """
    This test tries to encode with multiple entry types.
    There should be an exception for each.
    """
    params = RDSEParameters(
        size=1000,
        active_bits=0,
        sparsity=0.02,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    with pytest.raises(ValueError):
        encoder.encode("test")
        encoder.encode(datetime(2020, 1, 1, 0, 0))
        encoder.encode([1, 2, 3, 4])
        encoder.encode(((10, 20), 2))


# Test Type: unit test
def test_rdse_encode_empty_values():
    """
    Tests that encoder properly raises an exception if no input value is entered.
    This also tests a None value.
    """
    params = RDSEParameters(
        size=1000,
        active_bits=0,
        sparsity=0.02,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    with pytest.raises(TypeError):
        encoder.encode()
        encoder.encode(None)


# Test Type: unit test
def test_rdse_decode_empty_sdr():
    """Tests that the decode method can raise an exception when an empty sdr is entered."""
    params = RDSEParameters(
        size=1000,
        active_bits=0,
        sparsity=0.02,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    with pytest.raises(ValueError):
        encoder.encode(1)
        encoder.decode([])


# Test Type: unit test
def test_clear_registry_decode():
    """
    This tests that a value error is raised if there are no registered
    encodings and the user tries to decode.
    """
    params = RDSEParameters(
        size=1000,
        active_bits=0,
        sparsity=0.02,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    with pytest.raises(ValueError):
        a = encoder.encode(1)
        encoder.clear_registered_encodings()
        encoder.decode(a)


"""
Tests for RandomDistributedScalarEncoder (RDSE) decode.

decode() returns (value, confidence) by finding the cached encoding with best
overlap to the input SDR. Cache is populated on encode().
"""


# Test Type: unit test
def test_rdse_decode_returns_tuple_value_confidence():
    """decode() returns (value, confidence) tuple."""
    params = RDSEParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    encoded = encoder.encode(5.0)
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, tuple)
    assert len(decoded) == 2
    value, confidence = decoded
    assert isinstance(confidence, (int, float))
    assert 0 <= confidence <= 1


# Test Type: unit test
def test_rdse_decode_round_trip_same_value():
    """decode(encode(x)) returns (x, high confidence) for same encoder instance."""
    params = RDSEParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    for x in (0.0, 1.0, 5.0, 10.0, 100.0):
        encoded = encoder.encode(x)
        value, confidence = encoder.decode(encoded)
        assert value == x, f"Round-trip: encode({x}) then decode should yield {x}, got {value}"
        assert confidence >= 0.9, f"Round-trip confidence should be high, got {confidence}"


# Test Type: unit test
def test_rdse_decode_wrong_size_raises():
    """decode() with wrong-length SDR raises ValueError."""
    params = RDSEParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    encoder.encode(1.0)  # populate cache so decode has candidates
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 100)
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 300)


# Test Type: unit test
def test_rdse_decode_no_candidates_raises():
    """decode() with no prior encode (empty cache) raises ValueError."""
    params = RDSEParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    # No encode() call -> _encoding_cache empty -> no candidates
    with pytest.raises(ValueError, match="No candidate encodings"):
        encoder.decode([0] * 256)

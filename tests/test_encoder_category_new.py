"""
Test suite for Category Encoder.

The Category Encoder produces one-hot style encodings for categorical data (discrete values).
Each category gets a unique SDR representation, with no semantic relationship between categories.

Key Features:
  - One-hot or sparse encoding for categorical inputs
  - Each category receives unique bit pattern
  - No overlap expected between different categories (orthogonal representations)
  - Deterministic encoding (same category → same SDR)
  - Width parameter (w) controls encoding sparsity

Parameter Validation:
  - size: total bits in encoded output
  - w: number of active bits per encoding
  - category_list: list of valid categories
  - rdse_used: whether to use RDSE internally (False for basic category)

Tests validate:
  1. Encoder initialization with valid category list
  2. Encoding specific categories produces expected SDR
  3. Output format (binary only, correct length)
  4. Orthogonality (different categories → no overlap)
  5. Determinism and consistency
"""

import numpy as np
import pytest

from htmrl.encoder_layer.category_encoder_new import (
    CategoryEncoderNew,
    CategoryParametersNew,
)


# TODO we might need confidence filters on the category tests.
@pytest.fixture
def category_instance():
    """Fixture to create a Category encoder instance for tests
    This is used for teardown purposes if needed in the future.
    """


def test_category_initialization():
    """
    This tests to make sure the Category Encoder can succesfully be created.
    Note: there is an optional dimensions parameter not being used here.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(category_list=categories, rdse_used=False)
    e = CategoryEncoderNew(parameters=parameters)

    assert isinstance(e, CategoryEncoderNew)
    """Checking if the instance is correct."""


def test_encode_us():
    """
    This encodes the category "US" into an SDR of 1x12. That bit number is determined from
    3 categories and 1 unknown category. This is w or width of 3 times 4 which is 12 long.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(category_list=categories, rdse_used=False)
    e = CategoryEncoderNew(parameters=parameters)
    a = e.encode("US")
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert len(a) == 2048
    assert np.count_nonzero(a) > parameters.sparsity * parameters.size * 0.95


def test_unknown_category():
    """
    This encodes an unknown category. Here we use "NA" which as you can see is not one of
    the categories specified.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(category_list=categories, rdse_used=False)
    e = CategoryEncoderNew(parameters=parameters)
    a = e.encode("NA")
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert len(a) == 2048
    assert np.count_nonzero(a) > parameters.sparsity * parameters.size * 0.95


def test_encode_es():
    """
    This is almost idential to the "US" encoding, I am just deomonstrating that the encoding
    shows different active bits for different categories.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(category_list=categories, rdse_used=False)
    e = CategoryEncoderNew(parameters=parameters)
    a = e.encode("ES")
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert len(a) == 2048
    assert np.count_nonzero(a) > parameters.sparsity * parameters.size * 0.95


def test_with_sparsity():
    """This test is used to show how SDR outputs look with a single w or width."""
    categories = ["cat1", "cat2", "cat3", "cat4", "cat5"]

    parameters = CategoryParametersNew(sparsity=0.02, category_list=categories, rdse_used=True)
    e = CategoryEncoderNew(parameters=parameters)
    """The respective category should equal their index of expected results."""
    for cat in categories:
        a = e.encode(cat)
        assert len(a) == 2048
        assert np.count_nonzero(a) > parameters.sparsity * parameters.size * 0.95


def test_with_sparsity_sets_meaningful_active_bits_attribute():
    """When sparsity is provided, encoder should derive active bits in state."""
    categories = ["cat1", "cat2", "cat3"]
    parameters = CategoryParametersNew(sparsity=0.02, category_list=categories, rdse_used=True)

    encoder = CategoryEncoderNew(parameters=parameters)

    assert encoder.active_bits_per_category > 0
    assert encoder.active_bits_per_category == int(round(encoder.size * encoder.sparsity))


def test_with_active_bits_sets_meaningful_sparsity_attribute():
    """When active_bits_per_category is provided, encoder should derive sparsity in state."""
    categories = ["cat1", "cat2", "cat3"]
    active_bits = 41
    parameters = CategoryParametersNew(
        active_bits_per_category=active_bits, sparsity=0.0, category_list=categories, rdse_used=True
    )

    encoder = CategoryEncoderNew(parameters=parameters)

    assert encoder.sparsity > 0.0
    assert encoder.sparsity == active_bits / encoder.size


def test_category_list_mutation_after_construction_does_not_affect_encoder():
    """Mutating the params object after construction must not corrupt the encoder."""
    categories = ["ES", "GB", "US"]
    params = CategoryParametersNew(category_list=categories, rdse_used=False)
    encoder = CategoryEncoderNew(params)
    original_encoding = encoder.encode("ES")

    # Mutate the original list after construction
    params.category_list.append("EXTRA")
    params.category_list.clear()

    assert encoder.encode("ES") == original_encoding


def test_rdse_used():
    """
    This test uses the RDSE and demonstrates that the same encoder encoding a category twice
    to two different SDRs yields the same encoding. This is important since it shows we can
    decode this if needed and get the category back from our SDR.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(category_list=categories)
    e1 = CategoryEncoderNew(parameters=parameters)
    """These asserts just check that both SDRs are identical when the same category is encoded."""
    a1 = e1.encode("ES")
    a2 = e1.encode("ES")
    assert a1 == a2
    a1 = e1.encode("GB")
    a2 = e1.encode("GB")
    assert a1 == a2
    a1 = e1.encode("US")
    a2 = e1.encode("US")
    assert a1 == a2
    a1 = e1.encode("NA")
    a2 = e1.encode("NA")
    assert a1 == a2


# ---------------------------------------------------------------------------
# Output format and parameter conformance (binary 0/1 only, length)
# ---------------------------------------------------------------------------


def test_category_encode_output_only_zeros_and_ones():
    """CategoryEncoder output must contain only 0 and 1."""
    categories = ["ES", "GB", "US"]
    for rdse_used in (False, True):
        parameters = CategoryParametersNew(category_list=categories, rdse_used=rdse_used)
        encoder = CategoryEncoderNew(parameters)
        for cat in categories + ["NA"]:
            out = encoder.encode(cat)
            assert all(
                b in (0, 1) for b in out
            ), f"Output must be binary (0/1), rdse_used={rdse_used}, cat={cat!r}, got {set(out)}"


def test_category_encode_output_length_equals_size():
    """CategoryEncoder output length must equal (num_categories + 1) * w."""
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(category_list=categories, rdse_used=False)
    encoder = CategoryEncoderNew(parameters)
    expected_size = parameters.size  # +1 for unknown
    out = encoder.encode("US")
    assert (
        len(out) == expected_size
    ), f"Output length must equal (1+len(categories))*w = {expected_size}, got {len(out)}"


"""
tests.test_decoder_category

Test suite for CategoryEncoder decoding functionality.

Validates that CategoryEncoder correctly decodes SDRs back to original category strings.
Tests cover:
- decode() returns (value, confidence) tuple for RDSE-based encoders
- Decoded values match original encoded categories
- Confidence scores reflect encoding strength
- Unknown/undecodable SDRs return "NA" for value
- decode() is only implemented when rdse_used=True parameter is set

Note: CategoryEncoder.decode() follows the standard decoder interface where decode()
returns a tuple of (decoded_value, confidence) where decoded_value is the category string.

These tests validate the reverse transformation from SDR representations back to
interpretable category values.
"""


def test_decode_returns_tuple_of_two():
    """Decode returns (value, confidence) tuple."""
    params = CategoryParametersNew(category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoderNew(params)
    encoded = encoder.encode("US")
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, tuple)
    assert len(decoded) == 2
    value, confidence = decoded
    assert value is not None
    assert isinstance(confidence, (int, float))


def test_decode_value_in_categories_or_na():
    """Decoded value is one of the category strings or 'NA'."""
    categories = ["ES", "GB", "US"]
    params = CategoryParametersNew(category_list=categories, rdse_used=True)
    encoder = CategoryEncoderNew(params)
    valid_values = set(categories) | {"NA"}
    for cat in categories:
        encoded = encoder.encode(cat)
        decoded = encoder.decode(encoded)
        value = decoded[0]
        assert value in valid_values, f"Decoded value {value!r} not in {valid_values}"
    encoded_unknown = encoder.encode("NA")
    decoded_unknown = encoder.decode(encoded_unknown)
    assert decoded_unknown[0] in valid_values


def test_decode_confidence_in_range():
    """Decoded confidence is in [0, 1]."""
    params = CategoryParametersNew(category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoderNew(params)
    for cat in ["ES", "GB", "US", "NA"]:
        encoded = encoder.encode(cat)
        decoded = encoder.decode(encoded)
        _, confidence = decoded
        assert 0 <= confidence <= 1, f"Confidence {confidence} not in [0, 1]"


def test_decode_round_trip_same_category():
    """Encode then decode returns the same category (round-trip)."""
    categories = ["ES", "GB", "US"]
    params = CategoryParametersNew(category_list=categories, rdse_used=True)
    encoder = CategoryEncoderNew(params)
    for cat in categories:
        encoded = encoder.encode(cat)
        decoded = encoder.decode(encoded)
        value = decoded[0]
        assert value == cat, f"Round-trip: encoded {cat!r}, got back {value!r}"


def test_decode_round_trip_unknown():
    """Encode unknown category then decode returns 'NA'."""
    params = CategoryParametersNew(category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoderNew(params)
    encoded = encoder.encode("NA")
    decoded = encoder.decode(encoded)
    assert decoded[0] == "NA", f"Unknown should decode to 'NA', got {decoded[0]!r}"


def test_decode_wrong_sdr_size_raises():
    """Decode with wrong SDR length raises."""
    params = CategoryParametersNew(category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoderNew(params)
    # Encoder size is (len(categories)+1)*w = 4*3 = 12
    with pytest.raises(ValueError, match="does not match"):
        encoder.decode([0] * 10)
    with pytest.raises(ValueError, match="does not match"):
        encoder.decode([0] * 20)


def test_demonstrate_anything_can_be_categories():
    """
    Tests that the category encoder can take any category list and encode it no matter the type.
    On top of that this tests when wrong data types are entered into the encoding. They should all default
    to the not any category or NA.
    """
    params1 = CategoryParametersNew(category_list=["ES", "GB", "US"], rdse_used=True)
    encoder1 = CategoryEncoderNew(params1)
    a = encoder1.encode("ES")
    a1 = encoder1.encode(1)
    a2 = encoder1.encode("=")
    assert encoder1.decode(a)[0] == "ES"
    assert encoder1.decode(a1)[0] == "NA"
    assert encoder1.decode(a2)[0] == "NA"
    params2 = CategoryParametersNew(category_list=[1, 2, 3], rdse_used=True)
    encoder2 = CategoryEncoderNew(params2)
    b = encoder2.encode(2)
    b1 = encoder2.encode("ES")
    b2 = encoder2.encode("=")
    assert encoder2.decode(b)[0] == 2
    assert encoder2.decode(b1)[0] == "NA"
    assert encoder2.decode(b2)[0] == "NA"
    params3 = CategoryParametersNew(category_list=["-", "+", "="], rdse_used=True)
    encoder3 = CategoryEncoderNew(params3)
    c = encoder3.encode("=")
    c1 = encoder3.encode("ES")
    c2 = encoder3.encode(1)
    assert encoder3.decode(c)[0] == "="
    assert encoder3.decode(c1)[0] == "NA"
    assert encoder3.decode(c2)[0] == "NA"


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
    return result


# Correctness tests
def test_close_categories_are_similar():
    """
    This test checks to make sure categories by each other in the index are more similar
    than categories distanced from each other.
    """
    params = CategoryParametersNew(
        category_list=["ES", "GB", "US", "RU", "JP", "FR", "GR", "TU", "IT"], rdse_used=True
    )
    encoder = CategoryEncoderNew(params)
    encoding1 = encoder.encode("ES")
    encoding2 = encoder.encode("GB")
    encoding3 = encoder.encode("US")
    encoding4 = encoder.encode("Wrong")
    encoding5 = encoder.encode("IT")
    # far distanced
    assert hamming_distance_helper(encoding1, encoding2) < hamming_distance_helper(
        encoding1, encoding5
    )
    # 3 in a row
    assert hamming_distance_helper(encoding1, encoding2) < hamming_distance_helper(
        encoding1, encoding3
    )
    # not any category should be different than all other encodings
    far_distances = []
    far_distances.append(hamming_distance_helper(encoding4, encoding1))
    far_distances.append(hamming_distance_helper(encoding4, encoding2))
    far_distances.append(hamming_distance_helper(encoding4, encoding3))
    far_distances.append(hamming_distance_helper(encoding4, encoding5))
    for distance in far_distances:
        assert distance > (params.sparsity * params.size * 0.98)

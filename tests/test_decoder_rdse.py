"""
Tests for RandomDistributedScalarEncoder (RDSE) decode.

The decode() method reconstructs the original input value from an SDR by finding
the cached encoding with the best overlap (highest number of matching bits).

Return Format Change:
  - decode() now returns (value, confidence) tuple instead of a single value
  - confidence indicates how well the SDR matched the best cached encoding
  - Range: 0.0 (complete mismatch) to 1.0 (perfect match)

Cache Mechanism:
  - Cache is populated during encode() calls
  - decode() searches cache for encoding with maximum overlap
  - First decode() after init raises error (cache empty)
  - decode(encode(x)) typically returns (x, ~1.0) - perfect round-trip

Validation Points:
  1. Return type is tuple with exactly (value, confidence)
  2. Round-trip: encode(x) → decode(SDR) returns (x, high_confidence)
  3. Input validation: SDR length must match encoder size
  4. Cache requirement: must encode() values before decoding
  5. RDSE parameters: all tests use sparsity=0.0 with active_bits to satisfy
     the new mutual exclusivity constraint (exactly one must be set)
"""

import pytest

from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


# Test Type: unit test
def test_rdse_decode_returns_tuple_value_confidence():
    # TS-15 TC-104
    """
    Verify decode() returns (value, confidence) tuple (not single value).

    This validates the API change from returning just a value to returning
    a tuple that includes confidence information.

    Validates:
      - Return type is tuple
      - Tuple has exactly 2 elements
      - First element is the reconstructed value
      - Second element is float confidence in range [0.0, 1.0]

    Why it passes:
      - active_bits=20 with sparsity=0.0 satisfies mutual exclusivity constraint
      - encoder.encode(5.0) populates cache with this value
      - encoder.decode(SDR) finds matching cached encoding and returns (value, confidence)
      - Decoder implementation returns tuple with exact format expected
    """
    params: RDSEParameters = RDSEParameters(
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
    # TS-15 TC-105
    """
    Verify round-trip encoding/decoding returns original value with high confidence.

    This is the ideal case: encode a value, then decode the result should yield
    the same value with high confidence (~1.0).

    Validates:
      - decode(encode(x)) returns (x, confidence >= 0.9) for any x in test range
      - Works consistently across values: 0.0, 1.0, 5.0, 10.0, 100.0
      - Confidence is consistently high (no false positives)

    Why it passes:
      - active_bits=20 with sparsity=0.0 satisfies mutual exclusivity constraint
      - For each value x:
        1. encode(x) stores SDR in cache and returns it
        2. decode(same_SDR) finds exact match in cache
        3. Perfect overlap → confidence ~1.0
        4. Exact match → returns original value x
      - Tests across diverse input range to confirm consistency
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
    for x in (0.0, 1.0, 5.0, 10.0, 100.0):
        encoded = encoder.encode(x)
        value, confidence = encoder.decode(encoded)
        assert value == x, f"Round-trip: encode({x}) then decode should yield {x}, got {value}"
        assert confidence >= 0.9, f"Round-trip confidence should be high, got {confidence}"


# Test Type: unit test
def test_rdse_decode_wrong_size_raises():
    # TS-15 TC-106
    """
    Verify decode() validates input SDR length matches encoder size.

    The decoder expects SDR input to have exactly 'size' bits. Providing wrong
    length SDR is an error condition.

    Validates:
      - decode([0] * 100) raises ValueError when encoder size=256
      - decode([0] * 300) raises ValueError when encoder size=256
      - Error message contains "does not match encoder size"

    Why it passes:
      - active_bits=20 with sparsity=0.0 satisfies mutual exclusivity constraint
      - encoder.encode(1.0) populates cache so decode has candidates to search
      - decode() first validates input length == self.size
      - ValueError raised immediately on length mismatch, before cache search
      - Tests both "too short" (100) and "too long\" (300) cases
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
    encoder.encode(1.0)  # populate cache so decode has candidates
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 100)
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 300)


# Test Type: unit test
def test_rdse_decode_no_candidates_raises():
    # TS-15 TC-107
    """
    Verify decode() requires prior encode() to populate cache.

    The decoder uses a cache of previously encoded values to find the best match
    for an input SDR. Without any prior encode(), the cache is empty.

    Validates:
      - decode() with empty cache raises ValueError
      - Error message contains "No candidate encodings"
      - This prevents silent failures (can't decode without training data)

    Why it passes:
      - active_bits=20 with sparsity=0.0 satisfies mutual exclusivity constraint
      - Encoder is created but no encode() call is made
      - _encoding_cache remains empty
      - decode([0] * 256) finds no candidates in cache
      - ValueError raised with "No candidate encodings" error message
      - This is expected behavior - decoder needs training (cache population) first
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
    # No encode() call -> _encoding_cache empty -> no candidates
    with pytest.raises(ValueError, match="No candidate encodings"):
        encoder.decode([0] * 256)

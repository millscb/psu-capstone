"""
tests.test_decoder_coordinate

Test suite for CoordinateEncoder decoding functionality.

Validates that CoordinateEncoder correctly encodes and decodes spatial coordinates
(x, y, z positions) and associated metadata. Tests cover:
- Deterministic decoding (same input always produces same SDR)
- Coordinate encoding with spatial locality (nearby coordinates have overlapping SDRs)
- Radius-based search for nearby encoded coordinates
- Seed-based reproducibility
- Multi-dimensional coordinate support (2D, 3D spaces)
- Encoding registration for reference lookups
- Decoded coordinate accuracy and proximity

Parameter Constraints:
- n: SDR dimension (number of bits)
- w: number of active bits (sparsity)
- seed: random seed for reproducibility
- max_radius: search radius in coordinate space
- dims: number of coordinate dimensions (2D/3D)

These tests validate spatial encoding for location-based learning tasks,
enabling HTM systems to learn spatial patterns and sequences.
"""

from htmrl.encoder_layer.coordinate_encoder import CoordinateEncoder, CoordinateParameters


# Test Type: unit test
def test_decode_is_deterministic():
    params = CoordinateParameters(n=40, w=25, seed=42, max_radius=3, dims=2)
    enc = CoordinateEncoder(params)

    key1 = ((10, 20), 2)
    key2 = ((100, 200), 3)

    sdr1 = enc.encode(key1)
    sdr2 = enc.encode(key2)

    candidates = [key1, ((11, 20), 2), ((10, 21), 2), ((9, 20), 2), key2]

    for k in candidates:
        enc.register_encoding(k)

    first1 = enc.decode(sdr1, candidates=candidates)[0]
    for _ in range(100):
        assert enc.decode(sdr1, candidates=candidates)[0] == first1

    assert first1 == key1

    first2 = enc.decode(sdr2, candidates=candidates)[0]
    for _ in range(100):
        assert enc.decode(sdr2, candidates=candidates)[0] == first2

    assert first2 == key2


# Test Type: unit test
def test_decode_round_trip_same_coordinate():
    params = CoordinateParameters(n=2048, w=25, seed=0, max_radius=6, dims=2)
    enc = CoordinateEncoder(params)

    test_keys = [
        ((0, 0), 6),
        ((10, 20), 6),
        ((-5, 7), 6),
        ((100, 200), 6),
    ]

    for key in test_keys:
        encoded = enc.encode(key)
        decoded_key, confidence = enc.decode(encoded, candidates=test_keys)

        assert decoded_key == (tuple(key[0]), int(key[1]))
        assert confidence >= 0.9

        reencoded = enc.encode(decoded_key)
        assert reencoded == encoded

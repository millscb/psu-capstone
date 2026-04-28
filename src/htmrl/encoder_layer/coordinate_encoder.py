from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Iterable, override

import mmh3
import numpy as np

from htmrl.encoder_layer.base_encoder import BaseEncoder, ParameterMarker
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


class CoordinateEncoder(BaseEncoder[tuple[tuple[int, ...] | list[int], int]]):
    """Encode integer grid coordinates and radius into an SDR."""

    def __init__(
        self, parameters: CoordinateParameters | None = None, dimensions: list[int] | None = None
    ):

        self._parameters = (
            copy.deepcopy(parameters) if parameters is not None else CoordinateParameters()
        )

        self._n = self._parameters.n
        self._w = self._parameters.w
        self._seed = self._parameters.seed
        self._dims = self._parameters.dims
        self._max_radius = self._parameters.max_radius
        self._use_all_neighbors = self._parameters.use_all_neighbors

        enc_params = RDSEParameters(
            size=self._n,
            active_bits=self._w,
            sparsity=0.0,
            radius=0.0,
            resolution=1.0 / self._n,
            category=False,
            seed=self._seed,
        )

        self._encoder = RandomDistributedScalarEncoder(enc_params)
        self._overlap = self._encoder._overlap
        self._encoding_cache: dict[tuple[tuple[int, ...], int], list[int]] = {}
        self._max_neighbors = (2 * self._parameters.max_radius + 1) ** self._dims

        if self._parameters.use_all_neighbors:
            # all neighbors
            self._size = self._max_neighbors * self._n
        else:
            # all winners
            self._size = self._w * self._n

        super().__init__(self._size)

    @override
    def encode(self, input_value: tuple[list[int], int]) -> list[int]:
        """Encode ``(coordinate, radius)`` into a fixed-size binary SDR."""
        return self.register_encoding(input_value)

    def _compute_encoding(self, key: tuple[list[int], int]) -> list[int]:
        coordinate, radius = key

        if len(coordinate) != self._dims:
            raise ValueError(f"Expected coordinate with dims={self._dims}, got {len(coordinate)}")

        r = int(radius)
        if r < 0 or r > self._max_radius:
            raise ValueError(f"Radius must be in [0, {self._max_radius}], got {r}")

        neighbors = self._neighbors(coordinate, r)

        out: list[int] = []
        if self._use_all_neighbors:
            for c in neighbors:
                v = self._coord_to_unit_float(c)
                out.extend(int(b) for b in self._encoder.encode(v))

            expected = self._max_neighbors * self._n
        else:
            winners = self._topwcoordinates(neighbors, self._w)

            for c in winners:
                v = self._coord_to_unit_float(c)
                out.extend(int(b) for b in self._encoder.encode(v))

            expected = self._w * self._n

        if len(out) < expected:
            out.extend([0] * (expected - len(out)))
        else:
            out = out[:expected]

        return out

    def register_encoding(
        self, input_value: tuple[list[int], int], encoded: list[int] | None = None
    ) -> list[int]:
        """Cache and return the encoding for a coordinate/radius key."""
        coordinate, radius = input_value

        key = (tuple(int(v) for v in coordinate), int(radius))
        vector = encoded if encoded is not None else self._compute_encoding(key)

        if len(vector) != self._size:
            raise ValueError("Stored encoding must match encoder size")

        self._encoding_cache[key] = vector
        return vector

    @staticmethod
    def _neighbors(coordinate, radius):
        ranges = (range(int(n) - radius, int(n) + radius + 1) for n in coordinate)
        return itertools.product(*ranges)

    @classmethod
    def _topwcoordinates(cls, coordinates, w):
        scored = [(cls._order_for_coordinate(c), c) for c in coordinates]

        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored[-w:]]

    @staticmethod
    def _order_for_coordinate(coordinate) -> float:
        s = ",".join(str(int(v)) for v in coordinate)
        h = mmh3.hash(s, signed=False)
        return h / 2**32

    def _coord_to_unit_float(self, coordinate) -> float:
        s = ",".join(str(int(v)) for v in coordinate)
        h = mmh3.hash(s, seed=self._seed, signed=False)
        return h / 2**32

    @override
    def decode(
        self,
        encoded: list[int],
        candidates: Iterable[tuple[list[int], int]] | None = None,
    ) -> tuple[tuple[list[int], int] | None, float]:
        """Decode an SDR to the nearest cached coordinate/radius candidate."""
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        search_keys = (
            list(candidates) if candidates is not None else list(self._encoding_cache.keys())
        )
        if not search_keys:
            raise ValueError("No candidate encodings are available for decoding")

        best_key = None
        best_overlap = -1

        for key in search_keys:
            candidate_encoding = self._encoding_cache.get(key)
            if candidate_encoding is None:
                candidate_encoding = self.register_encoding(key)

            overlap = self._overlap(encoded, candidate_encoding)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        if self._use_all_neighbors:
            blocks = (2 * self._max_radius + 1) ** self._dims
        else:
            blocks = self._w
        expected_ones = blocks * self._w
        confidence = best_overlap / expected_ones if expected_ones > 0 else 0.0
        return best_key, confidence


@dataclass
class CoordinateParameters(ParameterMarker):
    """Configuration parameters for :class:`CoordinateEncoder`."""

    size: int | None = None
    n: int = 2048
    w: int = 25
    seed: int = 42
    max_radius: int = 10
    dims: int = 2
    use_all_neighbors: bool = False
    encoder_class = CoordinateEncoder

    def __post_init__(self) -> None:
        # Backward compatibility: some callers pass `size` instead of `n`.
        if self.size is not None:
            self.n = int(self.size)


if __name__ == "__main__":
    params = CoordinateParameters(n=400, w=25, max_radius=6)
    enc = CoordinateEncoder(params)

    params1 = CoordinateParameters(n=2048, w=20, max_radius=2)
    enc2 = CoordinateEncoder(params1)

    a = enc2.encode(((10, 20), 2))
    b = enc.encode(((11, 20), 2))
    c = enc.encode(((0, 0), 5))

    print(enc2.decode(a))
    print(enc.decode(c))

    print()

    test_keys = [
        ((0, 0), 6),
        ((10, 20), 6),
        ((-5, 7), 6),
        ((100, 200), 6),
    ]

    for key in test_keys:
        encoded = enc.encode(key)
        print(enc.decode(encoded))

    params2 = CoordinateParameters(n=400, w=20, max_radius=3, dims=3)

    enc3 = CoordinateEncoder(params2)

    d = enc3.encode(((10, 20, 30), 2))
    e = enc3.encode(((10, 20, 30), 2))

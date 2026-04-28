"""Random Distributed Scalar Encoder (RDSE) for flexible numeric encoding.

This module provides the RandomDistributedScalarEncoder, which encodes numeric
scalar values into SDRs using random, stable hash-based assignments. Unlike
ScalarEncoder, RDSE does not require knowing the input range in advance and
determines encodings at runtime using the MurmurHash3 algorithm.

Parameter constraints:
- "activeBits" & "sparsity" are mutually exclusive - specify exactly one
- "radius", "resolution", & "category" are mutually exclusive - specify exactly one

Based on NuPIC's RDSE implementation.
"""

from __future__ import annotations

import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import Iterable, override

import mmh3
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.log import get_logger


class RandomDistributedScalarEncoder(BaseEncoder[float | int]):
    """Random Distributed Scalar Encoder using MurmurHash3 for stable encodings.

    The RDSE encodes numeric scalar values into SDRs using hash-based random
    bit selection. It provides more flexibility than ScalarEncoder by not
    requiring pre-specified input ranges and determining encodings at runtime.
    """

    @property
    def encoding_cache(self) -> dict[float, list[int]]:
        """Public accessor for the encoding cache (for compatibility)."""
        return self._encoding_cache

    def __init__(self, parameters: RDSEParameters):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._size = self._parameters.size
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution
        self._category = self._parameters.category
        self._seed = self._parameters.seed
        self._encoding_cache: dict[float, list[int]] = {}
        self.knn: KNeighborsRegressor
        self.encoding: bool = False

        self.logger = get_logger(self)

        super().__init__(self._size)

    @override
    def encode(self, input_value: float | int) -> list[int]:
        """Encode the input value into a binary vector."""
        if type(input_value) is not int and type(input_value) is not float:
            raise ValueError("A scalar encoder can only encode floats or ints.")
        self.register_encoding(input_value)
        self.logger.info("RDSE encoded value: %s", input_value)
        return self._compute_encoding(input_value)

    def register_encoding(
        self, input_value: float | int, encoded: list[int] | None = None
    ) -> list[int]:
        """Caches an encoding so decode_closest can compare against it."""
        vector = encoded if encoded is not None else self._compute_encoding(input_value)
        if len(vector) != self.size:
            raise ValueError("Stored encoding must be the same length as encoder size")
        self._encoding_cache[input_value] = vector
        return vector

    def clear_registered_encodings(self) -> None:
        """Clears cached encodings used for nearest-neighbor decoding."""
        self._encoding_cache.clear()

    def _compute_encoding(self, input_value: float) -> list[int]:
        """
        Uses murmurhash3 algorithm to encode a float value.

        :param self: Description
        :param input_value: The value we want encoded.
        :type input_value: float
        :return: Returns a list of integers that signify an SDR.
        :rtype: list[int]
        """
        if self._category and input_value < 0:
            raise ValueError("Input to category encoder must be a non-negative integer")

        self.encoding = True

        # setup buffer of zeros to be filled in with 1s at the appropriate indices
        data = [0] * self.size

        assert self._resolution > 0, "Resolution must be greater than 0 to compute encoding"
        index = math.floor(input_value / self._resolution)

        for offset in range(self._active_bits):
            hash_buffer = index + offset

            """
                The lower case i in the struct.pack makes this take in signed 32 bit integers.
                It is important to note the previous iteration used an upper case I which
                made this not take negative values. The struct.pack converts an integer
                into a byte representation.
            """
            bucket = mmh3.hash(struct.pack("i", hash_buffer), self._seed, signed=False)
            bucket = bucket % self.size
            data[bucket] = 1
        return data

    @staticmethod
    def _overlap(first: list[int], second: list[int]) -> int:
        """
        Checks for overlapping bits in two Lists of integers.

        :param first: The first list in checking.
        :type first: list[int]
        :param second: The second list in checking.
        :type second: list[int]
        :return: Returns the overlapping bits.
        :rtype: int
        """
        if len(first) != len(second):
            raise ValueError("Vectors must be the same length to compute overlap")
        return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)

    def sparsify(self, vector: list[int]) -> list[int]:
        """Converts a sparse activity vector to a activation list."""
        return [i for i, bit in enumerate(vector) if bit == 1]

    @override
    def decode(
        self, encoded: list[int], candidates: Iterable[float] | None = None
    ) -> tuple[float | None, float]:
        """Returns the value whose encoding overlaps the most with the provided SDR."""
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        search_values = (
            list(candidates) if candidates is not None else list(self._encoding_cache.keys())
        )
        if not search_values:
            raise ValueError("No candidate encodings available for decoding")

        best_value: float | None = None
        best_overlap = -1

        for candidate in search_values:
            candidate_encoding = self._encoding_cache.get(candidate)
            if candidate_encoding is None:
                candidate_encoding = self.register_encoding(candidate)
            overlap = self._overlap(encoded, candidate_encoding)
            if overlap > best_overlap:
                best_overlap = overlap
                best_value = candidate

        confidence = (
            best_overlap / self._active_bits if best_overlap >= 0 and self._active_bits else 0.0
        )
        if best_overlap == 0:
            best_value = None
        self.logger.info("Decoded SDR into value: %s, with confidence: %s", best_value, confidence)
        return best_value, confidence

    def make_knn(self) -> None:
        """
        Alternate method to decode where it employs a knn regressor model.

        :param self: Description
        """
        if not self._encoding_cache:
            raise ValueError("No registered encodings available to build KNN")

        x = np.array(list(self._encoding_cache.values()), dtype=np.uint8)
        y = np.array(list(self._encoding_cache.keys()), dtype=np.float32)

        knn = KNeighborsRegressor(
            n_neighbors=min(2, len(y)),
            weights="distance",
            metric="hamming",
        )
        knn.fit(x, y)
        self.knn = knn

    def decode_knn(self, encoded: list[int]) -> float:
        """Returns the value whose encoding overlaps the most with the provided SDR."""
        if len(encoded) != self.size:
            raise ValueError("Encoded input must match encoder size")
        if self.encoding:
            self.make_knn()
            self.encoding = False
        query = np.asarray(encoded, dtype=np.int8).reshape(1, -1)
        result = self.knn.predict(query)
        return result.item()

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: "RDSEParameters"):
        """Method to check mutually exclusive parameters and fill in missing values.

        Args:
            parameters: The parameters to check and fill in.

        Returns:
            The checked and filled in parameters.

        Raises:
            ValueError: If the parameters are invalid.
        """
        # Check size parameter
        if not parameters.size > 0:
            raise ValueError("You have no size set.")

        num_active_args = 0

        # Check active bits / sparsity mutual exclusivity

        if parameters.active_bits > 0:
            num_active_args += 1
        if parameters.sparsity > 0.0:
            num_active_args += 1

        if num_active_args == 0:
            raise ValueError("Missing argument, need one of: 'activeBits' or 'sparsity'.")
        if num_active_args != 1:
            raise ValueError("Too many arguments, choose only one of: 'activeBits' or 'sparsity'.")

        # Check radius / resolution / category mutual exclusivity
        num_resolution_args = 0
        if parameters.radius > 0.0:
            num_resolution_args += 1
        if parameters.category:
            num_resolution_args += 1
        if parameters.resolution > 0.0:
            num_resolution_args += 1

        if num_resolution_args == 0:
            raise ValueError("Missing argument, need one of: 'radius', 'resolution', 'category'.")
        if num_resolution_args != 1:
            raise ValueError(
                "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."
            )

        # Finish filling in all of parameters.

        # Determine number of activeBits.
        if parameters.sparsity > 0:
            if not 0 <= parameters.sparsity <= 1:
                raise ValueError("Sparsity is not between 0 and 1 inclusive.")
            parameters.active_bits = int(round(parameters.size * parameters.sparsity))
            if not parameters.active_bits > 0:
                raise ValueError("Active bits are not greater than 0.")

        # Determine sparsity. Always calculate this even if it was given, to correct for rounding error.
        parameters.sparsity = float(parameters.active_bits / parameters.size)

        if parameters.category:
            parameters.active_bits = 1
        # Determine resolution.
        elif parameters.radius > 0.0:
            parameters.resolution = parameters.radius / float(parameters.active_bits)

        # Determine radius.
        elif parameters.resolution > 0.0:
            parameters.radius = float(parameters.active_bits) * parameters.resolution

        while parameters.seed == 0:
            parameters.seed = random.getrandbits(32)
        return parameters


@dataclass
class RDSEParameters:
    """Configuration parameters for :class:`RandomDistributedScalarEncoder`."""

    size: int = 2048
    """
    * Member "size" is the total number of bits in the encoded output SDR.
    """
    active_bits: int = 0
    """
    * Member "activeBits" is the number of true bits in the encoded output SDR.
    """
    sparsity: float = 0.02
    """
    * Member "sparsity" is the fraction of bits in the encoded output which this
    * encoder will activate. This is an alternative way to specify the member
    * "activeBits".
    """
    radius: float = 0.0
    """
    * Member "radius" Two inputs separated by more than the radius have
    * non-overlapping representations. Two inputs separated by less than the
    * radius will in general overlap in at least some of their bits. You can
    * think of this as the radius of the input.
    """
    resolution: float = 1.0
    """
    * Member "resolution" Two inputs separated by greater than, or equal to the
    * resolution will in general have different representations.
    """
    category: bool = False
    """
    * Member "category" means that the inputs are enumerated categories.
    * If true then this encoder will only encode unsigned integers, and all
    * inputs will have unique / non-overlapping representations.
    """
    seed: int = 42
    """
    * Member "seed" forces different encoders to produce different outputs, even
    * if the inputs and all other parameters are the same.  Two encoders with the
    * same seed, parameters, and input will produce identical outputs.
    *
    * The seed 0 is special.  Seed 0 is replaced with a random number.
    """
    encoder_class = RandomDistributedScalarEncoder


if __name__ == "__main__":
    params = RDSEParameters(
        size=2048,
        sparsity=0.02,
        radius=0.0,
        active_bits=0,
        category=False,
        seed=12345,
    )
    encoder = RandomDistributedScalarEncoder(params)
    a = encoder.encode(0.1)
    b = encoder.encode(5.1)

    def _overlap_count(first: list[int], second: list[int]) -> int:
        return int(np.count_nonzero(first == second))

    # print(_overlap_count(a, b))
    encoder.decode(a)
    encoder.decode(b)
    # print(encoder.decode(a))
    # print(encoder.decode(b))
    # Tests
    """
    params = RDSEParameters(
        size=2048, active_bits=0, sparsity=0.02, radius=0.0, resolution=1.0, category=False, seed=42
    )
    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(10.0, o1)
    print("Encoded 10.0 \n")
    print("Decode prediction is: \n")

    e2 = RandomDistributedScalarEncoder(params)

    o2 = SDR([e2.size])
    e2.encode(1.0, o2)
    print("Sparse is: \n")

    params2 = RDSEParameters(
        size=2048, active_bits=0, sparsity=0.02, radius=0.0, resolution=1.0, category=False, seed=42
    )
    e3 = RandomDistributedScalarEncoder(params2)
    o3 = SDR([e3.size])
    e3.encode(10.0, o3)
    print("Sparse is: \n")

    encoder = RandomDistributedScalarEncoder()
    output = SDR([encoder.size])
    encoder.encode(10.0, output)
    print("Sparse is: \n")
    output2 = SDR([encoder.size])
    encoder.encode(10.0, output2)
    print("Sparse is: \n")

    encoder = RandomDistributedScalarEncoder()
    output = SDR([encoder.size])

    train_values = [random.uniform(0.0, 100.0) for _ in range(200)]
    for v in train_values:
        encoder.encode(v, output)
    test_values = [random.uniform(0.0, 100.0) for _ in range(20)]
    print("__value_____predicted___error")
    y_true = []
    y_pred = []
    for v in test_values:
        encoder.encode(v, output)
        pred = encoder.decode(output)
        y_true.append(v)
        y_pred.append(pred)
        error = pred - v
        print(f"{v:7.3f}   {pred:9.3f}   {error:+7.3f}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print("RMSE:", rmse)
    """

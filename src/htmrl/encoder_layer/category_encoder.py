"""Category encoder for discrete, unrelated categorical values.

This module provides an encoder for discrete categories represented by strings
that have no inherent relationship or ordering. Each category is encoded to
a distinct, non-overlapping sparse distributed representation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, cast, override

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.log import get_logger


class CategoryEncoder(BaseEncoder[str]):
    """Encoder for discrete categorical values with no semantic relationships.

    This encoder converts string categories into sparse distributed representations
    where each category maps to a distinct, non-overlapping pattern. Unlike scalar
    encoders, there is no similarity between different categories' representations.

    The encoder reserves index 0 for unknown categories not in the category list.
    Internally, it uses either a ScalarEncoder or RandomDistributedScalarEncoder
    with appropriate parameters to ensure non-overlapping encodings.

    Args:
        parameters: Configuration specifying categories and encoder settings.
    """

    def __init__(self, parameters: CategoryParameters):
        self._parameters = self.check_parameters(copy.deepcopy(parameters))
        self._w = self._parameters.w
        self._category_list = list(self._parameters.category_list)
        self._RDSEused = self._parameters.rdse_used
        self._num_categories = len(self._category_list) + 1
        self.size = self._num_categories * self._w
        self.logger = get_logger(self)

        super().__init__(self._size)
        # Configure RDSE for random distributed encoding
        if self._RDSEused:
            self.rdsep = RDSEParameters(
                size=self._num_categories * self._w,
                active_bits=self._w,
                sparsity=0.0,
                radius=1.0,
                resolution=0.0,
                category=False,
                seed=0,
            )
            self.encoder = RandomDistributedScalarEncoder(self.rdsep)
        # Configure standard ScalarEncoder for deterministic encoding
        else:
            self.sp = ScalarEncoderParameters(
                minimum=0,
                maximum=int(self._num_categories - 1),
                clip_input=False,
                periodic=False,
                category=False,
                active_bits=self._w,
                sparsity=0.0,
                size=self._num_categories * self._w,
                radius=0.0,
                resolution=1.0,
            )
            self.encoder = ScalarEncoder(self.sp)

    @override
    def encode(self, input_value: str) -> list[int]:
        """Encode a category string into a sparse distributed representation.

        Maps the input category to its index in the category list (or 0 for
        unknown categories) and delegates to the underlying encoder.

        Args:
            input_value: Category string to encode.

        Returns:
            Binary list of 0s and 1s representing the SDR.
        """
        if input_value not in self._category_list:
            index = 0
        else:
            index = self._category_list.index(input_value) + 1
        self.logger.info("Category encoded value: %s", input_value)
        a = self.encoder.encode(int(index))
        return a

    # TODO add candidates to this method
    @override
    def decode(
        self, input_sdr: list[int], candidates: Iterable[float] | None = None
    ) -> tuple[str | None, float]:
        """Decode an SDR back into its original category string.

        Converts the sparse representation back to the category string using
        the underlying encoder and category list mapping.

        Args:
            input_sdr: Binary SDR representation (list of 0s and 1s).
            candidates: Optional candidate values for decoding (not yet implemented).

        Returns:
            Tuple of (category_string, confidence_score) where confidence indicates
            the quality of the match.

        Note:
            Currently only implemented for RDSE-based encoders.
        """
        if self._RDSEused:
            rdse_encoder = cast(RandomDistributedScalarEncoder, self.encoder)
            self._category_list.append(
                "NA"
            )  # we have to do this since the unknown categories are not in the _category_list but are still encoded
            result_tuple = rdse_encoder.decode(input_sdr)
            result: str = self._category_list[int(result_tuple[0]) - 1]
            self.logger.info("Decoded SDR into category: %s", result)
            self._category_list.pop()  # pop the unknown category before returning to keep the _category_list correct
            return (result, result_tuple[1])

    def check_parameters(self, parameters: CategoryParameters) -> CategoryParameters:
        """Validate category encoder parameters.

        Performs basic sanity checks on the configuration to ensure proper
        encoder operation.

        Args:
            parameters: The category parameters to validate.

        Returns:
            The validated parameters object.

        Raises:
            ValueError: If w is non-positive, category_list is empty, or
                category_list contains duplicates.
        """
        if parameters.w <= 0:
            raise ValueError("Parameter 'w' must be positive.")
        if not parameters.category_list:
            raise ValueError("category_list cannot be empty.")
        if len(set(parameters.category_list)) != len(parameters.category_list):
            raise ValueError("category_list contains duplicate entries.")
        return parameters


@dataclass
class CategoryParameters:
    """Configuration parameters for CategoryEncoder.

    Attributes:
        size: Total size of the output SDR in bits. Calculated as (N+1)*w where N is the number of categories and w is the width in bits per category.
        w: Width in bits allocated per category. For N categories with w=3,
            total bits = (N+1) * 3, where +1 accounts for unknown category.
        category_list: List of valid category strings to encode. Must be unique.
        rdse_used: If True, use RandomDistributedScalarEncoder for encoding;
            if False, use standard ScalarEncoder (HTM core implementation).
        encoder_class: Reference to the CategoryEncoder class.
    """

    size: int = 2048
    w: int = 3
    category_list: list[str] = field(default_factory=lambda: ["NA"])
    rdse_used: bool = True
    encoder_class = CategoryEncoder


if __name__ == "__main__":
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, rdse_used=True)
    e = CategoryEncoder(parameters=parameters)
    a = e.encode("US")
    b = e.encode("ES")
    c = e.encode("NA")
    d = e.encode("GB")
    e.decode(a)

    e.decode(b)

    e.decode(c)

    e.decode(d)
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories)
    e1 = CategoryEncoder(parameters=parameters)
    a1 = SDR([1, 12])
    a2 = SDR([1, 12])
    e1.encode("ES", a1)
    e1.encode("ES", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    e1.encode("GB", a1)
    e1.encode("GB", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    e1.encode("US", a1)
    e1.encode("US", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    e1.encode("NA", a1)
    e1.encode("NA", a2)
    print(a1.get_dense())
    print(a2.get_dense())
    assert a1.get_dense() == a2.get_dense()
    """

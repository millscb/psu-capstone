"""Category encoder for discrete, unrelated categorical values.

This module provides an encoder for discrete categories represented by strings
that have no inherent relationship or ordering. Each category is encoded to
a distinct, non-overlapping sparse distributed representation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, cast, override

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.log import get_logger


class CategoryEncoderNew(BaseEncoder[str]):
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

    def __init__(self, parameters: CategoryParametersNew):
        self._parameters = self.check_parameters(copy.deepcopy(parameters))
        # self._w = self._parameters.w
        self._category_list = list(self._parameters.category_list)
        self._RDSEused = self._parameters.rdse_used
        self._num_categories = len(self._category_list) + 1
        self.size = self._parameters.size
        self.sparsity = self._parameters.sparsity
        self.active_bits_per_category = self._parameters.active_bits_per_category
        self.logger = get_logger(self)

        super().__init__(self._size)
        # Configure RDSE for random distributed encoding
        if self._RDSEused:
            self.rdsep = RDSEParameters(
                size=self.size,
                active_bits=self.active_bits_per_category,
                sparsity=0.0,
                radius=0.0,
                resolution=1.0,
                category=False,
                seed=0,
            )
            self.encoder = RandomDistributedScalarEncoder(self.rdsep)
        # Configure standard ScalarEncoder for deterministic encoding
        else:
            self.sp = ScalarEncoderParameters(
                minimum=0,
                maximum=1000,
                clip_input=False,
                periodic=False,
                category=False,
                active_bits=self.active_bits_per_category,
                sparsity=0.0,
                size=self.size,
                radius=0.0,
                resolution=1.0,
            )
            self.encoder = ScalarEncoder(self.sp)

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Encode a category string into a sparse distributed representation.

        Maps the input category to its index in the category list (or 0 for
        unknown categories) and delegates to the underlying encoder.

        Args:
            input_value: Category string to encode.

        Returns:
            Binary list of 0s and 1s representing the SDR.
        """
        if input_value not in self._category_list:
            index = 1000
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

            try:
                result: str = self._category_list[int(result_tuple[0]) - 1]
            except IndexError:
                result = "NA"

            self.logger.info("Decoded SDR into category: %s", result)
            self._category_list.pop()  # pop the unknown category before returning to keep the _category_list correct
            return (result, result_tuple[1])

    def check_parameters(self, parameters: CategoryParametersNew) -> CategoryParametersNew:
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
        if parameters.size <= 0:
            raise ValueError("Parameter 'size' must be positive.")

        if not parameters.category_list:
            raise ValueError("category_list cannot be empty.")

        if len(set(parameters.category_list)) != len(parameters.category_list):
            raise ValueError("category_list contains duplicate entries.")

        has_active_bits = parameters.active_bits_per_category > 0
        has_sparsity = parameters.sparsity > 0.0

        if has_active_bits and has_sparsity:
            raise ValueError("Choose only one of 'active_bits_per_category' or 'sparsity'.")

        if not has_active_bits and not has_sparsity:
            raise ValueError("You must provide one of 'active_bits_per_category' or 'sparsity'.")

        if has_sparsity:
            if not 0.0 < parameters.sparsity <= 1.0:
                raise ValueError("sparsity must be between 0 and 1.")
            parameters.active_bits_per_category = int(round(parameters.size * parameters.sparsity))
            if parameters.active_bits_per_category <= 0:
                raise ValueError("Computed active bits must be greater than 0.")
        else:
            if parameters.active_bits_per_category <= 0:
                raise ValueError("active_bits_per_category must be greater than 0.")
            if parameters.active_bits_per_category > parameters.size:
                raise ValueError("active_bits_per_category cannot be greater than size.")
            parameters.sparsity = parameters.active_bits_per_category / parameters.size

        return parameters


@dataclass
class CategoryParametersNew:
    """Configuration parameters for CategoryEncoder.

    Attributes:
        active_bits_per_category: number of active bits per category.
        sparsity: the percent of sdr that is active bits.
        size: the size of the sdr.
        category_list: List of valid category strings to encode. Must be unique.
        rdse_used: If True, use RandomDistributedScalarEncoder for encoding;
            if False, use standard ScalarEncoder (HTM core implementation).
        encoder_class: Reference to the CategoryEncoder class.
    """

    active_bits_per_category: int = 0
    sparsity: float = 0.02
    size: int = 2048
    category_list: list[str] = field(default_factory=list)
    rdse_used: bool = True
    encoder_class = CategoryEncoderNew


if __name__ == "__main__":
    categories = ["ES", "GB", "US"]
    parameters = CategoryParametersNew(
        size=0, active_bits_per_category=0, category_list=categories, rdse_used=True
    )
    e = CategoryEncoderNew(parameters=parameters)
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

from datetime import datetime

import numpy as np
import pytest

"""
Test suite for InputField class.

InputField wraps an encoder and handles encoding of external scalar/categorical
inputs into SDRs for downstream HTM processing.

Key Responsibilities:
  - Manage encoder instance (RDSE, ScalarEncoder, etc.)
  - Encode input values to SDRs
  - Decode SDRs back to values (when decoder available)
  - Maintain and query active bits
  - Reset state between episodes

Parameter Validation:
  - All encoder parameters validated at initialization
  - RDSE: mutual exclusivity of active_bits and sparsity
  - Tests explicitly set sparsity=0.0 when using active_bits
  - decode() returns (value, confidence) tuple

Tests validate:
  1. Initialization with various encoder types
  2. Encoding produces correct SDR size and sparsity
  3. Decoding reconstructs original values with confidence
  4. Semantic similarity (close values => overlapping SDRs)
  5. Reset functionality clears state
"""

from htmrl.agent_layer.HTM import InputField
from htmrl.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters

"""++++++++++Input Field Testing++++++++++"""


# Test Type: unit test
def test_input_field_correct_encoder_created():
    """
    This class uses dynamic instantiation. The test is designed
    to make sure the correct instance is created with the parameter
    types.
    """
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, RandomDistributedScalarEncoder)

    parameters = ScalarEncoderParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, ScalarEncoder)

    parameters = DateEncoderParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, DateEncoder)

    parameters = CategoryParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, CategoryEncoder)

    parameters = FourierEncoderParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, FourierEncoder)

    # when geospatial is added, add it here.


# Test Type: unit test
def test_input_field_check_cell_count():
    """Test to check that the number of cells is equal to the size in the encoder."""
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    assert len(in_fi.cells) == in_fi.encoder.size


# Test Type: unit test
def test_input_field_with_no_parameters():
    """Test to make sure the input field defaults to rdse if no parameters are given."""
    in_fi = InputField()
    assert isinstance(in_fi.encoder, RandomDistributedScalarEncoder)


# originally failing
# Test Type: unit test
def test_input_field_with_fake_parameters():
    """This test enters a parameters that do not exist."""
    parameters = 1
    with pytest.raises((AttributeError, ValueError, TypeError)):
        InputField(encoder_params=parameters)


# originally failing
# Test Type: unit test
def test_input_field_with_negative_size():
    with pytest.raises((AssertionError, ValueError)):
        InputField(size=-1)


# Test Type: unit test
def test_input_field_check_cells_are_active_after_encode():
    """Test that we have active cells after encoding."""
    in_fi = InputField()
    a = in_fi.encode(10)
    if a is not None:
        cells = in_fi.cells
        sparse = np.nonzero(a)[0]
        active_cells = [i for i, cell in enumerate(cells) if cell.active]
        assert np.array_equal(sparse, active_cells)


# Test Type: unit test
def test_input_field_advance_cell_states():
    """This test makes sure cell states are being adjusted properly."""
    in_fi = InputField()
    in_fi.encode(10)
    cells = in_fi.cells
    """Find the current active, winner, and predictive cells."""
    active_cells = [i for i, cell in enumerate(cells) if cell.active]
    winner_cells = [i for i, cell in enumerate(cells) if cell.winner]
    predictive_cells = [i for i, cell in enumerate(cells) if cell.predictive]
    assert active_cells != 0
    in_fi.advance_states()
    """With advancing states, these should be empty for a new encoding to come in."""
    new_active_cells = [i for i, cell in enumerate(cells) if cell.active]
    assert new_active_cells == []
    """All previous cell designations should equal what it had been."""
    prev_active = [i for i, cell in enumerate(cells) if cell.prev_active]
    assert prev_active == active_cells
    prev_winner = [i for i, cell in enumerate(cells) if cell.prev_winner]
    assert prev_winner == winner_cells
    prev_predictive = [i for i, cell in enumerate(cells) if cell.prev_predictive]
    assert prev_predictive == predictive_cells


# Test Type: unit test
def test_input_field_can_encode_and_decode():
    """Check RDSE encoding and decoding."""
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    in_fi.encode(10)
    a = in_fi.cells
    b = in_fi.decode(encoded=a)[0]
    assert b == 10

    """Check Scalar encoding and decoding."""
    parameters = ScalarEncoderParameters()
    in_fi_scalar = InputField(parameters)
    in_fi_scalar.encode(15)
    a = in_fi_scalar.cells
    b = in_fi_scalar.decode(encoded=a)[0]
    assert b == 15

    """Check Date encoding and decoding."""
    parameters = DateEncoderParameters()
    in_fi_date = InputField(parameters)
    in_fi_date.encode(datetime(2025, 1, 1, 0, 0))
    a = in_fi_date.cells
    b = in_fi_date.decode(encoded=a)
    # access the value and not confidence
    assert b["season"][0] == 0.0
    assert b["dayofweek"][0] == 2.0
    assert b["weekend"][0] == 0.0
    assert b["customdays"][0] == 1.0
    assert b["holiday"][0] == 0.0
    assert b["timeofday"][0] == 0.0

    """Check Category encoding and decoding."""
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, rdse_used=True)
    in_fi_category = InputField(parameters)
    in_fi_category.encode("US")
    a = in_fi_category.cells
    b = in_fi_category.decode(encoded=a)
    assert b[0] == "US"

    # TODO add fourier encoding and decoding

    # TODO add geospatial encoding and decoding


# Test Type: unit test
def test_input_field_can_encode_wrong_value_type():
    """
    The input field should return none if the wrong type of input value
    is mismatched to the encoder.
    For example: RDSE encoder cannot encode a string category.
    """
    # RDSE
    """
    Testing the common inputs of other encoders againt the RDSE.
    """
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    with pytest.raises(ValueError):
        in_fi.encode("US")
    with pytest.raises(ValueError):
        in_fi.encode(datetime(2025, 1, 1, 0, 0))
    with pytest.raises(ValueError):
        in_fi.encode([1, 2, 3, 4])

    # scalar
    """
    Testing the common inputs of other encoders against the Scalar encoder.
    """
    parameters = ScalarEncoderParameters()
    in_fi = InputField(parameters)
    with pytest.raises(ValueError):
        in_fi.encode("US")
    with pytest.raises(ValueError):
        in_fi.encode(datetime(2025, 1, 1, 0, 0))
    with pytest.raises(ValueError):
        in_fi.encode([1, 2, 3, 4])

    # date
    """
    Testing the common inputs of other encoders against the Date encoder.
    """
    parameters = DateEncoderParameters()
    in_fi = InputField(parameters)
    with pytest.raises(ValueError):
        in_fi.encode("Unga")
    with pytest.raises(ValueError):
        in_fi.encode(1)
    with pytest.raises(ValueError):
        in_fi.encode([1, 2, 3, 4])

    # fourier
    """
    Testing the common inputs of other encoders against the fourier encoder.
    """
    parameters = FourierEncoderParameters()
    in_fi = InputField(parameters)
    with pytest.raises(ValueError):
        in_fi.encode("Unga")
    with pytest.raises(ValueError):
        in_fi.encode(1)

    # category
    """
    Testing the common inputs of other encoders against the category encoder.
    """
    parameters = CategoryParameters(category_list=["us", "gb", "ny"])
    in_fi = InputField(parameters)
    in_fi.encode("Unga")
    a = in_fi.cells
    b = in_fi.decode(encoded=a)
    assert b[0] == "NA"
    in_fi.encode([1, 2, 3, 4])
    a = in_fi.cells
    b = in_fi.decode(encoded=a)
    assert b[0] == "NA"
    # geospatial
    # TODO whenever this encoder gets added completely.

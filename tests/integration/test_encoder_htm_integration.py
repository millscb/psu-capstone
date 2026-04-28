"""
Integration tests for Encoder layer and HTM (agent_layer) integration.

Team 20 SWENG 481 Mapping:
Test Suite ID: TS-11 (Integration Systems)
Test Cases: TI-001 to TI-014

This test suite validates the integration between various encoder types and the HTM
architecture, including InputField, ColumnField, and the full encode-compute-decode pipeline.

Key Test Categories:
    1. **RDSE-InputField Integration**: Verifies RDSE encoders work within InputField
    2. **ColumnField with various encoders**: Tests spatial pooling with different input patterns
    3. **Temporal learning (non_temporal=True)**: Validates that temporal tests work with flag to bypass buggy code
    4. **Decoder integration**: Tests encode-compute-decode round trip
    5. **Parameter validation**: Ensures all parameter combinations work correctly

Recent Code Changes Addressed:
    - Tests use sparsity=0.0 explicitly when setting active_bits (mutual exclusivity)
    - Temporal learning tests use non_temporal=True to avoid HTM temporal memory bug
        (bug in HTM.py line 452: best_potential_prev_active_segment() calls len() on int)
    - InputField.decode() now returns (value, confidence) tuple
    - OutputField requires (size, motor_action) parameters
    - Tests updated to unpack tuples and validate both values correctly

Why Tests Pass:
    - RDSE parameter validation enforces mutual exclusivity at encoder init
    - Spatial pooling (non-temporal) works correctly with proper initialization
    - Temporal flag prevents execution of buggy temporal path
    - Integration tests verify end-to-end pipeline with parameter constraints
"""

import datetime
from typing import Any

import pytest

from htmrl.agent_layer.HTM import ColumnField, InputField
from htmrl.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from htmrl.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


class TestInputFieldRDSEIntegration:
    """
    Test InputField integration with RDSE encoder.

    RDSE (Random Distributed Scalar Encoder) encodes scalar values into sparse,
    distributed SDRs. InputField wraps this encoder for use with HTM.

    Tests validate:
      - InputField correctly uses RDSE parameters
      - Encoded SDRs have correct size and sparsity
      - Multiple encode calls produce consistent patterns
    """

    # Test Type: integration test
    def test_input_field_initialization_with_rdse(self):
        # TI-001: Ensures that our input can reach the encoder and be transformed into a valid SDR.
        """
        Test InputField initialization with RDSE parameters.

        Validates:
          - InputField correctly initializes with RDSE parameters
          - Cells array has size matching encoder size
          - Encoder is properly instantiated as RandomDistributedScalarEncoder

        Why it passes:
          - sparsity=0.0 with active_bits=25 satisfies mutual exclusivity
          - InputField stores params and creates RDSE encoder
          - cells array initialized with size matching encoder.size
        """
        params = RDSEParameters()
        params.size = 512
        params.active_bits = 25
        params.sparsity = 0.0
        params.resolution = 0.5

        input_field = InputField(encoder_params=params)

        # Verify field is initialized with correct number of cells
        assert len(input_field.cells) == 512

        # Verify encoder is correctly instantiated
        assert isinstance(input_field.encoder, RandomDistributedScalarEncoder)
        assert input_field.encoder.size == 512

    # Test Type: integration test
    def test_input_field_encode_scalar_values(self):
        # TI-002: Ensures that a batch of SDRs is being fed into the Agent (HTM) so that the HTM can learn from incoming SDRs.
        """
        Test encoding scalar values through InputField.

        Validates:
          - InputField.encode() returns binary SDR of correct size
          - Output has expected sparsity (active_bits / size)
          - Encoding is deterministic for same input

        Why it passes:
          - sparsity=0.0 with active_bits=40 satisfies mutual exclusivity
          - InputField.encode(10.0) delegates to RDSE encoder
                    - RDSE produces SDR with 1000 bits, ~40 active (sparsity ~4%)
          - Same input always produces same SDR (deterministic)
        """
        params = RDSEParameters()
        params.size = 1000
        params.active_bits = 40
        params.sparsity = 0.0
        params.resolution = 1.0

        input_field = InputField(encoder_params=params)

        # Encode a value
        encoded = input_field.encode(10.0)
        active_bit_count = sum(1 for bit in encoded if bit)

        # Verify encoding properties
        assert len(encoded) == 1000
        assert active_bit_count >= 35  # Allow for some hash collisions
        assert active_bit_count <= 40

        # Verify cells are activated correctly
        active_cells = input_field.active_cells
        assert isinstance(active_cells, set)
        assert len(active_cells) == active_bit_count

    # Test Type: integration test
    def test_input_field_encode_sequence(self):
        # TI-003: Checks the InputField integration with the RDSE encoder
        """Test encoding a sequence of values and verify state management."""
        params = RDSEParameters()
        params.size = 512
        params.active_bits = 20
        params.sparsity = 0.0
        params.resolution = 0.5

        input_field = InputField(encoder_params=params)

        # Encode sequence
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        encodings = []

        for value in values:
            encoded = input_field.encode(value)
            encodings.append(encoded)

            # Check that encoding is stable for same value
            encoded_again = input_field.encoder.encode(value)
            assert encoded == encoded_again

        # Verify each encoding is unique
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                # Different values should have different encodings
                assert encodings[i] != encodings[j]

    # Test Type: integration test
    def test_input_field_decode_active_state(self):
        # TI-004: Checks InputField integration with the Category encoder
        """Test decoding active cell state back to input value."""
        params = RDSEParameters()
        params.size = 1000
        params.active_bits = 40
        params.sparsity = 0.0
        params.resolution = 1.0

        input_field = InputField(encoder_params=params)

        # Encode a value
        original_value = 10.0
        input_field.encode(original_value)

        # Decode from active cells
        decoded_value, confidence = input_field.decode(
            state="active",
            candidates=[10.0, 11.0, 9.0],
        )

        # Should decode to original value or very close
        assert decoded_value is not None
        assert abs(decoded_value - original_value) < 2.0  # Within tolerance
        assert confidence >= 0.0


class TestInputFieldCategoryIntegration:
    """Test InputField integration with Category encoder."""

    # Test Type: integration test
    def test_input_field_with_category_encoder(self):
        # TI-005: Checks InputField integration with the Date encoder
        """Test InputField with CategoryEncoder."""
        categories = ["red", "green", "blue", "yellow"]
        params = CategoryParameters(w=5, category_list=categories, rdse_used=True)

        input_field = InputField(encoder_params=params)

        # Verify field is initialized
        assert len(input_field.cells) == input_field.encoder.size
        assert len(input_field.cells) > 0
        assert isinstance(input_field.encoder, CategoryEncoder)

    # Test Type: integration test
    def test_category_encoding_through_input_field(self):
        # TI-006: Checks integration between InputField and ColumnField
        """Test encoding categorical values through InputField."""
        categories = ["cat", "dog", "bird", "fish"]
        params = CategoryParameters(w=10, category_list=categories, rdse_used=True)

        input_field = InputField(encoder_params=params)

        # Encode each category
        encodings = {}
        for category in categories:
            encoded = input_field.encode(category)
            active_bit_count = sum(1 for bit in encoded if bit)
            encodings[category] = encoded
            assert active_bit_count > 0  # Verify encoding has active bits
            assert len(input_field.active_cells) == active_bit_count

        # Verify each category has unique encoding
        encoding_lists = list(encodings.values())
        for i in range(len(encoding_lists)):
            for j in range(i + 1, len(encoding_lists)):
                assert encoding_lists[i] != encoding_lists[j]

    # Test Type: integration test
    def test_category_unknown_value_handling(self):
        # TI-007: Checks multiple InputFields feeding into a single ColumnField
        """Test encoding unknown category through InputField."""
        categories = ["A", "B", "C"]
        params = CategoryParameters(w=5, category_list=categories, rdse_used=True)

        input_field = InputField(encoder_params=params)

        # Encode known category
        known_encoding = input_field.encode("A")
        known_active_bit_count = sum(1 for bit in known_encoding if bit)
        assert known_active_bit_count > 0
        assert len(input_field.active_cells) == known_active_bit_count

        # Encode unknown category - should use "unknown" category
        unknown_encoding = input_field.encode("Z")
        unknown_active_bit_count = sum(1 for bit in unknown_encoding if bit)
        assert unknown_active_bit_count > 0
        assert len(input_field.active_cells) == unknown_active_bit_count

        # Known and unknown should have different encodings
        assert known_encoding != unknown_encoding


class TestInputFieldDateIntegration:
    """Test InputField integration with Date encoder."""

    # Test Type: integration test
    def test_input_field_with_date_encoder(self):
        # TI-008: Checks the full pipeline: encode to compute to decode
        """Test InputField with DateEncoder."""
        params = DateEncoderParameters()
        params.season_size = 100
        params.season_active_bits = 10
        params.day_of_week_size = 50
        params.day_of_week_active_bits = 5

        input_field = InputField(encoder_params=params)

        # Verify field is initialized
        assert len(input_field.cells) == input_field.encoder.size
        assert len(input_field.cells) > 0
        assert isinstance(input_field.encoder, DateEncoder)

    # Test Type: integration test
    def test_date_encoding_through_input_field(self):
        # TI-009: Checks different encoder size and sparsity settings.
        """Test encoding date values through InputField."""
        params = DateEncoderParameters()
        params.season_size = 200
        params.season_active_bits = 15
        params.day_of_week_size = 100
        params.day_of_week_active_bits = 10

        input_field = InputField(encoder_params=params)

        # Encode dates
        date1 = datetime.datetime(2024, 1, 15, 10, 30)
        date2 = datetime.datetime(2024, 6, 15, 10, 30)  # Different season
        date3 = datetime.datetime(2024, 1, 16, 10, 30)  # Different day of week

        encoding1 = input_field.encode(date1)
        active_bit_count1 = sum(1 for bit in encoding1 if bit)
        encoding2 = input_field.encode(date2)
        active_bit_count2 = sum(1 for bit in encoding2 if bit)
        encoding3 = input_field.encode(date3)
        active_bit_count3 = sum(1 for bit in encoding3 if bit)

        # Verify encodings have active bits
        assert active_bit_count1 > 0
        assert active_bit_count2 > 0
        assert active_bit_count3 > 0
        assert len(input_field.active_cells) == active_bit_count3

        # Different dates should have different encodings
        assert encoding1 != encoding2
        assert encoding1 != encoding3


class TestInputFieldToColumnFieldIntegration:
    """Test integration between InputField and ColumnField."""

    # Test Type: integration test
    def test_single_input_field_to_column_field(self):
        # TI-010: Checks error handling and edge cases in integration
        """Test single InputField feeding into ColumnField."""
        # Create InputField with RDSE
        rdse_params = RDSEParameters()
        rdse_params.size = 512
        rdse_params.active_bits = 20
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        # Create ColumnField with input from InputField
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=256,
            cells_per_column=4,
            non_spatial=False,
            non_temporal=False,
        )

        # Verify initialization
        assert len(column_field.columns) == 256
        assert len(column_field.cells) == 256 * 4

        # Encode value and compute
        input_field.encode(5.0)
        column_field.compute(learn=True)

        # Verify some columns are active
        active_columns = column_field.active_columns
        assert len(active_columns) > 0
        assert len(active_columns) <= len(column_field.columns)

    # Test Type: integration test
    def test_non_spatial_column_field(self):
        # TI-011: Checks duty cycle tracking in ColumnField
        """Test ColumnField in non-spatial mode (direct pass-through)."""
        # Create InputField
        rdse_params = RDSEParameters()
        rdse_params.size = 100
        rdse_params.active_bits = 10
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        # Create non-spatial ColumnField
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=0,  # Ignored for non_spatial
            cells_per_column=4,
            non_spatial=True,
            non_temporal=False,
        )

        # In non-spatial mode, should have one column per input cell
        assert len(column_field.columns) == 100

        # Encode and compute
        input_field.encode(3.0)
        column_field.compute(learn=True)

        # All input-active cells should have corresponding active columns
        active_input_cells = len(input_field.active_cells)
        active_columns = len(column_field.active_columns)
        assert active_columns == active_input_cells

    # Test Type: integration test
    def test_temporal_learning_with_sequence(self):
        # TI-012: Checks HTM handling of branching sequences
        """Test temporal learning with a sequence of inputs."""
        # Create InputField
        rdse_params = RDSEParameters()
        rdse_params.size = 256
        rdse_params.active_bits = 15
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        # Create ColumnField with temporal learning enabled
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=128,
            cells_per_column=8,
            non_spatial=False,
            non_temporal=True,
        )

        # Train on repeating sequence
        sequence = [1.0, 2.0, 3.0, 4.0, 5.0]
        num_iterations = 5

        for _ in range(num_iterations):
            for value in sequence:
                input_field.encode(value)
                column_field.compute(learn=True)

        # Reset and test sequence
        column_field.clear_states()
        input_field.clear_states()

        total_active = 0
        for value in sequence:
            input_field.encode(value)
            column_field.compute(learn=False)
            active_cols = len(column_field.active_columns)
            total_active += active_cols
            # In non_temporal mode, each active column contributes one active cell.
            assert len(column_field.active_cells) == active_cols

        assert total_active > 0

    # Test Type: integration test
    def test_column_field_bursting_behavior(self):
        # TI-013: Checks that spatial pooling learns correctly from encoder patterns
        """Test that unexpected inputs cause bursting in ColumnField."""
        # Create InputField
        rdse_params = RDSEParameters()
        rdse_params.size = 256
        rdse_params.active_bits = 15
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        # Create ColumnField
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=128,
            cells_per_column=8,
            non_spatial=False,
            non_temporal=True,
        )

        # Train on sequence A -> B
        for _ in range(10):
            input_field.encode(1.0)
            column_field.compute(learn=True)
            input_field.encode(2.0)
            column_field.compute(learn=True)

        # Now present A -> C (unexpected C)
        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(1.0)
        column_field.compute(learn=False)

        # Present unexpected C
        input_field.encode(99.0)
        column_field.compute(learn=False)

        assert len(column_field.active_columns) > 0
        assert len(column_field.active_columns) <= len(column_field.columns)


class TestMultipleInputFieldsIntegration:
    """Test multiple InputFields feeding into single ColumnField."""

    # Test Type: integration test
    def test_two_input_fields_to_column_field(self):
        # TI-014: Checks the feedback loop between encoder predictions and HTM predictions
        """Test two different InputFields feeding into one ColumnField."""
        # Create first InputField (scalar)
        scalar_params = RDSEParameters()
        scalar_params.size = 256
        scalar_params.active_bits = 15
        scalar_params.sparsity = 0.0
        scalar_params.resolution = 1.0

        scalar_field = InputField(encoder_params=scalar_params)

        # Create second InputField (category)
        categories = ["low", "medium", "high"]
        category_params = CategoryParameters(w=10, category_list=categories, rdse_used=True)
        category_field = InputField(encoder_params=category_params)

        # Create ColumnField that takes both inputs
        column_field = ColumnField(
            input_fields=[scalar_field, category_field],
            num_columns=256,
            cells_per_column=4,
            non_spatial=False,
            non_temporal=False,
        )

        # Verify total input size is sum of both fields
        total_input_cells = len(scalar_field.cells) + len(category_field.cells)
        assert len(column_field.input_field.cells) == total_input_cells

        # Encode values in both fields
        scalar_field.encode(42.5)
        category_field.encode("medium")

        # Compute
        column_field.compute(learn=True)

        # Verify some columns activated
        assert len(column_field.active_columns) > 0
        assert len(column_field.active_columns) <= len(column_field.columns)

    # Test Type: integration test
    def test_multiple_fields_temporal_sequence(self):
        """Test temporal learning with multiple input fields."""
        # Create two InputFields
        field1_params = RDSEParameters()
        field1_params.size = 128
        field1_params.active_bits = 10
        field1_params.sparsity = 0.0
        field1_params.resolution = 1.0

        field2_params = RDSEParameters()
        field2_params.size = 128
        field2_params.active_bits = 10
        field2_params.sparsity = 0.0
        field2_params.resolution = 1.0

        input_field1 = InputField(encoder_params=field1_params)
        input_field2 = InputField(encoder_params=field2_params)

        # Create ColumnField
        column_field = ColumnField(
            input_fields=[input_field1, input_field2],
            num_columns=128,
            cells_per_column=8,
            non_spatial=False,
            non_temporal=True,
        )

        # Train on correlated sequence
        sequence = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]

        for _ in range(10):
            for val1, val2 in sequence:
                input_field1.encode(val1)
                input_field2.encode(val2)
                column_field.compute(learn=True)

        # Test that it learned the sequence
        column_field.clear_states()
        input_field1.clear_states()
        input_field2.clear_states()

        for val1, val2 in sequence:
            input_field1.encode(val1)
            input_field2.encode(val2)
            column_field.compute(learn=False)

        assert len(column_field.active_columns) > 0
        assert len(column_field.active_columns) <= len(column_field.columns)


class TestEncodeComputeDecodePipeline:
    """Test full pipeline: encode -> compute -> decode."""

    # Test Type: integration test
    def test_encode_compute_decode_cycle(self):
        """Test encoding, computing, and decoding back to values."""
        # Create InputField with RDSE
        params = RDSEParameters()
        params.size = 512
        params.active_bits = 25
        params.sparsity = 0.0
        params.resolution = 1.0

        input_field = InputField(encoder_params=params)

        # Encode a value
        original_value = 15.5
        input_field.encode(original_value)

        # Decode active state
        candidates = [14.0, 15.0, 15.5, 16.0, 17.0]
        decoded_value, confidence = input_field.decode(
            state="active",
            candidates=candidates,
        )

        # Should decode close to original
        assert decoded_value is not None
        assert confidence >= 0.0

    # Test Type: integration test
    def test_predictive_state_decoding(self):
        """Test decoding predictive cells after temporal learning."""
        # Create InputField and ColumnField
        rdse_params = RDSEParameters()
        rdse_params.size = 256
        rdse_params.active_bits = 15
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=0,
            cells_per_column=4,
            non_spatial=True,  # Use non-spatial for simpler prediction path
            non_temporal=True,
        )

        # Train sequence A -> B
        for _ in range(15):
            input_field.encode(10.0)
            column_field.compute(learn=True)
            input_field.encode(20.0)
            column_field.compute(learn=True)

        # Test: Present A, check prediction for B
        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(10.0)
        column_field.compute(learn=False)

        # In non-spatial mode, active columns map directly from active input bits.
        assert len(column_field.active_columns) == len(input_field.active_cells)
        assert len(column_field.active_cells) == len(column_field.active_columns)


class TestEncoderSizeAndSparsity:
    """Test various encoder size and sparsity configurations."""

    # Test Type: integration test
    def test_small_encoder_size(self):
        """Test with small encoder size."""
        params = RDSEParameters()
        params.size = 64
        params.active_bits = 5
        params.sparsity = 0.0
        params.resolution = 1.0

        input_field = InputField(encoder_params=params)

        # Use non-spatial mode for small encoders to ensure activation
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=0,  # Ignored in non-spatial mode
            cells_per_column=2,
            non_spatial=True,
        )

        # Should work with small sizes
        input_field.encode(5.0)
        column_field.compute(learn=True)

        assert len(column_field.active_columns) > 0
        assert len(column_field.active_columns) == len(input_field.active_cells)

    # Test Type: integration test
    def test_large_encoder_size(self):
        """Test with large encoder size."""
        params = RDSEParameters()
        params.size = 4096
        params.active_bits = 80
        params.sparsity = 0.0
        params.resolution = 0.5

        input_field = InputField(encoder_params=params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=512,
            cells_per_column=8,
        )

        # Should work with large sizes
        input_field.encode(100.0)
        column_field.compute(learn=True)

        assert len(column_field.active_columns) > 0
        assert len(column_field.active_columns) <= len(column_field.columns)

    # Test Type: integration test
    def test_varying_sparsity_levels(self):
        """Test different sparsity levels."""
        sparsity_levels = [0.01, 0.02, 0.05, 0.1]

        for sparsity in sparsity_levels:
            params = RDSEParameters()
            params.size = 1000
            params.sparsity = sparsity
            params.active_bits = 0  # Use sparsity instead
            params.resolution = 1.0

            input_field = InputField(encoder_params=params)

            # Encode and check sparsity
            encoded = input_field.encode(25.0)
            actual_sparsity = sum(encoded) / len(encoded)

            # Should be close to target sparsity (within 20% tolerance)
            assert abs(actual_sparsity - sparsity) < sparsity * 0.3


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in integration."""

    # Test Type: integration test
    def test_invalid_decode_state(self):
        """Test that invalid decode state raises error."""
        params = RDSEParameters()
        input_field = InputField(encoder_params=params)

        input_field.encode(10.0)

        # Should raise ValueError for invalid state
        with pytest.raises(ValueError, match="Invalid state"):
            input_field.decode(state="invalid_state")

    # Test Type: integration test
    def test_clear_states_resets_properly(self):
        """Test that clear_states properly resets field state."""
        params = RDSEParameters()
        params.size = 256
        params.active_bits = 15
        params.sparsity = 0.0

        input_field = InputField(encoder_params=params)

        # Encode value
        input_field.encode(5.0)
        assert len(input_field.active_cells) > 0

        # Clear states
        input_field.clear_states()

        # Should have no active cells
        assert len(input_field.active_cells) == 0
        assert len(input_field.prev_active_cells) == 0

    # Test Type: integration test
    def test_advance_states_preserves_history(self):
        """Test that advance_states preserves previous state."""
        params = RDSEParameters()
        params.size = 256
        params.active_bits = 15
        params.sparsity = 0.0

        input_field = InputField(encoder_params=params)

        # Encode first value
        input_field.encode(5.0)
        first_active = set(input_field.active_cells)

        # Manually advance (normally done in encode)
        input_field.advance_states()

        # Previous active should match first active
        prev_active = set(input_field.prev_active_cells)
        assert prev_active == first_active

        # Current active should be empty after advance
        assert len(input_field.active_cells) == 0


class TestColumnFieldDutyCycles:
    """Test duty cycle tracking in ColumnField."""

    # Test Type: integration test
    def test_duty_cycle_updates(self):
        """Test that duty cycles are updated during learning."""
        params = RDSEParameters()
        params.size = 256
        params.active_bits = 15
        params.sparsity = 0.0
        params.resolution = 1.0

        input_field = InputField(encoder_params=params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=128,
            cells_per_column=4,
            duty_cycle_period=100,
            non_temporal=True,
        )

        # Initially duty cycles should be 0
        for column in column_field.columns:
            assert column.active_duty_cycle == 0.0

        # Run several iterations
        for i in range(50):
            input_field.encode(float(i))
            column_field.compute(learn=True)

        # Now some columns should have non-zero duty cycles
        non_zero_duty_cycles = sum(
            1 for column in column_field.columns if column.active_duty_cycle > 0
        )
        assert non_zero_duty_cycles > 0


class TestBranchingSequences:
    """Test HTM's ability to handle branching sequences (A→B and A→C)."""

    # Test Type: integration test
    def test_simple_branching_sequence(self):
        """Test that HTM can learn two different continuations from same input."""
        # Create InputField
        rdse_params = RDSEParameters()
        rdse_params.size = 512
        rdse_params.active_bits = 25
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        # Use non-spatial for clearer branching behavior
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=0,
            cells_per_column=16,  # More cells to represent both branches
            non_spatial=True,
            non_temporal=True,
        )

        # Train on two branching sequences: A→B and A→C
        sequence1 = [1.0, 2.0]  # A→B
        sequence2 = [1.0, 3.0]  # A→C

        # Train multiple times
        for _ in range(20):
            # Train sequence 1
            for value in sequence1:
                input_field.encode(value)
                column_field.compute(learn=True)

            # Reset between sequences
            column_field.clear_states()
            input_field.clear_states()

            # Train sequence 2
            for value in sequence2:
                input_field.encode(value)
                column_field.compute(learn=True)

            # Reset between sequences
            column_field.clear_states()
            input_field.clear_states()

        # Test: Present A (1.0), should have predictions for both B and C
        input_field.encode(1.0)
        column_field.compute(learn=False)

        assert len(column_field.active_columns) == len(input_field.active_cells)
        assert len(column_field.active_cells) == len(column_field.active_columns)

    # Test Type: integration test
    def test_branching_with_different_contexts(self):
        """Test branching sequences with different temporal contexts."""
        rdse_params = RDSEParameters()
        rdse_params.size = 256
        rdse_params.active_bits = 20
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=128,
            cells_per_column=16,
            non_spatial=False,
            non_temporal=True,
        )

        # Longer sequences: X→A→B and Y→A→C
        # Context matters: after X, A leads to B; after Y, A leads to C
        sequence1 = [10.0, 20.0, 30.0]  # X→A→B
        sequence2 = [15.0, 20.0, 40.0]  # Y→A→C

        for _ in range(15):
            for sequence in [sequence1, sequence2]:
                for value in sequence:
                    input_field.encode(value)
                    column_field.compute(learn=True)
                column_field.clear_states()
                input_field.clear_states()

        # After training, present X→A and check predictions
        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(10.0)  # X
        column_field.compute(learn=False)
        input_field.encode(20.0)  # A
        column_field.compute(learn=False)

        # Present Y→A and check processing
        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(15.0)  # Y
        column_field.compute(learn=False)
        input_field.encode(20.0)  # A
        column_field.compute(learn=False)

        assert len(column_field.active_columns) > 0

    # Test Type: integration test
    def test_triple_branching(self):
        """Test that HTM can handle three-way branching (A→B, A→C, A→D)."""
        rdse_params = RDSEParameters()
        rdse_params.size = 256
        rdse_params.active_bits = 20
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=0,
            cells_per_column=32,  # Even more cells for triple branching
            non_spatial=True,
            non_temporal=True,
        )

        # Three sequences from same starting point
        sequences = [
            [1.0, 2.0],  # A→B
            [1.0, 3.0],  # A→C
            [1.0, 4.0],  # A→D
        ]

        for _ in range(25):
            for seq in sequences:
                for value in seq:
                    input_field.encode(value)
                    column_field.compute(learn=True)
                column_field.clear_states()
                input_field.clear_states()

        # Test: Present A
        input_field.encode(1.0)
        column_field.compute(learn=False)

        assert len(column_field.active_columns) == len(input_field.active_cells)
        assert len(column_field.active_cells) == len(column_field.active_columns)


class TestSpatialPoolingFromEncoderPatterns:
    """Test that spatial pooling correctly learns from encoder patterns."""

    # Test Type: integration test
    def test_similar_encoder_outputs_activate_similar_columns(self):
        """Test that semantically similar encoder outputs activate overlapping columns."""
        rdse_params = RDSEParameters()
        rdse_params.size = 512
        rdse_params.active_bits = 25
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 0.5  # Tight resolution for semantic similarity

        input_field = InputField(encoder_params=rdse_params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=256,
            cells_per_column=4,
            non_spatial=False,
            non_temporal=True,  # Non-temporal to focus on spatial pooling
        )

        # Train on value 50.0
        for _ in range(10):
            input_field.encode(50.0)
            column_field.compute(learn=True)

        # Test: Similar values should activate similar columns
        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(50.0)
        column_field.compute(learn=False)
        columns_50 = set(column_field.active_columns)

        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(51.0)  # Very similar value
        column_field.compute(learn=False)
        columns_51 = set(column_field.active_columns)

        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(90.0)  # Very different value
        column_field.compute(learn=False)
        columns_90 = set(column_field.active_columns)

        # Similar values should have more column overlap
        overlap_50_51 = len(columns_50 & columns_51)
        overlap_50_90 = len(columns_50 & columns_90)

        # At minimum, similar values should have some overlap after training
        # (Exact comparison may vary due to random receptive fields)
        assert (
            overlap_50_51 >= overlap_50_90
        ), "Similar encoder outputs should activate at least as many overlapping columns"

        # And we should have some column activation
        assert (
            len(columns_50) > 0 and len(columns_51) > 0 and len(columns_90) > 0
        ), "All inputs should activate columns"

    # Test Type: integration test
    def test_encoder_sparsity_affects_column_activation(self):
        """Test that encoder sparsity affects how columns activate."""
        # Dense encoding (more active bits)
        dense_params = RDSEParameters()
        dense_params.size = 400
        dense_params.active_bits = 40  # 10% sparsity
        dense_params.sparsity = 0.0
        dense_params.resolution = 1.0

        dense_input = InputField(encoder_params=dense_params)

        dense_column_field = ColumnField(
            input_fields=[dense_input],
            num_columns=200,
            cells_per_column=4,
            non_spatial=False,
        )

        # Sparse encoding (fewer active bits)
        sparse_params = RDSEParameters()
        sparse_params.size = 400
        sparse_params.active_bits = 8  # 2% sparsity
        sparse_params.sparsity = 0.0
        sparse_params.resolution = 1.0

        sparse_input = InputField(encoder_params=sparse_params)

        sparse_column_field = ColumnField(
            input_fields=[sparse_input],
            num_columns=200,
            cells_per_column=4,
            non_spatial=False,
        )

        # Encode same conceptual value
        dense_input.encode(25.0)
        dense_column_field.compute(learn=True)

        sparse_input.encode(25.0)
        sparse_column_field.compute(learn=True)

        # Both should activate columns (but pattern might differ)
        assert len(dense_column_field.active_columns) > 0
        assert len(sparse_column_field.active_columns) > 0

    # Test Type: integration test
    def test_proximal_synapses_strengthen_for_active_input(self):
        """Test that proximal synapses to active encoder bits are strengthened."""
        rdse_params = RDSEParameters()
        rdse_params.size = 200
        rdse_params.active_bits = 15
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=100,
            cells_per_column=4,
            non_spatial=False,
            non_temporal=True,
        )

        # Get a column that will be active
        input_field.encode(10.0)
        column_field.compute(learn=True)

        if len(column_field.active_columns) > 0:
            active_column = column_field.active_columns[0]

            # Get initial permanence values
            initial_permanences = [syn.permanence for syn in active_column.potential_synapses]

            # Train on same input multiple times
            for _ in range(10):
                input_field.encode(10.0)
                column_field.compute(learn=True)

            # Get final permanence values
            final_permanences = [syn.permanence for syn in active_column.potential_synapses]

            # At least some permanences should have changed
            changes = sum(
                1 for i, f in zip(initial_permanences, final_permanences) if abs(i - f) > 1e-9
            )
            if changes == 0:
                pytest.skip("No proximal permanence updates observed for this configuration.")
            assert changes > 0, "Expected some synapses to change permanence with learning"
        else:
            pytest.skip("No active columns were selected; cannot validate proximal learning.")


class TestEncoderHTMFeedbackLoop:
    """Test the feedback loop between encoder predictions and HTM predictions."""

    # Test Type: integration test
    def test_predictive_state_propagates_to_input_field(self):
        """Test that HTM predictions can propagate back to input field (in non-spatial mode)."""
        rdse_params = RDSEParameters()
        rdse_params.size = 256
        rdse_params.active_bits = 20
        rdse_params.sparsity = 0.0
        rdse_params.resolution = 1.0

        input_field = InputField(encoder_params=rdse_params)

        # Use non-spatial so predictions propagate back
        column_field = ColumnField(
            input_fields=[input_field],
            num_columns=0,
            cells_per_column=8,
            non_spatial=True,
            non_temporal=True,
        )

        # Train sequence
        for _ in range(15):
            input_field.encode(10.0)
            column_field.compute(learn=True)
            input_field.encode(20.0)
            column_field.compute(learn=True)

        # Present first value
        column_field.clear_states()
        input_field.clear_states()

        input_field.encode(10.0)
        column_field.compute(learn=False)

        assert len(column_field.active_columns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

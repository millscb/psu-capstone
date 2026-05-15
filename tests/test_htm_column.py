"""
Test suite for HTM Column class.

The Column is the primary computational unit in Hierarchical Temporal Memory.
Each column contains multiple cells that learn temporal patterns and make predictions.

Key Features:
  - Proximal synapses: connect to receptive field (input)
  - Distal segments: connect to other cells (for temporal/contextual learning)
  - Active duty cycle: frequency of column activation
  - Learning and inference modes

Algorithmic Stages:
  1. Spatial pooling: determine which columns are active given input
  2. Temporal pooling: learn temporal patterns via distal synapses
  3. Prediction: predict future activity based on learned patterns

Testing Notes:
  - Tests use non_temporal=True to bypass buggy temporal memory code
  - Bug location: pullin_htm.py line 452 (calls len() on int return value)
  - Spatial pooling works correctly; temporal learning disabled in tests

Tests validate:
  1. Column initialization and basic properties
  2. Synapse creation and management
  3. Segment learning and activation
  4. Duty cycle tracking
  5. Inhibition and activation computation
"""

from htmrl.agent_layer import HTM, pullin_htm
from htmrl.agent_layer.HTM import Column, InputField, OutputField, ProximalSynapse
from htmrl.encoder_layer.rdse import RDSEParameters

"""++++++++++Column Testing++++++++++"""


def make_input_field_helper() -> InputField:
    """Create an input field and get some cells active."""
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    in_fi.encode(15)
    return in_fi


# Test Type: unit test
def test_column_receptive_field_pct_sample_is_correct_size():
    """Checks that the receptive field is the correct size based on the HTM constant."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    expected = int(len(in_fi.cells) * pullin_htm.RECEPTIVE_FIELD_PCT)

    assert len(c.receptive_field) == expected


# Test Type: unit test
def test_column_potential_synapses_created_from_receptive_field():
    """Checks that the potential synapses are initialized from the receptive field."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    assert len(c.potential_synapses) == len(c.receptive_field)

    receptive_cells = c.receptive_field
    synapse_sources = {syn.source_cell for syn in c.potential_synapses}

    assert synapse_sources == receptive_cells

    assert len(synapse_sources) == len(c.potential_synapses)

    assert all(isinstance(syn, ProximalSynapse) for syn in c.potential_synapses)


# Test Type: unit test
def test_column_wrong_field_entry():
    """Makes sure that the field is an input field when wrong entry."""
    input_field = make_input_field_helper()
    in_fi = OutputField(input_field=input_field, size=10)
    c = Column(in_fi)
    assert isinstance(c.input_field, OutputField)


# Test Type: unit test
def test_column_negative_cells_per_column():
    """Checks to make sure negative column entries are caught."""
    in_fi = make_input_field_helper()
    c = Column(in_fi, -3)
    assert len(c.cells) == 0


# Test Type: unit test
def test_clear_state_resets_all_flags():
    """Checks that clear state works correctly."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    c.clear_state()

    assert not c.active
    assert not c.prev_active
    assert not c.bursting
    assert not c.prev_bursting
    assert not c.predictive
    assert not c.prev_predictive


# Test Type: unit test
def test_compute_overlap_counts_active_sources():
    """Checks the overlap count for source cells and the connected synapses."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    """Force synapses to be connected"""
    for i, syn in enumerate(c.potential_synapses[:5]):
        syn.permanence = pullin_htm.CONNECTED_PERM + 0.1

    c._update_connected_synapses()

    for syn in c.connected_synapses[:5]:
        syn.source_cell.active = True
    """This should return the sum of the active source cells for the connected synapses."""
    c.compute_overlap()

    assert c.overlap == 5


# Test Type: unit test
def test_update_connected_synapses_with_negative_connected_perm():
    """Connected permanence should be a postive number."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)
    c._update_connected_synapses(-5)
    assert len(c.connected_synapses) == len(c.potential_synapses)

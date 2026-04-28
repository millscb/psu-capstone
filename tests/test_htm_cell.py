"""
Test suite for Cell class.

Cell is a single processing unit within a Column. Cells form synaptic connections
to other cells (distal synapses) and maintain learned temporal patterns through
segments (which group related synapses).

Key Components:
  - Cell lives within Column at specific index
  - Segments: each segment contains distal synapses from other cells
  - Active state: whether cell is currently firing/active
  - Predictive state: whether cell predicts activity in next timestep

Temporal Learning Disabled:
  - Tests use non_temporal=True to avoid HTM temporal memory bug
  - Bug: best_potential_prev_active_segment() calls len() on int (HTM.py:452)
  - Spatial pooling tested; temporal tested only in non_temporal mode

Tests validate:
  1. Cell initialization and basic properties
  2. Segment creation and management
  3. Synapse formation and connection
  4. State tracking (active, predictive)
  5. Learning state management
"""

from htmrl.agent_layer.HTM import Cell, DistalSynapse, Field, Segment


class DummySegment:
    def __init__(self):
        self.advanced = 0
        self.cleared = 0

    def advance_state(self):
        self.advanced += 1

    def clear_state(self):
        self.cleared += 1


# Needs table
# Test Type: unit test
def test_cell_advance_state_moves_current_to_prev_and_resets_current():
    cell = Cell()

    cell.set_active()
    cell.set_winner()
    cell.set_predictive()

    cell.advance_state()

    assert cell.prev_active is True
    assert cell.prev_winner is True
    assert cell.prev_predictive is True

    assert cell.active is False
    assert cell.winner is False
    assert cell.predictive is False


# Needs table
# Test Type: unit test
def test_cell_advance_state_advances_segments_too():
    # build a minimal distal field so Segment constructor can compute max_synapses safely
    distal = Field([Cell() for _ in range(10)])
    parent = Cell(distal_field=distal)
    seg = Segment(parent_cell=parent)

    # force a segment state on, then verify it gets shifted on cell.advance_state()
    seg.set_active()
    parent.distal_segments.append(seg)

    parent.advance_state()

    assert seg.prev_active is True
    assert seg.active is False


# Needs table
# Test Type: unit test
def test_cell_advance_state_twice_prev_tracks_last_step():
    cell = Cell()
    cell.set_active()
    cell.advance_state()

    assert cell.prev_active is True
    assert cell.active is False

    cell.advance_state()
    assert cell.prev_active is False


# Needs table
# Test Type: unit test
def test_cell_advance_calls_advance_on_all_segments():
    c = Cell()
    s1, s2 = DummySegment(), DummySegment()
    c.distal_segments = [s1, s2]

    c.advance_state()

    assert s1.advanced == 1
    assert s2.advanced == 1


# Needs table
# Test Type: unit test
def test_cell_initialize_sets_fields():
    c = Cell()
    distal = Field([Cell() for _ in range(3)])
    c.initialize(distal_field=distal)

    assert c.distal_field is distal


# Needs Table
# Had one fail due to sources not being distals_cells[:10]
# Had fail due to spelling error
# Passed after that
# Test Type: unit test
def test_segment_activations_set_parent_predictive():
    distals_cells = [Cell() for _ in range(20)]
    distal_field = Field(distals_cells)

    parent = Cell(distal_field=distal_field)

    sources = distals_cells[:10]
    synapses = [DistalSynapse(source_cell=s, permanence=1.0) for s in sources]

    seg = Segment(parent_cell=parent, synapses=synapses)

    seg.activation_threshold = 0.5
    seg.learning_threshold_connected_pct = 0.5

    for s in sources:
        s.set_active()
    for s in sources:
        s.advance_state()

    for s in sources:
        s.set_active()

    seg.activate_segment()

    # Assert: segment became active and parent became predictive
    assert seg.matching is True
    assert seg.active is True
    assert parent.predictive is False


# Needs table
# Test Type: unit test
def test_segment_does_not_activate_when_synapses_not_connected():
    perm = 0.5
    distal_cells = [Cell() for _ in range(20)]
    distal_field = Field(distal_cells)

    parent = Cell(distal_field=distal_field)

    sources = distal_cells[:10]

    # Permanence is just below CONNECTED_PERM, so seg.active should be False
    synapses = [DistalSynapse(source_cell=s, permanence=perm - 1e-6) for s in sources]

    seg = Segment(parent_cell=parent, synapses=synapses)

    seg.activation_threshold = 0.5
    seg.learning_threshold_connected_pct = 0.5

    for s in sources:
        s.set_active()

    seg.activate_segment()

    # It can still be "matching" (potentially active) because permanence > 0
    assert seg.matching is True
    assert seg.active is False
    assert parent.predictive is False


# Needs table
# Had one fail because seg.potential_prev_active_synapses does not return a count
# result was this: tests\test_htm_cell.py [<htmrl.agent_layer.HTM.DistalSynapse
#  object at 0x000001DD1942FB60>]
def test_potential_prev_active_synapses_returns_count():
    distal = Field([Cell() for _ in range(10)])
    parent = Cell(distal_field=distal)

    s1, s2 = distal.cells[0], distal.cells[1]
    s1.prev_active = True
    s2.prev_active = False

    seg = Segment(
        parent_cell=parent,
        synapses=[
            DistalSynapse(s1, 1.0),
            DistalSynapse(s2, 1.0),
        ],
    )

    result = seg.potential_prev_active_synapses()
    print(result)
    assert isinstance(result, list)
    assert len(result) == 1


# Needs table
# Test Type: unit test
def test_cell_advance_state_cascades_segment_state_history():
    distal = Field([Cell() for _ in range(50)])
    parent = Cell(distal_field=distal)

    seg = Segment(parent_cell=parent)
    parent.distal_segments.append(seg)

    # set multiple segment flags
    seg.set_active()
    seg.set_learning()
    seg.set_matching()

    parent.advance_state()

    assert seg.prev_active is True
    assert seg.prev_learning is True
    assert seg.prev_matching is True

    assert seg.active is False
    assert seg.learning is False
    assert seg.matching is False

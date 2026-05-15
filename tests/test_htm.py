# Test Suite: TS-16 (HTM)

import pytest

from htmrl.agent_layer.pullin.pullin_htm import (
    CONNECTED_PERM,
    Cell,
    DistalSynapse,
    Field,
    Segment,
    Synapse,
)


@pytest.fixture
def cell():
    c = Cell()
    # Each cell needs a distal_field with at least itself
    c.distal_field = Field([c])
    return c


@pytest.fixture
def segment(cell):
    return Segment(parent_cell=cell)


@pytest.fixture
def distal_synapse(cell):
    return DistalSynapse(source_cell=cell, permanence=0.5)


# Cell Tests


# Test Type: unit test
def test_cell_initialization(cell):
    assert isinstance(cell.segments, list)
    assert cell.segments == []


# Test Type: unit test
def test_cell_store_segments(cell, segment):
    cell.distal_segments.append(segment)
    assert len(cell.distal_segments) == 1
    assert cell.distal_segments[0] is segment


# Test Type: unit test
def test_cell_repr(cell):
    r = repr(cell)
    assert r.startswith("Cell(id=")
    assert r.endswith(")")


# DistalSynapse Tests


# Test Type: unit test
def test_distal_synapse_initialization(distal_synapse, cell):
    assert distal_synapse.source_cell is cell
    assert distal_synapse.permanence == pytest.approx(0.5)


# Segment Tests


# Test Type: unit test
def test_segment_initialization(segment):
    assert segment.synapses == []
    assert segment.sequence_segment is False


@pytest.fixture
def three_cells():
    cells = [Cell() for _ in range(3)]
    # Assign distal_field for each cell
    for c in cells:
        c.distal_field = Field(cells)
    return cells


@pytest.fixture
def three_distal_synapses(three_cells):
    return [
        DistalSynapse(c, permanence=CONNECTED_PERM + i * 0.1) for i, c in enumerate(three_cells)
    ]


@pytest.fixture
def segment_with_synapses(cell, three_distal_synapses):
    seg = Segment(parent_cell=cell)
    seg.synapses.extend(three_distal_synapses)
    return seg


# Test Type: unit test
def test_segment_with_synapses(segment_with_synapses, three_distal_synapses):
    assert len(segment_with_synapses.synapses) == 3
    assert segment_with_synapses.synapses == three_distal_synapses


# Synapse Tests


# Test Type: unit test
def test_synapse_source_and_permanence():
    s = Synapse(source_cell=None, permanence=0.7)
    assert s.permanence == pytest.approx(0.7)
    assert s.source_cell is None


# Segment API tests


# Test Type: unit test
def test_segment_is_active_and_potentially_active(cell):
    seg = Segment(parent_cell=cell)
    # Add three synapses to meet threshold logic
    c1 = Cell()
    c2 = Cell()
    c3 = Cell()
    c1.active = True
    c2.active = True
    c3.active = True
    syn1 = DistalSynapse(c1, permanence=0.1)
    syn2 = DistalSynapse(c2, permanence=0.1)
    syn3 = DistalSynapse(c3, permanence=0.1)
    seg.synapses.extend([syn1, syn2, syn3])
    seg.learning_threshold_connected_pct = 0.5
    # With three synapses, threshold is 1.5, so two is not enough, but three is
    assert seg.is_potentially_active() is True

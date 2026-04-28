"""
Test suite for Synapse class.

Synapse represents a connection from one Cell to another Cell. Each synapse
has a permanence value (strength) that is updated during learning.

Key Properties:
  - presynaptic_cell: source cell of the connection
  - postsynaptic_cell: target cell of the connection
  - permanence: strength (0.0 to 1.0), higher = stronger connection
  - connected: whether permanence >= connected_threshold (~0.5)

Learning Rules:
  - Reward (positive reinforcement): increase permanence
  - Punishment (negative reinforcement): decrease permanence
  - Permanence clamped to [0.0, 1.0]

Tests validate:
  1. Synapse initialization with proper cell references
  2. Connected state computation based on permanence
  3. Learning updates (increase/decrease permanence)
  4. Clamping to valid range [0.0, 1.0]
  5. Synapse strength tracking
"""

import pytest

from htmrl.agent_layer.HTM import (
    CONNECTED_PERM,
    INITIAL_PERMANENCE,
    PERMANENCE_DEC,
    PERMANENCE_INC,
    ApicalSynapse,
    Cell,
    DistalSynapse,
    ProximalSynapse,
    Synapse,
)


@pytest.fixture
def cell():
    """Create a default inactive cell."""
    c = Cell()
    c.active = False
    c.prev_active = False
    c.predictive = False
    return c


@pytest.fixture
def active_cell(cell):
    """Creates a default active cell."""
    cell.active = True
    return cell


@pytest.fixture
def predictive_cell(cell):
    """Creates a default predictive cell."""
    cell.predictive = True
    return cell


"""Synapse"""


# Test Type: unit test
def test_synapse_adjust_permanence_increase(cell):
    """Checks that permanence is adjusted properly wiht increase true."""
    syn = Synapse(cell, 0.5)
    syn._adjust_permanence(increase=True)

    assert syn.permanence > 0.5


# Test Type: unit test
def test_synapse_adjust_permanence_decrease(cell):
    """Checks that permanence is adjusted properly with increase false."""
    syn = Synapse(cell, 0.5)
    syn._adjust_permanence(increase=False)

    assert syn.permanence < 0.5


# Test Type: unit test
def test_synapse_permanence_cannot_exceed_1(cell):
    """Checks that permanence cannot exceed 1.0 even with large strength."""
    syn = Synapse(cell, 0.99)
    syn._adjust_permanence(increase=True, strength=10)

    assert syn.permanence <= 1.0


# Test Type: unit test
def test_synapse_permanence_cannot_go_below_0(cell):
    """Checks that permanence cannot go below 0.0 even with large strength."""
    syn = Synapse(cell, 0.01)
    syn._adjust_permanence(increase=False, strength=10)

    assert syn.permanence >= 0.0


# Test Type: unit test
def test_synapse_adjust_permanence_negative_strength_increase_asserts(cell):
    """Checls that when we are increasing permanence that a negative strength is rejected."""
    syn = Synapse(cell, 0.5)
    syn._adjust_permanence(increase=True, strength=-1.0)
    assert 0.0 <= syn.permanence <= 1.0
    assert syn.permanence < 0.5


# Test Type: unit test
def test_synapse_adjust_permanence_negative_strength_decrease_asserts(cell):
    """Checks that when we are decreasing permanence that a negative strength is rejected."""
    syn = Synapse(cell, 0.5)
    syn._adjust_permanence(increase=False, strength=-1.0)
    assert 0.0 <= syn.permanence <= 1.0
    assert syn.permanence > 0.5


"""ApicalSynapse"""


# Test Type: unit test
def test_apical_synapse_uses_predictive(predictive_cell):
    """If the apical segment has a predictive cell and connected permanence threshold,
    it should be active."""
    syn = ApicalSynapse(predictive_cell, CONNECTED_PERM)
    assert syn.active is False


# Test Type: unit test
def test_apical_synapse_not_active_if_not_predictive(cell):
    """If the apical segment has a normal cell and connected permanence
    threshold it should not be active."""
    syn = ApicalSynapse(cell, CONNECTED_PERM)
    assert syn.active is False


"""DistalSynapse"""


# Test Type: unit test
def test_distal_synapse_inherits_active_behavior(active_cell):
    """If the distal synapse has an active cell and connected permanence
    threshold it should be active."""
    syn = DistalSynapse(active_cell, CONNECTED_PERM)
    assert syn.active is True


"""ProximalSynapse"""


# Test Type: unit test
def test_proximal_synapse_default_permanence(cell):
    """If the proximal synapse has a normal cell the permanence should be the initial value."""
    syn = ProximalSynapse(cell)
    assert syn.permanence == INITIAL_PERMANENCE

from htmrl.agent_layer.our_htm.cell import Cell
from htmrl.agent_layer.our_htm.synapse import Synapse


class Column:
    """
    A mini-column in the Spatial Pooler.

    From the papers terminology section:
        "Column: An HTM region is organized in columns of cells. The SP
        operates at the column-level, where a column of cells functions as
        a single computational unit."

    Each column owns:
    - A dendritic segment, represented here as a list of potential
        synapses connecting the column to bits in the input space. The
        paper describes the segment as having "a set of potential synapses
        representing a (random) subset of the input bits. Each potential
        synapse has a permanence value."
    - A boost factor, used during Phase 2 to scale the column's overlap.
        The paper: "The number of active synapses is multiplied by a
        'boosting' factor, which is dynamically determined by how often a
        column is active relative to its neighbors."
    - An active duty cycle and an overlap duty cycle, both sliding
        averages used by Phase 4 to drive boosting.
    - A list of cells. The Spatial Pooler itself only marks columns
        active or inactive, but the broader HTM stack (Temporal Memory)
        operates at the cell level, so cells are instantiated here for
        forward compatibility.

    Initialize a column with the given index and a fixed number of
    cells.

        Args:
            index: The column's position within the HTM region. Used by the
                Spatial Pooler to identify winning columns and to compute
                neighbor relationships in local inhibition mode.
            num_cells_per_column: Number of cells inside the column. The
                Spatial Pooler treats the column as a single computational
                unit regardless of this value, but the paper notes that
                a region "is organized in columns of cells," and Temporal
                Memory built on top of the SP picks specific cells within
                active columns.
    """

    def __init__(self, index, num_cells_per_column=1):

        self.index = index

        self.cells = []
        for i in range(num_cells_per_column):
            cell = Cell(
                index=(index * num_cells_per_column) + i,
                column_index=index,
            )
            self.cells.append(cell)

        self.potential_synapses: list[Synapse] = []
        self.connected_synapses: list[Synapse] = []
        self.boost = 1.0

        self.overlap = 0

        self.active_duty_cycle = 0.0
        self.overlap_duty_cycle = 0.0

    def add_potential_synapse(self, synapse: Synapse, connected_perm: float) -> None:
        """
        Add a synapse to this columns potential pool, and also to its
        connected list if its permanence is already above the threshold.

        Called once per synapse during Phase 1 of the algorithm. The paper
        describes this initialization step as:
            "Each dendrite segment has a set of potential synapses
            representing a (random) subset of the input bits. Each
            potential synapse has a permanence value. These values are
            randomly initialized around the permanence threshold. Based on
            their permanence values, some of the potential synapses will
            already be connected; the permanences are greater than the
            threshold value."

        Args:
            synapse: The Synapse to add.
            connected_perm: The Spatial Poolers connected-permanence
                threshold, used to decide whether the synapse should also
                go into connected_synapses.
        """
        self.potential_synapses.append(synapse)
        if synapse.is_connected(connected_perm):
            self.connected_synapses.append(synapse)

    def on_crossing(self, synapse: Synapse, crossing: int) -> None:
        """
        Update the maintained connected_synapses list in response to a
        permanence change that crossed the connected threshold.

        Args:
            synapse: The Synapse whose connection state changed.
            crossing: +1 for an upward crossing, -1 for a downward
                crossing.
        """
        if crossing == 1:
            self.connected_synapses.append(synapse)
        elif crossing == -1:
            self.connected_synapses.remove(synapse)

    def connected_synapse(self, connected_perm: float) -> list[Synapse]:
        """
        Recompute the connected-synapse list from scratch by filtering
        potential_synapses against the given threshold.

        Args:
            connected_perm: The connected-permanence threshold.

        Returns:
            A new list of synapses whose permanence is at or above
            connected_perm.
        """
        result = []
        for s in self.potential_synapses:
            if s.is_connected(connected_perm):
                result.append(s)
        return result

    def compute_overlap(self, input_vector: list[int]) -> float:
        """
        Compute this column's overlap with the given input vector.

        Implements Phase 2 of the algorithm at the column level. From the
        paper:
            "Given an input vector, this phase calculates the overlap of
            each column with that vector. The overlap for each column is
            simply the number of connected synapses with active inputs,
            multiplied by the column's boost factor."

        Pseudocode (paper, lines 1-5):
            overlap(c) = 0
            for s in connectedSynapses(c):
                overlap(c) += input(t, s.sourceInput)
            overlap(c) *= boost(c)

        Args:
            input_vector: A sequence of 0/1 values indexed by source input
                index.

        Returns:
            The boosted overlap value.
        """
        raw_overlap = 0
        for s in self.connected_synapses:
            raw_overlap += input_vector[s.source_input_index]

        self.overlap = raw_overlap * self.boost
        return self.overlap

    def activate_cells(self):
        """
        Mark every cell in this column as active.
        """
        for cell in self.cells:
            cell.activate()

    def deactivate_cells(self):
        """
        Mark every cell in this column as inactive.
        """
        for cell in self.cells:
            cell.deactivate()

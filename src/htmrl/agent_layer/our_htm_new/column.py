from htmrl.agent_layer.our_htm_new.cell import Cell
from htmrl.agent_layer.our_htm_new.synapse import Synapse


class Column:
    """Represents a column in the HTM spatial pooler.

    Args:
        index: The index of the column.
        num_cells_per_column: Number of cells in this column.
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

    def add_potential_synapse(self, synapse, connected_perm):
        """Add a potential synapse to the column.

        Args:
            synapse: The synapse to add.
            connected_perm: The permanence threshold for connection.
        """
        self.potential_synapses.append(synapse)
        if synapse.is_connected(connected_perm):
            self.connected_synapses.append(synapse)

    def on_crossing(self, synapse, crossing):
        """Handle synapse permanence crossing the threshold.

        Args:
            synapse: The synapse that crossed.
            crossing: 1 for connected, -1 for disconnected.
        """
        if crossing == 1:
            self.connected_synapses.append(synapse)
        elif crossing == -1:
            self.connected_synapses.remove(synapse)

    def connected_synapse(self, connected_perm):
        """Get connected synapses based on permanence.

        Args:
            connected_perm: The permanence threshold.

        Returns:
            List of connected synapses.
        """
        result = []
        for s in self.potential_synapses:
            if s.is_connected(connected_perm):
                result.append(s)
        return result

    def compute_overlap(self, input_vector):
        """Compute the overlap score with the input vector.

        Args:
            input_vector: The input SDR.

        Returns:
            The overlap score.
        """
        raw_overlap = 0
        for s in self.connected_synapses:
            raw_overlap += input_vector[s.source_input_index]

        self.overlap = raw_overlap * self.boost
        return self.overlap

    def activate_cells(self):
        """Activate all cells in the column."""
        for cell in self.cells:
            cell.activate()

    def deactivate_cells(self):
        """Deactivate all cells in the column."""
        for cell in self.cells:
            cell.deactivate()

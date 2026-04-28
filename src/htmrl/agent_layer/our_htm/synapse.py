class Synapse:
    """
    A synapse on a columns dendritic segment in the Spatial Pooler.

    From the paper's terminology section:
        "Synapse: A junction between cells. In the Spatial Pooling
        algorithm, synapses on a column's dendritic segment connect to
        bits in the input space. A synapse can be in the following states:
            Connected - permanence is above the threshold.
            Potential - permanence is below the threshold.
            Unconnected - does not have the ability to connect."
        Initialize a synapse.

        Args:
            source_input_index: The index of the input bit this synapse
                connects to.
            permanence: The initial permanence value, in the range [0, 1].
    """

    def __init__(self, source_input_index: int, permanence: float) -> None:

        self.source_input_index = source_input_index

        self.permanence = permanence

    def is_connected(self, connected_perm: float) -> bool:
        """
        Return whether this synapse is currently connected.

        From the paper's terminology section:
            "Permanence threshold: If a synapse's permanence is above
            this value, it is considered fully connected. Acceptable
            values are [0, 1]."

        Args:
            connected_perm: The connected-permanence threshold from the
                Spatial Pooler.

        Returns:
            True if permanence >= connected_perm, False otherwise.
        """
        return self.permanence >= connected_perm

    def increment_permanence(self, amount: float, connected_perm: float) -> int:
        """
        Increase the permanence value, clamping to 1.0.

        Called by Phase 4 of the Spatial Pooler in two situations:

          1. Hebbian learning. The paper's pseudocode (lines 13-15):
                if active(s) then
                    s.permanence += synPermActiveInc
                    s.permanence = min(1.0, s.permanence)
             A synapse on a winning column whose source input bit was on
             during this iteration gets its permanence incremented by
             `synPermActiveInc`.

          2. Duty-cycle-driven boosting. The paper's pseudocode (line 25)
             calls `increasePermanences(c, 0.1*connectedPerm)` on any
             column whose overlap duty cycle has fallen below its
             minimum, raising every synapse in the column by a small
             amount so that the column can "search for new inputs."
        Args:
            amount: The amount to add to the permanence before clamping.
            connected_perm: The connected-permanence threshold, used to
                detect a crossing.

        Returns:
            +1 if this update moved the synapse from unconnected to
            connected, 0 if no crossing occurred.
        """
        was_connected = self.permanence >= connected_perm
        self.permanence = min(1.0, self.permanence + amount)
        now_connected = self.permanence >= connected_perm
        return 1 if (now_connected and not was_connected) else 0

    def decrement_permanence(self, amount: float, connected_perm: float) -> int:
        """
        Decrease the permanence value, clamping to 0.0.

        Called by Phase 4 of the Spatial Pooler during Hebbian learning.
        The paper's pseudocode (lines 16-18):
            else
                s.permanence -= synPermInactiveDec
                s.permanence = max(0.0, s.permanence)
        Args:
            amount: The amount to subtract from the permanence before
                clamping.
            connected_perm: The connected-permanence threshold, used to
                detect a crossing.

        Returns:
            -1 if this update moved the synapse from connected to
            unconnected, 0 if no crossing occurred.
        """
        was_connected = self.permanence >= connected_perm
        self.permanence = max(0.0, self.permanence - amount)
        now_connected = self.permanence >= connected_perm
        return -1 if (was_connected and not now_connected) else 0

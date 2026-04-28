class Synapse:
    """Represents a synapse connecting a column to an input bit.

    Args:
        source_input_index: Index of the input bit this synapse connects to.
        permanence: The permanence value of the synapse.
    """

    def __init__(self, source_input_index, permanence):
        self.source_input_index = source_input_index

        self.permanence = permanence

    def is_connected(self, connected_perm):
        """Check if the synapse is connected.

        Args:
            connected_perm: The permanence threshold for connection.

        Returns:
            True if connected, False otherwise.
        """
        return self.permanence >= connected_perm

    def increment_permanence(self, amount, connected_perm):
        """Increment the permanence and check for crossing.

        Args:
            amount: Amount to increment.
            connected_perm: The permanence threshold.

        Returns:
            1 if became connected, 0 otherwise.
        """
        was_connected = self.permanence >= connected_perm
        self.permanence = min(1.0, self.permanence + amount)
        now_connected = self.permanence >= connected_perm
        return 1 if (now_connected and not was_connected) else 0

    def decrement_permanence(self, amount, connected_perm):
        """Decrement the permanence and check for crossing.

        Args:
            amount: Amount to decrement.
            connected_perm: The permanence threshold.

        Returns:
            -1 if became disconnected, 0 otherwise.
        """
        was_connected = self.permanence >= connected_perm
        self.permanence = max(0.0, self.permanence - amount)
        now_connected = self.permanence >= connected_perm
        return -1 if (was_connected and not now_connected) else 0

class Cell:
    """Represents a single cell in the HTM column.

    Args:
        index: The index of the cell within its column.
        column_index: The index of the column this cell belongs to.
    """

    def __init__(self, index, column_index):
        self.index = index

        self.column_index = column_index

        self.active = False

    def activate(self):
        """Activate the cell."""
        self.active = True

    def deactivate(self):
        """Deactivate the cell."""
        self.active = False

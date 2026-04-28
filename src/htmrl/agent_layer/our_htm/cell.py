class Cell:
    """
    A single cell within a Spatial Pooler column.

    From the paper's terminology section:
        "Column: An HTM region is organized in columns of cells. The SP
        operates at the column-level, where a column of cells functions as
        a single computational unit."
    Initialize a cell.
        Args:
            index: The cell's global index across the entire HTM region.
            column_index: The index of the column this cell belongs to.
    """

    def __init__(self, index, column_index):

        self.index = index

        self.column_index = column_index

        self.active = False

    def activate(self):
        """Mark this cell as active."""
        self.active = True

    def deactivate(self):
        """Mark this cell as inactive."""
        self.active = False

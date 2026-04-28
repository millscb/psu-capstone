import random
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from htmrl.agent_layer.pullin.pullin_htm import Cell


class Field:
    """A collection of cells."""

    def __init__(self, cells: Iterable["Cell"], name: str | None = None) -> None:
        self.cells: list["Cell"] = list(cells)
        self.name: str = name if name is not None else ""

    def __iter__(self):
        return iter(self.cells)

    def sample(self, pct: float) -> set["Cell"]:
        """Sample 'pct' percent cells from the field."""
        n = int(len(self.cells) * pct)
        if n > len(self.cells):
            raise ValueError("Cannot sample more cells than are in the field.")
        return set(random.sample(self.cells, n))

    def reset(self) -> None:
        """Reset all cells in the field to initial state."""
        for cell in self.cells:
            cell.clear_state()

    @property
    def active_cells(self) -> set["Cell"]:
        """Return set of currently active cells in the field."""
        return {cell for cell in self.cells if cell.active}

    @property
    def prev_active_cells(self) -> set["Cell"]:
        """Return set of previously active cells in the field."""
        return {cell for cell in self.cells if cell.prev_active}

    @property
    def predictive_cells(self) -> set["Cell"]:
        """Return set of currently predictive cells in the field."""
        return {cell for cell in self.cells if cell.predictive}

    @property
    def prev_predictive_cells(self) -> set["Cell"]:
        """Return set of previously predictive cells in the field."""
        return {cell for cell in self.cells if cell.prev_predictive}

    @property
    def prev_learning_cells(self) -> set["Cell"]:
        """Return set of previously learning cells in the field."""
        return {cell for cell in self.cells if cell.prev_learning}

    @property
    def prev_winner_cells(self) -> set["Cell"]:
        """Return set of previously winning cells in the field."""
        return {cell for cell in self.cells if cell.prev_winner}

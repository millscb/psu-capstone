"""HTM Temporal Memory (TM): learns and predicts temporal sequences over columns.

Overview:
- TM operates over the set of columns produced by the Spatial Pooler (SP).
- Each column contains multiple cells to represent different temporal contexts.
- Distal segments on cells aggregate synapses to previously active cells; when
  enough synapses are active/connected, the segment activates and the cell becomes predictive.

Core phases per timestep:
1) Active state:
   - If a column was correctly predicted, only its predictive cells become active.
   - Otherwise, the column bursts (all its cells become active) and a winner cell is chosen.
2) Predictive state:
   - Segments evaluate active synapses against current active cells; cells with
     sufficiently active segments become predictive for the next timestep.
3) Learning:
   - Reinforce segments associated with correct predictions and winning contexts:
     increase permanence on synapses whose source cells were active, decrease otherwise;
     grow new synapses to recently active cells (up to NEW_SYNAPSE_MAX).
   - Punish segments that predicted incorrectly by decreasing permanence.

Outputs:
- Binary vectors over cells for active/predictive/learning (winner) states.
"""

# htm_core/temporal_memory.py
from __future__ import annotations

import random

import numpy as np

from legacy.legacy_htm.cell import Cell
from legacy.legacy_htm.column import Column
from legacy.legacy_htm.constants import (
    INITIAL_DISTAL_PERM,
    NEW_SYNAPSE_MAX,
    PERMANENCE_DEC,
    PERMANENCE_INC,
    SEGMENT_ACTIVATION_THRESHOLD,
)
from legacy.legacy_htm.distal_synapse import DistalSynapse
from legacy.legacy_htm.segment import Segment


class TemporalMemory:
    """Temporal Memory: learns transitions between column activations.

    Responsibilities:
    - Maintain per-timestep sets of active, predictive, and winner cells.
    - Compute active and predictive states from current inputs and learned segments.
    - Update distal synapse permanence and grow new synapses to encode correct transitions.

    Notes:
    - cells_per_column controls sequence capacity; more cells allow richer contexts.
    - Learning thresholds and permanence increments/decays govern stability/sensitivity.
    """

    def __init__(
        self,
        columns: list[Column],
        cells_per_column: int,
    ) -> None:
        self.columns: list[Column] = list(columns)
        self.cells_per_column: int = cells_per_column

        # Attach cells to each column
        for c in self.columns:
            c.cells = [Cell() for _ in range(cells_per_column)]

        # Time-indexed TM state
        self.active_cells: dict[int, set[Cell]] = {}
        self.winner_cells: dict[int, set[Cell]] = {}
        self.predictive_cells: dict[int, set[Cell]] = {}
        self.learning_segments: dict[int, set[Segment]] = {}
        self.negative_segments: dict[int, set[Segment]] = {}

        self.current_t: int = 0

        # Optional column -> field mapping if the SP builds one
        self.column_field_map: dict[Column, str | None] = {}

    # ---------- Core step API ----------

    def step(self, active_columns: list[Column]) -> dict[str, np.ndarray]:
        """Advance TM one time step given the active columns.

        Process:
        - Filter/validate active columns.
        - Compute active state, predictive state, and perform learning.
        - Return binary vectors for active, predictive, and winner (learning) cells.

        Args:
            active_columns: List of Column objects that are currently active.

        Returns:
            Dictionary with binary vectors for active_cells, predictive_cells, and learning_cells.
        """
        t = self.current_t
        active_columns = [c for c in active_columns if isinstance(c, Column)]

        self._compute_active_state(active_columns)
        self._compute_predictive_state()
        self._learn()

        self.current_t += 1

        active_cells_vec = self.cells_to_binary(self.active_cells.get(t, set()))
        predictive_cells_vec = self.cells_to_binary(self.predictive_cells.get(t, set()))
        learning_cells_vec = self.cells_to_binary(self.winner_cells.get(t, set()))

        return {
            "active_cells": active_cells_vec,
            "predictive_cells": predictive_cells_vec,
            "learning_cells": learning_cells_vec,
        }

    # ---------- Core TM logic ----------

    def _compute_active_state(self, active_columns: list[Column]) -> None:
        t = self.current_t
        prev_predictive = self.predictive_cells.get(t - 1, set())
        active_cells_t: set[Cell] = set()
        winner_cells_t: set[Cell] = set()
        learning_segments_t: set[Segment] = set()

        for column in active_columns:
            predictive_cells_prev = [cell for cell in column.cells if cell in prev_predictive]
            if predictive_cells_prev:
                # Correctly predicted column
                for cell in predictive_cells_prev:
                    active_cells_t.add(cell)
                    winner_cells_t.add(cell)
                    for seg in self._active_segments_of(cell, t - 1):
                        learning_segments_t.add(seg)
            else:
                # Bursting: all cells active
                for cell in column.cells:
                    active_cells_t.add(cell)
                best_cell, best_segment = self._best_matching_cell(column, t - 1)
                if best_segment is None:
                    if best_cell is None:
                        best_cell = column.cells[0]
                    best_segment = Segment()
                    best_cell._segments.append(best_segment)
                if best_cell is not None:
                    winner_cells_t.add(best_cell)
                learning_segments_t.add(best_segment)

        self.active_cells[t] = active_cells_t
        self.winner_cells[t] = winner_cells_t
        self.learning_segments[t] = learning_segments_t
        print(f"[TM] Active state at t={t}: {len(active_cells_t)} cells active.")

    def _compute_predictive_state(self) -> None:
        """Determine cells that become predictive based on active segments.

        Rule:
        - A cell is predictive at t if any of its segments has at least
          SEGMENT_ACTIVATION_THRESHOLD active synapses whose source cells are active at t.

        Side effects:
        - Updates self.predictive_cells[t].
        """
        t = self.current_t
        active_cells_t = self.active_cells.get(t, set())
        predictive_cells_t: set[Cell] = set()
        for column in self.columns:
            for cell in column.cells:
                for seg in cell._segments:
                    if len(seg.active_synapses(active_cells_t)) >= SEGMENT_ACTIVATION_THRESHOLD:
                        predictive_cells_t.add(cell)
                        break
        self.predictive_cells[t] = predictive_cells_t
        print(f"[TM] Predictive state at t={t}: {len(predictive_cells_t)} cells predictive.")

    def _learn(self) -> None:
        """Apply positive and negative learning to distal segments.

        Steps:
        - Identify negative segments: segments that were active at t-1 but whose columns did not become active at t.
        - Reinforce learning segments: increase permanence for synapses whose source cells were active at t-1,
          decrease permanence otherwise; grow new synapses to recently active cells.
        - Punish negative segments: decrease permanence.

        Side effects:
        - Updates self.learning_segments[t], self.negative_segments[t], and synapse permanence.
        """
        t = self.current_t
        prev_predictive = self.predictive_cells.get(t - 1, set())
        active_columns = {
            c
            for c in self.columns
            if any(cell in self.active_cells.get(t, set()) for cell in c.cells)
        }
        negative_segments: set[Segment] = set()

        # Identify segments that predicted but whose columns did not become active
        for column in self.columns:
            if column not in active_columns:
                for cell in column.cells:
                    if cell in prev_predictive:
                        for seg in self._active_segments_of(cell, t - 1):
                            negative_segments.add(seg)
        self.negative_segments[t] = negative_segments

        # Positive reinforcement
        for seg in self.learning_segments.get(t, set()):
            self._reinforce_segment(seg)

        # Negative reinforcement
        for seg in negative_segments:
            self._punish_segment(seg)

        print(
            f"[TM] Learning at t={t}: +{len(self.learning_segments.get(t, set()))} / "
            f"-{len(negative_segments)} segments."
        )

    # ---------- Helpers (belong with TM) ----------

    def cells_to_binary(self, cells: set[Cell]) -> np.ndarray:
        """Return binary vector over all cells (flattened across columns)."""
        total_cells = len(self.columns) * self.cells_per_column
        vec = np.zeros(total_cells, dtype=int)
        for col_idx, col in enumerate(self.columns):
            base = col_idx * self.cells_per_column
            for local_idx, cell in enumerate(col.cells):
                if cell in cells:
                    vec[base + local_idx] = 1
        return vec

    def get_predictive_columns_mask(self, t: int | None = None) -> np.ndarray:
        """Return binary vector of predictive columns for time t."""
        if not self.predictive_cells:
            return np.zeros(len(self.columns), dtype=int)
        max_t = max(self.predictive_cells.keys())
        if t is None:
            query_t = max_t
        elif t == -1:
            query_t = max_t - 1
        else:
            query_t = t
        if query_t < 0:
            return np.zeros(len(self.columns), dtype=int)
        pred_cells = self.predictive_cells.get(query_t, set())
        cols = {col for col in self.columns if any(cell in pred_cells for cell in col.cells)}
        mask = np.zeros(len(self.columns), dtype=int)
        for idx, col in enumerate(self.columns):
            if col in cols:
                mask[idx] = 1
        return mask

    def reset_state(self) -> None:
        """Reset transient TM state while preserving learned segments."""
        self.active_cells.clear()
        self.winner_cells.clear()
        self.predictive_cells.clear()
        self.learning_segments.clear()
        self.negative_segments.clear()
        self.current_t = 0

    # ---------- Lower-level TM helpers ----------

    def _best_matching_cell(
        self, column: Column, prev_t: int
    ) -> tuple[Cell | None, Segment | None]:
        prev_active_cells = self.active_cells.get(prev_t, set())
        best_cell: Cell | None = None
        best_segment: Segment | None = None
        best_match = -1

        for cell in column.cells:
            if not cell._segments:
                # Prefer unused cell if no better match yet
                if best_match == -1:
                    best_cell = cell
                    best_segment = None
                    best_match = 0
                continue
            for seg in cell._segments:
                match_count = len(seg.matching_synapses(prev_active_cells))
                if match_count > best_match:
                    best_match = match_count
                    best_cell = cell
                    best_segment = seg
        return best_cell, best_segment

    def _active_segments_of(self, cell: Cell, t: int) -> list[Segment]:
        prev_active_cells = self.active_cells.get(t, set())
        active_list: list[Segment] = []
        for seg in cell._segments:
            if len(seg.active_synapses(prev_active_cells)) >= SEGMENT_ACTIVATION_THRESHOLD:
                active_list.append(seg)
        return active_list

    def _reinforce_segment(self, segment: Segment) -> None:
        """Apply positive learning to a segment: adjust permanence and grow new synapses.

        Steps:
        - Increase permanence of synapses whose source cells were active at t-1.
        - Decrease permanence otherwise.
        - Add up to NEW_SYNAPSE_MAX new synapses to previously active cells not yet connected.
        - Mark the segment as a sequence_segment.
        """
        t = self.current_t
        prev_active_cells = self.active_cells.get(t - 1, set())
        # Strengthen existing active synapses, weaken others
        for syn in segment.synapses:
            if syn.source_cell in prev_active_cells:
                syn.permanence = min(1.0, syn.permanence + PERMANENCE_INC)
            else:
                syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)
        # Grow new synapses
        existing_sources = {syn.source_cell for syn in segment.synapses}
        candidates = [c for c in prev_active_cells if c not in existing_sources]
        random.shuffle(candidates)
        for cell_src in candidates[:NEW_SYNAPSE_MAX]:
            segment.synapses.append(DistalSynapse(cell_src, INITIAL_DISTAL_PERM))
        segment.sequence_segment = True

    def _punish_segment(self, segment: Segment) -> None:
        """Apply negative learning to a segment: decrease permanence on all synapses."""
        for syn in segment.synapses:
            syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)


# smoke check

if __name__ == "__main__":
    from typing import Any, cast

    from htmrl.agent_layer.htm.spatial_pooler import SpatialPooler

    # Create a simple SP and TM
    input_size = 20
    num_columns = 10
    cells_per_column = 4

    sp = SpatialPooler(
        input_space_size=input_size,
        column_count=num_columns,
        initial_synapses_per_column=int(0.5 * input_size),
        random_seed=42,
    )

    tm = TemporalMemory(
        columns=sp.columns,
        cells_per_column=cells_per_column,
    )

    # Dummy input sequence
    input_sequence = [np.random.randint(0, 2, size=input_size) for _ in range(5)]

    for step_idx, input_vector in enumerate(input_sequence):
        print(f"\n=== Step {step_idx} ===")
        active_columns_mask, active_columns = sp.compute_active_columns(
            input_vector, inhibition_radius=2.0
        )

        tm_output = tm.step(active_columns)
        print(f"Active Cells: {tm_output['active_cells']}")
        print(f"Predictive Cells: {tm_output['predictive_cells']}")
        print(f"Learning Cells: {tm_output['learning_cells']}")

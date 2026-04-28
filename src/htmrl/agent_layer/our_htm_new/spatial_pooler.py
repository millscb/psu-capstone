import math
import random

from htmrl.agent_layer.our_htm_new.cell import Cell
from htmrl.agent_layer.our_htm_new.column import Column
from htmrl.agent_layer.our_htm_new.synapse import Synapse


class SpatialPooler:
    """Implements the HTM Spatial Pooler algorithm.

    Args:
        input_size: Size of the input SDR.
        column_count: Number of columns.
        potential_radius: Radius for potential synapses.
        potential_pct: Percentage of potential synapses.
        global_inhibition: Whether to use global inhibition.
        num_active_columns_per_inh_area: Number of active columns per inhibition area.
        stimulus_threshold: Minimum overlap for activation.
        syn_perm_active_inc: Permanence increment for active synapses.
        syn_perm_inactive_dec: Permanence decrement for inactive synapses.
        connected_perm: Permanence threshold for connection.
        min_pct_overlap_duty_cycle: Minimum overlap duty cycle percentage.
        duty_cycle_period: Period for duty cycle calculation.
        boost_strength: Strength of boosting.
        seed: Random seed.
    """

    def __init__(
        self,
        input_size,
        column_count=2048,
        potential_radius=None,
        potential_pct=0.75,
        global_inhibition=True,
        num_active_columns_per_inh_area=40,
        stimulus_threshold=0,
        syn_perm_active_inc=0.03,
        syn_perm_inactive_dec=0.015,
        connected_perm=0.2,
        min_pct_overlap_duty_cycle=0.001,
        duty_cycle_period=1000,
        boost_strength=2.0,
        seed=42,
    ):

        self.input_size = input_size
        self.column_count = column_count

        self.potential_radius = potential_radius if potential_radius is not None else input_size
        self.potential_pct = potential_pct
        self.global_inhibition = global_inhibition
        self.num_active_columns_per_inh_area = num_active_columns_per_inh_area
        self.stimulus_threshold = stimulus_threshold
        self.syn_perm_active_inc = syn_perm_active_inc
        self.syn_perm_inactive_dec = syn_perm_inactive_dec
        self.connected_perm = connected_perm
        self.min_pct_overlap_duty_cycle = min_pct_overlap_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.boost_strength = boost_strength

        self._rng = random.Random(seed)

        self.inhibition_radius = input_size

        self.iteration = 0

        self.columns: list[Column] = []
        for i in range(column_count):
            self.columns.append(Column(index=i))
        self._initialize_potential_synapses()

    def _initialize_potential_synapses(self):

        for col in self.columns:
            center = int(col.index * (self.input_size / self.column_count))
            low = max(0, center - self.potential_radius)
            high = min(self.input_size, center + self.potential_radius + 1)

            window = []
            for i in range(low, high):
                window.append(i)

            num_potential = max(1, int(len(window) * self.potential_pct))
            chosen = self._rng.sample(window, num_potential)

            for src in chosen:
                perm = self._rng.uniform(
                    max(0.0, self.connected_perm - 0.1), min(1.0, self.connected_perm + 0.1)
                )
                syn = Synapse(src, perm)
                col.add_potential_synapse(syn, self.connected_perm)

    def compute(self, input_vector, learn=True):
        """Compute the spatial pooling for the given input.

        Args:
            input_vector: The input SDR.
            learn: Whether to perform learning.

        Returns:
            List of active column indices.
        """
        if len(input_vector) != self.input_size:
            raise ValueError(
                f"Input vector length {len(input_vector)} != expected {self.input_size}"
            )
        self.iteration += 1

        self._phase2_overlap(input_vector)
        active_columns = self._phase3_inhibition()

        if learn:
            self._phase4_learn(input_vector, active_columns)

        for col in self.columns:
            col.deactivate_cells()
        for c_idx in active_columns:
            self.columns[c_idx].activate_cells()

        return active_columns

    def _phase2_overlap(self, input_vector):
        for col in self.columns:
            col.compute_overlap(input_vector)

    def _phase3_inhibition(self):
        active = []
        k = self.num_active_columns_per_inh_area

        if self.global_inhibition:
            sorted_cols = sorted(self.columns, key=lambda c: c.overlap, reverse=True)
            if len(sorted_cols) < k:
                threshold = 0
            else:
                threshold = sorted_cols[k - 1].overlap
            for col in self.columns:
                if (
                    col.overlap >= self.stimulus_threshold
                    and col.overlap >= threshold
                    and col.overlap > 0
                ):
                    active.append(col.index)
                    if len(active) >= k:
                        break
        else:
            for col in self.columns:
                neighbors = self._neighbors(col.index)

                # collect overlap values from neighbors, then sort descending.
                neighbor_overlaps = []
                for n in neighbors:
                    neighbor_overlaps.append(self.columns[n].overlap)
                neighbor_overlaps.sort(reverse=True)

                if len(neighbor_overlaps) < k:
                    min_local_activity = 0
                else:
                    min_local_activity = neighbor_overlaps[k - 1]
                if (
                    col.overlap > self.stimulus_threshold
                    and col.overlap >= min_local_activity
                    and col.overlap > 0
                ):
                    active.append(col.index)

        return active

    def _neighbors(self, column_index):
        low = max(0, column_index - self.inhibition_radius)
        high = min(self.column_count, column_index + self.inhibition_radius + 1)

        neighbors = []
        for i in range(low, high):
            neighbors.append(i)
        return neighbors

    def _phase4_learn(self, input_vector, active_columns):
        cp = self.connected_perm
        inc = self.syn_perm_active_inc
        dec = self.syn_perm_inactive_dec

        for c_idx in active_columns:
            col = self.columns[c_idx]
            for syn in col.potential_synapses:
                if input_vector[syn.source_input_index] == 1:
                    crossing = syn.increment_permanence(inc, cp)
                else:
                    crossing = syn.decrement_permanence(dec, cp)
                if crossing:
                    col.on_crossing(syn, crossing)

        active_set = set(active_columns)
        for col in self.columns:
            self._update_active_duty_cycle(col, col.index in active_set)
            self._update_overlap_duty_cycle(col)

        if self.global_inhibition:
            total_adc = 0.0
            max_odc = 0.0
            for col in self.columns:
                total_adc += col.active_duty_cycle
                if col.overlap_duty_cycle > max_odc:
                    max_odc = col.overlap_duty_cycle
            mean_neighbor_active = total_adc / self.column_count
            min_duty_cycle = self.min_pct_overlap_duty_cycle * max_odc

            for col in self.columns:
                col.boost = self._boost_function(col.active_duty_cycle, mean_neighbor_active)
                if col.overlap_duty_cycle < min_duty_cycle:
                    self._increase_permanences(col, 0.1 * self.connected_perm)
        else:
            for col in self.columns:
                neighbors = self._neighbors(col.index)

                # sum active duty cycles across neighbors to get the mean.
                total_neighbor_adc = 0.0
                for n in neighbors:
                    total_neighbor_adc += self.columns[n].active_duty_cycle
                mean_neighbor_active = total_neighbor_adc / max(1, len(neighbors))

                col.boost = self._boost_function(col.active_duty_cycle, mean_neighbor_active)

                # find the max overlap duty cycle across neighbors.
                max_neighbor_overlap_dc = 0.0
                for n in neighbors:
                    neighbor_odc = self.columns[n].overlap_duty_cycle
                    if neighbor_odc > max_neighbor_overlap_dc:
                        max_neighbor_overlap_dc = neighbor_odc

                min_duty_cycle = self.min_pct_overlap_duty_cycle * max_neighbor_overlap_dc
                if col.overlap_duty_cycle < min_duty_cycle:
                    self._increase_permanences(col, 0.1 * self.connected_perm)

        self.inhibition_radius = self._average_receptive_field_size()

    def _update_active_duty_cycle(self, col, c_active):
        period = min(self.iteration, self.duty_cycle_period)
        col.active_duty_cycle = (
            col.active_duty_cycle * (period - 1) + (1.0 if c_active else 0.0)
        ) / period

    def _update_overlap_duty_cycle(self, col):
        period = min(self.iteration, self.duty_cycle_period)
        had_overlap = 1.0 if col.overlap >= self.stimulus_threshold and col.overlap > 0 else 0.0
        col.overlap_duty_cycle = (col.overlap_duty_cycle * (period - 1) + had_overlap) / period

    def _boost_function(self, active_duty_cycle, neighbor_mean_duty_cycle):
        return math.exp(-self.boost_strength * (active_duty_cycle - neighbor_mean_duty_cycle))

    def _increase_permanences(self, col, scale):
        cp = self.connected_perm
        for syn in col.potential_synapses:
            crossing = syn.increment_permanence(scale, cp)
            if crossing:
                col.on_crossing(syn, crossing)

    def _average_receptive_field_size(self):
        total = 0
        n = 0
        for col in self.columns:
            if not col.connected_synapses:
                continue

            # collect source indices of all connected synapses
            indices = []
            for s in col.connected_synapses:
                indices.append(s.source_input_index)

            radius = (max(indices) - min(indices)) / 2.0
            total += radius
            n += 1
        if n == 0:
            return self.input_size
        return max(1, int(total / n))

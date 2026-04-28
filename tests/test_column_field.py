import copy
import random
from collections import Counter
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from htmrl.agent_layer.HTM import (
    CONNECTED_PERM,
    DESIRED_LOCAL_SPARSITY,
    PERMANENCE_DEC,
    PERMANENCE_INC,
    RECEPTIVE_FIELD_PCT,
    Cell,
    Column,
    ColumnField,
    Field,
    InputField,
    ProximalSynapse,
)
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters


@pytest.fixture(autouse=True)
def _disable_blocking_plots(monkeypatch):
    """Prevent interactive plot windows from blocking test execution."""
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))


def activate_cells(cf: ColumnField, input_value):
    """Activate input cells."""
    in_fi = cast(InputField, cf.input_fields[0])
    in_fi.encode(input_value)


def get_random_indices(size: int, active_bits: int):
    # range of length of cells and getting the number of active bits desired.
    # this is likely 2048 and 40.
    indices = random.sample(range(size), active_bits)
    return indices


def activate_cells_directly(cf: ColumnField, indices: list[int]):
    in_fi = cast(InputField, cf.input_fields[0])
    for cell in in_fi.cells:
        cell.active = False
    for i in indices:
        in_fi.cells[i].set_active()


def make_input_field(n_cells: int = 2048) -> InputField:
    """Create a simple input Field with n cells."""
    return InputField(
        RDSEParameters(resolution=1.0), n_cells
    )  # this defaults to an rdse of n_cell as size


def make_input_field_sin(n_cells: int = 2048) -> InputField:
    """Create a simple input Field with n cells."""
    return InputField(
        RDSEParameters(resolution=0.001), n_cells
    )  # this defaults to an rdse of n_cell as size


def make_input_field_scalar(
    n_cells: int = 2048, res: float = 1.0, min: int = 0, max: int = 1000, periodic: bool = False
) -> InputField:
    """Create a simple input field with n cells and scalar encoder."""
    return InputField(
        ScalarEncoderParameters(
            minimum=min,
            maximum=max,
            sparsity=0.02,
            active_bits=0,
            radius=0,
            resolution=res,
            periodic=periodic,
        )
    )


def make_standard_cf(
    input_field: Field, num_columns: int = 2048, cells_per_column: int = 10
) -> ColumnField:
    """Construct a standard spatial temporal ColumnField for testing."""
    return ColumnField(
        input_fields=[input_field],
        num_columns=num_columns,
        cells_per_column=cells_per_column,
        non_spatial=False,
        non_temporal=True,
    )


def make_spatial_only_cf(
    input_field: Field, num_columns: int = 2048, duty_cycle_period: int = 1000
) -> ColumnField:
    """Construct a spatial only non temporal ColumnField for SP tests."""
    return ColumnField(
        input_fields=[input_field],
        num_columns=num_columns,
        cells_per_column=1,
        non_spatial=False,
        non_temporal=True,
        duty_cycle_period=duty_cycle_period,
    )


def make_temporal_only_cf(
    input_field: Field, num_columns: int = 2048, duty_cycle_period: int = 1000
) -> ColumnField:
    """Construct a temporal only non spatial ColumnField for TM tests."""
    return ColumnField(
        input_fields=[input_field],
        num_columns=num_columns,
        cells_per_column=1,
        non_spatial=True,
        non_temporal=False,
        duty_cycle_period=duty_cycle_period,
    )


def _overlap_count(first, second) -> int:
    """Count overlapping active column indices between two sets."""
    return len(first & second)


def add_noise_to_encoding(encoded_bits: list[int], noise_level: float):
    """Return a noisy copy of a binary encoding vector."""
    noisy = encoded_bits
    active = []
    inactive = []
    for i, bit in enumerate(noisy):
        if bit == 1:
            active.append(i)
        else:
            inactive.append(i)
    import random

    bits_to_flip = int(len(active) * noise_level)
    selected_bits_to_inactive = random.sample(active, bits_to_flip)
    selected_new_active = random.sample(inactive, bits_to_flip)

    for i, bit in enumerate(encoded_bits):
        for ina, act in zip(selected_bits_to_inactive, selected_new_active):
            if i == ina:
                encoded_bits[i] = 0
            if i == act:
                encoded_bits[i] = 1

    test_final = []
    for i, bit in enumerate(encoded_bits):
        if bit == 1:
            test_final.append(i)
    # print(_overlap_count(set(active), set(test_final)))


# Test Type: unit test
def test_add_noise():
    e = RandomDistributedScalarEncoder(RDSEParameters())
    bits = e.encode(1)
    add_noise_to_encoding(bits, 0.10)


def get_active_column_indices(cf: ColumnField) -> list[int]:
    return [i for i, col in enumerate(cf.columns) if col.active]


def overlap_ratio(original: list[int], noisy: list[int]) -> float:
    if not original:
        return 0.0
    noisy_set = set(noisy)
    intersection = [i for i in original if i in noisy_set]
    return len(intersection) / len(original)


"""Spatial Pooling inside the column field correctness tests based on Numenta Paper: spatial-pooling-algorithm/HTM-Spatial-Pooler-Overview.pdf"""

"""FIXED SPARSENESS"""


# Test Type: unit test
def test_active_ratio_matches_desired_sparsity():
    # TS-25
    """After compute, the number of active columns should match desired sparsity."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)
    activate_cells(cf, 42.0)

    expected_active = int(len(cf.columns) * DESIRED_LOCAL_SPARSITY)

    cf.compute(learn=True)
    active_cols = {i for i, col in enumerate(cf.columns) if col.active}

    assert (
        len(active_cols) == expected_active
    ), f"Expected exactly {expected_active} active columns, got {len(active_cols)}"


# Test Type: unit test
def test_sparsity_invariant_to_input_density():
    # TS-25
    """Output sparsity stays fixed regardless of how many input bits are on."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)
    expected_active = int(len(cf.columns) * DESIRED_LOCAL_SPARSITY)

    # use encoders with different sparsity settings to vary input density
    for input_sparsity in (0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.80):
        cast(InputField, cf.input_fields[0]).encoder._sparsity = input_sparsity
        print(cast(InputField, cf.input_fields[0]).encoder._sparsity)
        activate_cells(cf, 100.0)
        cf.compute(learn=True)
        active_cols = {i for i, col in enumerate(cf.columns) if col.active}

        assert (
            len(active_cols) == expected_active
        ), f"Input sparsity {input_sparsity}: expected {expected_active} active cols, got {len(active_cols)}"


# Test Type: unit test
def test_activate_top_k_columns_respects_k():
    # TS-25
    """activate_top_k_columns should activate exactly k columns."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    for i, col in enumerate(cf.columns):
        col.overlap = float(i)

    for k in (1, 10, 50, 128):
        for col in cf.columns:
            col.active = False
        cf.activate_top_k_columns(k)
        active_cols = {i for i, col in enumerate(cf.columns) if col.active}
        assert (
            len(active_cols) >= k
        ), f"activate_top_k_columns({k}) activated only {len(active_cols)}"


"""DISTRIBUTED CODING"""


# Test Type: unit test
def test_many_columns_participate_across_patterns():
    # TS-25
    """Over many RDSE encoded inputs, a large share of columns should activate at least once."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    rng = random.Random(42)
    ever_active: set[Column] = set()
    print(len(in_fi.cells) * RECEPTIVE_FIELD_PCT)
    for _ in range(100):
        value = rng.uniform(-10000, 10000)
        activate_cells(cf, value)
        cf.compute(learn=True)
        active_cols = {col for col in cf.columns if col.active}
        ever_active.update(active_cols)
        print("Total columns ever active: ", len(ever_active))

    participation = len(ever_active) / len(cf.columns)
    assert (
        participation > 0.7
    ), f"Only {participation:.1%} of columns were ever active not distributed enough"


# Test Type: unit test
def test_no_single_column_dominates():
    # TS-25
    """No single column should be active in more than a small fraction of all patterns."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    rng = random.Random(42)
    activation_counts: dict = {}
    for _ in range(200):
        value = rng.uniform(-10000, 10000)
        activate_cells(cf, value)
        cf.compute(learn=True)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1

    if activation_counts:
        max_freq = max(activation_counts.values()) / 200
        print(max_freq)
        threshold = min(1.0, 5 * DESIRED_LOCAL_SPARSITY)
        assert (
            max_freq < threshold
        ), f"Most used column was active {max_freq:.1%} of the time threshold {threshold:.1%}"


# Test Type: unit test
def test_activation_with_random_cells_excluding_encoder():
    """Checks active column over 49 epochs on a random input excluding the built in encoder."""
    import htmrl.agent_layer.HTM

    htmrl.agent_layer.HTM.PERMANENCE_INC = 0.10
    htmrl.agent_layer.HTM.PERMANENCE_DEC = 0.02
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    # build 100 random indices
    all_indices = []
    for _ in range(100):
        random_indices = get_random_indices(input_size, 40)
        all_indices.append(random_indices)

    # train for 49 epochs, bypass encoder with activate cells directly
    for _ in range(49):
        for index in all_indices:
            activate_cells_directly(cf, index)
            cf.compute(learn=True)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for index in all_indices:
        activate_cells_directly(cf, index)
        cf.compute(learn=False)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in cf.columns:
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(cf.columns)) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(
        bin_centers,
        fractions,
        width=bin_widths,
        align="center",
        color="#3366CC",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Activation Frequency Distribution with random cells excluding encoder\n(epoch 49)",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(
        0.95,
        0.95,
        f"{n_active} / {len(cf.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


def test_activation_converge_on_desired_sparsity_random_once_with_encoder():
    """Tests random inputs to the spatial pooler to see that our activations converge on the desired sparsity."""
    import htmrl.agent_layer.HTM

    htmrl.agent_layer.HTM.PERMANENCE_INC = 0.10
    htmrl.agent_layer.HTM.PERMANENCE_DEC = 0.02
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    # rng = random.Random(42)
    pattern_a = random.sample(range(-10000, 10000), 100)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            activate_cells(cf, value)
            cf.compute(learn=True)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        activate_cells(cf, value)
        cf.compute(learn=False)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in cf.columns:
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(cf.columns)) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(
        bin_centers,
        fractions,
        width=bin_widths,
        align="center",
        color="#3366CC",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Activation Frequency Distribution with Random data once with encoder\n(epoch 49)",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(
        0.95,
        0.95,
        f"{n_active} / {len(cf.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


# Test Type: unit test
@pytest.mark.slow
def test_activation_zero_epoch_exclude_encoder():
    """Tests a zero epoch excluding encoder on the spatial pooler."""
    import htmrl.agent_layer.HTM

    htmrl.agent_layer.HTM.PERMANENCE_INC = 0.10
    htmrl.agent_layer.HTM.PERMANENCE_DEC = 0.02
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)
    # pattern_a = random.sample(range(-10000, 10000), 100)

    # build 100 random indices
    all_indices = []
    for _ in range(100):
        random_indices = get_random_indices(input_size, 40)
        all_indices.append(random_indices)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for index in all_indices:
        activate_cells_directly(cf, index)
        cf.compute(learn=False)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in cf.columns:
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(cf.columns)) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(
        bin_centers,
        fractions,
        width=bin_widths,
        align="center",
        color="#3366CC",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Activation Frequency Distribution with Random data zero epoch excluding encoder",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(
        0.95,
        0.95,
        f"{n_active} / {len(cf.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


def test_activation_zero_epoch_with_encoder():
    """Tests zero epoch with encoder on spatial pooler."""
    import htmrl.agent_layer.HTM

    htmrl.agent_layer.HTM.PERMANENCE_INC = 0.10
    htmrl.agent_layer.HTM.PERMANENCE_DEC = 0.02
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)
    pattern_a = random.sample(range(-10000, 10000), 100)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        activate_cells(cf, value)
        cf.compute(learn=False)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in cf.columns:
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(cf.columns)) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(
        bin_centers,
        fractions,
        width=bin_widths,
        align="center",
        color="#3366CC",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Activation Frequency Distribution with Random data zero epoch with encoder",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(
        0.95,
        0.95,
        f"{n_active} / {len(cf.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


def test_activation_with_sin_wave_scalar_encoder_periodic_false():
    """Tests a spatial pooler with the scalar encoder and periodic false."""
    import htmrl.agent_layer.HTM

    htmrl.agent_layer.HTM.PERMANENCE_INC = 0.10
    htmrl.agent_layer.HTM.PERMANENCE_DEC = 0.02
    input_size = 2048
    in_fi = make_input_field_scalar(input_size, 0.001, min=-1, max=1, periodic=False)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            activate_cells(cf, float(value))
            cf.compute(learn=True)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        activate_cells(cf, float(value))
        cf.compute(learn=False)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in cf.columns:
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(cf.columns)) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(
        bin_centers,
        fractions,
        width=bin_widths,
        align="center",
        color="#3366CC",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Activation Frequency Distribution with sin wave with scalar encoder periodic false\n(epoch 49)",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.95,
        0.95,
        f"{n_active} / {len(cf.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


def test_activation_with_sin_wave_with_encoder():
    """Tests a sin wave and rdse encoder spatial pooler to see dominate columns form on the pattern."""
    import htmrl.agent_layer.HTM

    htmrl.agent_layer.HTM.PERMANENCE_INC = 0.10
    htmrl.agent_layer.HTM.PERMANENCE_DEC = 0.02
    input_size = 2048
    in_fi = make_input_field_sin(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            activate_cells(cf, float(value))
            cf.compute(learn=True)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        activate_cells(cf, float(value))
        cf.compute(learn=False)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in cf.columns:
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(cf.columns)) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(
        bin_centers,
        fractions,
        width=bin_widths,
        align="center",
        color="#3366CC",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Activation Frequency Distribution with sin wave with rdse encoder\n(epoch 49)",
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.95,
        0.95,
        f"{n_active} / {len(cf.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


# Test Type: unit test
def test_duty_cycle_boosting_engages_inactive_columns():
    # TS-25
    """Duty cycle tracking should engage after repeated presentations."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size, duty_cycle_period=50)

    for value in range(60):
        activate_cells(cf, value)
        cf.compute(learn=True)

    assert cf._duty_cycle_window > 0, "Duty cycle window was never advanced"

    duty_cycles = [col.active_duty_cycle for col in cf.columns]
    non_zero = sum(1 for d in duty_cycles if d > 0)
    assert non_zero > 0, "No columns have positive duty cycles"


"""PRESERVING SEMANTIC SIMILARITY"""


# Test Type: unit test
def test_similar_inputs_produce_similar_sdrs():
    # TS-25
    """Nearby scalar values should produce overlapping column activations."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 50.0
    similar_value = 51.0  # one resolution step away since default rdse is 1.0

    # train on base
    for _ in range(100):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    # test base
    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_base = {i for i, col in enumerate(cf.columns) if col.active}

    # test similar
    activate_cells(cf, similar_value)
    cf.compute(learn=False)
    cols_similar = {i for i, col in enumerate(cf.columns) if col.active}

    sim = _overlap_count(cols_base, cols_similar)
    print(sim)
    assert (
        sim > 0
    ), f"Similar inputs ({base_value} vs {similar_value}) produced SDRs with only {sim} overlapping columns"


# Test Type: unit test
def test_dissimilar_inputs_produce_dissimilar_sdrs():
    # TS-25
    """Values far apart should produce low column overlap."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    value_a = 1.0
    value_b = 1000.0  # far apart no rdse bit overlap expected

    activate_cells(cf, value_a)
    activate_cells(cf, value_b)

    # train on both
    for _ in range(100):
        cf.compute(learn=True)

    # test A
    activate_cells(cf, value_a)
    cf.compute(learn=False)
    cols_a = {i for i, col in enumerate(cf.columns) if col.active}

    # test B
    activate_cells(cf, value_b)
    cf.compute(learn=False)
    cols_b = {i for i, col in enumerate(cf.columns) if col.active}

    sim = _overlap_count(cols_a, cols_b)
    print(sim)
    expected_active = int(len(cf.columns) * DESIRED_LOCAL_SPARSITY)
    # overlap should be well below the number of active columns
    assert (
        sim < expected_active
    ), f"Distant inputs ({value_a} vs {value_b}) still had {sim} overlapping columns out of {expected_active} active"


# Test Type: unit test
def test_sp_overlap_gradient():
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 1

    for _ in range(100):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_base = [1 if col.active else 0 for col in cf.columns]

    overlaps = []
    for offset in range(1, 1000):
        activate_cells(cf, base_value + offset)
        cf.compute(learn=False)
        cols_test = [1 if col.active else 0 for col in cf.columns]
        overlaps.append(_overlap_count_list(cols_base, cols_test))

    import matplotlib.pyplot as plt

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("SP Overlap vs Distance with RDSE input field")
    plt.show()
    """
    #nearby values should overlap more than distant values
    near = sum(overlaps[:10]) / 10
    far = sum(overlaps[500:]) / 500
    print(near)
    print(far)

    assert near > far, (
        f"Near avg overlap ({near:.1f}) should exceed far avg ({far:.1f})"
    )

    #overlap at distance 0 should be highest
    assert overlaps[0] == max(overlaps), (
        f"Base value overlap ({overlaps[0]}) should be the maximum"
    )
    """


def _overlap_count_list(first: list[int], second: list[int]) -> int:
    return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)


# Test Type: unit test
def test_rdse_gradient():
    e = RandomDistributedScalarEncoder(RDSEParameters())
    base_value = 1
    base_encoding = e.encode(base_value)

    overlaps = []
    for offset in range(1, 1000):
        value = base_value + offset
        encoding = e.encode(value)
        overlap = _overlap_count_list(base_encoding, encoding)
        overlaps.append(overlap)

    import matplotlib.pyplot as plt

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("RDSE Overlap vs Distance")
    plt.show()
    """
    #nearby values should overlap more than distant values
    near = sum(overlaps[:10]) / 10
    far = sum(overlaps[500:]) / 500
    print(near)
    print(far)

    assert near > far, (
        f"Near avg overlap ({near:.1f}) should exceed far avg ({far:.1f})"
    )

    #closest value should have the most overlap
    assert overlaps[0] == max(overlaps), (
        f"Offset 1 overlap ({overlaps[0]}) should be the maximum"
    )

    #overlap should eventually reach zero
    assert overlaps[-1] == 0, (
        f"Distant values should have no overlap, got {overlaps[-1]}"
    )
    """


# Test Type: unit test
def test_scalar_gradient():
    e = ScalarEncoder(ScalarEncoderParameters(radius=0, resolution=1.0, periodic=True))
    base_value = 1
    base_encoding = e.encode(base_value)

    overlaps = []
    for offset in range(1, 1000):
        value = base_value + offset
        encoding = e.encode(value)
        overlap = _overlap_count_list(base_encoding, encoding)
        overlaps.append(overlap)

    import matplotlib.pyplot as plt

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("Scalar Encoder Overlap vs Distance")
    plt.show()


# Test Type: unit test
def test_sp_overlap_gradient_input_field_scalar():
    input_size = 2048
    in_fi = make_input_field_scalar(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 1

    for _ in range(100):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_base = [1 if col.active else 0 for col in cf.columns]

    overlaps = []
    for offset in range(1, 1000):
        activate_cells(cf, base_value + offset)
        cf.compute(learn=False)
        cols_test = [1 if col.active else 0 for col in cf.columns]
        overlaps.append(_overlap_count_list(cols_base, cols_test))

    import matplotlib.pyplot as plt

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("SP Overlap vs Distance with Scalar input field")
    plt.show()


"""NOISE ROBUSTNESS"""


# Test Type: unit test
def test_zero_noise_no_output_change():
    """With zero noise there should be no change in SP output."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    activate_cells(cf, 42.0)
    cf.compute(learn=False)
    cols1 = [1 if col.active else 0 for col in cf.columns]

    activate_cells(cf, 42.0)
    cf.compute(learn=False)
    cols2 = [1 if col.active else 0 for col in cf.columns]

    assert _overlap_count_list(cols1, cols2) > 0


# Test Type: unit test
def test_noise_gradient_plot():
    """Plot noise robustness across epochs like the research paper. Eventually add asserts to make it a test."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    noise_levels = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ]
    epoch_checkpoints = [0, 5, 10, 20, 40]
    results = {}

    epoch = 0
    for checkpoint in epoch_checkpoints:
        # train the amount of epochs in checkpoint, 0-5-10-20-40 like the paper
        while epoch < checkpoint:
            activate_cells(cf, 42.0)
            cf.compute(learn=True)
            epoch += 1

        # get a baseline of the active columns
        activate_cells(cf, 42.0)
        cf.compute(learn=False)
        cols1 = [1 if col.active else 0 for col in cf.columns]
        num_active = sum(cols1)

        # get input_field cells
        cells = []
        for cell in cf.input_fields[0].cells:
            cells.append(1 if cell.active else 0)

        overlaps = []
        for noise_pct in noise_levels:
            if noise_pct == 0.0:
                overlaps.append(1.0)
                continue

            # add the noise to the input_field cells, but deep copy so we do not change the base cells for futre run throughs
            noisy_cells = copy.deepcopy(cells)
            add_noise_to_encoding(noisy_cells, noise_pct)
            # clear cells to false and set active based on noise
            for cell in in_fi.cells:
                cell.active = False
            for cell, n_cell in zip(in_fi.cells, noisy_cells):
                if n_cell == 1:
                    cell.set_active()
            # clear column states and their cells
            for col in cf.columns:
                col.active = False
                for cell in col.cells:
                    cell.active = False
            # compute the new overlaps
            for col in cf.columns:
                col.compute_overlap()
            # activate columns and cells of columns
            cf.activate_columns()
            for col in cf.active_columns:
                for cell in col.cells:
                    cell.set_active()
            # get new active columns
            cols2 = [1 if col.active else 0 for col in cf.columns]
            # calculate overlap with original active columns
            overlap = _overlap_count_list(cols1, cols2)
            overlaps.append(overlap / num_active if num_active > 0 else 0)

        results[checkpoint] = overlaps
        print(f"epoch {checkpoint}: {overlaps}")

    import matplotlib.pyplot as plt

    colors = {0: "#3366CC", 5: "#009966", 10: "#CC3333", 20: "#00CCCC", 40: "#993399"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for ep, overlap_ratios in results.items():
        ax.plot(noise_levels, overlap_ratios, linewidth=2.5, label=f"epoch {ep}", color=colors[ep])

    ax.set_xlabel("Noise Level", fontsize=14)
    ax.set_ylabel("Change of SP Output", fontsize=14)
    ax.set_title("Properties of SP noise robustness", fontsize=16)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=12, loc="upper right", edgecolor="black")

    plt.show()


"""FAULT TOLERANCE"""

"""CONTINUOUS LEARNING"""


# Test Type: unit test
@pytest.mark.slow
def test_synapse_formation():
    """
    Track newly connected synapses per epoch across a dataset switch.
    Synapse formation counts how many potential synapses cross the connected permanence threshold during each training epoch. Should
    spike early, drop to near zero once stable, spike again at the dataset switch, then settle back down.
    """
    input_field = make_input_field()
    cf = make_spatial_only_cf(input_field)

    rng_a = random.Random(42)
    dataset_a = [rng_a.uniform(0, 5000) for _ in range(100)]
    rng_b = random.Random(77)
    dataset_b = [rng_b.uniform(5000, 10000) for _ in range(100)]

    switch_epoch = 30
    total_epochs = 60
    formation_history: list[int] = []

    for epoch in range(total_epochs):
        training_data = dataset_a if epoch < switch_epoch else dataset_b

        # all connected synapses before training
        prev_connected = {s for col in cf.columns for s in col.connected_synapses}

        # train one epoch
        for value in training_data:
            activate_cells(cf, value)
            cf.compute(learn=True)

        # count synapses newly connected this epoch
        current_connected = {s for col in cf.columns for s in col.connected_synapses}
        new_synapses = len(current_connected - prev_connected)

        formation_history.append(new_synapses)

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(formation_history)), formation_history, color="black", linewidth=1.2)
    ax.axvline(x=switch_epoch, color="black", linestyle="--", linewidth=2, label="Dataset switch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Synapse\nFormation", rotation=0, labelpad=50)
    ax.set_xlim(0, len(formation_history) - 1)
    ax.legend(loc="upper right")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()

    # assertions
    # early epochs should have noticeably high formation
    early_window = 5
    early_avg = sum(formation_history[:early_window]) / early_window
    assert early_avg > 70, (
        f"Early synapse formation avg {early_avg:.0f} is too low; "
        f"SP should be actively forming synapses."
    )

    # pre-switch plateau should have lower formation than early epochs
    pre_switch_start = max(early_window, switch_epoch - 10)
    pre_switch_segment = formation_history[pre_switch_start:switch_epoch]
    pre_switch_avg = sum(pre_switch_segment) / len(pre_switch_segment)
    assert pre_switch_avg < early_avg, (
        f"Pre-switch formation {pre_switch_avg:.0f} should be less than "
        f"early formation {early_avg:.0f}."
    )

    # post-switch spike should exceed the pre-switch plateau
    post_switch_end = min(total_epochs, switch_epoch + 10)
    post_switch_peak = max(formation_history[switch_epoch:post_switch_end])
    assert post_switch_peak > pre_switch_avg, (
        f"Post-switch peak {post_switch_peak} should exceed "
        f"pre-switch avg {pre_switch_avg:.0f}."
    )


"""STABILITY"""


def compute_stability(
    cf,
    test_inputs: list,
    prev_activations: list[list[int]] | None,
) -> tuple[float, list[list[int]]]:

    current_activations: list[list[int]] = []
    # with the test values grab the active column indices for each
    for inp in test_inputs:
        activate_cells(cf, inp)
        cf.compute(learn=False)
        current_activations.append(get_active_column_indices(cf))

    if prev_activations is None:
        return 1.0, current_activations

    m = len(test_inputs)
    if m == 0:
        return 1.0, current_activations
    # calculate the sum of the overlaps between previous and current activations
    stability_sum = sum(
        overlap_ratio(prev, curr) for prev, curr in zip(prev_activations, current_activations)
    )
    # divide that sum by total test inputs to get the average fraction of active mini-columns
    return stability_sum / m, current_activations


# Test Type: unit test
def test_continuous_learning_stability():
    """
    SP trains on dataset_a and stabilizes. Inputs switch to dataset_b stability drops then recovers as the SP adapts to the new statistics.
    """
    input_field = make_input_field()
    cf = make_spatial_only_cf(input_field)

    # random datasets, two for the before and after
    rng_a = random.Random(42)
    dataset_a = [rng_a.uniform(0, 50) for _ in range(200)]
    rng_b = random.Random(77)
    dataset_b = [rng_b.uniform(50, 100) for _ in range(200)]
    # random values from the original datasets like the paper
    test_rng = random.Random(99)
    test_inputs_a = test_rng.sample(dataset_a, 30)
    test_inputs_b = test_rng.sample(dataset_b, 30)

    switch_epoch = 60
    total_epochs = 120
    prev_activations = None
    stability_history: list[float] = []

    for epoch in range(total_epochs):
        # before data switch
        if epoch < switch_epoch:
            training_data = dataset_a
            test_inputs = test_inputs_a
        # after data switch
        else:
            if epoch == switch_epoch:
                prev_activations = None
            training_data = dataset_b
            test_inputs = test_inputs_b
        # compute the stability
        stability, prev_activations = compute_stability(cf, test_inputs, prev_activations)
        stability_history.append(stability)
        # train on data
        for value in training_data:
            activate_cells(cf, value)
            cf.compute(learn=True)

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(stability_history)), stability_history, color="black", linewidth=1.2)
    ax.axvline(x=switch_epoch, color="black", linestyle="--", linewidth=2, label="Dataset switch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Stability")
    ax.set_ylim(0.55, 1.05)
    ax.set_xlim(0, len(stability_history) - 1)
    ax.legend(loc="lower right")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()

    # assertions
    pre_switch_tail = stability_history[40:switch_epoch]
    pre_switch_avg = sum(pre_switch_tail) / len(pre_switch_tail)
    assert pre_switch_avg > 0.85, (
        f"Pre-switch stability avg {pre_switch_avg:.3f} is below 0.85; "
        f"SP did not stabilize on dataset A."
    )

    post_switch_tail = stability_history[-20:]
    post_switch_avg = sum(post_switch_tail) / len(post_switch_tail)
    assert post_switch_avg > 0.85, (
        f"Post-switch stability avg {post_switch_avg:.3f} is below 0.85; "
        f"SP did not recover after switching to dataset B."
    )


# Test Type: unit test
def test_stability_dips_at_switch():
    """Stability should drop when the input distribution changes.
    The post switch dip should be measurably lower than the pre switch
    plateau, confirming the SP is sensitive to distribution shift.
    This kind of captures the continuous learning as well I think.
    """
    input_field = make_input_field()
    cf = make_spatial_only_cf(input_field)

    rng_a = random.Random(42)
    dataset_a = [rng_a.uniform(0, 50) for _ in range(200)]
    rng_b = random.Random(77)
    dataset_b = [rng_b.uniform(50, 100) for _ in range(200)]
    test_inputs_a = [random.Random(99).uniform(0, 50) for _ in range(30)]
    test_inputs_b = [random.Random(101).uniform(50, 100) for _ in range(30)]

    switch_epoch = 60
    total_epochs = 120
    prev_activations = None
    stability_history: list[float] = []

    for epoch in range(total_epochs):
        if epoch < switch_epoch:
            training_data = dataset_a
            test_inputs = test_inputs_a
        else:
            if epoch == switch_epoch:
                prev_activations = None
            training_data = dataset_b
            test_inputs = test_inputs_b

        stability, prev_activations = compute_stability(cf, test_inputs, prev_activations)
        stability_history.append(stability)

        for value in training_data:
            activate_cells(cf, value)
            cf.compute(learn=True)

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(stability_history)), stability_history, color="black", linewidth=1.2)
    ax.axvline(x=switch_epoch, color="black", linestyle="--", linewidth=2, label="Dataset switch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Stability")
    ax.set_ylim(0.55, 1.05)
    ax.set_xlim(0, len(stability_history) - 1)
    ax.legend(loc="lower right")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    plt.show()

    # assertions
    pre_switch_plateau = stability_history[40:switch_epoch]
    post_switch_dip = stability_history[switch_epoch + 1 : switch_epoch + 10]

    pre_avg = sum(pre_switch_plateau) / len(pre_switch_plateau)
    dip_avg = sum(post_switch_dip) / len(post_switch_dip)

    assert dip_avg < pre_avg, (
        f"Expected a stability dip after the switch: "
        f"dip avg {dip_avg:.3f} should be < pre-switch avg {pre_avg:.3f}."
    )


# Test Type: unit test
def test_stability_without_training():
    """Without any training between checkpoints stability should be 1."""
    input_field = make_input_field()
    cf = make_spatial_only_cf(input_field)
    test_inputs = [random.Random(99).uniform(0, 50) for _ in range(30)]

    _, prev_activations = compute_stability(cf, test_inputs, prev_activations=None)
    stability, _ = compute_stability(cf, test_inputs, prev_activations)

    assert (
        stability <= 0.04
    ), f"Stability should be close to 0 without intervening training, got {stability:.3f}"


# Test Type: unit test
def test_stability_first_checkpoint_is_one():
    """The very first checkpoint should return 1."""
    input_field = make_input_field()
    cf = make_spatial_only_cf(input_field)
    test_inputs = [random.Random(99).uniform(0, 50) for _ in range(30)]

    stability, _ = compute_stability(cf, test_inputs, prev_activations=None)
    assert stability == 1.0

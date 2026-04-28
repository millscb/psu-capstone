import random

import numpy as np
import pytest

from htmrl.agent_layer.our_htm.column import Column
from htmrl.agent_layer.our_htm.spatial_pooler import SpatialPooler
from htmrl.encoder_layer import RandomDistributedScalarEncoder, RDSEParameters


def make_spatial_pooler(
    input_size=2048,
    potential_radius=None,
    potential_pct=0.75,
    global_inhibition=True,
    stimulus_threshold=0,
):
    return SpatialPooler(
        input_size=2048,
        potential_radius=potential_radius,
        potential_pct=potential_pct,
        global_inhibition=global_inhibition,
        stimulus_threshold=stimulus_threshold,
    )


def make_rdse():
    return RandomDistributedScalarEncoder(RDSEParameters())


# Test Type: unit test
def test_active_ratio_matches_desired_sparsity():
    """After compute, the number of active columns should match desired sparsity."""
    input_size = 2048
    desired_sparsity = 0.02
    e = make_rdse()
    sp = make_spatial_pooler(input_size=input_size)

    input_vector = e.encode(42.0)
    active_cols = sp.compute(input_vector, learn=True)
    print(len(active_cols))
    expected_active = int(len(sp.columns) * desired_sparsity)
    print(expected_active)
    assert (
        len(active_cols) == expected_active
    ), f"Expected exactly {expected_active} active columns, got {len(active_cols)}"


# Test Type: unit test
def test_sparsity_invariant_to_input_density():
    """Output sparsity stays fixed regardless of how many input bits are on."""
    input_size = 2048
    desired_sparsity = 0.02
    e = make_rdse()
    sp = make_spatial_pooler(input_size=input_size)

    expected_active = int(len(sp.columns) * desired_sparsity)

    # use encoders with different sparsity settings to vary input density
    for input_sparsity in (0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.80):
        e._sparsity = input_sparsity
        print(e._sparsity)
        input_vector = e.encode(100.0)
        active_cols = sp.compute(input_vector, learn=True)

        assert (
            len(active_cols) == expected_active
        ), f"Input sparsity {input_sparsity}: expected {expected_active} active cols, got {len(active_cols)}"


"""DISTRIBUTED CODING"""


# Test Type: unit test
def test_many_columns_participate_across_patterns():
    """Over many RDSE encoded inputs, a large share of columns should activate at least once."""
    input_size = 2048
    e = make_rdse()
    sp = make_spatial_pooler(input_size=input_size)

    rng = random.Random(42)
    ever_active: set[Column] = set()
    for _ in range(100):
        value = rng.uniform(-10000, 10000)
        input_vector = e.encode(value)
        active_cols = sp.compute(input_vector, learn=True)
        ever_active.update(active_cols)
        print("Total columns ever active: ", len(ever_active))

    participation = len(ever_active) / len(sp.columns)
    assert (
        participation > 0.7
    ), f"Only {participation:.1%} of columns were ever active not distributed enough"


# Test Type: unit test
def test_no_single_column_dominates():
    """No single column should be active in more than a small fraction of all patterns."""
    input_size = 2048
    desired_sparsity = 0.02
    e = make_rdse()
    sp = make_spatial_pooler(input_size=input_size)

    rng = random.Random(42)
    activation_counts: dict = {}
    for _ in range(49):
        value = rng.uniform(-10000, 10000)
        input_vector = e.encode(value)
        active_cols = sp.compute(input_vector, learn=True)
        for col in active_cols:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    if activation_counts:
        max_freq = max(activation_counts.values()) / 49
        print(max_freq)
        threshold = min(1.0, 5 * desired_sparsity)
        assert (
            max_freq < threshold
        ), f"Most used column was active {max_freq:.1%} of the time threshold {threshold:.1%}"


# Test Type: unit test
def test_activation_converge_on_desired_sparsity_random_once():
    input_size = 2048
    e = make_rdse()
    sp = make_spatial_pooler(
        input_size=input_size,
        potential_radius=None,
        potential_pct=0.10,
        global_inhibition=True,
        stimulus_threshold=1,
    )
    sp.syn_perm_active_inc = 0.10
    sp.syn_perm_inactive_dec = 0.02

    pattern_a = random.sample(range(-10000, 10000), 100)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            input_vector = e.encode(value)
            sp.compute(input_vector, learn=True)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        input_vector = e.encode(value)
        active_cols = sp.compute(input_vector, learn=False)
        for col in active_cols:
            activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in sp.columns:
        freqs.append(activation_counts.get(col.index, 0) / len(pattern_a))
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(sp.columns)) * len(pattern_a)
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
        f"{n_active} / {len(sp.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


# Test Type: unit test
def test_activation_converge_on_desired_sparsity_with_sin_wave():
    input_size = 2048
    e = make_rdse()
    sp = make_spatial_pooler(
        input_size=input_size,
        potential_radius=None,
        potential_pct=0.20,
        global_inhibition=True,
        stimulus_threshold=1,
    )
    sp.syn_perm_active_inc = 0.10
    sp.syn_perm_inactive_dec = 0.02

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            input_vector = e.encode(float(value))
            sp.compute(input_vector, learn=True)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        input_vector = e.encode(float(value))
        active_cols = sp.compute(input_vector, learn=False)
        for col in active_cols:
            activation_counts[col] = activation_counts.get(col, 0) + 1
    freqs = []
    for col in sp.columns:
        freqs.append(activation_counts.get(col.index, 0) / len(pattern_a))
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / len(sp.columns)) * 100
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
        "Activation Frequency Distribution with sin wave data once with encoder\n(epoch 49)",
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
        f"{n_active} / {len(sp.columns)} cols active ({pct_active:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout()
    plt.show()


"""PRESERVING SEMANTIC SIMILARITY"""


def _overlap_count(first, second) -> int:
    """Count overlapping active column indices between two sets."""
    return len(first & second)


# Test Type: unit test
def test_similar_inputs_produce_similar_sdrs():
    """Nearby scalar values should produce overlapping column activations."""
    input_size = 2048
    e = make_rdse()
    sp = make_spatial_pooler(
        input_size=input_size,
        potential_radius=None,
        potential_pct=0.20,
        global_inhibition=True,
        stimulus_threshold=1,
    )

    base_value = 50.0
    similar_value = 51.0  # one resolution step away since default rdse is 1.0

    # train on base
    for _ in range(100):
        input_vector = e.encode(base_value)
        sp.compute(input_vector, learn=True)

    # test base
    input_vector = e.encode(base_value)
    cols_base = sp.compute(input_vector, learn=False)

    # test similar
    input_vector = e.encode(similar_value)
    cols_similar = sp.compute(input_vector, learn=False)

    sim = _overlap_count(set(cols_base), set(cols_similar))
    print(sim)
    assert (
        sim > 0
    ), f"Similar inputs ({base_value} vs {similar_value}) produced SDRs with only {sim} overlapping columns"


# Test Type: unit test
def test_dissimilar_inputs_produce_dissimilar_sdrs():
    """Values far apart should produce low column overlap."""
    input_size = 2048
    desired_sparsity = 0.02
    e = make_rdse()
    sp = make_spatial_pooler(
        input_size=input_size,
        potential_radius=None,
        potential_pct=0.20,
        global_inhibition=True,
        stimulus_threshold=1,
    )

    value_a = 1.0
    value_b = 1000.0  # far apart no rdse bit overlap expected

    sdr_a = e.encode(value_a)
    sdr_b = e.encode(value_b)

    # train on both
    for _ in range(100):
        sp.compute(sdr_a, learn=True)
        sp.compute(sdr_b, learn=True)

    # test A
    cols_a = sp.compute(sdr_a, learn=False)

    # test B
    cols_b = sp.compute(sdr_b, learn=False)

    sim = _overlap_count(set(cols_a), set(cols_b))
    print(sim)
    expected_active = int(len(sp.columns) * desired_sparsity)
    # overlap should be well below the number of active columns
    assert (
        sim < expected_active
    ), f"Distant inputs ({value_a} vs {value_b}) still had {sim} overlapping columns out of {expected_active} active"


"""NOISE ROBUSTNESS"""


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


# Test Type: unit test
def test_zero_noise_no_output_change():
    """With zero noise there should be no change in SP output."""
    input_size = 2048
    e = make_rdse()
    sp = make_spatial_pooler(
        input_size=input_size,
        potential_radius=None,
        potential_pct=0.20,
        global_inhibition=True,
        stimulus_threshold=1,
    )

    input_vector = e.encode(42.0)

    cols1 = sp.compute(input_vector, learn=False)

    cols2 = sp.compute(input_vector, learn=False)

    assert _overlap_count(set(cols1), set(cols2)) == 40

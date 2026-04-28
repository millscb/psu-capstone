"""Visualization utilities for SDRs and encoder analysis.

This module provides plotting functions for visualizing sparse distributed
representations, FFT analysis of time-series data, and encoder behaviors.
Includes tools for comparing encodings, analyzing frequency spectra, and
displaying SDR patterns as 2D grids.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import ListedColormap, PowerNorm
from scipy.fft import fft, fftfreq

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.input_layer.input_handler import InputHandler
from htmrl.log import logger
from htmrl.utils import DATA_PATH, PROJECT_ROOT
from legacy.sdr_layer.sdr import SDR

plt.style.use("seaborn-v0_8-poster")


def show_active_columns(brain: Any, dataset_name: str | None = None) -> None:
    """Visualize active columns for each column field in a brain."""

    for column_field in brain._column_fields.values():
        sdr = [
            (1 if column in column_field.active_columns else 0) for column in column_field.columns
        ]

        num_active = sum(sdr)
        sparsity = (num_active / len(sdr)) * 100 if sdr else 0
        dataset_info = f" - {dataset_name}" if dataset_name else ""
        plot_sdr(
            sdr,
            title=(
                f"Active Columns: {column_field.name}{dataset_info}\n"
                f"({num_active}/{len(sdr)} active, {sparsity:.1f}% sparsity)"
            ),
        )


def show_heat_map(brain: Any, dataset_name: str | None = None) -> None:
    """Visualize column duty-cycle activity as a heat map."""

    if not brain._column_fields:
        raise ValueError("No column fields available to visualize.")

    column_field = brain.column_fields[0]
    duty_cycles = np.array([column.active_duty_cycle for column in column_field.columns])

    if duty_cycles.size == 0:
        raise ValueError("Column field has no columns to visualize.")

    side = int(np.ceil(np.sqrt(duty_cycles.size)))
    heat_map = np.zeros((side, side))
    heat_map.flat[: duty_cycles.size] = duty_cycles

    positive = duty_cycles[duty_cycles > 0]
    active_columns = len(positive)
    max_duty = float(positive.max()) if positive.size > 0 else 0.0

    dataset_info = f" - {dataset_name}" if dataset_name else ""
    title = (
        f"Column Duty Cycle Heat Map: {column_field.name}{dataset_info}\n"
        f"({active_columns}/{len(duty_cycles)} active columns, max duty={max_duty:.3f})"
    )

    if positive.size > 0:
        vmax = max(max_duty, 1e-6)
        norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=vmax)
        vmin = None
        vmax_for_plot = None
    else:
        norm = None
        vmin = 0.0
        vmax_for_plot = 1.0

    plt.figure(figsize=(12, 12))
    plt.imshow(
        heat_map,
        cmap="hot",
        interpolation="nearest",
        norm=norm,
        vmin=vmin,
        vmax=vmax_for_plot,
    )
    plt.title(title)
    plt.colorbar(label="Duty Cycle")
    plt.xticks([])
    plt.yticks([])
    plt.show(block=True)


def plot_sdr(data: list[int], title: str | None = None) -> None:
    """Plot a visual representation of an SDR as a 2D grid.

    Converts the 1D binary SDR into a square grid visualization where
    active bits are shown in blue and inactive bits in white.

    Args:
        data: Binary list representing the SDR (0s and 1s).
        title: Optional title for the plot.
    """

    sdr = SDR([len(data)])
    sdr.set_dense(data)
    dense = np.array(sdr.get_dense(), dtype=int)

    # compute square grid size
    n = dense.size
    side = int(np.ceil(np.sqrt(n)))  # smallest square that fits SDR

    # pad if needed
    padded = np.zeros(side * side, dtype=int)
    padded[:n] = dense

    grid = padded.reshape(side, side)

    # colormap: white for 0, blue for 1
    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(12, 12))
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    plot_title = title or "SDR Visualization"
    plt.title(plot_title, fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)


def _sdr_to_grid(data: list[int]) -> np.ndarray:
    """Convert 1D SDR bits into a square 2D grid with zero-padding."""
    sdr = SDR([len(data)])
    sdr.set_dense(data)
    dense = np.array(sdr.get_dense(), dtype=int)

    n = dense.size
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side, dtype=int)
    padded[:n] = dense
    return padded.reshape(side, side)


def plot_sdr_sequence(items: list[tuple[list[int], str]]) -> None:
    """Interactive SDR gallery.

    Controls:
    - Right arrow / 'n': next SDR
    - Left arrow / 'p': previous SDR
    - Escape / 'q': close viewer
    """
    if not items:
        raise ValueError("items must contain at least one SDR plot entry.")

    cmap = ListedColormap(["white", "blue"])
    index = 0

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(_sdr_to_grid(items[index][0]), cmap=cmap, interpolation="nearest")
    ax.set_title(items[index][1], fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    def _render(i: int) -> None:
        nonlocal index
        index = i % len(items)
        im.set_data(_sdr_to_grid(items[index][0]))
        ax.set_title(items[index][1], fontsize=14)
        fig.canvas.draw_idle()

    def _on_key(event: Any) -> None:
        key = (event.key or "").lower()
        if key in {"right", "n"}:
            _render(index + 1)
        elif key in {"left", "p"}:
            _render(index - 1)
        elif key in {"escape", "q"}:
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show(block=True)


def plot_signal(
    signal: list[float] | np.ndarray | pd.Series,
    sample_rate: float = 1.0,
    domain: str = "both",
    title: str | None = None,
) -> None:
    """Plot a numeric signal in the time domain, frequency domain, or both.

    Args:
        signal: 1D numeric sequence to visualize.
        sample_rate: Sampling rate in Hz used for frequency axis scaling.
        domain: One of "time", "frequency", or "both".
        title: Optional label appended to plot titles.

    Raises:
        ValueError: If the signal is empty, ``sample_rate`` is not positive,
            or ``domain`` is not one of ``'time'``, ``'frequency'``, or ``'both'``.
    """

    values = np.asarray(signal, dtype=float).flatten()
    if values.size == 0:
        raise ValueError("Signal is empty. Provide at least one value.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")

    selected_domain = domain.lower()
    if selected_domain not in {"time", "frequency", "both"}:
        raise ValueError("domain must be one of: 'time', 'frequency', 'both'.")

    plot_label = f" - {title}" if title else ""

    if selected_domain in {"time", "both"}:
        time_axis = np.arange(values.size, dtype=float) / sample_rate
        plt.figure(figsize=(16, 8))
        plt.plot(time_axis, values, "r")
        plt.title(f"Signal in Time Domain{plot_label}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show(block=True)

    if selected_domain in {"frequency", "both"}:
        freq_data = fft(values)
        samples = len(freq_data)
        if samples < 2:
            raise ValueError("Signal must contain at least 2 samples for frequency plotting.")

        magnitudes = np.abs(freq_data[1 : samples // 2])
        freq_bin = fftfreq(samples, 1 / sample_rate)[1 : samples // 2]

        plt.figure(figsize=(16, 8))
        plt.plot(freq_bin, magnitudes)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.title(f"FFT Magnitude Spectrum{plot_label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(which="both", axis="both", linestyle="--", linewidth=0.8)
        plt.show(block=True)


def visualize_signal_fft(dataset: str, sample_rate: int) -> None:
    """Plot time-domain data and FFT magnitude spectrum for the specified dataset."""
    ih = InputHandler()

    signal = ih.input_data(os.path.join(PROJECT_ROOT, "data", dataset))

    columns: list[str] = []

    for k in signal.keys():
        if k.lower() == "timestamp":
            continue
        columns.append(k)

        print(f"Columns: {columns}")

    for column in columns:
        values = np.array(signal[column], dtype=float)
        values[0] = 0.0  # remove DC component by zeroing the first value
        values = values[:4096]

        print(f"Plotting column: {column}")

        plot_signal(values, sample_rate=sample_rate, domain="both", title=column)

        freq_data = fft(values)
        samples = len(freq_data)
        magnitudes = np.abs(freq_data[1 : samples // 2])
        freq_bin = fftfreq(samples, 1 / sample_rate)[1 : samples // 2]
        peak_index = np.argmax(magnitudes)
        peak_freq = freq_bin[peak_index]
        print(f"Plot Peak Frequency: {peak_freq} Hz")

        fft_encoder = FourierEncoder(FourierEncoderParameters())

        sdr = fft_encoder.encode(values)

        plot_sdr(sdr)


if __name__ == "__main__":

    fft_encoder = FourierEncoder(
        FourierEncoderParameters(
            resolutions_in_ranges=[1.0, 1.0],
            frequency_ranges=[(0, 100), (100, 500)],
            size=2048,
            # active bits in range times number of ranges
            sparsity_in_ranges=[0.02, 0.02],
            sensitivity_threshold=0.01,
        )
    )

    a, b, c, d, e, f = 10, 2, 30, 2, 50, 60
    y1 = np.sin(2 * np.pi * a * np.linspace(0, 1, 2048, endpoint=False))
    y1 *= np.sin(2 * np.pi * b * np.linspace(0, 1, 2048, endpoint=False))
    y2 = np.sin(2 * np.pi * d * np.linspace(0, 1, 2048, endpoint=False))

    """
    fft_one = fft_encoder.encode(y1)
    fft_two = fft_encoder.encode(y2)

    print(f"SDR One: {len(fft_one)}")
    print(f"SDR active bits One: {sum(fft_one)}")
    print(f"SDR Two: {len(fft_two)}")
    print(f"SDR active bits Two: {sum(fft_two)}")

    overlap_bits = overlap(fft_one, fft_two)
    hamming_dist = hamming_distance(fft_one, fft_two)
    print(f"Overlap: {overlap_bits} bits")
    print(f"Hamming Distance: {hamming_dist} bits")

    #plot_sdr(fft_one)
    #plot_sdr(fft_two)

    """
    visualize_signal_fft("fin_test.csv", sample_rate=4096)

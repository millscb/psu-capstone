"""
Driver script to demonstrate the capabilities of the Brain and Trainer classes.
Author: Chris Mills @millscb

Date: 2025-02-21

This script includes several demonstration functions that can be run to visualize and
understand different components of the Brain and Trainer classes.
Each function focuses on a specific aspect, such as data ingestion, encoding, brain
structure, and training on datasets.
The main function at the bottom can be modified to call any of these demonstration
functions as needed.

Note: Some of the demonstration functions may require specific datasets to be available
in the DATA_PATH directory. Ensure that the necessary datasets are in place before
running those demos.
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Import each layer once so the demos read like the architecture they exercise.
# This keeps example code focused on workflow rather than repeated module paths.
import htmrl.agent_layer as ag
import htmrl.encoder_layer as en
import htmrl.grapher as grapher
import htmrl.input_layer as il
from htmrl.log import logger
from htmrl.utils import DATA_PATH, PROJECT_ROOT, hamming_distance, overlap

ESD = os.path.join(DATA_PATH, "concat_ESData.xlsx")
REC_CENTER = os.path.join(DATA_PATH, "rec_center.csv")
DATA_COLUMN_LOG_MESSAGE = "Data column '%s': %d records"

DATA_DICT = {}


def build_demo_records(row_count: int = 32) -> list[dict[str, Any]]:
    """Build deterministic mixed-type records used by demo/integration tests."""

    records: list[dict[str, Any]] = []
    base_time = datetime(2025, 1, 1, 9, 0, 0)
    labels = ["A", "B", "C", "D"]

    for index in range(max(1, row_count)):
        records.append(
            {
                "float_col": float(index) * 0.5 + 1.0,
                "int_col": int((index * 3) % 100),
                "str_col": labels[index % len(labels)],
                "date_col": base_time + timedelta(hours=index),
            }
        )

    return records


def run_full_demo(row_limit: int = 32, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Run a compact, test-friendly demo and emit visual artifacts to disk."""

    import matplotlib.pyplot as plt
    import numpy as np

    records = build_demo_records(row_count=row_limit)
    floats = np.asarray([record["float_col"] for record in records], dtype=float)
    ints = np.asarray([record["int_col"] for record in records], dtype=float)

    destination = (
        Path(output_dir) if output_dir is not None else Path(PROJECT_ROOT) / "docs" / "reports"
    )
    destination.mkdir(parents=True, exist_ok=True)

    signal_plot = destination / "demo_signal.png"
    plt.figure(figsize=(8, 4))
    plt.plot(floats, label="float_col")
    plt.plot(ints, label="int_col", alpha=0.8)
    plt.title("Demo Input Signals")
    plt.xlabel("Record Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(signal_plot)
    plt.close()

    fft_plot = destination / "demo_signal_fft.png"
    magnitude = np.abs(np.fft.rfft(floats - floats.mean()))
    freqs = np.fft.rfftfreq(floats.size, d=1.0)
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, magnitude)
    plt.title("Demo Signal FFT Magnitude")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(fft_plot)
    plt.close()

    return {
        "signal": signal_plot,
        "fft": fft_plot,
    }


def show_input_data_demo() -> None:
    """Demonstrate the input handler can digest large datasets.
    Use Case Input Data --> Validate Data
    """

    ih = il.InputHandler()
    data = ih.input_data(ESD)

    for name, value in data.items():
        logger.info(DATA_COLUMN_LOG_MESSAGE, name, len(value))


def show_input_to_encoder_demo(s: int = 5) -> None:
    """Demonstrate the InputHandler's ability to convert raw input data to encoder-ready format."""

    ih = il.InputHandler()
    brain = ag.Brain()
    trainer = ag.Trainer(brain)

    columns: list[str] = []
    data = ih.input_data(ESD)

    for key in data.keys():
        logger.debug(DATA_COLUMN_LOG_MESSAGE, key, len(data[key]))

        columns.append(key)

    column_data = ih.get_column_data(column=columns[1])

    trainer.main_brain = trainer.build_brain(
        [(f"{columns[1]}_input", 2048, en.RDSEParameters(resolution=0.01))]
    )  # fin markets

    values = column_data[:s]  # Take the first 's' values for demonstration

    field = trainer.main_brain._input_fields[f"{columns[1]}_input"]
    encoder = field.encoder

    sdr = []

    for i, value in enumerate(values):
        encoded: list[int] = field.encode(value)
        decoded_value, confidence = field.decode("active", field, encoder._encoding_cache)  # type: ignore
        print(f"Decoded: {decoded_value} (Confidence: {confidence})")
        grapher.plot_sdr(encoded, title=f"RDSE Encoding: {columns[1]}={value:.2f} (SDR {i + 1})")

        sdr.append(encoded)

    for i, encoded in enumerate(sdr):
        print(
            f"Hamming distance from first SDR: {hamming_distance(sdr[0], encoded)} (Step {i + 1})"
        )
        print(f"Overlap with first SDR: {overlap(sdr[0], encoded)} (Step {i + 1})")


def show_brain_creation_demo() -> None:
    """Demonstrate creating a Brain and inspecting its structure.
    Use Case Input Parameters
    """

    brain = ag.Brain()
    trainer = ag.Trainer(brain)

    input_fields: list[tuple[str, int, en.ParameterMarker]] = []

    # Example of building a brain with specific input fields
    input_fields = [
        ("temperature_input", 2048, en.RDSEParameters()),
        ("humidity_input", 2048, en.RDSEParameters()),
        ("energy_consumption_input", 2048, en.RDSEParameters()),
        ("date_input", 2048, en.DateEncoderParameters()),
        ("category_input", 2048, en.CategoryParameters()),
        ("wave_input", 2048, en.FourierEncoderParameters()),
    ]

    trainer.main_brain = trainer.build_brain(input_fields)

    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer.main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer.main_brain._column_fields.values():
        print(f"- {field.name}")


def sine_wave_demo(steps: int = 100) -> None:
    """Demonstrate encoding and learning on a simple sine wave dataset.

    Use Case Train Model -> Train Brain -> Use HTM -> Encode SDR / Decode SDR

    """

    import numpy as np

    # Generate a sine wave dataset
    x = np.linspace(0, 1, 2048, endpoint=False)
    y = np.sin(2 * np.pi * 1 * x)
    data = {"sine_wave_input": y}
    brain = ag.Brain()
    trainer = ag.Trainer(brain)
    trainer.main_brain = trainer.build_brain(
        [("sine_wave_input", 2048, en.RDSEParameters(resolution=0.001))]
    )
    brain = trainer.main_brain

    for name, value in data.items():
        logger.debug(DATA_COLUMN_LOG_MESSAGE, name, len(value))

    column = {"sine_wave_input": y.tolist()}

    trainer.train_column(column, steps, brain)

    # show predicted vs actual values for the last 100 steps
    test_results = trainer.test(brain, column, steps)

    trainer.show_active_columns(brain, dataset_name="sine wave")
    trainer.show_heat_map(brain, dataset_name="sine wave")

    report_path = os.path.join(PROJECT_ROOT, "docs/reports/sine_wave_training_stats.txt")
    trainer.print_train_stats(report_path, test_results=test_results, training_steps=steps)


def fin_data_demo(column: str | None = None, steps: int = 100) -> None:
    """Demonstrate loading and visualizing data from the dataset.

    Use Cace Input Data -> Validate Data -> Input Parameters -> Train Model -> Train Brain -> Use HTM -> Encode/Decode SDR
    """

    ih = il.InputHandler()
    data = ih.input_data(ESD)

    brain = ag.Brain()
    trainer = ag.Trainer(brain)
    # resolution changes the spatial pooler representation
    trainer.main_brain = trainer.build_full_brain(data, 2048, en.RDSEParameters(resolution=0.01))
    brain = trainer.main_brain

    if not column:
        for name, value in data.items():
            logger.debug(DATA_COLUMN_LOG_MESSAGE, name, len(value))

            trainer.train_column(brain, column={f"{name}_input": value}, steps=steps)

    elif column not in data:
        logger.error("Specified column '%s' not found in dataset.", column)
        return
    else:
        trainer.train_column(brain, column={f"{column}_input": data[column]}, steps=steps)

    brain.print_stats()

    # trainer.test(trainer._main_brain, {f"{column}_input": data[column]}, steps=steps)

    trainer.show_active_columns(brain, dataset_name="financial data")
    trainer.show_heat_map(brain, dataset_name="financial data")


def rec_center_demo(steps: int = 100) -> None:
    """Demonstrate loading and visualizing the rec_center dataset."""

    columns: list[str] = []
    ih = il.InputHandler()
    brain = ag.Brain()
    trainer = ag.Trainer(brain)
    data = ih.input_data(REC_CENTER)

    for key in data.keys():
        logger.debug(DATA_COLUMN_LOG_MESSAGE, key, len(data[key]))

        columns.append(key)

    trainer.main_brain = trainer.build_brain(
        [
            (f"{columns[0]}_input", 2048, en.DateEncoderParameters()),
            (f"{columns[1]}_input", 2048, en.RDSEParameters(resolution=0.1)),
        ]
    )

    brain = trainer.main_brain

    for field in brain._input_fields.values():
        logger.debug(
            f"Brain input field: {field.name} with encoder: {type(field.encoder).__name__}"
        )

        trainer.train_column(
            brain, column={field.name: data[field.name.replace("_input", "")]}, steps=steps
        )

    brain.print_stats()
    trainer.show_active_columns(brain, dataset_name="recreation center data")
    trainer.show_heat_map(brain, dataset_name="recreation center data")


def show_field_single_encoding_demo() -> None:
    """Demonstrate encoding a sample input through the Brain's input fields."""

    brain = ag.Brain()
    trainer = ag.Trainer(brain)

    input_fields: list[tuple[str, int, en.ParameterMarker]] = []
    # Create a simple brain with one input field and one column field
    input_fields = [
        ("temperature_input", 2048, en.RDSEParameters(resolution=0.1)),
    ]
    trainer.main_brain = trainer.build_brain(input_fields)

    # Example input value to encode
    sample_input = {"temperature_input": 25.8}

    # Step the brain with the sample input
    output = trainer.main_brain._input_fields["temperature_input"].encode(
        sample_input["temperature_input"]
    )

    grapher.plot_sdr(
        output, title=f"RDSE Encoding: temperature_input={sample_input['temperature_input']}°C"
    )


if __name__ == "__main__":
    # Ensure we're in the project root directory for consistent file operations
    os.chdir(PROJECT_ROOT)

    # Example usage of the Brain and Trainer classes

    show_input_data_demo()
    show_input_to_encoder_demo(3)
    show_brain_creation_demo()
    sine_wave_demo(200)
    fin_data_demo(steps=3)
    rec_center_demo(200)
    show_field_single_encoding_demo()

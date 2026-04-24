#!/usr/bin/env python3
"""Demo all encoder types with valid sample inputs and SDR visualization.

Usage:
        python scripts/encoder_types.py
        python scripts/encoder_types.py --encoder rdse
        python scripts/encoder_types.py --no-plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import grapher  # noqa: E402
import psu_capstone.encoder_layer as el  # noqa: E402
from psu_capstone.encoder_layer.encoder_factory import EncoderFactory  # noqa: E402

# Keep demo/probe resolutions in one place so behavior stays consistent.
RDSE_RESOLUTION = 0.01
FOURIER_BIN_RESOLUTION = 1.0
GEOSPATIAL_DEFAULT_SPEED_MPS = 5.0


@dataclass
class EncoderDemo:
    name: str
    build_and_encode: Callable[[], list[int]]
    input_summary: str
    semantic_probe: Callable[[], tuple[list[int], list[int], list[int], str, str, str]] | None = (
        None
    )


def _build_encoder(encoder_type: str, params: dict[str, Any]) -> Any:
    """Create an encoder through the shared EncoderFactory."""
    return cast(Any, EncoderFactory.create_encoder(encoder_type, params))


def _rdse_params() -> dict[str, Any]:
    return {"size": 256, "sparsity": 0.1, "resolution": RDSE_RESOLUTION, "seed": 42}


def _fourier_params() -> dict[str, Any]:
    return {
        "size": 512,
        "frequency_ranges": [(0, 32), (32, 64), (64, 128)],
        "sparsity_in_ranges": [0.05, 0.05, 0.05],
        "resolutions_in_ranges": [
            FOURIER_BIN_RESOLUTION,
            FOURIER_BIN_RESOLUTION,
            FOURIER_BIN_RESOLUTION,
        ],
        "total_samples": 256,
        "start_time": 0.0,
        "stop_time": 1.0,
    }


def _rdse_demo() -> list[int]:
    params = _rdse_params()
    encoder = _build_encoder("rdse", params)
    return encoder.encode(12.34)


def _scalar_demo() -> list[int]:
    params = {
        "minimum": 0,
        "maximum": 100,
        "active_bits": 16,
        "size": 256,
        "radius": 0.0,
        "resolution": 1.0,
    }
    encoder = _build_encoder("scalar", params)
    return encoder.encode(42)


def _category_demo() -> list[int]:
    params = {"w": 8, "category_list": ["cat", "dog", "bird"], "rdse_used": True}
    encoder = _build_encoder("category", params)
    return encoder.encode("dog")


def _category_new_demo() -> list[int]:
    params = {
        "active_bits_per_category": 12,
        "sparsity": 0.0,
        "size": 256,
        "category_list": ["red", "green", "blue"],
        "rdse_used": True,
    }
    encoder = _build_encoder("new_category", params)
    return encoder.encode("green")


def _date_demo() -> list[int]:
    params = {
        "size": 512,
        "year_size": 64,
        "season_size": 64,
        "day_of_week_size": 64,
        "weekend_size": 64,
        "holiday_size": 64,
        "time_of_day_size": 64,
        "custom_size": 64,
    }
    encoder = _build_encoder("date", params)
    return encoder.encode(datetime(2026, 4, 19, 15, 30, 0))


def _fourier_demo() -> list[int]:
    params = _fourier_params()
    encoder = _build_encoder("fourier", params)
    signal = _fourier_sample_signal()
    return encoder.encode(signal)


def _fourier_sample_signal() -> np.ndarray:
    """Return the canonical Fourier demo signal used for plotting and encoding."""
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    return np.sin(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 24 * t)


def _geo_input(lon: float, lat: float) -> tuple[float, float, float]:
    """Build geospatial encoder input with a fixed internal speed."""
    return (GEOSPATIAL_DEFAULT_SPEED_MPS, lon, lat)


def _coordinate_demo() -> list[int]:
    params = {"n": 128, "w": 8, "max_radius": 4, "dims": 2, "use_all_neighbors": False}
    encoder = _build_encoder("coordinate", params)
    return encoder.encode(([10, 20], 2))


def _geospatial_demo() -> list[int]:
    params = {
        "xy_scale": 5.0,
        "z_scale": 1.0,
        "timestep": 1.0,
        "max_radius": 4,
        "use_altitude": False,
    }
    encoder = _build_encoder("geospatial", params)
    return encoder.encode(_geo_input(-77.0365, 38.8977))


def _delta_demo() -> list[int]:
    params = {"size": 256, "sparsity": 0.1}
    encoder = _build_encoder("delta", params)
    return encoder.encode((12.5, 9.75))


def _rdse_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = _rdse_params()
    encoder = _build_encoder("rdse", params)
    return (
        encoder.encode(12.0),
        encoder.encode(12.05),
        encoder.encode(35.0),
        "12.0",
        "12.05",
        "35.0",
    )


def _scalar_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = {
        "minimum": 0,
        "maximum": 100,
        "active_bits": 16,
        "size": 256,
        "radius": 0.0,
        "resolution": 1.0,
    }
    encoder = _build_encoder("scalar", params)
    return (
        encoder.encode(42),
        encoder.encode(43),
        encoder.encode(90),
        "42",
        "43",
        "90",
    )


def _date_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = {
        "size": 512,
        "year_size": 64,
        "season_size": 64,
        "day_of_week_size": 64,
        "weekend_size": 64,
        "holiday_size": 64,
        "time_of_day_size": 64,
        "custom_size": 64,
    }
    encoder = _build_encoder("date", params)
    base = datetime(2026, 4, 19, 15, 30, 0)
    close = datetime(2026, 4, 19, 16, 0, 0)
    far = datetime(2026, 10, 19, 15, 30, 0)
    return (
        encoder.encode(base),
        encoder.encode(close),
        encoder.encode(far),
        str(base),
        str(close),
        str(far),
    )


def _coordinate_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = {"n": 128, "w": 8, "max_radius": 4, "dims": 2, "use_all_neighbors": False}
    encoder = _build_encoder("coordinate", params)
    return (
        encoder.encode(([10, 20], 2)),
        encoder.encode(([11, 20], 2)),
        encoder.encode(([60, 80], 2)),
        "([10, 20], r=2)",
        "([11, 20], r=2)",
        "([60, 80], r=2)",
    )


def _geospatial_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = {
        "xy_scale": 5.0,
        "z_scale": 1.0,
        "timestep": 1.0,
        "max_radius": 4,
        "use_altitude": False,
    }
    encoder = _build_encoder("geospatial", params)
    return (
        encoder.encode(_geo_input(-77.0365, 38.8977)),
        encoder.encode(_geo_input(-77.0360, 38.8979)),
        encoder.encode(_geo_input(-0.1276, 51.5072)),
        "(-77.0365, 38.8977)",
        "(-77.0360, 38.8979)",
        "(-0.1276, 51.5072)",
    )


def _delta_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = {"size": 256, "sparsity": 0.1}
    encoder = _build_encoder("delta", params)
    return (
        encoder.encode((12.5, 9.75)),
        encoder.encode((12.6, 9.8)),
        encoder.encode((60.0, -30.0)),
        "(12.5, 9.75)",
        "(12.6, 9.8)",
        "(60.0, -30.0)",
    )


def _fourier_probe() -> tuple[list[int], list[int], list[int], str, str, str]:
    params = _fourier_params()
    encoder = _build_encoder("fourier", params)
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    base = np.sin(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 24 * t)
    close = np.sin(2 * np.pi * 9 * t) + 0.5 * np.sin(2 * np.pi * 24 * t)
    far = np.sin(2 * np.pi * 60 * t)
    return (
        encoder.encode(base),
        encoder.encode(close),
        encoder.encode(far),
        "sin(8Hz)+0.5*sin(24Hz)",
        "sin(9Hz)+0.5*sin(24Hz)",
        "sin(60Hz)",
    )


def _build_demos() -> list[EncoderDemo]:
    return [
        EncoderDemo("rdse", _rdse_demo, "12.34 (float)", semantic_probe=_rdse_probe),
        EncoderDemo("scalar", _scalar_demo, "42 (int)", semantic_probe=_scalar_probe),
        EncoderDemo("category", _category_demo, '"dog" (str category)'),
        EncoderDemo("category_new", _category_new_demo, '"green" (str category)'),
        EncoderDemo(
            "date", _date_demo, "datetime(2026-04-19 15:30:00)", semantic_probe=_date_probe
        ),
        EncoderDemo(
            "fourier",
            _fourier_demo,
            "1D sine mixture numpy array (256 samples)",
            semantic_probe=_fourier_probe,
        ),
        EncoderDemo(
            "coordinate",
            _coordinate_demo,
            "([10, 20], radius=2)",
            semantic_probe=_coordinate_probe,
        ),
        EncoderDemo(
            "geospatial",
            _geospatial_demo,
            "(lon, lat) = (-77.0365, 38.8977)",
            semantic_probe=_geospatial_probe,
        ),
        EncoderDemo("delta", _delta_demo, "(12.5, 9.75) numeric pair", semantic_probe=_delta_probe),
    ]


def _active_count(sdr: list[int]) -> int:
    return int(np.count_nonzero(np.asarray(sdr)))


def _overlap_score(a: list[int], b: list[int]) -> float:
    a_arr = np.asarray(a, dtype=np.uint8)
    b_arr = np.asarray(b, dtype=np.uint8)
    intersection = int(np.count_nonzero(a_arr & b_arr))
    min_active = max(1, min(_active_count(a), _active_count(b)))
    return intersection / min_active


def _hamming_distance(a: list[int], b: list[int]) -> int:
    a_arr = np.asarray(a, dtype=np.uint8)
    b_arr = np.asarray(b, dtype=np.uint8)
    return int(np.count_nonzero(a_arr != b_arr))


def main() -> None:
    demos = _build_demos()
    demo_names = [demo.name for demo in demos]

    parser = argparse.ArgumentParser(
        description="Show each encoder type with valid input and resulting SDR plots."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="all",
        choices=["all", *demo_names],
        help="Run a single encoder demo or all demos.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip grapher.plot_sdr windows (useful for headless runs).",
    )
    args = parser.parse_args()

    selected = (
        demos if args.encoder == "all" else [next(d for d in demos if d.name == args.encoder)]
    )

    print("\\nEncoder Demo: valid input -> SDR")
    print("=" * 64)

    for demo in selected:
        print(f"\\n[{demo.name}] input: {demo.input_summary}")
        sdr = demo.build_and_encode()
        active = _active_count(sdr)
        print(f"[{demo.name}] SDR length={len(sdr)} active_bits={active}")

        if demo.semantic_probe is not None:
            base, close, far, base_label, close_label, far_label = demo.semantic_probe()
            close_overlap = _overlap_score(base, close)
            far_overlap = _overlap_score(base, far)
            close_hamming = _hamming_distance(base, close)
            far_hamming = _hamming_distance(base, far)
            print(f"[{demo.name}] semantic probe base={base_label}")
            print(
                f"[{demo.name}] close={close_label} overlap={close_overlap:.3f} hamming={close_hamming}"
            )
            print(f"[{demo.name}] far={far_label} overlap={far_overlap:.3f} hamming={far_hamming}")
        else:
            print(f"[{demo.name}] semantic probe: n/a (categorical example)")

        if not args.no_plot:
            if demo.name == "fourier":
                grapher.plot_signal(
                    _fourier_sample_signal(),
                    sample_rate=256,
                    domain="both",
                    title="fourier_demo_signal",
                )
            grapher.plot_sdr(
                sdr,
                title=(
                    f"{demo.name} encoder\\n"
                    f"input={demo.input_summary}\\n"
                    f"len={len(sdr)} active={active}"
                ),
            )

    print("\\nDone.")


if __name__ == "__main__":
    main()

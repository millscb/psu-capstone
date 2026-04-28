from datetime import datetime

import numpy as np
import pytest

from htmrl.encoder_layer.encoder_factory import EncoderFactory
from htmrl.input_layer.input_handler import InputHandler


def fourier_helper():
    time = np.linspace(0, 1, 128, endpoint=False)
    signal = np.zeros_like(time)
    components = [(3, 1.0, 0.0), (7, 0.5, 0.25)]

    for frequency, amplitude, phase in components:
        signal += amplitude * np.sin(2 * np.pi * frequency * time + phase)

    return signal


@pytest.mark.parametrize(
    "column,payload,encoder_type,parameters",
    [
        (
            "dt",
            [{"dt": datetime(2026, 3, 1, 12, 0, 0)}, {"dt": datetime(2026, 3, 1, 12, 0, 1)}],
            "date",
            {"size": 2048},
        ),
        (
            "random",
            [{"random": 1.5}, {"random": 2.5}, {"random": 3.5}],
            "rdse",
            {"size": 128, "sparsity": 0.1, "resolution": 1.0, "seed": 42},
        ),
        (
            "scal",
            [{"scal": 400}, {"scal": 500}, {"scal": 600}],
            "scalar",
            {"minimum": 0, "maximum": 1000},
        ),
        (  # Had errors here because of the inputs being incorrect
            "coor",
            [
                {"coor": ((10, 20, 1), 2)},
                {"coor": ((40, 40, 2), 4)},
                {"coor": ((50, 50, 3), 3)},
            ],
            "coordinate",
            {"n": 2048, "w": 25, "dims": 3},
        ),
        (  # Had errors because I forgot to give it coordinate defaults
            "geo",
            [{"geo": ((5.0, -177.0365, 38.8977, 10.0))}],
            "geospatial",
            {},
        ),
        (
            "categore",
            [
                {"categore": "dog"},
                {"categore": "CA"},
                {"categore": "PF"},
            ],
            "new_category",
            {"size": 2048, "category_list": ["dog", "CA", "PF"]},
        ),
        (
            "four",
            [{"four": fourier_helper()}],
            "fourier",
            {"size": 2048},
        ),
        (
            "delt",
            [{"delt": [(10.0, 5.0), (1.0, 3.5)]}],
            "delta",
            {"size": 2048},
        ),
    ],
)
# Test Type: integration test
def test_input_handler_to_encoder_pipeline(column, payload, encoder_type, parameters):
    # TI-016
    ih = InputHandler()
    ih.input_data(payload)
    values = ih.get_column_data(column)

    encoder = EncoderFactory.create_encoder(encoder_type, parameters)

    for value in values:
        sdr = encoder.encode(value)
        assert isinstance(sdr, list)
        assert len(sdr) == encoder.size
        assert all(bit in (0, 1) for bit in sdr)
        assert sum(sdr) > 0

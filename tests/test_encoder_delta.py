"""Regression tests for DeltaEncoder parameter contract behavior."""

import pytest

from htmrl.encoder_layer.delta_encoder import DeltaEncoder, DeltaEncoderParameters


# Test Type: unit test
# TS-31 TC-263
def test_delta_sparsity_config_computes_meaningful_active_bits_attribute():
    """When only sparsity is configured, active_bits should still be meaningful."""
    encoder = DeltaEncoder(
        DeltaEncoderParameters(
            size=1000,
            sparsity=0.02,
            active_bits=0,
        )
    )

    assert encoder._active_bits == 20
    assert encoder._sparsity == pytest.approx(0.02)


# Test Type: unit test
# TS-31 TC-264
def test_delta_active_bits_config_computes_meaningful_sparsity_attribute():
    """When only active_bits is configured, sparsity should still be meaningful."""
    encoder = DeltaEncoder(
        DeltaEncoderParameters(
            size=500,
            active_bits=25,
            sparsity=0.0,
        )
    )

    assert encoder._active_bits == 25
    assert encoder._sparsity == pytest.approx(0.05)
    out = encoder.encode((10.0, 7.0))
    assert len(out) == 500
    assert sum(out) == 25

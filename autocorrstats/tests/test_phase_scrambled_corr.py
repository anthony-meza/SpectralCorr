"""Tests for phase scrambling helpers."""

import numpy as np

from autocorrstats import AR1_process
from autocorrstats.ebisuzaki_surrogate_generation import phase_scrambled_surrogates


def test_phase_scrambled_surrogates_shape_and_dims():
    ts = AR1_process(0.9, 1.0, 1.0, 128, seed=42)
    surrogates = phase_scrambled_surrogates(ts, n_surrogates=5)

    assert surrogates.dims == ("surrogate", "time")
    assert surrogates.shape == (5, 128)
    assert surrogates.name == "phase_scrambled_surrogates"


def test_phase_scrambled_surrogates_single_output_squeezes_surrogate_dim():
    ts = AR1_process(0.9, 1.0, 1.0, 64, seed=42)
    surrogate = phase_scrambled_surrogates(ts, n_surrogates=1)

    assert surrogate.dims == ("time",)
    assert surrogate.shape == (64,)


def test_phase_scrambled_surrogates_preserve_time_coordinate():
    ts = AR1_process(0.7, 1.0, 0.0, 50, seed=42, dt=0.25)
    surrogates = phase_scrambled_surrogates(ts, n_surrogates=3)

    np.testing.assert_allclose(surrogates.time.values, ts.time.values)

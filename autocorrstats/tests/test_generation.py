import numpy as np
import pytest

from autocorrstats import AR1_process


def test_ar1_process_returns_named_time_series():
    ts = AR1_process(0.8, 1.0, 0.5, 5, seed=42, dt=0.25, name="series")

    assert ts.name == "series"
    assert ts.sizes["time"] == 5
    assert ts.values[0] == 0.5
    np.testing.assert_allclose(ts.time.values, np.arange(5) * 0.25)


def test_ar1_process_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        AR1_process(1.0, 1.0, 0.0, 10)

    with pytest.raises(ValueError):
        AR1_process(0.5, 0.0, 0.0, 10)

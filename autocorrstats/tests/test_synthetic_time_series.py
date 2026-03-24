import numpy as np
import pytest
import xarray as xr

from autocorrstats import AR1_process
from autocorrstats.ebisuzaki_significance_testing import empirical_p_value
from autocorrstats.ebisuzaki_surrogate_generation import phase_scrambled_surrogates


def test_synthetic_time_series_keep_shape_and_time():
    ts = AR1_process(0.7, 1.0, 0.0, 16, seed=42, dt=0.25)

    synthetic_series = phase_scrambled_surrogates(ts, n_surrogates=3)

    assert synthetic_series.dims == ("surrogate", "time")
    assert synthetic_series.shape == (3, 16)
    np.testing.assert_allclose(synthetic_series.time.values, ts.time.values)


def test_synthetic_time_series_p_value_rejects_invalid_alternative():
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'greater', or 'less'"):
        empirical_p_value(
            xr.DataArray([0.5], dims=["lag"]),
            xr.DataArray([[0.1]], dims=["surrogate", "lag"]),
            alternative="invalid",
        )

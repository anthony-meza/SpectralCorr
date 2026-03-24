import numpy as np
import pytest
import xarray as xr

from autocorrstats import polynomial_coefficients, polynomial_coefficient_significance


def test_polynomial_coefficients_fit_a_line():
    time = np.arange(6, dtype=float)
    ts = xr.DataArray(2.0 * time + 1.0, dims=["time"], coords={"time": time})

    coeffs = polynomial_coefficients(ts, degree=1)

    assert coeffs.sel(degree=1).item() == pytest.approx(2.0)
    assert coeffs.sel(degree=0).item() == pytest.approx(1.0)


def test_polynomial_significance_returns_a_pvalue():
    time = np.arange(32, dtype=float)
    ts = xr.DataArray(0.1 * time + np.sin(time / 4), dims=["time"], coords={"time": time})

    result = polynomial_coefficient_significance(ts, degree=1, n_surrogates=4)

    assert "polynomial_coefficient" in result
    assert "polynomial_coefficient_pvalue" in result

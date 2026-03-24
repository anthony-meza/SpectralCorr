"""Tests for polynomial coefficient utilities."""

import numpy as np
import xarray as xr

from autocorrstats import (
    polynomial_coefficients,
    polynomial_coefficient_significance,
)


def test_polynomial_coefficients_linear_trend():
    time = np.arange(20, dtype=float)
    ts = xr.DataArray(2.0 * time + 1.0, dims=["time"], coords={"time": time})

    coeffs = polynomial_coefficients(ts, degree=1)

    np.testing.assert_allclose(coeffs.sel(degree=1), 2.0, atol=1e-10)
    np.testing.assert_allclose(coeffs.sel(degree=0), 1.0, atol=1e-10)


def test_polynomial_coefficient_significance_returns_expected_vars():
    np.random.seed(0)
    time = np.arange(64, dtype=float)
    ts = xr.DataArray(
        0.05 * time + np.sin(time / 8),
        dims=["time"],
        coords={"time": time},
    )

    result = polynomial_coefficient_significance(
        ts,
        degree=1,
        n_surrogates=8,
        return_distributions=True,
    )

    assert "polynomial_coefficient" in result
    assert "polynomial_coefficient_pvalue" in result
    assert "polynomial_coefficient_distribution" in result
    assert "degree" in result["polynomial_coefficient"].dims
    assert result["polynomial_coefficient_pvalue"].dims == ()
    assert result["polynomial_coefficient_distribution"].dims == (
        "bootstrap_iter",
        "degree",
    )


def test_polynomial_coefficient_pvalue_only_covers_selected_degree():
    np.random.seed(0)
    time = np.arange(64, dtype=float)
    ts = xr.DataArray(
        -0.05 * time + np.sin(time / 8),
        dims=["time"],
        coords={"time": time},
    )

    result = polynomial_coefficient_significance(
        ts,
        degree=1,
        n_surrogates=8,
        alternative="less",
    )

    assert result["polynomial_coefficient_pvalue"].dims == ()

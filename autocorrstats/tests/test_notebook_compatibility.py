"""
Test that the notebook examples work with the new xarray interface.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import pandas as pd
from autocorrstats import (
    AR1_process,
    cross_correlation,
    maximum_cross_correlation,
    polynomial_coefficient_significance,
)


def test_ar1_lagged_example_notebook():
    """Test the core code path used in AR1_lagged_example.ipynb."""
    # Time series parameters
    N = 500
    dt = 1.0

    # AR(1) model parameters
    rho_x = 0.9
    rho_y = 0.9
    noise_std = 1.0

    # Generate time series (returns xarray by default)
    ts1 = AR1_process(rho_x, noise_std, 1, N, seed=42, dt=dt)
    ts2 = AR1_process(rho_y, noise_std, 1, N, seed=31, dt=dt)

    # Verify xarray structure
    assert 'time' in ts1.coords
    assert ts1.shape == (N,)

    # Compute cross-correlation
    ccf_maxlag = 36
    ccf_ds = cross_correlation(ts1, ts2, maxlags=ccf_maxlag, method='pearson')
    lag_max, ccf_max = maximum_cross_correlation(ts1, ts2, maxlags=ccf_maxlag)

    # Verify results structure
    assert 'cross_correlation' in ccf_ds.data_vars
    assert 'cross_correlation_pvalue' in ccf_ds.data_vars
    assert isinstance(lag_max, (int, float, np.number))
    assert isinstance(ccf_max, (float, np.number))


def test_ebisuzaki_method():
    """Test Ebisuzaki bootstrap method with distribution output."""
    # Generate test time series
    N = 500
    dt = 1.0
    ts1 = AR1_process(0.9, 1.0, 1, N, seed=42, dt=dt)
    ts2 = AR1_process(0.9, 1.0, 1, N, seed=31, dt=dt)

    # Run Ebisuzaki method with small n_surrogates for speed
    n_surrogates = 100
    ccf_maxlag = 36
    result = cross_correlation(
        ts1, ts2,
        maxlags=ccf_maxlag,
        method='ebisuzaki',
        n_surrogates=n_surrogates,
        return_distributions=True,
        detrend="constant"
    )

    # Verify bootstrap distributions are returned
    assert 'cross_correlation_distribution' in result
    assert result.cross_correlation_distribution.shape == (n_surrogates, 2 * ccf_maxlag + 1)


def test_xarray_plotting_compatibility():
    """Test that xarray plotting works correctly."""
    # Generate test data
    N = 100
    ts1 = AR1_process(0.9, 1.0, 1, N, seed=42)
    ts2 = AR1_process(0.8, 1.0, 1, N, seed=31)
    ccf_ds = cross_correlation(ts1, ts2, maxlags=10, method='pearson')

    # Create plots (don't display, just verify no errors)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    ts1.plot(ax=axes[0], label='Time Series 1')
    ts2.plot(ax=axes[0], label='Time Series 2')
    axes[0].legend()

    ccf_ds["cross_correlation"].plot(ax=axes[1])

    plt.close(fig)


def test_polynomial_trend_example_notebook():
    """Test the core code path used in a polynomial trend notebook."""
    time = np.arange(120, dtype=float)
    signal = 0.03 * time + np.sin(time / 12)

    import xarray as xr

    ts = xr.DataArray(signal, dims=["time"], coords={"time": time})
    result = polynomial_coefficient_significance(
        ts,
        degree=1,
        n_surrogates=16,
        return_distributions=True,
        alternative="greater",
    )

    assert "polynomial_coefficient" in result
    assert "polynomial_coefficient_pvalue" in result
    assert "polynomial_coefficient_distribution" in result
    assert result["polynomial_coefficient_pvalue"].dims == ()


def test_rapid_example_notebook():
    """Test the core code path used in RAPID_example.ipynb."""
    df = pd.read_csv("notebook_examples/amoc_rapid_RAPID.csv", comment="#")
    decimal_time = df["Year"] + (df["Month"] - 1) / 12

    import xarray as xr

    ts = xr.DataArray(
        df["RAPID (Sv)"].to_numpy(),
        dims=["time"],
        coords={"time": decimal_time.to_numpy()},
        name="rapid_transport",
    )
    result = polynomial_coefficient_significance(
        ts,
        degree=1,
        n_surrogates=16,
        alternative="two-sided",
        return_distributions=True,
    )

    assert "polynomial_coefficient" in result
    assert "polynomial_coefficient_pvalue" in result
    assert "polynomial_coefficient_distribution" in result
    assert result["polynomial_coefficient_pvalue"].dims == ()

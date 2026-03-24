import numpy as np
import pytest
import xarray as xr

from autocorrstats import AR1_process, cross_correlation, maximum_cross_correlation


def make_series(rho: float, seed: int, length: int = 50):
    return AR1_process(rho, 1.0, 0.0, length, seed=seed)


def test_cross_correlation_peaks_at_zero_for_identical_series():
    ts = make_series(0.8, seed=42)

    result = cross_correlation(ts, ts, maxlags=3, method="pearson")

    assert "cross_correlation" in result
    assert "cross_correlation_pvalue" in result
    assert result["cross_correlation"].sel(lag=0.0).item() == pytest.approx(1.0)


def test_maximum_cross_correlation_finds_known_shift():
    base = make_series(0.9, seed=42, length=40)
    ts1 = xr.DataArray(base.values[:-2], coords={"time": base.time.values[:-2]}, dims=["time"])
    ts2 = xr.DataArray(base.values[2:], coords={"time": base.time.values[:-2]}, dims=["time"])

    lag, corr = maximum_cross_correlation(ts1, ts2, maxlags=5)

    assert abs(lag) == 2.0
    assert corr > 0.5


def test_cross_correlation_rejects_bad_input():
    ts1 = make_series(0.8, seed=1, length=20)
    ts2 = make_series(0.7, seed=2, length=20)
    ts_with_gap = xr.DataArray(ts2.values.copy(), coords=ts2.coords, dims=ts2.dims)
    ts_with_gap[5] = np.nan

    with pytest.raises(ValueError, match="Method must be 'pearson' or 'ebisuzaki'"):
        cross_correlation(ts1, ts2, method="invalid")

    with pytest.raises(ValueError, match="missing values"):
        cross_correlation(ts1, ts_with_gap)

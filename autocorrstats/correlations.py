"""Correlation and phase scrambling utilities for time series analysis."""

from math import erf, sqrt
from statistics import NormalDist

import numpy as np
import xarray as xr

from .ebisuzaki_significance_testing import (
    empirical_p_value,
)
from .ebisuzaki_surrogate_generation import _detrend_1d, phase_scrambled_surrogates
from .time_series_validation import check_time_step, validate_time_series


def _pearson_correlation(ts1: xr.DataArray, ts2: xr.DataArray, time_dim="time"):
    if time_dim not in ts1.dims or time_dim not in ts2.dims:
        raise ValueError(f"{time_dim!r} must be a dimension of both inputs")
    if ts1.sizes[time_dim] != ts2.sizes[time_dim]:
        raise ValueError(
            f"Length mismatch: {ts1.sizes[time_dim]} vs {ts2.sizes[time_dim]}"
        )

    valid = ts1.notnull() & ts2.notnull()
    ts1_valid = ts1.where(valid)
    ts2_valid = ts2.where(valid)

    ts1_centered = ts1_valid - ts1_valid.mean(dim=time_dim, skipna=True)
    ts2_centered = ts2_valid - ts2_valid.mean(dim=time_dim, skipna=True)

    numerator = (ts1_centered * ts2_centered).mean(dim=time_dim, skipna=True)
    denominator = ts1_centered.std(dim=time_dim, skipna=True) * ts2_centered.std(
        dim=time_dim, skipna=True
    )

    return numerator / denominator


def _lagged_correlation(ts1: xr.DataArray, ts2: xr.DataArray, lag: int, time_dim="time"):
    if lag < 0:
        shifted_ts1 = ts1.shift({time_dim: -lag})
        valid = shifted_ts1.notnull() & ts2.notnull()
        corr = _pearson_correlation(
            shifted_ts1.where(valid),
            ts2.where(valid),
            time_dim=time_dim,
        )
    else:
        shifted_ts2 = ts2.shift({time_dim: lag})
        valid = ts1.notnull() & shifted_ts2.notnull()
        corr = _pearson_correlation(
            ts1.where(valid),
            shifted_ts2.where(valid),
            time_dim=time_dim,
        )

    sample_size = valid.sum(dim=time_dim).rename("sample_size")
    corr = corr.rename("cross_correlation")
    return corr, sample_size


def _cross_correlation_coefficients(
    ts1: xr.DataArray, ts2: xr.DataArray, maxlags=None, time_dim="time"
):
    n = ts1.sizes[time_dim]
    dt = check_time_step(ts1, time_dim=time_dim)
    ts1_name = ts1.name or "ts1"
    ts2_name = ts2.name or "ts2"

    if maxlags is None:
        maxlags = n - 1
    maxlags = min(maxlags, n - 1)

    lag_indices = np.arange(-maxlags, maxlags + 1, dtype=int)
    lag_values = lag_indices * dt
    correlations = []
    sample_sizes = []
    for lag, lag_value in zip(lag_indices, lag_values, strict=False):
        corr, sample_size = _lagged_correlation(ts1, ts2, lag=lag, time_dim=time_dim)
        correlations.append(corr.expand_dims(lag=[lag_value]))
        sample_sizes.append(sample_size.expand_dims(lag=[lag_value]))

    ccf = xr.concat(correlations, dim="lag").assign_coords(lag=lag_values)
    ccf = ccf.rename("cross_correlation")
    sample_sizes_da = xr.concat(sample_sizes, dim="lag").assign_coords(lag=lag_values)
    sample_sizes_da = sample_sizes_da.rename("sample_size")
    lag_attrs = {
        "description": f"Lag in {time_dim} units",
        "sign_convention": (
            f"negative: {ts1_name} leads {ts2_name}; "
            f"positive: {ts2_name} leads {ts1_name}"
        ),
    }
    ccf["lag"].attrs.update(lag_attrs)
    sample_sizes_da["lag"].attrs.update(lag_attrs)
    return ccf, sample_sizes_da, dt


def cross_correlation(
    ts1,
    ts2,
    maxlags=None,
    method="pearson",
    n_surrogates=5000,
    return_distributions=False,
    detrend="constant",
    significance_level=0.05,
):
    """
    Compute cross-correlation and p-values between two time series.
    """
    if method not in ["pearson", "ebisuzaki"]:
        raise ValueError(f"Method must be 'pearson' or 'ebisuzaki', got '{method}'")

    time_dim = "time"
    validate_time_series(ts1, time_dim=time_dim, function_name="cross_correlation")
    validate_time_series(ts2, time_dim=time_dim, function_name="cross_correlation")

    if ts1.sizes[time_dim] != ts2.sizes[time_dim]:
        raise ValueError(
            "cross_correlation: Time series must have same length: "
            f"ts1 has {ts1.sizes[time_dim]}, ts2 has {ts2.sizes[time_dim]}"
        )

    time1 = ts1[time_dim].astype(float)
    time2 = ts2[time_dim].astype(float)
    if not time1.equals(time2):
        raise ValueError(
            "cross_correlation: Time coordinates must be aligned (same time values for both series)"
        )

    ts1 = _detrend_1d(ts1, detrend=detrend, time_dim=time_dim)
    ts2 = _detrend_1d(ts2, detrend=detrend, time_dim=time_dim)
    ccf, sample_sizes, dt = _cross_correlation_coefficients(
        ts1,
        ts2,
        maxlags=maxlags,
        time_dim=time_dim,
    )
    significance_vars = cross_correlation_significance_test(
        ts1,
        ts2,
        ccf,
        sample_sizes,
        method=method,
        n_surrogates=n_surrogates,
        return_distributions=return_distributions,
        detrend=detrend,
        significance_level=significance_level,
        time_dim=time_dim,
    )

    return xr.Dataset(
        {
            "cross_correlation": ccf.assign_attrs(
                {"description": "Correlation by lag"}
            ),
            **significance_vars,
        },
        coords={"lag": ccf["lag"]},
        attrs={
            "description": f"Cross-correlation ({method})",
            "n_samples": ts1.sizes[time_dim],
            "dt": dt,
        },
    )


def cross_correlation_significance_test(
    ts1: xr.DataArray,
    ts2: xr.DataArray,
    ccf: xr.DataArray,
    sample_sizes: xr.DataArray,
    method="pearson",
    n_surrogates=1000,
    return_distributions=False,
    detrend="constant",
    significance_level=0.05,
    time_dim="time",
):
    if method == "pearson":
        return _cross_correlation_pearson_significance_test(
            ccf, sample_sizes, significance_level
        )
    if method == "ebisuzaki":
        return _cross_correlation_ebisuzaki_significance_test(
            ts1,
            ts2,
            ccf,
            n_surrogates=n_surrogates,
            return_distributions=return_distributions,
            detrend=detrend,
            time_dim=time_dim,
        )
    raise ValueError(f"Method must be 'pearson' or 'ebisuzaki', got '{method}'")


def _cross_correlation_ebisuzaki_significance_test(
    ts1: xr.DataArray,
    ts2: xr.DataArray,
    ccf: xr.DataArray,
    n_surrogates=1000,
    return_distributions=False,
    detrend="constant",
    time_dim="time",
):
    maxlags = (ccf.sizes["lag"] - 1) // 2
    xs_surrogates = phase_scrambled_surrogates(
        ts1,
        detrend=detrend,
        n_surrogates=n_surrogates,
        time_dim=time_dim,
    )
    ys_surrogates = phase_scrambled_surrogates(
        ts2,
        detrend=detrend,
        n_surrogates=n_surrogates,
        time_dim=time_dim,
    )
    surrogate_ccf, _, _ = _cross_correlation_coefficients(
        xs_surrogates,
        ys_surrogates,
        maxlags=maxlags,
        time_dim=time_dim,
    )
    
    ccf_pval = empirical_p_value(
        ccf,
        surrogate_ccf,
        surrogate_dim="surrogate",
        alternative="two-sided",
    )

    data_vars = {
        "cross_correlation_pvalue": ccf_pval.rename("cross_correlation_pvalue"),
    }
    data_vars["cross_correlation_pvalue"].attrs = {
        "description": "Bootstrap p value by lag"
    }

    if return_distributions:
        data_vars["cross_correlation_distribution"] = surrogate_ccf.rename(
            "cross_correlation_distribution"
        ).rename({"surrogate": "bootstrap_iter"})
        data_vars["cross_correlation_distribution"] = data_vars[
            "cross_correlation_distribution"
        ].transpose("bootstrap_iter", "lag", ...)
        data_vars["cross_correlation_distribution"].attrs = {
            "description": "Bootstrap correlation by lag"
        }

    return data_vars


def maximum_cross_correlation(ts1, ts2, maxlags=None, method="pearson"):
    """
    Find the lag and value of maximum cross-correlation.
    """
    ds = cross_correlation(ts1, ts2, maxlags, method=method)
    idx = ds["cross_correlation"].fillna(-np.inf).argmax(dim="lag")
    lag_max = ds["lag"].isel(lag=idx)
    ccf_max = ds["cross_correlation"].isel(lag=idx)

    if lag_max.ndim == 0 and ccf_max.ndim == 0:
        return lag_max.item(), ccf_max.item()
    return lag_max, ccf_max


def _normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _pearson_stats(r, n, significance_level):
    if np.isnan(r) or n < 4:
        return np.nan, np.nan, np.nan

    r_clipped = np.clip(r, -1 + 1e-12, 1 - 1e-12)
    z = np.arctanh(r_clipped) * np.sqrt(n - 3)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    z_crit = NormalDist().inv_cdf(1.0 - significance_level / 2.0)
    se = 1.0 / np.sqrt(n - 3)
    limit = np.tanh(z_crit * se)
    return p_value, -limit, limit


def _cross_correlation_pearson_significance_test(
    ccf: xr.DataArray, sample_sizes: xr.DataArray, significance_level: float
):
    pvals, ci_low, ci_high = xr.apply_ufunc(
        _pearson_stats,
        ccf,
        sample_sizes,
        input_core_dims=[[], []],
        output_core_dims=[[], [], []],
        kwargs={"significance_level": significance_level},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )

    return {
        "cross_correlation_pvalue": pvals.rename("cross_correlation_pvalue").assign_attrs(
            {
                "description": "P value by lag"
            }
        ),
        "pearson_ci_lower": ci_low.rename("pearson_ci_lower").assign_attrs(
            {
                "description": f"Lower {100 * (1 - significance_level):.0f}% Pearson CI"
            }
        ),
        "pearson_ci_upper": ci_high.rename("pearson_ci_upper").assign_attrs(
            {
                "description": f"Upper {100 * (1 - significance_level):.0f}% Pearson CI"
            }
        ),
    }

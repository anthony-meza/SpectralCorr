"""Polynomial coefficient estimation and surrogate significance testing.

In this package, surrogate coefficients are coefficients computed from
synthetic time series generated from the original data.
"""

import xarray as xr

from .ebisuzaki_significance_testing import empirical_p_value
from .ebisuzaki_surrogate_generation import phase_scrambled_surrogates
from .time_series_validation import check_time_step, validate_time_series


def polynomial_coefficients(
    ts: xr.DataArray,
    degree: int = 1,
    time_dim: str = "time",
) -> xr.DataArray:
    """Fit a polynomial along the time dimension and return its coefficients."""
    validate_time_series(ts, time_dim=time_dim, function_name="polynomial_coefficients")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    coeffs = ts.polyfit(dim=time_dim, deg=degree).polyfit_coefficients
    return coeffs.rename("polynomial_coefficient")


def polynomial_coefficient_significance(
    ts: xr.DataArray,
    degree: int = 1,
    *,
    n_surrogates: int = 1000,
    detrend: str | None = None,
    alternative: str = "two-sided",
    return_distributions: bool = False,
    time_dim: str = "time",
) -> xr.Dataset:
    """Test polynomial coefficients against phase-scrambled surrogate coefficients.

    Here, the surrogate coefficients come from synthetic time series produced
    by phase scrambling the original series.
    """
    validate_time_series(
        ts,
        time_dim=time_dim,
        function_name="polynomial_coefficient_significance",
    )
    if degree < 0:
        raise ValueError("degree must be non-negative")

    observed = polynomial_coefficients(ts, degree=degree, time_dim=time_dim)
    surrogates = phase_scrambled_surrogates(
        ts,
        detrend=detrend,
        n_surrogates=n_surrogates,
        time_dim=time_dim,
    )
    surrogate_coeffs = polynomial_coefficients(
        surrogates,
        degree=degree,
        time_dim=time_dim,
    )
    pvalue = empirical_p_value(
        observed.sel(degree=degree),
        surrogate_coeffs.sel(degree=degree),
        surrogate_dim="surrogate",
        alternative=alternative,
    )

    data_vars = {
        "polynomial_coefficient": observed.assign_attrs(
            {"description": "Polynomial coefficient"}
        ),
        "polynomial_coefficient_pvalue": pvalue.rename(
            "polynomial_coefficient_pvalue"
        ).assign_attrs({"description": f"Empirical p value for degree {degree}"}),
    }

    if return_distributions:
        data_vars["polynomial_coefficient_distribution"] = surrogate_coeffs.rename(
            "polynomial_coefficient_distribution"
        ).rename({"surrogate": "bootstrap_iter"}).transpose(
            "bootstrap_iter", ...
        ).assign_attrs(
            {"description": "Surrogate coefficient distribution"}
        )

    return xr.Dataset(
        data_vars,
        attrs={
            "description": "Polynomial coefficient significance",
            "degree": degree,
            "n_samples": ts.sizes[time_dim],
            "n_surrogates": n_surrogates,
            "dt": check_time_step(ts, time_dim=time_dim),
            "alternative": alternative,
            "surrogate_detrend": detrend,
        },
    )

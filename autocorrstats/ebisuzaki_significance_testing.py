"""Utilities for testing observed parameters against surrogate distributions."""

import numpy as np
import xarray as xr


def empirical_p_value(
    test_parameter: xr.DataArray,
    empirical_distribution: xr.DataArray,
    *,
    surrogate_dim: str = "surrogate",
    alternative: str = "two-sided",
) -> xr.DataArray:
    """Compute an empirical p-value from a surrogate distribution."""
    if alternative == "two-sided":
        comparison = np.abs(empirical_distribution) >= np.abs(test_parameter)
    elif alternative == "greater":
        comparison = empirical_distribution >= test_parameter
    elif alternative == "less":
        comparison = empirical_distribution <= test_parameter
    else:
        raise ValueError(
            "alternative must be 'two-sided', 'greater', or 'less'"
        )

    return comparison.mean(dim=surrogate_dim)

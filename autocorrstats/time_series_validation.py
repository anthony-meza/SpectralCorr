"""Validation helpers for xarray time series inputs."""

import numpy as np
import xarray as xr


def _scalar_bool(value) -> bool:
    if isinstance(value, xr.DataArray):
        return bool(np.asarray(value.compute().data).item())
    return bool(np.asarray(value).item())


def check_time_step(ts: xr.DataArray, time_dim="time") -> float:
    if "dt" in ts.attrs:
        return float(ts.attrs["dt"])
    if ts.sizes[time_dim] <= 1:
        return 1.0

    dt_values = ts[time_dim].astype(float).diff(time_dim)
    dt = float(dt_values.isel({time_dim: 0}).item())
    is_constant_step = abs(dt_values - dt) < 1e-10
    if not _scalar_bool(is_constant_step.all()):
        raise ValueError(
            f"Time step ({time_dim}) must be constant throughout the series"
        )
    return dt


def validate_time_series(ts, time_dim="time", function_name="function"):
    """Validate an xarray time series input along a named time dimension."""
    if not isinstance(ts, xr.DataArray):
        raise ValueError(
            f"{function_name}: input must be an xarray.DataArray, got {type(ts)}"
        )
    if time_dim not in ts.coords:
        raise ValueError(f"{function_name}: input must have {time_dim!r} coordinate")
    if time_dim not in ts.dims:
        raise ValueError(f"{function_name}: input must have {time_dim!r} dimension")

    try:
        ts[time_dim].astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"{function_name}: Time coordinates must be numeric (convertible to float): {e}"
        )
    check_time_step(ts, time_dim=time_dim)

    missing = ts.isnull()
    partial_missing = missing.any(dim=time_dim) & ~missing.all(dim=time_dim)
    if _scalar_bool(partial_missing.any()):
        raise ValueError(
            f"{function_name}: missing values along {time_dim!r} must be either all present or all missing for each series"
        )

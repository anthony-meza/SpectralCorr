"""Autoregressive time series generation utilities."""

from typing import Optional

import numpy as np
import xarray as xr

def AR1_process(
    rho: float,
    sigma: float,
    y0: float,
    N: int,
    seed: Optional[int] = None,
    dt: float = 1.0,
    name: Optional[str] = None,
):
    """
    Simulate a first-order autoregressive AR(1) process.

    Generates a time series following the equation:
    y[t] = rho * y[t-1] + eps[t]

    where eps[t] ~ N(0, sigma²) is white Gaussian noise.

    Parameters
    ----------
    rho : float
        AR(1) coefficient (-1 < rho < 1 for stationarity).
        Values closer to 1 indicate stronger persistence.
    sigma : float
        Standard deviation of Gaussian noise innovations.
        Must be positive.
    y0 : float
        Initial value y[0] of the time series.
    N : int
        Number of time points to generate. Must be positive.
    seed : int, optional
        Random seed for reproducible results. If None, uses
        current random state.
    dt : float, optional, default=1.0
        Time step between consecutive samples.
    name : str, optional
        Name for the returned xarray.DataArray. Defaults to ``"timeseries"``.
    Returns
    -------
    xarray.DataArray
        Simulated AR(1) time series with time coordinate.
        
    Raises
    ------
    ValueError
        If rho is not in (-1, 1), sigma <= 0, or N <= 0.
        
    Notes
    -----
    The AR(1) process is stationary when |rho| < 1, with:
    - Mean: y0 / (1 - rho) (asymptotic)
    - Variance: sigma² / (1 - rho²) (asymptotic)
    - Autocorrelation at lag k: rho^k
    
    References
    ----------
    Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). 
    Time series analysis: forecasting and control. John Wiley & Sons.
    
    Examples
    --------
    Generate a stationary AR(1) process with strong persistence:
    
    >>> ts = AR1_process(rho=0.9, sigma=1.0, y0=0.0, N=100, seed=42)
    >>> print(f"Generated {ts.sizes['time']} points with dt={ts.attrs['dt']}")
    Generated 100 points with dt=1.0
    """
    # Input validation
    if not (-1 < rho < 1):
        raise ValueError(f"AR(1) coefficient rho must be in (-1, 1) for stationarity, got {rho}")
    if sigma <= 0:
        raise ValueError(f"Noise standard deviation sigma must be positive, got {sigma}")
    if N <= 0:
        raise ValueError(f"Number of points N must be positive, got {N}")
    if dt <= 0:
        raise ValueError(f"Time step dt must be positive, got {dt}")
    
    rng = np.random.default_rng(seed)
    
    # Initialize time series
    data = np.zeros(N)
    data[0] = y0
    
    for t in range(1, N):
        data[t] = rho * data[t - 1] + rng.normal(0.0, sigma)

    time = np.arange(N) * dt

    return xr.DataArray(
        data,
        coords={"time": time},
        dims=["time"],
        name=name or "timeseries",
        attrs={
            "description": "Time series data",
            "dt": dt,
            "length": N,
            "process": "AR1",
            "rho": rho,
            "sigma": sigma,
        },
    )

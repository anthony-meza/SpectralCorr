"""Phase scrambling utilities for generating surrogate time series."""

import numpy as np
import xarray as xr

from .time_series_validation import check_time_step, validate_time_series


def _detrend_1d(ts: xr.DataArray, detrend: str | None, time_dim="time") -> xr.DataArray:
    if detrend is None:
        return ts
    if detrend == "constant":
        return ts - ts.mean(dim=time_dim)
    if detrend == "linear":
        coeffs = ts.polyfit(dim=time_dim, deg=1)
        trend = xr.polyval(ts[time_dim], coeffs.polyfit_coefficients)
        return ts - trend
    raise ValueError("detrend must be 'constant', 'linear', or None")


def _fft_1d(signal: np.ndarray) -> np.ndarray:
    return np.fft.rfft(np.asarray(signal, dtype=float))


def _xr_rfft(ts: xr.DataArray, time_dim="time") -> xr.DataArray:
    dt = check_time_step(ts, time_dim=time_dim)
    spectrum = xr.apply_ufunc(
        _fft_1d,
        ts,
        input_core_dims=[[time_dim]],
        output_core_dims=[["freq"]],
        vectorize=True,
        output_dtypes=[complex],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"freq": ts.sizes[time_dim] // 2 + 1}},
    )

    freqs = np.fft.rfftfreq(ts.sizes[time_dim], d=dt)
    
    return spectrum.assign_coords({"freq": freqs})


def _xr_irfft(spectrum: xr.DataArray, n: int, time_dim="time") -> xr.DataArray:
    return xr.apply_ufunc(
        np.fft.irfft,
        spectrum,
        input_core_dims=[["freq"]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        kwargs={"n": n},
        output_dtypes=[float],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {time_dim: n}},
    )


def phase_scrambled_surrogates(
    ts: xr.DataArray,
    detrend: str | None = "constant",
    n_surrogates: int = 1,
    time_dim: str = "time",
):
    """
    Generate phase-scrambled surrogates of an xarray DataArray.

    Phase scrambling is applied independently along `time_dim` for every slice
    of the remaining dimensions using NumPy FFTs.

    Parameters
    ----------
    ts : xarray.DataArray
        Input data containing `time_dim`.
    detrend : {"constant", "linear", None}, default "constant"
        Detrending option applied before the FFT.
    n_surrogates : int, default 1
        Number of surrogate series to generate.
    time_dim : str, default "time"
        Name of the dimension along which to phase scramble.

    Returns
    -------
    xarray.DataArray
        Phase-scrambled surrogate(s). A new dimension `"surrogate"` is added
        when `n_surrogates > 1`.
    """
    validate_time_series(ts, time_dim=time_dim, function_name="phase_scrambled_surrogates")
    if detrend not in {"constant", "linear", None}:
        raise ValueError("detrend must be 'constant', 'linear', or None")

    if n_surrogates <= 0:
        raise ValueError("n_surrogates must be positive")

    dt = check_time_step(ts, time_dim=time_dim)
    nt = ts.sizes[time_dim]
    ts = _detrend_1d(ts, detrend, time_dim=time_dim)
    spectrum = _xr_rfft(ts, time_dim=time_dim)

    nf = spectrum.sizes["freq"]
    dc = xr.DataArray(
        np.arange(nf) == 0,
        dims=["freq"],
        coords={"freq": spectrum["freq"]},
    )
    nyq = xr.DataArray(
        (np.arange(nf) == nf - 1) if nt % 2 == 0 and nf > 1 else np.zeros(nf, dtype=bool),
        dims=["freq"],
        coords={"freq": spectrum["freq"]},
    )
    interior = ~(dc | nyq)
    amp = np.abs(spectrum)
    phase = xr.DataArray(
        np.random.uniform(
            0.0,
            2.0 * np.pi,
            size=(n_surrogates, *[spectrum.sizes[d] for d in spectrum.dims]),
        ),
        dims=("surrogate", *spectrum.dims),
        coords={"surrogate": np.arange(n_surrogates), **spectrum.coords},
    )
    
    Fp = xr.where(interior, amp * np.exp(1j * phase), spectrum)
    # Fp = xr.where(dc, 0.0 + 0.0j, Fp)
    Fp = xr.where(nyq, amp * np.cos(phase) * np.sqrt(2.0), Fp)
    # Fp = xr.where(nyq, amp * np.sign(phase - np.pi), Fp)

    out = _xr_irfft(Fp, n=nt, time_dim=time_dim)
    out = out.transpose("surrogate", ...)
    out = out.assign_coords(
        {"surrogate": np.arange(n_surrogates), time_dim: ts[time_dim]}
    )

    out = out.rename("phase_scrambled_surrogates")

    out.attrs = {
        "description": f"{n_surrogates} phase-scrambled surrogate time series",
        "original_detrend": detrend,
        "dt": dt,
    }

    if n_surrogates == 1:
        out = out.squeeze("surrogate", drop=True)

    return out

"""
Tests for AR process generation functions.
"""

import numpy as np
import pytest
from autocorrstats.ar_processes import AR1_process


class TestAR1Process:
    """Test suite for AR1_process function."""
    
    def test_basic_functionality(self):
        """Test basic AR1 generation works."""
        N = 100
        rho = 0.8
        sigma = 1.0
        y0 = 0.5
        dt = 1.0
        
        ts = AR1_process(rho, sigma, y0, N, seed=42, dt=dt)

        assert ts.sizes["time"] == N
        assert ts.attrs["dt"] == dt
        assert len(ts.values) == N
        assert len(ts.time) == N
        assert ts.values[0] == y0
        assert np.allclose(ts.time, np.arange(N) * dt)
    
    def test_reproducibility(self):
        """Test that same seed produces identical results."""
        params = {'rho': 0.9, 'sigma': 1.5, 'y0': 1.0, 'N': 50, 'dt': 0.5}
        
        ts1 = AR1_process(**params, seed=123)
        ts2 = AR1_process(**params, seed=123)

        assert np.array_equal(ts1.values, ts2.values)
        assert np.array_equal(ts1.time, ts2.time)

    def test_custom_name(self):
        """Test that a custom DataArray name is preserved."""
        ts = AR1_process(0.8, 1.0, 0.0, 20, seed=42, name="custom_series")

        assert ts.name == "custom_series"
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        params = {'rho': 0.7, 'sigma': 1.0, 'y0': 0.0, 'N': 100, 'dt': 1.0}
        
        ts1 = AR1_process(**params, seed=1)
        ts2 = AR1_process(**params, seed=2)

        assert not np.array_equal(ts1.values, ts2.values)
        assert ts1.values[0] == ts2.values[0]
    
    def test_statistical_properties(self):
        """Test that generated series has expected statistical properties."""
        rho = 0.8
        sigma = 1.0
        y0 = 0.0
        N = 10000  # Large N for statistical accuracy
        
        ts = AR1_process(rho, sigma, y0, N, seed=42)
        
        # Theoretical variance for large N
        theoretical_var = sigma**2 / (1 - rho**2)
        empirical_var = np.var(ts.values)
        
        # Allow 10% tolerance for finite sample effects
        assert abs(empirical_var - theoretical_var) / theoretical_var < 0.1
        
        # Test autocorrelation at lag 1
        autocorr_lag1 = np.corrcoef(ts.values[:-1], ts.values[1:])[0, 1]
        assert abs(autocorr_lag1 - rho) < 0.05
    
    def test_input_validation(self):
        """Test input validation raises appropriate errors."""
        base_params = {'sigma': 1.0, 'y0': 0.0, 'N': 100, 'dt': 1.0}
        
        # Test rho validation
        with pytest.raises(ValueError, match="rho must be in \\(-1, 1\\)"):
            AR1_process(rho=1.0, **base_params)

        with pytest.raises(ValueError, match="rho must be in \\(-1, 1\\)"):
            AR1_process(rho=-1.0, **base_params)

        with pytest.raises(ValueError, match="rho must be in \\(-1, 1\\)"):
            AR1_process(rho=1.5, **base_params)

        # Test sigma validation
        with pytest.raises(ValueError, match="sigma must be positive"):
            AR1_process(rho=0.5, sigma=0.0, **{k: v for k, v in base_params.items() if k != 'sigma'})

        with pytest.raises(ValueError, match="sigma must be positive"):
            AR1_process(rho=0.5, sigma=-1.0, **{k: v for k, v in base_params.items() if k != 'sigma'})

        # Test N validation
        with pytest.raises(ValueError, match="N must be positive"):
            AR1_process(rho=0.5, N=0, **{k: v for k, v in base_params.items() if k != 'N'})

        with pytest.raises(ValueError, match="N must be positive"):
            AR1_process(rho=0.5, N=-5, **{k: v for k, v in base_params.items() if k != 'N'})

        # Test dt validation
        with pytest.raises(ValueError, match="dt must be positive"):
            AR1_process(rho=0.5, dt=0.0, **{k: v for k, v in base_params.items() if k != 'dt'})

    def test_removed_return_xarray_argument(self):
        """Test legacy return_xarray kwarg is no longer accepted."""
        with pytest.raises(TypeError):
            AR1_process(rho=0.5, sigma=1.0, y0=0.0, N=10, return_xarray=True)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very small rho (close to white noise)
        ts_small_rho = AR1_process(rho=0.01, sigma=1.0, y0=0.0, N=100, seed=42)
        assert len(ts_small_rho.values) == 100

        # rho close to 1 (highly persistent)
        ts_high_rho = AR1_process(rho=0.99, sigma=1.0, y0=1.0, N=50, seed=42)
        assert len(ts_high_rho.values) == 50

        # Very small sigma
        ts_small_sigma = AR1_process(rho=0.5, sigma=1e-6, y0=1.0, N=50, seed=42)
        assert len(ts_small_sigma.values) == 50

        # Single point
        ts_single = AR1_process(rho=0.5, sigma=1.0, y0=2.5, N=1)
        assert ts_single.sizes["time"] == 1
        assert ts_single.values[0] == 2.5
    
    def test_time_array_properties(self):
        """Test properties of generated time array."""
        N = 50
        dt = 0.25
        
        ts = AR1_process(rho=0.6, sigma=1.0, y0=0.0, N=N, dt=dt)
        
        assert len(ts.time) == N
        assert ts.time[0] == 0.0
        assert np.allclose(np.diff(ts.time), dt)
        assert ts.time[-1] == (N - 1) * dt
    
    def test_no_seed_randomness(self):
        """Test that not providing seed gives random results."""
        params = {'rho': 0.7, 'sigma': 1.0, 'y0': 0.0, 'N': 50, 'dt': 1.0}
        
        # Generate multiple series without seed
        series = [AR1_process(**params) for _ in range(5)]
        
        # Check that not all series are identical
        data_arrays = [ts.values for ts in series]
        all_identical = all(np.array_equal(data_arrays[0], arr) for arr in data_arrays[1:])
        assert not all_identical, "All series should be different without fixed seed"

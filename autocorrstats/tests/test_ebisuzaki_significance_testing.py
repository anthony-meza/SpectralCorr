"""Tests for empirical significance testing helpers."""

import numpy as np
import xarray as xr
import pytest

from autocorrstats.ebisuzaki_significance_testing import (
    empirical_p_value,
)


def test_two_sided_empirical_pvalue():
    test_parameter = xr.DataArray([0.5, 1.5], dims=["lag"])
    empirical_distribution = xr.DataArray(
        [[0.1, 1.0], [-0.6, 2.0], [0.7, -1.4], [-0.2, 1.6]],
        dims=["surrogate", "lag"],
    )

    result = empirical_p_value(
        test_parameter,
        empirical_distribution,
    )

    np.testing.assert_allclose(result, [0.5, 0.5])


def test_one_sided_empirical_pvalue():
    test_parameter = xr.DataArray([0.5], dims=["lag"])
    empirical_distribution = xr.DataArray(
        [[0.1], [0.6], [0.7], [-0.2]],
        dims=["surrogate", "lag"],
    )

    result = empirical_p_value(
        test_parameter,
        empirical_distribution,
        alternative="greater",
    )

    np.testing.assert_allclose(result, [0.5])


def test_invalid_alternative_raises():
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'greater', or 'less'"):
        empirical_p_value(
            xr.DataArray([0.5], dims=["lag"]),
            xr.DataArray([[0.1]], dims=["surrogate", "lag"]),
            alternative="invalid",
        )

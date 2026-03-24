"""Public package surface used by the shipped examples."""

from .version import __version__
from .ar_processes import AR1_process
from .correlations import cross_correlation, maximum_cross_correlation
from .ebisuzaki_surrogate_generation import phase_scrambled_surrogates
from .polynomial_coefficients import (
    polynomial_coefficients,
    polynomial_coefficient_significance,
)

__all__ = [
    "__version__",
    "AR1_process",
    "cross_correlation",
    "maximum_cross_correlation",
    "phase_scrambled_surrogates",
    "polynomial_coefficients",
    "polynomial_coefficient_significance",
]

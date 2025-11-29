"""Utility functions for projection-hyperparameter experiments."""

from .fisher_utils import (
    # Eigenspectrum (eigenvalues + effective dimensions)
    compute_eigenspectrum,
    # Projection utilities
    project_gradients_to_cpu,
    compute_kernel_from_projected,
)
from .gradient_cache import (
    GradientCache,
    create_gradient_cache,
)

__all__ = [
    # Fisher utilities
    "compute_eigenspectrum",
    # Projection utilities
    "project_gradients_to_cpu",
    "compute_kernel_from_projected",
    # Gradient cache
    "GradientCache",
    "create_gradient_cache",
]

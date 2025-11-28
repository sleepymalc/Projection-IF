"""Utility functions for projection-hyperparameter experiments."""

from .fisher_utils import (
    # Memory management
    clear_memory,
    # Gradient computation (vmap-based)
    compute_per_sample_gradients,
    # Eigenvalue and effective dimension
    compute_eigenvalues,
    compute_effective_dimension,
    estimate_effective_dimension,
    # Gram matrix
    compute_gram_matrix,
    # Sandwich bounds (unified interface)
    compute_sandwich_bounds,
)
from .gradient_cache import (
    GradientCache,
)

__all__ = [
    # Memory management
    "clear_memory",
    # Gradient computation
    "compute_per_sample_gradients",
    # Fisher utilities
    "compute_eigenvalues",
    "compute_effective_dimension",
    "estimate_effective_dimension",
    "compute_gram_matrix",
    # Sandwich bounds
    "compute_sandwich_bounds",
    # Gradient cache
    "GradientCache",
]

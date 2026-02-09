"""Utility functions for projection-hyperparameter experiments."""

from .fisher import (
    # Safe projection
    safe_project,
    # Eigenspectrum (eigenvalues + effective dimensions)
    compute_eigenspectrum,
    # Effective dimension
    compute_effective_dimension,
    # Unregularized self-influence
    compute_unregularized_self_influence,
    # Projection utilities
    project_gradients,
    compute_kernel_from_projected,
)
from .gradient_cache import (
    GradientCache,
    create_gradient_cache,
)
from .metrics import (
    lds,
    spearman_correlation,
    rank_data,
)
from .data import (
    get_validation_split_indices,
)

__all__ = [
    # Fisher utilities
    "safe_project",
    "compute_eigenspectrum",
    "compute_effective_dimension",
    "compute_unregularized_self_influence",
    # Projection utilities
    "project_gradients",
    "compute_kernel_from_projected",
    # Gradient cache
    "GradientCache",
    "create_gradient_cache",
    # Metrics
    "lds",
    "spearman_correlation",
    "rank_data",
    # Data utilities
    "get_validation_split_indices",
]

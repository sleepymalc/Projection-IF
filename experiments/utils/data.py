"""Data splitting utilities for experiments."""

import numpy as np


def get_validation_split_indices(test_sampler, val_ratio=0.1, seed=0):
    """
    Get indices for validation and test splits without creating dataloaders.

    Args:
        test_sampler: Sampler for the test dataset
        val_ratio: Fraction to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (val_indices, test_indices)
    """
    test_indices = list(test_sampler)
    num_test = len(test_indices)

    np.random.seed(seed)
    np.random.shuffle(test_indices)
    num_val = int(val_ratio * num_test)
    val_indices = test_indices[:num_val]
    new_test_indices = test_indices[num_val:]

    return val_indices, new_test_indices

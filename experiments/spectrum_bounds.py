"""
Spectrum Bounds Validation: Validate Regularized Sketching Bounds
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor

from dattri.func.projection import (
    make_random_projector,
    ProjectionType,
)
from dattri.benchmark.load import load_benchmark
from utils.fisher import (
    compute_eigenspectrum,
    safe_project,
    compute_unregularized_self_influence,
)
from utils.gradient_cache import GradientCache
from typing import Tuple


# =============================================================================
# Robust Float64 Gram Matrix Computation (Fixes Mixed Precision Instability)
# =============================================================================

def compute_robust_gram(
    grad_cache: GradientCache,
    lambda_values: list,
    device: str = "cuda",
    batch_size: int = 64,
    return_eigenvectors: bool = False,
) -> Tuple[Tensor, dict, Tensor, Tensor]:
    """
    Compute exact Gram matrix K = (1/n) G @ G^T in float64 to avoid catastrophic cancellation.

    For high-dimensional models, float32 precision causes catastrophic cancellation
    in influence score computation: ||v||^2 - u^T (G G^T + λI)^{-1} u. Both the numerator
    (sketched) and denominator (exact) must use float64 for stable ratios.

    Args:
        grad_cache: GradientCache containing the training gradients
        lambda_values: List of regularization parameters λ to compute d_λ for
        device: GPU device
        batch_size: Batch size for streaming
        return_eigenvectors: If True, compute and return eigenvectors for φ₀ computation

    Returns:
        eigenvalues: Eigenvalues of K in descending order (float64)
        effective_dims: Dictionary mapping λ -> d_λ
        K: Gram matrix (n, n) in float64, normalized by 1/n
        eigenvectors: Eigenvectors of K (n, n) in descending order (None if return_eigenvectors=False)
    """
    n = grad_cache.n_samples
    print(f"  Computing Exact Gram Matrix (n={n}) in float64...")

    # Initialize accumulator in double precision
    K = torch.zeros(n, n, dtype=torch.float64, device=device)

    # Stream over gradients to build K
    stream_batch = batch_size

    for i_start in range(0, n, stream_batch):
        i_end = min(i_start + stream_batch, n)
        # Load batch (float32 storage) -> Cast to float64
        G_i = grad_cache.get_batch(i_start, i_end, device=device).to(dtype=torch.float64)

        for j_start in range(i_start, n, stream_batch):
            j_end = min(j_start + stream_batch, n)

            if i_start == j_start:
                G_j = G_i
            else:
                G_j = grad_cache.get_batch(j_start, j_end, device=device).to(dtype=torch.float64)

            # Accumulate block
            block = G_i @ G_j.T
            K[i_start:i_end, j_start:j_end] = block

            if i_start != j_start:
                K[j_start:j_end, i_start:i_end] = block.T

    # Normalize
    K.div_(n)

    # Compute eigendecomposition (in float64)
    if return_eigenvectors:
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
        eigenvectors = eigenvectors.flip(1)
    else:
        eigenvalues = torch.linalg.eigvalsh(K)
        eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
        eigenvectors = None

    # Compute effective dimensions
    effective_dims = {}
    for lamb in lambda_values:
        d_lambda = torch.sum(eigenvalues / (eigenvalues + lamb)).item()
        effective_dims[lamb] = d_lambda

    return eigenvalues, effective_dims, K, eigenvectors


# =============================================================================
# Woodbury Precomputation
# =============================================================================

@dataclass
class WoodburyPrecomputed:
    """Precomputed quantities for fast Woodbury solves across multiple λ values.

    Given projected gradients PG (n, m), we precompute:
    - K_proj = (1/n) PG @ PG^T  (n×n kernel matrix)
    - eigenvalues, eigenvectors of K_proj
    - PV = projected test vectors (k, m)
    - U = (1/√n) PG @ PV^T  (n×k matrix for Woodbury formula)
    - pv_norms_sq = ||Pv_i||² for each test vector

    Then for any λ, sketched scores can be computed in O(n²) instead of O(n³).

    For cross-scores (test mode), we also need K_proj to compute the full
    (n_train, k_test) score matrix.
    """
    eigenvalues: Tensor      # (n,) eigenvalues of K_proj
    eigenvectors: Tensor     # (n, n) eigenvectors of K_proj
    U: Tensor                # (n, k) matrix for Woodbury: (1/√n) PG @ PV^T
    pv_norms_sq: Tensor      # (k,) squared norms of projected test vectors
    n: int                   # number of training samples
    K_proj: Tensor = None    # (n, n) kernel matrix (optional, for cross-scores)


def precompute_woodbury(
    grad_cache: GradientCache,
    projector,
    test_vectors: Tensor,
    device: str = "cuda",
    batch_size: int = 64,
    store_k_proj: bool = False,
) -> WoodburyPrecomputed:
    """
    Precompute quantities using Streaming Gram Matrix in Float64.

    This avoids the memory/indexing issues of SVD on the massive (n, m) matrix:
    1. OOM / Indexing: We never materialize the full (n, m) PG matrix.
       We only store the (n, n) kernel K and (n, k) U matrix.
    2. PSD / Stability: We accumulate K in float64. This prevents the
       catastrophic cancellation that breaks PSD-ness in float32.
       Tiny negative eigenvalues (< machine epsilon) are clamped to 0.

    After this, solving for any λ is O(n²) instead of O(n³).

    Args:
        grad_cache: GradientCache containing the training gradients
        projector: dattri projector with .project() method
        test_vectors: Test vectors (k, d) on device
        device: GPU device
        batch_size: Batch size for streaming
        store_k_proj: If True, store K_proj in result (needed for cross-scores)

    Returns:
        WoodburyPrecomputed containing all precomputed quantities
    """
    n = grad_cache.n_samples
    k = test_vectors.shape[0]
    d = grad_cache.dim

    # Step 1: Project test vectors → PV (k, m) in float64 for precision
    # Use safe_project to handle m >= d case (identity projection)
    # Project in batches to avoid OOM on large test sets
    pv_batches = []
    for i_start in range(0, k, batch_size):
        i_end = min(i_start + batch_size, k)
        test_batch = test_vectors[i_start:i_end]
        pv_batch = safe_project(test_batch, projector, d, ensemble_id=0).to(dtype=torch.float64, device=device)
        pv_batches.append(pv_batch)
    PV = torch.cat(pv_batches, dim=0)
    pv_norms_sq = (PV ** 2).sum(dim=1)  # (k,)

    # Step 2: Initialize accumulators in float64 (critical for numerical stability)
    K = torch.zeros(n, n, dtype=torch.float64, device=device)
    U = torch.zeros(n, k, dtype=torch.float64, device=device)

    # Step 3: Stream over gradients to build K and U without storing full PG
    # We project each batch and immediately accumulate into K
    # This requires O(n^2 / batch_size) projection calls but uses O(n^2) memory

    # First pass: project all gradients and store temporarily for K computation
    # We store projections in CPU memory in chunks to avoid GPU OOM
    print(f"  Streaming Gram matrix (n={n}) in float64...")

    # Project all gradients to CPU storage (float32 for storage, cast to float64 for ops)
    pg_storage = []
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        g_batch = grad_cache.get_batch(i_start, i_end, device=device)
        pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
        pg_storage.append(pg_batch.cpu())  # Store on CPU in float32

        # Compute U for this batch: U[i] = pg_i @ PV.T
        pg_batch_f64 = pg_batch.to(dtype=torch.float64)
        U[i_start:i_end] = pg_batch_f64 @ PV.T
        del pg_batch, pg_batch_f64

    # Clean up PV to free GPU memory
    del PV
    torch.cuda.empty_cache()

    # Second pass: compute K from stored projections
    for i, i_start in enumerate(range(0, n, batch_size)):
        i_end = min(i_start + batch_size, n)
        pg_i = pg_storage[i].to(device=device, dtype=torch.float64)

        for j, j_start in enumerate(range(0, n, batch_size)):
            j_end = min(j_start + batch_size, n)

            if j < i:
                # Already computed by symmetry
                continue
            elif i == j:
                # Diagonal block
                K[i_start:i_end, j_start:j_end] = pg_i @ pg_i.T
            else:
                # Off-diagonal block
                pg_j = pg_storage[j].to(device=device, dtype=torch.float64)
                block = pg_i @ pg_j.T
                K[i_start:i_end, j_start:j_end] = block
                K[j_start:j_end, i_start:i_end] = block.T  # Symmetry
                del pg_j

        del pg_i

    # Clean up storage
    del pg_storage
    torch.cuda.empty_cache()

    # Step 4: Normalize
    K.div_(n)
    U.div_(n ** 0.5)

    # Step 5: Eigendecomposition on the (n x n) kernel matrix
    # eigh is numerically stable for symmetric matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(K)

    # Step 6: Enforce PSD by clamping tiny negative eigenvalues to 0
    # In float64, anything < -1e-12 * max_eigenvalue is just numerical noise
    threshold = -1e-12 * eigenvalues.abs().max()
    n_negative = (eigenvalues < threshold).sum().item()
    if n_negative > 0:
        print(f"  Warning: {n_negative} eigenvalues below threshold, clamping to 0")
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    # Flip to descending order (eigh returns ascending)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    return WoodburyPrecomputed(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        U=U,
        pv_norms_sq=pv_norms_sq,
        n=n,
        K_proj=K if store_k_proj else None,
    )


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    grad_cache: GradientCache,
    lambda_values: list,
    m_values: list,
    proj_type: ProjectionType = ProjectionType.normal,
    num_trials: int = 5,
    num_test_grads: int = 200,
    seed: int = 42,
    device: str = "cuda",
    batch_size: int = 32,
    min_d_lambda: float = 5.0,
    test_mode: str = "self",
    test_grad_cache: GradientCache = None,
) -> dict:
    """
    Run spectrum bounds experiment to validate sketched influence approximation.

    Uses fixed m values grid and fast eigendecomposition-based λ sweep for efficiency.
    For each (m, trial), we project gradients and eigendecompose ONCE, then sweep
    all λ values in O(n²) per λ instead of O(n³).

    Supports two modes:
    - "self": Self-influence scores φ_λ(g_i, g_i) using training gradients only.
              Error is computed over diagonal self-scores.
    - "test": Cross-influence scores φ_λ(g_train, g_test) using test set gradients.
              Computes full (n_train, k_test) score matrix and aggregates error.

    Args:
        grad_cache: GradientCache containing training gradients (forms the Fisher matrix)
        lambda_values: List of regularization values
        m_values: Fixed list of projection dimensions to test
        proj_type: Type of random projection
        num_trials: Number of random projection trials per configuration
        num_test_grads: Number of gradients to use for testing (from train or test set)
        seed: Random seed
        device: GPU device
        batch_size: Batch size for GPU operations
        min_d_lambda: Skip λ values where d_λ < this threshold
        test_mode: "self" for self-influence, "test" for cross-influence with test set
        test_grad_cache: GradientCache for test gradients (required if test_mode="test")

    Returns:
        Dictionary with all results
    """
    # Validate test_mode
    if test_mode not in ("self", "test"):
        raise ValueError(f"test_mode must be 'self' or 'test', got '{test_mode}'")
    if test_mode == "test" and test_grad_cache is None:
        raise ValueError("test_grad_cache is required when test_mode='test'")

    n = grad_cache.n_samples
    d = grad_cache.dim

    print(f"\nRunning experiment:")
    print(f"  - test_mode = {test_mode}")
    print(f"  - batch_size = {batch_size}")
    print(f"  - n = {n}, d = {d:,}")
    print(f"  - {len(m_values)} m values × {num_trials} trials × {len(lambda_values)} λ")

    # Filter out m values that are >= parameter dimension (no projection needed)
    # Random projection is only meaningful when projecting to lower dimensions
    original_m_values = m_values
    m_values = [m for m in m_values if m < d]
    if len(m_values) < len(original_m_values):
        print(f"  - Filtered m values: {len(original_m_values)} -> {len(m_values)} "
              f"(removed m >= d={d})")
        if not m_values:
            raise ValueError(f"No valid m values remaining after filtering (all >= d={d})")
    print(f"  - Testing {len(m_values)} valid m values (m < d={d}): {m_values[:5]}...{m_values[-3:] if len(m_values) > 5 else []}")

    # Compute eigenvalues, effective dimensions, and Gram matrix in FLOAT64
    # We always need eigenvectors now for fast exact score computation
    print(f"\nComputing eigenvalues and Gram matrix (Robust Float64)...")
    eigenvalues, eff_dims, K, eigenvectors = compute_robust_gram(
        grad_cache, lambda_values, device=device, batch_size=batch_size*4,
        return_eigenvectors=True,  # Always need eigenvectors for fast λ sweep
    )

    eig_nonzero = eigenvalues[eigenvalues > 1e-10]
    print(f"  Eigenvalue stats: max={eigenvalues[0]:.2e}, min_nonzero={eig_nonzero[-1]:.2e}, "
          f"n_nonzero={len(eig_nonzero)}/{len(eigenvalues)}")

    # Compute numerical rank
    threshold = 1e-10 * eigenvalues[0]
    rank = (eigenvalues > threshold).sum().item()

    # Prepare test vectors based on mode
    if test_mode == "self":
        k_test = min(num_test_grads, n)
        test_vectors = [grad_cache.get_sample(i, device="cpu") for i in range(k_test)]
        V = torch.stack(test_vectors).to(device)  # (k, d)
        print(f"  - Self-influence mode: using {k_test} training gradients as test vectors")
    else:
        k_test = min(num_test_grads, test_grad_cache.n_samples)
        test_vectors = [test_grad_cache.get_sample(i, device="cpu") for i in range(k_test)]
        V = torch.stack(test_vectors).to(device)  # (k, d)
        print(f"  - Test mode: using {k_test} test gradients as test vectors")

    # Filter valid λ values (d_λ >= min_d_lambda)
    valid_lambdas = [lamb for lamb in lambda_values if eff_dims[lamb] >= min_d_lambda]
    skipped_lambdas = [lamb for lamb in lambda_values if eff_dims[lamb] < min_d_lambda]

    if skipped_lambdas:
        print(f"\nSkipping {len(skipped_lambdas)} λ values with d_λ < {min_d_lambda}")

    # =========================================================================
    # OPTIMIZATION: Precompute exact scores using eigendecomposition for fast λ sweep
    # Instead of O(n³) per λ, we do O(n³) once + O(n²) per λ
    # =========================================================================
    print(f"\nPrecomputing exact scores using eigendecomposition (fast λ sweep)...")

    # Compute U = G @ V^T once (used for all λ)
    V_dbl = V.to(dtype=torch.float64, device=device)
    U_exact = torch.zeros(n, k_test, device=device, dtype=torch.float64)
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        G_batch = grad_cache.get_batch(i_start, i_end, device=device)
        U_exact[i_start:i_end, :] = G_batch.to(dtype=torch.float64) @ V_dbl.T

    # For self mode, also need ||g_i||² for each test sample
    if test_mode == "self":
        g_norms_sq = (V_dbl ** 2).sum(dim=1)  # (k,)
        # U_self = (1/√n) G @ V^T for self-influence Woodbury
        U_self = U_exact / (n ** 0.5)

    # Precompute W = eigenvectors^T @ U for fast eigenvalue-based solve
    # For self mode: W_self = eigenvectors^T @ U_self
    # For test mode: W_test = eigenvectors^T @ U_exact
    if test_mode == "self":
        W_exact = eigenvectors.T @ U_self  # (n, k)
    else:
        W_exact = eigenvectors.T @ U_exact  # (n, k)

    # Compute φ₀ for test mode (doesn't depend on λ)
    train_phi0 = None
    test_phi0 = None
    if test_mode == "test":
        print("  Computing unregularized self-influence φ₀...")
        train_phi0, test_phi0 = compute_unregularized_self_influence(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            U_test=U_exact,
            n=n,
        )
        # Move to CPU for later error computation
        train_phi0 = train_phi0.cpu()
        test_phi0 = test_phi0.cpu()

    # Compute exact scores for all λ using eigendecomposition (FAST!)
    exact_scores_by_lambda = {}
    print(f"  Computing exact scores for {len(valid_lambdas)} λ values...")

    for lamb in valid_lambdas:
        inv_eigvals = 1.0 / (eigenvalues + lamb)

        if test_mode == "self":
            # Self-influence: score_i = (1/λ)[||g_i||² - u_i^T (K + λI)^{-1} u_i]
            # where u_i = (1/√n) G @ g_i and K = (1/n) G @ G^T
            # Using eigendecomposition: X = eigenvectors @ (inv_eigvals * W)
            X = eigenvectors @ (inv_eigvals.unsqueeze(1) * W_exact)  # (n, k)
            ux_dots = (U_self * X).sum(dim=0)  # (k,)
            scores = (g_norms_sq - ux_dots) / lamb
            exact_scores_by_lambda[lamb] = scores.cpu()
        else:
            # Test mode: B_λ(g_i, v_j) = (1/λ)[g_i · v_j - (1/n) u_i^T (K + λI)^{-1} w_j]
            # where u_i = G @ g_i = n * K[:, i], w_j = G @ v_j = U_exact[:, j]
            # X_B = (K + λI)^{-1} @ (U_exact / n)
            X_B = eigenvectors @ (inv_eigvals.unsqueeze(1) * W_exact) / n  # (n, k)
            # Cross term: n * K @ X_B
            K_dbl = K.to(dtype=torch.float64, device=device)
            cross_term = n * (K_dbl @ X_B)
            scores = (U_exact - cross_term) / lamb  # G_dot_V = U_exact
            exact_scores_by_lambda[lamb] = scores.cpu()

    print(f"  Done. Exact scores cached for all λ values.")

    # =========================================================================
    # Initialize per_trial_stats for all (m, λ) combinations
    # =========================================================================
    per_trial_stats = {}
    for m in m_values:
        for lamb in valid_lambdas:
            per_trial_stats[(m, lamb)] = {}

    # =========================================================================
    # Main loop: trial → m → λ sweep (FAST: eigendecompose once per m, then sweep λ)
    # =========================================================================
    print(f"\nRunning {num_trials} trials × {len(m_values)} m values...")

    # Iterate from largest m first (most memory intensive) to fail fast if OOM
    m_values_desc = sorted(m_values, reverse=True)

    for trial in range(num_trials):
        trial_seed = seed + trial
        print(f"\nTrial {trial + 1}/{num_trials} (seed={trial_seed})")

        for m in tqdm(m_values_desc, desc="  projection dimension"):
            # When m >= d, safe_project will use identity (no projection) as baseline

            # Create projector for this (m, trial)
            projector = make_random_projector(
                param_shape_list=[d],
                feature_batch_size=batch_size,
                proj_dim=m,
                proj_max_batch_size=128,
                device=torch.device(device),
                proj_seed=trial_seed,
                proj_type=proj_type,
                dtype=torch.float32,
            )

            # Precompute Woodbury quantities ONCE per (m, trial)
            # This includes eigendecomposition of projected Gram matrix
            precomputed = precompute_woodbury(
                grad_cache=grad_cache,
                projector=projector,
                test_vectors=V,
                device=device,
                batch_size=batch_size,
                store_k_proj=(test_mode == "test"),
            )

            # Precompute W_sketched for fast λ sweep
            W_sketched = precomputed.eigenvectors.T @ precomputed.U  # (n, k)

            # Sweep all λ values CHEAPLY using eigendecomposition
            for lamb in valid_lambdas:
                inv_eigvals = 1.0 / (precomputed.eigenvalues + lamb)

                if test_mode == "self":
                    # Sketched self-influence using eigendecomposition
                    X = precomputed.eigenvectors @ (inv_eigvals.unsqueeze(1) * W_sketched)
                    ux_dots = (precomputed.U * X).sum(dim=0)  # (k,)
                    sketched_scores = (precomputed.pv_norms_sq - ux_dots) / lamb

                    # Compute error metrics
                    exact_scores = exact_scores_by_lambda[lamb]
                    sketched_scores_cpu = sketched_scores.cpu()

                    # Filter valid scores
                    max_score = exact_scores.abs().max()
                    relative_threshold = max(1e-10 * max_score.item(), 1e-12)
                    valid_mask = exact_scores.abs() > relative_threshold

                    ratios = torch.zeros_like(exact_scores)
                    ratios[valid_mask] = sketched_scores_cpu[valid_mask] / exact_scores[valid_mask]
                    valid_ratios = ratios[valid_mask].numpy()

                    eps_values = np.abs(valid_ratios - 1)
                    per_trial_stats[(m, lamb)][trial] = {
                        "mean_ratio": np.mean(valid_ratios),
                        "std_ratio": np.std(valid_ratios),
                        "min_ratio": np.min(valid_ratios),
                        "max_ratio": np.max(valid_ratios),
                        "eps_mean": np.mean(eps_values),
                        "eps_max": np.max(eps_values),
                        "eps_95": np.percentile(eps_values, 95),
                        "eps_99": np.percentile(eps_values, 99),
                        "n_samples": len(valid_ratios),
                    }
                else:
                    # Sketched cross-scores using eigendecomposition
                    X = precomputed.eigenvectors @ (inv_eigvals.unsqueeze(1) * W_sketched)
                    pg_dot_pv = (n ** 0.5) * precomputed.U
                    cross_term = (n ** 0.5) * (precomputed.K_proj @ X)
                    sketched_scores = (pg_dot_pv - cross_term) / lamb

                    # Compute normalized error (all on CPU)
                    exact_scores = exact_scores_by_lambda[lamb]
                    sketched_scores_cpu = sketched_scores.cpu()

                    # Compute normalizer: sqrt(φ₀(g_i)) * sqrt(φ₀(v_j)) - already on CPU
                    sqrt_train = torch.sqrt(torch.clamp(train_phi0, min=1e-12)).unsqueeze(1)
                    sqrt_test = torch.sqrt(torch.clamp(test_phi0, min=1e-12)).unsqueeze(0)
                    normalizer = sqrt_train * sqrt_test

                    additive_error = (sketched_scores_cpu - exact_scores).abs()
                    valid_mask = normalizer > 1e-12
                    epsilon = torch.zeros_like(additive_error)
                    epsilon[valid_mask] = additive_error[valid_mask] / normalizer[valid_mask]

                    valid_epsilon = epsilon[valid_mask].numpy()
                    per_trial_stats[(m, lamb)][trial] = {
                        "eps_mean": np.mean(valid_epsilon),
                        "eps_max": np.max(valid_epsilon),
                        "eps_median": np.median(valid_epsilon),
                        "eps_95": np.percentile(valid_epsilon, 95),
                        "eps_99": np.percentile(valid_epsilon, 99),
                        "n_samples": len(valid_epsilon),
                    }

            # Explicit memory cleanup after processing each m value
            del precomputed, W_sketched
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =========================================================================
    # Build results dict with metadata and aggregated experiment results
    # =========================================================================
    print("Aggregating results...")
    results = {
        "n_samples": n,
        "d_params": d,
        "proj_type": proj_type.value,
        "lambda_values": lambda_values,
        "m_values": m_values,
        "effective_dims": eff_dims,
        "batch_size": batch_size,
        "min_d_lambda": min_d_lambda,
        "num_trials": num_trials,
        "skipped_configs": [
            {"lambda": lamb, "d_lambda": eff_dims[lamb],
             "reason": f"d_lambda ({eff_dims[lamb]:.2f}) < min_d_lambda ({min_d_lambda})"}
            for lamb in skipped_lambdas
        ],
        "test_mode": test_mode,
        "num_test_grads": k_test,
        "eigenvalues": eigenvalues.cpu().numpy(),
        "rank": rank,
        "experiments": [],
    }

    for m in m_values:
        for lamb in valid_lambdas:
            trial_stats = per_trial_stats[(m, lamb)]
            if len(trial_stats) == 0:
                continue

            trial_list = list(trial_stats.values())
            n_trials_actual = len(trial_list)
            d_lambda = eff_dims[lamb]

            if test_mode == "self":
                trial_mean_ratios = np.array([t["mean_ratio"] for t in trial_list])
                trial_eps_means = np.array([t["eps_mean"] for t in trial_list])
                trial_eps_maxs = np.array([t["eps_max"] for t in trial_list])
                trial_eps_95s = np.array([t["eps_95"] for t in trial_list])
                trial_eps_99s = np.array([t["eps_99"] for t in trial_list])

                results["experiments"].append({
                    "m": m,
                    "lambda": lamb,
                    "d_lambda": d_lambda,
                    "m_over_d_lambda": m / d_lambda if d_lambda > 0 else float('inf'),
                    "n_trials": n_trials_actual,
                    "per_trial": trial_stats,
                    "mean_ratio": np.mean(trial_mean_ratios),
                    "mean_ratio_std": np.std(trial_mean_ratios, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_mean": np.mean(trial_eps_means),
                    "empirical_eps_mean_std": np.std(trial_eps_means, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_max": np.mean(trial_eps_maxs),
                    "empirical_eps_max_std": np.std(trial_eps_maxs, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_95": np.mean(trial_eps_95s),
                    "empirical_eps_95_std": np.std(trial_eps_95s, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_99": np.mean(trial_eps_99s),
                    "empirical_eps_99_std": np.std(trial_eps_99s, ddof=1) if n_trials_actual > 1 else 0.0,
                })
            else:
                trial_eps_means = np.array([t["eps_mean"] for t in trial_list])
                trial_eps_maxs = np.array([t["eps_max"] for t in trial_list])
                trial_eps_medians = np.array([t["eps_median"] for t in trial_list])
                trial_eps_95s = np.array([t["eps_95"] for t in trial_list])
                trial_eps_99s = np.array([t["eps_99"] for t in trial_list])

                results["experiments"].append({
                    "m": m,
                    "lambda": lamb,
                    "d_lambda": d_lambda,
                    "m_over_d_lambda": m / d_lambda if d_lambda > 0 else float('inf'),
                    "n_trials": n_trials_actual,
                    "per_trial": trial_stats,
                    "empirical_eps_mean": np.mean(trial_eps_means),
                    "empirical_eps_mean_std": np.std(trial_eps_means, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_max": np.mean(trial_eps_maxs),
                    "empirical_eps_max_std": np.std(trial_eps_maxs, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_median": np.mean(trial_eps_medians),
                    "empirical_eps_median_std": np.std(trial_eps_medians, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_95": np.mean(trial_eps_95s),
                    "empirical_eps_95_std": np.std(trial_eps_95s, ddof=1) if n_trials_actual > 1 else 0.0,
                    "empirical_eps_99": np.mean(trial_eps_99s),
                    "empirical_eps_99_std": np.std(trial_eps_99s, ddof=1) if n_trials_actual > 1 else 0.0,
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Practical validation of Theorem 2 for regularized sketching"
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar2", "maestro"])
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["lr", "mlp", "resnet9", "musictransformer"])
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per configuration")
    parser.add_argument("--proj_type", type=str, default="normal",
                       choices=["normal", "rademacher", "sjlt"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for GPU operations. Increase for better GPU utilization, "
                            "decrease if running out of memory. Suggested: 50-200 for d~1M, "
                            "10-50 for d~100M.")

    # Gradient offloading options
    parser.add_argument("--offload", type=str, default="cpu",
                       choices=["none", "cpu", "disk"],
                       help="Gradient storage: none (GPU), cpu (RAM), disk (files). "
                            "Use 'disk' for very large models (>10M parameters).")
    parser.add_argument("--cache_dir", type=str, default="./grad_cache",
                       help="Directory for gradient cache (only used with --offload disk)")

    # Numerical stability options
    parser.add_argument("--min_d_lambda", type=float, default=1.0,
                       help="Skip λ values where d_λ < this threshold (numerically degenerate). "
                            "(default: 1.0)")

    # Test mode options
    parser.add_argument("--test_mode", type=str, default="self",
                       choices=["self", "test"],
                       help="Test mode: 'self' for self-influence (diagonal scores), "
                            "'test' for cross-influence with test set gradients.")
    parser.add_argument("--num_test_grads", type=int, default=500,
                       help="Number of test gradients to use (from train set for 'self', "
                            "from test set for 'test').")

    # Lambda sweep configuration (integer powers of 10, or log-spaced if lambda_steps specified)
    parser.add_argument("--lambda_exp_min", type=int, default=-8,
                       help="Minimum exponent for λ sweep: λ_min = 10^exp_min (default: -8 → 1e-8)")
    parser.add_argument("--lambda_exp_max", type=int, default=2,
                       help="Maximum exponent for λ sweep: λ_max = 10^exp_max (default: 2 → 1e2)")
    parser.add_argument("--lambda_steps", type=int, default=0,
                       help="Number of log-spaced λ values. If 0 (default), uses integer powers of 10.")

    # Projection dimension grid (powers of 2, like lambda uses powers of 10)
    parser.add_argument("--m_exp_min", type=int, default=2,
                       help="Minimum exponent for m sweep: m_min = 2^exp_min (default: 2 → 4)")
    parser.add_argument("--m_exp_max", type=int, default=20,
                       help="Maximum exponent for m sweep: m_max = 2^exp_max (default: 20 → 1048576)")
    parser.add_argument("--m_steps", type=int, default=0,
                       help="Number of log2-spaced m values. If 0 (default), uses integer powers of 2.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    proj_type = ProjectionType(args.proj_type)

    print("\n" + "="*60)
    print(f"Offload Mode: {args.offload.upper()}")
    print("="*60)
    if args.offload == "disk":
        print("- Gradients will be cached to disk")
        print(f"- Cache directory: {args.cache_dir}")
    elif args.offload == "cpu":
        print("- Gradients will be stored in CPU RAM")
    else:
        print("- Gradients will be stored on GPU")
    print(f"- Batch size: {args.batch_size}")
    print("="*60 + "\n")

    # Create GradientCache with appropriate offload mode
    grad_cache = GradientCache(
        offload=args.offload,
        cache_dir=args.cache_dir if args.offload == "disk" else None,
    )

    # Load benchmark to get training data
    print(f"\nLoading {args.model} on {args.dataset}...")

    model_details, _ = load_benchmark(model=args.model, dataset=args.dataset, metric="lds")

    # CRITICAL FIX: Load the checkpoint that corresponds to the ground truth
    # This ensures d_λ values are consistent with hyperparam_selection.py
    nn_model = model_details["model"]
    checkpoint = torch.load(model_details["models_full"][0], map_location=args.device)
    nn_model.load_state_dict(checkpoint)
    nn_model.to(args.device)
    nn_model.eval()
    print(f"Loaded checkpoint: {model_details['models_full'][0]}")

    train_dataset = model_details["train_dataset"]
    train_sampler = model_details["train_sampler"]
    indices = list(train_sampler)
    n_samples = len(indices)

    # For disk mode, check if cache is valid and reuse
    # Note: is_valid() already loads metadata when it returns True
    train_cache_dir = f"{args.cache_dir}/train" if args.offload == "disk" else None
    if args.offload == "disk":
        grad_cache = GradientCache(offload="disk", cache_dir=train_cache_dir)
    if args.offload == "disk" and grad_cache.is_valid(expected_samples=n_samples):
        print(f"Loading cached training gradients from {train_cache_dir}...")
    else:
        # Need to compute and cache gradients
        model_type = "musictransformer" if args.model == "musictransformer" else "default"
        grad_cache.cache(
            model=nn_model,
            dataset=train_dataset,
            indices=indices,
            device=args.device,
            model_type=model_type,
            batch_size=args.batch_size,
        )

    # For test mode, also compute test gradients
    test_grad_cache = None
    if args.test_mode == "test":
        test_dataset = model_details["test_dataset"]
        test_sampler = model_details.get("test_sampler", None)
        if test_sampler is not None:
            test_indices = list(test_sampler)
        else:
            # Use all test samples if no sampler specified
            test_indices = list(range(len(test_dataset)))

        # Limit to num_test_grads
        test_indices = test_indices[:args.num_test_grads]
        n_test = len(test_indices)

        print(f"\nLoading test gradients for test mode (n_test={n_test})...")

        test_cache_dir = f"{args.cache_dir}/test" if args.offload == "disk" else None
        test_grad_cache = GradientCache(
            offload=args.offload,
            cache_dir=test_cache_dir,
        )

        if args.offload == "disk" and test_grad_cache.is_valid(expected_samples=n_test):
            print(f"Loading cached test gradients from {test_cache_dir}...")
        else:
            model_type = "musictransformer" if args.model == "musictransformer" else "default"
            test_grad_cache.cache(
                model=nn_model,
                dataset=test_dataset,
                indices=test_indices,
                device=args.device,
                model_type=model_type,
                batch_size=args.batch_size,
            )

    # Experiment configuration
    # Generate λ values: integer powers of 10 if lambda_steps=0, else log-spaced
    if args.lambda_steps == 0:
        lambda_values = [10.0 ** exp for exp in range(args.lambda_exp_min, args.lambda_exp_max + 1)]
        print(f"\nλ sweep configuration:")
        print(f"  Exponent range: [{args.lambda_exp_min}, {args.lambda_exp_max}]")
        print(f"  λ values: {[f'1e{exp}' for exp in range(args.lambda_exp_min, args.lambda_exp_max + 1)]}")
    else:
        lambda_values = np.logspace(args.lambda_exp_min, args.lambda_exp_max, num=args.lambda_steps).tolist()
        print(f"\nλ sweep configuration:")
        print(f"  Range: [1e{args.lambda_exp_min}, 1e{args.lambda_exp_max}] ({args.lambda_steps} log-spaced steps)")

    # Generate m values: integer powers of 2 if m_steps=0, else log2-spaced
    if args.m_steps == 0:
        m_values = [2 ** exp for exp in range(args.m_exp_min, args.m_exp_max + 1)]
        print(f"\nm sweep configuration:")
        print(f"  Exponent range: [{args.m_exp_min}, {args.m_exp_max}]")
        print(f"  m values: {[f'2^{exp}' for exp in range(args.m_exp_min, args.m_exp_max + 1)]}")
        print(f"  m values (decimal): {m_values}")
    else:
        m_values = np.unique(np.logspace(
            args.m_exp_min * np.log10(2),  # Convert base-2 exponents to base-10
            args.m_exp_max * np.log10(2),
            num=args.m_steps,
            base=10
        ).astype(int)).tolist()
        print(f"\nm sweep configuration:")
        print(f"  Range: [2^{args.m_exp_min}, 2^{args.m_exp_max}] ({args.m_steps} log2-spaced steps)")
        print(f"  m values: {m_values}")

    # Run experiment
    results = run_experiment(
        grad_cache=grad_cache,
        lambda_values=lambda_values,
        m_values=m_values,
        proj_type=proj_type,
        num_trials=args.num_trials,
        num_test_grads=args.num_test_grads,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        min_d_lambda=args.min_d_lambda,
        test_mode=args.test_mode,
        test_grad_cache=test_grad_cache,
    )

    # Add metadata
    results["dataset"] = args.dataset
    results["model"] = args.model
    results["offload_mode"] = args.offload
    results["batch_size"] = args.batch_size

    # Save results with organized directory structure
    experiment_dir = os.path.join(args.output_dir, "spectrum_bounds")
    os.makedirs(experiment_dir, exist_ok=True)

    # Build filename with key settings
    results_filename = f"{args.dataset}_{args.model}_{args.proj_type}_{args.test_mode}.pt"
    results_path = os.path.join(experiment_dir, results_filename)
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

"""
Faithfulness-Utility Alignment Experiment

Investigates the relationship between empirical faithfulness (measured by comparing
exact vs sketched influence scores) and downstream utility (measured by LDS).

Key Questions:
1. Alignment: For fixed m, does λ*(m) = argmax LDS(m, λ) fall in the empirically
   "unfaithful" region where sketched scores deviate significantly from exact scores?
2. Monotonicity: Does the optimal utility LDS(m, λ*(m)) increase with m?

Faithfulness is measured empirically by computing the ratio of sketched to exact
influence scores, avoiding reliance on theoretical bounds with unknown constants.
"""

import argparse
import gc
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from dattri.benchmark.load import load_benchmark
from dattri.func.projection import make_random_projector, ProjectionType

from utils.fisher import (
    compute_eigenspectrum,
    safe_project,
    compute_unregularized_self_influence,
    compute_effective_dimension,
)
from utils.gradient_cache import GradientCache, create_gradient_cache
from utils.metrics import lds
from utils.data import get_validation_split_indices


# =============================================================================
# Exact Score Computation (for faithfulness measurement)
# =============================================================================

def compute_exact_gram_matrix(
    grad_cache: GradientCache,
    device: str = "cuda",
    batch_size: int = 64,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute exact Gram matrix K = (1/n) G @ G^T in float64.

    Args:
        grad_cache: GradientCache containing the training gradients
        device: GPU device
        batch_size: Batch size for streaming

    Returns:
        eigenvalues: Eigenvalues of K in descending order (float64)
        eigenvectors: Eigenvectors of K, columns in descending order (float64)
        K: Gram matrix (n, n) in float64, normalized by 1/n (ON CPU)
    """
    n = grad_cache.n_samples
    print(f"  Computing Exact Gram Matrix (n={n}) in float64...")
    print(f"  Allocating K on CPU to save GPU memory...")

    # Allocate K on CPU to avoid GPU OOM issues
    K = torch.zeros(n, n, dtype=torch.float64, device="cpu")

    for i_start in tqdm(range(0, n, batch_size), desc="  Gram matrix", leave=False):
        i_end = min(i_start + batch_size, n)
        G_i = grad_cache.get_batch(i_start, i_end, device=device).to(dtype=torch.float64)

        for j_start in range(i_start, n, batch_size):
            j_end = min(j_start + batch_size, n)

            if i_start == j_start:
                G_j = G_i
            else:
                G_j = grad_cache.get_batch(j_start, j_end, device=device).to(dtype=torch.float64)

            # Compute block on GPU, then move to CPU to save GPU memory
            block = (G_i @ G_j.T).cpu()
            K[i_start:i_end, j_start:j_end] = block

            if i_start != j_start:
                K[j_start:j_end, i_start:i_end] = block.T

            # Free GPU memory for the block
            del block
            if i_start != j_start:
                del G_j

        del G_i

    K.div_(n)

    # Compute full eigendecomposition on CPU (K is already on CPU)
    print(f"  Computing eigendecomposition on CPU...")
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
    eigenvectors = eigenvectors.flip(1)

    # Clear GPU cache to free fragmented memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eigenvalues, eigenvectors, K


def compute_exact_bilinear_scores(
    train_grad_cache: GradientCache,
    test_grads: Tensor,
    K: Tensor,
    eigenvalues: Tensor,
    eigenvectors: Tensor,
    lambda_values: List[float],
    device: str = "cuda",
    batch_size: int = 64,
) -> Tuple[Dict[float, Tensor], Tensor, Tensor, Tensor]:
    """
    Compute exact bilinear form scores B_λ(g_train, g_test) = g_train^T (F + λI)^{-1} g_test.

    Also returns φ₀ for both training and test samples for error normalization.

    Args:
        train_grad_cache: GradientCache containing training gradients
        test_grads: Test gradient vectors (k, d) on device
        K: Pre-computed Gram matrix (n, n), normalized by 1/n, in float64
        eigenvalues: Eigenvalues of K (n,)
        eigenvectors: Eigenvectors of K (n, n)
        lambda_values: List of λ values
        device: GPU device
        batch_size: Batch size for streaming

    Returns:
        exact_scores: Dictionary mapping λ -> exact bilinear scores (n, k)
        U_B: G @ V^T matrix (n, k)
        train_phi0: Unregularized self-influence for training samples (n,)
        test_phi0: Unregularized self-influence for test samples (k,)
    """
    n = train_grad_cache.n_samples
    k = test_grads.shape[0]

    print(f"  Computing exact bilinear scores for {n} train × {k} test samples...")

    # Compute U_B = G @ V^T (n, k)
    V_dbl = test_grads.to(dtype=torch.float64, device=device)
    U_B = torch.zeros(n, k, device=device, dtype=torch.float64)
    for i_start in tqdm(range(0, n, batch_size), desc="  Computing G @ V^T", leave=False):
        i_end = min(i_start + batch_size, n)
        G_batch = train_grad_cache.get_batch(i_start, i_end, device=device)
        U_B[i_start:i_end, :] = G_batch.to(dtype=torch.float64) @ V_dbl.T

    # Move eigenvectors and eigenvalues to GPU for efficient computation
    # (K eigendecomposition was done on CPU to save GPU memory, but now we need GPU for speed)
    eigenvectors_gpu = eigenvectors.to(device)
    eigenvalues_gpu = eigenvalues.to(device)

    # Compute φ₀ for training and test samples
    print(f"  Computing φ₀ for error normalization...")
    train_phi0, test_phi0 = compute_unregularized_self_influence(
        eigenvalues=eigenvalues_gpu,
        eigenvectors=eigenvectors_gpu,
        U_test=U_B,
        n=n,
    )

    # Compute exact bilinear scores for each λ using Woodbury identity
    # B_λ(g_i, v_j) = (1/λ)[g_i · v_j - (1/n) u_i^T (K + λI)^{-1} w_j]
    # where u_i = G @ g_i = n * K[:, i], w_j = G @ v_j = U_B[:, j]
    G_dot_V = U_B  # g_i · v_j = (G @ V^T)[i, j]

    # Move K to GPU for fast λ sweep computations
    # (eigenvectors_gpu and eigenvalues_gpu already moved above)
    K_gpu = K.to(device)

    # Precompute W = eigenvectors^T @ U_B for fast λ sweep
    W = eigenvectors_gpu.T @ U_B  # (n, k)

    exact_scores = {}
    for lamb in lambda_values:
        inv_eigvals = 1.0 / (eigenvalues_gpu + lamb)
        # X_B = (K + λI)^{-1} @ (U_B / n)
        X_B = eigenvectors_gpu @ (inv_eigvals.unsqueeze(1) * W) / n  # (n, k)

        # Cross term: n * K @ X_B = n * K @ (K + λI)^{-1} @ (U_B / n) = K @ (K + λI)^{-1} @ U_B
        cross_term = n * (K_gpu @ X_B)  # (n, k)

        # B_λ(g_i, v_j) = (g_i · v_j - cross_term[i, j]) / λ
        scores = (G_dot_V - cross_term) / lamb
        exact_scores[lamb] = scores.cpu()

    # Free GPU memory after all λ computations
    del K_gpu, eigenvectors_gpu, eigenvalues_gpu, W, X_B, cross_term, G_dot_V
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return exact_scores, U_B, train_phi0.cpu(), test_phi0.cpu()


# =============================================================================
# Sketched Score Computation with Faithfulness Measurement
# =============================================================================

def compute_sketched_bilinear_scores(
    train_grad_cache: GradientCache,
    test_grads: Tensor,
    projector,
    lambda_values: List[float],
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[float, Tensor]:
    """
    Compute sketched bilinear form scores B̃_λ(g_train, g_test).

    Args:
        train_grad_cache: GradientCache containing training gradients
        test_grads: Test gradient vectors (k, d), can be on CPU or GPU
        projector: Projector with .project() method
        lambda_values: List of λ values
        device: GPU device
        batch_size: Batch size

    Returns:
        Dictionary mapping λ -> sketched bilinear scores (n, k)
    """
    n = train_grad_cache.n_samples
    k = test_grads.shape[0]
    m = projector.proj_dim
    d = train_grad_cache.dim

    # Project test gradients in batches to avoid CUDA memory issues with SJLT
    # CRITICAL FIX: Projecting all test grads at once causes illegal memory access with SJLT
    pv_batches = []
    proj_batch_size = batch_size  # Use same batch size as training projection

    for i_start in range(0, k, proj_batch_size):
        i_end = min(i_start + proj_batch_size, k)
        test_batch = test_grads[i_start:i_end].to(device)
        pv_batch = safe_project(test_batch, projector, d, ensemble_id=0).to(dtype=torch.float64, device=device)
        pv_batches.append(pv_batch)
        del test_batch

    PV = torch.cat(pv_batches, dim=0)  # (k, m)
    del pv_batches

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build projected Gram matrix K_proj and U = PG @ PV^T
    K_proj = torch.zeros(n, n, dtype=torch.float64, device=device)
    U = torch.zeros(n, k, dtype=torch.float64, device=device)

    # Project all training gradients
    pg_storage = []
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        g_batch = train_grad_cache.get_batch(i_start, i_end, device=device)
        pg_batch = safe_project(g_batch, projector, d, ensemble_id=0).to(dtype=torch.float64)
        pg_storage.append(pg_batch.cpu())
        U[i_start:i_end] = pg_batch @ PV.T

        # Explicitly free batch tensors
        del g_batch, pg_batch

    # Build K_proj from stored projections
    for i, i_start in enumerate(range(0, n, batch_size)):
        i_end = min(i_start + batch_size, n)
        pg_i = pg_storage[i].to(device=device, dtype=torch.float64)

        for j, j_start in enumerate(range(0, n, batch_size)):
            j_end = min(j_start + batch_size, n)
            if j < i:
                continue
            elif i == j:
                K_proj[i_start:i_end, j_start:j_end] = pg_i @ pg_i.T
            else:
                pg_j = pg_storage[j].to(device=device, dtype=torch.float64)
                block = pg_i @ pg_j.T
                K_proj[i_start:i_end, j_start:j_end] = block
                K_proj[j_start:j_end, i_start:i_end] = block.T
                del pg_j, block

        del pg_i

    del pg_storage

    # Normalize
    K_proj.div_(n)
    U.div_(n ** 0.5)

    # PG[i] · PV[j] = √n * U[i,j]
    pg_dot_pv = (n ** 0.5) * U

    # Eigendecompose for fast λ sweep
    eigenvalues, eigenvectors = torch.linalg.eigh(K_proj)
    eigenvalues = eigenvalues.clamp(min=0.0)
    W = eigenvectors.T @ U  # (n, k)

    # Compute sketched scores for each λ
    sketched_scores = {}
    for lamb in lambda_values:
        inv_eigvals = 1.0 / (eigenvalues + lamb)
        X = eigenvectors @ (inv_eigvals.unsqueeze(1) * W)  # (n, k)

        # Cross term = √n * K_proj @ X
        cross_term = (n ** 0.5) * (K_proj @ X)

        scores = (pg_dot_pv - cross_term) / lamb
        sketched_scores[lamb] = scores.cpu()

    # Explicitly delete large GPU tensors
    del PV, K_proj, U, pg_dot_pv, eigenvalues, eigenvectors, W

    # Clean up GPU tensors and synchronize before returning
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return sketched_scores


def compute_bilinear_faithfulness_metrics(
    exact_scores: Tensor,
    sketched_scores: Tensor,
    train_phi0: Tensor,
    test_phi0: Tensor,
    threshold: float = 0.1,
) -> Dict:
    """
    Compute empirical faithfulness metrics for bilinear form scores with φ₀ normalization.

    Error metric: ε = |B̃_λ - B_λ| / (sqrt(φ₀(g)) * sqrt(φ₀(v)))
    This matches the theoretical bound from Theorem 1.

    Args:
        exact_scores: Exact bilinear scores (n, k)
        sketched_scores: Sketched bilinear scores (n, k)
        train_phi0: Unregularized self-influence for training samples (n,)
        test_phi0: Unregularized self-influence for test samples (k,)
        threshold: Threshold for defining "faithful" (default 0.1)

    Returns:
        Dictionary with faithfulness metrics
    """
    exact_f64 = exact_scores.to(torch.float64)
    sketched_f64 = sketched_scores.to(torch.float64)
    train_phi0_f64 = train_phi0.to(torch.float64)
    test_phi0_f64 = test_phi0.to(torch.float64)

    # Compute normalizer: sqrt(φ₀(g_i)) * sqrt(φ₀(v_j)) for each (i, j) pair
    sqrt_train = torch.sqrt(torch.clamp(train_phi0_f64, min=1e-12)).unsqueeze(1)  # (n, 1)
    sqrt_test = torch.sqrt(torch.clamp(test_phi0_f64, min=1e-12)).unsqueeze(0)    # (1, k)
    normalizer = sqrt_train * sqrt_test  # (n, k)

    # Compute additive error
    additive_error = (sketched_f64 - exact_f64).abs()  # (n, k)

    # Compute normalized epsilon = |error| / normalizer
    valid_mask = normalizer > 1e-12
    epsilon = torch.zeros_like(additive_error)
    epsilon[valid_mask] = additive_error[valid_mask] / normalizer[valid_mask]

    n_valid = valid_mask.sum().item()
    n_total = exact_scores.numel()

    if n_valid == 0:
        return {
            "eps_mean": float('nan'),
            "eps_median": float('nan'),
            "eps_max": float('nan'),
            "p95_error": float('nan'),
            "p99_error": float('nan'),
            "is_faithful": False,
            "fraction_within_threshold": 0.0,
            "n_valid": 0,
            "n_total": n_total,
        }

    valid_epsilon = epsilon[valid_mask]

    # Fraction of scores within threshold
    within_threshold = (valid_epsilon <= threshold).float().mean().item()

    # Define faithful: p95 error <= threshold
    p95_error = torch.quantile(valid_epsilon, 0.95).item()
    is_faithful = p95_error <= threshold

    return {
        "eps_mean": valid_epsilon.mean().item(),
        "eps_median": valid_epsilon.median().item(),
        "eps_max": valid_epsilon.max().item(),
        "p95_error": p95_error,
        "p99_error": torch.quantile(valid_epsilon, 0.99).item(),
        "is_faithful": is_faithful,
        "fraction_within_threshold": within_threshold,
        "n_valid": n_valid,
        "n_total": n_total,
    }


# =============================================================================
# Joint Sweep with Empirical Faithfulness
# =============================================================================

def run_lambda_sweep_for_m(
    train_grad_cache: GradientCache,
    val_grad_cache: GradientCache,
    test_grad_cache: GradientCache,
    val_gt: tuple,
    test_gt: tuple,
    proj_dim: int,
    lambda_values: List[float],
    exact_bilinear_scores: Dict[float, Tensor],
    train_phi0: Tensor,
    test_phi0: Tensor,
    test_grads: Tensor,
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    faithfulness_threshold: float = 0.1,
) -> Dict:
    """
    Sweep over λ values with fixed projection dimension m, measuring both
    utility (LDS) and empirical faithfulness (bilinear form error with φ₀ normalization).

    Efficiently computes BOTH validation and test LDS for all λ values using a single
    projection and eigendecomposition. This is much faster than projecting separately
    for validation and test evaluation.

    Args:
        train_grad_cache: Training gradient cache
        val_grad_cache: Validation gradient cache (for λ selection)
        test_grad_cache: Test gradient cache (for final evaluation)
        val_gt: Validation ground truth for LDS (for λ selection)
        test_gt: Test ground truth for LDS (for final evaluation)
        proj_dim: Projection dimension m (fixed)
        lambda_values: List of λ values to sweep
        exact_bilinear_scores: Pre-computed exact bilinear scores {λ -> (n, k) scores}
        train_phi0: Unregularized self-influence for training samples (n,)
        test_phi0: Unregularized self-influence for test samples (k,)
        test_grads: Test gradient vectors (k, d) for bilinear score computation
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for projection
        faithfulness_threshold: Threshold for defining faithful (default 0.1)

    Returns:
        Dictionary with sweep results for this m (including both val and test LDS)
    """
    n_train = train_grad_cache.n_samples
    n_val = val_grad_cache.n_samples
    n_test = test_grad_cache.n_samples
    d = train_grad_cache.dim
    dtype = train_grad_cache.dtype

    # Reduce batch size for very large m to avoid GPU memory issues
    # For m > 256K, use smaller batches to reduce peak memory from temporary tensors
    if proj_dim > 262144:  # 2^18
        batch_size = min(batch_size, 32)
    elif proj_dim > 65536:  # 2^16
        batch_size = min(batch_size, 48)

    # Setup projector
    proj_type_enum = ProjectionType(proj_type)

    proj_bs = min(batch_size, 32)
    proj_bs = max(8, (proj_bs // 8) * 8)

    projector = make_random_projector(
        param_shape_list=[d],
        feature_batch_size=batch_size,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_bs,
        device=torch.device(device),
        proj_seed=seed,
        proj_type=proj_type_enum,
        dtype=dtype,
    )

    # Validate projection dimension
    if proj_dim >= d:
        raise ValueError(
            f"Invalid projection dimension: proj_dim={proj_dim} must be < d={d}. "
            f"Random projection is only meaningful when projecting to lower dimensions."
        )

    # Compute sketched bilinear scores for faithfulness measurement
    sketched_bilinear_scores = compute_sketched_bilinear_scores(
        train_grad_cache=train_grad_cache,
        test_grads=test_grads,
        projector=projector,
        lambda_values=lambda_values,
        device=device,
        batch_size=batch_size,
    )

    # Free GPU memory from sketched bilinear computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Project gradients for LDS computation (train, val, AND test - all with same projector)
    PG_train = torch.zeros(n_train, proj_dim, dtype=dtype, device="cpu")
    for i in range(0, n_train, batch_size):
        i_end = min(i + batch_size, n_train)
        g_batch = train_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
        PG_train[i:i_end] = pg_batch.cpu()

    PG_val = torch.zeros(n_val, proj_dim, dtype=dtype, device="cpu")
    for i in range(0, n_val, batch_size):
        i_end = min(i + batch_size, n_val)
        g_batch = val_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
        PG_val[i:i_end] = pg_batch.cpu()

    PG_test = torch.zeros(n_test, proj_dim, dtype=dtype, device="cpu")
    for i in range(0, n_test, batch_size):
        i_end = min(i + batch_size, n_test)
        g_batch = test_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
        PG_test[i:i_end] = pg_batch.cpu()

    # Compute kernel matrices for LDS (both val and test cross-kernels)
    PG_train_gpu = PG_train.to(device)
    K_train = PG_train_gpu @ PG_train_gpu.T

    PG_val_gpu = PG_val.to(device)
    K_val_cross = PG_val_gpu @ PG_train_gpu.T

    PG_test_gpu = PG_test.to(device)
    K_test_cross = PG_test_gpu @ PG_train_gpu.T

    # Eigendecompose ONCE for fast λ sweep (used for both val and test LDS)
    eigenvalues, U = torch.linalg.eigh(K_train.double())
    eigenvalues = torch.clamp(eigenvalues, min=0).to(dtype).to(device)
    U = U.to(dtype).to(device)

    # Precompute for efficient λ sweep
    UT_Kval_T = U.T @ K_val_cross.T
    UT_Ktest_T = U.T @ K_test_cross.T

    # Sweep λ values - compute BOTH val and test LDS efficiently
    results = {
        "proj_dim": proj_dim,
        "lambda_values": [],
        "val_lds": [],
        "test_lds": [],
        "faithfulness": [],
    }

    for lamb in tqdm(lambda_values, desc=f"    λ sweep (m={proj_dim})", leave=False):
        # Compute inverse diagonal (shared between val and test)
        n_lamb = n_train * lamb
        inv_diag = 1.0 / (eigenvalues + n_lamb)

        # Compute validation LDS (GPU-accelerated)
        val_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Kval_T)
        val_lds_score = lds(val_scores_T, val_gt, device=device)
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

        # Compute test LDS (using same eigendecomposition - very efficient!)
        test_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Ktest_T)
        test_lds_score = lds(test_scores_T, test_gt, device=device)
        mean_test_lds = torch.mean(test_lds_score[~torch.isnan(test_lds_score)]).item()

        # Compute bilinear form faithfulness metrics with φ₀ normalization
        if lamb in exact_bilinear_scores and lamb in sketched_bilinear_scores:
            faith_metrics = compute_bilinear_faithfulness_metrics(
                exact_bilinear_scores[lamb],
                sketched_bilinear_scores[lamb],
                train_phi0,
                test_phi0,
                threshold=faithfulness_threshold,
            )
        else:
            faith_metrics = {"is_faithful": None, "p95_error": float('nan')}

        results["lambda_values"].append(lamb)
        results["val_lds"].append(mean_val_lds)
        results["test_lds"].append(mean_test_lds)
        results["faithfulness"].append(faith_metrics)

    # Find best λ for this m (based on VALIDATION LDS for λ selection)
    best_idx = np.argmax(results["val_lds"])
    results["best_lambda"] = results["lambda_values"][best_idx]
    results["best_val_lds"] = results["val_lds"][best_idx]
    results["best_test_lds"] = results["test_lds"][best_idx]
    results["best_lds"] = results["val_lds"][best_idx]  # For backward compatibility
    results["best_faithfulness"] = results["faithfulness"][best_idx]

    # Explicitly delete large GPU tensors
    del PG_train, PG_val, PG_test, PG_train_gpu, PG_val_gpu, PG_test_gpu
    del K_train, K_val_cross, K_test_cross, eigenvalues, U, UT_Kval_T, UT_Ktest_T

    # Clean up GPU tensors and synchronize before returning
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return results


def run_joint_sweep(
    train_grad_cache: GradientCache,
    val_grad_cache: GradientCache,
    test_grad_cache: GradientCache,
    val_gt: tuple,
    test_gt: tuple,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    K: torch.Tensor,
    m_values: List[int],
    lambda_values: List[float],
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    faithfulness_threshold: float = 0.1,
    num_test_samples: int = 200,
    num_trials: int = 1,
) -> Dict:
    """
    Run the full joint sweep experiment with empirical faithfulness measurement.

    Uses bilinear form error with φ₀ normalization for faithfulness, matching
    the theoretical bounds from Theorem 1.

    **Important**: λ selection uses the validation set, but the final LDS values
    reported for each (m, λ*(m)) are evaluated on the TEST set. This prevents
    overfitting to the validation set during hyperparameter selection.

    Args:
        train_grad_cache: Training gradient cache
        val_grad_cache: Validation gradient cache (used for λ selection)
        test_grad_cache: Test gradient cache (used for final evaluation)
        val_gt: Validation ground truth for LDS (for λ selection)
        test_gt: Test ground truth for LDS (for final evaluation)
        eigenvalues: Pre-computed eigenvalues of full Gram matrix (descending)
        eigenvectors: Pre-computed eigenvectors of full Gram matrix
        K: Pre-computed Gram matrix (for exact score computation)
        m_values: List of projection dimensions to test
        lambda_values: List of λ values to sweep for each m
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for computation
        faithfulness_threshold: Threshold for empirical faithfulness (ε₉₅ threshold)
        num_test_samples: Number of test samples for faithfulness measurement
        num_trials: Number of random projection trials for statistical analysis

    Returns:
        Dictionary with comprehensive results including per-trial statistics
    """
    n = train_grad_cache.n_samples

    print(f"\n{'='*60}")
    print("Joint Sweep: Faithfulness-Utility Alignment")
    print("(Bilinear form error with φ₀ normalization)")
    print(f"{'='*60}")
    print(f"  m values: {len(m_values)} ({min(m_values)} to {max(m_values)})")
    print(f"  λ values: {len(lambda_values)} ({min(lambda_values):.1e} to {max(lambda_values):.1e})")
    print(f"  Faithfulness threshold (ε₉₅): {faithfulness_threshold}")
    print(f"  Number of trials: {num_trials}")

    # Get test gradients for bilinear form computation
    k_test = min(num_test_samples, test_grad_cache.n_samples)
    test_grads_list = [test_grad_cache.get_sample(i, device="cpu") for i in range(k_test)]
    test_grads = torch.stack(test_grads_list).to(device)  # (k, d)

    print(f"  Test gradients: {k_test} samples, {test_grads.shape[1]:,} dims, "
          f"{test_grads.element_size() * test_grads.numel() / 1e9:.2f} GB on GPU")

    # Compute exact bilinear scores and φ₀ ONCE (reused across all m values)
    print(f"\n[1/3] Computing exact bilinear scores and φ₀...")
    exact_bilinear_scores, U_B, train_phi0, test_phi0 = compute_exact_bilinear_scores(
        train_grad_cache=train_grad_cache,
        test_grads=test_grads,
        K=K,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_values=lambda_values,
        device=device,
        batch_size=batch_size,
    )

    # Move test_grads to CPU to free GPU memory (will be moved back as needed)
    print(f"  Moving test_grads to CPU to save {test_grads.element_size() * test_grads.numel() / 1e9:.2f} GB GPU memory...")
    test_grads = test_grads.cpu()

    # Clean up U_B from GPU to save memory (it's no longer needed)
    del U_B
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Compute d_λ for all λ values
    print(f"\n[2/3] Computing effective dimensions d_λ...")
    d_lambda_dict = {}
    for lamb in lambda_values:
        d_lambda_dict[lamb] = compute_effective_dimension(eigenvalues, lamb)

    # Main loop: trial → m → λ (with multi-seed support)
    print(f"\n[3/3] Running joint sweep ({num_trials} trials)...")

    results = {
        "m_values": m_values,
        "lambda_values": lambda_values,
        "faithfulness_threshold": faithfulness_threshold,
        "d_lambda": d_lambda_dict,
        "num_trials": num_trials,
        "grid": {},                    # Full (m, λ) -> val_lds grid (averaged over trials)
        "test_grid": {},               # Full (m, λ) -> test_lds grid (averaged over trials)
        "faithfulness_grid": {},       # Full (m, λ) -> faithfulness metrics (averaged)
        "per_m_results": {},           # Per-m summary with test LDS
        "per_trial_results": {},       # Detailed per-trial results
    }

    # Initialize per-trial storage
    for trial in range(num_trials):
        results["per_trial_results"][trial] = {}

    # Collect results across trials for each m
    per_m_trial_data = {m: [] for m in m_values}

    # Iterate from largest m first (most memory intensive) to fail fast if OOM
    m_values_desc = sorted(m_values, reverse=True)

    for trial in range(num_trials):
        trial_seed = seed + trial
        print(f"\nTrial {trial + 1}/{num_trials} (seed={trial_seed})")

        # Clean GPU state at the start of each trial
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        for m in tqdm(m_values_desc, desc=f"  Sweeping m (trial {trial+1})"):
            try:
                # Efficient sweep: computes BOTH val and test LDS for all λ values
                # using a single projection and eigendecomposition
                sweep_result = run_lambda_sweep_for_m(
                    train_grad_cache=train_grad_cache,
                    val_grad_cache=val_grad_cache,
                    test_grad_cache=test_grad_cache,
                    val_gt=val_gt,
                    test_gt=test_gt,
                    proj_dim=m,
                    lambda_values=lambda_values,
                    exact_bilinear_scores=exact_bilinear_scores,
                    train_phi0=train_phi0,
                    test_phi0=test_phi0,
                    test_grads=test_grads,
                    proj_type=proj_type,
                    device=device,
                    seed=trial_seed,
                    batch_size=batch_size,
                    faithfulness_threshold=faithfulness_threshold,
                )
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    print(f"\n[ERROR] GPU error at m={m}, trial={trial+1}")
                    print(f"  Error: {e}")
                    print(f"  Try reducing m_exp_max or batch_size")
                    # Clean up GPU state
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
                raise

            # λ selection uses validation set, but test LDS is computed in same sweep
            lambda_star = sweep_result["best_lambda"]
            val_lds_star = sweep_result["best_val_lds"]
            test_lds_star = sweep_result["best_test_lds"]  # Already computed efficiently!
            faith_at_star = sweep_result["best_faithfulness"]

            # Store trial data (now includes full test LDS grid)
            trial_data = {
                "lambda_star": lambda_star,
                "val_lds_star": val_lds_star,
                "test_lds_star": test_lds_star,
                "faithfulness_at_star": faith_at_star,
                "all_val_lds": sweep_result["val_lds"],
                "all_test_lds": sweep_result["test_lds"],  # Full test LDS grid
                "all_faithfulness": sweep_result["faithfulness"],
            }
            per_m_trial_data[m].append(trial_data)
            results["per_trial_results"][trial][m] = trial_data

            # Progress output
            p95_error = faith_at_star.get("p95_error", float('nan'))
            is_faithful = faith_at_star.get("is_faithful", False)
            region_str = "faithful" if is_faithful else "UNFAITHFUL"
            tqdm.write(f"    m={m:6d}: λ*(m)={lambda_star:.2e}, val_LDS={val_lds_star:.4f}, "
                       f"test_LDS={test_lds_star:.4f}, ε₉₅={p95_error:.3f} [{region_str}]")

            # Explicit memory cleanup after each m iteration
            del sweep_result, lambda_star, val_lds_star, test_lds_star, faith_at_star, trial_data
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

    # Aggregate results across trials
    print(f"\nAggregating results across {num_trials} trials...")

    for m in m_values:
        trial_data_list = per_m_trial_data[m]

        # Aggregate λ* (use mode/median, or just report all)
        lambda_stars = [td["lambda_star"] for td in trial_data_list]
        val_lds_stars = np.array([td["val_lds_star"] for td in trial_data_list])
        test_lds_stars = np.array([td["test_lds_star"] for td in trial_data_list])

        # Use the most common λ* (mode) or the λ* from first trial
        # For consistency, use median λ* based on val_lds ranking
        best_trial_idx = np.argmax(val_lds_stars)
        lambda_star = lambda_stars[best_trial_idx]
        faith_at_star = trial_data_list[best_trial_idx]["faithfulness_at_star"]

        # Find faithful region
        all_faithfulness = trial_data_list[0]["all_faithfulness"]  # Same for all trials
        faithful_lambdas = [
            lamb for i, lamb in enumerate(lambda_values)
            if all_faithfulness[i].get("is_faithful", False)
        ]
        min_faithful_lambda = min(faithful_lambdas) if faithful_lambdas else float('inf')
        in_unfaithful_region = not faith_at_star.get("is_faithful", False)

        # Store aggregated results (use TEST LDS for reporting)
        results["per_m_results"][m] = {
            "lambda_star": lambda_star,
            "val_lds_star": float(np.mean(val_lds_stars)),
            "val_lds_star_std": float(np.std(val_lds_stars, ddof=1)) if num_trials > 1 else 0.0,
            "test_lds_star": float(np.mean(test_lds_stars)),
            "test_lds_star_std": float(np.std(test_lds_stars, ddof=1)) if num_trials > 1 else 0.0,
            "lds_star": float(np.mean(test_lds_stars)),  # Use test LDS as primary metric
            "lds_star_std": float(np.std(test_lds_stars, ddof=1)) if num_trials > 1 else 0.0,
            "faithfulness_at_star": faith_at_star,
            "in_unfaithful_region": in_unfaithful_region,
            "min_faithful_lambda": min_faithful_lambda,
            "all_lds": trial_data_list[0]["all_val_lds"],  # Val LDS grid for λ selection viz
            "all_test_lds": trial_data_list[0]["all_test_lds"],  # Test LDS grid
            "all_faithfulness": all_faithfulness,
            "n_trials": num_trials,
            "per_trial_lambda_stars": lambda_stars,
            "per_trial_val_lds": val_lds_stars.tolist(),
            "per_trial_test_lds": test_lds_stars.tolist(),
        }

        # Store grid values (averaged over trials)
        for i, lamb in enumerate(lambda_values):
            val_lds_values = [td["all_val_lds"][i] for td in trial_data_list]
            test_lds_values = [td["all_test_lds"][i] for td in trial_data_list]
            results["grid"][(m, lamb)] = float(np.mean(val_lds_values))  # Val for backward compat
            results["test_grid"][(m, lamb)] = float(np.mean(test_lds_values))  # Test grid
            results["faithfulness_grid"][(m, lamb)] = all_faithfulness[i]

        # Summary output
        region_str = "UNFAITHFUL" if in_unfaithful_region else "faithful"
        p95_error = faith_at_star.get("p95_error", float('nan'))
        print(f"  m={m:6d}: λ*(m)={lambda_star:.2e}, test_LDS={np.mean(test_lds_stars):.4f}±{np.std(test_lds_stars):.4f}, "
              f"ε₉₅={p95_error:.3f} [{region_str}]")

    # Find best configuration (already evaluated on test set during sweep)
    print(f"\n[4/4] Summarizing best configuration...")

    best_m = max(results["per_m_results"].keys(),
                 key=lambda m: results["per_m_results"][m]["test_lds_star"])
    best_lambda = results["per_m_results"][best_m]["lambda_star"]
    best_m_results = results["per_m_results"][best_m]

    results["best_config"] = {
        "m": best_m,
        "lambda": best_lambda,
        "val_lds": best_m_results["val_lds_star"],
        "val_lds_std": best_m_results.get("val_lds_star_std", 0.0),
        "test_lds": best_m_results["test_lds_star"],
        "test_lds_std": best_m_results.get("test_lds_star_std", 0.0),
    }

    print(f"\nBest configuration: m={best_m}, λ={best_lambda:.2e}")
    print(f"  Val LDS:  {results['best_config']['val_lds']:.4f} ± {results['best_config']['val_lds_std']:.4f}")
    print(f"  Test LDS: {results['best_config']['test_lds']:.4f} ± {results['best_config']['test_lds_std']:.4f}")

    return results


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_results(results: Dict) -> Dict:
    """Analyze the joint sweep results."""
    per_m = results["per_m_results"]
    m_values = sorted(per_m.keys())

    analysis = {
        "alignment": {},
        "monotonicity": {},
    }

    # Question 1: Alignment - is λ*(m) empirically faithful?
    n_unfaithful = sum(1 for m in m_values if per_m[m]["in_unfaithful_region"])
    analysis["alignment"] = {
        "n_total": len(m_values),
        "n_in_unfaithful_region": n_unfaithful,
        "fraction_unfaithful": n_unfaithful / len(m_values),
        "conclusion": "λ*(m) tends to be empirically UNFAITHFUL" if n_unfaithful > len(m_values) / 2
                      else "λ*(m) tends to be empirically faithful",
    }

    # Question 2: Monotonicity (use test LDS, which is the primary metric)
    # Handle both old format (lds_star) and new format (test_lds_star)
    lds_stars = []
    lds_stds = []
    for m in m_values:
        if "test_lds_star" in per_m[m]:
            lds_stars.append(per_m[m]["test_lds_star"])
            lds_stds.append(per_m[m].get("test_lds_star_std", 0.0))
        else:
            lds_stars.append(per_m[m]["lds_star"])
            lds_stds.append(per_m[m].get("lds_star_std", 0.0))

    monotonic_violations = 0
    for i in range(1, len(lds_stars)):
        if lds_stars[i] < lds_stars[i-1] - 0.01:
            monotonic_violations += 1

    correlation = np.corrcoef(m_values, lds_stars)[0, 1]

    analysis["monotonicity"] = {
        "lds_stars": lds_stars,
        "lds_stds": lds_stds,
        "m_values": m_values,
        "monotonic_violations": monotonic_violations,
        "correlation_m_lds": correlation,
        "is_monotonic": monotonic_violations == 0,
        "conclusion": "LDS*(m) is monotonically increasing with m" if monotonic_violations == 0
                      else f"LDS*(m) has {monotonic_violations} violations of monotonicity",
    }

    return analysis


def print_analysis(analysis: Dict):
    """Print a summary of the analysis."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY (Empirical Faithfulness)")
    print("(Note: LDS* values are from TEST set evaluation)")
    print("="*60)

    print("\n[Question 1: Alignment]")
    align = analysis["alignment"]
    print(f"  λ*(m) empirically unfaithful: {align['n_in_unfaithful_region']}/{align['n_total']} "
          f"({align['fraction_unfaithful']*100:.1f}%)")
    print(f"  Conclusion: {align['conclusion']}")

    print("\n[Question 2: Monotonicity]")
    mono = analysis["monotonicity"]
    print(f"  Correlation(m, test_LDS*): {mono['correlation_m_lds']:.3f}")
    print(f"  Monotonic violations: {mono['monotonic_violations']}")
    print(f"  Conclusion: {mono['conclusion']}")


# =============================================================================
# Data Loading Utilities
# =============================================================================

# =============================================================================
# Main Experiment Entry Point
# =============================================================================

def run_experiment(
    dataset: str,
    model_name: str,
    m_values: List[int],
    lambda_values: List[float],
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    offload: str = "cpu",
    cache_dir: Optional[str] = None,
    proj_type: str = "normal",
    val_ratio: float = 0.1,
    faithfulness_threshold: float = 0.1,
    num_test_samples: int = 200,
    num_trials: int = 5,
) -> Dict:
    """
    Run the full faithfulness-utility alignment experiment with empirical faithfulness.

    Args:
        dataset: Dataset name (mnist, cifar2, maestro)
        model_name: Model name (lr, mlp, resnet9, musictransformer)
        m_values: List of projection dimensions to test
        lambda_values: List of λ values to sweep
        device: Device for computation
        seed: Random seed
        batch_size: Batch size
        offload: Gradient storage mode
        cache_dir: Directory for gradient cache
        proj_type: Type of random projection
        val_ratio: Fraction of test set for validation
        faithfulness_threshold: Threshold for empirical faithfulness (default 0.1)
        num_test_samples: Number of samples for faithfulness measurement
        num_trials: Number of random projection trials for statistical analysis

    Returns:
        Dictionary with all results and analysis
    """
    print(f"\n{'='*60}")
    print(f"Faithfulness-Utility Alignment Experiment (Empirical)")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"{'='*60}")

    # Load benchmark
    model_details, groundtruth = load_benchmark(
        model=model_name, dataset=dataset, metric="lds"
    )

    # Load the checkpoint that corresponds to the ground truth
    # The ground truth LDS values were computed using models_full[0]
    model = model_details["model"]
    checkpoint = torch.load(model_details["models_full"][0], map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {model_details['models_full'][0]}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = "musictransformer" if model_name == "musictransformer" else "default"

    print(f"Model parameters: {n_params:,}")

    # Get indices and ground truth
    train_indices = list(model_details["train_sampler"])
    val_indices, test_indices = get_validation_split_indices(
        model_details["test_sampler"], val_ratio=val_ratio, seed=seed
    )

    # Compute ground truth for val/test splits
    original_gt_values, subset_indices = groundtruth
    test_indices_dict = {idx: pos for pos, idx in enumerate(model_details["test_sampler"])}

    val_gt_indices = [test_indices_dict[idx] for idx in val_indices]
    val_gt_values = original_gt_values[:, val_gt_indices]
    val_gt = (val_gt_values, subset_indices)

    test_gt_indices = [test_indices_dict[idx] for idx in test_indices]
    test_gt_values = original_gt_values[:, test_gt_indices]
    test_gt = (test_gt_values, subset_indices)

    # Compute gradient caches
    print("\n[Setup] Computing gradient caches...")

    train_grad_cache = create_gradient_cache(
        model=model,
        dataset=model_details["train_dataset"],
        indices=train_indices,
        device=device,
        offload=offload,
        model_type=model_type,
        batch_size=batch_size,
        cache_dir=f"{cache_dir}/train" if cache_dir else None,
    )

    val_grad_cache = create_gradient_cache(
        model=model,
        dataset=model_details["test_dataset"],
        indices=val_indices,
        device=device,
        offload=offload,
        model_type=model_type,
        batch_size=batch_size,
        cache_dir=f"{cache_dir}/val" if cache_dir else None,
    )

    test_grad_cache = create_gradient_cache(
        model=model,
        dataset=model_details["test_dataset"],
        indices=test_indices,
        device=device,
        offload=offload,
        model_type=model_type,
        batch_size=batch_size,
        cache_dir=f"{cache_dir}/test" if cache_dir else None,
    )

    print(f"  Caches ready: train={train_grad_cache.n_samples}, "
          f"val={val_grad_cache.n_samples}, test={test_grad_cache.n_samples}")

    # Filter out m values that are >= parameter dimension (no projection needed)
    d = train_grad_cache.dim
    original_m_values = m_values
    m_values = [m for m in m_values if m < d]
    if len(m_values) < len(original_m_values):
        print(f"\n  Filtered m values: {len(original_m_values)} -> {len(m_values)} "
              f"(removed m >= d={d})")
        print(f"  Valid m values: {m_values}")
        if not m_values:
            raise ValueError(f"No valid m values remaining after filtering (all >= d={d})")

    # Compute exact Gram matrix and eigenspectrum (including eigenvectors for φ₀)
    print("\n[Setup] Computing exact Gram matrix and eigenspectrum...")
    eigenvalues, eigenvectors, K = compute_exact_gram_matrix(
        train_grad_cache, device=device, batch_size=batch_size * 4
    )
    print(f"  Eigenvalues: max={eigenvalues[0]:.2e}, "
          f"rank={torch.sum(eigenvalues > 1e-10 * eigenvalues[0]).item()}")
    print(f"  K is stored on CPU to save GPU memory")

    # Ensure GPU memory is clean before starting sweep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Run the joint sweep (uses bilinear form error with φ₀ normalization)
    results = run_joint_sweep(
        train_grad_cache=train_grad_cache,
        val_grad_cache=val_grad_cache,
        test_grad_cache=test_grad_cache,
        val_gt=val_gt,
        test_gt=test_gt,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        K=K,
        m_values=m_values,
        lambda_values=lambda_values,
        proj_type=proj_type,
        device=device,
        seed=seed,
        batch_size=batch_size,
        faithfulness_threshold=faithfulness_threshold,
        num_test_samples=num_test_samples,
        num_trials=num_trials,
    )

    # Add metadata
    results["metadata"] = {
        "dataset": dataset,
        "model": model_name,
        "n_params": n_params,
        "n_train": train_grad_cache.n_samples,
        "n_val": val_grad_cache.n_samples,
        "n_test": test_grad_cache.n_samples,
        "seed": seed,
        "proj_type": proj_type,
        "batch_size": batch_size,
        "num_test_samples": num_test_samples,
        "num_trials": num_trials,
    }

    # Run analysis
    analysis = analyze_results(results)
    results["analysis"] = analysis
    print_analysis(analysis)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Faithfulness-Utility Alignment Experiment (Empirical)"
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar2", "maestro"])
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["lr", "mlp", "resnet9", "musictransformer"])
    parser.add_argument("--proj_type", type=str, default="normal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--offload", type=str, default="cpu",
                       choices=["none", "cpu", "disk"])
    parser.add_argument("--cache_dir", type=str, default="./grad_cache")

    # Faithfulness settings
    parser.add_argument("--faithfulness_threshold", type=float, default=0.01,
                       help="Threshold for empirical faithfulness (default 0.01 = 1%% error)")
    parser.add_argument("--num_test_samples", type=int, default=500,
                       help="Number of samples for faithfulness measurement")
    parser.add_argument("--num_trials", type=int, default=5,
                       help="Number of random projection trials for statistical analysis")

    # Validation configuration
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Fraction of test set to use for validation (default: 0.1)")

    # Sweep configuration (powers of 2, matching spectrum_bounds.py)
    parser.add_argument("--m_exp_min", type=int, default=5,
                       help="Minimum exponent for m sweep: m_min = 2^exp_min (default: 5 → 32)")
    parser.add_argument("--m_exp_max", type=int, default=20,
                       help="Maximum exponent for m sweep: m_max = 2^exp_max (default: 20 → 1048576)")
    parser.add_argument("--m_steps", type=int, default=0,
                       help="Number of log2-spaced m values. If 0 (default), uses integer powers of 2.")

    # Lambda sweep configuration (integer powers of 10, or log-spaced if lambda_steps specified)
    parser.add_argument("--lambda_exp_min", type=int, default=-8,
                       help="Minimum exponent for λ sweep: λ_min = 10^exp_min (default: -8 → 1e-8)")
    parser.add_argument("--lambda_exp_max", type=int, default=2,
                       help="Maximum exponent for λ sweep: λ_max = 10^exp_max (default: 2 → 1e2)")
    parser.add_argument("--lambda_steps", type=int, default=0,
                       help="Number of log-spaced λ values. If 0 (default), uses integer powers of 10.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate m values: integer powers of 2 if m_steps=0, else log2-spaced
    if args.m_steps == 0:
        m_values = [2 ** exp for exp in range(args.m_exp_min, args.m_exp_max + 1)]
    else:
        m_values = np.unique(np.logspace(
            args.m_exp_min * np.log10(2),  # Convert base-2 exponents to base-10
            args.m_exp_max * np.log10(2),
            num=args.m_steps,
            base=10
        ).astype(int)).tolist()

    # Generate λ values: integer powers of 10 if lambda_steps=0, else log-spaced
    if args.lambda_steps == 0:
        lambda_values = [10.0 ** exp for exp in range(args.lambda_exp_min, args.lambda_exp_max + 1)]
    else:
        lambda_values = np.logspace(args.lambda_exp_min, args.lambda_exp_max, num=args.lambda_steps).tolist()
        if args.lambda_steps > 500:
            print(f"\n⚠️  NOTE: lambda_steps={args.lambda_steps} is very high.")
            print(f"   Using GPU-accelerated Spearman correlation (~20x faster than CPU).")
            print(f"   Estimated time per m value: ~{args.lambda_steps * 0.02:.1f}s\n")

    print(f"\nConfiguration:")
    if args.m_steps == 0:
        print(f"  m sweep: 2^[{args.m_exp_min}, {args.m_exp_max}] = {[f'2^{exp}' for exp in range(args.m_exp_min, args.m_exp_max + 1)]}")
        print(f"  m values (decimal): {m_values}")
    else:
        print(f"  m sweep: [2^{args.m_exp_min}, 2^{args.m_exp_max}] ({args.m_steps} log2-spaced steps)")
        print(f"  m values: {m_values}")
    if args.lambda_steps == 0:
        print(f"  λ sweep: 10^[{args.lambda_exp_min}, {args.lambda_exp_max}] = {[f'1e{e}' for e in range(args.lambda_exp_min, args.lambda_exp_max + 1)]}")
    else:
        print(f"  λ sweep: [1e{args.lambda_exp_min}, 1e{args.lambda_exp_max}] ({args.lambda_steps} log-spaced steps)")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Faithfulness threshold: {args.faithfulness_threshold}")
    print(f"  Number of trials: {args.num_trials}")

    # Run experiment
    results = run_experiment(
        dataset=args.dataset,
        model_name=args.model,
        m_values=m_values,
        lambda_values=lambda_values,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        offload=args.offload,
        cache_dir=args.cache_dir,
        proj_type=args.proj_type,
        val_ratio=args.val_ratio,
        faithfulness_threshold=args.faithfulness_threshold,
        num_test_samples=args.num_test_samples,
        num_trials=args.num_trials,
    )

    # Save results
    experiment_dir = os.path.join(args.output_dir, "faithfulness_utility")
    os.makedirs(experiment_dir, exist_ok=True)

    results_filename = f"{args.dataset}_{args.model}_{args.proj_type}.pt"
    results_path = os.path.join(experiment_dir, results_filename)
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

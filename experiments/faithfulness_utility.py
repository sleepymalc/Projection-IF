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
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from dattri.benchmark.load import load_benchmark
from dattri.metric import lds
from dattri.func.projection import make_random_projector, ProjectionType

from utils.fisher_utils import compute_eigenspectrum
from utils.gradient_cache import GradientCache, create_gradient_cache


# =============================================================================
# Exact Score Computation (for faithfulness measurement)
# =============================================================================

def compute_exact_gram_matrix(
    grad_cache: GradientCache,
    device: str = "cuda",
    batch_size: int = 64,
) -> Tuple[Tensor, Tensor]:
    """
    Compute exact Gram matrix K = (1/n) G @ G^T in float64.

    Args:
        grad_cache: GradientCache containing the training gradients
        device: GPU device
        batch_size: Batch size for streaming

    Returns:
        eigenvalues: Eigenvalues of K in descending order (float64)
        K: Gram matrix (n, n) in float64, normalized by 1/n
    """
    n = grad_cache.n_samples
    print(f"  Computing Exact Gram Matrix (n={n}) in float64...")

    K = torch.zeros(n, n, dtype=torch.float64, device=device)

    for i_start in tqdm(range(0, n, batch_size), desc="  Gram matrix", leave=False):
        i_end = min(i_start + batch_size, n)
        G_i = grad_cache.get_batch(i_start, i_end, device=device).to(dtype=torch.float64)

        for j_start in range(i_start, n, batch_size):
            j_end = min(j_start + batch_size, n)

            if i_start == j_start:
                G_j = G_i
            else:
                G_j = grad_cache.get_batch(j_start, j_end, device=device).to(dtype=torch.float64)

            block = G_i @ G_j.T
            K[i_start:i_end, j_start:j_end] = block

            if i_start != j_start:
                K[j_start:j_end, i_start:i_end] = block.T

    K.div_(n)

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(K)
    eigenvalues = eigenvalues.flip(0).clamp(min=0.0)

    return eigenvalues, K


def compute_exact_self_scores(
    grad_cache: GradientCache,
    K: Tensor,
    lambda_values: List[float],
    device: str = "cuda",
    batch_size: int = 64,
    num_test_samples: int = 200,
) -> Dict[float, Tensor]:
    """
    Compute exact self-influence scores for a subset of training samples.

    Uses Woodbury identity for efficiency:
        score_i = (1/λ)[||g_i||² - u_i^T (K + λI)^{-1} u_i]
    where u_i = (1/√n) G @ g_i and K = (1/n) G @ G^T.

    Args:
        grad_cache: GradientCache containing the gradients
        K: Pre-computed Gram matrix (n, n), normalized by 1/n, in float64
        lambda_values: List of λ values
        device: GPU device
        batch_size: Batch size for streaming
        num_test_samples: Number of samples to compute scores for

    Returns:
        Dictionary mapping λ -> exact self-scores tensor (num_test_samples,)
    """
    n = grad_cache.n_samples
    k = min(num_test_samples, n)

    print(f"  Computing exact self-scores for {k} samples...")

    # Get test gradients (first k samples)
    test_grads = []
    for i in range(k):
        test_grads.append(grad_cache.get_sample(i, device="cpu"))
    V = torch.stack(test_grads).to(device=device, dtype=torch.float64)  # (k, d)

    # Compute ||g_i||² for each test sample
    g_norms_sq = (V ** 2).sum(dim=1)  # (k,)

    # Compute U = (1/√n) G @ V^T, shape (n, k)
    U = torch.zeros(n, k, device=device, dtype=torch.float64)
    for i_start in tqdm(range(0, n, batch_size), desc="  Computing U", leave=False):
        i_end = min(i_start + batch_size, n)
        G_batch = grad_cache.get_batch(i_start, i_end, device=device).to(dtype=torch.float64)
        U[i_start:i_end] = G_batch @ V.T

    U.div_(n ** 0.5)

    # Eigendecompose K for fast solves across λ
    print(f"  Eigendecomposing K ({n}x{n})...")
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = eigenvalues.clamp(min=0.0)

    # Precompute W = U^T @ eigenvectors for fast λ sweep
    W = eigenvectors.T @ U  # (n, k)

    # Compute exact scores for each λ
    exact_scores = {}
    for lamb in lambda_values:
        inv_eigvals = 1.0 / (eigenvalues + lamb)
        # X = (K + λI)^{-1} @ U = eigenvectors @ (inv_eigvals * W)
        X = eigenvectors @ (inv_eigvals.unsqueeze(1) * W)  # (n, k)

        # score_i = (1/λ)[||g_i||² - u_i^T x_i]
        ux_dots = (U * X).sum(dim=0)  # (k,)
        scores = (g_norms_sq - ux_dots) / lamb

        exact_scores[lamb] = scores.cpu()

    return exact_scores


# =============================================================================
# Sketched Score Computation with Faithfulness Measurement
# =============================================================================

def compute_sketched_self_scores(
    grad_cache: GradientCache,
    projector,
    lambda_values: List[float],
    device: str = "cuda",
    batch_size: int = 64,
    num_test_samples: int = 200,
) -> Dict[float, Tensor]:
    """
    Compute sketched self-influence scores for a subset of samples.

    Args:
        grad_cache: GradientCache containing the gradients
        projector: Projector with .project() method
        lambda_values: List of λ values
        device: GPU device
        batch_size: Batch size
        num_test_samples: Number of samples to compute scores for

    Returns:
        Dictionary mapping λ -> sketched self-scores tensor (num_test_samples,)
    """
    n = grad_cache.n_samples
    k = min(num_test_samples, n)
    m = projector.proj_dim

    # Project test gradients
    test_grads = []
    for i in range(k):
        test_grads.append(grad_cache.get_sample(i, device=device))
    V = torch.stack(test_grads)  # (k, d)
    PV = projector.project(V, ensemble_id=0).to(dtype=torch.float64, device=device)  # (k, m)

    # Compute ||Pg_i||² for each test sample
    pv_norms_sq = (PV ** 2).sum(dim=1)  # (k,)

    # Build projected Gram matrix and U
    K_proj = torch.zeros(n, n, dtype=torch.float64, device=device)
    U = torch.zeros(n, k, dtype=torch.float64, device=device)

    # Project all gradients and build K_proj
    pg_storage = []
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        g_batch = grad_cache.get_batch(i_start, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0).to(dtype=torch.float64)
        pg_storage.append(pg_batch.cpu())

        # Compute U for this batch
        U[i_start:i_end] = pg_batch @ PV.T

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

    # Clean up
    del pg_storage

    # Normalize
    K_proj.div_(n)
    U.div_(n ** 0.5)

    # Eigendecompose for fast λ sweep
    eigenvalues, eigenvectors = torch.linalg.eigh(K_proj)
    eigenvalues = eigenvalues.clamp(min=0.0)

    W = eigenvectors.T @ U  # (n, k)

    # Compute sketched scores for each λ
    sketched_scores = {}
    for lamb in lambda_values:
        inv_eigvals = 1.0 / (eigenvalues + lamb)
        X = eigenvectors @ (inv_eigvals.unsqueeze(1) * W)
        ux_dots = (U * X).sum(dim=0)
        scores = (pv_norms_sq - ux_dots) / lamb
        sketched_scores[lamb] = scores.cpu()

    return sketched_scores


def compute_faithfulness_metrics(
    exact_scores: Tensor,
    sketched_scores: Tensor,
    threshold: float = 0.1,
) -> Dict:
    """
    Compute empirical faithfulness metrics.

    Args:
        exact_scores: Exact influence scores (k,)
        sketched_scores: Sketched influence scores (k,)
        threshold: Threshold for defining "faithful" (default 0.1 = 10% error)

    Returns:
        Dictionary with faithfulness metrics
    """
    # Filter out near-zero scores to avoid numerical issues
    exact_f64 = exact_scores.to(torch.float64)
    sketched_f64 = sketched_scores.to(torch.float64)

    max_score = exact_f64.abs().max()
    relative_threshold = max(1e-10 * max_score.item(), 1e-12)
    valid_mask = exact_f64.abs() > relative_threshold

    if valid_mask.sum() == 0:
        return {
            "mean_ratio": float('nan'),
            "std_ratio": float('nan'),
            "correlation": float('nan'),
            "mean_error": float('nan'),
            "max_error": float('nan'),
            "is_faithful": False,
            "fraction_within_threshold": 0.0,
            "n_valid": 0,
        }

    # Compute ratios
    ratios = torch.zeros_like(exact_f64)
    ratios[valid_mask] = sketched_f64[valid_mask] / exact_f64[valid_mask]
    valid_ratios = ratios[valid_mask]

    # Compute errors (|ratio - 1|)
    errors = (valid_ratios - 1).abs()

    # Compute correlation
    valid_exact = exact_f64[valid_mask]
    valid_sketched = sketched_f64[valid_mask]

    mean_exact = valid_exact.mean()
    mean_sketched = valid_sketched.mean()
    cov = ((valid_exact - mean_exact) * (valid_sketched - mean_sketched)).mean()
    std_exact = valid_exact.std()
    std_sketched = valid_sketched.std()

    if std_exact > 0 and std_sketched > 0:
        correlation = (cov / (std_exact * std_sketched)).item()
    else:
        correlation = float('nan')

    # Fraction of scores within threshold
    within_threshold = (errors <= threshold).float().mean().item()

    # Define faithful: mean ratio within [1-threshold, 1+threshold]
    mean_ratio = valid_ratios.mean().item()
    is_faithful = abs(mean_ratio - 1) <= threshold

    return {
        "mean_ratio": mean_ratio,
        "std_ratio": valid_ratios.std().item(),
        "correlation": correlation,
        "mean_error": errors.mean().item(),
        "max_error": errors.max().item(),
        "p95_error": torch.quantile(errors, 0.95).item(),
        "is_faithful": is_faithful,
        "fraction_within_threshold": within_threshold,
        "n_valid": valid_mask.sum().item(),
    }


# =============================================================================
# Effective Dimension Computation
# =============================================================================

def compute_effective_dimension(eigenvalues: torch.Tensor, lamb: float) -> float:
    """Compute effective dimension d_λ = Σ_i σ_i / (σ_i + λ)."""
    return torch.sum(eigenvalues / (eigenvalues + lamb)).item()


# =============================================================================
# Joint Sweep with Empirical Faithfulness
# =============================================================================

def run_lambda_sweep_for_m(
    train_grad_cache: GradientCache,
    val_grad_cache: GradientCache,
    val_gt: tuple,
    proj_dim: int,
    lambda_values: List[float],
    exact_scores: Dict[float, Tensor],
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    faithfulness_threshold: float = 0.1,
    num_test_samples: int = 200,
) -> Dict:
    """
    Sweep over λ values with fixed projection dimension m, measuring both
    utility (LDS) and empirical faithfulness.

    Args:
        train_grad_cache: Training gradient cache
        val_grad_cache: Validation gradient cache
        val_gt: Validation ground truth for LDS
        proj_dim: Projection dimension m (fixed)
        lambda_values: List of λ values to sweep
        exact_scores: Pre-computed exact scores {λ -> scores}
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for projection
        faithfulness_threshold: Threshold for defining faithful (default 0.1)
        num_test_samples: Number of samples for faithfulness measurement

    Returns:
        Dictionary with sweep results for this m
    """
    n_train = train_grad_cache.n_samples
    n_val = val_grad_cache.n_samples
    d = train_grad_cache.dim
    dtype = train_grad_cache.dtype

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

    # Compute sketched self-scores for faithfulness measurement
    sketched_scores = compute_sketched_self_scores(
        grad_cache=train_grad_cache,
        projector=projector,
        lambda_values=lambda_values,
        device=device,
        batch_size=batch_size,
        num_test_samples=num_test_samples,
    )

    # Project gradients for LDS computation
    PG_train = torch.zeros(n_train, proj_dim, dtype=dtype, device="cpu")
    for i in range(0, n_train, batch_size):
        i_end = min(i + batch_size, n_train)
        g_batch = train_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_train[i:i_end] = pg_batch.cpu()

    PG_val = torch.zeros(n_val, proj_dim, dtype=dtype, device="cpu")
    for i in range(0, n_val, batch_size):
        i_end = min(i + batch_size, n_val)
        g_batch = val_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_val[i:i_end] = pg_batch.cpu()

    # Compute kernel matrices for LDS
    PG_train_gpu = PG_train.to(device)
    K_train = PG_train_gpu @ PG_train_gpu.T

    PG_val_gpu = PG_val.to(device)
    K_val_cross = PG_val_gpu @ PG_train_gpu.T

    # Eigendecompose for fast λ sweep
    eigenvalues, U = torch.linalg.eigh(K_train.double())
    eigenvalues = torch.clamp(eigenvalues, min=0).to(dtype).to(device)
    U = U.to(dtype).to(device)

    UT_Kval_T = U.T @ K_val_cross.T

    # Sweep λ values
    results = {
        "proj_dim": proj_dim,
        "lambda_values": [],
        "val_lds": [],
        "faithfulness": [],
    }

    for lamb in lambda_values:
        # Compute LDS
        n_lamb = n_train * lamb
        inv_diag = 1.0 / (eigenvalues + n_lamb)
        val_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Kval_T)

        val_lds_score = lds(val_scores_T, val_gt)[0]
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

        # Compute faithfulness metrics
        if lamb in exact_scores and lamb in sketched_scores:
            faith_metrics = compute_faithfulness_metrics(
                exact_scores[lamb],
                sketched_scores[lamb],
                threshold=faithfulness_threshold,
            )
        else:
            faith_metrics = {"is_faithful": None, "mean_ratio": float('nan')}

        results["lambda_values"].append(lamb)
        results["val_lds"].append(mean_val_lds)
        results["faithfulness"].append(faith_metrics)

    # Find best λ for this m
    best_idx = np.argmax(results["val_lds"])
    results["best_lambda"] = results["lambda_values"][best_idx]
    results["best_lds"] = results["val_lds"][best_idx]
    results["best_faithfulness"] = results["faithfulness"][best_idx]

    return results


def run_joint_sweep(
    train_grad_cache: GradientCache,
    val_grad_cache: GradientCache,
    test_grad_cache: GradientCache,
    val_gt: tuple,
    test_gt: tuple,
    eigenvalues: torch.Tensor,
    K: torch.Tensor,
    m_values: List[int],
    lambda_values: List[float],
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    faithfulness_threshold: float = 0.1,
    num_test_samples: int = 200,
) -> Dict:
    """
    Run the full joint sweep experiment with empirical faithfulness measurement.

    Args:
        train_grad_cache: Training gradient cache
        val_grad_cache: Validation gradient cache
        test_grad_cache: Test gradient cache
        val_gt: Validation ground truth for LDS
        test_gt: Test ground truth for LDS
        eigenvalues: Pre-computed eigenvalues of full Gram matrix
        K: Pre-computed Gram matrix (for exact score computation)
        m_values: List of projection dimensions to test
        lambda_values: List of λ values to sweep for each m
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for computation
        faithfulness_threshold: Threshold for empirical faithfulness
        num_test_samples: Number of samples for faithfulness measurement

    Returns:
        Dictionary with comprehensive results
    """
    print(f"\n{'='*60}")
    print("Joint Sweep: Faithfulness-Utility Alignment (Empirical)")
    print(f"{'='*60}")
    print(f"  m values: {len(m_values)} ({min(m_values)} to {max(m_values)})")
    print(f"  λ values: {len(lambda_values)} ({min(lambda_values):.1e} to {max(lambda_values):.1e})")
    print(f"  Faithfulness threshold: {faithfulness_threshold}")

    # Compute exact scores ONCE (reused across all m values)
    print(f"\n[1/3] Computing exact influence scores...")
    exact_scores = compute_exact_self_scores(
        grad_cache=train_grad_cache,
        K=K,
        lambda_values=lambda_values,
        device=device,
        batch_size=batch_size,
        num_test_samples=num_test_samples,
    )

    # Compute d_λ for all λ values
    print(f"\n[2/3] Computing effective dimensions d_λ...")
    d_lambda_dict = {}
    for lamb in lambda_values:
        d_lambda_dict[lamb] = compute_effective_dimension(eigenvalues, lamb)

    # Main loop: sweep m, then λ
    print(f"\n[3/3] Running joint sweep...")

    results = {
        "m_values": m_values,
        "lambda_values": lambda_values,
        "faithfulness_threshold": faithfulness_threshold,
        "d_lambda": d_lambda_dict,
        "grid": {},                    # Full (m, λ) -> LDS grid
        "faithfulness_grid": {},       # Full (m, λ) -> faithfulness metrics
        "per_m_results": {},
    }

    for m in tqdm(m_values, desc="Sweeping m"):
        sweep_result = run_lambda_sweep_for_m(
            train_grad_cache=train_grad_cache,
            val_grad_cache=val_grad_cache,
            val_gt=val_gt,
            proj_dim=m,
            lambda_values=lambda_values,
            exact_scores=exact_scores,
            proj_type=proj_type,
            device=device,
            seed=seed,
            batch_size=batch_size,
            faithfulness_threshold=faithfulness_threshold,
            num_test_samples=num_test_samples,
        )

        # Store full grids
        for i, lamb in enumerate(lambda_values):
            results["grid"][(m, lamb)] = sweep_result["val_lds"][i]
            results["faithfulness_grid"][(m, lamb)] = sweep_result["faithfulness"][i]

        # Store per-m summary
        lambda_star = sweep_result["best_lambda"]
        lds_star = sweep_result["best_lds"]
        faith_at_star = sweep_result["best_faithfulness"]

        # Find the faithful region for this m
        faithful_lambdas = [
            lamb for i, lamb in enumerate(lambda_values)
            if sweep_result["faithfulness"][i].get("is_faithful", False)
        ]

        if faithful_lambdas:
            min_faithful_lambda = min(faithful_lambdas)
        else:
            min_faithful_lambda = float('inf')

        in_unfaithful_region = not faith_at_star.get("is_faithful", False)

        results["per_m_results"][m] = {
            "lambda_star": lambda_star,
            "lds_star": lds_star,
            "faithfulness_at_star": faith_at_star,
            "in_unfaithful_region": in_unfaithful_region,
            "min_faithful_lambda": min_faithful_lambda,
            "all_lds": sweep_result["val_lds"],
            "all_faithfulness": sweep_result["faithfulness"],
        }

        # Progress output
        region_str = "UNFAITHFUL" if in_unfaithful_region else "faithful"
        mean_ratio = faith_at_star.get("mean_ratio", float('nan'))
        tqdm.write(f"  m={m:6d}: λ*(m)={lambda_star:.2e}, LDS*={lds_star:.4f}, "
                   f"ratio={mean_ratio:.3f} [{region_str}]")

    # Evaluate best configuration on test set
    print(f"\n[4/4] Evaluating on test set...")

    best_m = max(results["per_m_results"].keys(),
                 key=lambda m: results["per_m_results"][m]["lds_star"])
    best_lambda = results["per_m_results"][best_m]["lambda_star"]

    test_sweep = run_lambda_sweep_for_m(
        train_grad_cache=train_grad_cache,
        val_grad_cache=test_grad_cache,
        val_gt=test_gt,
        proj_dim=best_m,
        lambda_values=[best_lambda],
        exact_scores={best_lambda: exact_scores[best_lambda]},
        proj_type=proj_type,
        device=device,
        seed=seed,
        batch_size=batch_size,
        faithfulness_threshold=faithfulness_threshold,
        num_test_samples=num_test_samples,
    )

    results["best_config"] = {
        "m": best_m,
        "lambda": best_lambda,
        "val_lds": results["per_m_results"][best_m]["lds_star"],
        "test_lds": test_sweep["val_lds"][0],
    }

    print(f"\nBest configuration: m={best_m}, λ={best_lambda:.2e}")
    print(f"  Val LDS:  {results['best_config']['val_lds']:.4f}")
    print(f"  Test LDS: {results['best_config']['test_lds']:.4f}")

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

    # Question 2: Monotonicity
    lds_stars = [per_m[m]["lds_star"] for m in m_values]

    monotonic_violations = 0
    for i in range(1, len(lds_stars)):
        if lds_stars[i] < lds_stars[i-1] - 0.01:
            monotonic_violations += 1

    correlation = np.corrcoef(m_values, lds_stars)[0, 1]

    analysis["monotonicity"] = {
        "lds_stars": lds_stars,
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
    print("="*60)

    print("\n[Question 1: Alignment]")
    align = analysis["alignment"]
    print(f"  λ*(m) empirically unfaithful: {align['n_in_unfaithful_region']}/{align['n_total']} "
          f"({align['fraction_unfaithful']*100:.1f}%)")
    print(f"  Conclusion: {align['conclusion']}")

    print("\n[Question 2: Monotonicity]")
    mono = analysis["monotonicity"]
    print(f"  Correlation(m, LDS*): {mono['correlation_m_lds']:.3f}")
    print(f"  Monotonic violations: {mono['monotonic_violations']}")
    print(f"  Conclusion: {mono['conclusion']}")


# =============================================================================
# Data Loading Utilities
# =============================================================================

def get_validation_split_indices(test_sampler, val_ratio=0.1, seed=0):
    """Split test set into validation and test."""
    test_indices = list(test_sampler)
    num_test = len(test_indices)

    np.random.seed(seed)
    np.random.shuffle(test_indices)
    num_val = int(val_ratio * num_test)
    val_indices = test_indices[:num_val]
    new_test_indices = test_indices[num_val:]

    return val_indices, new_test_indices


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

    model = model_details["model"]
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

    # Compute exact Gram matrix and eigenspectrum
    print("\n[Setup] Computing exact Gram matrix and eigenspectrum...")
    eigenvalues, K = compute_exact_gram_matrix(
        train_grad_cache, device=device, batch_size=batch_size * 4
    )
    print(f"  Eigenvalues: max={eigenvalues[0]:.2e}, "
          f"rank={torch.sum(eigenvalues > 1e-10 * eigenvalues[0]).item()}")

    # Run the joint sweep
    results = run_joint_sweep(
        train_grad_cache=train_grad_cache,
        val_grad_cache=val_grad_cache,
        test_grad_cache=test_grad_cache,
        val_gt=val_gt,
        test_gt=test_gt,
        eigenvalues=eigenvalues,
        K=K,
        m_values=m_values,
        lambda_values=lambda_values,
        proj_type=proj_type,
        device=device,
        seed=seed,
        batch_size=batch_size,
        faithfulness_threshold=faithfulness_threshold,
        num_test_samples=num_test_samples,
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
    parser.add_argument("--faithfulness_threshold", type=float, default=0.1,
                       help="Threshold for empirical faithfulness (default 0.1 = 10% error)")
    parser.add_argument("--num_test_samples", type=int, default=200,
                       help="Number of samples for faithfulness measurement")

    # Sweep configuration
    parser.add_argument("--m_min", type=int, default=32,
                       help="Minimum projection dimension")
    parser.add_argument("--m_max", type=int, default=16384,
                       help="Maximum projection dimension")
    parser.add_argument("--m_steps", type=int, default=10,
                       help="Number of m values (log-spaced)")
    parser.add_argument("--lambda_min", type=float, default=1e-5,
                       help="Minimum λ value")
    parser.add_argument("--lambda_max", type=float, default=1e2,
                       help="Maximum λ value")
    parser.add_argument("--lambda_steps", type=int, default=50,
                       help="Number of λ values (log-spaced)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate sweep grids
    m_values = np.unique(np.logspace(
        np.log10(args.m_min),
        np.log10(args.m_max),
        num=args.m_steps
    ).astype(int)).tolist()

    lambda_values = np.logspace(
        np.log10(args.lambda_min),
        np.log10(args.lambda_max),
        num=args.lambda_steps
    ).tolist()

    print(f"\nConfiguration:")
    print(f"  m values: {m_values}")
    print(f"  λ range: [{args.lambda_min:.1e}, {args.lambda_max:.1e}] ({args.lambda_steps} steps)")
    print(f"  Faithfulness threshold: {args.faithfulness_threshold}")

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
        faithfulness_threshold=args.faithfulness_threshold,
        num_test_samples=args.num_test_samples,
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

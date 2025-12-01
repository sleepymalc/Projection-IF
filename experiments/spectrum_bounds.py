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
from utils.fisher_utils import (
    compute_eigenspectrum,
    project_gradients_to_cpu,
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
    batch_size: int = 64
) -> Tuple[Tensor, dict, Tensor]:
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

    Returns:
        eigenvalues: Eigenvalues of K in descending order (float64)
        effective_dims: Dictionary mapping λ -> d_λ
        K: Gram matrix (n, n) in float64, normalized by 1/n
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

    # Compute eigenvalues for effective dimension calc (in float64)
    eigenvalues = torch.linalg.eigvalsh(K)
    eigenvalues = eigenvalues.flip(0).clamp(min=0.0)

    # Compute effective dimensions
    effective_dims = {}
    for lamb in lambda_values:
        d_lambda = torch.sum(eigenvalues / (eigenvalues + lamb)).item()
        effective_dims[lamb] = d_lambda

    return eigenvalues, effective_dims, K


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
    """
    eigenvalues: Tensor      # (n,) eigenvalues of K_proj
    eigenvectors: Tensor     # (n, n) eigenvectors of K_proj
    U: Tensor                # (n, k) matrix for Woodbury
    pv_norms_sq: Tensor      # (k,) squared norms of projected test vectors
    n: int                   # number of training samples


def precompute_woodbury(
    grad_cache: GradientCache,
    projector,
    test_vectors: Tensor,
    device: str = "cuda",
    batch_size: int = 64,
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

    Returns:
        WoodburyPrecomputed containing all precomputed quantities
    """
    n = grad_cache.n_samples
    k = test_vectors.shape[0]

    # Step 1: Project test vectors → PV (k, m) in float64 for precision
    PV = projector.project(test_vectors, ensemble_id=0).to(dtype=torch.float64, device=device)
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
        pg_batch = projector.project(g_batch, ensemble_id=0)
        pg_storage.append(pg_batch.cpu())  # Store on CPU in float32

        # Compute U for this batch: U[i] = pg_i @ PV.T
        pg_batch_f64 = pg_batch.to(dtype=torch.float64)
        U[i_start:i_end] = pg_batch_f64 @ PV.T

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

    # Clean up storage
    del pg_storage

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
    )


def compute_scores(
    A: Tensor,
    B: Tensor,
    grad_cache: GradientCache,
    lamb: float,
    device: str,
    batch_size: int,
    K: Tensor,
) -> Tensor:
    """
    Compute exact influence scores in Float64 to ensure valid baseline.

    For self-scores, pass A=B and use result.diag().

    Uses Woodbury identity:
        scores[i,j] = (1/λ)[a_i^T b_j - u_a[i]^T (K + λI)^{-1} u_b[j]]
    where u_a = G @ a, u_b = G @ b, and K is normalized (1/n G G^T).

    Note: K should be the normalized Gram matrix from compute_robust_gram().
    The math: Score = (1/λ)[ a.b - u_a^T (G G^T + nλI)^-1 u_b ]
              G G^T = n * K, so matrix = n*K + n*λ*I = n(K + λI)
              Inverse = (1/n)(K + λI)^-1
              We solve (K + λI) X = U/n to get X = (K + λI)^-1 (U/n)

    Args:
        A: Left vectors of shape (k_A, d) on device
        B: Right vectors of shape (k_B, d) on device
        grad_cache: GradientCache containing the gradients
        lamb: Regularization parameter
        device: GPU device
        batch_size: Batch size for streaming
        K: Pre-computed Gram matrix (n, n), normalized by 1/n, in float64

    Returns:
        Tensor of shape (k_A, k_B) containing influence scores
    """
    n = grad_cache.n_samples

    # 1. Cast inputs to float64 for precision
    A_dbl = A.to(dtype=torch.float64, device=device)
    B_dbl = B.to(dtype=torch.float64, device=device)
    K_dbl = K.to(dtype=torch.float64, device=device)
    lamb_dbl = float(lamb)

    k_A = A.shape[0]
    k_B = B.shape[0]

    # 2. Compute U_A = G @ A^T in float64
    U_A = torch.zeros(n, k_A, device=device, dtype=torch.float64)
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        G_batch = grad_cache.get_batch(i_start, i_end, device=device)
        # Cast to double for precision
        U_A[i_start:i_end, :] = G_batch.to(dtype=torch.float64) @ A_dbl.T

    # Check if A and B are the same tensor (self-scores optimization)
    if A.data_ptr() == B.data_ptr():
        U_B = U_A
    else:
        U_B = torch.zeros(n, k_B, device=device, dtype=torch.float64)
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            G_batch = grad_cache.get_batch(i_start, i_end, device=device)
            U_B[i_start:i_end, :] = G_batch.to(dtype=torch.float64) @ B_dbl.T

    # 3. Solve (K + λI) X = U/n in float64
    # K is already normalized by 1/n in compute_robust_gram
    reg_matrix = K_dbl + lamb_dbl * torch.eye(n, device=device, dtype=torch.float64)
    rhs = U_B / n  # Scale by 1/n to account for the normalization
    X_B = torch.linalg.solve(reg_matrix, rhs)  # (n, k_B)

    # 4. Compute scores: (A.B - U_A^T @ X_B) / λ
    A_dot_B = A_dbl @ B_dbl.T  # (k_A, k_B)
    cross_term = U_A.T @ X_B  # (k_A, k_B)
    scores = (A_dot_B - cross_term) / lamb_dbl

    return scores


# =============================================================================
# Sandwich Bounds (local to this experiment)
# =============================================================================

def compute_sandwich_bounds(
    lamb: float,
    exact_scores: Tensor,
    precomputed: WoodburyPrecomputed,
) -> dict:
    """
    Compute sandwich bounds for sketched inverse approximation quality.

    Uses the Woodbury method with precomputed eigendecomposition for O(n²) solve.

    Args:
        lamb: Regularization parameter λ
        exact_scores: Pre-computed exact scores (k,)
        precomputed: WoodburyPrecomputed from precompute_woodbury()

    Returns:
        Dictionary with exact_scores, sketched_scores, ratios, and statistics
    """
    # Use precomputed eigendecomposition for O(n²) solve
    inv_eigvals = 1.0 / (precomputed.eigenvalues + lamb)  # (n,)
    W = precomputed.eigenvectors.T @ precomputed.U  # (n, k)
    X = precomputed.eigenvectors @ (inv_eigvals.unsqueeze(1) * W)  # (n, k)

    # score_i = (1/λ)[||Pv_i||² - u_i^T x_i]
    ux_dots = (precomputed.U * X).sum(dim=0)  # (k,)
    sketched_scores = (precomputed.pv_norms_sq - ux_dots) / lamb

    exact_scores = exact_scores.detach().cpu()
    sketched_scores = sketched_scores.detach().cpu()

    # Cast to same dtype for ratio computation (use float64 for precision)
    exact_scores_f64 = exact_scores.to(torch.float64)
    sketched_scores_f64 = sketched_scores.to(torch.float64)

    # Use relative threshold based on score magnitude to handle different scales
    max_score = exact_scores_f64.abs().max()
    relative_threshold = max(1e-10 * max_score.item(), 1e-12)
    valid_mask = exact_scores_f64.abs() > relative_threshold

    ratios = torch.zeros_like(exact_scores_f64)
    ratios[valid_mask] = sketched_scores_f64[valid_mask] / exact_scores_f64[valid_mask]

    n_valid = valid_mask.sum().item()
    n_total = len(exact_scores)

    return {
        "exact_scores": exact_scores,
        "sketched_scores": sketched_scores,
        "ratios": ratios,
        "mean_ratio": ratios[valid_mask].mean().item() if valid_mask.any() else float('nan'),
        "std_ratio": ratios[valid_mask].std().item() if valid_mask.any() else float('nan'),
        "max_ratio": ratios[valid_mask].max().item() if valid_mask.any() else float('nan'),
        "min_ratio": ratios[valid_mask].min().item() if valid_mask.any() else float('nan'),
        "n_valid": n_valid,
        "n_total": n_total,
        "valid_fraction": n_valid / n_total if n_total > 0 else 0.0,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    grad_cache: GradientCache,
    lambda_values: list,
    m_multipliers: list,
    proj_type: ProjectionType = ProjectionType.normal,
    num_trials: int = 5,
    num_test_grads: int = 200,
    seed: int = 42,
    device: str = "cuda",
    batch_size: int = 32,
    min_m: int = 10,
    min_d_lambda: float = 5.0,
) -> dict:
    """

    Args:
        grad_cache: GradientCache
        lambda_values: List of regularization values
        m_multipliers: Multipliers of d_λ to test
        proj_type: Type of random projection
        num_trials: Number of random projection trials per configuration
        num_test_grads: Number of gradients to use for testing
        seed: Random seed
        device: GPU device
        batch_size: Batch size for GPU operations
        min_m: Minimum projection dimension
        min_d_lambda: Skip λ values where d_λ < this threshold

    Returns:
        Dictionary with all results
    """
    n = grad_cache.n_samples
    d = grad_cache.dim

    print(f"\nRunning experiment:")
    print(f"  - batch_size = {batch_size}")
    print(f"  - n = {n}, d = {d:,}")
    print(f"  - min_m = {min_m}, min_d_lambda = {min_d_lambda}")
    print(f"  - {len(m_multipliers)} dim ratio × {num_trials} trials × {len(lambda_values)} λ")

    # Compute eigenvalues, effective dimensions, and Gram matrix in FLOAT64
    # This fixes mixed precision instability where float32 causes catastrophic
    # cancellation in influence score computation for high-dimensional models
    print(f"\nComputing eigenvalues and Gram matrix (Robust Float64)...")
    eigenvalues, eff_dims, K = compute_robust_gram(
        grad_cache, lambda_values, device=device, batch_size=batch_size*4
    )

    eig_nonzero = eigenvalues[eigenvalues > 1e-10]
    print(f"  Eigenvalue stats: max={eigenvalues[0]:.2e}, min_nonzero={eig_nonzero[-1]:.2e}, "
          f"n_nonzero={len(eig_nonzero)}/{len(eigenvalues)}")
    # K is already normalized by 1/n and in float64 - no unnormalization needed
    # The new compute_scores() expects the normalized K

    # Compute numerical rank
    threshold = 1e-10 * eigenvalues[0]
    rank = (eigenvalues > threshold).sum().item()

    # Prepare test vectors
    test_vectors = [grad_cache.get_sample(i, device="cpu") for i in range(min(num_test_grads, n))]
    V = torch.stack(test_vectors).to(device)  # (k, d)

    # Filter valid λ values (d_λ >= min_d_lambda)
    valid_lambdas = [lamb for lamb in lambda_values if eff_dims[lamb] >= min_d_lambda]
    skipped_lambdas = [lamb for lamb in lambda_values if eff_dims[lamb] < min_d_lambda]

    if skipped_lambdas:
        print(f"\nSkipping {len(skipped_lambdas)} λ values with d_λ < {min_d_lambda}")

    # Precompute exact scores for all λ values (doesn't depend on m or trial)
    # Note: compute_scores now uses float64 internally for numerical stability
    print(f"\nPrecomputing exact scores for {len(valid_lambdas)} λ values...")
    exact_scores_by_lambda = {}
    for lamb in valid_lambdas:
        exact_scores_by_lambda[lamb] = compute_scores(V, V, grad_cache, lamb, device, batch_size, K).diag()
    print(f"  Done. Exact scores cached for all λ values.")

    # Collect all unique m values and build config mapping
    # m_to_configs: m -> list of config dicts
    # all_config_ratios: (m, lamb, mult) -> list of ratios (accumulated across trials)
    m_to_configs = {}
    all_config_ratios = {}

    for lamb in valid_lambdas:
        d_lambda = eff_dims[lamb]
        for mult in m_multipliers:
            m_raw = mult * d_lambda
            m = max(min_m, min(int(m_raw), d))
            m_clamped = (m == min_m and m_raw < min_m)

            if m not in m_to_configs:
                m_to_configs[m] = []
            m_to_configs[m].append({
                "lamb": lamb,
                "mult": mult,
                "d_lambda": d_lambda,
                "m_raw": m_raw,
                "m_clamped": m_clamped,
            })
            all_config_ratios[(m, lamb, mult)] = []

    m_values_sorted = sorted(m_to_configs.keys())
    print(f"\nUnique projection dimension to test: {len(m_values_sorted)}")

    # Main loop: trial → m → λ
    for trial in range(num_trials):
        trial_seed = seed + trial
        print(f"\nTrial {trial + 1}/{num_trials} (seed={trial_seed})")

        for m in tqdm(m_values_sorted, desc="  projection dimension"):
            configs = m_to_configs[m]

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

            # Always precompute for Woodbury path (more numerically stable than direct m×m)
            precomputed = precompute_woodbury(
                grad_cache=grad_cache,
                projector=projector,
                test_vectors=V,
                device=device,
                batch_size=batch_size,
            )

            # Compute bounds for each lambda using precomputed Woodbury quantities
            for config in configs:
                lamb = config["lamb"]
                bounds = compute_sandwich_bounds(
                    lamb=lamb,
                    exact_scores=exact_scores_by_lambda[lamb],
                    precomputed=precomputed,
                )
                ratios = bounds["ratios"].numpy()
                valid_ratios = ratios[~np.isnan(ratios)]
                all_config_ratios[(m, lamb, config["mult"])].extend(valid_ratios.tolist())

    # Build results dict with metadata and aggregated experiment results
    print("Aggregating results...")
    results = {
        "n_samples": n,
        "d_params": d,
        "proj_type": proj_type.value,
        "lambda_values": lambda_values,
        "m_multipliers": m_multipliers,
        "effective_dims": eff_dims,
        "batch_size": batch_size,
        "min_m": min_m,
        "min_d_lambda": min_d_lambda,
        "skipped_configs": [
            {"lambda": lamb, "d_lambda": eff_dims[lamb],
             "reason": f"d_lambda ({eff_dims[lamb]:.2f}) < min_d_lambda ({min_d_lambda})"}
            for lamb in skipped_lambdas
        ],
        "use_held_out_gradients": False,
        "eigenvalues": eigenvalues.cpu().numpy(),
        "rank": rank,
        "experiments": [],
    }

    for m, configs in m_to_configs.items():
        for config in configs:
            ratios_list = all_config_ratios[(m, config["lamb"], config["mult"])]
            if len(ratios_list) == 0:
                continue

            all_ratios = np.array(ratios_list)
            eps_values = np.abs(all_ratios - 1)

            # Build result by extending config with computed statistics
            results["experiments"].append({
                "m": m,
                "lambda": config["lamb"],
                "multiplier": config["mult"],
                "d_lambda": config["d_lambda"],
                "m_raw": config["m_raw"],
                "m_clamped": config["m_clamped"],
                "m_over_d_lambda": m / config["d_lambda"] if config["d_lambda"] > 0 else float('inf'),
                # Ratio statistics
                "mean_ratio": np.mean(all_ratios),
                "std_ratio": np.std(all_ratios),
                "min_ratio": np.min(all_ratios),
                "max_ratio": np.max(all_ratios),
                # Epsilon statistics
                "empirical_eps_max": np.max(eps_values),
                "empirical_eps_mean": np.mean(eps_values),
                "empirical_eps_std": np.std(eps_values),
                "empirical_eps_95": np.percentile(eps_values, 95),
                "empirical_eps_99": np.percentile(eps_values, 99),
                "n_samples": len(all_ratios),
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
    parser.add_argument("--min_m", type=int, default=1,
                       help="Minimum projection dimension to avoid numerical degeneracy. "
                            "When mult * d_λ < min_m, m is clamped to min_m. (default: 1)")
    parser.add_argument("--min_d_lambda", type=float, default=1.0,
                       help="Skip λ values where d_λ < this threshold (numerically degenerate). "
                            "(default: 1.0)")

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
    nn_model = model_details["model"]
    train_dataset = model_details["train_dataset"]
    train_sampler = model_details["train_sampler"]
    indices = list(train_sampler)
    n_samples = len(indices)

    # For disk mode, check if cache is valid and reuse
    # Note: is_valid() already loads metadata when it returns True
    if args.offload == "disk" and grad_cache.is_valid(expected_samples=n_samples):
        print(f"Loading cached gradients from {args.cache_dir}...")
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

    # Experiment configuration
    lambda_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    m_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0]

    # Run experiment
    results = run_experiment(
        grad_cache=grad_cache,
        lambda_values=lambda_values,
        m_multipliers=m_multipliers,
        proj_type=proj_type,
        num_trials=args.num_trials,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        min_m=args.min_m,
        min_d_lambda=args.min_d_lambda,
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
    results_filename = f"{args.dataset}_{args.model}_{args.proj_type}.pt"
    results_path = os.path.join(experiment_dir, results_filename)
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

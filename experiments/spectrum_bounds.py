"""
Spectrum Bounds Validation: Validate Regularized Sketching Bounds
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from typing import Union
from torch import Tensor

from dattri.func.projection import (
    make_random_projector,
    ProjectionType,
)
from dattri.benchmark.load import load_benchmark
from utils.fisher_utils import (
    compute_eigenspectrum,
    project_gradients_to_cpu,
    compute_kernel_from_projected,
)
from utils.gradient_cache import GradientCache


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
    Precompute quantities for fast Woodbury solves across multiple λ values.

    This function:
    1. Projects test vectors → PV
    2. Projects all gradients → PG, computing U = PG @ PV^T simultaneously
    3. Computes K_proj = (1/n) PG @ PG^T
    4. Computes eigendecomposition of K_proj

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

    # Step 1: Project test vectors → PV (k, m)
    PV = projector.project(test_vectors, ensemble_id=0)
    pv_norms_sq = (PV ** 2).sum(dim=1)  # (k,)

    # Step 2: Project all gradients and compute U = PG @ PV^T simultaneously
    PG_cpu, U = project_gradients_to_cpu(grad_cache, projector, device, batch_size, test_vectors=PV)

    # Scale U by 1/√n
    U = U / (n ** 0.5)

    # Step 3: Compute kernel K_proj = (1/n) PG @ PG^T
    K_proj = compute_kernel_from_projected(PG_cpu, device, batch_size, normalize=True)

    # Step 4: Eigendecomposition (O(n³) but done only once)
    eigenvalues, eigenvectors = torch.linalg.eigh(K_proj)

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
    Compute influence scores A^T (F + λI)^{-1} B (unsketched).

    For self-scores, pass A=B and use result.diag().

    Uses Woodbury identity:
        scores[i,j] = (1/λ)[a_i^T b_j - u_a[i]^T (K + nλI)^{-1} u_b[j]]
    where u_a = G @ a and u_b = G @ b.

    Args:
        A: Left vectors of shape (k_A, d) on device
        B: Right vectors of shape (k_B, d) on device
        grad_cache: GradientCache containing the gradients
        lamb: Regularization parameter
        device: GPU device
        batch_size: Batch size for streaming
        K: Pre-computed Gram matrix (n, n), unnormalized

    Returns:
        Tensor of shape (k_A, k_B) containing influence scores
    """
    k_A = A.shape[0]
    k_B = B.shape[0]
    dtype = A.dtype
    n = grad_cache.n_samples

    # Compute U_A = G @ A^T -> (n, k_A)
    U_A = torch.zeros(n, k_A, device=device, dtype=dtype)
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        G_batch = grad_cache.get_batch(i_start, i_end, device=device)
        U_A[i_start:i_end, :] = G_batch @ A.T

    # Check if A and B are the same tensor (self-scores optimization)
    same_AB = A.data_ptr() == B.data_ptr()

    if same_AB:
        U_B = U_A
    else:
        # Compute U_B = G @ B^T -> (n, k_B)
        U_B = torch.zeros(n, k_B, device=device, dtype=dtype)
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            G_batch = grad_cache.get_batch(i_start, i_end, device=device)
            U_B[i_start:i_end, :] = G_batch @ B.T

    # Solve (nλI + K) X_B = U_B
    K_dev = K.to(device)
    reg_matrix = K_dev + n * lamb * torch.eye(n, device=device, dtype=K_dev.dtype)
    X_B = torch.linalg.solve(reg_matrix, U_B)  # (n, k_B)

    # Compute scores: scores[i,j] = (1/λ)[a_i^T b_j - u_a[i]^T x_b[j]]
    A_dot_B = A @ B.T  # (k_A, k_B)
    cross_term = U_A.T @ X_B  # (k_A, k_B)
    scores = (A_dot_B - cross_term) / lamb

    return scores


# =============================================================================
# Sandwich Bounds (local to this experiment)
# =============================================================================

def compute_sandwich_bounds(
    grad_cache: GradientCache,
    projector,
    lamb: float,
    test_vectors: Union[list, Tensor],
    exact_scores: Tensor,
    device: str = "cuda",
    batch_size: int = 500,
    precomputed: WoodburyPrecomputed = None,
) -> dict:
    """
    Compute sandwich bounds for sketched inverse approximation quality.

    Uses adaptive strategy: kernel method (Woodbury) when m > n to avoid
    allocating the m×m matrix PFP^T.

    Args:
        grad_cache: GradientCache containing the gradients
        projector: dattri projector with .project() method (from make_random_projector)
        lamb: Regularization parameter λ
        test_vectors: Test vectors (list of tensors or stacked tensor)
        exact_scores: Pre-computed exact scores
        device: GPU device for computation
        batch_size: Batch size for streaming operations
        precomputed: Optional WoodburyPrecomputed from precompute_woodbury().
                     If provided and m > n, uses O(n²) solve instead of O(n³).

    Returns:
        Dictionary with exact_scores, sketched_scores, ratios, and statistics
    """
    n = grad_cache.n_samples
    m = projector.proj_dim
    dtype = grad_cache.dtype

    # Convert test vectors to tensor if needed
    if isinstance(test_vectors, list):
        V = torch.stack(test_vectors).to(device)  # (k, d)
    else:
        V = test_vectors.to(device)

    # Adaptive strategy for sketched scores:
    if m > n: # Use Woodbury method
        assert precomputed is not None
        # Use precomputed eigendecomposition for O(n²) solve
        inv_eigvals = 1.0 / (precomputed.eigenvalues + lamb)  # (n,)
        W = precomputed.eigenvectors.T @ precomputed.U  # (n, k)
        X = precomputed.eigenvectors @ (inv_eigvals.unsqueeze(1) * W)  # (n, k)

        # score_i = (1/λ)[||Pv_i||² - u_i^T x_i]
        ux_dots = (precomputed.U * X).sum(dim=0)  # (k,)
        sketched_scores = (precomputed.pv_norms_sq - ux_dots) / lamb
    else:
        # Standard method: form m×m PFP^T directly and solve
        PFPT = torch.zeros(m, m, device=device, dtype=dtype)

        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            G_batch = grad_cache.get_batch(i_start, i_end, device=device)
            PG_batch = projector.project(G_batch, ensemble_id=0)
            PFPT += PG_batch.T @ PG_batch

        PFPT /= n
        PFPT.diagonal().add_(lamb)

        # Project test vectors and solve
        PV = projector.project(V, ensemble_id=0)
        X = torch.linalg.solve(PFPT, PV.T)  # (m, k)
        sketched_scores = (PV * X.T).sum(dim=1)  # (k,)

    exact_scores = exact_scores.detach().cpu()
    sketched_scores = sketched_scores.detach().cpu()

    # Use relative threshold based on score magnitude to handle different scales
    max_score = exact_scores.abs().max()
    relative_threshold = max(1e-10 * max_score.item(), 1e-12)
    valid_mask = exact_scores.abs() > relative_threshold

    ratios = torch.zeros_like(exact_scores)
    ratios[valid_mask] = sketched_scores[valid_mask] / exact_scores[valid_mask]

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

    # Compute eigenvalues, effective dimensions, and Gram matrix together
    print(f"\nComputing eigenvalues and Gram matrix...")
    eigenvalues, eff_dims, K = compute_eigenspectrum(
        grad_cache, lambda_values, device=device, batch_size=batch_size*4, return_gram=True
    )

    eig_nonzero = eigenvalues[eigenvalues > 1e-10]
    print(f"  Eigenvalue stats: max={eigenvalues[0]:.2e}, min_nonzero={eig_nonzero[-1]:.2e}, "
          f"n_nonzero={len(eig_nonzero)}/{len(eigenvalues)}")
    # Unnormalize for Woodbury formula
    K_unnorm = K * n

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
    print(f"\nPrecomputing exact scores for {len(valid_lambdas)} λ values...")
    exact_scores_by_lambda = {}
    for lamb in valid_lambdas:
        exact_scores_by_lambda[lamb] = compute_scores(V, V, grad_cache, lamb, device, batch_size, K_unnorm).diag()
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

            # For Woodbury (m > n), precompute eigendecomposition once for O(n²) solves
            precomputed = None
            if m > n: # Use Woodbury method
                precomputed = precompute_woodbury(
                    grad_cache=grad_cache,
                    projector=projector,
                    test_vectors=V,
                    device=device,
                    batch_size=batch_size,
                )

            # Unified loop: compute_sandwich_bounds handles both paths
            for config in configs:
                lamb = config["lamb"]
                bounds = compute_sandwich_bounds(
                    grad_cache=grad_cache,
                    projector=projector,
                    lamb=lamb,
                    test_vectors=test_vectors,
                    exact_scores=exact_scores_by_lambda[lamb],
                    device=device,
                    batch_size=batch_size,
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
    parser.add_argument("--min_d_lambda", type=float, default=5.0,
                       help="Skip λ values where d_λ < this threshold (numerically degenerate). "
                            "(default: 5.0)")

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
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
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

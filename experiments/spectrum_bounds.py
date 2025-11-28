"""
Spectrum Bounds Validation: Validate Regularized Sketching Bounds

Validates that the sketched influence score approximation satisfies the (1±ε) sandwich bounds
for vectors in range(F), as claimed by the regularized sketching theorem.

Theory:
    For g in range(F):
        (1-ε) g^T M(λ) g ≤ g^T S(λ) g ≤ (1+ε) g^T M(λ) g

    where:
        M(λ) = (F + λI)^{-1}           (true regularized inverse)
        S(λ) = P^T (PFP^T + λI)^{-1} P  (sketched approximation)
        ε scales as O(√(d_λ/m)) for m >= d_λ

Key Questions:
    1. For given (m, λ), what is the empirical ε?
    2. Does ε scale as O(1/√(m/d_λ)) as predicted?
    3. Is m ≈ d_λ/ε² sufficient for ε-accurate approximation?

Usage:
    python spectrum_bounds.py --dataset mnist --model mlp
    python spectrum_bounds.py --dataset maestro --model musictransformer --offload disk

Interpreting Results:
    - "ratio" = sketched_score / exact_score (should be ≈ 1.0)
    - "ε_95" = 95th percentile of |ratio - 1| (main metric)
    - When m/d_λ ≥ 1, expect ε < 0.3 (reasonable approximation)
    - When m/d_λ ≥ 4, expect ε < 0.1 (good approximation)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add experiments directory to path for local utils
experiments_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(experiments_dir))

from utils.fisher_utils import (
    compute_eigenvalues,
    compute_effective_dimension,
    compute_sandwich_bounds,
    compute_gram_matrix,
    compute_per_sample_gradients,
    clear_memory,
)
from dattri.func.projection import (
    make_random_projector,
    ProjectionType,
)
from utils.gradient_cache import GradientCache


def load_gradients(
    dataset: str,
    model: str,
    device: str,
    n_samples: int,
    batch_size: int = 32,
) -> list:
    """
    Load model and compute per-sample gradients using vmap for efficiency.

    Args:
        dataset: Dataset name (e.g., "mnist", "cifar10", "maestro")
        model: Model name (e.g., "mlp", "resnet9", "musictransformer")
        device: Device for computation
        n_samples: Number of samples to compute gradients for
        batch_size: Batch size for vmap gradient computation

    Returns:
        List of gradient tensors on CPU, each of shape (d,)
    """
    from dattri.benchmark.load import load_benchmark

    model_details, _ = load_benchmark(model=model, dataset=dataset, metric="lds")
    nn_model = model_details["model"]
    train_dataset = model_details["train_dataset"]
    train_sampler = model_details["train_sampler"]
    indices = list(train_sampler)[:n_samples]

    model_type = "musictransformer" if model == "musictransformer" else "default"
    return compute_per_sample_gradients(
        model=nn_model,
        dataset=train_dataset,
        indices=indices,
        device=device,
        batch_size=batch_size,
        model_type=model_type,
    )


def compute_empirical_epsilon(
    gradients_cpu: list,
    lamb: float,
    m: int,
    proj_type: ProjectionType,
    num_trials: int = 5,
    num_test_grads: int = 200,
    seed: int = 42,
    device: str = "cuda",
    batch_size: int = 100,
    K: torch.Tensor = None,
    use_held_out_gradients: bool = False,
) -> dict:
    """
    Compute the empirical ε for a given (m, λ) configuration.

    By default, uses all gradients to form F and tests on a subset of those same
    gradients (which are truly in range(F)).

    Args:
        gradients_cpu: List of gradient tensors on CPU
        lamb: Regularization parameter
        m: Projection dimension
        proj_type: Type of random projection
        num_trials: Number of projection trials
        num_test_grads: Number of gradients to use for testing
        seed: Random seed
        device: GPU device
        batch_size: Batch size for GPU operations
        K: Pre-computed Gram matrix (n, n) for Woodbury. If None, computed on demand.
        use_held_out_gradients: If True, hold out test gradients from FIM construction
                                (old behavior). If False (default), use all gradients
                                for FIM and test on a subset of them.

    Returns:
        Dictionary with statistics about the approximation quality.
    """
    # Handle both list and GradientCache
    if hasattr(gradients_cpu, 'n_samples'):
        # GradientCache
        n = gradients_cpu.n_samples
        d = gradients_cpu.dim
        dtype = torch.float32  # Default dtype for GradientCache
    else:
        # List of tensors
        n = len(gradients_cpu)
        d = gradients_cpu[0].shape[0]
        dtype = gradients_cpu[0].dtype

    if use_held_out_gradients:
        # Old behavior: hold out some gradients from FIM construction
        # This mode requires list access - convert GradientCache to list
        if hasattr(gradients_cpu, 'to_list'):
            gradients_list = gradients_cpu.to_list(device="cpu")
        else:
            gradients_list = gradients_cpu
        n_fisher = n - num_test_grads
        gradients_fisher = gradients_list[:n_fisher]
        gradients_test = gradients_list[n_fisher:]

        # Compute Gram matrix K for Woodbury (efficient for exact score computation)
        if K is not None:
            # K was computed on full gradients, recompute for fisher subset
            print(f"  Computing Gram matrix for n={n_fisher} samples...")
            K_fisher = compute_gram_matrix(gradients_fisher, device=device, batch_size=batch_size)
            K_fisher = K_fisher * n_fisher  # Unnormalize for Woodbury formula
        else:
            K_fisher = None
    else:
        # Default: use all gradients for FIM, test on a subset of them
        gradients_fisher = gradients_cpu
        # Get test gradients - handle both list and GradientCache
        if hasattr(gradients_cpu, 'get_batch'):
            # GradientCache - use get_batch for efficient access
            gradients_test = [gradients_cpu.get_sample(i, device="cpu") for i in range(min(num_test_grads, n))]
        else:
            gradients_test = gradients_cpu[:num_test_grads]
        # Unnormalize K for Woodbury: K was computed as G G^T / n, but Woodbury needs G G^T
        K_fisher = K * n if K is not None else None

    all_ratios = []
    all_errors = []

    for trial in range(num_trials):
        trial_seed = seed + trial

        # Use dattri's make_random_projector factory
        # proj_max_batch_size must be multiple of 8
        proj_max_batch_size = min(batch_size, 32)
        proj_max_batch_size = max(8, (proj_max_batch_size // 8) * 8)

        projector = make_random_projector(
            param_shape_list=[d],  # flat gradients of dimension d
            feature_batch_size=batch_size,
            proj_dim=m,
            proj_max_batch_size=proj_max_batch_size,
            device=torch.device(device),
            proj_seed=trial_seed,
            proj_type=proj_type,
            dtype=dtype,
        )

        bounds = compute_sandwich_bounds(
            gradients=gradients_fisher,
            projector=projector,
            lamb=lamb,
            test_vectors=gradients_test,
            num_test_vectors=num_test_grads,
            device=device,
            batch_size=batch_size,
            K=K_fisher,
        )

        if hasattr(projector, 'free_memory'):
            projector.free_memory()
        del projector

        for ratio in bounds["ratios"].tolist():
            if not np.isnan(ratio):
                all_ratios.append(ratio)
                all_errors.append(abs(ratio - 1))

        clear_memory(device)

    # Cleanup (only delete K_fisher if it was newly computed, not if it's the passed-in K)
    if use_held_out_gradients and K_fisher is not None:
        del K_fisher
        clear_memory(device)

    all_ratios = np.array(all_ratios)
    all_errors = np.array(all_errors)

    # Compute empirical ε: max deviation from 1
    # Theorem says (1-ε) ≤ ratio ≤ (1+ε), so ε = max(|ratio - 1|)
    empirical_eps_max = np.max(np.abs(all_ratios - 1))
    empirical_eps_mean = np.mean(np.abs(all_ratios - 1))
    empirical_eps_std = np.std(np.abs(all_ratios - 1))

    # Percentiles for understanding distribution
    eps_95 = np.percentile(np.abs(all_ratios - 1), 95)
    eps_99 = np.percentile(np.abs(all_ratios - 1), 99)

    return {
        "m": m,
        "lambda": lamb,
        "mean_ratio": np.mean(all_ratios),
        "std_ratio": np.std(all_ratios),
        "min_ratio": np.min(all_ratios),
        "max_ratio": np.max(all_ratios),
        "empirical_eps_max": empirical_eps_max,
        "empirical_eps_mean": empirical_eps_mean,
        "empirical_eps_std": empirical_eps_std,
        "empirical_eps_95": eps_95,
        "empirical_eps_99": eps_99,
        "n_samples": len(all_ratios),
    }


def run_experiment(
    gradients_cpu: list,
    lambda_values: list,
    m_multipliers: list,
    proj_type: ProjectionType = ProjectionType.normal,
    num_trials: int = 5,
    seed: int = 42,
    device: str = "cuda",
    batch_size: int = 100,
    min_m: int = 10,
    min_d_lambda: float = 5.0,
    use_held_out_gradients: bool = False,
) -> dict:
    """
    Run the full experiment: for each λ, compute d_λ and test various m values.

    Args:
        gradients_cpu: List of gradient tensors on CPU
        lambda_values: List of regularization values
        m_multipliers: Multipliers of d_λ to test (e.g., [0.25, 0.5, 1, 2, 4])
        proj_type: Type of random projection
        num_trials: Number of random projection trials per configuration
        seed: Random seed
        device: GPU device
        batch_size: Batch size for GPU operations (increase for better utilization)
        min_m: Minimum projection dimension to avoid numerical degeneracy (default: 10)
        min_d_lambda: Skip λ values where d_λ < this threshold (default: 5.0)
        use_held_out_gradients: If True, hold out test gradients from FIM construction.
                                If False (default), use all gradients for FIM.

    Returns:
        Dictionary with all results
    """
    # Handle both list and GradientCache
    if hasattr(gradients_cpu, 'n_samples'):
        # GradientCache
        n = gradients_cpu.n_samples
        d = gradients_cpu.dim
    else:
        # List of tensors
        n = len(gradients_cpu)
        d = gradients_cpu[0].shape[0]

    print(f"\nRunning experiment with:")
    print(f"  - batch_size = {batch_size}")
    print(f"  - n = {n}, d = {d:,}")
    print(f"  - min_m = {min_m}, min_d_lambda = {min_d_lambda}")

    # Compute eigenvalues
    print(f"\nComputing eigenvalues...")
    eigenvalues = compute_eigenvalues(gradients_cpu, device=device, batch_size=batch_size)
    eff_dim_func = lambda lamb: compute_effective_dimension(eigenvalues, lamb)

    # Report eigenvalue statistics for debugging numerical issues
    eig_nonzero = eigenvalues[eigenvalues > 1e-10]
    print(f"  Eigenvalue stats: max={eigenvalues[0]:.2e}, min_nonzero={eig_nonzero[-1]:.2e}, "
          f"n_nonzero={len(eig_nonzero)}/{len(eigenvalues)}")

    # Pre-compute Gram matrix K for Woodbury identity (efficient exact score computation)
    print("Pre-computing Gram matrix K...")
    K = compute_gram_matrix(gradients_cpu, device=device, batch_size=batch_size)

    # Compute numerical rank
    threshold = 1e-10 * eigenvalues[0]
    rank = (eigenvalues > threshold).sum().item()

    results = {
        "n_samples": n,
        "d_params": d,
        "proj_type": proj_type.value,
        "lambda_values": lambda_values,
        "m_multipliers": m_multipliers,
        "effective_dims": {},
        "experiments": [],
        "batch_size": batch_size,
        "min_m": min_m,
        "min_d_lambda": min_d_lambda,
        "skipped_configs": [],
        "use_held_out_gradients": use_held_out_gradients,
        "eigenvalues": eigenvalues.cpu().numpy(),
        "rank": rank,
    }

    for lamb in lambda_values:
        d_lambda = eff_dim_func(lamb)
        results["effective_dims"][lamb] = d_lambda

        print(f"\n{'='*60}")
        print(f"λ = {lamb:.1e}, d_λ = {d_lambda:.1f}")

        # Skip λ values with too small effective dimension (numerically degenerate)
        if d_lambda < min_d_lambda:
            print(f"  SKIPPED: d_λ = {d_lambda:.2f} < {min_d_lambda} (numerically degenerate)")
            results["skipped_configs"].append({
                "lambda": lamb,
                "d_lambda": d_lambda,
                "reason": f"d_lambda ({d_lambda:.2f}) < min_d_lambda ({min_d_lambda})"
            })
            continue

        print(f"{'='*60}")

        for mult in m_multipliers:
            # Compute m as multiple of d_λ, with minimum threshold to avoid degeneracy
            m_raw = mult * d_lambda
            m = max(min_m, min(int(m_raw), d))

            # Flag if m was clamped to minimum
            m_clamped = (m == min_m and m_raw < min_m)

            clamped_note = " [CLAMPED]" if m_clamped else ""
            print(f"\n  Testing m = {m} ({mult}× d_λ = {m_raw:.1f}){clamped_note}...")

            exp_result = compute_empirical_epsilon(
                gradients_cpu, lamb, m, proj_type,
                num_trials=num_trials,
                seed=seed,
                device=device,
                batch_size=batch_size,
                K=K,
                use_held_out_gradients=use_held_out_gradients,
            )
            exp_result["d_lambda"] = d_lambda
            exp_result["m_over_d_lambda"] = m / d_lambda if d_lambda > 0 else float('inf')
            exp_result["m_raw"] = m_raw
            exp_result["m_clamped"] = m_clamped
            exp_result["multiplier"] = mult

            results["experiments"].append(exp_result)

            print(f"    Ratio: {exp_result['mean_ratio']:.4f} ± {exp_result['std_ratio']:.4f}")
            print(f"    ε (95th percentile): {exp_result['empirical_eps_95']:.4f}")
            print(f"    ε (max): {exp_result['empirical_eps_max']:.4f}")

            # Memory cleanup between iterations
            clear_memory(device)

    # Final cleanup
    del eigenvalues
    if K is not None:
        del K
    clear_memory(device)

    return results


def print_summary_table(results: dict):
    """Print a summary table of the key findings."""
    print("\n" + "="*80)
    print("SUMMARY: Spectrum Bounds Validation")
    print("="*80)

    print(f"\nDataset: n={results['n_samples']}, d={results['d_params']}")
    print(f"Projection type: {results['proj_type']}")

    # Show stability parameters if present
    if "min_m" in results:
        print(f"Stability: min_m={results['min_m']}, min_d_lambda={results['min_d_lambda']}")

    # Report skipped configurations
    if "skipped_configs" in results and results["skipped_configs"]:
        print(f"\nSkipped {len(results['skipped_configs'])} λ values (d_λ too small):")
        for skip in results["skipped_configs"]:
            print(f"  λ={skip['lambda']:.1e}: {skip['reason']}")

    print("\n" + "-"*90)
    print(f"{'λ':>10} | {'d_λ':>8} | {'m':>8} | {'m/d_λ':>8} | {'ε_95':>8} | {'ε_max':>8} | {'ratio':>10}")
    print("-"*90)

    for exp in results["experiments"]:
        # Mark clamped m values with asterisk
        m_str = f"{exp['m']:>6}"
        if exp.get('m_clamped', False):
            m_str = f"{exp['m']:>5}*"

        print(f"{exp['lambda']:>10.1e} | {exp['d_lambda']:>8.1f} | {m_str:>8} | "
              f"{exp['m_over_d_lambda']:>8.2f} | {exp['empirical_eps_95']:>8.4f} | "
              f"{exp['empirical_eps_max']:>8.4f} | {exp['mean_ratio']:>10.4f}")

    print("-"*90)
    print("  * = m was clamped to min_m (theoretical m/d_λ ratio not meaningful)")

    # Find the m/d_λ threshold where ε < 0.1
    threshold_eps = 0.1
    print(f"\nλ values achieving ε_95 < {threshold_eps}:")
    for lamb in results["lambda_values"]:
        exps = [e for e in results["experiments"] if e["lambda"] == lamb and not e.get('m_clamped', False)]
        if not exps:
            continue
        exps = sorted(exps, key=lambda x: x["m_over_d_lambda"])
        for exp in exps:
            if exp["empirical_eps_95"] < threshold_eps:
                print(f"  λ={lamb:.0e}: ε < {threshold_eps} when m/d_λ ≥ {exp['m_over_d_lambda']:.2f}")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Practical validation of Theorem 2 for regularized sketching"
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar2", "maestro"])
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["mlp", "resnet9", "musictransformer"])
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num_trials", type=int, default=5, help="Trials per configuration")
    parser.add_argument("--proj_type", type=str, default="normal",
                       choices=["normal", "rademacher", "sjlt"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=100,
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
    parser.add_argument("--min_m", type=int, default=10,
                       help="Minimum projection dimension to avoid numerical degeneracy. "
                            "When mult * d_λ < min_m, m is clamped to min_m. (default: 10)")
    parser.add_argument("--min_d_lambda", type=float, default=5.0,
                       help="Skip λ values where d_λ < this threshold (numerically degenerate). "
                            "(default: 5.0)")

    # Gradient handling options
    parser.add_argument("--hold_out", action="store_true",
                       help="Hold out test gradients from FIM construction. "
                            "By default, all gradients are used for FIM and a subset is tested.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Experiment configuration
    lambda_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    m_multipliers = [0.25, 0.4, 0.5, 0.7, 1.0, 2.0, 4.0]  # multiples of d_λ

    proj_type = ProjectionType(args.proj_type)

    print("\n" + "="*60)
    print(f"OFFLOAD MODE: {args.offload.upper()}")
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
    cache = GradientCache(
        offload=args.offload,
        cache_dir=args.cache_dir if args.offload == "disk" else None,
    )

    # For disk mode, check if cache is valid and reuse
    # Note: is_valid() already loads metadata when it returns True
    if args.offload == "disk" and cache.is_valid(expected_samples=args.n_samples):
        print(f"Loading cached gradients from {args.cache_dir}...")
    else:
        # Need to compute and cache gradients
        print(f"\nLoading {args.model} on {args.dataset}...")
        from dattri.benchmark.load import load_benchmark

        model_details, _ = load_benchmark(model=args.model, dataset=args.dataset, metric="lds")
        nn_model = model_details["model"]
        train_dataset = model_details["train_dataset"]
        train_sampler = model_details["train_sampler"]
        indices = list(train_sampler)[:args.n_samples]

        model_type = "musictransformer" if args.model == "musictransformer" else "default"
        cache.store_gradients(
            model=nn_model,
            dataset=train_dataset,
            indices=indices,
            device=args.device,
            model_type=model_type,
            batch_size=args.batch_size,
        )

    # Pass cache to experiment
    gradients_cpu = cache

    # Run experiment
    results = run_experiment(
        gradients_cpu=gradients_cpu,
        lambda_values=lambda_values,
        m_multipliers=m_multipliers,
        proj_type=proj_type,
        num_trials=args.num_trials,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        min_m=args.min_m,
        min_d_lambda=args.min_d_lambda,
        use_held_out_gradients=args.hold_out,
    )

    # Add metadata
    results["dataset"] = args.dataset
    results["model"] = args.model
    results["offload_mode"] = args.offload
    results["batch_size"] = args.batch_size

    # Print summary
    print_summary_table(results)

    # Save results with organized directory structure
    experiment_dir = os.path.join(args.output_dir, "spectrum_bounds")
    os.makedirs(experiment_dir, exist_ok=True)

    # Build filename with key settings
    holdout_suffix = "_holdout" if args.hold_out else ""
    results_filename = (
        f"{args.dataset}_{args.model}_{args.proj_type}"
        f"_n{args.n_samples}{holdout_suffix}.pt"
    )
    results_path = os.path.join(experiment_dir, results_filename)
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

"""
Hyperparameter Selection: Compare λ and m Selection Strategies

Compares two approaches for selecting the regularization λ and sketch dimension m:

1. **Theory-Driven (Proposal):**
   Choose m based on effective dimension: m ≈ d_λ/ε²

2. **Utility-Driven (Sequential / "Taming" paper approach):**
   a) Fix m large, sweep λ to maximize LDS (utility metric)
   b) Then verify m ≥ d_λ* for the optimal λ*
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from dattri.benchmark.load import load_benchmark
from dattri.metric import lds
from dattri.func.projection import make_random_projector, ProjectionType

from utils.fisher_utils import (
    compute_eigenspectrum,
    project_gradients_to_cpu,
    compute_kernel_from_projected,
)
from utils.gradient_cache import GradientCache, create_gradient_cache


# =============================================================================
# Sketched Score Computation (specific to this experiment)
# =============================================================================

def compute_scores_sketched(
    train_grad_cache: GradientCache,
    test_grad_cache: GradientCache,
    projector,
    lamb: float,
    device: str = "cuda",
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Compute sketched influence scores (Pg_train)^T (PFP^T + λI)^{-1} (Pg_test).

    For self-scores, pass train_grad_cache=test_grad_cache and use result.diag().

    Args:
        train_grad_cache: Training gradients in GradientCache
        test_grad_cache: Test gradients in GradientCache
        projector: Projector with .project() method and .proj_dim attribute
        lamb: Regularization parameter λ
        device: GPU device
        batch_size: Batch size for streaming

    Returns:
        scores: (n_train, n_test) influence score matrix
    """
    n_train = train_grad_cache.n_samples
    m = projector.proj_dim

    # Auto-select method based on which dimension is smaller
    if n_train < m: # Use Woodbury method
        # 1. Project all gradients
        PG_train = project_gradients_to_cpu(train_grad_cache, projector, device, batch_size)
        PG_test = project_gradients_to_cpu(test_grad_cache, projector, device, batch_size)

        # 2. Compute kernel K = PG_train @ PG_train^T (not normalized)
        K = compute_kernel_from_projected(PG_train, device, batch_size, normalize=False)

        # 3. Compute scores using Woodbury
        k_A = PG_train.shape[0]
        k_B = PG_test.shape[0]
        dtype = PG_train.dtype

        # K_cross = PA @ PB^T
        print(f"  Computing cross-product ({k_A}×{k_B})...")
        K_cross = torch.zeros(k_A, k_B, device=device, dtype=dtype)
        cross_batch = batch_size * 4
        for i_start in tqdm(range(0, k_A, cross_batch), desc="  Cross-product", leave=False):
            i_end = min(i_start + cross_batch, k_A)
            pa_i = PG_train[i_start:i_end].to(device)
            PB_gpu = PG_test.to(device)
            K_cross[i_start:i_end] = pa_i @ PB_gpu.T

        # Solve (K + k_A*λI) X = K_cross using eigendecomposition
        print(f"  Solving linear system ({k_A}×{k_A}) via eigendecomposition...")
        K_reg = K.clone()
        K_reg.diagonal().add_(k_A * lamb)

        eigenvalues, eigenvectors = torch.linalg.eigh(K_reg)
        inv_eigenvalues = 1.0 / eigenvalues
        X = eigenvectors @ (inv_eigenvalues.unsqueeze(1) * (eigenvectors.T @ K_cross))

        # Scores = (1/λ)(K_cross - K @ X)
        return (1.0 / lamb) * (K_cross - K @ X)
    else:
        n_train = train_grad_cache.n_samples
        n_test = test_grad_cache.n_samples
        m = projector.proj_dim
        dtype = train_grad_cache.dtype

        # Pass 1: Build H = (1/n) Σ (Pg)(Pg)^T + λI
        print(f"    Building projected Fisher matrix ({m}×{m})...")
        H = torch.zeros(m, m, device=device, dtype=dtype)

        for i_start in tqdm(range(0, n_train, batch_size), desc="    Pass 1: Building H", leave=False):
            i_end = min(i_start + batch_size, n_train)
            g_batch = train_grad_cache.get_batch(i_start, i_end, device=device)
            pg_batch = projector.project(g_batch, ensemble_id=0)
            H.add_(pg_batch.T @ pg_batch)

        H.div_(n_train)
        H.diagonal().add_(lamb)

        # Eigendecomposition of regularized Fisher for numerical stability
        print(f"    Computing eigendecomposition of H ({m}×{m})...")
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        inv_eigenvalues = 1.0 / eigenvalues

        # Pre-compute Q_test = H^{-1} @ (P G_test)^T
        print(f"    Computing test scores...")
        Q_test = torch.zeros(m, n_test, device=device, dtype=dtype)

        for j_start in tqdm(range(0, n_test, batch_size), desc="    Projecting test grads", leave=False):
            j_end = min(j_start + batch_size, n_test)
            g_batch = test_grad_cache.get_batch(j_start, j_end, device=device)
            pg_batch = projector.project(g_batch, ensemble_id=0)
            Q_test[:, j_start:j_end] = eigenvectors @ (inv_eigenvalues.unsqueeze(1) * (eigenvectors.T @ pg_batch.T))

        # Pass 2: Stream train gradients to compute scores
        scores = torch.zeros(n_train, n_test, device=device, dtype=dtype)

        for i_start in tqdm(range(0, n_train, batch_size), desc="    Pass 2: Computing scores", leave=False):
            i_end = min(i_start + batch_size, n_train)
            g_batch = train_grad_cache.get_batch(i_start, i_end, device=device)
            pg_batch = projector.project(g_batch, ensemble_id=0)
            scores[i_start:i_end, :] = pg_batch @ Q_test

        return scores


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


def _get_gradient_info(grad_cache: GradientCache) -> Tuple[int, int, torch.dtype]:
    """Get n_samples, dim, dtype from GradientCache."""
    return grad_cache.n_samples, grad_cache.dim, grad_cache.dtype


def run_lambda_sweep(
    train_grad_cache: GradientCache,
    val_grad_cache: GradientCache,
    test_grad_cache: GradientCache,
    val_gt: tuple,
    test_gt: tuple,
    proj_dim: int,
    lambda_values: List[float],
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
) -> Dict:
    """
    Sweep over λ values with fixed projection dimension using influence functions.

    Strategy (Spectral Method for λ sweep):
    1. Project all train/val/test gradients once -> PG_train, PG_val, PG_test
    2. Compute kernel matrices once: K = PG_train @ PG_train^T, K_cross = PG @ PG_train^T
    3. Eigendecompose K once: K = U Σ U^T
    4. For each λ: solve via eigenvalues (milliseconds per λ!)

    Args:
        train_grad_cache: Training gradient cache (required)
        val_grad_cache: Validation gradient cache (required)
        test_grad_cache: Test gradient cache (required)
        val_gt: Validation ground truth for LDS computation
        test_gt: Test ground truth for LDS computation
        proj_dim: Projection dimension m (fixed)
        lambda_values: List of λ values to sweep
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for projection

    Returns:
        Dictionary with sweep results
    """
    n_train, d, dtype = _get_gradient_info(train_grad_cache)
    n_val, _, _ = _get_gradient_info(val_grad_cache)
    n_test, _, _ = _get_gradient_info(test_grad_cache)

    results = {"lambda_values": [], "val_lds": [], "proj_dim": proj_dim}

    # Setup projector (created once, reused for all projections)
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

    # =========================================================================
    # Step 1: Project all gradients ONCE
    # =========================================================================
    print(f"  [1/4] Projecting {n_train} train gradients (m={proj_dim})...")
    PG_train = torch.zeros(n_train, proj_dim, dtype=dtype, device="cpu")
    for i in tqdm(range(0, n_train, batch_size), desc="    Train", leave=False):
        i_end = min(i + batch_size, n_train)
        g_batch = train_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_train[i:i_end] = pg_batch.cpu()

    print(f"  [2/4] Projecting {n_val} val + {n_test} test gradients...")
    PG_val = torch.zeros(n_val, proj_dim, dtype=dtype, device="cpu")
    for i in tqdm(range(0, n_val, batch_size), desc="    Val", leave=False):
        i_end = min(i + batch_size, n_val)
        g_batch = val_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_val[i:i_end] = pg_batch.cpu()

    PG_test = torch.zeros(n_test, proj_dim, dtype=dtype, device="cpu")
    for i in tqdm(range(0, n_test, batch_size), desc="    Test", leave=False):
        i_end = min(i + batch_size, n_test)
        g_batch = test_grad_cache.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_test[i:i_end] = pg_batch.cpu()

    # =========================================================================
    # Step 2: Compute kernel matrices (on GPU in blocks if needed)
    # =========================================================================
    print(f"  [3/4] Computing kernel matrices...")

    # K_train = PG_train @ PG_train^T (n_train × n_train)
    PG_train_gpu = PG_train.to(device)
    K_train = PG_train_gpu @ PG_train_gpu.T

    # K_val_cross = PG_val @ PG_train^T (n_val × n_train)
    PG_val_gpu = PG_val.to(device)
    K_val_cross = PG_val_gpu @ PG_train_gpu.T

    # K_test_cross = PG_test @ PG_train^T (n_test × n_train)
    PG_test_gpu = PG_test.to(device)
    K_test_cross = PG_test_gpu @ PG_train_gpu.T

    # =========================================================================
    # Step 3: Eigendecompose K_train ONCE
    # =========================================================================
    print(f"  [4/4] Eigendecomposing K_train ({n_train}×{n_train})...")
    # Use double precision for numerical stability, then convert back
    eigenvalues, U = torch.linalg.eigh(K_train.double())
    eigenvalues = torch.clamp(eigenvalues, min=0).to(dtype).to(device)
    U = U.to(dtype).to(device)

    # Precompute U^T @ K_cross^T for fast score computation
    UT_Kval_T = U.T @ K_val_cross.T  # (n_train, n_val)
    UT_Ktest_T = U.T @ K_test_cross.T  # (n_train, n_test)

    # =========================================================================
    # Step 4: Fast λ sweep (milliseconds per λ!)
    # =========================================================================
    best_lambda = None
    best_val_lds = float('-inf')

    print(f"\n  Sweeping {len(lambda_values)} λ values...")
    for lamb in lambda_values:
        # Solve via eigendecomposition:
        # (K + nλI)^{-1} = U @ diag(1/(σ + nλ)) @ U^T
        n_lamb = n_train * lamb
        inv_diag = 1.0 / (eigenvalues + n_lamb)

        # Scores = K_cross @ (K + nλI)^{-1}
        #        = K_cross @ U @ diag(inv) @ U^T
        # We have UT_Kcross_T = U^T @ K_cross^T, so:
        # Scores^T = U @ (inv_diag * UT_Kcross_T)
        val_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Kval_T)
        # lds expects (n_train, n_val), which is val_scores_T
        val_lds_score = lds(val_scores_T, val_gt)[0]
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

        results["lambda_values"].append(lamb)
        results["val_lds"].append(mean_val_lds)

        if mean_val_lds > best_val_lds:
            best_val_lds = mean_val_lds
            best_lambda = lamb

        print(f"    λ = {lamb:.1e}: Val LDS = {mean_val_lds:.4f}")

    # Evaluate best λ on test set
    n_lamb = n_train * best_lambda
    inv_diag = 1.0 / (eigenvalues + n_lamb)
    test_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Ktest_T)
    # lds expects (n_train, n_test), which is test_scores_T
    test_lds_score = lds(test_scores_T, test_gt)[0]
    mean_test_lds = torch.mean(test_lds_score[~torch.isnan(test_lds_score)]).item()

    results["best_lambda"] = best_lambda
    results["best_val_lds"] = best_val_lds
    results["test_lds"] = mean_test_lds

    print(f"\n  Best λ = {best_lambda:.1e}: Val LDS = {best_val_lds:.4f}, Test LDS = {mean_test_lds:.4f}")

    return results


def run_m_sweep(
    train_grad_cache: GradientCache,
    val_grad_cache: GradientCache,
    val_gt: tuple,
    lamb: float,
    m_values: List[int],
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
) -> Dict:
    """
    Sweep over projection dimensions with fixed λ using influence functions.

    After selecting λ via LDS, this validates the theory:
    LDS should stabilize once m ≥ d_λ/ε² for desired ε.

    Args:
        train_grad_cache: Training gradient cache (required)
        val_grad_cache: Validation gradient cache (required)
        val_gt: Validation ground truth for LDS computation
        lamb: Fixed regularization parameter λ
        m_values: List of projection dimensions to sweep
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for computation

    Returns:
        Dictionary with sweep results
    """
    _, d, dtype = _get_gradient_info(train_grad_cache)
    proj_type_enum = ProjectionType(proj_type)
    proj_bs = min(batch_size, 32)
    proj_bs = max(8, (proj_bs // 8) * 8)

    results = {"m_values": [], "val_lds": [], "lambda": lamb}

    for proj_dim in tqdm(m_values, desc=f"m sweep (λ={lamb:.1e})"):
        # Create projector for this dimension
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

        # Compute sketched influence scores
        val_score = compute_scores_sketched(
            train_grad_cache, val_grad_cache, projector, lamb, device, batch_size
        )

        val_lds_score = lds(val_score, val_gt)[0]
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

        results["m_values"].append(proj_dim)
        results["val_lds"].append(mean_val_lds)

        print(f"  m = {proj_dim}: Val LDS = {mean_val_lds:.4f}")

    return results


def run_experiment(
    dataset: str,
    model_name: str,
    lambda_values: List[float],
    m_values: List[int],
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    offload: str = "cpu",
    cache_dir: Optional[str] = None,
    proj_type: ProjectionType = ProjectionType.normal,
    val_ratio: float = 0.1,
) -> Dict:
    """
    Run full comparison between joint and sequential hyperparameter selection.

    Uses true influence functions with sketched approximation.
    Theory predicts: m ≥ d_λ/ε² for ε-accurate approximation.

    All gradient caches are computed upfront at the beginning and passed to
    sub-functions, ensuring gradients are computed exactly once.

    Args:
        dataset: Dataset name
        model_name: Model name
        lambda_values: List of regularization values to sweep
        m_values: List of projection dimensions to sweep
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for gradient computation
        offload: Gradient storage mode ("none", "cpu", "disk")
        cache_dir: Directory for disk cache (required if offload="disk")
        proj_type: Type of random projection
        val_ratio: Fraction of test set to use for validation
    """
    print(f"\n{'='*60}")
    print(f"Full Hyperparameter Selection Comparison (Influence Functions)")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"{'='*60}")

    # Load benchmark
    model_details, groundtruth = load_benchmark(model=model_name, dataset=dataset, metric="lds")

    # Count model parameters
    model = model_details["model"]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = "musictransformer" if model_name == "musictransformer" else "default"

    results = {
        "dataset": dataset,
        "model": model_name,
        "n_params": n_params,
        "lambda_values": lambda_values,
        "m_values": m_values,
        "batch_size": batch_size,
        "proj_type": proj_type,
        "offload": offload,
        "method": "influence_function",
    }

    # =========================================================================
    # Compute ALL gradient caches upfront
    # =========================================================================
    print("\n[Setup] Computing gradient caches...")

    # Get indices
    train_indices = list(model_details["train_sampler"])
    val_indices, test_indices = get_validation_split_indices(
        model_details["test_sampler"], val_ratio=val_ratio, seed=seed
    )

    # Compute ground truth for val/test splits
    original_gt_values, subset_indices = groundtruth
    test_indices_dict = {idx: pos for pos, idx in enumerate(model_details["test_sampler"])}
    val_gt_indices = [test_indices_dict[idx] for idx in val_indices]
    val_gt_values = original_gt_values[:, val_gt_indices]
    test_gt_indices = [test_indices_dict[idx] for idx in test_indices]
    test_gt_values = original_gt_values[:, test_gt_indices]
    val_gt = (val_gt_values, subset_indices)
    test_gt = (test_gt_values, subset_indices)

    # Compute training gradients
    print("  Computing training gradients...")
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

    # Compute validation gradients
    print("  Computing validation gradients...")
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

    # Compute test gradients
    print("  Computing test gradients...")
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

    print(f"  Gradient caches ready: train={train_grad_cache.n_samples}, "
          f"val={val_grad_cache.n_samples}, test={test_grad_cache.n_samples}")

    # =========================================================================
    # Step 1: Compute effective dimensions
    # =========================================================================
    print("\n[Step 1] Computing effective dimensions d_λ(F)...")
    gram_batch_size = batch_size * 4  # Gram computation can handle larger batches
    _, d_lambda_dict = compute_eigenspectrum(
        train_grad_cache, lambda_values, device=device, batch_size=gram_batch_size
    )
    results["d_lambda"] = d_lambda_dict

    print("\nEffective dimensions:")
    for lamb, d_l in d_lambda_dict.items():
        print(f"  λ = {lamb:.1e}: d_λ = {d_l:.1f}")
        if d_l > 0:
            print(f"           m for ε=0.1: {d_l / 0.01:.0f}, ε=0.2: {d_l / 0.04:.0f}, ε=0.3: {d_l / 0.09:.0f}")

    # =========================================================================
    # Step 2: λ sweep with large fixed m
    # =========================================================================
    print("\n[Step 2] Sweeping λ with fixed large m...")
    large_m = max(m_values)
    lambda_sweep_results = run_lambda_sweep(
        train_grad_cache=train_grad_cache,
        val_grad_cache=val_grad_cache,
        test_grad_cache=test_grad_cache,
        val_gt=val_gt,
        test_gt=test_gt,
        proj_dim=large_m,
        lambda_values=lambda_values,
        proj_type=proj_type,
        device=device,
        seed=seed,
        batch_size=batch_size,
    )

    results["lambda_sweep"] = lambda_sweep_results

    best_lambda = lambda_sweep_results["best_lambda"]
    best_d_lambda = d_lambda_dict.get(best_lambda, d_lambda_dict[min(d_lambda_dict.keys(), key=lambda x: abs(x - best_lambda))])

    print(f"\nBest λ* = {best_lambda:.1e}")
    print(f"Corresponding d_λ* = {best_d_lambda:.1f}")
    print(f"Theoretical minimum m for ε=0.1: {best_d_lambda / 0.01:.0f}")
    print(f"Theoretical minimum m for ε=0.2: {best_d_lambda / 0.04:.0f}")
    print(f"Theoretical minimum m for ε=0.3: {best_d_lambda / 0.09:.0f}")

    # =========================================================================
    # Step 3: m sweep with fixed best λ
    # =========================================================================
    print(f"\n[Step 3] Sweeping m with fixed λ* = {best_lambda:.1e}...")
    m_sweep_results = run_m_sweep(
        train_grad_cache=train_grad_cache,
        val_grad_cache=val_grad_cache,
        val_gt=val_gt,
        lamb=best_lambda,
        m_values=m_values,
        proj_type=proj_type,
        device=device,
        seed=seed,
        batch_size=batch_size,
    )
    results["m_sweep"] = m_sweep_results

    # Find minimum m that achieves ~95% of max LDS
    max_lds = max(m_sweep_results["val_lds"])
    threshold_lds = 0.95 * max_lds
    for m, lds_val in zip(m_sweep_results["m_values"], m_sweep_results["val_lds"]):
        if lds_val >= threshold_lds:
            results["empirical_min_m"] = m
            break
    else:
        results["empirical_min_m"] = max(m_values)

    # Compute implied ε from empirical_min_m
    if best_d_lambda > 0:
        implied_eps = np.sqrt(best_d_lambda / results["empirical_min_m"])
        results["implied_epsilon"] = implied_eps
    else:
        implied_eps = float('nan')
        results["implied_epsilon"] = implied_eps

    print(f"\nEmpirical minimum m (95% of max LDS): {results['empirical_min_m']}")
    print(f"Theoretical d_λ*: {best_d_lambda:.0f}")
    print(f"Implied ε (from m = d_λ/ε²): {implied_eps:.3f}")
    print(f"Theory validation: m/d_λ = {results['empirical_min_m'] / best_d_lambda:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter selection comparison for regularized sketching"
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar2", "maestro"])
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["lr", "mlp", "resnet9", "musictransformer"])
    parser.add_argument("--proj_type", type=str, default="normal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for gradient computation. Decrease if running out of memory.")
    # Gradient storage options
    parser.add_argument("--offload", type=str, default="cpu",
                       choices=["none", "cpu", "disk"],
                       help="Gradient storage: none (GPU), cpu (RAM), disk (files)")
    parser.add_argument("--cache_dir", type=str, default="./grad_cache",
                       help="Directory for gradient cache (only used with --offload disk)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    proj_type = ProjectionType(args.proj_type)

    # Print configuration
    print("\n" + "="*60)
    print(f"Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Offload Mode: {args.offload.upper()}")
    if args.offload == "disk":
        print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print("="*60 + "\n")

    # Define search spaces
    lambda_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    m_values = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    # Run experiment
    results = run_experiment(
        args.dataset, args.model, lambda_values, m_values,
        device=args.device, seed=args.seed,
        batch_size=args.batch_size, offload=args.offload,
        cache_dir=args.cache_dir,
        proj_type=proj_type,
    )

    # Add metadata
    results["dataset"] = args.dataset
    results["model"] = args.model
    results["offload_mode"] = args.offload
    results["batch_size"] = args.batch_size

    # Save results with organized directory structure
    experiment_dir = os.path.join(args.output_dir, "hyperparam_selection")
    os.makedirs(experiment_dir, exist_ok=True)

    # Build filename with key settings
    results_filename = f"{args.dataset}_{args.model}_{args.proj_type}.pt"
    results_path = os.path.join(experiment_dir, results_filename)
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

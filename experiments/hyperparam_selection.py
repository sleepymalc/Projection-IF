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
from dattri.func.projection import make_random_projector, ProjectionType

from utils.fisher_utils import (
    compute_eigenspectrum,
    project_gradients_to_cpu,
    compute_kernel_from_projected,
)
from utils.gradient_cache import GradientCache, create_gradient_cache
from utils.metrics_gpu import lds_gpu


# =============================================================================
# Safe Projection Helper
# =============================================================================

def safe_project(data: torch.Tensor, projector, d: int, ensemble_id: int = 0) -> torch.Tensor:
    """
    Safely project data, using identity (no projection) when m >= d.

    Args:
        data: Input tensor of shape (batch, d)
        projector: Projector with .project() method and .proj_dim attribute
        d: Original dimension
        ensemble_id: Ensemble ID for projection

    Returns:
        Projected tensor of shape (batch, m) if m < d, else original tensor
    """
    m = projector.proj_dim
    if m >= d:
        # No projection needed when target dimension >= source dimension
        return data
    else:
        return projector.project(data, ensemble_id=ensemble_id)


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
        d = train_grad_cache.dim
        dtype = train_grad_cache.dtype

        # Validate projection dimension
        if m >= d:
            raise ValueError(
                f"Invalid projection dimension: m={m} must be < d={d}. "
                f"Random projection is only meaningful when projecting to lower dimensions."
            )

        # Pass 1: Build H = (1/n) Σ (Pg)(Pg)^T + λI
        print(f"    Building projected Fisher matrix ({m}×{m})...")
        H = torch.zeros(m, m, device=device, dtype=dtype)

        for i_start in tqdm(range(0, n_train, batch_size), desc="    Pass 1: Building H", leave=False):
            i_end = min(i_start + batch_size, n_train)
            g_batch = train_grad_cache.get_batch(i_start, i_end, device=device)
            pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
            H.add_(pg_batch.T @ pg_batch)
            del g_batch, pg_batch
            if (i_start // batch_size) % 20 == 0:
                torch.cuda.empty_cache()

        H.div_(n_train)
        H.diagonal().add_(lamb)

        # Eigendecomposition of regularized Fisher for numerical stability
        print(f"    Computing eigendecomposition of H ({m}×{m})...")
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        inv_eigenvalues = 1.0 / eigenvalues
        del H  # Free H after eigendecomposition

        # Pre-compute Q_test = H^{-1} @ (P G_test)^T
        print(f"    Computing test scores...")
        Q_test = torch.zeros(m, n_test, device=device, dtype=dtype)

        for j_start in tqdm(range(0, n_test, batch_size), desc="    Projecting test grads", leave=False):
            j_end = min(j_start + batch_size, n_test)
            g_batch = test_grad_cache.get_batch(j_start, j_end, device=device)
            pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
            Q_test[:, j_start:j_end] = eigenvectors @ (inv_eigenvalues.unsqueeze(1) * (eigenvectors.T @ pg_batch.T))
            del g_batch, pg_batch

        # Pass 2: Stream train gradients to compute scores
        scores = torch.zeros(n_train, n_test, device=device, dtype=dtype)

        for i_start in tqdm(range(0, n_train, batch_size), desc="    Pass 2: Computing scores", leave=False):
            i_end = min(i_start + batch_size, n_train)
            g_batch = train_grad_cache.get_batch(i_start, i_end, device=device)
            pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
            scores[i_start:i_end, :] = pg_batch @ Q_test
            del g_batch, pg_batch

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
    num_trials: int = 1,
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
        num_trials: Number of trials with different random projections

    Returns:
        Dictionary with sweep results (averaged over trials if num_trials > 1)
    """
    n_train, d, dtype = _get_gradient_info(train_grad_cache)
    n_val, _, _ = _get_gradient_info(val_grad_cache)
    n_test, _, _ = _get_gradient_info(test_grad_cache)

    # Store per-trial results
    per_trial_results = []

    proj_type_enum = ProjectionType(proj_type)
    proj_bs = min(batch_size, 32)
    proj_bs = max(8, (proj_bs // 8) * 8)

    for trial in range(num_trials):
        trial_seed = seed + trial
        print(f"  Trial {trial + 1}/{num_trials} (seed={trial_seed})")

        # Create projector for this trial
        projector = make_random_projector(
            param_shape_list=[d],
            feature_batch_size=batch_size,
            proj_dim=proj_dim,
            proj_max_batch_size=proj_bs,
            device=torch.device(device),
            proj_seed=trial_seed,
            proj_type=proj_type_enum,
            dtype=dtype,
        )

        # Validate projection dimension
        if proj_dim >= d:
            raise ValueError(
                f"Invalid projection dimension: proj_dim={proj_dim} must be < d={d}. "
                f"Random projection is only meaningful when projecting to lower dimensions."
            )

        # =========================================================================
        # Memory estimation and warning
        # =========================================================================
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        pg_train_gb = (n_train * proj_dim * bytes_per_elem) / (1024**3)
        pg_val_gb = (n_val * proj_dim * bytes_per_elem) / (1024**3)
        pg_test_gb = (n_test * proj_dim * bytes_per_elem) / (1024**3)
        total_gb = pg_train_gb + pg_val_gb + pg_test_gb

        print(f"    Memory estimate for projected gradients (CPU):")
        print(f"      PG_train: {pg_train_gb:.2f} GB")
        print(f"      PG_val: {pg_val_gb:.2f} GB")
        print(f"      PG_test: {pg_test_gb:.2f} GB")
        print(f"      Total: {total_gb:.2f} GB")

        if total_gb > 20:
            print(f"    WARNING: Large memory footprint ({total_gb:.1f} GB). Consider reducing m or using smaller batches.")

        # =========================================================================
        # Step 1: Project all gradients ONCE
        # =========================================================================
        print(f"    [1/4] Projecting {n_train} train gradients (m={proj_dim})...")
        PG_train = torch.zeros(n_train, proj_dim, dtype=dtype, device="cpu")
        for i in tqdm(range(0, n_train, batch_size), desc="      Train", leave=False):
            i_end = min(i + batch_size, n_train)
            g_batch = train_grad_cache.get_batch(i, i_end, device=device)
            pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
            PG_train[i:i_end] = pg_batch.cpu()
            # Explicitly free GPU memory
            del g_batch, pg_batch
            # Periodically clear CUDA cache to avoid fragmentation
            if (i // batch_size) % 20 == 0:
                torch.cuda.empty_cache()

        print(f"    [2/4] Projecting {n_val} val + {n_test} test gradients...")
        PG_val = torch.zeros(n_val, proj_dim, dtype=dtype, device="cpu")
        for i in tqdm(range(0, n_val, batch_size), desc="      Val", leave=False):
            i_end = min(i + batch_size, n_val)
            g_batch = val_grad_cache.get_batch(i, i_end, device=device)
            pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
            PG_val[i:i_end] = pg_batch.cpu()
            del g_batch, pg_batch

        PG_test = torch.zeros(n_test, proj_dim, dtype=dtype, device="cpu")
        for i in tqdm(range(0, n_test, batch_size), desc="      Test", leave=False):
            i_end = min(i + batch_size, n_test)
            g_batch = test_grad_cache.get_batch(i, i_end, device=device)
            pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
            PG_test[i:i_end] = pg_batch.cpu()
            del g_batch, pg_batch

        # Final cleanup before kernel computation
        torch.cuda.empty_cache()

        # =========================================================================
        # Step 2: Compute kernel matrices (in chunks to avoid OOM)
        # =========================================================================
        print(f"    [3/4] Computing kernel matrices...")

        # Check if we can fit PG_train on GPU at once
        pg_train_size_gb = PG_train.element_size() * PG_train.numel() / (1024**3)
        print(f"      PG_train size: {pg_train_size_gb:.2f} GB")

        # If PG_train is too large (>10GB), compute in chunks
        if pg_train_size_gb > 10.0:
            print(f"      Computing kernels in chunks (PG_train too large for GPU)...")
            chunk_size = max(1, int(10.0 / pg_train_size_gb * n_train))  # ~10GB chunks

            # K_train = PG_train @ PG_train^T
            K_train = torch.zeros(n_train, n_train, dtype=dtype, device=device)
            for i in tqdm(range(0, n_train, chunk_size), desc="        K_train", leave=False):
                i_end = min(i + chunk_size, n_train)
                pg_chunk = PG_train[i:i_end].to(device)
                K_train[i:i_end] = pg_chunk @ PG_train.T.to(device)
                del pg_chunk
                if i % (chunk_size * 5) == 0:  # Periodically clear cache
                    torch.cuda.empty_cache()

            # K_val_cross = PG_val @ PG_train^T
            PG_val_gpu = PG_val.to(device)
            PG_train_T_gpu = PG_train.T.to(device)
            K_val_cross = PG_val_gpu @ PG_train_T_gpu
            del PG_val_gpu  # Free immediately after use

            # K_test_cross = PG_test @ PG_train^T
            PG_test_gpu = PG_test.to(device)
            K_test_cross = PG_test_gpu @ PG_train_T_gpu
            del PG_test_gpu, PG_train_T_gpu  # Free immediately after use
        else:
            # K_train = PG_train @ PG_train^T (n_train × n_train)
            PG_train_gpu = PG_train.to(device)
            K_train = PG_train_gpu @ PG_train_gpu.T

            # K_val_cross = PG_val @ PG_train^T (n_val × n_train)
            PG_val_gpu = PG_val.to(device)
            K_val_cross = PG_val_gpu @ PG_train_gpu.T

            # K_test_cross = PG_test @ PG_train^T (n_test × n_train)
            PG_test_gpu = PG_test.to(device)
            K_test_cross = PG_test_gpu @ PG_train_gpu.T

            del PG_train_gpu

        # =========================================================================
        # Step 3: Eigendecompose K_train ONCE
        # =========================================================================
        print(f"    [4/4] Eigendecomposing K_train ({n_train}×{n_train})...")
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
        trial_results = {
            "lambda_values": [],
            "val_lds": [],
            "test_lds": [],
        }

        best_lambda = None
        best_val_lds = float('-inf')

        print(f"\n    Sweeping {len(lambda_values)} λ values...")
        for lamb in lambda_values:
            # Solve via eigendecomposition:
            # (K + nλI)^{-1} = U @ diag(1/(σ + nλ)) @ U^T
            n_lamb = n_train * lamb
            inv_diag = 1.0 / (eigenvalues + n_lamb)

            # Validation LDS (GPU-accelerated)
            val_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Kval_T)
            val_lds_score = lds_gpu(val_scores_T, val_gt, device=device)
            mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

            # Test LDS (GPU-accelerated)
            test_scores_T = U @ (inv_diag.unsqueeze(1) * UT_Ktest_T)
            test_lds_score = lds_gpu(test_scores_T, test_gt, device=device)
            mean_test_lds = torch.mean(test_lds_score[~torch.isnan(test_lds_score)]).item()

            trial_results["lambda_values"].append(lamb)
            trial_results["val_lds"].append(mean_val_lds)
            trial_results["test_lds"].append(mean_test_lds)

            if mean_val_lds > best_val_lds:
                best_val_lds = mean_val_lds
                best_lambda = lamb

            print(f"      λ = {lamb:.1e}: Val LDS = {mean_val_lds:.4f}, Test LDS = {mean_test_lds:.4f}")

        trial_results["best_lambda"] = best_lambda
        trial_results["best_val_lds"] = best_val_lds
        # Get test LDS at best lambda
        best_idx = trial_results["lambda_values"].index(best_lambda)
        trial_results["test_lds_at_best"] = trial_results["test_lds"][best_idx]

        per_trial_results.append(trial_results)

        # Clean up GPU memory before next trial
        if 'PG_train_gpu' in locals():
            del PG_train_gpu
        del PG_val_gpu, PG_test_gpu
        del K_train, K_val_cross, K_test_cross
        del eigenvalues, U, UT_Kval_T, UT_Ktest_T
        if 'val_scores_T' in locals():
            del val_scores_T
        if 'test_scores_T' in locals():
            del test_scores_T
        torch.cuda.empty_cache()

    # =========================================================================
    # Aggregate results across trials
    # =========================================================================
    print(f"\n  Aggregating results across {num_trials} trials...")

    # Average val_lds and test_lds across trials for each lambda
    avg_val_lds = []
    avg_test_lds = []
    for i, lamb in enumerate(lambda_values):
        val_lds_values = [trial["val_lds"][i] for trial in per_trial_results]
        test_lds_values = [trial["test_lds"][i] for trial in per_trial_results]
        avg_val_lds.append(np.mean(val_lds_values))
        avg_test_lds.append(np.mean(test_lds_values))

    # Select best lambda based on average validation LDS
    best_idx = np.argmax(avg_val_lds)
    best_lambda = lambda_values[best_idx]
    best_val_lds = avg_val_lds[best_idx]
    best_test_lds = avg_test_lds[best_idx]

    # Compute std for best lambda across trials
    best_val_lds_trials = [trial["val_lds"][best_idx] for trial in per_trial_results]
    best_test_lds_trials = [trial["test_lds"][best_idx] for trial in per_trial_results]
    val_lds_std = np.std(best_val_lds_trials, ddof=1) if num_trials > 1 else 0.0
    test_lds_std = np.std(best_test_lds_trials, ddof=1) if num_trials > 1 else 0.0

    results = {
        "lambda_values": lambda_values,
        "val_lds": avg_val_lds,
        "test_lds": avg_test_lds,
        "best_lambda": best_lambda,
        "best_val_lds": best_val_lds,
        "best_val_lds_std": val_lds_std,
        "test_lds": best_test_lds,
        "test_lds_std": test_lds_std,
        "proj_dim": proj_dim,
        "num_trials": num_trials,
        "per_trial_results": per_trial_results,
    }

    print(f"\n  Best λ = {best_lambda:.1e}: Val LDS = {best_val_lds:.4f}±{val_lds_std:.4f}, "
          f"Test LDS = {best_test_lds:.4f}±{test_lds_std:.4f}")

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
    num_trials: int = 1,
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
        num_trials: Number of trials with different random projections

    Returns:
        Dictionary with sweep results (averaged over trials if num_trials > 1)
    """
    _, d, dtype = _get_gradient_info(train_grad_cache)
    proj_type_enum = ProjectionType(proj_type)
    proj_bs = min(batch_size, 32)
    proj_bs = max(8, (proj_bs // 8) * 8)

    # Store per-trial results for each m
    per_m_trial_results = {m: [] for m in m_values}

    # Iterate from largest m first (most memory intensive) to fail fast if OOM
    m_values_desc = sorted(m_values, reverse=True)

    for trial in range(num_trials):
        trial_seed = seed + trial
        print(f"  Trial {trial + 1}/{num_trials} (seed={trial_seed})")

        for proj_dim in tqdm(m_values_desc, desc=f"    m sweep (λ={lamb:.1e})", leave=False):
            # Create projector for this dimension
            projector = make_random_projector(
                param_shape_list=[d],
                feature_batch_size=batch_size,
                proj_dim=proj_dim,
                proj_max_batch_size=proj_bs,
                device=torch.device(device),
                proj_seed=trial_seed,
                proj_type=proj_type_enum,
                dtype=dtype,
            )

            # Compute sketched influence scores
            val_score = compute_scores_sketched(
                train_grad_cache, val_grad_cache, projector, lamb, device, batch_size
            )

            val_lds_score = lds_gpu(val_score, val_gt, device=device)
            mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

            per_m_trial_results[proj_dim].append(mean_val_lds)

            print(f"      m = {proj_dim}: Val LDS = {mean_val_lds:.4f}")

    # Aggregate results across trials
    print(f"\n  Aggregating results across {num_trials} trials...")
    avg_val_lds = []
    val_lds_std = []

    for m in m_values:
        trial_values = per_m_trial_results[m]
        avg_val_lds.append(np.mean(trial_values))
        val_lds_std.append(np.std(trial_values, ddof=1) if num_trials > 1 else 0.0)

    results = {
        "m_values": m_values,
        "val_lds": avg_val_lds,
        "val_lds_std": val_lds_std,
        "lambda": lamb,
        "num_trials": num_trials,
        "per_trial_results": per_m_trial_results,
    }

    for i, m in enumerate(m_values):
        print(f"    m = {m}: Val LDS = {avg_val_lds[i]:.4f}±{val_lds_std[i]:.4f}")

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
    num_test_grads: int = 500,
    num_trials: int = 1,
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
        num_test_grads: Number of test gradients (before val/test split)
        num_trials: Number of random projection trials for statistical analysis
    """
    print(f"\n{'='*60}")
    print(f"Full Hyperparameter Selection Comparison (Influence Functions)")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"{'='*60}")

    # Load benchmark
    model_details, groundtruth = load_benchmark(model=model_name, dataset=dataset, metric="lds")

    # Load the checkpoint that corresponds to the ground truth
    # The ground truth LDS values were computed using models_full[0]
    model = model_details["model"]
    checkpoint = torch.load(model_details["models_full"][0], map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {model_details['models_full'][0]}")

    # Count model parameters
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

    # Get indices - limit test set to num_test_grads before splitting
    train_indices = list(model_details["train_sampler"])
    full_test_indices = list(model_details["test_sampler"])

    # Limit to num_test_grads (shuffle first for randomness)
    np.random.seed(seed)
    np.random.shuffle(full_test_indices)
    limited_test_indices = full_test_indices[:num_test_grads]

    # Create a mock sampler-like object for get_validation_split_indices
    class MockSampler:
        def __init__(self, indices):
            self._indices = indices
        def __iter__(self):
            return iter(self._indices)

    val_indices, test_indices = get_validation_split_indices(
        MockSampler(limited_test_indices), val_ratio=val_ratio, seed=seed
    )

    print(f"  Test set: {len(full_test_indices)} total -> {num_test_grads} sampled -> "
          f"{len(val_indices)} val + {len(test_indices)} test")

    # Compute ground truth for val/test splits
    # Map from dataset index to position in original test_sampler
    original_gt_values, subset_indices = groundtruth
    original_test_indices = list(model_details["test_sampler"])
    test_indices_dict = {idx: pos for pos, idx in enumerate(original_test_indices)}
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
    results["m_values"] = m_values  # Update with filtered values

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
    print(f"\n[Step 2] Sweeping λ with fixed large m ({num_trials} trials)...")
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
        num_trials=num_trials,
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
    print(f"\n[Step 3] Sweeping m with fixed λ* = {best_lambda:.1e} ({num_trials} trials)...")
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
        num_trials=num_trials,
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
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for gradient computation. Decrease if running out of memory.")

    # Gradient storage options
    parser.add_argument("--offload", type=str, default="cpu",
                       choices=["none", "cpu", "disk"],
                       help="Gradient storage: none (GPU), cpu (RAM), disk (files)")
    parser.add_argument("--cache_dir", type=str, default="./grad_cache",
                       help="Directory for gradient cache (only used with --offload disk)")

    # Lambda sweep configuration (integer powers of 10, or log-spaced if lambda_steps specified)
    parser.add_argument("--lambda_exp_min", type=int, default=-8,
                       help="Minimum exponent for λ sweep: λ_min = 10^exp_min (default: -8 → 1e-8)")
    parser.add_argument("--lambda_exp_max", type=int, default=2,
                       help="Maximum exponent for λ sweep: λ_max = 10^exp_max (default: 2 → 1e2)")
    parser.add_argument("--lambda_steps", type=int, default=0,
                       help="Number of log-spaced λ values. If 0 (default), uses integer powers of 10.")

    # Projection dimension grid (powers of 2, matching spectrum_bounds.py)
    parser.add_argument("--m_exp_min", type=int, default=5,
                       help="Minimum exponent for m sweep: m_min = 2^exp_min (default: 5 → 32)")
    parser.add_argument("--m_exp_max", type=int, default=20,
                       help="Maximum exponent for m sweep: m_max = 2^exp_max (default: 20 → 1048576)")
    parser.add_argument("--m_steps", type=int, default=0,
                       help="Number of log2-spaced m values. If 0 (default), uses integer powers of 2.")

    # Validation and test configuration
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Fraction of test set to use for validation (default: 0.1)")
    parser.add_argument("--num_test_grads", type=int, default=500,
                       help="Number of test gradients to use (before val/test split, default: 500)")
    parser.add_argument("--num_trials", type=int, default=5,
                       help="Number of random projection trials for statistical analysis (default: 5)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    proj_type = ProjectionType(args.proj_type)

    # Generate λ values: integer powers of 10 if lambda_steps=0, else log-spaced
    if args.lambda_steps == 0:
        lambda_values = [10.0 ** exp for exp in range(args.lambda_exp_min, args.lambda_exp_max + 1)]
    else:
        lambda_values = np.logspace(args.lambda_exp_min, args.lambda_exp_max, num=args.lambda_steps).tolist()

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

    # Print configuration
    print("\n" + "="*60)
    print(f"Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Offload Mode: {args.offload.upper()}")
    if args.offload == "disk":
        print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    if args.lambda_steps == 0:
        print(f"λ sweep: 10^[{args.lambda_exp_min}, {args.lambda_exp_max}] = {[f'1e{e}' for e in range(args.lambda_exp_min, args.lambda_exp_max + 1)]}")
    else:
        print(f"λ sweep: [1e{args.lambda_exp_min}, 1e{args.lambda_exp_max}] ({args.lambda_steps} log-spaced steps)")
    print(f"m values: {m_values}")
    print(f"Val ratio: {args.val_ratio}, num_test_grads: {args.num_test_grads}")
    print("="*60 + "\n")

    # Run experiment
    results = run_experiment(
        args.dataset, args.model, lambda_values, m_values,
        device=args.device, seed=args.seed,
        batch_size=args.batch_size, offload=args.offload,
        cache_dir=args.cache_dir,
        proj_type=proj_type,
        val_ratio=args.val_ratio,
        num_test_grads=args.num_test_grads,
        num_trials=args.num_trials,
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

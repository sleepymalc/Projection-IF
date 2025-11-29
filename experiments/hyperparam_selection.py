"""
Hyperparameter Selection: Compare λ and m Selection Strategies

Compares two approaches for selecting the regularization λ and sketch dimension m:

1. **Theory-Driven (Proposal):**
   Choose m based on effective dimension: m ≈ d_λ/ε²

2. **Utility-Driven (Sequential / "Taming" paper approach):**
   a) Fix m large, sweep λ to maximize LDS (utility metric)
   b) Then verify m ≥ d_λ* for the optimal λ*

Usage:
    python hyperparam_selection.py --dataset mnist --model mlp --mode full
    python hyperparam_selection.py --mode lambda_sweep --proj_dim 2048
    python hyperparam_selection.py --mode m_sweep --lamb 1e-3

    # Offload modes for memory management:
    python hyperparam_selection.py --offload none   # Keep gradients on GPU (fastest)
    python hyperparam_selection.py --offload cpu    # Offload to CPU RAM (default)
    python hyperparam_selection.py --offload disk   # Offload to disk (for large models)

Interpreting Results:
    - LDS (Linear Datamodeling Score): Higher = better attribution quality
    - λ_opt: Regularization that maximizes LDS on validation set
    - m_threshold: Minimum m where LDS stabilizes (should be ≈ d_λ)
    - If m_threshold ≈ d_λ, theory matches practice

Memory Efficiency:
    - --offload controls gradient storage: none (GPU), cpu (RAM), disk (files)
    - Uses vmap-based gradient computation for efficiency
    - --batch_size controls GPU memory usage (decrease if running out of memory)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add experiments directory to path for local utils
experiments_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(experiments_dir))

from dattri.benchmark.load import load_benchmark
from dattri.benchmark.utils import SubsetSampler
from dattri.metric import lds
from dattri.func.projection import make_random_projector, ProjectionType

from utils.fisher_utils import (
    compute_eigenvalues,
    compute_effective_dimension,
    clear_memory,
)
from utils.gradient_cache import GradientCache, create_gradient_cache


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


def create_validation_split(test_dataset, test_sampler, groundtruth, val_ratio=0.1, seed=0):
    """Split test set into validation and test sets."""
    test_indices = list(test_sampler)
    num_test = len(test_indices)

    np.random.seed(seed)
    np.random.shuffle(test_indices)
    num_val = int(val_ratio * num_test)
    val_indices = test_indices[:num_val]
    new_test_indices = test_indices[num_val:]

    val_sampler = SubsetSampler(val_indices)
    new_test_sampler = SubsetSampler(new_test_indices)

    val_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, sampler=val_sampler,
    )
    new_test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, sampler=new_test_sampler,
    )

    original_gt_values, subset_indices = groundtruth
    test_indices_dict = {idx: pos for pos, idx in enumerate(test_sampler)}

    val_gt_indices = [test_indices_dict[idx] for idx in val_indices]
    val_gt_values = original_gt_values[:, val_gt_indices]
    test_gt_indices = [test_indices_dict[idx] for idx in new_test_indices]
    test_gt_values = original_gt_values[:, test_gt_indices]

    return (
        val_dataloader, new_test_dataloader,
        (val_gt_values, subset_indices), (test_gt_values, subset_indices)
    )


def _get_gradient_info(gradients: GradientCache) -> Tuple[int, int, torch.dtype]:
    """Get n_samples, dim, dtype from GradientCache."""
    return gradients.n_samples, gradients.dim, gradients.dtype


def _get_gradient_batch(gradients: GradientCache, start: int, end: int, device: str) -> torch.Tensor:
    """Get a batch of gradients from cache."""
    return gradients.get_batch(start, end, device=device)


def influence_function(
    train_gradients: GradientCache,
    test_gradients: GradientCache,
    lamb: float,
    proj_dim: int,
    proj_type: str = "normal",
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Memory-Optimized IF Attribution using "Double Streaming" strategy.

    Computes: score[i,j] = g_test[j]^T (F + λI)^{-1} g_train[i]
    Using sketched approximation: g_test[j]^T P^T (PFP^T + λI)^{-1} P g_train[i]

    Strategy:
    1. Stream gradients to form the operator matrix (H or K) block-by-block.
    2. Invert the small operator matrix.
    3. Stream gradients again to compute scores against the inverted operator.

    Never stores the full (n_train × proj_dim) projection matrix.

    Adaptive switching based on bottleneck:
    - If m < n: Form m×m projected Fisher (PFP^T) - "Projected Fisher Method"
    - If n < m: Form n×n kernel (K) - "Kernel Method" via Woodbury identity

    Args:
        train_gradients: Training gradients stored in GradientCache
        test_gradients: Test gradients stored in GradientCache
        lamb: Regularization parameter λ
        proj_dim: Projection dimension m
        proj_type: Type of random projection ("normal", "rademacher", "sjlt")
        device: Device for computation
        seed: Random seed for projection
        batch_size: Batch size for GPU operations

    Returns:
        Attribution scores of shape (n_train, n_test)
    """
    n_train, d, dtype = _get_gradient_info(train_gradients)
    n_test, _, _ = _get_gradient_info(test_gradients)

    # Heuristic: Work in whichever dimension is smaller (m or n)
    use_kernel_method = n_train < proj_dim

    # Setup Projector
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

    scores = torch.zeros(n_train, n_test, device=device, dtype=dtype)

    if use_kernel_method:
        # === STRATEGY A: Kernel Method (Woodbury) ===
        # Efficient when n_train is small (e.g., 500-5000) and proj_dim is large
        # Form K = (PG)(PG)^T by projecting all gradients once and storing on CPU.
        #
        # Math: Using Woodbury identity on (PFP^T + λI)^{-1}
        # Final scores = (1/λ)(K_cross - K @ M^{-1} @ K_cross)
        # where K = PG @ PG^T (n×n), M = nλI + K, K_cross = PG @ PH^T

        print(f"  Using Kernel Method (n={n_train} < m={proj_dim})...")

        # 1. Project ALL training gradients once and store on CPU
        # Memory: n_train × proj_dim × 4 bytes (e.g., 5000 × 32768 × 4 = 655 MB)
        pg_train_mem_mb = n_train * proj_dim * 4 / (1024**2)
        print(f"    Projecting {n_train} train gradients (will use ~{pg_train_mem_mb:.0f} MB CPU RAM)...")

        PG_train_cpu = torch.zeros(n_train, proj_dim, dtype=dtype, device="cpu")
        for i in tqdm(range(0, n_train, batch_size), desc="    Projecting train grads", leave=False):
            i_end = min(i + batch_size, n_train)
            g_batch = _get_gradient_batch(train_gradients, i, i_end, device)
            pg_batch = projector.project(g_batch, ensemble_id=0)
            PG_train_cpu[i:i_end] = pg_batch.cpu()
            del g_batch, pg_batch
            clear_memory(device)

        # 2. Project test gradients and store on CPU
        print(f"    Projecting {n_test} test gradients...")
        PG_test_cpu = torch.zeros(n_test, proj_dim, dtype=dtype, device="cpu")
        for j in tqdm(range(0, n_test, batch_size), desc="    Projecting test grads", leave=False):
            j_end = min(j + batch_size, n_test)
            g_batch = _get_gradient_batch(test_gradients, j, j_end, device)
            pg_batch = projector.project(g_batch, ensemble_id=0)
            PG_test_cpu[j:j_end] = pg_batch.cpu()
            del g_batch, pg_batch
            clear_memory(device)

        # 3. Compute kernel matrices on GPU in batched fashion
        print(f"    Computing kernel matrix ({n_train}x{n_train})...")
        K = torch.zeros(n_train, n_train, device=device, dtype=dtype)
        K_cross = torch.zeros(n_train, n_test, device=device, dtype=dtype)

        # Process in row batches to limit GPU memory usage
        kernel_batch_size = min(batch_size * 4, n_train)  # Larger batches for efficiency
        for i in tqdm(range(0, n_train, kernel_batch_size), desc="    Building kernel", leave=False):
            i_end = min(i + kernel_batch_size, n_train)
            pg_i = PG_train_cpu[i:i_end].to(device)

            # K_cross: (batch, m) @ (m, n_test) -> (batch, n_test)
            PG_test_gpu = PG_test_cpu.to(device)
            K_cross[i:i_end] = pg_i @ PG_test_gpu.T
            del PG_test_gpu

            # K: compute full row strip at once (batch, m) @ (n_train, m)^T -> (batch, n_train)
            # Process in column batches to avoid OOM
            for j in range(0, n_train, kernel_batch_size):
                j_end = min(j + kernel_batch_size, n_train)
                pg_j = PG_train_cpu[j:j_end].to(device)
                K[i:i_end, j:j_end] = pg_i @ pg_j.T
                del pg_j

            del pg_i
            clear_memory(device)

        del PG_train_cpu, PG_test_cpu

        # 4. Solve: M = K + nλI, then x = M^{-1} @ K_cross using eigendecomposition
        print(f"    Solving linear system ({n_train}x{n_train}) via eigendecomposition...")
        K.diagonal().add_(n_train * lamb)

        # Use eigendecomposition for numerical stability with small λ
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        inv_eigenvalues = 1.0 / eigenvalues
        # x = V @ diag(1/λ) @ V^T @ K_cross
        x = eigenvectors @ (inv_eigenvalues.unsqueeze(1) * (eigenvectors.T @ K_cross))
        del eigenvalues, inv_eigenvalues

        # Restore K (remove regularization for final computation)
        K.diagonal().add_(-n_train * lamb)

        # Score = (1/λ) * (K_cross - K @ x)
        scores = (1.0 / lamb) * (K_cross - K @ x)
        print(f"    Done.")

        del K, K_cross, eigenvectors, x

    else:
        # === STRATEGY B: Direct Projected Fisher (Standard Sketching) ===
        # Efficient when m is small (e.g., 4096) and n is large (e.g., 50k)
        # Form H = PFP^T (m × m) by streaming. Never stores (n × m) matrix.
        #
        # Pass 1: Stream gradients → Project → Accumulate H = (1/n)∑(Pg)(Pg)^T
        # Pass 2: Stream gradients → Project → Multiply with H^{-1} → Final Score

        print(f"  Using Projected Fisher Method (m={proj_dim} <= n={n_train})...")

        # Pass 1: Accumulate H = (1/n) * sum((Pg)(Pg)^T)
        print(f"    Building projected Fisher matrix ({proj_dim}x{proj_dim})...")
        H = torch.zeros(proj_dim, proj_dim, device=device, dtype=dtype)

        for i in tqdm(range(0, n_train, batch_size), desc="    Pass 1: Building H", leave=False):
            i_end = min(i + batch_size, n_train)
            g_batch = _get_gradient_batch(train_gradients, i, i_end, device)
            pg_batch = projector.project(g_batch, ensemble_id=0)
            del g_batch

            # H += pg^T @ pg
            H.add_(pg_batch.T @ pg_batch)
            del pg_batch
            clear_memory(device)

        H.div_(n_train)
        H.diagonal().add_(lamb)

        # Eigendecomposition of regularized Fisher for numerical stability
        print(f"    Computing eigendecomposition of H ({proj_dim}x{proj_dim})...")
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        inv_eigenvalues = 1.0 / eigenvalues
        del H, eigenvalues

        # Pre-compute Q_test = H^{-1} @ (P G_test)^T
        # Size: (m, n_test) - acceptable since m is small in this branch
        print(f"    Computing test scores...")
        Q_test = torch.zeros(proj_dim, n_test, device=device, dtype=dtype)
        for j in tqdm(range(0, n_test, batch_size), desc="    Projecting test grads", leave=False):
            j_end = min(j + batch_size, n_test)
            g_batch = _get_gradient_batch(test_gradients, j, j_end, device)
            pg_batch = projector.project(g_batch, ensemble_id=0)  # (batch, m)
            del g_batch

            # Solve H * q = pg_batch^T using eigendecomposition
            # q = V @ diag(1/λ) @ V^T @ pg_batch^T
            Q_test[:, j:j_end] = eigenvectors @ (inv_eigenvalues.unsqueeze(1) * (eigenvectors.T @ pg_batch.T))
            del pg_batch
            clear_memory(device)

        del eigenvectors, inv_eigenvalues

        # Pass 2: Stream Train Gradients AGAIN to compute scores
        # Score_i = (P g_i)^T @ Q_test = (P g_i) @ (H^{-1} @ P G_test^T)
        print(f"    Computing final scores...")
        for i in tqdm(range(0, n_train, batch_size), desc="    Pass 2: Computing scores", leave=False):
            i_end = min(i + batch_size, n_train)
            g_batch = _get_gradient_batch(train_gradients, i, i_end, device)
            pg_batch = projector.project(g_batch, ensemble_id=0)  # (batch, m)
            del g_batch

            # (batch, m) @ (m, n_test) -> (batch, n_test)
            scores[i:i_end, :] = pg_batch @ Q_test
            del pg_batch
            clear_memory(device)

        print(f"    Done.")
        del Q_test

    if hasattr(projector, 'free_memory'):
        projector.free_memory()
    clear_memory(device)

    return scores


def run_lambda_sweep(
    train_gradients: GradientCache,
    val_gradients: GradientCache,
    test_gradients: GradientCache,
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

    OPTIMIZED: Projects gradients ONCE and reuses for all λ values.

    Strategy (Spectral Method for λ sweep):
    1. Project all train/val/test gradients once -> PG_train, PG_val, PG_test
    2. Compute kernel matrices once: K = PG_train @ PG_train^T, K_cross = PG @ PG_train^T
    3. Eigendecompose K once: K = U Σ U^T
    4. For each λ: solve via eigenvalues (milliseconds per λ!)

    This reduces complexity from O(num_λ × n × d) to O(n × d + num_λ × n²).

    Args:
        train_gradients: Training gradient cache (required)
        val_gradients: Validation gradient cache (required)
        test_gradients: Test gradient cache (required)
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
    n_train, d, dtype = _get_gradient_info(train_gradients)
    n_val, _, _ = _get_gradient_info(val_gradients)
    n_test, _, _ = _get_gradient_info(test_gradients)

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
        g_batch = train_gradients.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_train[i:i_end] = pg_batch.cpu()
        del g_batch, pg_batch
    clear_memory(device)

    print(f"  [2/4] Projecting {n_val} val + {n_test} test gradients...")
    PG_val = torch.zeros(n_val, proj_dim, dtype=dtype, device="cpu")
    for i in tqdm(range(0, n_val, batch_size), desc="    Val", leave=False):
        i_end = min(i + batch_size, n_val)
        g_batch = val_gradients.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_val[i:i_end] = pg_batch.cpu()
        del g_batch, pg_batch

    PG_test = torch.zeros(n_test, proj_dim, dtype=dtype, device="cpu")
    for i in tqdm(range(0, n_test, batch_size), desc="    Test", leave=False):
        i_end = min(i + batch_size, n_test)
        g_batch = test_gradients.get_batch(i, i_end, device=device)
        pg_batch = projector.project(g_batch, ensemble_id=0)
        PG_test[i:i_end] = pg_batch.cpu()
        del g_batch, pg_batch

    if hasattr(projector, 'free_memory'):
        projector.free_memory()
    del projector
    clear_memory(device)

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
    del PG_val_gpu, PG_val

    # K_test_cross = PG_test @ PG_train^T (n_test × n_train)
    PG_test_gpu = PG_test.to(device)
    K_test_cross = PG_test_gpu @ PG_train_gpu.T
    del PG_test_gpu, PG_test, PG_train_gpu, PG_train

    clear_memory(device)

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

    del K_train  # No longer needed after eigendecomp
    clear_memory(device)

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

        del val_scores_T, val_lds_score

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

    del eigenvalues, U, UT_Kval_T, UT_Ktest_T, K_val_cross, K_test_cross
    del test_scores_T, test_lds_score
    clear_memory(device)

    return results


def run_m_sweep(
    train_gradients: GradientCache,
    val_gradients: GradientCache,
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
        train_gradients: Training gradient cache (required)
        val_gradients: Validation gradient cache (required)
        val_gt: Validation ground truth for LDS computation
        lamb: Fixed regularization parameter λ
        m_values: List of projection dimensions to sweep
        proj_type: Type of random projection
        device: Device for computation
        seed: Random seed
        batch_size: Batch size for influence function computation

    Returns:
        Dictionary with sweep results
    """
    results = {"m_values": [], "val_lds": [], "lambda": lamb}

    for proj_dim in tqdm(m_values, desc=f"m sweep (λ={lamb:.1e})"):
        # Compute attribution scores using influence functions
        val_score = influence_function(
            train_gradients=train_gradients,
            test_gradients=val_gradients,
            lamb=lamb,
            proj_dim=proj_dim,
            proj_type=proj_type,
            device=device,
            seed=seed,
            batch_size=batch_size,
        )

        val_lds_score = lds(val_score, val_gt)[0]
        mean_val_lds = torch.mean(val_lds_score[~torch.isnan(val_lds_score)]).item()

        results["m_values"].append(proj_dim)
        results["val_lds"].append(mean_val_lds)

        print(f"  m = {proj_dim}: Val LDS = {mean_val_lds:.4f}")

        # Memory cleanup
        del val_score, val_lds_score
        clear_memory(device)

    return results


def effective_dimension(
    lambda_values: List[float],
    gradients: GradientCache,
    device: str = "cuda",
    batch_size: int = 32,
    eigenvalues: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute effective dimension d_λ for the model.

    Uses exact eigendecomposition of the Gram matrix K = G G^T, which shares
    eigenvalues with the Fisher F = G^T G / n. This is efficient because:
    - K is n×n (small when n << d)
    - Eigenvalues are computed once and reused for all λ values
    - d_λ = Σ σᵢ/(σᵢ + λ) is computed analytically

    Args:
        lambda_values: List of regularization values to compute d_λ for
        gradients: Pre-computed gradient cache (required)
        device: Device for computation
        batch_size: Batch size for Gram matrix computation
        eigenvalues: Pre-computed eigenvalues (optional, will compute if None)

    Returns:
        Dictionary mapping λ -> d_λ
    """
    # Compute eigenvalues once (O(n²d) for Gram + O(n³) for eigendecomp)
    # Use larger batch size for Gram matrix computation (less memory-intensive than gradients)
    if eigenvalues is None:
        print("  Computing eigenvalues of Gram matrix...")
        gram_batch_size = batch_size * 4  # Gram computation can handle larger batches
        eigenvalues = compute_eigenvalues(gradients, device=device, batch_size=gram_batch_size)

    # Compute d_λ analytically for each λ (O(n) per λ - instant!)
    d_lambda_dict = {}
    for lamb in lambda_values:
        d_lambda = compute_effective_dimension(eigenvalues, lamb)
        d_lambda_dict[lamb] = d_lambda
        print(f"  λ = {lamb:.1e}: d_λ = {d_lambda:.1f}")

    clear_memory(device)

    return d_lambda_dict


def run_full_comparison(
    dataset: str,
    model_name: str,
    lambda_values: List[float],
    m_values: List[int],
    device: str = "cuda",
    seed: int = 42,
    batch_size: int = 32,
    offload: str = "cpu",
    cache_dir: Optional[str] = None,
    proj_type: str = "normal",
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
        proj_type: Projection type ("normal", "rademacher", "sjlt")
        val_ratio: Fraction of test set to use for validation
    """
    print(f"\n{'='*60}")
    print(f"Full Hyperparameter Selection Comparison (Influence Functions)")
    print(f"Dataset: {dataset}, Model: {model_name}")
    print(f"Projection type: {proj_type}, Batch size: {batch_size}")
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
    train_gradients = create_gradient_cache(
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
    val_gradients = create_gradient_cache(
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
    test_gradients = create_gradient_cache(
        model=model,
        dataset=model_details["test_dataset"],
        indices=test_indices,
        device=device,
        offload=offload,
        model_type=model_type,
        batch_size=batch_size,
        cache_dir=f"{cache_dir}/test" if cache_dir else None,
    )

    print(f"  Gradient caches ready: train={train_gradients.n_samples}, "
          f"val={val_gradients.n_samples}, test={test_gradients.n_samples}")

    # =========================================================================
    # Step 1: Compute effective dimensions
    # =========================================================================
    print("\n[Step 1] Computing effective dimensions d_λ(F)...")
    d_lambda_dict = effective_dimension(
        lambda_values=lambda_values,
        gradients=train_gradients,
        device=device,
        batch_size=batch_size,
    )
    results["d_lambda"] = d_lambda_dict
    clear_memory(device)

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
        train_gradients=train_gradients,
        val_gradients=val_gradients,
        test_gradients=test_gradients,
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
    clear_memory(device)

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
        train_gradients=train_gradients,
        val_gradients=val_gradients,
        val_gt=val_gt,
        lamb=best_lambda,
        m_values=m_values,
        proj_type=proj_type,
        device=device,
        seed=seed,
        batch_size=batch_size,
    )
    results["m_sweep"] = m_sweep_results

    # =========================================================================
    # Cleanup and final analysis
    # =========================================================================
    del train_gradients, val_gradients, test_gradients
    clear_memory(device)

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
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "lambda_sweep", "m_sweep"],
                       help="Run mode: full comparison, just λ sweep, or just m sweep")
    parser.add_argument("--proj_dim", type=int, default=2048,
                       help="Projection dimension for λ sweep mode")
    parser.add_argument("--lamb", type=float, default=1e-3,
                       help="λ value for m sweep mode")
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

    # Print configuration
    offload_desc = {"none": "GPU (fastest)", "cpu": "CPU RAM", "disk": "Disk files"}
    print("\n" + "="*60)
    print(f"Configuration")
    print("="*60)
    print(f"Dataset: {args.dataset}, Model: {args.model}")
    print(f"Gradient storage: {offload_desc[args.offload]}")
    if args.offload == "disk":
        print(f"Cache directory: {args.cache_dir}")
    print(f"Batch size: {args.batch_size}")
    print("="*60 + "\n")

    # Define search spaces
    lambda_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    m_values = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]

    # Save results with organized directory structure
    experiment_dir = os.path.join(args.output_dir, "hyperparam_selection")
    os.makedirs(experiment_dir, exist_ok=True)

    if args.mode == "full":
        # Full comparison mode - run_full_comparison handles all gradient computation
        results = run_full_comparison(
            args.dataset, args.model, lambda_values, m_values,
            device=args.device, seed=args.seed,
            batch_size=args.batch_size, offload=args.offload,
            cache_dir=args.cache_dir,
            proj_type=args.proj_type,
        )

        results_filename = f"{args.dataset}_{args.model}_full.pt"
        results_path = os.path.join(experiment_dir, results_filename)
        torch.save(results, results_path)
        print(f"\nResults saved to {results_path}")

    elif args.mode == "lambda_sweep":
        # Lambda sweep mode - compute gradients here, then call run_lambda_sweep
        model_details, groundtruth = load_benchmark(model=args.model, dataset=args.dataset, metric="lds")
        model = model_details["model"]
        model_type = "musictransformer" if args.model == "musictransformer" else "default"

        # Get indices
        train_indices = list(model_details["train_sampler"])
        val_indices, test_indices = get_validation_split_indices(
            model_details["test_sampler"], val_ratio=0.1, seed=args.seed
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

        # Compute gradients
        print("Computing gradient caches...")
        train_gradients = create_gradient_cache(
            model=model, dataset=model_details["train_dataset"], indices=train_indices,
            device=args.device, offload=args.offload, model_type=model_type,
            batch_size=args.batch_size,
            cache_dir=f"{args.cache_dir}/train" if args.cache_dir else None,
        )
        val_gradients = create_gradient_cache(
            model=model, dataset=model_details["test_dataset"], indices=val_indices,
            device=args.device, offload=args.offload, model_type=model_type,
            batch_size=args.batch_size,
            cache_dir=f"{args.cache_dir}/val" if args.cache_dir else None,
        )
        test_gradients = create_gradient_cache(
            model=model, dataset=model_details["test_dataset"], indices=test_indices,
            device=args.device, offload=args.offload, model_type=model_type,
            batch_size=args.batch_size,
            cache_dir=f"{args.cache_dir}/test" if args.cache_dir else None,
        )

        results = run_lambda_sweep(
            train_gradients=train_gradients,
            val_gradients=val_gradients,
            test_gradients=test_gradients,
            val_gt=val_gt,
            test_gt=test_gt,
            proj_dim=args.proj_dim,
            lambda_values=lambda_values,
            proj_type=args.proj_type,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
        )

        del train_gradients, val_gradients, test_gradients
        clear_memory(args.device)

        results_filename = f"{args.dataset}_{args.model}_lambda_sweep_m{args.proj_dim}.pt"
        results_path = os.path.join(experiment_dir, results_filename)
        torch.save(results, results_path)
        print(f"Results saved to {results_path}")

    elif args.mode == "m_sweep":
        # M sweep mode - compute gradients here, then call run_m_sweep
        model_details, groundtruth = load_benchmark(model=args.model, dataset=args.dataset, metric="lds")
        model = model_details["model"]
        model_type = "musictransformer" if args.model == "musictransformer" else "default"

        # Get indices
        train_indices = list(model_details["train_sampler"])
        val_indices, _ = get_validation_split_indices(
            model_details["test_sampler"], val_ratio=0.1, seed=args.seed
        )

        # Compute ground truth for val split
        original_gt_values, subset_indices = groundtruth
        test_indices_dict = {idx: pos for pos, idx in enumerate(model_details["test_sampler"])}
        val_gt_indices = [test_indices_dict[idx] for idx in val_indices]
        val_gt_values = original_gt_values[:, val_gt_indices]
        val_gt = (val_gt_values, subset_indices)

        # Compute gradients
        print("Computing gradient caches...")
        train_gradients = create_gradient_cache(
            model=model, dataset=model_details["train_dataset"], indices=train_indices,
            device=args.device, offload=args.offload, model_type=model_type,
            batch_size=args.batch_size,
            cache_dir=f"{args.cache_dir}/train" if args.cache_dir else None,
        )
        val_gradients = create_gradient_cache(
            model=model, dataset=model_details["test_dataset"], indices=val_indices,
            device=args.device, offload=args.offload, model_type=model_type,
            batch_size=args.batch_size,
            cache_dir=f"{args.cache_dir}/val" if args.cache_dir else None,
        )

        results = run_m_sweep(
            train_gradients=train_gradients,
            val_gradients=val_gradients,
            val_gt=val_gt,
            lamb=args.lamb,
            m_values=m_values,
            proj_type=args.proj_type,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
        )

        del train_gradients, val_gradients
        clear_memory(args.device)

        results_filename = f"{args.dataset}_{args.model}_m_sweep_lamb{args.lamb}.pt"
        results_path = os.path.join(experiment_dir, results_filename)
        torch.save(results, results_path)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

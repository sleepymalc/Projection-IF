"""
Fisher Information Matrix utilities for theoretical validation experiments.

This module provides memory-efficient functions to compute:
- Per-sample gradients using vmap (vectorized map)
- Eigenvalues of the empirical Fisher Information Matrix (eFIM)
- Effective dimension d_λ(F)
- Sandwich bounds for sketched inverses
- Hutchinson trace estimator for scalable d_λ estimation

Key insight: For n samples with d parameters, we always use the Gram matrix
G @ G^T / n (n×n) instead of Fisher F = G^T @ G / n (d×d) when n < d,
since they share the same nonzero eigenvalues. This is critical because
typically n << d in neural networks.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Any, TYPE_CHECKING
import gc
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.func import grad, vmap, functional_call
from tqdm import tqdm

# Import GradientCache for explicit type checking (avoids circular import at runtime)
if TYPE_CHECKING:
    from .gradient_cache import GradientCache


# =============================================================================
# Memory Management Utilities
# =============================================================================

def clear_memory(device: str = "cuda"):
    """Explicitly clear GPU and CPU memory."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Gradient Computation (vmap-based)
# =============================================================================

def compute_per_sample_gradients(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    device: str = "cuda",
    batch_size: int = 32,
    model_type: str = "default",
) -> list[Tensor]:
    """
    Compute per-sample gradients using vmap for efficiency.

    Uses torch.func's vmap(grad(...)) approach which is significantly faster
    than computing gradients one sample at a time with backward().

    Args:
        model: The neural network model
        dataset: Dataset containing training samples
        indices: List of sample indices to compute gradients for
        device: Device for computation
        batch_size: Batch size for vmap gradient computation
        model_type: "musictransformer" for sequence models, "default" for image classification

    Returns:
        List of gradient tensors on CPU, each of shape (d,)
    """
    model = model.to(device)
    model.eval()

    # Get named parameters for functional_call
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    d_params = sum(p.numel() for p in params.values())

    print(f"Computing gradients for {len(indices)} samples using vmap...")
    print(f"Model has {d_params:,} parameters")
    print(f"Batch size: {batch_size}")

    # Define loss function in torch.func style
    if model_type == "musictransformer":
        def loss_func(params_dict, data):
            input_seq, target_seq = data
            input_seq = input_seq.unsqueeze(0)
            output = functional_call(model, params_dict, (input_seq,))
            output_flat = output.view(-1, output.size(-1))
            target_flat = target_seq.view(-1)
            return nn.CrossEntropyLoss()(output_flat, target_flat)
    else:
        def loss_func(params_dict, data):
            image, label = data
            image = image.unsqueeze(0)
            output = functional_call(model, params_dict, (image,))
            return nn.CrossEntropyLoss()(output.squeeze(0), label)

    # Create vmap'd gradient function
    grad_func = vmap(grad(loss_func), in_dims=(None, 0), randomness="different")

    gradients_cpu = []

    for batch_start in tqdm(range(0, len(indices), batch_size), desc="Computing gradients"):
        batch_end = min(batch_start + batch_size, len(indices))
        batch_indices = indices[batch_start:batch_end]

        # Collate batch data
        if model_type == "musictransformer":
            inputs = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
            targets = torch.stack([dataset[idx][1] for idx in batch_indices]).to(device)
            batch_data = (inputs, targets)
        else:
            images = torch.stack([dataset[idx][0] for idx in batch_indices]).to(device)
            labels = torch.tensor([dataset[idx][1] for idx in batch_indices]).to(device)
            batch_data = (images, labels)

        # Compute per-sample gradients using vmap
        with torch.no_grad():
            grad_dict = grad_func(params, batch_data)

        # Flatten and concatenate gradients for each sample
        for i in range(len(batch_indices)):
            grad_flat = torch.cat([
                grad_dict[k][i].view(-1) for k in params.keys()
            ])
            gradients_cpu.append(grad_flat.cpu())

        # Memory cleanup
        del grad_dict, batch_data
        if model_type == "musictransformer":
            del inputs, targets
        else:
            del images, labels
        clear_memory(device)

    return gradients_cpu


class _DiskBatchDataset(torch.utils.data.Dataset):
    """
    Dataset for loading gradient batches from disk files.

    Designed for use with DataLoader's multiprocessing workers to parallelize
    disk I/O. Only stores paths (picklable), not GradientCache object.
    """

    def __init__(self, cache_dir: str, n_batches: int):
        self.cache_dir = cache_dir
        self.n_batches = n_batches

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> Tuple[int, Tensor]:
        """Load a batch from disk. Called by DataLoader workers (separate processes)."""
        from pathlib import Path
        batch_file = Path(self.cache_dir) / f"batch_{batch_idx}.pt"
        data = torch.load(batch_file, map_location="cpu", weights_only=True)
        return batch_idx, data


def compute_gram_matrix(
    gradients: "GradientCache",
    device: str = "cuda",
    batch_size: int = 128,
) -> Tensor:
    """
    Compute Gram matrix K = G @ G^T / n from GradientCache.

    Works with all GradientCache offload modes (none, cpu, disk).

    Uses block-wise computation to minimize memory usage:
    - Only loads O(batch_size) gradients at a time
    - Exploits symmetry: only computes upper triangle
    - Memory: O(batch_size * d) for gradients + O(n²) for result

    Args:
        gradients: GradientCache object (any offload mode)
        device: Device for computation (default: "cuda")
        batch_size: Batch size for block computation

    Returns:
        K: Gram matrix (n, n) on the specified device, normalized by n
    """
    n = gradients.n_samples
    effective_batch_size = batch_size if batch_size else 100
    n_batches = math.ceil(n / effective_batch_size)

    print(f"Computing Gram matrix for {n} samples ({n_batches} batches, size={effective_batch_size})...")

    # Initialize output matrix
    K = torch.zeros(n, n, device=device)

    # Determine mode
    use_disk_mode = gradients.offload == "disk"
    use_cuda = device == "cuda" and torch.cuda.is_available()

    def get_batch_range(batch_idx: int) -> Tuple[int, int]:
        """Get (start, end) sample indices for a batch."""
        start = batch_idx * effective_batch_size
        end = min(start + effective_batch_size, n)
        return start, end

    # Setup CUDA stream for overlapped transfers
    transfer_stream = torch.cuda.Stream() if use_cuda else None
    compute_stream = torch.cuda.current_stream() if use_cuda else None

    if use_disk_mode:
        # =======================================================================
        # Disk mode: Double-buffering with background disk prefetch
        #
        # Strategy:
        # 1. Use ThreadPoolExecutor to load next batch from disk in background
        # 2. Use CUDA streams to overlap CPU->GPU transfer with computation
        # 3. LRU cache on CPU avoids repeated disk reads for same batch
        #
        # Timeline per inner iteration:
        #   [Compute j] [Transfer j+1] [Disk load j+2 (background)]
        # =======================================================================
        from concurrent.futures import ThreadPoolExecutor

        disk_executor = ThreadPoolExecutor(max_workers=2)  # 2 workers for better overlap

        def load_to_cpu(batch_idx):
            """Load batch to CPU, using cache if available."""
            start, end = get_batch_range(batch_idx)
            return gradients.get_batch(start, end, device="cpu")

        for i_batch in tqdm(range(n_batches), desc="Computing Gram matrix"):
            i_start, i_end = get_batch_range(i_batch)

            # Load G_i
            G_i_cpu = load_to_cpu(i_batch)
            G_i_gpu = G_i_cpu.to(device)

            # Double-buffering state
            next_future = None
            next_gpu = None

            for j_batch in range(i_batch, n_batches):
                j_start, j_end = get_batch_range(j_batch)

                # Get current G_j
                if j_batch == i_batch:
                    G_j_gpu = G_i_gpu
                elif next_gpu is not None:
                    # Wait for prefetched GPU tensor
                    if transfer_stream is not None:
                        compute_stream.wait_stream(transfer_stream)
                    G_j_gpu = next_gpu
                    next_gpu = None
                else:
                    # No prefetch available, load synchronously
                    G_j_cpu = load_to_cpu(j_batch)
                    G_j_gpu = G_j_cpu.to(device)

                # Start prefetching j+2 from disk (if not already cached)
                # This runs in background while we compute and transfer
                if j_batch + 2 < n_batches and j_batch + 2 > i_batch:
                    disk_executor.submit(load_to_cpu, j_batch + 2)

                # Start async transfer of j+1 to GPU while computing with j
                if j_batch + 1 < n_batches and j_batch + 1 > i_batch:
                    # Get j+1 from disk/cache (may block if not prefetched)
                    if next_future is not None:
                        next_cpu = next_future.result()
                        next_future = None
                    else:
                        next_cpu = load_to_cpu(j_batch + 1)

                    # Async transfer to GPU
                    if transfer_stream is not None:
                        with torch.cuda.stream(transfer_stream):
                            next_gpu = next_cpu.to(device, non_blocking=True)
                    else:
                        next_gpu = next_cpu.to(device)

                # Compute block K[i,j] = G_i @ G_j^T
                block = G_i_gpu @ G_j_gpu.T
                K[i_start:i_end, j_start:j_end] = block

                # Fill symmetric block K[j,i] = K[i,j]^T
                if i_batch != j_batch:
                    K[j_start:j_end, i_start:i_end] = block.T
                    del G_j_gpu

            del G_i_gpu

            if i_batch % 5 == 0 and use_cuda:
                torch.cuda.empty_cache()

        disk_executor.shutdown(wait=True)
        gradients.clear_cache()

    else:
        # =======================================================================
        # CPU/none mode: Direct memory access (original efficient path)
        # =======================================================================
        storage = gradients._memory_storage

        for i_batch in tqdm(range(n_batches), desc="Computing Gram matrix"):
            i_start, i_end = get_batch_range(i_batch)
            G_i_cpu = storage[i_start:i_end]

            if transfer_stream is not None:
                with torch.cuda.stream(transfer_stream):
                    G_i_gpu = G_i_cpu.to(device, non_blocking=True)
                compute_stream.wait_stream(transfer_stream)
            else:
                G_i_gpu = G_i_cpu.to(device)

            next_j_gpu = None

            for j_batch in range(i_batch, n_batches):
                j_start, j_end = get_batch_range(j_batch)

                if j_batch == i_batch:
                    G_j_gpu = G_i_gpu
                elif next_j_gpu is not None:
                    if transfer_stream is not None:
                        compute_stream.wait_stream(transfer_stream)
                    G_j_gpu = next_j_gpu
                    next_j_gpu = None
                else:
                    G_j_cpu = storage[j_start:j_end]
                    G_j_gpu = G_j_cpu.to(device)

                # Prefetch next j
                if j_batch + 1 < n_batches and j_batch + 1 > i_batch and transfer_stream is not None:
                    next_j_start, next_j_end = get_batch_range(j_batch + 1)
                    next_j_cpu = storage[next_j_start:next_j_end]
                    with torch.cuda.stream(transfer_stream):
                        next_j_gpu = next_j_cpu.to(device, non_blocking=True)

                block = G_i_gpu @ G_j_gpu.T
                K[i_start:i_end, j_start:j_end] = block
                if i_batch != j_batch:
                    K[j_start:j_end, i_start:i_end] = block.T

                if j_batch != i_batch:
                    del G_j_gpu

            del G_i_gpu

            if i_batch % 20 == 0 and use_cuda:
                torch.cuda.empty_cache()

    gc.collect()
    clear_memory(device)

    return K / n


def compute_eigenvalues(
    gradients: "GradientCache",
    device: str = "cuda",
    batch_size: int = 128,
) -> Tensor:
    """
    Compute eigenvalues of F = G^T @ G / n.

    Uses Gram matrix K = G @ G^T / n (n×n) since K and F share
    the same nonzero eigenvalues. This is efficient because typically n << d.

    Args:
        gradients: GradientCache containing the gradients
        device: Device for computation
        batch_size: Batch size for streaming

    Returns:
        eigenvalues: Eigenvalues of F in descending order
    """
    K = compute_gram_matrix(gradients, device, batch_size)
    eigenvalues = torch.linalg.eigvalsh(K)
    del K
    clear_memory(device)

    eigenvalues = eigenvalues.flip(0)
    return torch.clamp(eigenvalues, min=0)


def compute_effective_dimension(eigenvalues: Tensor, lamb: float) -> float:
    """
    Compute effective dimension from eigenvalues.

    d_λ(F) = tr(F(F + λI)^{-1}) = Σ_i λ_i / (λ_i + λ)

    Args:
        eigenvalues: Eigenvalues of F (from compute_eigenvalues)
        lamb: Regularization parameter λ

    Returns:
        d_lambda: The effective dimension
    """
    eigenvalues = torch.clamp(eigenvalues, min=0)
    d_lambda = torch.sum(eigenvalues / (eigenvalues + lamb))
    return d_lambda.item()


def estimate_effective_dimension(
    gradients: "GradientCache",
    lamb: float,
    device: str = "cuda",
    num_probes: int = 100,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    Estimate effective dimension using Hutchinson's trace estimator with Woodbury formula.

    d_λ = tr(F(F + λI)^{-1}) ≈ E[z^T F (F + λI)^{-1} z] where z ~ N(0, I)

    Uses the Woodbury identity to work in n-dimensional space (n = num samples)
    instead of d-dimensional space (d = num parameters), which is much more efficient
    when gradients are stored on CPU.

    Key insight from proposal Section 4:
        z^T F(F + λI)^{-1} z = u^T (λnI + K)^{-1} u, where u = G^T z and K = G^T G

    This reduces the cost from O(num_probes * cg_iters * n * d) to O(n^2 * d + n^3 + num_probes * n * d).

    Args:
        gradients: GradientCache containing the gradients
        lamb: Regularization parameter λ
        device: GPU device for computation
        num_probes: Number of random probe vectors
        batch_size: Batch size for streaming operations

    Returns:
        Tuple of (estimated d_lambda, standard error)
    """
    n = gradients.n_samples
    d = gradients.dim
    dtype = gradients.dtype

    # Step 1: Compute Gram matrix K = G^T G (n x n) via streaming
    # This is the key: work in n-dimensional space, not d-dimensional
    print(f"  Building Gram matrix K ({n}x{n}) for Woodbury formula...")
    K = compute_gram_matrix(gradients, device=device, batch_size=batch_size)
    # Note: compute_gram_matrix returns K/n, we need K = n * (K/n)
    K = K * n  # Now K = G^T G (unnormalized)

    # Step 2: Precompute M = (λnI + K)^{-1} using eigendecomposition
    # Eigendecomposition is more numerically stable than Cholesky for small λ
    print(f"  Computing eigendecomposition of (λnI + K)...")
    reg_matrix = K + lamb * n * torch.eye(n, device=device, dtype=dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(reg_matrix)
    inv_eigenvalues = 1.0 / eigenvalues
    del reg_matrix, eigenvalues
    clear_memory(device)

    # Step 3: For each probe, compute u = G^T z, then estimate = u^T M^{-1} u
    # We batch the probes for efficiency: compute U = G^T Z for all probes at once
    print(f"  Computing {num_probes} Hutchinson probes (batched)...")

    # Generate all probe vectors at once
    Z = torch.randn(d, num_probes, device=device, dtype=dtype)

    # Compute U = G^T Z by streaming over gradients (one pass!)
    U = torch.zeros(n, num_probes, device=device, dtype=dtype)
    for i_start in tqdm(range(0, n, batch_size), desc="  Computing G^T Z"):
        i_end = min(i_start + batch_size, n)
        G_batch = gradients.get_batch(i_start, i_end, device=device)
        # G_batch is (batch_size, d), Z is (d, num_probes)
        # U[i_start:i_end] = G_batch @ Z is (batch_size, num_probes)
        U[i_start:i_end] = G_batch @ Z
        del G_batch

    del Z
    clear_memory(device)

    # Step 4: Compute estimates = diag(U^T M^{-1} U) = diag(U^T (λnI + K)^{-1} U)
    # M^{-1} = V @ diag(1/eigenvalues) @ V^T, so X = V @ diag(1/λ) @ V^T @ U
    X = eigenvectors @ (inv_eigenvalues.unsqueeze(1) * (eigenvectors.T @ U))
    estimates = (U * X).sum(dim=0)  # Element-wise multiply and sum over n dimension

    del U, X, eigenvectors, inv_eigenvalues, K
    clear_memory(device)

    mean_estimate = estimates.mean().item()
    std_error = (estimates.std() / (num_probes ** 0.5)).item()

    return mean_estimate, std_error

def _matvec_streaming(
    cache: "GradientCache",
    v: Tensor,
    lamb: float,
    device: str,
    batch_size: int = 500,
) -> Tensor:
    """
    Compute (F + λI) v = (1/n) G^T (G v) + λv using streaming.

    Args:
        cache: GradientCache containing gradients
        v: Vector to multiply (d,) on GPU
        lamb: Regularization parameter
        device: GPU device
        batch_size: Batch size for streaming

    Returns:
        result: (F + λI) v on GPU
    """
    n = cache.n_samples
    d = v.shape[0]

    # First compute G @ v in streaming fashion
    Gv = torch.zeros(n, device=device)
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        G_batch = cache.get_batch(i_start, i_end, device=device)
        Gv[i_start:i_end] = G_batch @ v
        del G_batch

    # Then compute G^T @ (G @ v) in streaming fashion
    GTGv = torch.zeros(d, device=device)
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        G_batch = cache.get_batch(i_start, i_end, device=device)
        GTGv += G_batch.T @ Gv[i_start:i_end]
        del G_batch

    # F v = (1/n) G^T G v
    Fv = GTGv / n

    return Fv + lamb * v


def _cg_solve_fisher_streaming(
    cache: "GradientCache",
    b: Tensor,
    lamb: float,
    device: str,
    max_iter: int = 100,
    tol: float = 1e-6,
    batch_size: int = 50,
) -> Tensor:
    """
    Solve (F + λI) x = b using CG with streaming gradient access.

    Args:
        cache: GradientCache containing gradients
        b: RHS vector (d,) on GPU
        lamb: Regularization
        device: GPU device
        max_iter: Maximum iterations
        tol: Convergence tolerance
        batch_size: Batch size for streaming

    Returns:
        x: Solution vector on GPU
    """
    d = b.shape[0]
    dtype = b.dtype

    def matvec(v):
        return _matvec_streaming(cache, v, lamb, device, batch_size)

    # Initialize
    x = torch.zeros(d, device=device, dtype=dtype)
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)

        if rs_new.sqrt() < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x

def compute_sandwich_bounds(
    gradients: "GradientCache",
    projector: Union[Tensor, Any],
    lamb: float,
    test_vectors: Optional[Union[list, Tensor]] = None,
    num_test_vectors: int = 50,
    device: str = "cuda",
    batch_size: int = 500,
    K: Optional[Tensor] = None,
) -> dict:
    """
    Compute sandwich bounds for sketched inverse approximation quality.

    Automatically handles:
    - Dense projection matrix (Tensor) or dattri projector (from make_random_projector)
    - Woodbury identity (if K provided) or CG for exact scores

    Args:
        gradients: GradientCache containing the gradients
        projector: Either a (m, d) projection matrix or a dattri projector with .project() method
        lamb: Regularization parameter λ
        test_vectors: Test vectors. If None, uses first num_test_vectors from gradients.
        num_test_vectors: Number of test vectors if not provided
        device: GPU device for computation
        batch_size: Batch size for streaming operations
        K: Pre-computed Gram matrix (n, n) for Woodbury. If None, uses CG.

    Returns:
        Dictionary with exact_scores, sketched_scores, ratios, and statistics
    """
    has_project_method = hasattr(projector, 'project')

    n = gradients.n_samples
    d = gradients.dim
    dtype = gradients.dtype

    if has_project_method:
        m = projector.proj_dim
    else:
        m = projector.shape[0]

    # Handle test vectors
    if test_vectors is None:
        test_vectors = [gradients.get_sample(i, device="cpu") for i in range(min(num_test_vectors, n))]

    k = len(test_vectors) if isinstance(test_vectors, list) else test_vectors.shape[0]

    # Convert test vectors to tensor if needed
    if isinstance(test_vectors, list):
        V = torch.stack(test_vectors).to(device)  # (k, d)
    else:
        V = test_vectors.to(device)

    # Compute PFP^T
    PFPT = _compute_pfpt(gradients, projector, n, m, dtype, device, batch_size, has_project_method)

    # Add regularization
    PFPT_reg = PFPT + lamb * torch.eye(m, device=device, dtype=dtype)
    del PFPT

    # Compute all scores (batched for efficiency)
    exact_scores = _compute_exact_scores(
        V, gradients, lamb, device, batch_size, K
    )

    sketched_scores = _compute_sketched_scores(
        V, projector, PFPT_reg, has_project_method, batch_size
    )

    del PFPT_reg, V
    clear_memory(device)

    return _format_results(exact_scores, sketched_scores)


def _compute_pfpt(
    gradients: "GradientCache",
    projector,
    n: int,
    m: int,
    dtype,
    device: str,
    batch_size: int,
    has_project_method: bool,
) -> Tensor:
    """Compute projected Fisher PFP^T."""
    PFPT = torch.zeros(m, m, device=device, dtype=dtype)

    for i_start in tqdm(range(0, n, batch_size), desc="Computing PFP^T"):
        i_end = min(i_start + batch_size, n)
        G_batch = gradients.get_batch(i_start, i_end, device=device)

        if has_project_method:
            PG_batch = projector.project(G_batch, ensemble_id=0)
        else:
            PG_batch = (projector @ G_batch.T).T

        PFPT += PG_batch.T @ PG_batch
        del G_batch, PG_batch
        clear_memory(device)

    return PFPT / n


def _compute_exact_scores(
    V: Tensor,
    gradients: "GradientCache",
    lamb: float,
    device: str,
    batch_size: int,
    K: Optional[Tensor],
) -> Tensor:
    """
    Compute exact scores v^T (F + λI)^{-1} v for all test vectors in batch.

    Args:
        V: Test vectors of shape (k, d) on device
        gradients: GradientCache containing the gradients
        lamb: Regularization parameter
        device: GPU device
        batch_size: Batch size for streaming
        K: Pre-computed Gram matrix (n, n) or None

    Returns:
        Tensor of shape (k,) containing exact scores
    """
    k, d = V.shape
    dtype = V.dtype
    n = gradients.n_samples

    if K is not None:
        # Batched Woodbury: score_i = (1/λ) [||v_i||² - u_i^T x_i]
        # where u_i = G @ v_i and (nλI + K) x_i = u_i

        # Compute U = G @ V^T -> (n, k), so U[:, i] = G @ v_i
        U = torch.zeros(n, k, device=device, dtype=dtype)
        for i_start in range(0, n, batch_size):
            i_end = min(i_start + batch_size, n)
            G_batch = gradients.get_batch(i_start, i_end, device=device)
            U[i_start:i_end, :] = G_batch @ V.T  # (batch, k)
            del G_batch

        # Solve (nλI + K) X = U for all columns at once
        K_dev = K.to(device)
        reg_matrix = K_dev + n * lamb * torch.eye(n, device=device, dtype=K_dev.dtype)
        X = torch.linalg.solve(reg_matrix, U)  # (n, k)

        # Compute scores: score_i = (1/λ) [||v_i||² - u_i^T x_i]
        v_norms_sq = (V ** 2).sum(dim=1)  # (k,)
        ux_dots = (U * X).sum(dim=0)  # (k,) - dot products for each column
        scores = (v_norms_sq - ux_dots) / lamb

        del U, X, K_dev, reg_matrix
        return scores
    else:
        # Fall back to CG for each vector (can't easily batch CG)
        scores = torch.zeros(k, device=device, dtype=dtype)
        for i in tqdm(range(k), desc="Computing exact scores (CG)"):
            v = V[i]
            exact_v = _cg_solve_fisher_streaming(
                gradients, v, lamb, device,
                max_iter=200, tol=1e-8, batch_size=batch_size
            )
            scores[i] = torch.dot(v, exact_v)
            del exact_v
        return scores


def _compute_sketched_scores(
    V: Tensor,
    projector,
    PFPT_reg: Tensor,
    has_project_method: bool,
    batch_size: int = 500,
) -> Tensor:
    """
    Compute sketched scores v^T P^T (PFP^T + λI)^{-1} P v for all test vectors.

    Args:
        V: Test vectors of shape (k, d) on device
        projector: Either a (m, d) projection matrix or a dattri projector
        PFPT_reg: Regularized PFP^T matrix of shape (m, m)
        has_project_method: Whether projector is a dattri projector (has .project() method)
        batch_size: Batch size for operator projection

    Returns:
        Tensor of shape (k,) containing sketched scores
    """
    k = V.shape[0]
    device = V.device
    dtype = V.dtype

    # Project all test vectors: PV has shape (k, m)
    if has_project_method:
        # Process in batches if needed
        if k <= batch_size:
            PV = projector.project(V, ensemble_id=0)  # (k, m)
        else:
            PV_list = []
            for i_start in range(0, k, batch_size):
                i_end = min(i_start + batch_size, k)
                PV_batch = projector.project(V[i_start:i_end], ensemble_id=0)
                PV_list.append(PV_batch)
            PV = torch.cat(PV_list, dim=0)  # (k, m)
    else:
        PV = (projector @ V.T).T  # (k, m)

    # Solve (PFP^T + λI) X = PV^T for all columns at once
    # X has shape (m, k)
    X = torch.linalg.solve(PFPT_reg, PV.T)  # (m, k)

    # Compute scores: score_i = PV[i] @ X[:, i]
    scores = (PV * X.T).sum(dim=1)  # (k,)

    del PV, X
    return scores


def _format_results(exact_scores: Tensor, sketched_scores: Tensor) -> dict:
    """Format results into standard output dictionary."""
    # Ensure tensors are on CPU for output
    exact_scores = exact_scores.detach().cpu()
    sketched_scores = sketched_scores.detach().cpu()

    # Use relative threshold based on score magnitude to handle different scales
    # A score is valid if it's > 1e-10 * max(|scores|) or > 1e-12 (absolute floor)
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

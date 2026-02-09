"""
Fisher Information Matrix utilities for theoretical validation experiments.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Tuple, TYPE_CHECKING
import math
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .gradient_cache import GradientCache


# =============================================================================
# Safe Projection Helper
# =============================================================================

def safe_project(data: Tensor, projector, d: int, ensemble_id: int = 0) -> Tensor:
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
# Gram Matrix Computation
# =============================================================================

def _compute_gram_matrix(
    grad_cache: GradientCache,
    device: str = "cuda",
    batch_size: int = 128,
    projector: Optional[Any] = None,
) -> Tensor:
    """
    Compute Gram matrix K = G @ G^T / n (or (PG) @ (PG)^T / n) via streaming.

    Args:
        grad_cache: GradientCache object (any offload mode)
        device: Device for computation (default: "cuda")
        batch_size: Batch size for block computation
        projector: Optional dattri projector. If provided, computes
                   K = (PG)(PG)^T / n instead of G G^T / n.

    Returns:
        K: Gram matrix (n, n) on the specified device, normalized by n
    """
    n = grad_cache.n_samples
    effective_batch_size = batch_size if batch_size else 100
    n_batches = math.ceil(n / effective_batch_size)

    # Initialize output matrix
    K = torch.zeros(n, n, device=device)

    # Determine mode
    use_disk_mode = grad_cache.offload == "disk"
    use_cuda = device == "cuda" and torch.cuda.is_available()

    def get_batch_range(batch_idx: int) -> Tuple[int, int]:
        """Get (start, end) sample indices for a batch."""
        start = batch_idx * effective_batch_size
        end = min(start + effective_batch_size, n)
        return start, end

    # Setup CUDA stream for overlapped transfers
    transfer_stream = torch.cuda.Stream() if use_cuda else None
    compute_stream = torch.cuda.current_stream() if use_cuda else None

    # Helper to apply projection (if projector provided)
    def maybe_project(data_gpu):
        if projector is not None:
            d = grad_cache.dim
            return safe_project(data_gpu, projector, d, ensemble_id=0)
        return data_gpu

    if use_disk_mode: # Disk mode: Double-buffering with background disk prefetch
        from concurrent.futures import ThreadPoolExecutor

        disk_executor = ThreadPoolExecutor(max_workers=2)  # 2 workers for better overlap

        def load_to_cpu(batch_idx):
            """Load batch to CPU, using cache if available."""
            start, end = get_batch_range(batch_idx)
            return grad_cache.get_batch(start, end, device="cpu")

        for i_batch in range(n_batches):
            i_start, i_end = get_batch_range(i_batch)

            # Load G_i and project immediately to reduce memory
            # Clone to avoid race condition with LRU cache eviction
            G_i_cpu = load_to_cpu(i_batch).clone()
            G_i_gpu = maybe_project(G_i_cpu.to(device))

            # Double-buffering state
            next_future = None
            next_gpu = None

            for j_batch in range(i_batch, n_batches):
                j_start, j_end = get_batch_range(j_batch)

                # Get current G_j (projected if projector provided)
                if j_batch == i_batch:
                    G_j_gpu = G_i_gpu
                elif next_gpu is not None:
                    # Wait for prefetched GPU tensor (already projected)
                    if transfer_stream is not None:
                        compute_stream.wait_stream(transfer_stream)
                    G_j_gpu = next_gpu
                    next_gpu = None
                else:
                    # No prefetch available, load and project synchronously
                    # Clone to avoid race condition with LRU cache eviction
                    G_j_cpu = load_to_cpu(j_batch).clone()
                    G_j_gpu = maybe_project(G_j_cpu.to(device))

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

                    # Transfer to GPU then project
                    # NOTE: For disk mode, we use synchronous transfer to avoid race
                    # conditions with LRU cache eviction. The async stream approach
                    # can cause "illegal memory access" when cache evicts a tensor
                    # that's still being transferred to GPU.
                    # Clone to ensure we own the tensor (cache may evict original)
                    next_cpu = next_cpu.clone()
                    next_gpu = maybe_project(next_cpu.to(device))

                # Compute block K[i,j] = G_i @ G_j^T
                block = G_i_gpu @ G_j_gpu.T
                K[i_start:i_end, j_start:j_end] = block

                # Fill symmetric block K[j,i] = K[i,j]^T
                if i_batch != j_batch:
                    K[j_start:j_end, i_start:i_end] = block.T

        disk_executor.shutdown(wait=True)

    else:
        # =======================================================================
        # CPU/none mode: Direct memory access (original efficient path)
        # =======================================================================
        storage = grad_cache._memory_storage

        for i_batch in range(n_batches):
            i_start, i_end = get_batch_range(i_batch)
            G_i_cpu = storage[i_start:i_end]

            if transfer_stream is not None:
                with torch.cuda.stream(transfer_stream):
                    G_i_gpu = G_i_cpu.to(device, non_blocking=True)
                compute_stream.wait_stream(transfer_stream)
                G_i_gpu = maybe_project(G_i_gpu)
            else:
                G_i_gpu = maybe_project(G_i_cpu.to(device))

            next_j_gpu = None

            for j_batch in range(i_batch, n_batches):
                j_start, j_end = get_batch_range(j_batch)

                if j_batch == i_batch:
                    G_j_gpu = G_i_gpu
                elif next_j_gpu is not None:
                    if transfer_stream is not None:
                        compute_stream.wait_stream(transfer_stream)
                    # Project now that transfer is complete
                    G_j_gpu = maybe_project(next_j_gpu)
                    next_j_gpu = None
                else:
                    G_j_cpu = storage[j_start:j_end]
                    G_j_gpu = maybe_project(G_j_cpu.to(device))

                # Prefetch next j (transfer only - projection on default stream)
                if j_batch + 1 < n_batches and j_batch + 1 > i_batch and transfer_stream is not None:
                    next_j_start, next_j_end = get_batch_range(j_batch + 1)
                    next_j_cpu = storage[next_j_start:next_j_end]
                    with torch.cuda.stream(transfer_stream):
                        next_j_gpu = next_j_cpu.to(device, non_blocking=True)
                    # Projection will happen when we use it (after wait_stream)

                block = G_i_gpu @ G_j_gpu.T
                K[i_start:i_end, j_start:j_end] = block
                if i_batch != j_batch:
                    K[j_start:j_end, i_start:i_end] = block.T

    return K / n


def compute_eigenspectrum(
    grad_cache: GradientCache,
    lambda_values: list,
    device: str = "cuda",
    batch_size: int = 128,
    return_gram: bool = False,
) -> Union[Tuple[Tensor, dict], Tuple[Tensor, dict, Tensor]]:
    """
    Compute eigenvalues of F = G^T @ G / n and effective dimensions for given λ values.

    Uses Gram matrix K = G @ G^T / n (n×n) since K and F share
    the same nonzero eigenvalues. This is efficient because typically n << d.

    Effective dimension: d_λ(F) = tr(F(F + λI)^{-1}) = Σ_i λ_i / (λ_i + λ)

    Args:
        grad_cache: GradientCache containing the gradients
        lambda_values: List of regularization parameters λ to compute d_λ for
        device: Device for computation
        batch_size: Batch size for streaming
        return_gram: If True, also return the Gram matrix K

    Returns:
        eigenvalues: Eigenvalues of F in descending order
        effective_dims: Dictionary mapping λ -> d_λ
        K (optional): Gram matrix if return_gram=True
    """
    K = _compute_gram_matrix(grad_cache, device, batch_size)
    eigenvalues = torch.linalg.eigvalsh(K)

    eigenvalues = eigenvalues.flip(0)
    eigenvalues = torch.clamp(eigenvalues, min=0)

    # Compute effective dimension for each lambda
    effective_dims = {}
    for lamb in lambda_values:
        d_lambda = torch.sum(eigenvalues / (eigenvalues + lamb))
        effective_dims[lamb] = d_lambda.item()

    if return_gram:
        return eigenvalues, effective_dims, K
    else:
        return eigenvalues, effective_dims


# =============================================================================
# Shared Projection and Kernel Utilities
# =============================================================================

def project_gradients(
    grad_cache: GradientCache,
    projector,
    device: str = "cuda",
    batch_size: int = 64,
    test_vectors: Optional[Tensor] = None,
    storage_device: str = "cpu",
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Project all gradients and store on storage_device, optionally computing cross-products.

    This is the first step in Woodbury-based methods when m > n.

    Args:
        grad_cache: GradientCache containing the gradients
        projector: dattri projector with .project() method
        device: GPU device for projection
        batch_size: Batch size for streaming
        test_vectors: Optional projected test vectors (k, m) on device.
                      If provided, also computes U = PG @ test_vectors^T
        storage_device: Device to store projected gradients (default "cpu")

    Returns:
        PG: Projected gradients (n, m) on storage_device
        U (optional): Cross-product matrix (n, k) on device if test_vectors provided
    """
    n = grad_cache.n_samples
    m = projector.proj_dim
    d = grad_cache.dim
    dtype = grad_cache.dtype

    # Validate projection dimension
    if m >= d:
        raise ValueError(
            f"Invalid projection dimension: m={m} must be < d={d}. "
            f"Random projection is only meaningful when projecting to lower dimensions."
        )

    PG = torch.zeros(n, m, dtype=dtype, device=storage_device)

    # Optionally compute U = PG @ PV^T
    if test_vectors is not None:
        k = test_vectors.shape[0]
        U = torch.zeros(n, k, device=device, dtype=dtype)

    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        g_batch = grad_cache.get_batch(i_start, i_end, device=device)
        pg_batch = safe_project(g_batch, projector, d, ensemble_id=0)
        PG[i_start:i_end] = pg_batch.to(storage_device)

        if test_vectors is not None:
            U[i_start:i_end] = pg_batch @ test_vectors.T

        # Explicitly free GPU memory
        del g_batch, pg_batch
        # Periodically clear CUDA cache to avoid fragmentation
        if (i_start // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

    if test_vectors is not None:
        return PG, U
    return PG


def compute_kernel_from_projected(
    PG_cpu: Tensor,
    device: str = "cuda",
    batch_size: int = 128,
    normalize: bool = True,
) -> Tensor:
    """
    Compute kernel matrix K = PG @ PG^T from projected gradients.

    Args:
        PG_cpu: Projected gradients (n, m) on CPU
        device: GPU device for computation
        batch_size: Batch size for block computation
        normalize: If True, divide by n (default). If False, return raw kernel.

    Returns:
        K: Kernel matrix (n, n) on device
    """
    n = PG_cpu.shape[0]
    dtype = PG_cpu.dtype

    K = torch.zeros(n, n, device=device, dtype=dtype)

    kernel_batch_size = min(batch_size * 4, n)

    for i_start in range(0, n, kernel_batch_size):
        i_end = min(i_start + kernel_batch_size, n)
        pg_i = PG_cpu[i_start:i_end].to(device=device, dtype=dtype)

        for j_start in range(i_start, n, kernel_batch_size):
            j_end = min(j_start + kernel_batch_size, n)
            pg_j = PG_cpu[j_start:j_end].to(device=device, dtype=dtype)

            block = pg_i @ pg_j.T
            K[i_start:i_end, j_start:j_end] = block

            if i_start != j_start:
                K[j_start:j_end, i_start:i_end] = block.T

    if normalize:
        K = K / n

    return K


# =============================================================================
# Effective Dimension
# =============================================================================

def compute_effective_dimension(eigenvalues: Tensor, lamb: float) -> float:
    """Compute effective dimension d_λ = Σ_i σ_i / (σ_i + λ)."""
    return torch.sum(eigenvalues / (eigenvalues + lamb)).item()


# =============================================================================
# Unregularized Self-Influence (φ₀)
# =============================================================================

def compute_unregularized_self_influence(
    eigenvalues: Tensor,
    eigenvectors: Tensor,
    U_test: Tensor = None,
    n: int = None,
    eps: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """
    Compute unregularized self-influence φ₀(g) = g^T F^† g for training and test samples.

    For training gradients g_i (rows of G):
        φ₀(g_i) = n × Σ_{j: λ_j > ε} Q[i,j]²

    For test gradients v:
        w = G @ v (projection into Gram space)
        φ₀(v) = (1/n) × ||K^† w||² = (1/n) × Σ_{j: λ_j > ε} (Q^T w)_j² / λ_j²

    Args:
        eigenvalues: Eigenvalues of K in descending order (n,)
        eigenvectors: Eigenvectors of K, columns in descending order (n, n)
        U_test: G @ V^T for test gradients V, shape (n, k). If None, only train φ₀ is computed.
        n: Number of training samples (for normalization)
        eps: Threshold for non-zero eigenvalues (relative to max eigenvalue)

    Returns:
        train_phi0: φ₀ for training samples, shape (n,)
        test_phi0: φ₀ for test samples, shape (k,), or None if U_test is None
    """
    if n is None:
        n = eigenvalues.shape[0]

    # Determine non-zero eigenvalues (relative threshold)
    threshold = eps * eigenvalues[0].item() if eigenvalues[0] > 0 else eps
    nonzero_mask = eigenvalues > threshold
    n_nonzero = nonzero_mask.sum().item()

    print(f"  φ₀ computation: {n_nonzero}/{len(eigenvalues)} non-zero eigenvalues (threshold={threshold:.2e})")

    # Training samples: φ₀(g_i) = n × Σ_{j: λ_j > ε} Q[i,j]²
    # This uses the fact that g_i is the i-th row of G, so u_i = G @ g_i = n * K[:, i]
    # and K^† K is the projection onto the range of K
    Q_nonzero = eigenvectors[:, nonzero_mask]  # (n, r) where r = n_nonzero
    train_phi0 = n * (Q_nonzero ** 2).sum(dim=1)  # (n,)

    # Test samples: φ₀(v_j) = (1/n) × ||K^† w_j||²
    # where w_j = G @ v_j = U_test[:, j]
    # K^† w = Q Λ^† Q^T w, so ||K^† w||² = Σ_j (Q^T w)_j² / λ_j² for non-zero λ
    if U_test is not None:
        # Project U_test onto eigenbasis: Z = Q^T @ U_test, shape (n, k)
        Z = eigenvectors.T @ U_test  # (n, k)

        # For non-zero eigenvalues, compute (z / λ)² and sum
        inv_lambda_sq = torch.zeros_like(eigenvalues)
        inv_lambda_sq[nonzero_mask] = 1.0 / (eigenvalues[nonzero_mask] ** 2)

        # ||K^† w_j||² = Σ_i (Z[i,j] / λ_i)² for non-zero λ
        test_phi0 = (1.0 / n) * (inv_lambda_sq.unsqueeze(1) * (Z ** 2)).sum(dim=0)  # (k,)
    else:
        test_phi0 = None

    return train_phi0, test_phi0

"""
Unified gradient storage with configurable offload strategies.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Literal
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import math

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


OffloadMode = Literal["none", "cpu", "disk"]


class GradientCache:
    """
    Unified gradient storage with configurable offload strategies.

    Supports three modes:
    - "none": Keep gradients on compute device (e.g., GPU)
    - "cpu": Offload gradients to CPU RAM
    - "disk": Offload gradients to disk files

    All modes provide the same interface: get_batch(), get_sample(), iterate().
    """

    def __init__(
        self,
        offload: OffloadMode = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
        dtype: torch.dtype = torch.float32,
        storage_batch_size: int = 32,
        prefetch_batches: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize gradient cache.

        Args:
            offload: Storage strategy - "none" (GPU), "cpu", or "disk"
            cache_dir: Directory for disk storage (required if offload="disk")
            dtype: Data type for stored gradients
            storage_batch_size: Number of gradients per file (disk mode only)
            prefetch_batches: Number of batches to prefetch ahead (disk mode, default 2)
            pin_memory: Pin CPU tensors for faster GPU transfers (default True)
        """
        self.offload = offload
        self.dtype = dtype
        self.storage_batch_size = storage_batch_size
        self.prefetch_batches = prefetch_batches
        self.pin_memory = pin_memory and torch.cuda.is_available()

        # Metadata
        self.n_samples = 0
        self.dim = 0
        self._storage_device: Optional[str] = None  # Device where gradients are stored

        # Storage backends
        self._memory_storage: Optional[Tensor] = None  # For "none" and "cpu" modes
        self._disk_cache_dir: Optional[Path] = None  # For "disk" mode
        self._disk_batch_cache: dict = {}  # LRU cache for disk mode

        # Prefetching infrastructure (disk mode)
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_cache: dict = {}  # {batch_idx: Future or Tensor}
        self._prefetch_lock = Lock()

        if offload == "disk":
            if cache_dir is None:
                raise ValueError("cache_dir is required when offload='disk'")
            self._disk_cache_dir = Path(cache_dir)
            # Initialize prefetch executor
            self._prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_batches)

    @property
    def _metadata_file(self) -> Optional[Path]:
        """Path to metadata file (disk mode only)."""
        if self._disk_cache_dir is None:
            return None
        return self._disk_cache_dir / "metadata.pt"

    def _ensure_dir(self):
        """Create cache directory if needed (disk mode only)."""
        if self._disk_cache_dir is not None:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def clear(self):
        """Clear all stored gradients."""
        # Clear memory storage
        self._memory_storage = None

        # Clear disk storage
        if self._disk_cache_dir is not None and self._disk_cache_dir.exists():
            for f in self._disk_cache_dir.glob("batch_*.pt"):
                f.unlink()
            if self._metadata_file is not None and self._metadata_file.exists():
                self._metadata_file.unlink()

        self._disk_batch_cache.clear()
        self._clear_prefetch_cache()
        self.n_samples = 0
        self.dim = 0
        self._storage_device = None

    def _clear_prefetch_cache(self):
        """Clear the prefetch cache."""
        with self._prefetch_lock:
            self._prefetch_cache.clear()

    def shutdown(self):
        """Shutdown the prefetch executor. Call when done with the cache."""
        self._clear_prefetch_cache()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False)
            self._prefetch_executor = None

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()

    # =========================================================================
    # Disk Storage Helpers
    # =========================================================================

    def _get_batch_file(self, batch_idx: int) -> Path:
        """Get path to batch file (disk mode)."""
        assert self._disk_cache_dir is not None
        return self._disk_cache_dir / f"batch_{batch_idx}.pt"

    def _sample_to_batch_idx(self, sample_idx: int) -> tuple:
        """Convert sample index to (batch_idx, offset_in_batch)."""
        batch_idx = sample_idx // self.storage_batch_size
        offset = sample_idx % self.storage_batch_size
        return batch_idx, offset

    def _save_batch_to_disk(self, batch_idx: int, gradients: Tensor):
        """Save a batch of gradients to disk."""
        self._ensure_dir()
        torch.save(gradients.cpu().to(self.dtype), self._get_batch_file(batch_idx))

    def _load_batch_from_disk(self, batch_idx: int, device: str) -> Tensor:
        """
        Load a batch file from disk with CPU-side LRU caching.

        IMPORTANT: Always caches on CPU to avoid GPU memory bloat.
        Transfers to target device on demand.

        This is faster because:
        1. torch.load(..., map_location="cpu") is faster than direct GPU load
        2. CPU cache doesn't consume GPU memory
        3. CPU->GPU transfer is fast with pinned memory
        """
        # Check CPU cache first (cache is always on CPU)
        if batch_idx in self._disk_batch_cache:
            data_cpu = self._disk_batch_cache[batch_idx]
            if device == "cpu":
                return data_cpu
            else:
                return data_cpu.to(device, non_blocking=self.pin_memory)

        batch_file = self._get_batch_file(batch_idx)
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file {batch_file} not found")

        # Always load to CPU first (faster than direct GPU load)
        data_cpu = torch.load(batch_file, map_location="cpu", weights_only=True)

        # Pin memory for faster GPU transfers
        if self.pin_memory and torch.cuda.is_available():
            data_cpu = data_cpu.pin_memory()

        # LRU eviction - STRICT memory limit to prevent OOM
        # For large models (e.g., MusicTransformer with 13M params), each batch of 128
        # samples can be ~6.6GB. We limit cache to ~4GB to stay safe.
        MAX_CACHE_RAM_GB = 4.0
        bytes_per_batch = data_cpu.element_size() * data_cpu.numel()
        max_batches_by_memory = max(1, int((MAX_CACHE_RAM_GB * 1e9) / bytes_per_batch))
        # Hard cap at 4 batches regardless of size (for very small batches)
        max_cache_size = min(4, max_batches_by_memory)

        while len(self._disk_batch_cache) >= max_cache_size:
            oldest_key = next(iter(self._disk_batch_cache))
            del self._disk_batch_cache[oldest_key]

        self._disk_batch_cache[batch_idx] = data_cpu

        if device == "cpu":
            return data_cpu
        else:
            return data_cpu.to(device, non_blocking=self.pin_memory)

    def _load_batch_to_cpu_pinned(self, batch_idx: int) -> Tensor:
        """
        Load a batch file to CPU with optional memory pinning.

        Pinned memory enables faster CPU->GPU transfers via non-blocking copies.
        """
        batch_file = self._get_batch_file(batch_idx)
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file {batch_file} not found")

        data = torch.load(batch_file, map_location="cpu", weights_only=True)

        if self.pin_memory:
            data = data.pin_memory()

        return data

    # =========================================================================
    # Prefetching Methods (Disk Mode)
    # =========================================================================

    def _prefetch_batch(self, batch_idx: int):
        """Submit a batch for background loading."""
        if self._prefetch_executor is None or self.offload != "disk":
            return

        with self._prefetch_lock:
            if batch_idx in self._prefetch_cache:
                return  # Already prefetching or loaded

            # Submit background load
            future = self._prefetch_executor.submit(self._load_batch_to_cpu_pinned, batch_idx)
            self._prefetch_cache[batch_idx] = future

    def _get_prefetched_batch(self, batch_idx: int, device: str) -> Tensor:
        """
        Get a batch, using prefetched data if available.

        If the batch was prefetched, retrieves it from the prefetch cache.
        Otherwise, loads it synchronously.
        """
        with self._prefetch_lock:
            cached = self._prefetch_cache.pop(batch_idx, None)

        if cached is not None:
            # Get result from future (blocks if not ready yet)
            from concurrent.futures import Future
            if isinstance(cached, Future):
                data = cached.result()
            else:
                data = cached

            # Transfer to target device (non-blocking if pinned)
            if device != "cpu":
                data = data.to(device, non_blocking=self.pin_memory)
            return data

        # Fallback to regular loading
        return self._load_batch_from_disk(batch_idx, device)

    def _prefetch_upcoming_batches(self, current_batch_idx: int, num_batches: int):
        """Prefetch the next few batches in the background."""
        n_batches = self.get_num_batch_files()
        for i in range(1, self.prefetch_batches + 1):
            next_idx = current_batch_idx + i
            if next_idx < n_batches:
                self._prefetch_batch(next_idx)

    def load_batch_to_cpu(self, batch_idx: int) -> Tensor:
        """
        Load a batch file directly to CPU (bypasses LRU cache).

        Used by algorithms that manage their own caching strategy.
        """
        batch_file = self._get_batch_file(batch_idx)
        if not batch_file.exists():
            raise FileNotFoundError(f"Batch file {batch_file} not found")
        return torch.load(batch_file, map_location="cpu", weights_only=True)

    def _save_disk_metadata(self):
        """Save metadata to disk."""
        if self._metadata_file is not None:
            self._ensure_dir()
            torch.save({
                "n_samples": self.n_samples,
                "dim": self.dim,
                "dtype": str(self.dtype),
                "storage_batch_size": self.storage_batch_size,
                "offload": self.offload,
            }, self._metadata_file)

    def load_metadata(self) -> bool:
        """Load metadata from disk. Returns True if successful."""
        if self._metadata_file is None or not self._metadata_file.exists():
            return False

        metadata = torch.load(self._metadata_file, weights_only=True)
        self.n_samples = metadata["n_samples"]
        self.dim = metadata["dim"]
        self.storage_batch_size = metadata.get("storage_batch_size", 100)
        # Restore dtype from metadata (backwards compatible with caches that don't have it)
        dtype_str = metadata.get("dtype")
        if dtype_str is not None:
            # Convert string like "torch.float32" to actual dtype
            self.dtype = getattr(torch, dtype_str.replace("torch.", ""), torch.float32)
        return True

    def is_valid(self, expected_samples: Optional[int] = None) -> bool:
        """
        Check if cache is valid and complete.

        Note: This method loads metadata as a side effect when returning True.
        Callers should NOT call load_metadata() again after this returns True.
        """
        if self.offload != "disk":
            return self._memory_storage is not None and self.n_samples > 0

        if self._metadata_file is None or not self._metadata_file.exists():
            return False

        if not self.load_metadata():
            return False

        if expected_samples is not None and self.n_samples != expected_samples:
            return False

        # Verify all batch files exist
        n_batches = math.ceil(self.n_samples / self.storage_batch_size)
        for i in range(n_batches):
            if not self._get_batch_file(i).exists():
                return False

        return True

    # =========================================================================
    # Gradient Storage
    # =========================================================================

    def cache(
        self,
        model: nn.Module,
        dataset,
        indices: list,
        device: str = "cuda",
        model_type: str = "default",
        batch_size: int = 32,
    ):
        """
        Compute and store gradients using vmap.

        Args:
            model: Neural network model
            dataset: Dataset to sample from
            indices: Indices of samples to compute gradients for
            device: Device for computation
            model_type: Type of model ("default", "musictransformer")
            batch_size: Batch size for vmap gradient computation
        """
        from torch.func import grad, vmap, functional_call

        self.clear()

        mode_desc = {
            "none": f"GPU ({device})",
            "cpu": "CPU RAM",
            "disk": f"disk ({self._disk_cache_dir})",
        }
        print(f"Setting up {len(indices)} gradients with {mode_desc[self.offload]}...")

        model = model.to(device)
        model.eval()

        # Get named parameters
        params = {k: p for k, p in model.named_parameters() if p.requires_grad}
        d_params = sum(p.numel() for p in params.values())
        print(f"Model has {d_params:,} parameters")

        # Define loss function
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

        grad_func = vmap(grad(loss_func), in_dims=(None, 0), randomness="different")

        # Determine storage device
        if self.offload == "none":
            self._storage_device = device
        elif self.offload == "cpu":
            self._storage_device = "cpu"
        else:  # disk
            self._storage_device = "cpu"  # Disk reads go through CPU

        # Collect gradients for storage modes
        if self.offload == "disk":
            self._store_to_disk(
                grad_func, params, dataset, indices, device, model_type, batch_size
            )
        else:
            self._store_to_memory(
                grad_func, params, dataset, indices, device, model_type, batch_size
            )

        print(f"Stored {self.n_samples} gradients (dim={self.dim:,})")

    def _store_to_memory(
        self, grad_func, params, dataset, indices, device, model_type, batch_size
    ):
        """Store gradients in memory (GPU or CPU)."""
        n_samples = len(indices)
        storage_device = "cpu" if self.offload == "cpu" else device

        # First pass: compute one gradient to determine dimension
        first_batch_indices = indices[:min(batch_size, n_samples)]
        if model_type == "musictransformer":
            inputs = torch.stack([dataset[idx][0] for idx in first_batch_indices]).to(device)
            targets = torch.stack([dataset[idx][1] for idx in first_batch_indices]).to(device)
            first_batch_data = (inputs, targets)
        else:
            images = torch.stack([dataset[idx][0] for idx in first_batch_indices]).to(device)
            labels = torch.tensor([dataset[idx][1] for idx in first_batch_indices]).to(device)
            first_batch_data = (images, labels)

        with torch.no_grad():
            first_grad_dict = grad_func(params, first_batch_data)

        # Compute dimension from first gradient
        first_grad = torch.cat([first_grad_dict[k][0].view(-1) for k in params.keys()])
        self.dim = first_grad.shape[0]

        # Pre-allocate storage tensor (avoids memory doubling from torch.stack)
        self._memory_storage = torch.empty(
            n_samples, self.dim, dtype=self.dtype, device=storage_device
        )

        # Store first batch gradients
        for i in range(len(first_batch_indices)):
            grad_flat = torch.cat([first_grad_dict[k][i].view(-1) for k in params.keys()])
            if self.offload == "cpu":
                self._memory_storage[i] = grad_flat.cpu().to(self.dtype)
            else:
                self._memory_storage[i] = grad_flat.to(self.dtype)

        # Process remaining batches
        sample_idx = len(first_batch_indices)
        for batch_start in tqdm(range(batch_size, n_samples, batch_size), desc="Computing gradients"):
            batch_end = min(batch_start + batch_size, n_samples)
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

            # Compute gradients
            with torch.no_grad():
                grad_dict = grad_func(params, batch_data)

            # Flatten and store directly into pre-allocated tensor
            for i in range(len(batch_indices)):
                grad_flat = torch.cat([
                    grad_dict[k][i].view(-1) for k in params.keys()
                ])
                if self.offload == "cpu":
                    self._memory_storage[sample_idx] = grad_flat.cpu().to(self.dtype)
                else:
                    self._memory_storage[sample_idx] = grad_flat.to(self.dtype)
                sample_idx += 1

        self.n_samples = n_samples

    def _store_to_disk(
        self, grad_func, params, dataset, indices, device, model_type, batch_size
    ):
        """Store gradients to disk in batched files."""
        print(f"Storage batch size: {self.storage_batch_size} gradients per file")

        current_batch = []
        current_batch_idx = 0

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

            # Compute gradients
            with torch.no_grad():
                grad_dict = grad_func(params, batch_data)

            # Flatten and collect
            for i in range(len(batch_indices)):
                grad_flat = torch.cat([
                    grad_dict[k][i].view(-1) for k in params.keys()
                ])
                current_batch.append(grad_flat.cpu())

                if self.dim == 0:
                    self.dim = grad_flat.shape[0]

                # Save batch when full
                if len(current_batch) >= self.storage_batch_size:
                    self._save_batch_to_disk(current_batch_idx, torch.stack(current_batch))
                    current_batch = []
                    current_batch_idx += 1

        # Save remaining gradients
        if current_batch:
            self._save_batch_to_disk(current_batch_idx, torch.stack(current_batch))

        self.n_samples = len(indices)
        self._save_disk_metadata()

        n_batches = math.ceil(len(indices) / self.storage_batch_size)
        print(f"Cached in {n_batches} batch files")

    # =========================================================================
    # Unified Access Interface
    # =========================================================================

    def get_sample(self, idx: int, device: str) -> Tensor:
        """
        Get a single gradient.

        Args:
            idx: Index of the gradient
            device: Target device

        Returns:
            Gradient tensor of shape (dim,)
        """
        if self.offload == "disk":
            batch_idx, offset = self._sample_to_batch_idx(idx)
            batch_data = self._load_batch_from_disk(batch_idx, device)
            return batch_data[offset]
        else:
            assert self._memory_storage is not None
            return self._memory_storage[idx].to(device)

    def get_batch(self, start: int, end: int, device: str) -> Tensor:
        """
        Get a range of gradients.

        Args:
            start: Starting index (inclusive)
            end: Ending index (exclusive)
            device: Target device

        Returns:
            Batched gradient tensor of shape (end - start, dim)
        """
        if self.offload == "disk":
            return self._get_batch_disk(start, end, device)
        else:
            assert self._memory_storage is not None
            return self._memory_storage[start:end].to(device)

    def _get_batch_disk(self, start: int, end: int, device: str) -> Tensor:
        """Get batch from disk storage."""
        start_batch, start_offset = self._sample_to_batch_idx(start)
        end_batch, end_offset = self._sample_to_batch_idx(end - 1)

        if start_batch == end_batch:
            batch_data = self._load_batch_from_disk(start_batch, device)
            return batch_data[start_offset:end_offset + 1]
        else:
            result = []
            for batch_idx in range(start_batch, end_batch + 1):
                batch_data = self._load_batch_from_disk(batch_idx, device)
                if batch_idx == start_batch:
                    result.append(batch_data[start_offset:])
                elif batch_idx == end_batch:
                    result.append(batch_data[:end_offset + 1])
                else:
                    result.append(batch_data)
            return torch.cat(result, dim=0)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_num_batch_files(self) -> int:
        """Get number of storage batch files (disk mode only)."""
        return math.ceil(self.n_samples / self.storage_batch_size)

    def get_batch_file_range(self, batch_idx: int) -> tuple:
        """Get (start_idx, end_idx) for samples in a batch file."""
        start = batch_idx * self.storage_batch_size
        end = min(start + self.storage_batch_size, self.n_samples)
        return start, end

    def clear_cache(self):
        """Clear the in-memory LRU cache (disk mode only)."""
        self._disk_batch_cache.clear()

    def to_list(self, device: str = "cpu") -> list:
        """
        Convert all gradients to a list of tensors.

        Warning: This loads all gradients into memory. Use only when necessary
        (e.g., for algorithms that require list-based random access).

        Args:
            device: Target device for the tensors (default: "cpu")

        Returns:
            List of gradient tensors, each of shape (dim,)
        """
        return [self.get_sample(i, device=device) for i in range(self.n_samples)]

    def __len__(self) -> int:
        """Return number of stored gradients."""
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"GradientCache(offload='{self.offload}', n_samples={self.n_samples}, "
            f"dim={self.dim}, device='{self._storage_device}')"
        )


def create_gradient_cache(
    model: nn.Module,
    dataset,
    indices: list,
    device: str = "cuda",
    offload: OffloadMode = "cpu",
    model_type: str = "default",
    batch_size: int = 32,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False,
    prefetch_batches: int = 4,
    pin_memory: bool = True,
) -> GradientCache:
    """
    Convenience function to create and populate a gradient cache.

    For disk mode, will load from cache if valid, otherwise recompute.

    Args:
        model: Neural network model
        dataset: Dataset to sample from
        indices: Indices of samples
        device: Device for computation
        offload: Storage strategy - "none" (GPU), "cpu", or "disk"
        model_type: Type of model
        batch_size: Batch size for gradient computation
        cache_dir: Directory for disk storage (required if offload="disk")
        force_recompute: If True, always recompute gradients
        prefetch_batches: Number of batches to prefetch ahead (disk mode, default 2)
        pin_memory: Pin CPU tensors for faster GPU transfers (default True)

    Returns:
        Populated GradientCache object
    """
    grad_cache = GradientCache(
        offload=offload,
        cache_dir=cache_dir,
        storage_batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        pin_memory=pin_memory,
    )

    # For disk mode, try to load existing cache
    # Note: is_valid() already loads metadata when it returns True
    if offload == "disk" and not force_recompute:
        if grad_cache.is_valid(expected_samples=len(indices)):
            print(f"Loading cached gradients from {cache_dir}")
            grad_cache._storage_device = "cpu"
            return grad_cache

    # Compute and store gradients
    grad_cache.cache(
        model=model,
        dataset=dataset,
        indices=indices,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
    )

    return grad_cache

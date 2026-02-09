"""
Factorized Effective Dimension: K-FAC vs Unstructured Comparison

Empirically compares the effective dimension governing unstructured sketches
(Theorem 2) with the factorized effective dimension from Kronecker-structured
sketches (Theorem 6), addressing the trade-off described in Remark 7.

For each layer with K-FAC approximation F_l ≈ A_l ⊗ E_l, computes:
  - d_λ(A_l ⊗ E_l): unstructured effective dimension
  - d_{λ_E}(A_l) × d_{λ_A}(E_l): factorized effective dimension
    where λ_E = λ/‖E_l‖₂ and λ_A = λ/‖A_l‖₂

Optionally also computes d_λ(F_full) for the full empirical Fisher for
a three-way comparison.
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch import Tensor
from tqdm import tqdm

from dattri.benchmark.load import load_benchmark
from utils.gradient_cache import GradientCache
from utils.fisher import compute_eigenspectrum, compute_effective_dimension


# =============================================================================
# Effective Dimension Helpers
# =============================================================================

def compute_kronecker_effective_dim(
    eig_A: Tensor, eig_E: Tensor, lamb: float,
) -> float:
    """
    Compute d_λ(A ⊗ E) from eigenvalues of A and E.

    The eigenvalues of A ⊗ E are {α_i * γ_j}, so:
    d_λ(A ⊗ E) = Σ_{i,j} (α_i γ_j) / (α_i γ_j + λ)
    """
    products = eig_A.unsqueeze(1) * eig_E.unsqueeze(0)  # (d_A, d_E)
    return torch.sum(products / (products + lamb)).item()


# =============================================================================
# K-FAC Factor Extraction
# =============================================================================

def extract_kfac_factors(
    model: nn.Module,
    dataset,
    indices: List[int],
    device: str = "cuda",
    batch_size: int = 32,
) -> Dict[str, dict]:
    """
    Extract K-FAC factors (A, E) for each Linear/Conv2d layer.

    Runs forward/backward on training samples with hooks to capture
    activations and backpropagated gradients, then accumulates the
    K-FAC covariance matrices A = (1/n) Σ a_i a_i^T, E = (1/n) Σ e_i e_i^T.

    Args:
        model: Neural network model (on device, eval mode)
        dataset: Training dataset (indexable, returns (input, target))
        indices: Training sample indices
        device: GPU device
        batch_size: Mini-batch size for forward/backward

    Returns:
        Dictionary mapping layer_name -> {
            'A': Activation covariance (d_A, d_A) float64
            'E': Backprop gradient covariance (d_E, d_E) float64
            'd_A', 'd_E', 'layer_type', 'param_count'
        }
    """
    model.eval()
    n = len(indices)

    # Identify target layers and register hooks
    target_layers = OrderedDict()
    hooks = []
    activations = {}
    output_grads = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            target_layers[name] = module

            def make_fwd_hook(layer_name):
                def hook(mod, inp, out):
                    activations[layer_name] = inp[0].detach()
                return hook

            def make_bwd_hook(layer_name):
                def hook(mod, grad_input, grad_output):
                    output_grads[layer_name] = grad_output[0].detach()
                return hook

            hooks.append(module.register_forward_hook(make_fwd_hook(name)))
            hooks.append(module.register_full_backward_hook(make_bwd_hook(name)))

    print(f"  Found {len(target_layers)} layers: {list(target_layers.keys())}")

    # Accumulators
    A_accum = {}
    E_accum = {}
    layer_info = {}
    spatial_counts = {}  # Track total spatial samples for conv layers

    # Process training data in batches
    print(f"  Processing {n} samples in batches of {batch_size}...")

    for batch_start in tqdm(range(0, n, batch_size), desc="  K-FAC extraction"):
        batch_end = min(batch_start + batch_size, n)
        batch_indices = indices[batch_start:batch_end]
        actual_batch_size = len(batch_indices)

        # Collect batch data
        batch_data = [dataset[i] for i in batch_indices]
        inputs = torch.stack([d[0] for d in batch_data]).to(device)
        targets = torch.tensor(
            [d[1] if isinstance(d[1], int) else d[1].item() for d in batch_data],
            dtype=torch.long, device=device,
        )

        # Forward + backward (sum reduction → per-sample backprop gradients)
        model.zero_grad()
        outputs = model(inputs)
        loss = F_func.cross_entropy(outputs, targets, reduction='sum')
        loss.backward()

        # Extract factors from hooks and accumulate
        for lname, module in target_layers.items():
            act = activations[lname]
            grad = output_grads[lname]

            if isinstance(module, nn.Linear):
                a = act.to(dtype=torch.float64)  # (batch, d_in)
                e = grad.to(dtype=torch.float64)  # (batch, d_out)

                if module.bias is not None:
                    ones = torch.ones(
                        actual_batch_size, 1,
                        device=device, dtype=torch.float64,
                    )
                    a = torch.cat([a, ones], dim=1)

                d_A = a.shape[1]
                d_E = e.shape[1]
                count = actual_batch_size

            elif isinstance(module, nn.Conv2d):
                # Unfold activations to patches for K-FAC
                unfold = nn.Unfold(
                    kernel_size=module.kernel_size,
                    padding=module.padding,
                    stride=module.stride,
                    dilation=module.dilation,
                )
                patches = unfold(act)  # (batch, C_in*K_h*K_w, L)
                L = patches.shape[2]

                # (batch*L, d_A)
                a = patches.permute(0, 2, 1).reshape(
                    -1, patches.shape[1],
                ).to(dtype=torch.float64)

                if module.bias is not None:
                    ones = torch.ones(
                        a.shape[0], 1, device=device, dtype=torch.float64,
                    )
                    a = torch.cat([a, ones], dim=1)

                d_A = a.shape[1]

                # (batch*L, C_out)
                e = grad.reshape(
                    actual_batch_size, grad.shape[1], -1,
                ).permute(0, 2, 1).reshape(
                    -1, grad.shape[1],
                ).to(dtype=torch.float64)

                d_E = e.shape[1]
                count = actual_batch_size * L

            # Initialize accumulators on first batch
            if lname not in A_accum:
                A_accum[lname] = torch.zeros(
                    d_A, d_A, device=device, dtype=torch.float64,
                )
                E_accum[lname] = torch.zeros(
                    d_E, d_E, device=device, dtype=torch.float64,
                )
                spatial_counts[lname] = 0

                p_count = module.weight.numel()
                if module.bias is not None:
                    p_count += module.bias.numel()
                layer_info[lname] = {
                    'd_A': d_A,
                    'd_E': d_E,
                    'layer_type': type(module).__name__,
                    'param_count': p_count,
                }

            A_accum[lname] += a.T @ a
            E_accum[lname] += e.T @ e
            spatial_counts[lname] += count

    # Normalize and build results
    results = {}
    for lname in target_layers:
        total = spatial_counts[lname]
        A = A_accum[lname] / total
        E = E_accum[lname] / total

        results[lname] = {
            'A': A,
            'E': E,
            **layer_info[lname],
        }

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    return results


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    model: nn.Module,
    dataset,
    indices: List[int],
    lambda_values: List[float],
    device: str = "cuda",
    batch_size: int = 32,
    grad_cache: Optional[GradientCache] = None,
) -> dict:
    """
    Run the factorized effective dimension comparison experiment.

    Args:
        model: Neural network model
        dataset: Training dataset
        indices: Training sample indices
        lambda_values: List of regularization parameters to sweep
        device: GPU device
        batch_size: Batch size for forward/backward
        grad_cache: Optional GradientCache for computing full Fisher d_λ(F)

    Returns:
        Dictionary with all results
    """
    n = len(indices)

    # Step 1: Extract K-FAC factors
    print(f"\nStep 1: Extracting K-FAC factors...")
    kfac_factors = extract_kfac_factors(
        model=model,
        dataset=dataset,
        indices=indices,
        device=device,
        batch_size=batch_size,
    )

    # Step 2: Eigendecompose A and E for each layer
    print(f"\nStep 2: Computing eigendecompositions...")
    layer_results = {}

    for name, factors in kfac_factors.items():
        A = factors['A']
        E = factors['E']
        d_A = factors['d_A']
        d_E = factors['d_E']

        eig_A = torch.linalg.eigvalsh(A).flip(0).clamp(min=0)
        eig_E = torch.linalg.eigvalsh(E).flip(0).clamp(min=0)

        rank_A = (eig_A > 1e-10 * eig_A[0]).sum().item() if eig_A[0] > 0 else 0
        rank_E = (eig_E > 1e-10 * eig_E[0]).sum().item() if eig_E[0] > 0 else 0

        print(f"  Layer '{name}' ({factors['layer_type']}):")
        print(f"    A: {d_A}x{d_A}, rank={rank_A}, ||A||_2={eig_A[0]:.4e}")
        print(f"    E: {d_E}x{d_E}, rank={rank_E}, ||E||_2={eig_E[0]:.4e}")
        print(f"    d = d_A x d_E = {d_A * d_E:,}")

        norm_A = eig_A[0].item()
        norm_E = eig_E[0].item()

        # Compute effective dimensions for all lambda
        dim_results = {}
        for lamb in lambda_values:
            # Unstructured: d_λ(A ⊗ E)
            d_lambda_F = compute_kronecker_effective_dim(eig_A, eig_E, lamb)

            # Factorized: d_{λ_E}(A) × d_{λ_A}(E)
            lambda_E = lamb / norm_E if norm_E > 0 else float('inf')
            lambda_A = lamb / norm_A if norm_A > 0 else float('inf')

            d_lE_A = compute_effective_dimension(eig_A, lambda_E)
            d_lA_E = compute_effective_dimension(eig_E, lambda_A)
            factorized = d_lE_A * d_lA_E

            ratio = factorized / d_lambda_F if d_lambda_F > 0 else float('inf')

            dim_results[lamb] = {
                'd_lambda_unstructured': d_lambda_F,
                'd_lE_A': d_lE_A,
                'd_lA_E': d_lA_E,
                'd_lambda_factorized': factorized,
                'ratio': ratio,
                'lambda_E': lambda_E,
                'lambda_A': lambda_A,
            }

        layer_results[name] = {
            'eigenvalues_A': eig_A.cpu().numpy(),
            'eigenvalues_E': eig_E.cpu().numpy(),
            'norm_A': norm_A,
            'norm_E': norm_E,
            'rank_A': rank_A,
            'rank_E': rank_E,
            'd_A': d_A,
            'd_E': d_E,
            'layer_type': factors['layer_type'],
            'param_count': factors['param_count'],
            'effective_dims': dim_results,
        }

    # Step 3 (optional): Compute full Fisher effective dimension
    full_fisher_eff_dims = None
    full_fisher_eigenvalues = None
    if grad_cache is not None:
        print(f"\nStep 3: Computing full Fisher effective dimension...")
        eigenvalues_F, eff_dims_F = compute_eigenspectrum(
            grad_cache, lambda_values, device=device, batch_size=batch_size * 4,
        )
        full_fisher_eff_dims = eff_dims_F
        full_fisher_eigenvalues = eigenvalues_F.cpu().numpy()

        eig_nz = eigenvalues_F[eigenvalues_F > 1e-10 * eigenvalues_F[0]]
        print(f"  Full Fisher: rank={len(eig_nz)}, "
              f"||F||_2={eigenvalues_F[0]:.4e}")

    # Build results
    results = {
        'n_samples': n,
        'lambda_values': lambda_values,
        'layers': layer_results,
        'full_fisher_eff_dims': full_fisher_eff_dims,
        'full_fisher_eigenvalues': full_fisher_eigenvalues,
    }

    # Print summary
    _print_summary(layer_results, lambda_values, full_fisher_eff_dims)

    return results


def _print_summary(
    layer_results: dict,
    lambda_values: list,
    full_fisher_eff_dims: Optional[dict] = None,
):
    """Print a formatted summary table."""
    has_full = full_fisher_eff_dims is not None
    single_layer = len(layer_results) == 1

    print(f"\n{'='*80}")
    print(f"Summary: Factorized vs Unstructured Effective Dimension")
    print(f"{'='*80}")

    for name, lr in layer_results.items():
        print(f"\nLayer: {name} ({lr['layer_type']}, "
              f"d_A={lr['d_A']}, d_E={lr['d_E']})")

        # Show d_lam(F) per-layer only for single-layer models
        show_full_here = has_full and single_layer

        # Header
        hdr = (f"  {'lambda':>12s} | {'d_lam(A*E)':>12s} | "
               f"{'d_lE(A)*d_lA(E)':>16s} | {'Ratio':>8s}")
        sep = f"  {'-'*12}-+-{'-'*12}-+-{'-'*16}-+-{'-'*8}"
        if show_full_here:
            hdr += f" | {'d_lam(F)':>12s}"
            sep += f"-+-{'-'*12}"
        print(hdr)
        print(sep)

        for lamb in lambda_values:
            dr = lr['effective_dims'][lamb]
            line = (f"  {lamb:12.2e} | "
                    f"{dr['d_lambda_unstructured']:12.2f} | "
                    f"{dr['d_lambda_factorized']:16.2f} | "
                    f"{dr['ratio']:8.2f}")
            if show_full_here:
                line += f" | {full_fisher_eff_dims[lamb]:12.2f}"
            print(line)

    # Aggregate summary across layers (always shown for multi-layer,
    # and also for single-layer when full Fisher is available)
    if len(layer_results) > 1 or has_full:
        if len(layer_results) > 1:
            print(f"\nAggregate (sum over {len(layer_results)} layers):")
        else:
            print(f"\nComparison with full Fisher:")

        hdr = (f"  {'lambda':>12s} | {'Sum d_lam':>12s} | "
               f"{'Sum factorized':>16s} | {'Ratio':>8s}")
        sep = f"  {'-'*12}-+-{'-'*12}-+-{'-'*16}-+-{'-'*8}"
        if has_full:
            hdr += f" | {'d_lam(F)':>12s}"
            sep += f"-+-{'-'*12}"
        print(hdr)
        print(sep)

        for lamb in lambda_values:
            total_unstr = sum(
                lr['effective_dims'][lamb]['d_lambda_unstructured']
                for lr in layer_results.values()
            )
            total_fact = sum(
                lr['effective_dims'][lamb]['d_lambda_factorized']
                for lr in layer_results.values()
            )
            ratio = total_fact / total_unstr if total_unstr > 0 else float('inf')
            line = (f"  {lamb:12.2e} | "
                    f"{total_unstr:12.2f} | "
                    f"{total_fact:16.2f} | "
                    f"{ratio:8.2f}")
            if has_full:
                line += f" | {full_fisher_eff_dims[lamb]:12.2f}"
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Compare K-FAC factorized vs unstructured effective dimensions"
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar2", "maestro"])
    parser.add_argument("--model", type=str, default="lr",
                       choices=["lr", "mlp", "resnet9", "musictransformer"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for forward/backward passes")

    # Gradient offloading for full Fisher computation
    parser.add_argument("--offload", type=str, default="cpu",
                       choices=["none", "cpu", "disk"])
    parser.add_argument("--cache_dir", type=str, default="./grad_cache")

    # Whether to also compute full Fisher d_λ(F) for comparison
    parser.add_argument("--compute_full_fisher", action="store_true",
                       help="Also compute d_λ(F) for the full empirical Fisher")

    # Lambda sweep (integer powers of 10)
    parser.add_argument("--lambda_exp_min", type=int, default=-8,
                       help="Minimum exponent for λ sweep (default: -8 -> 1e-8)")
    parser.add_argument("--lambda_exp_max", type=int, default=2,
                       help="Maximum exponent for λ sweep (default: 2 -> 1e2)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"Factorized Effective Dimension Comparison")
    print(f"  Dataset: {args.dataset}, Model: {args.model}")
    print("="*60)

    # Load benchmark
    print(f"\nLoading {args.model} on {args.dataset}...")
    model_details, _ = load_benchmark(
        model=args.model, dataset=args.dataset, metric="lds",
    )

    nn_model = model_details["model"]
    checkpoint = torch.load(
        model_details["models_full"][0], map_location=args.device,
    )
    nn_model.load_state_dict(checkpoint)
    nn_model.to(args.device)
    nn_model.eval()
    print(f"Loaded checkpoint: {model_details['models_full'][0]}")

    train_dataset = model_details["train_dataset"]
    train_sampler = model_details["train_sampler"]
    indices = list(train_sampler)
    n_samples = len(indices)
    print(f"Training samples: {n_samples}")

    # Lambda values
    lambda_values = [
        10.0 ** exp
        for exp in range(args.lambda_exp_min, args.lambda_exp_max + 1)
    ]
    print(f"lambda values: {[f'1e{e}' for e in range(args.lambda_exp_min, args.lambda_exp_max + 1)]}")

    # Optionally compute full Fisher
    grad_cache = None
    if args.compute_full_fisher:
        print(f"\nComputing full Fisher gradient cache...")
        cache_dir = (
            f"{args.cache_dir}/train" if args.offload == "disk" else None
        )
        grad_cache = GradientCache(offload=args.offload, cache_dir=cache_dir)

        if not (args.offload == "disk"
                and grad_cache.is_valid(expected_samples=n_samples)):
            model_type = (
                "musictransformer"
                if args.model == "musictransformer"
                else "default"
            )
            grad_cache.cache(
                model=nn_model,
                dataset=train_dataset,
                indices=indices,
                device=args.device,
                model_type=model_type,
                batch_size=args.batch_size,
            )

    # Run experiment
    results = run_experiment(
        model=nn_model,
        dataset=train_dataset,
        indices=indices,
        lambda_values=lambda_values,
        device=args.device,
        batch_size=args.batch_size,
        grad_cache=grad_cache,
    )

    # Add metadata
    results["dataset"] = args.dataset
    results["model"] = args.model

    # Save results
    experiment_dir = os.path.join(args.output_dir, "factorized_edim")
    os.makedirs(experiment_dir, exist_ok=True)
    results_path = os.path.join(
        experiment_dir, f"{args.dataset}_{args.model}.pt",
    )
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

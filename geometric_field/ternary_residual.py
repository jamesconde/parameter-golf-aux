"""Experiment A: Ternary Residual — two-scale quantization of the same weights.

The same continuous weight matrix is quantized twice with different group sizes.
The primary quantization (fine groups) handles most of the computation.
The residual (coarse groups) captures what the primary missed.

y = ternary(W, g=128)·x + ε·(ternary(W, g=coarse)·x - ternary(W, g=128)·x)

The difference (y_coarse - y_fine) is largest where the fine-group quantization
was poorest — the residual patches exactly those dimensions.

Zero extra parameters. Zero extra bytes. ~15% compute overhead.

Usage:
    apply_ternary_residual(model, epsilon=0.1, coarse_group_size=512)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def ternary_quantize(w: Tensor, group_size: int) -> Tensor:
    """Quantize weight to ternary {-1, 0, +1} × scale with given group size."""
    original_shape = w.shape
    w_flat = w.reshape(-1, group_size)
    scale = w_flat.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    q = (w_flat / scale).round().clamp(-1, 1)
    w_ternary = w + ((q * scale).reshape(original_shape) - w).detach()
    return w_ternary


def patch_residual_forward(module: nn.Module, epsilon: float, coarse_group_size: int,
                            is_normed: bool = False):
    """Monkey-patch a TernaryLinear to use two-scale residual quantization."""
    _epsilon = epsilon
    _coarse_gs = coarse_group_size
    _fine_gs = getattr(module, 'group_size', 128)
    _is_normed = is_normed

    def residual_forward(x: Tensor) -> Tensor:
        if _is_normed:
            x = F.rms_norm(x, (x.size(-1),))

        w = module.weight.bfloat16()

        # Primary: fine-group ternary (standard)
        w_fine = ternary_quantize(w, _fine_gs)

        if _epsilon > 0:
            # Residual: coarse-group ternary (different scale regime)
            w_coarse = ternary_quantize(w, _coarse_gs)
            # Blend: primary + epsilon * (coarse - primary)
            w_effective = w_fine + _epsilon * (w_coarse - w_fine)
        else:
            w_effective = w_fine

        return F.linear(x, w_effective.to(x.dtype),
                        module.bias.to(x.dtype) if module.bias is not None else None)

    module.forward = residual_forward
    module._has_residual = True


def apply_ternary_residual(model: nn.Module, epsilon: float = 0.1,
                            coarse_group_size: int = 512,
                            mlp_only: bool = False):
    """Apply ternary residual to all TernaryLinear modules in the model.

    Args:
        model: the GPT model
        epsilon: residual weight (0 = disabled, 0.1 = gentle, 0.2 = moderate)
        coarse_group_size: group size for the residual quantization
        mlp_only: if True, only patch MLP layers (save compute on attention)
    """
    if epsilon <= 0:
        print("Ternary residual disabled (epsilon=0)")
        return

    patched = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Ternary" not in cls_name or not hasattr(module, "weight"):
            continue
        if hasattr(module, "_has_residual"):
            continue
        if mlp_only and "attn" in name:
            continue

        is_normed = "Normed" in cls_name
        patch_residual_forward(module, epsilon, coarse_group_size, is_normed)
        patched += 1

    print(f"Ternary residual applied: {patched} layers, epsilon={epsilon}, "
          f"coarse_group={coarse_group_size}")

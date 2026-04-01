"""Asymmetric Ternary: learned positive/negative scales per group.

Standard ternary: q ∈ {-1, 0, +1} × scale
  → effective values: {-scale, 0, +scale}
  → symmetric: positive and negative have same magnitude

Asymmetric ternary: q ∈ {-1, 0, +1} with separate pos/neg scales
  → effective values: {-scale_neg, 0, +scale_pos}
  → each group of 128 weights learns whether positive or negative
    values should be larger

This breaks an arbitrary symmetry constraint. Some groups may naturally
need asymmetric range (e.g., bias-like corrections, rectified features).

Cost: 1 extra float per group of 128 weights
  → ~512 groups per layer × 4 matrix types × 10 layers = ~20K floats
  → At FP8: ~20KB artifact cost (0.13% of 16MB budget)
  → Step time: ~0% overhead (same matmul, different scale in STE)

Usage:
    apply_asymmetric_ternary(model, init_asymmetry=0.0)
    # init_asymmetry=0.0 → starts symmetric (identical to baseline)
    # the optimizer learns per-group asymmetry during training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def patch_asymmetric_forward(module: nn.Module, is_normed: bool = False):
    """Monkey-patch TernaryLinear to use asymmetric positive/negative scales.

    Adds a learned `asymmetry` parameter per quantization group.
    asymmetry=0 → symmetric (baseline behavior)
    asymmetry>0 → positive ternary values are larger
    asymmetry<0 → negative ternary values are larger
    """
    group_size = getattr(module, 'group_size', 128)
    w = module.weight
    n_groups = (w.numel() + group_size - 1) // group_size

    # Learned asymmetry per group, initialized to 0 (symmetric = baseline)
    asymmetry = nn.Parameter(torch.zeros(n_groups, dtype=torch.float32))
    module.register_parameter('_asymmetry', asymmetry)

    _gs = group_size
    _is_normed = is_normed

    def asymmetric_forward(x: Tensor) -> Tensor:
        if _is_normed:
            x = F.rms_norm(x, (x.size(-1),))

        w = module.weight.bfloat16()
        g = _gs

        # Standard ternary quantization
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)

        # Standard STE: quantize to ternary (this part is detached)
        w_ternary_sym = (q * scale).reshape(w.shape)
        w_ste = w + (w_ternary_sym - w).detach()  # STE for weight gradients

        # Asymmetric correction: applied OUTSIDE detach so gradients flow to asymmetry
        asym = module._asymmetry[:w_g.shape[0]].clamp(-0.5, 0.5)
        asym = asym.to(dtype=w_g.dtype).unsqueeze(-1)  # (n_groups, 1)

        # Correction: positive ternary values scaled by (1+asym), negative by (1-asym)
        # When asym=0: correction=0 (baseline behavior)
        pos_mask = (q > 0).float()
        neg_mask = (q < 0).float()
        correction = scale * asym * (pos_mask - neg_mask)  # (n_groups, g)
        w_out = w_ste + correction.reshape(w.shape)  # gradient flows through correction

        return F.linear(x, w_out.to(x.dtype),
                        module.bias.to(x.dtype) if module.bias is not None else None)

    module.forward = asymmetric_forward
    module._has_asymmetric = True


def apply_asymmetric_ternary(model: nn.Module, mlp_only: bool = False):
    """Apply asymmetric ternary to all TernaryLinear modules.

    Args:
        model: the GPT model
        mlp_only: if True, only patch MLP layers
    """
    patched = 0
    total_asym_params = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Ternary" not in cls_name or not hasattr(module, "weight"):
            continue
        if hasattr(module, "_has_asymmetric"):
            continue
        if mlp_only and "attn" in name:
            continue

        is_normed = "Normed" in cls_name
        patch_asymmetric_forward(module, is_normed)

        n_groups = module._asymmetry.numel()
        total_asym_params += n_groups
        patched += 1

    print(f"Asymmetric ternary applied to {patched} layers, "
          f"{total_asym_params} asymmetry params "
          f"({total_asym_params * 4 / 1024:.1f} KB fp32, "
          f"{total_asym_params * 1 / 1024:.1f} KB fp8)")

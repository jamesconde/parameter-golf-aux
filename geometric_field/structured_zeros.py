"""Experiment D: Structured Zero Placement — data-driven zero thresholds.

The ternary STE rounds small weights to zero. The threshold is implicit:
weights with |w/scale| < 0.5 become zero. This experiment shifts the
threshold per-column based on input feature importance.

Important input dimensions (high activation variance) get a LOWER zero
threshold → fewer zeros → more ±1 weights → more information retained.
Unimportant dimensions get a HIGHER threshold → more zeros → those
weights contribute less → freed "capacity" goes to important dims.

Zero extra parameters (zero_bias is precomputed from C_diag, not learned).
Zero extra bytes. ~0% compute overhead.

Usage:
    apply_structured_zeros(model, signals_path="geometric_field/g_signals.pt",
                           bias_range=0.1)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_zero_bias(C_diag: Tensor, bias_range: float = 0.1) -> Tensor:
    """Compute per-column zero threshold bias from input covariance.

    Args:
        C_diag: (M,) input variance per dimension
        bias_range: max absolute bias (e.g., 0.1 means threshold shifts ±0.1)

    Returns:
        zero_bias: (M,) values in [-bias_range, +bias_range]
        Positive = more zeros (higher threshold), negative = fewer zeros
    """
    if C_diag is None or torch.isnan(C_diag).any():
        return torch.zeros_like(C_diag) if C_diag is not None else None

    # Relative importance: high-variance dims are important
    importance = C_diag / C_diag.mean().clamp(min=1e-8)

    # Map to [-bias_range, +bias_range]:
    # importance > 1 (important) → negative bias (fewer zeros)
    # importance < 1 (unimportant) → positive bias (more zeros)
    # Centered and clamped
    bias = -bias_range * (importance - 1.0).clamp(-1.0, 1.0)

    return bias


def patch_structured_zeros(module: nn.Module, zero_bias: Tensor,
                            is_normed: bool = False):
    """Monkey-patch a TernaryLinear to use structured zero thresholds."""
    _zero_bias = zero_bias.clone()
    _group_size = getattr(module, 'group_size', 128)
    _is_normed = is_normed

    def structured_forward(x: Tensor) -> Tensor:
        if _is_normed:
            x = F.rms_norm(x, (x.size(-1),))

        w = module.weight.bfloat16()
        g = _group_size
        N, M = w.shape

        # Expand zero_bias to match weight matrix columns
        bias = _zero_bias.to(device=w.device, dtype=w.dtype)
        if bias.shape[0] != M:
            # Dimension mismatch — skip structured zeros for this layer
            bias = torch.zeros(M, device=w.device, dtype=w.dtype)

        # Standard ternary quantization with modified threshold
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        normalized = w_g / scale

        # Per-column threshold: 0.5 + bias
        # Weight shape (N, M) flattened to (N*M/g, g)
        # Column index for element [row, col] in group [row*M+col]//g
        # Expand bias to full weight shape then reshape to groups
        bias_2d = bias.unsqueeze(0).expand(N, M)  # (N, M)
        bias_grouped = bias_2d.reshape(-1, g)  # (N*M/g, g)
        threshold = 0.5 + bias_grouped

        # Ternary with shifted threshold
        q = torch.where(normalized.abs() < threshold,
                        torch.zeros_like(normalized),
                        normalized.sign())

        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary.to(x.dtype),
                        module.bias.to(x.dtype) if module.bias is not None else None)

    module.forward = structured_forward
    module._has_structured_zeros = True


def apply_structured_zeros(model: nn.Module, signals_path: str = "",
                            bias_range: float = 0.1, C_diag: dict = None):
    """Apply structured zero placement to all ternary layers.

    Args:
        model: the GPT model
        signals_path: path to g_signals.pt (from compute_signals.py)
        bias_range: max zero threshold shift (0.05=gentle, 0.1=moderate, 0.2=aggressive)
        C_diag: precomputed covariance dict (overrides signals_path)
    """
    if bias_range <= 0:
        print("Structured zeros disabled (bias_range=0)")
        return

    # Load C_diag
    if C_diag is None and signals_path and os.path.exists(signals_path):
        signals = torch.load(signals_path, map_location="cpu", weights_only=True)
        C_diag = signals.get("C_diag", {})

    if not C_diag:
        print("WARNING: No C_diag signals available. Structured zeros disabled.")
        return

    patched = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Ternary" not in cls_name or not hasattr(module, "weight"):
            continue
        if hasattr(module, "_has_structured_zeros"):
            continue

        w = module.weight
        N, M = w.shape
        is_normed = "Normed" in cls_name

        # Find matching C_diag
        layer_C = None
        for ckey in C_diag:
            if name in ckey or ckey in name:
                c = C_diag[ckey]
                if c.shape[0] == M:
                    layer_C = c
                    break

        if layer_C is None:
            continue

        zero_bias = compute_zero_bias(layer_C, bias_range)
        patch_structured_zeros(module, zero_bias, is_normed)
        patched += 1

    print(f"Structured zeros applied: {patched} layers, bias_range={bias_range}")

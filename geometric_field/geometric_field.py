"""Geometric field G for ternary weight modulation.

G(i,j) modulates continuous weights before ternary STE quantization.
It's computed from a compact formula — NOT stored in the artifact.

Usage:
    # After model construction, apply G to all ternary layers:
    apply_geometric_field(model, signals_path="geometric_field/g_signals.pt",
                          alpha=0.3, beta=0.3)

    # Or with env vars:
    GEOM_ALPHA=0.3 GEOM_BETA=0.3 GEOM_SIGNALS=geometric_field/g_signals.pt
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_G_column(C_diag: Tensor, delta_e: Tensor,
                     alpha: float, beta: float) -> Tensor:
    """Compute column-wise G from data signals.

    Args:
        C_diag: (M,) input covariance diagonal for this layer
        delta_e: (M,) word-boundary direction (projected to this layer's input dim)
        alpha: weight for covariance signal (0 = off, 0.1-0.5 typical)
        beta: weight for word-boundary signal (0 = off, 0.1-0.5 typical)

    Returns:
        G_col: (M,) column-wise geometric field, mean=1.0
    """
    G_col = torch.ones_like(C_diag)

    # Guard against NaN inputs
    if torch.isnan(C_diag).any() or (delta_e is not None and torch.isnan(delta_e).any()):
        print("  WARNING: NaN in signals, returning identity G")
        return G_col

    if alpha > 0 and C_diag is not None:
        # Relative importance: high-variance dims get compressed
        covar_signal = (C_diag.clamp(min=1e-10) / C_diag.mean().clamp(min=1e-8)).sqrt()
        G_col = G_col / (1.0 + alpha * (covar_signal - 1.0)).clamp(min=0.1)

    if beta > 0 and delta_e is not None:
        # Word-boundary alignment: dims carrying boundary info get compressed
        boundary_signal = delta_e.abs()
        boundary_signal = boundary_signal / boundary_signal.mean().clamp(min=1e-8)
        G_col = G_col / (1.0 + beta * (boundary_signal - 1.0)).clamp(min=0.1)

    # Normalize to mean=1 (G redistributes, doesn't globally scale)
    G_col = G_col / G_col.mean().clamp(min=1e-8)

    return G_col


def patch_ternary_forward(module: nn.Module, G_col: Tensor, is_normed: bool = False):
    """Monkey-patch a TernaryLinear's forward to apply G before STE.

    This is the least invasive integration — no class replacement needed.
    The original forward is preserved and called with modulated weights.
    """
    _G_col = G_col.clone()
    _is_normed = is_normed
    _group_size = getattr(module, 'group_size', 128)

    def geometric_forward(x: Tensor) -> Tensor:
        w = module.weight.bfloat16()

        # Apply G column-wise before STE
        G = _G_col.to(device=w.device, dtype=w.dtype)
        w_mod = w * G.unsqueeze(0)  # (N, M) * (1, M)

        # Ternary STE quantization on modulated weights
        g = _group_size
        w_g = w_mod.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w_mod + ((q * scale).reshape(w_mod.shape) - w_mod).detach()

        # NormedTernaryLinear applies RMSNorm to input
        if _is_normed:
            x = F.rms_norm(x, (x.size(-1),))

        return F.linear(x, w_ternary.to(x.dtype),
                        module.bias.to(x.dtype) if module.bias is not None else None)

    module.forward = geometric_forward
    module._has_geometric_field = True


def apply_geometric_field(model: nn.Module, signals_path: str = "",
                          alpha: float = 0.0, beta: float = 0.0,
                          delta_e: Tensor = None, C_diag: dict = None):
    """Apply geometric field G to all ternary layers in the model.

    Args:
        model: the GPT model
        signals_path: path to g_signals.pt (from compute_signals.py)
        alpha: covariance signal weight
        beta: word-boundary signal weight
        delta_e: precomputed word-boundary direction (overrides signals_path)
        C_diag: precomputed covariance dict (overrides signals_path)
    """
    if alpha == 0 and beta == 0:
        print("Geometric field disabled (alpha=0, beta=0)")
        return

    # Load signals
    if signals_path and os.path.exists(signals_path) and (delta_e is None or C_diag is None):
        signals = torch.load(signals_path, map_location="cpu", weights_only=True)
        if delta_e is None:
            delta_e = signals.get("delta_e")
        if C_diag is None:
            C_diag = signals.get("C_diag", {})

    patched = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        is_ternary = "Ternary" in cls_name
        if not is_ternary or not hasattr(module, "weight"):
            continue
        if hasattr(module, "_has_geometric_field"):
            continue  # Already patched

        w = module.weight
        N, M = w.shape  # (out_features, in_features)

        is_normed = "Normed" in cls_name

        # Get C_diag for this layer
        layer_C = None
        if C_diag:
            # Try exact match first, then partial
            for ckey in C_diag:
                if name in ckey or ckey in name:
                    layer_C = C_diag[ckey]
                    break
            # If C_diag dim doesn't match input dim, skip or truncate
            if layer_C is not None and layer_C.shape[0] != M:
                layer_C = None

        # Get delta_e for this layer's input dimension
        layer_delta_e = None
        if delta_e is not None:
            if delta_e.shape[0] == M:
                layer_delta_e = delta_e
            elif delta_e.shape[0] > M:
                layer_delta_e = delta_e[:M]  # Truncate if needed
            # For layers with different input dim (e.g., mlp.proj has M=3072),
            # delta_e (768-dim) doesn't directly apply — skip boundary signal
            # unless dimensions match

        # Compute G
        if layer_C is None and layer_delta_e is None:
            continue  # No signals available for this layer

        # Use whichever signals are available
        effective_alpha = alpha if layer_C is not None else 0.0
        effective_beta = beta if layer_delta_e is not None else 0.0

        if effective_alpha == 0 and effective_beta == 0:
            continue

        # Create dummy signals if one is missing
        if layer_C is None:
            layer_C = torch.ones(M)
        if layer_delta_e is None:
            layer_delta_e = torch.zeros(M)

        G_col = compute_G_column(layer_C, layer_delta_e, effective_alpha, effective_beta)

        # Patch the forward method
        patch_ternary_forward(module, G_col, is_normed=is_normed)
        patched += 1

        g_range = f"[{G_col.min():.3f}, {G_col.max():.3f}]"
        print(f"  G applied to {name}: shape ({N},{M}), alpha={effective_alpha}, "
              f"beta={effective_beta}, G range={g_range}")

    print(f"Geometric field applied to {patched} layers")

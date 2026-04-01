"""Architecture Experiments for Ternary Parameter Golf.

Three activation/architecture modifications that don't touch the STE:

Experiment 1: Parametric Power — learned exponent per layer
Experiment 2: Stochastic Depth — random block skipping during training
Experiment 3: GaugeReLU — phase-magnitude decomposition

All are applied via monkey-patching after model construction.
Env vars: POWER_ACT=1, STOCH_DEPTH=0.2, GAUGE_RELU=1

Usage:
    apply_activation_experiments(model, args)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# Experiment 1: Per-Layer Parametric Power Activation
# ============================================================

def apply_parametric_power(model: nn.Module):
    """Replace relu² with relu^p where p is learned per layer.

    Initializes p=2.0 (identical to baseline). The optimizer learns
    the optimal power for each layer.
    """
    patched = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'activation') or not hasattr(module, 'proj'):
            continue
        if module.activation != 'relu2':
            continue

        # Add learnable power parameter
        power = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        module.register_parameter('_power', power)

        # Monkey-patch forward
        original_fc = module.fc if hasattr(module, 'fc') else None
        original_proj = module.proj

        def make_power_forward(fc, proj, power_param):
            def forward(x: Tensor) -> Tensor:
                h = fc(x)
                # leaky_relu preserves gradient for negative values
                h = F.leaky_relu(h, negative_slope=0.01)
                # Learned power exponent (clamped to [1.0, 4.0] to prevent gradient issues)
                p = power_param.clamp(1.0, 4.0).to(dtype=h.dtype)
                h = h.abs().clamp(min=1e-6).pow(p) * h.sign()
                return proj(h)
            return forward

        if original_fc is not None:
            module.forward = make_power_forward(original_fc, original_proj, power)
            patched += 1

    print(f"Parametric power applied to {patched} MLP layers (init p=2.0)")
    return patched


# ============================================================
# Experiment 2: Stochastic Depth
# ============================================================

def apply_stochastic_depth(model: nn.Module, max_drop_rate: float = 0.2,
                            skip_first: int = 2, skip_last: int = 2):
    """Apply stochastic depth to transformer blocks.

    Deeper blocks have higher drop probability (linear schedule).
    First and last N blocks are never dropped (they're critical for
    input processing and output prediction).

    At eval time, outputs are scaled by (1 - drop_rate) for each block.
    """
    blocks = None
    for name, module in model.named_modules():
        if hasattr(module, 'blocks') and isinstance(getattr(module, 'blocks'), nn.ModuleList):
            blocks = module.blocks
            break

    if blocks is None:
        print("Stochastic depth: could not find blocks ModuleList")
        return 0

    num_blocks = len(blocks)
    patched = 0

    for i, block in enumerate(blocks):
        # Skip first and last N blocks
        if i < skip_first or i >= num_blocks - skip_last:
            drop_rate = 0.0
        else:
            # Linear schedule: increases with depth
            progress = (i - skip_first) / max(num_blocks - skip_first - skip_last - 1, 1)
            drop_rate = max_drop_rate * progress

        if drop_rate <= 0:
            continue

        original_forward = block.forward
        _drop = drop_rate

        def make_stoch_forward(orig_fwd, drop_r):
            def forward(x: Tensor, x0: Tensor) -> Tensor:
                if block.training and torch.rand(1).item() < drop_r:
                    # Skip this block entirely — return input unchanged
                    return x
                out = orig_fwd(x, x0)
                if not block.training:
                    # At eval: scale output to compensate for training drops
                    # The residual contribution is scaled down
                    # out = x + scale * (attn + mlp), so we scale the delta
                    delta = out - x
                    out = x + (1.0 - drop_r) * delta
                return out
            return forward

        block.forward = make_stoch_forward(original_forward, drop_rate)
        patched += 1

    print(f"Stochastic depth applied to {patched}/{num_blocks} blocks "
          f"(max_drop={max_drop_rate}, skip_first={skip_first}, skip_last={skip_last})")
    return patched


# ============================================================
# Experiment 3: GaugeReLU — Phase-Magnitude Activation
# ============================================================

def gauge_relu(x: Tensor) -> Tensor:
    """Phase-magnitude activation treating dimension pairs as complex numbers.

    Applies nonlinearity to magnitude only, preserving phase.
    This retains relative phase information between paired dimensions
    that component-wise relu² destroys.
    """
    shape = x.shape
    dim = shape[-1]

    # Ensure even dimension (pad with zero if odd)
    if dim % 2 != 0:
        x = F.pad(x, (0, 1))
        dim = dim + 1
        padded = True
    else:
        padded = False

    x_pairs = x.reshape(*shape[:-1], -1, 2)  # (..., dim//2, 2)

    # Magnitude: sqrt(a² + b² + eps) — eps prevents NaN gradient at zero
    magnitude = (x_pairs[..., 0].square() + x_pairs[..., 1].square() + 1e-8).sqrt()

    # Phase: atan2(b, a) — safe because magnitude > 0 from eps
    phase = torch.atan2(x_pairs[..., 1], x_pairs[..., 0])

    # Apply nonlinearity to magnitude only (leaky_relu then square, like relu²)
    mag_activated = F.leaky_relu(magnitude, negative_slope=0.01).square()

    # Reconstruct from polar coordinates
    out = torch.stack([mag_activated * phase.cos(),
                       mag_activated * phase.sin()], dim=-1)
    out = out.reshape(*shape[:-1], -1)

    if padded:
        out = out[..., :shape[-1]]

    return out


def apply_gauge_relu(model: nn.Module):
    """Replace relu² with GaugeReLU in all MLP layers."""
    patched = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'activation') or not hasattr(module, 'proj'):
            continue
        if module.activation != 'relu2':
            continue

        original_fc = module.fc if hasattr(module, 'fc') else None
        original_proj = module.proj

        if original_fc is None:
            continue

        def make_gauge_forward(fc, proj):
            def forward(x: Tensor) -> Tensor:
                h = fc(x)
                h = gauge_relu(h)
                return proj(h)
            return forward

        module.forward = make_gauge_forward(original_fc, original_proj)
        patched += 1

    print(f"GaugeReLU applied to {patched} MLP layers")
    return patched


# ============================================================
# Master application function
# ============================================================

def apply_activation_experiments(model: nn.Module, power_act: bool = False,
                                  stoch_depth: float = 0.0,
                                  gauge_relu_enabled: bool = False,
                                  stoch_skip_first: int = 2,
                                  stoch_skip_last: int = 2):
    """Apply all enabled activation experiments.

    Args:
        model: the GPT model
        power_act: enable parametric power activation
        stoch_depth: max drop rate for stochastic depth (0 = disabled)
        gauge_relu_enabled: enable GaugeReLU activation
        stoch_skip_first: number of first blocks to never drop
        stoch_skip_last: number of last blocks to never drop
    """
    total = 0

    # Power and GaugeReLU are mutually exclusive (both modify MLP activation)
    if power_act and gauge_relu_enabled:
        print("WARNING: Both POWER_ACT and GAUGE_RELU enabled. Using GAUGE_RELU.")
        power_act = False

    if power_act:
        total += apply_parametric_power(model)

    if gauge_relu_enabled:
        total += apply_gauge_relu(model)

    if stoch_depth > 0:
        total += apply_stochastic_depth(model, max_drop_rate=stoch_depth,
                                         skip_first=stoch_skip_first,
                                         skip_last=stoch_skip_last)

    if total == 0:
        print("No activation experiments enabled")

    return total

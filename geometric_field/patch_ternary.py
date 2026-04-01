#!/usr/bin/env python3
"""Patch the ternary training script for non-Hopper GPUs.

Modifies train_gpt_cuda_ternary.py to:
1. Add FlashAttention fallback (FA3 → FA2 → SDPA with GQA expansion)
2. Enable SDP backend fallbacks
3. Add USE_COMPILE flag

Usage:
    python geometric_field/patch_ternary.py train_gpt_cuda_ternary.py
"""
import sys
import re


def patch(source: str) -> str:
    """Apply all patches to the ternary training script."""

    # 1. Flash attention import fallback
    # Rename FA3 import to _flash_attn_3 so step 2's replacement doesn't create recursion
    source = source.replace(
        'from flash_attn_interface import flash_attn_func',
        '''try:
    from flash_attn_interface import flash_attn_func as _flash_attn_3
except ImportError:
    _flash_attn_3 = None
try:
    from flash_attn import flash_attn_func as _flash_attn_2
except ImportError:
    _flash_attn_2 = None

def _sdpa_fallback(q, k, v, causal=True):
    """SDPA fallback with GQA head expansion for non-Hopper GPUs."""
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    if kt.size(1) != qt.size(1):
        n_rep = qt.size(1) // kt.size(1)
        kt = kt.repeat_interleave(n_rep, dim=1)
        vt = vt.repeat_interleave(n_rep, dim=1)
    import torch.nn.functional as _F
    return _F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)

def _fa_dispatch(q, k, v, causal=True):
    """Dispatch to best available attention implementation."""
    if _flash_attn_3 is not None:
        return _flash_attn_3(q, k, v, causal=causal)
    elif _flash_attn_2 is not None:
        return _flash_attn_2(q, k, v, causal=causal)
    else:
        return _sdpa_fallback(q, k, v, causal=causal)'''
    )

    # 2. Replace all remaining flash_attn_func calls with _fa_dispatch
    # Safe because FA3 was renamed to _flash_attn_3 above
    source = source.replace('flash_attn_func(', '_fa_dispatch(')

    # 3. SDP backend fallbacks
    source = source.replace(
        'enable_mem_efficient_sdp(False)',
        'enable_mem_efficient_sdp(True)   # Fallback for non-Hopper'
    )
    source = source.replace(
        'enable_math_sdp(False)',
        'enable_math_sdp(True)            # Fallback'
    )

    # 4. torch.compile flag (if not already present)
    if 'use_compile' not in source:
        # Add use_compile to the Hyperparameters class
        source = source.replace(
            '    compile_mode = _e("COMPILE_MODE", "default")',
            '    compile_mode = _e("COMPILE_MODE", "default")\n'
            '    use_compile = _e("USE_COMPILE", 1, int)'
        )
        # Gate specific torch.compile calls using args.use_compile
        source = source.replace(
            "ns_orth = torch.compile(ns_orth)",
            "ns_orth = torch.compile(ns_orth) if args.use_compile else ns_orth"
        )
        source = source.replace(
            'compiled_model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None)',
            'compiled_model = torch.compile(base_model, mode=args.compile_mode if args.compile_mode != "default" else None) if args.use_compile else base_model'
        )

    # 5. Geometric field integration
    # Add G env vars to Hyperparameters and apply after model construction
    if 'geom_alpha' not in source:
        source = source.replace(
            '    use_compile = _e("USE_COMPILE", 1, int)',
            '    use_compile = _e("USE_COMPILE", 1, int)\n'
            '    geom_alpha = _e("GEOM_ALPHA", 0.0, float)\n'
            '    geom_beta = _e("GEOM_BETA", 0.0, float)\n'
            '    geom_signals = _e("GEOM_SIGNALS", "")'
        )

        # Add residual, structured zeros, and activation experiment hyperparameters
        source = source.replace(
            '    geom_signals = _e("GEOM_SIGNALS", "")',
            '    geom_signals = _e("GEOM_SIGNALS", "")\n'
            '    residual_epsilon = _e("RESIDUAL_EPSILON", 0.0, float)\n'
            '    residual_coarse_gs = _e("RESIDUAL_COARSE_GS", 512, int)\n'
            '    zero_bias_range = _e("ZERO_BIAS_RANGE", 0.0, float)\n'
            '    power_act = _e("POWER_ACT", 0, int)\n'
            '    stoch_depth = _e("STOCH_DEPTH", 0.0, float)\n'
            '    gauge_relu = _e("GAUGE_RELU", 0, int)'
        )

        # Inject all experiment applications after model construction
        source = source.replace(
            '    compiled_model = torch.compile(base_model',
            '    # Apply experiments if configured\n'
            '    import sys as _sys; _sys.path.insert(0, ".")\n'
            '    if args.geom_alpha > 0 or args.geom_beta > 0:\n'
            '        from geometric_field.geometric_field import apply_geometric_field\n'
            '        apply_geometric_field(base_model, signals_path=args.geom_signals,\n'
            '                              alpha=args.geom_alpha, beta=args.geom_beta)\n'
            '    if args.residual_epsilon > 0:\n'
            '        from geometric_field.ternary_residual import apply_ternary_residual\n'
            '        apply_ternary_residual(base_model, epsilon=args.residual_epsilon,\n'
            '                               coarse_group_size=args.residual_coarse_gs)\n'
            '    if args.zero_bias_range > 0:\n'
            '        from geometric_field.structured_zeros import apply_structured_zeros\n'
            '        apply_structured_zeros(base_model, signals_path=args.geom_signals,\n'
            '                               bias_range=args.zero_bias_range)\n'
            '    if args.power_act or args.stoch_depth > 0 or args.gauge_relu:\n'
            '        from geometric_field.activation_experiments import apply_activation_experiments\n'
            '        apply_activation_experiments(base_model, power_act=bool(args.power_act),\n'
            '                                     stoch_depth=args.stoch_depth,\n'
            '                                     gauge_relu_enabled=bool(args.gauge_relu))\n'
            '    compiled_model = torch.compile(base_model'
        )

    # 6. Save raw state_dict before quantization (for Phase 0 analysis)
    source = source.replace(
        '        # Two methods: Standard Base-3 vs Bitmask Mapping',
        '        # Save raw state_dict for Phase 0 analysis\n'
        '        torch.save(sd, "final_model_raw_sd.pt")\n'
        '        log0(f"saved raw state_dict: final_model_raw_sd.pt ({len(sd)} keys)")\n'
        '        # Two methods: Standard Base-3 vs Bitmask Mapping'
    )

    return source


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_ternary.py <train_gpt_cuda_ternary.py>")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, 'r') as f:
        source = f.read()

    patched = patch(source)

    with open(filepath, 'w') as f:
        f.write(patched)

    print(f"Patched {filepath}")
    print("  - FlashAttention fallback (FA3 → FA2 → SDPA+GQA)")
    print("  - SDP backend fallbacks enabled")
    print("  - USE_COMPILE flag added")


if __name__ == "__main__":
    main()

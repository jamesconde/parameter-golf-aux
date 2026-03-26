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
    source = source.replace(
        'from flash_attn_interface import flash_attn_func',
        '''try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None
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
    if flash_attn_func is not None:
        return flash_attn_func(q, k, v, causal=causal)
    elif _flash_attn_2 is not None:
        return _flash_attn_2(q, k, v, causal=causal)
    else:
        return _sdpa_fallback(q, k, v, causal=causal)'''
    )

    # 2. Replace all flash_attn_func calls with _fa_dispatch
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
    if 'USE_COMPILE' not in source:
        # Add USE_COMPILE to Hyperparameters-like section
        source = source.replace(
            "COMPILE_MODE = os.environ.get('COMPILE_MODE', 'default')",
            "COMPILE_MODE = os.environ.get('COMPILE_MODE', 'default')\n"
            "USE_COMPILE = bool(int(os.environ.get('USE_COMPILE', '1')))"
        )
        # Gate torch.compile calls
        source = re.sub(
            r'(\w+)\s*=\s*torch\.compile\(([^)]+)\)',
            r'\1 = torch.compile(\2) if USE_COMPILE else \2',
            source
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

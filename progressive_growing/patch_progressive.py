#!/usr/bin/env python3
"""Patch the int6 SOTA train_gpt.py to add progressive depth growing.

The model starts training with fewer layers (faster steps, more data)
and switches to full depth partway through training.

Usage:
    python progressive_growing/patch_progressive.py train_gpt.py

Env vars added:
    GROW_FRACTION: 0.0 = disabled (default), 0.33 = grow at 33% of wallclock
    GROW_INITIAL_LAYERS: number of layers in Phase 1 (default: 7)
"""
import sys


def patch(source: str) -> str:
    """Apply progressive growing patches to the SOTA train_gpt.py."""

    # 1. Add flash attention fallback (same as ternary patches)
    source = source.replace(
        'from flash_attn_interface import flash_attn_func as flash_attn_3_func',
        '''try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    flash_attn_3_func = None
try:
    from flash_attn import flash_attn_func as flash_attn_2_func
except ImportError:
    flash_attn_2_func = None'''
    )

    # Replace flash_attn_3_func calls with fallback
    if 'flash_attn_3_func' in source and '_fa_dispatch' not in source:
        # Add dispatch function after imports
        # Use _fa3_orig to avoid infinite recursion when replacing flash_attn_3_func( calls
        source = source.replace(
            'class Hyperparameters:',
            '''_fa3_orig = flash_attn_3_func  # save reference before renaming calls

def _fa_dispatch(q, k, v, causal=True):
    if _fa3_orig is not None:
        return _fa3_orig(q, k, v, causal=causal)
    elif flash_attn_2_func is not None:
        return flash_attn_2_func(q, k, v, causal=causal)
    else:
        import torch.nn.functional as _F
        qt, kt, vt = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        if kt.size(1) != qt.size(1):
            n_rep = qt.size(1) // kt.size(1)
            kt = kt.repeat_interleave(n_rep, dim=1)
            vt = vt.repeat_interleave(n_rep, dim=1)
        return _F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1,2)

class Hyperparameters:'''
        )
        source = source.replace('flash_attn_3_func(', '_fa_dispatch(')

    # 2. Add SDP backend fallbacks
    source = source.replace(
        'enable_mem_efficient_sdp(False)',
        'enable_mem_efficient_sdp(True)'
    )
    source = source.replace(
        'enable_math_sdp(False)',
        'enable_math_sdp(True)'
    )

    # 3. Add USE_COMPILE flag
    if 'USE_COMPILE' not in source:
        source = source.replace(
            '    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))',
            '    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))\n'
            '    use_compile = int(os.environ.get("USE_COMPILE", "1"))'
        )
        source = source.replace(
            '    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)',
            '    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if args.use_compile else base_model'
        )

    # 4. Add progressive growing hyperparameters
    if 'GROW_FRACTION' not in source:
        source = source.replace(
            '    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))',
            '    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))\n'
            '    grow_fraction = float(os.environ.get("GROW_FRACTION", 0.0))\n'
            '    grow_initial_layers = int(os.environ.get("GROW_INITIAL_LAYERS", 7))'
        )

    # 5. Add the shallow forward step function
    # Insert after the GPT class (before the eval functions)
    eval_func_marker = 'def eval_val('
    if eval_func_marker in source and 'run_shallow_step' not in source:
        source = source.replace(
            eval_func_marker,
            '''def run_shallow_step(base_model, x, y, active_layers, full_layers):
    """Run forward with fewer layers by temporarily modifying model attributes."""
    orig_enc = base_model.num_encoder_layers
    orig_dec = base_model.num_decoder_layers
    orig_skip = base_model.num_skip_weights

    active_enc = active_layers // 2
    active_dec = active_layers - active_enc

    base_model.num_encoder_layers = active_enc
    base_model.num_decoder_layers = active_dec
    base_model.num_skip_weights = min(active_enc, active_dec)

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = base_model(x, y)
    finally:
        base_model.num_encoder_layers = orig_enc
        base_model.num_decoder_layers = orig_dec
        base_model.num_skip_weights = orig_skip

    return loss


''' + eval_func_marker
        )

    # 6. Modify the training loop to support progressive growing
    # Find the training loop's forward pass and wrap it
    training_forward = '''        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps'''

    progressive_forward = '''        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            if _growing and elapsed_ms < _grow_at_ms:
                # Phase 1: shallow, uncompiled, fast steps
                loss = run_shallow_step(base_model, x, y,
                    active_layers=args.grow_initial_layers,
                    full_layers=args.num_layers)
            else:
                if _growing:
                    log0(f"GROW: {args.grow_initial_layers}L -> {args.num_layers}L "
                         f"at step {step}, train_time:{elapsed_ms:.0f}ms")
                    _growing = False
                    # Restore dormant layer banks if weight decay may have shrunk them
                    if args.muon_wd > 0 and _initial_banks is not None:
                        with torch.no_grad():
                            n = args.num_layers
                            for li in range(args.grow_initial_layers, n):
                                base_model.qo_bank.data[li] = _initial_banks['qo'][li]
                                base_model.qo_bank.data[n+li] = _initial_banks['qo'][n+li]
                                base_model.kv_bank.data[li] = _initial_banks['kv'][li]
                                base_model.kv_bank.data[n+li] = _initial_banks['kv'][n+li]
                                base_model.mlp_up_bank.data[li] = _initial_banks['mlp_up'][li]
                                base_model.mlp_down_bank.data[li] = _initial_banks['mlp_down'][li]
                        log0(f"GROW: restored dormant banks for layers {args.grow_initial_layers}-{n-1}")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps'''

    if training_forward in source:
        source = source.replace(training_forward, progressive_forward)
    else:
        print("WARNING: Could not find training forward pass to patch")

    # 7. Add growing state initialization before the training loop
    source = source.replace(
        '    training_time_ms = 0.0\n'
        '    stop_after_step: int | None = None',
        '    # Progressive growing state\n'
        '    _growing = args.grow_fraction > 0\n'
        '    if _growing and args.grow_initial_layers >= args.num_layers:\n'
        '        log0(f"WARNING: grow_initial_layers={args.grow_initial_layers} >= num_layers={args.num_layers}, disabling growing")\n'
        '        _growing = False\n'
        '    _grow_at_ms = args.max_wallclock_seconds * 1000.0 * args.grow_fraction if _growing else 0\n'
        '    _initial_banks = None\n'
        '    if _growing:\n'
        '        log0(f"progressive_growing: {args.grow_initial_layers}L -> {args.num_layers}L "\n'
        '             f"at {args.grow_fraction*100:.0f}% wallclock ({_grow_at_ms/1000:.0f}s)")\n'
        '        _initial_banks = {\n'
        '            "qo": base_model.qo_bank.data.clone(),\n'
        '            "kv": base_model.kv_bank.data.clone(),\n'
        '            "mlp_up": base_model.mlp_up_bank.data.clone(),\n'
        '            "mlp_down": base_model.mlp_down_bank.data.clone(),\n'
        '        }\n'
        '    training_time_ms = 0.0\n'
        '    stop_after_step: int | None = None'
    )

    return source


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_progressive.py <train_gpt.py>")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, 'r') as f:
        source = f.read()

    patched = patch(source)

    with open(filepath, 'w') as f:
        f.write(patched)

    print(f"Patched {filepath}")
    print("  - Flash attention fallback (FA3 → FA2 → SDPA+GQA)")
    print("  - SDP backend fallbacks")
    print("  - USE_COMPILE flag")
    print("  - Progressive growing (GROW_FRACTION, GROW_INITIAL_LAYERS)")
    print("  - run_shallow_step() for Phase 1")
    print("  - Dormant bank restoration at growth transition")


if __name__ == "__main__":
    main()

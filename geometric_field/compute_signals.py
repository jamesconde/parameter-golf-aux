#!/usr/bin/env python3
"""Experiment 1: Compute data-driven G signals.

Computes the two signals that inform G's shape:
1. Input covariance diagonal (C_diag) — which input dims have high variance
2. Word-boundary direction (delta_e) — which dims separate word-start from continuation

Saves signals to a .pt file for use by GeometricTernaryLinear.

Usage:
    python geometric_field/compute_signals.py \
        --checkpoint final_model_raw_sd.pt \
        --tokenizer fineweb_8192_bpe.model \
        --data-path ./data/datasets/fineweb10B_sp8192 \
        --output geometric_field/g_signals.pt
"""
import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F


def compute_word_boundary_direction(tokenizer_path: str, embeddings: torch.Tensor,
                                     embed_proj_weight: torch.Tensor = None) -> torch.Tensor:
    """Compute the direction in embedding space separating word-starts from continuations.

    Returns a unit vector in model_dim space.
    """
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    word_start_ids = []
    continuation_ids = []

    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece.startswith("▁") or piece.startswith(" "):
            word_start_ids.append(i)
        elif piece.startswith("<") and piece.endswith(">"):
            pass  # skip special tokens
        else:
            continuation_ids.append(i)

    print(f"  Word-start tokens: {len(word_start_ids)}")
    print(f"  Continuation tokens: {len(continuation_ids)}")

    emb = embeddings.float()
    e_start = emb[word_start_ids].mean(dim=0)
    e_continue = emb[continuation_ids].mean(dim=0)
    delta_e = e_start - e_continue
    delta_e = delta_e / delta_e.norm().clamp(min=1e-8)

    # Project to model_dim if embed_proj exists
    if embed_proj_weight is not None:
        delta_e = F.linear(delta_e, embed_proj_weight.float())
        delta_e = delta_e / delta_e.norm().clamp(min=1e-8)

    print(f"  Delta_e shape: {delta_e.shape}")
    print(f"  Delta_e norm: {delta_e.norm().item():.4f}")
    return delta_e


def compute_input_covariance(model, val_tokens: np.ndarray, device: torch.device,
                              seq_len: int = 1024, n_batches: int = 10,
                              batch_size: int = 8) -> dict:
    """Run validation data through model, collect input activation statistics.

    Returns dict mapping module name → C_diag (shape (in_features,))
    """
    activation_stats = {}

    def make_hook(name):
        def hook(module, input, output):
            x = input[0].detach().float()
            x_sq = (x ** 2).mean(dim=tuple(range(x.dim() - 1)))  # shape (in_features,)
            if name not in activation_stats:
                activation_stats[name] = {"sum": torch.zeros_like(x_sq), "count": 0}
            activation_stats[name]["sum"] += x_sq
            activation_stats[name]["count"] += 1
        return hook

    # Register hooks on ternary linear layers
    hooks = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Ternary" in cls_name or "ternary" in cls_name:
            hooks.append(module.register_forward_hook(make_hook(name)))
            # Also hook the inner .linear if it exists (NormedTernaryLinear)
            if hasattr(module, "linear"):
                hooks.append(module.linear.register_forward_hook(make_hook(name + ".linear")))

    if not hooks:
        print("  WARNING: No ternary layers found for hooks. Trying all nn.Linear...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.shape[0] > 100:
                hooks.append(module.register_forward_hook(make_hook(name)))

    print(f"  Registered {len(hooks)} hooks")

    # Prepare validation batches
    # Load tokens with header skip (same as Phase 0 fix)
    n_seqs = min(n_batches * batch_size, len(val_tokens) // (seq_len + 1))
    tokens = torch.from_numpy(val_tokens[:n_seqs * (seq_len + 1)].astype(np.int64))
    tokens = tokens.reshape(n_seqs, seq_len + 1)

    model.eval()
    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch = tokens[i:i + batch_size, :seq_len].to(device)
            targets = tokens[i:i + batch_size, 1:seq_len + 1].to(device)
            try:
                model(batch, targets)
            except Exception:
                # Some models need different calling conventions
                try:
                    model(batch)
                except Exception as e:
                    print(f"  WARNING: forward pass failed: {e}")
                    break
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Batch {i // batch_size + 1}/{n_seqs // batch_size}")

    for h in hooks:
        h.remove()

    # Compute C_diag
    C_diag = {}
    for name, stats in activation_stats.items():
        if stats["count"] > 0:
            C_diag[name] = stats["sum"] / stats["count"]

    print(f"  Collected C_diag for {len(C_diag)} layers")
    return C_diag


def load_val_tokens(data_path: str, max_tokens: int = 500000) -> np.ndarray:
    """Load validation tokens from binary files, skipping header."""
    pattern = os.path.join(data_path, "fineweb_val_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No validation files matching {pattern}")

    header_bytes = 256 * np.dtype("<i4").itemsize
    arrays = []
    total = 0
    for f in files:
        header = np.fromfile(f, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(f, dtype="<u2", count=num_tokens, offset=header_bytes)
        arrays.append(tokens)
        total += len(tokens)
        if total >= max_tokens:
            break

    result = np.concatenate(arrays)[:max_tokens]
    print(f"  Loaded {len(result):,} validation tokens")
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute G signals")
    parser.add_argument("--checkpoint", default="final_model_raw_sd.pt")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_8192_bpe.model")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp8192")
    parser.add_argument("--output", default="geometric_field/g_signals.pt")
    parser.add_argument("--max-val-tokens", type=int, default=500000)
    parser.add_argument("--n-batches", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    sys.path.insert(0, ".")
    from train_gpt_cuda_ternary import Hyperparameters, GPT
    hp = Hyperparameters()
    model = GPT(hp).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Signal 1: Word-boundary direction
    print("\nComputing word-boundary direction (delta_e)...")
    embeddings = model.tok_emb.weight.detach()
    embed_proj = getattr(model, "embed_proj", None)
    embed_proj_weight = embed_proj.weight.detach() if embed_proj is not None else None
    delta_e = compute_word_boundary_direction(args.tokenizer, embeddings, embed_proj_weight)

    # Signal 2: Input covariance
    print("\nComputing input covariance (C_diag)...")
    val_tokens = load_val_tokens(args.data_path, max_tokens=args.max_val_tokens)
    C_diag = compute_input_covariance(model, val_tokens, device, n_batches=args.n_batches)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "delta_e": delta_e.cpu(),
        "C_diag": {k: v.cpu() for k, v in C_diag.items()},
        "embed_dim": embeddings.shape[1],
        "model_dim": hp.model_dim,
    }, args.output)
    print(f"\nSignals saved to {args.output}")
    print(f"  delta_e: {delta_e.shape}")
    print(f"  C_diag layers: {len(C_diag)}")


if __name__ == "__main__":
    main()

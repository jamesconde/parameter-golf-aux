#!/usr/bin/env python3
"""Phase 0: Measure Ternary Quantization Error Structure.

DECISION GATE: If no spatial structure exists, G cannot help.

Analyzes the trained ternary model's weight matrices to determine:
- Row error profile (handled by per-group scale → less interesting)
- Column error profile (NOT handled by any existing mechanism → our target)
- Within-group position profile
- Layer-wise variation
- Fused matrix sub-component analysis

Classifies into scenarios:
  A: No structure → STOP
  B: Row-only structure → STOP (per-group scale handles this)
  C: Column structure → PROCEED with column-dependent G
  D: Both row + column → PROCEED with full G(i,j)
  E: Layer-position variation → Add per-layer modulation

Usage (on Colab, after training a ternary model):
    python geometric_field/phase0_analysis.py --checkpoint final_model_state.pt

    Or with the full model (loads from training script):
    python geometric_field/phase0_analysis.py --train-script train_gpt_cuda_ternary.py
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def compute_ternary_error(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Compute element-wise ternary quantization error for a weight matrix."""
    w_float = weight.detach().float()
    w_g = w_float.reshape(-1, group_size)
    scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    q = (w_g / scale).round().clamp(-1, 1)
    w_ternary = (q * scale).reshape(w_float.shape)
    error = (w_float - w_ternary).abs()
    return error


def analyze_matrix(name: str, weight: torch.Tensor, group_size: int = 128) -> dict:
    """Analyze quantization error structure for one weight matrix."""
    N, M = weight.shape  # (out_features, in_features)
    error = compute_ternary_error(weight, group_size)

    # Basic stats
    mean_error = error.mean().item()
    std_error = error.std().item()

    # Row error profile: mean error per output feature
    row_error = error.mean(dim=1)  # shape (N,)
    row_std = row_error.std().item()
    # Expected row std under null (no structure): error.std() / sqrt(M)
    null_row_std = std_error / math.sqrt(M)
    row_structure_ratio = row_std / max(null_row_std, 1e-12)

    # Column error profile: mean error per input feature
    col_error = error.mean(dim=0)  # shape (M,)
    col_std = col_error.std().item()
    null_col_std = std_error / math.sqrt(N)
    col_structure_ratio = col_std / max(null_col_std, 1e-12)

    # Within-group position profile
    n_groups = (N * M) // group_size
    error_flat = error.reshape(-1)
    if len(error_flat) >= group_size:
        # Reshape into groups and average across groups
        usable = (len(error_flat) // group_size) * group_size
        error_grouped = error_flat[:usable].reshape(-1, group_size)
        position_profile = error_grouped.mean(dim=0).tolist()  # shape (group_size,)
        position_std = error_grouped.std(dim=0).mean().item()  # within-group positional variation
    else:
        position_profile = []
        position_std = 0.0

    # Weight distribution stats
    w_float = weight.detach().float()
    zero_frac = (compute_ternary_error(weight, group_size) < 1e-10).float().mean().item()
    w_g = w_float.reshape(-1, group_size)
    scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    q = (w_g / scale).round().clamp(-1, 1)
    ternary_zero_frac = (q == 0).float().mean().item()

    result = {
        "name": name,
        "shape": [N, M],
        "group_size": group_size,
        "mean_error": mean_error,
        "std_error": std_error,
        "row_structure_ratio": row_structure_ratio,
        "col_structure_ratio": col_structure_ratio,
        "row_error_std": row_std,
        "col_error_std": col_std,
        "position_profile_std": position_std,
        "ternary_zero_frac": ternary_zero_frac,
        # Store profiles for visualization
        "row_error_profile": row_error.tolist(),
        "col_error_profile": col_error.tolist(),
        "position_profile": position_profile,
    }

    return result


def analyze_fused_matrix(name: str, weight: torch.Tensor, group_size: int,
                         splits: dict) -> list:
    """Analyze a fused matrix (c_qkv or gate_up) by sub-component."""
    results = []
    # Full matrix analysis
    results.append(analyze_matrix(name, weight, group_size))

    # Sub-component analysis
    for sub_name, (start, end) in splits.items():
        sub_weight = weight[start:end, :]
        sub_result = analyze_matrix(f"{name}.{sub_name}", sub_weight, group_size)
        results.append(sub_result)

    return results


def classify_scenario(matrix_results: list) -> dict:
    """Classify the overall scenario based on all matrix analyses."""
    row_ratios = [r["row_structure_ratio"] for r in matrix_results if "." not in r["name"].split("/")[-1]]
    col_ratios = [r["col_structure_ratio"] for r in matrix_results if "." not in r["name"].split("/")[-1]]

    mean_row = np.mean(row_ratios) if row_ratios else 0
    mean_col = np.mean(col_ratios) if col_ratios else 0
    max_row = max(row_ratios) if row_ratios else 0
    max_col = max(col_ratios) if col_ratios else 0

    # Layer-position variation
    layer_errors = defaultdict(list)
    for r in matrix_results:
        name = r["name"]
        if "blocks." in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "blocks" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    layer_errors[layer_idx].append(r["mean_error"])
                    break

    layer_means = {k: np.mean(v) for k, v in sorted(layer_errors.items())}
    if len(layer_means) > 1:
        layer_vals = list(layer_means.values())
        layer_variation = np.std(layer_vals) / np.mean(layer_vals) if np.mean(layer_vals) > 0 else 0
    else:
        layer_variation = 0

    # Classify
    if mean_col > 2.0 and mean_row > 2.0:
        scenario = "D"
        description = "Both row AND column structure — full G(i,j) possible"
    elif mean_col > 2.0:
        scenario = "C"
        description = "Column structure — column-dependent G viable"
    elif mean_row > 2.0:
        scenario = "B"
        description = "Row-only structure — per-group scale already handles this, STOP"
    else:
        scenario = "A"
        description = "No significant spatial structure — G cannot help, STOP"

    if layer_variation > 0.15:
        scenario += "+E"
        description += " + significant layer-position variation"

    return {
        "scenario": scenario,
        "description": description,
        "proceed": scenario.startswith("C") or scenario.startswith("D"),
        "mean_row_structure": float(mean_row),
        "mean_col_structure": float(mean_col),
        "max_row_structure": float(max_row),
        "max_col_structure": float(max_col),
        "layer_variation": float(layer_variation),
        "layer_mean_errors": {str(k): float(v) for k, v in layer_means.items()},
    }


def print_report(all_results: list, classification: dict):
    """Print human-readable analysis report."""
    print("\n" + "=" * 75)
    print("PHASE 0: TERNARY QUANTIZATION ERROR STRUCTURE ANALYSIS")
    print("=" * 75)

    print(f"\n{'Matrix':<45} {'Shape':>12} {'MeanErr':>8} {'RowStr':>7} {'ColStr':>7} {'ZeroFr':>7}")
    print("-" * 90)
    for r in all_results:
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
        print(f"{r['name']:<45} {shape_str:>12} {r['mean_error']:>8.5f} "
              f"{r['row_structure_ratio']:>7.2f} {r['col_structure_ratio']:>7.2f} "
              f"{r['ternary_zero_frac']:>7.3f}")

    print(f"\n--- Scenario Classification ---")
    c = classification
    print(f"  Scenario: {c['scenario']}")
    print(f"  {c['description']}")
    print(f"  Mean row structure ratio:  {c['mean_row_structure']:.2f} (threshold: 2.0)")
    print(f"  Mean col structure ratio:  {c['mean_col_structure']:.2f} (threshold: 2.0)")
    print(f"  Max row structure ratio:   {c['max_row_structure']:.2f}")
    print(f"  Max col structure ratio:   {c['max_col_structure']:.2f}")
    print(f"  Layer variation:           {c['layer_variation']:.3f} (threshold: 0.15)")

    if c["layer_mean_errors"]:
        print(f"\n--- Error by Layer ---")
        for layer, err in sorted(c["layer_mean_errors"].items(), key=lambda x: int(x[0])):
            bar = "█" * int(err / max(c["layer_mean_errors"].values()) * 30)
            role = "encoder" if int(layer) < 5 else "decoder"
            print(f"  Layer {layer:>2} ({role}): {err:.5f} {bar}")

    print(f"\n--- Decision ---")
    if c["proceed"]:
        print(f"  ✓ PROCEED with geometric field G experiments")
        print(f"  Column structure ratio {c['mean_col_structure']:.2f} > 2.0 threshold")
        print(f"  The quantization error varies systematically across input dimensions")
        print(f"  G can exploit this by modulating weights per-column before STE")
    else:
        print(f"  ✗ STOP — no exploitable spatial structure")
        if c["scenario"] == "A":
            print(f"  Neither row nor column structure exceeds threshold")
            print(f"  Quantization error is uniformly distributed — G cannot help")
        elif c["scenario"].startswith("B"):
            print(f"  Row structure exists but per-group scale already handles it")
            print(f"  No additional benefit from G")
        print(f"  Write up as negative result (still valuable for competition)")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Ternary quantization error analysis")
    parser.add_argument("--checkpoint", default="",
                        help="Path to trained model state_dict (.pt file)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Ternary quantization group size")
    parser.add_argument("--output", default="geometric_field/phase0_results.json",
                        help="Output JSON path")
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Compute expected shapes
    dim = args.model_dim
    head_dim = dim // args.num_heads
    q_size = dim  # num_heads * head_dim
    kv_size = args.num_kv_heads * head_dim
    hidden = dim * args.mlp_mult

    # Expected weight matrix shapes
    expected_matrices = {
        "attn.c_qkv": (q_size + 2 * kv_size, dim),  # (1536, 768)
        "attn.proj": (dim, dim),                       # (768, 768)
        "mlp.gate_up": (hidden * 2, dim),              # (6144, 768)
        "mlp.proj": (dim, hidden),                      # nn.Linear(hidden, dim) → weight shape (dim, hidden) = (768, 3072)
    }

    # Fused matrix splits
    qkv_splits = {
        "Q": (0, q_size),
        "K": (q_size, q_size + kv_size),
        "V": (q_size + kv_size, q_size + 2 * kv_size),
    }
    gate_up_splits = {
        "gate": (0, hidden),
        "up": (hidden, hidden * 2),
    }

    # Load checkpoint — handle multiple formats
    def load_checkpoint(path):
        """Load checkpoint, handling raw state_dict, full model, or LZMA-compressed."""
        import io
        try:
            # Try raw state_dict first
            sd = torch.load(path, map_location=device, weights_only=True)
            if isinstance(sd, dict) and any("weight" in k for k in sd):
                return sd
        except Exception:
            pass
        try:
            # Try LZMA-compressed (ternary submission format)
            import lzma
            with open(path, "rb") as f:
                raw = f.read()
            try:
                decompressed = lzma.decompress(raw)
            except lzma.LZMAError:
                decompressed = raw  # Not LZMA compressed
            loaded = torch.load(io.BytesIO(decompressed), map_location=device, weights_only=False)
            if isinstance(loaded, dict):
                # Could be quantized format — look for state_dict inside
                if "state_dict" in loaded:
                    return loaded["state_dict"]
                elif any("weight" in k for k in loaded):
                    return loaded
                else:
                    return loaded
            elif hasattr(loaded, "state_dict"):
                return loaded.state_dict()
        except Exception:
            pass
        # Last resort: load without weights_only
        try:
            loaded = torch.load(path, map_location=device, weights_only=False)
            if hasattr(loaded, "state_dict"):
                return loaded.state_dict()
            return loaded
        except Exception as e:
            print(f"ERROR: Could not load {path}: {e}")
            sys.exit(1)

    checkpoint_path = args.checkpoint
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Search for common checkpoint names
        for candidate in ["final_model_state.pt", "final_model.pt", "checkpoint.pt"]:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("ERROR: No checkpoint found. Train a ternary model first.")
        print("  Expected: final_model.pt, final_model_state.pt, or --checkpoint PATH")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_checkpoint(checkpoint_path)
    print(f"  Loaded {len(state_dict)} keys")

    # Find and analyze all ternary weight matrices
    all_results = []
    print(f"\nAnalyzing {args.num_layers} layers × 4 matrix types = {args.num_layers * 4} matrices...")

    for layer_idx in range(args.num_layers):
        for mat_type in ["attn.c_qkv", "attn.proj", "mlp.gate_up", "mlp.proj"]:
            # Try different key formats
            candidates = [
                f"blocks.{layer_idx}.{mat_type}.weight",
                f"blocks.{layer_idx}.{mat_type.replace('.', '_')}.weight",
            ]

            weight = None
            for key in candidates:
                if key in state_dict:
                    weight = state_dict[key]
                    break

            if weight is None:
                # Search for partial match
                for k in state_dict:
                    if f"blocks.{layer_idx}" in k and mat_type.split(".")[-1] in k and "weight" in k:
                        weight = state_dict[k]
                        break

            if weight is None:
                print(f"  WARNING: Could not find weight for blocks.{layer_idx}.{mat_type}")
                continue

            name = f"blocks.{layer_idx}.{mat_type}"

            # Analyze based on matrix type
            if mat_type == "attn.c_qkv":
                results = analyze_fused_matrix(name, weight, args.group_size, qkv_splits)
            elif mat_type == "mlp.gate_up":
                results = analyze_fused_matrix(name, weight, args.group_size, gate_up_splits)
            else:
                results = [analyze_matrix(name, weight, args.group_size)]

            all_results.extend(results)

    if not all_results:
        print("\nERROR: No weight matrices found in checkpoint!")
        print("Available keys (sample):")
        for i, k in enumerate(sorted(state_dict.keys())):
            print(f"  {k}")
            if i > 20:
                print(f"  ... ({len(state_dict)} total keys)")
                break
        sys.exit(1)

    # Classify scenario
    classification = classify_scenario(all_results)

    # Print report
    print_report(all_results, classification)

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "classification": classification,
        "matrices": [{k: v for k, v in r.items()
                       if k not in ("row_error_profile", "col_error_profile", "position_profile")}
                      for r in all_results],
        "config": {
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "mlp_mult": args.mlp_mult,
            "num_layers": args.num_layers,
            "group_size": args.group_size,
        },
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save full profiles (for visualization) separately
    profiles_path = args.output.replace(".json", "_profiles.pt")
    torch.save({r["name"]: {
        "row_error": r["row_error_profile"],
        "col_error": r["col_error_profile"],
        "position": r["position_profile"],
    } for r in all_results}, profiles_path)
    print(f"Profiles saved to {profiles_path}")


if __name__ == "__main__":
    main()

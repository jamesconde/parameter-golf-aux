#!/usr/bin/env python3
"""Error analysis of a trained Parameter Golf model.

Runs the model on validation data and decomposes the loss to find
where the model struggles. This informs what auxiliary losses could help.

Analyses:
1. Per-token loss distribution (easy/medium/hard breakdown)
2. Loss by token type (letter, digit, punctuation, space, etc.)
3. Loss by position in sequence
4. Top-K accuracy (is correct answer in top-5, top-10?)
5. Confidence calibration (overconfident vs underconfident)
6. Worst predictions (highest loss tokens and their contexts)
7. Per-document loss variance (some docs much harder?)
8. Loss dynamics: which tokens would benefit from reweighting?

Usage:
    # On Colab after training, from /content/parameter-golf:
    python3 scripts/error_analysis.py --model final_model.pt

    # Or train a fresh short baseline and analyze:
    python3 scripts/error_analysis.py --train-steps 100

All parameters configurable via CLI.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm


def load_validation_tokens(data_path: str, max_tokens: int = 0) -> np.ndarray:
    """Load validation tokens from binary files (skipping 256-int32 header)."""
    pattern = os.path.join(data_path, "fineweb_val_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No validation files matching {pattern}")
    header_bytes = 256 * np.dtype("<i4").itemsize  # 1024 bytes
    arrays = []
    total = 0
    for f in files:
        header = np.fromfile(f, dtype="<i4", count=256)
        num_tokens = int(header[2])
        tokens = np.fromfile(f, dtype="<u2", count=num_tokens, offset=header_bytes)
        arrays.append(tokens)
        total += len(tokens)
        print(f"  Loaded {f}: {len(tokens):,} tokens (max_id={tokens.max()})")
        if max_tokens > 0 and total >= max_tokens:
            break
    result = np.concatenate(arrays)
    if max_tokens > 0:
        result = result[:max_tokens]
    print(f"Loaded {len(result):,} validation tokens from {len(files)} shards")
    return result


def build_token_type_map(sp: spm.SentencePieceProcessor, max_id: int) -> dict:
    """Classify each token ID into a type category."""
    type_map = {}
    for tid in range(min(sp.vocab_size(), max_id + 1)):
        piece = sp.id_to_piece(tid)
        if piece.startswith("<") and piece.endswith(">"):
            type_map[tid] = "special"
        elif piece.replace("▁", "").isdigit():
            type_map[tid] = "number"
        elif piece.replace("▁", "").isalpha():
            type_map[tid] = "word"
        elif piece.replace("▁", "").isspace() or piece == "▁":
            type_map[tid] = "space"
        elif piece.replace("▁", "").strip() == "":
            type_map[tid] = "whitespace"
        else:
            has_alpha = any(c.isalpha() for c in piece.replace("▁", ""))
            has_digit = any(c.isdigit() for c in piece.replace("▁", ""))
            if has_alpha and has_digit:
                type_map[tid] = "alphanum"
            elif has_alpha:
                type_map[tid] = "word_punct"
            elif has_digit:
                type_map[tid] = "num_punct"
            else:
                type_map[tid] = "punctuation"
    # Fill remaining IDs (byte fallbacks)
    for tid in range(max_id + 1):
        if tid not in type_map:
            type_map[tid] = "byte_fallback"
    return type_map


@torch.no_grad()
def analyze_model(model, val_tokens: np.ndarray, device: torch.device,
                  seq_len: int = 2048, batch_size: int = 8,
                  max_sequences: int = 500, sp=None) -> dict:
    """Run validation inference and collect per-token statistics."""
    model.eval()

    n_seqs = min(max_sequences, len(val_tokens) // (seq_len + 1))
    tokens = torch.from_numpy(val_tokens[:n_seqs * (seq_len + 1)].astype(np.int64))
    tokens = tokens.reshape(n_seqs, seq_len + 1)

    all_losses = []          # Per-token loss
    all_targets = []         # Target token IDs
    all_positions = []       # Position in sequence
    all_top1_correct = []    # Whether top-1 prediction is correct
    all_top5_correct = []    # Whether correct is in top-5
    all_top10_correct = []   # Whether correct is in top-10
    all_entropies = []       # Output distribution entropy
    all_max_probs = []       # Confidence (max probability)
    all_seq_ids = []         # Which sequence this token belongs to

    print(f"Analyzing {n_seqs} sequences of {seq_len} tokens ({n_seqs * seq_len:,} total)...")

    for batch_start in range(0, n_seqs, batch_size):
        batch_end = min(batch_start + batch_size, n_seqs)
        batch = tokens[batch_start:batch_end].to(device)
        x = batch[:, :-1]   # Input
        y = batch[:, 1:]     # Target

        # Forward pass to get logits
        if hasattr(model, 'forward_logits'):
            logits = model.forward_logits(x)  # [B, T, V]
        else:
            # Fallback: manual forward
            logits = model(x)

        B, T, V = logits.shape

        # Per-token cross-entropy loss
        logits_flat = logits.reshape(-1, V).float()
        targets_flat = y.reshape(-1)
        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Top-K accuracy
        _, top_k_preds = logits_flat.topk(10, dim=-1)
        target_expanded = targets_flat.unsqueeze(-1)
        top1_correct = (top_k_preds[:, :1] == target_expanded).any(dim=-1)
        top5_correct = (top_k_preds[:, :5] == target_expanded).any(dim=-1)
        top10_correct = (top_k_preds[:, :10] == target_expanded).any(dim=-1)

        # Entropy of output distribution
        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        max_prob = probs.max(dim=-1).values

        # Positions
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
        seq_ids = torch.arange(batch_start, batch_end, device=device).unsqueeze(1).expand(-1, T).reshape(-1)

        all_losses.append(per_token_loss.cpu())
        all_targets.append(targets_flat.cpu())
        all_positions.append(positions.cpu())
        all_top1_correct.append(top1_correct.cpu())
        all_top5_correct.append(top5_correct.cpu())
        all_top10_correct.append(top10_correct.cpu())
        all_entropies.append(entropy.cpu())
        all_max_probs.append(max_prob.cpu())
        all_seq_ids.append(seq_ids.cpu())

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Batch {batch_start // batch_size + 1}/{(n_seqs + batch_size - 1) // batch_size}")

    # Concatenate
    losses = torch.cat(all_losses).numpy()
    targets = torch.cat(all_targets).numpy()
    positions = torch.cat(all_positions).numpy()
    top1 = torch.cat(all_top1_correct).numpy()
    top5 = torch.cat(all_top5_correct).numpy()
    top10 = torch.cat(all_top10_correct).numpy()
    entropies = torch.cat(all_entropies).numpy()
    max_probs = torch.cat(all_max_probs).numpy()
    seq_ids = torch.cat(all_seq_ids).numpy()

    report = {}

    # 1. Overall statistics
    report["overall"] = {
        "mean_loss": float(losses.mean()),
        "median_loss": float(np.median(losses)),
        "std_loss": float(losses.std()),
        "mean_bpb": float(losses.mean() / math.log(2) * (seq_len / (seq_len * 1.2))),  # approx
        "top1_accuracy": float(top1.mean()),
        "top5_accuracy": float(top5.mean()),
        "top10_accuracy": float(top10.mean()),
        "mean_entropy": float(entropies.mean()),
        "mean_confidence": float(max_probs.mean()),
        "n_tokens": len(losses),
    }

    # 2. Loss distribution buckets
    easy = (losses < 1.0).sum()
    medium = ((losses >= 1.0) & (losses < 4.0)).sum()
    hard = ((losses >= 4.0) & (losses < 8.0)).sum()
    impossible = (losses >= 8.0).sum()
    n = len(losses)
    report["difficulty_distribution"] = {
        "easy_pct": float(easy / n * 100),
        "easy_threshold": "loss < 1.0",
        "medium_pct": float(medium / n * 100),
        "medium_threshold": "1.0 <= loss < 4.0",
        "hard_pct": float(hard / n * 100),
        "hard_threshold": "4.0 <= loss < 8.0",
        "impossible_pct": float(impossible / n * 100),
        "impossible_threshold": "loss >= 8.0",
    }

    # Loss at percentiles
    report["loss_percentiles"] = {
        f"p{p}": float(np.percentile(losses, p))
        for p in [5, 10, 25, 50, 75, 90, 95, 99]
    }

    # 3. Loss by position
    pos_bins = [0, 16, 64, 256, 512, 1024, 2048]
    pos_stats = {}
    for i in range(len(pos_bins) - 1):
        lo, hi = pos_bins[i], pos_bins[i + 1]
        mask = (positions >= lo) & (positions < hi)
        if mask.sum() > 0:
            pos_stats[f"{lo}-{hi-1}"] = {
                "mean_loss": float(losses[mask].mean()),
                "top1_acc": float(top1[mask].mean()),
                "n": int(mask.sum()),
            }
    report["loss_by_position"] = pos_stats

    # 4. Loss by token type
    if sp is not None:
        max_id = int(targets.max())
        type_map = build_token_type_map(sp, max_id)
        type_stats = defaultdict(lambda: {"losses": [], "correct": []})
        for i in range(len(targets)):
            tid = int(targets[i])
            ttype = type_map.get(tid, "unknown")
            type_stats[ttype]["losses"].append(losses[i])
            type_stats[ttype]["correct"].append(top1[i])

        type_report = {}
        for ttype, data in sorted(type_stats.items()):
            ls = np.array(data["losses"])
            cs = np.array(data["correct"])
            type_report[ttype] = {
                "mean_loss": float(ls.mean()),
                "top1_acc": float(cs.mean()),
                "count": len(ls),
                "pct_of_total": float(len(ls) / n * 100),
            }
        report["loss_by_token_type"] = type_report

    # 5. Confidence calibration (binned)
    conf_bins = np.linspace(0, 1, 11)
    cal_stats = {}
    for i in range(len(conf_bins) - 1):
        lo, hi = conf_bins[i], conf_bins[i + 1]
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() > 0:
            cal_stats[f"{lo:.1f}-{hi:.1f}"] = {
                "mean_confidence": float(max_probs[mask].mean()),
                "accuracy": float(top1[mask].mean()),
                "count": int(mask.sum()),
                "pct_of_total": float(mask.sum() / n * 100),
            }
    report["confidence_calibration"] = cal_stats

    # 6. Per-document (sequence) loss variance
    seq_losses = []
    unique_seqs = np.unique(seq_ids)
    for sid in unique_seqs:
        mask = seq_ids == sid
        seq_losses.append(float(losses[mask].mean()))
    seq_losses = np.array(seq_losses)
    report["per_document"] = {
        "mean_doc_loss": float(seq_losses.mean()),
        "std_doc_loss": float(seq_losses.std()),
        "min_doc_loss": float(seq_losses.min()),
        "max_doc_loss": float(seq_losses.max()),
        "p10_doc_loss": float(np.percentile(seq_losses, 10)),
        "p90_doc_loss": float(np.percentile(seq_losses, 90)),
        "n_documents": len(seq_losses),
    }

    # 7. Focal loss simulation: what would different gammas do?
    focal_analysis = {}
    for gamma in [0.5, 1.0, 2.0, 3.0]:
        p_correct = np.exp(-losses)  # Approximate p(correct) from CE loss
        focal_weight = (1.0 - p_correct) ** gamma
        # Effective loss with focal weighting
        weighted_loss = (focal_weight * losses).mean()
        # What fraction of gradient comes from each difficulty bucket
        easy_mask = losses < 1.0
        med_mask = (losses >= 1.0) & (losses < 4.0)
        hard_mask = losses >= 4.0
        total_weight = (focal_weight * losses).sum()
        focal_analysis[f"gamma_{gamma}"] = {
            "effective_loss": float(weighted_loss),
            "easy_gradient_pct": float((focal_weight[easy_mask] * losses[easy_mask]).sum() / total_weight * 100),
            "medium_gradient_pct": float((focal_weight[med_mask] * losses[med_mask]).sum() / total_weight * 100),
            "hard_gradient_pct": float((focal_weight[hard_mask] * losses[hard_mask]).sum() / total_weight * 100),
        }
    report["focal_simulation"] = focal_analysis

    # 8. Where should we focus? Tokens where model is wrong but close
    # These are tokens with high loss but correct answer in top-10
    close_wrong = (~top1.astype(bool)) & top10.astype(bool)
    far_wrong = (~top1.astype(bool)) & (~top10.astype(bool))
    report["improvement_opportunity"] = {
        "correct_top1_pct": float(top1.mean() * 100),
        "wrong_but_in_top10_pct": float(close_wrong.mean() * 100),
        "wrong_not_in_top10_pct": float(far_wrong.mean() * 100),
        "close_wrong_mean_loss": float(losses[close_wrong].mean()) if close_wrong.sum() > 0 else 0,
        "far_wrong_mean_loss": float(losses[far_wrong].mean()) if far_wrong.sum() > 0 else 0,
        "insight": (
            "Tokens where model is wrong but answer is in top-10 are the highest-value "
            "targets for improvement. An auxiliary loss that sharpens the distribution "
            "around these tokens would directly improve BPB."
        ),
    }

    return report


def print_report(report: dict):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("MODEL ERROR ANALYSIS")
    print("=" * 70)

    o = report["overall"]
    print(f"\n--- Overall ---")
    print(f"  Mean loss: {o['mean_loss']:.4f}")
    print(f"  Top-1 accuracy: {o['top1_accuracy']:.1%}")
    print(f"  Top-5 accuracy: {o['top5_accuracy']:.1%}")
    print(f"  Top-10 accuracy: {o['top10_accuracy']:.1%}")
    print(f"  Mean confidence: {o['mean_confidence']:.3f}")
    print(f"  Mean entropy: {o['mean_entropy']:.3f}")

    d = report["difficulty_distribution"]
    print(f"\n--- Difficulty Distribution ---")
    print(f"  Easy  (loss < 1.0):  {d['easy_pct']:5.1f}%")
    print(f"  Medium (1-4):        {d['medium_pct']:5.1f}%")
    print(f"  Hard  (4-8):         {d['hard_pct']:5.1f}%")
    print(f"  Impossible (>= 8):   {d['impossible_pct']:5.1f}%")

    p = report["loss_percentiles"]
    print(f"\n--- Loss Percentiles ---")
    for k, v in p.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n--- Loss by Position ---")
    for pos_range, stats in report["loss_by_position"].items():
        print(f"  pos {pos_range}: loss={stats['mean_loss']:.4f} top1={stats['top1_acc']:.1%}")

    if "loss_by_token_type" in report:
        print(f"\n--- Loss by Token Type ---")
        sorted_types = sorted(report["loss_by_token_type"].items(),
                              key=lambda x: -x[1]["count"])
        for ttype, stats in sorted_types:
            print(f"  {ttype:<15} loss={stats['mean_loss']:.4f} "
                  f"top1={stats['top1_acc']:.1%} "
                  f"({stats['pct_of_total']:.1f}% of tokens)")

    print(f"\n--- Confidence Calibration ---")
    for conf_range, stats in report["confidence_calibration"].items():
        print(f"  conf {conf_range}: acc={stats['accuracy']:.1%} "
              f"({stats['pct_of_total']:.1f}% of tokens)")

    print(f"\n--- Per-Document Variance ---")
    doc = report["per_document"]
    print(f"  Mean doc loss: {doc['mean_doc_loss']:.4f} ± {doc['std_doc_loss']:.4f}")
    print(f"  Range: [{doc['min_doc_loss']:.4f}, {doc['max_doc_loss']:.4f}]")
    print(f"  P10-P90: [{doc['p10_doc_loss']:.4f}, {doc['p90_doc_loss']:.4f}]")

    print(f"\n--- Focal Loss Simulation ---")
    for gamma_key, stats in report["focal_simulation"].items():
        print(f"  {gamma_key}: effective_loss={stats['effective_loss']:.4f} "
              f"easy={stats['easy_gradient_pct']:.1f}% "
              f"med={stats['medium_gradient_pct']:.1f}% "
              f"hard={stats['hard_gradient_pct']:.1f}%")

    imp = report["improvement_opportunity"]
    print(f"\n--- Improvement Opportunity ---")
    print(f"  Correct (top-1):            {imp['correct_top1_pct']:.1f}%")
    print(f"  Wrong but in top-10:        {imp['wrong_but_in_top10_pct']:.1f}% (mean loss: {imp['close_wrong_mean_loss']:.4f})")
    print(f"  Wrong, not in top-10:       {imp['wrong_not_in_top10_pct']:.1f}% (mean loss: {imp['far_wrong_mean_loss']:.4f})")
    print(f"\n  {imp['insight']}")


def main():
    parser = argparse.ArgumentParser(description="Error analysis of trained Parameter Golf model")
    parser.add_argument("--model", default="final_model.pt",
                        help="Path to trained model checkpoint (default: final_model.pt)")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024",
                        help="Path to dataset directory")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to SentencePiece model")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sequences", type=int, default=500,
                        help="Max validation sequences to analyze")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Max validation tokens to load (0 = all)")
    parser.add_argument("--output", default="experiments/error_analysis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    print(f"Tokenizer: {sp.vocab_size()} tokens")

    # Load validation data
    val_tokens = load_validation_tokens(args.data_path, max_tokens=args.max_tokens)

    # Load model
    print(f"Loading model from {args.model}...")
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Run a baseline training first, or specify --model path")
        sys.exit(1)

    # Import GPT class from training script
    sys.path.insert(0, ".")
    from train_gpt_aux import GPT, Hyperparameters

    hp = Hyperparameters()
    model = GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        mtp_num_heads=getattr(hp, 'mtp_num_heads', 0),
        mtp_loss_weight=getattr(hp, 'mtp_loss_weight', 0.2),
        bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim,
        xsa_last_n=hp.xsa_last_n,
        rope_dims=hp.rope_dims,
        ln_scale=hp.ln_scale,
        dtg=getattr(hp, 'dtg_enabled', False),
        ve_enabled=hp.ve_enabled,
        ve_dim=hp.ve_dim,
        ve_layers=hp.ve_layers,
        gated_attention=getattr(hp, 'gated_attention', False),
        value_residual=getattr(hp, 'value_residual', False),
    ).to(device)

    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    # Filter out MTP heads if not in saved state
    model_keys = set(model.state_dict().keys())
    load_keys = set(state_dict.keys())
    missing = model_keys - load_keys
    if missing:
        print(f"  Warning: {len(missing)} missing keys (MTP heads etc.)")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Run analysis
    report = analyze_model(
        model, val_tokens, device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_sequences=args.max_sequences,
        sp=sp,
    )

    # Print report
    print_report(report)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()

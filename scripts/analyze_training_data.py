"""Analyze FineWeb training data to inform auxiliary loss design.

Computes statistics that directly map to loss function decisions:
1. Token frequency distribution (Zipf analysis) → unigram KL lambda
2. Bigram transition entropy → per-token predictability scores
3. Per-token difficulty distribution → focal loss gamma tuning
4. Document-level topic diversity → mixture entropy gap
5. Positional entropy profile → where in the sequence is prediction hardest?

Usage:
    python scripts/analyze_training_data.py [--shards N] [--output PATH]

Outputs a JSON report + printed summary.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

def load_tokens(pattern: str, max_shards: int = 1) -> np.ndarray:
    """Load token IDs from binary shard files."""
    files = sorted(glob.glob(pattern))[:max_shards]
    if not files:
        print(f"ERROR: No files matching {pattern}")
        sys.exit(1)
    arrays = []
    for f in files:
        tokens = np.memmap(f, dtype=np.uint16, mode="r")
        arrays.append(np.array(tokens))  # copy to memory
        print(f"  Loaded {f}: {len(tokens):,} tokens")
    return np.concatenate(arrays)


def zipf_analysis(tokens: np.ndarray, vocab_size: int) -> dict:
    """Analyze token frequency distribution and Zipf fit."""
    actual_vocab = int(tokens.max()) + 1
    counts = np.bincount(tokens, minlength=actual_vocab).astype(np.float64)
    total = counts.sum()
    probs = counts / total

    # Sort descending
    sorted_counts = np.sort(counts)[::-1]
    sorted_probs = sorted_counts / total

    # Entropy of unigram distribution
    nonzero = probs[probs > 0]
    entropy = -(nonzero * np.log2(nonzero)).sum()

    # Zipf exponent estimate (log-log regression on top 500 tokens)
    top_n = min(500, len(sorted_counts))
    ranks = np.arange(1, top_n + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_counts[:top_n].clip(min=1))
    # Linear regression: log(freq) = -alpha * log(rank) + c
    A = np.stack([log_ranks, np.ones(top_n)], axis=1)
    result = np.linalg.lstsq(A, log_freqs, rcond=None)
    alpha = -result[0][0]

    # Coverage: what fraction of tokens do top-K account for?
    cumsum = np.cumsum(sorted_probs)
    coverage_10 = float(cumsum[9]) if len(cumsum) >= 10 else 0.0
    coverage_50 = float(cumsum[49]) if len(cumsum) >= 50 else 0.0
    coverage_100 = float(cumsum[99]) if len(cumsum) >= 100 else 0.0
    coverage_500 = float(cumsum[499]) if len(cumsum) >= 500 else 0.0

    # Tokens with zero occurrences (out of all possible IDs)
    zero_count = int((counts == 0).sum())
    n_unique = int((counts > 0).sum())

    return {
        "entropy_bits": float(entropy),
        "zipf_alpha": float(alpha),
        "top10_coverage": coverage_10,
        "top50_coverage": coverage_50,
        "top100_coverage": coverage_100,
        "top500_coverage": coverage_500,
        "zero_count_tokens": zero_count,
        "unique_tokens": n_unique,
        "actual_vocab_size": actual_vocab,
        "nominal_vocab_size": vocab_size,
        "total_tokens": int(total),
        "most_common_10": [(int(i), int(sorted_counts[i - 1])) for i in range(1, 11)],
    }


def bigram_entropy(tokens: np.ndarray, vocab_size: int, sample_size: int = 5_000_000) -> dict:
    """Compute per-token conditional entropy from bigram statistics.

    H(next | prev) for each prev token — tells us how predictable
    the next token is given just the previous one.
    """
    # Sample for speed
    if len(tokens) > sample_size:
        start = np.random.randint(0, len(tokens) - sample_size)
        tokens = tokens[start:start + sample_size]

    # Count bigrams
    prev = tokens[:-1]
    curr = tokens[1:]

    # Per-token conditional entropy
    bigram_counts = defaultdict(Counter)
    for p, c in zip(prev, curr):
        bigram_counts[int(p)][int(c)] += 1

    conditional_entropies = {}
    for tok, next_counts in bigram_counts.items():
        total = sum(next_counts.values())
        h = 0.0
        for count in next_counts.values():
            p = count / total
            if p > 0:
                h -= p * math.log2(p)
        conditional_entropies[tok] = h

    entropies = list(conditional_entropies.values())
    if not entropies:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

    arr = np.array(entropies)

    # Weighted mean (by token frequency)
    unigram_counts = np.bincount(tokens, minlength=vocab_size).astype(np.float64)
    total = unigram_counts.sum()
    weighted_entropy = 0.0
    for tok, h in conditional_entropies.items():
        weighted_entropy += (unigram_counts[tok] / total) * h

    # Difficulty buckets: what fraction of tokens are easy/medium/hard to predict?
    easy = sum(1 for h in entropies if h < 2.0)
    medium = sum(1 for h in entropies if 2.0 <= h < 6.0)
    hard = sum(1 for h in entropies if h >= 6.0)
    n = len(entropies)

    return {
        "unweighted_mean": float(arr.mean()),
        "weighted_mean": float(weighted_entropy),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "easy_frac": easy / n,  # H < 2 bits
        "medium_frac": medium / n,  # 2 <= H < 6 bits
        "hard_frac": hard / n,  # H >= 6 bits
        "n_tokens_observed": n,
    }


def positional_entropy(tokens: np.ndarray, seq_len: int = 2048, n_seqs: int = 5000) -> dict:
    """Measure how token entropy varies by position in the sequence.

    Early positions may be more predictable (document headers, common openings)
    while later positions may be more diverse.
    """
    # Reshape into sequences
    n_available = len(tokens) // seq_len
    n_seqs = min(n_seqs, n_available)
    seqs = tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)

    # Per-position token distribution entropy
    position_entropies = []
    for pos in range(seq_len):
        col = seqs[:, pos]
        counts = np.bincount(col, minlength=1024).astype(np.float64)
        probs = counts / counts.sum()
        nonzero = probs[probs > 0]
        h = -(nonzero * np.log2(nonzero)).sum()
        position_entropies.append(float(h))

    arr = np.array(position_entropies)

    # Summarize by chunks
    chunk_size = seq_len // 8
    chunks = []
    for i in range(8):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunks.append({
            "positions": f"{start}-{end-1}",
            "mean_entropy": float(arr[start:end].mean()),
        })

    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "pos0_entropy": position_entropies[0],
        "pos1_entropy": position_entropies[1],
        "mid_entropy": float(arr[seq_len // 2]),
        "last_entropy": position_entropies[-1],
        "chunks": chunks,
    }


def document_diversity(tokens: np.ndarray, seq_len: int = 2048, n_docs: int = 2000, vocab_size: int = 1024) -> dict:
    """Estimate topic diversity via per-document unigram divergence.

    If all documents have similar token distributions, mixture entropy gap is small.
    If documents are topically diverse, the gap is large → topic-aware losses may help.
    """
    n_available = len(tokens) // seq_len
    n_docs = min(n_docs, n_available)
    seqs = tokens[:n_docs * seq_len].reshape(n_docs, seq_len)

    # Global distribution — use actual max token ID for consistent array size
    actual_vocab = int(tokens[:n_docs * seq_len].max()) + 1
    bin_size = max(vocab_size, actual_vocab)
    global_counts = np.bincount(tokens[:n_docs * seq_len], minlength=bin_size).astype(np.float64)
    global_probs = global_counts / global_counts.sum()

    # Per-document KL divergence from global
    kl_divs = []
    doc_entropies = []
    for i in range(n_docs):
        doc_counts = np.bincount(seqs[i], minlength=bin_size).astype(np.float64)
        doc_probs = doc_counts / doc_counts.sum()

        # Document entropy
        nonzero = doc_probs[doc_probs > 0]
        h = -(nonzero * np.log2(nonzero)).sum()
        doc_entropies.append(h)

        # KL(doc || global) — how different is this doc from the average?
        mask = (doc_probs > 0) & (global_probs > 0)
        if mask.any():
            kl = (doc_probs[mask] * np.log2(doc_probs[mask] / global_probs[mask])).sum()
            kl_divs.append(float(kl))

    doc_h = np.array(doc_entropies)
    kl_arr = np.array(kl_divs) if kl_divs else np.array([0.0])

    # Mixture entropy gap = H(global) - mean(H(doc))
    global_nonzero = global_probs[global_probs > 0]
    global_entropy = -(global_nonzero * np.log2(global_nonzero)).sum()
    mean_doc_entropy = float(doc_h.mean())
    mixture_gap = float(global_entropy) - mean_doc_entropy

    return {
        "global_entropy_bits": float(global_entropy),
        "mean_doc_entropy_bits": mean_doc_entropy,
        "mixture_entropy_gap": mixture_gap,
        "mixture_gap_ratio": mixture_gap / float(global_entropy) if global_entropy > 0 else 0,
        "mean_kl_from_global": float(kl_arr.mean()),
        "std_kl_from_global": float(kl_arr.std()),
        "max_kl_from_global": float(kl_arr.max()),
        "doc_entropy_std": float(doc_h.std()),
        "n_docs": n_docs,
    }


def loss_recommendations(report: dict) -> list[str]:
    """Generate concrete recommendations from the statistics."""
    recs = []

    # Focal loss
    easy = report["bigram_entropy"]["easy_frac"]
    hard = report["bigram_entropy"]["hard_frac"]
    if easy > 0.3:
        recs.append(
            f"FOCAL LOSS: {easy:.0%} of tokens are easy (bigram H < 2 bits). "
            f"Focal loss with gamma=2.0 will significantly downweight these, "
            f"redirecting gradients to the {report['bigram_entropy']['medium_frac']:.0%} medium-difficulty tokens."
        )
    if hard > 0.3:
        recs.append(
            f"FOCAL LOSS: {hard:.0%} of tokens are hard (bigram H >= 6 bits). "
            f"Consider gamma=1.0 (gentler) — too aggressive downweighting may "
            f"waste capacity on truly unpredictable tokens."
        )

    # Unigram KL
    zipf = report["zipf_analysis"]
    recs.append(
        f"UNIGRAM KL: Top-100 tokens cover {zipf['top100_coverage']:.1%} of all text. "
        f"Unigram entropy = {zipf['entropy_bits']:.2f} bits. "
        f"Lambda_unigram=0.1 with decay to zero by 50% training should help the model "
        f"learn these base rates faster."
    )

    # Decorrelation
    recs.append(
        f"DECORRELATION: With 11 layers and {zipf['entropy_bits']:.1f}-bit unigram entropy, "
        f"the model has ~{11 * 512:,} hidden dimensions to learn ~{zipf['entropy_bits']:.0f} bits of base-rate info. "
        f"Layer redundancy is plausible — decorrelation loss should help."
    )

    # Mixture entropy
    mix = report["document_diversity"]
    if mix["mixture_gap_ratio"] > 0.05:
        recs.append(
            f"TOPIC DIVERSITY: Mixture entropy gap = {mix['mixture_entropy_gap']:.3f} bits "
            f"({mix['mixture_gap_ratio']:.1%} of global entropy). Documents are topically diverse. "
            f"Topic-aware or difficulty-weighted loss could exploit this structure."
        )
    else:
        recs.append(
            f"TOPIC DIVERSITY: Mixture entropy gap = {mix['mixture_entropy_gap']:.3f} bits "
            f"({mix['mixture_gap_ratio']:.1%}) — documents are relatively uniform. "
            f"Topic-aware losses unlikely to help much."
        )

    # Positional
    pos = report["positional_entropy"]
    if abs(pos["pos0_entropy"] - pos["last_entropy"]) > 0.5:
        recs.append(
            f"POSITIONAL: Entropy varies from {pos['pos0_entropy']:.2f} bits (pos 0) to "
            f"{pos['last_entropy']:.2f} bits (last pos). "
            f"Position-dependent loss weighting could help but adds complexity."
        )

    return recs


def main():
    parser = argparse.ArgumentParser(description="Analyze FineWeb training data for loss design")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024",
                        help="Path to dataset directory")
    parser.add_argument("--shards", type=int, default=1, help="Number of training shards to analyze")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Vocabulary size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--output", default="experiments/data_analysis.json", help="Output JSON path")
    args = parser.parse_args()

    train_pattern = os.path.join(args.data_path, "fineweb_train_*.bin")
    print(f"Loading training data from {train_pattern} ({args.shards} shards)...")
    tokens = load_tokens(train_pattern, max_shards=args.shards)
    print(f"Total tokens: {len(tokens):,}\n")

    print("=" * 60)
    print("1. ZIPF / TOKEN FREQUENCY ANALYSIS")
    print("=" * 60)
    zipf = zipf_analysis(tokens, args.vocab_size)
    print(f"  Unigram entropy: {zipf['entropy_bits']:.3f} bits")
    print(f"  Zipf exponent (alpha): {zipf['zipf_alpha']:.3f}")
    print(f"  Top-10 coverage: {zipf['top10_coverage']:.1%}")
    print(f"  Top-50 coverage: {zipf['top50_coverage']:.1%}")
    print(f"  Top-100 coverage: {zipf['top100_coverage']:.1%}")
    print(f"  Top-500 coverage: {zipf['top500_coverage']:.1%}")
    print(f"  Unique tokens: {zipf['unique_tokens']}/{zipf['actual_vocab_size']} (nominal vocab: {zipf['nominal_vocab_size']})")
    print()

    print("=" * 60)
    print("2. BIGRAM CONDITIONAL ENTROPY (per-token difficulty)")
    print("=" * 60)
    bigram = bigram_entropy(tokens, args.vocab_size)
    print(f"  Weighted mean H(next|prev): {bigram['weighted_mean']:.3f} bits")
    print(f"  Unweighted mean: {bigram['unweighted_mean']:.3f} bits")
    print(f"  Median: {bigram['median']:.3f} bits")
    print(f"  Range: [{bigram['min']:.3f}, {bigram['max']:.3f}]")
    print(f"  P10-P90: [{bigram['p10']:.3f}, {bigram['p90']:.3f}]")
    print(f"  Easy tokens (H<2): {bigram['easy_frac']:.1%}")
    print(f"  Medium tokens (2<=H<6): {bigram['medium_frac']:.1%}")
    print(f"  Hard tokens (H>=6): {bigram['hard_frac']:.1%}")
    print()

    print("=" * 60)
    print("3. POSITIONAL ENTROPY PROFILE")
    print("=" * 60)
    pos = positional_entropy(tokens, seq_len=args.seq_len)
    print(f"  Mean positional entropy: {pos['mean']:.3f} bits")
    print(f"  Position 0: {pos['pos0_entropy']:.3f} bits")
    print(f"  Position 1: {pos['pos1_entropy']:.3f} bits")
    print(f"  Mid-sequence: {pos['mid_entropy']:.3f} bits")
    print(f"  Last position: {pos['last_entropy']:.3f} bits")
    print(f"  By chunk:")
    for chunk in pos["chunks"]:
        print(f"    {chunk['positions']}: {chunk['mean_entropy']:.3f} bits")
    print()

    print("=" * 60)
    print("4. DOCUMENT DIVERSITY / MIXTURE ENTROPY")
    print("=" * 60)
    diversity = document_diversity(tokens, seq_len=args.seq_len, vocab_size=args.vocab_size)
    print(f"  Global entropy: {diversity['global_entropy_bits']:.3f} bits")
    print(f"  Mean per-doc entropy: {diversity['mean_doc_entropy_bits']:.3f} bits")
    print(f"  Mixture entropy gap: {diversity['mixture_entropy_gap']:.3f} bits ({diversity['mixture_gap_ratio']:.1%})")
    print(f"  Mean KL(doc||global): {diversity['mean_kl_from_global']:.3f} bits")
    print(f"  Max KL(doc||global): {diversity['max_kl_from_global']:.3f} bits")
    print(f"  Doc entropy std: {diversity['doc_entropy_std']:.3f} bits")
    print()

    report = {
        "zipf_analysis": zipf,
        "bigram_entropy": bigram,
        "positional_entropy": pos,
        "document_diversity": diversity,
    }

    print("=" * 60)
    print("5. LOSS FUNCTION RECOMMENDATIONS")
    print("=" * 60)
    recs = loss_recommendations(report)
    report["recommendations"] = recs
    for i, rec in enumerate(recs, 1):
        print(f"\n  [{i}] {rec}")
    print()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to {args.output}")


if __name__ == "__main__":
    main()

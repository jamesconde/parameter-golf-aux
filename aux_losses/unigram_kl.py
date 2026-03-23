"""Unigram prior KL loss for Parameter Golf.

Regularizes the model's average output distribution toward the known
marginal token frequency, providing free base-rate information.
"""
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def compute_unigram_distribution(
    train_files_pattern: str, vocab_size: int
) -> Tensor:
    """Count token frequencies across training shards.

    Args:
        train_files_pattern: glob pattern for training .bin files
        vocab_size: number of tokens in vocabulary

    Returns:
        [vocab_size] tensor of log-probabilities
    """
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    files = sorted(glob.glob(train_files_pattern))
    for f in files:
        tokens = np.memmap(f, dtype=np.uint16, mode="r")
        # Vectorized counting — filter to valid vocab range
        valid = tokens[tokens < vocab_size]
        token_ids, token_counts = np.unique(valid, return_counts=True)
        for tid, cnt in zip(token_ids, token_counts):
            counts[int(tid)] += cnt

    # Normalize with Laplace smoothing
    probs = (counts + 1.0) / (counts.sum() + vocab_size)
    return probs.float().log()


def unigram_kl_loss(logits: Tensor, unigram_log_probs: Tensor) -> Tensor:
    """KL divergence between model's average prediction and unigram prior.

    Args:
        logits: [N, vocab_size] raw logits
        unigram_log_probs: [vocab_size] precomputed log(unigram_probs)
    """
    # Average the model's predicted distribution across the batch
    mean_log_probs = F.log_softmax(logits.float(), dim=-1).mean(dim=0)

    # KL(unigram || model_avg) = sum(p_unigram * (log_p_unigram - log_p_model))
    # Compute manually since F.kl_div reduction="batchmean" is invalid for 1D
    unigram_probs = unigram_log_probs.exp()
    kl = (unigram_probs * (unigram_log_probs - mean_log_probs)).sum()
    return kl

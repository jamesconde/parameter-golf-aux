"""Representation rank loss for Parameter Golf.

Encourages the model to use its full representational bandwidth
by penalizing low effective rank of hidden state distributions.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def representation_rank_loss(
    hidden_states: Tensor,
    target_effective_rank_ratio: float = 0.8,
    subsample: int = 256,
) -> Tensor:
    """Penalize when effective rank of hidden states is below target.

    Uses eigenvalue entropy of the covariance matrix as a differentiable
    proxy for effective rank.

    Args:
        hidden_states: [batch, seq_len, dim] from one layer
        target_effective_rank_ratio: target ratio of effective_rank / dim
        subsample: max number of vectors to use (for speed)
    """
    flat = hidden_states.reshape(-1, hidden_states.size(-1))

    if flat.size(0) > subsample:
        idx = torch.randint(0, flat.size(0), (subsample,), device=flat.device)
        flat = flat[idx]

    # Must compute in float32 for numerical stability
    with torch.autocast(device_type="cuda", enabled=False):
        flat = flat.float()
        flat = flat - flat.mean(dim=0, keepdim=True)

        # Covariance matrix
        cov = (flat.T @ flat) / flat.size(0)  # [dim, dim]

        # Eigenvalues of symmetric matrix (faster than full SVD)
        eigvals = torch.linalg.eigvalsh(cov)
        eigvals = eigvals.clamp(min=1e-8)

        # Effective rank = exp(entropy of normalized eigenvalue distribution)
        p = eigvals / eigvals.sum()
        entropy = -(p * p.log()).sum()
        effective_rank = torch.exp(entropy)

        max_rank = float(hidden_states.size(-1))
        rank_ratio = effective_rank / max_rank

        # Penalize when below target
        penalty = F.relu(target_effective_rank_ratio - rank_ratio)

    return penalty

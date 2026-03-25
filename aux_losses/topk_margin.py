"""Top-K margin loss for Parameter Golf.

Targets the 27.8% of tokens where the model has the correct answer
in its top-K predictions but can't rank it #1. Two complementary
approaches:

1. Margin ranking: directly penalizes the logit gap between the
   correct token and the model's top prediction.

2. Close-wrong boost: upweights the CE loss on tokens where the
   model is close-but-wrong, causing the optimizer to reallocate
   capacity from hopeless tokens to improvable ones.

Designed from error analysis showing:
- 55.0% correct (top-1) — leave these alone
- 27.8% close wrong (in top-10, not top-1) — TARGET THESE
- 17.3% far wrong (not in top-10) — can't help, don't waste capacity
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def topk_margin_loss(
    logits: Tensor,
    targets: Tensor,
    k: int = 10,
    margin: float = 1.0,
) -> Tensor:
    """Margin ranking loss between correct token and top-1 prediction.

    Only applies to tokens where correct answer is in top-K but not top-1.
    Penalizes: max(0, margin - (logit_correct - logit_top1))

    Args:
        logits: [N, vocab_size] raw logits
        targets: [N] target token ids
        k: consider correct "close" if in top-K
        margin: desired logit gap (higher = more aggressive)
    """
    # Logit of correct token
    correct_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Top-1 logit (the model's best guess)
    top1_logits = logits.max(dim=-1).values

    # Find where correct is in top-K but not top-1
    _, top_k_indices = logits.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1)
    in_top1 = (top_k_indices[:, :1] == targets_expanded).any(dim=-1)
    in_topk = (top_k_indices == targets_expanded).any(dim=-1)
    close_wrong = in_topk & ~in_top1

    # Margin loss: want correct_logit to be within `margin` of top1_logit
    gap = margin - (correct_logits - top1_logits)
    margin_penalty = F.relu(gap)

    # Only apply to close-wrong tokens
    margin_penalty = margin_penalty * close_wrong.float()

    # Normalize by number of close-wrong tokens (not total tokens)
    n_close = close_wrong.float().sum().clamp(min=1.0)
    return margin_penalty.sum() / n_close


def close_wrong_boost_loss(
    logits: Tensor,
    targets: Tensor,
    k: int = 10,
    boost: float = 2.0,
) -> Tensor:
    """CE loss with boosted weight on close-wrong tokens.

    Tokens where correct is in top-K but not top-1 get `boost`x weight.
    All other tokens get 1x weight. This causes the optimizer to
    reallocate capacity from hopeless tokens to improvable ones.

    Args:
        logits: [N, vocab_size] raw logits
        targets: [N] target token ids
        k: consider correct "close" if in top-K
        boost: multiplier on CE for close-wrong tokens
    """
    ce = F.cross_entropy(logits, targets, reduction='none')

    # Find close-wrong tokens
    _, top_k_indices = logits.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1)
    in_top1 = (top_k_indices[:, :1] == targets_expanded).any(dim=-1)
    in_topk = (top_k_indices == targets_expanded).any(dim=-1)
    close_wrong = in_topk & ~in_top1

    # Weight: boost for close-wrong, 1.0 for everything else
    weight = torch.ones_like(ce)
    weight[close_wrong] = boost

    return (weight * ce).sum() / weight.sum()

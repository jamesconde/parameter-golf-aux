"""Focal cross-entropy loss for Parameter Golf.

Downweights well-classified tokens so gradient signal focuses on
medium-difficulty tokens where the model can actually improve.
gamma=0 reduces to standard cross-entropy.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Focal loss variant of cross-entropy.

    Args:
        logits: [N, vocab_size] raw logits
        targets: [N] target token ids
        gamma: focusing parameter. 0 = standard CE, 2 = standard focal
        label_smoothing: label smoothing epsilon
    """
    ce_loss = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing
    )

    # p_correct = probability assigned to the correct class
    log_probs = F.log_softmax(logits, dim=-1)
    p_correct = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))

    # Focal weight: (1 - p_correct)^gamma
    focal_weight = (1.0 - p_correct.detach()) ** gamma

    return (focal_weight * ce_loss).mean()

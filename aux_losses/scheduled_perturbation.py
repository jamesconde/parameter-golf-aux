"""Scheduled loss perturbations for faster convergence.

These modify the loss AFTER the compiled model's forward() returns,
so they work with fullgraph=True and add zero compilation overhead.

Designed from training curve analysis showing:
- Steps 0-500: rapid descent (91.5% of learning)
- Steps 500-3500: plateau with oscillation (only 6.6% of learning)
- Steps 3500+: warmdown (1.8% of learning)

The plateau is where we lose the most potential. These perturbations
aim to help the model converge faster through the plateau by:
1. Smoothing the loss landscape (scheduled label smoothing)
2. Adding exploration noise (gradient noise injection)
3. Dampening hopeless tokens (loss truncation)
"""
import math
import torch
from torch import Tensor


class LossScheduler:
    """Applies scheduled perturbations to the loss based on training progress.

    All perturbations modify the scalar loss AFTER model.forward() returns,
    so they're compatible with torch.compile(fullgraph=True).

    Usage:
        scheduler = LossScheduler(config)
        for step in range(total_steps):
            loss = model(x, y)
            loss = scheduler.perturb(loss, step, total_steps)
            loss.backward()
    """

    def __init__(
        self,
        # Scheduled label smoothing: softens targets during plateau
        label_smoothing_peak: float = 0.0,     # 0 = disabled, try 0.05-0.1
        label_smoothing_start_frac: float = 0.05,  # start at 5% of training
        label_smoothing_end_frac: float = 0.5,     # end at 50% of training
        vocab_size: int = 1024,

        # Loss truncation: cap per-token loss to ignore hopeless tokens
        # (requires per-token loss, so this modifies CE computation)
        loss_truncation_max: float = 0.0,  # 0 = disabled, try 6.0-8.0

        # Gradient noise: Langevin-style exploration during plateau
        grad_noise_scale: float = 0.0,     # 0 = disabled, try 0.001-0.01
        grad_noise_start_frac: float = 0.05,
        grad_noise_end_frac: float = 0.5,
    ):
        self.label_smoothing_peak = label_smoothing_peak
        self.label_smoothing_start_frac = label_smoothing_start_frac
        self.label_smoothing_end_frac = label_smoothing_end_frac
        self.vocab_size = vocab_size
        self.loss_truncation_max = loss_truncation_max
        self.grad_noise_scale = grad_noise_scale
        self.grad_noise_start_frac = grad_noise_start_frac
        self.grad_noise_end_frac = grad_noise_end_frac

    def _schedule_weight(self, step: int, total_steps: int,
                         start_frac: float, end_frac: float) -> float:
        """Triangular schedule: ramp up from start, peak in middle, ramp down to end."""
        progress = step / max(total_steps, 1)
        if progress < start_frac or progress > end_frac:
            return 0.0
        mid = (start_frac + end_frac) / 2
        if progress < mid:
            # Ramp up
            return (progress - start_frac) / (mid - start_frac)
        else:
            # Ramp down
            return (end_frac - progress) / (end_frac - mid)

    def perturb(self, loss: Tensor, step: int, total_steps: int) -> Tensor:
        """Apply scheduled perturbations to the loss."""

        # 1. Scheduled label smoothing (post-hoc approximation)
        # Standard CE = -log(p_correct)
        # Smoothed CE ≈ (1-α)*CE + α*log(V) where V=vocab_size
        # This softens the loss during plateau, removed before warmdown
        if self.label_smoothing_peak > 0:
            alpha = self.label_smoothing_peak * self._schedule_weight(
                step, total_steps,
                self.label_smoothing_start_frac,
                self.label_smoothing_end_frac,
            )
            if alpha > 0:
                uniform_loss = math.log(self.vocab_size)
                loss = (1.0 - alpha) * loss + alpha * uniform_loss

        # 2. Gradient noise injection (Langevin dynamics)
        # Adds exploration during plateau to help escape local minima
        # Noise scales with the loss magnitude so it's proportional
        if self.grad_noise_scale > 0:
            noise_weight = self.grad_noise_scale * self._schedule_weight(
                step, total_steps,
                self.grad_noise_start_frac,
                self.grad_noise_end_frac,
            )
            if noise_weight > 0:
                noise = torch.randn(1, device=loss.device, dtype=loss.dtype) * noise_weight
                loss = loss + loss.detach() * noise  # Scale noise by loss magnitude

        return loss


def truncated_ce_loss(logits: Tensor, targets: Tensor, max_loss: float = 6.0) -> Tensor:
    """Cross-entropy with per-token loss capped at max_loss.

    Tokens with loss > max_loss are "hopeless" — their gradients are
    pure noise that slows convergence. Truncating caps the gradient
    contribution of these tokens, reducing noise during the plateau.

    From error analysis: 17.3% of tokens have loss > 4.0, and 1.1% > 8.0.
    Setting max_loss=6.0 caps the noisiest ~5% of tokens.

    This IS a loss function change but requires logits, so it needs
    forward_aux() OR can be integrated directly into forward().
    """
    per_token_loss = torch.nn.functional.cross_entropy(
        logits, targets, reduction='none'
    )
    # Soft truncation (smooth gradient at boundary)
    truncated = torch.where(
        per_token_loss < max_loss,
        per_token_loss,
        max_loss + torch.log1p(per_token_loss - max_loss),  # Log-dampened tail
    )
    return truncated.mean()

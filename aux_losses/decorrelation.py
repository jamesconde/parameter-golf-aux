"""Inter-layer decorrelation loss for Parameter Golf.

Forces adjacent transformer layers to learn different representations,
preventing redundant computation in a small (10-11 layer) model.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def inter_layer_decorrelation_loss(
    hidden_states_list: list[Tensor],
    sample_size: int = 64,
) -> Tensor:
    """Penalize high cosine similarity between adjacent layer hidden states.

    Args:
        hidden_states_list: list of [batch, seq_len, dim] tensors from each layer
        sample_size: number of random (batch*seq) positions to sample for speed
    """
    loss = torch.tensor(0.0, device=hidden_states_list[0].device)
    n_pairs = 0

    for i in range(len(hidden_states_list) - 1):
        h_i = hidden_states_list[i]  # [batch, seq, dim]
        h_j = hidden_states_list[i + 1]

        # Flatten to [batch*seq, dim], sample random subset for speed
        flat_i = h_i.reshape(-1, h_i.size(-1))
        flat_j = h_j.reshape(-1, h_j.size(-1))

        if flat_i.size(0) > sample_size:
            idx = torch.randint(
                0, flat_i.size(0), (sample_size,), device=flat_i.device
            )
            flat_i = flat_i[idx]
            flat_j = flat_j[idx]

        # Cosine similarity — we want this LOW (layers should differ)
        with torch.autocast(device_type="cuda", enabled=False):
            cos_sim = F.cosine_similarity(
                flat_i.float(), flat_j.float(), dim=-1
            ).mean()
        loss = loss + cos_sim
        n_pairs += 1

    return loss / max(n_pairs, 1)

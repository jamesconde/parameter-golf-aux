"""Auxiliary loss modules for Parameter Golf."""
from aux_losses.decorrelation import inter_layer_decorrelation_loss
from aux_losses.rank_loss import representation_rank_loss
from aux_losses.focal_loss import focal_cross_entropy
from aux_losses.unigram_kl import unigram_kl_loss, compute_unigram_distribution
from aux_losses.topk_margin import topk_margin_loss, close_wrong_boost_loss

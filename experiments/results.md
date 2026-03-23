# Parameter Golf — Experiment Log

## Base Recipe
- **Forked from:** thwu1's "10L Int5-MLP + BigramHash(10240)" (1.1428 BPB, merged 2026-03-20)
- **Techniques included:** 10L (5E+5D), int5 MLP / int6 attn, SmearGate, BigramHash(10240), OrthoInit, Muon WD=0.04, SWA(0.4), magnitude pruning, zstd-22, sliding window eval stride=64
- **File:** `train_gpt_sota.py` (1231 lines, 52930 bytes)
- **Our modified file:** `train_gpt_aux.py` (adds ~60 lines for aux loss integration)

## Smoke Test (WSL, RTX 4060 8GB)
- **Date:** 2026-03-22
- **Config:** 30 iterations, batch_tokens=32768, decorr_lambda=0.01
- **Result:** Training loop runs correctly, aux_loss logged at each step
- **Note:** Eval too slow for local testing (969K sliding windows). Real experiments on Colab/RunPod.

## Experiment Results

### Phase 1: Individual Auxiliary Loss Testing
_Run each on Colab/RunPod with 500 iterations, 3 seeds (42, 1337, 7)_

| Experiment | Lambda | val_bpb (mean±std) | Delta vs Baseline | Notes |
|-----------|--------|-------------------|-------------------|-------|
| SOTA baseline (no aux) | — | TBD | — | |
| Focal loss (gamma=1.0) | — | TBD | TBD | |
| Focal loss (gamma=2.0) | — | TBD | TBD | |
| Focal loss (gamma=3.0) | — | TBD | TBD | |
| Decorrelation | 0.001 | TBD | TBD | |
| Decorrelation | 0.01 | TBD | TBD | |
| Decorrelation | 0.05 | TBD | TBD | |
| Decorrelation | 0.1 | TBD | TBD | |
| Rank loss | 0.01 | TBD | TBD | |
| Rank loss | 0.05 | TBD | TBD | |
| Rank loss | 0.1 | TBD | TBD | |
| Unigram KL | 0.01 | TBD | TBD | |
| Unigram KL | 0.05 | TBD | TBD | |
| Unigram KL | 0.1 | TBD | TBD | |

### Phase 2: Stacking
_Combine auxiliary losses that individually helped_

| Combination | val_bpb | Delta vs Baseline | Notes |
|------------|---------|-------------------|-------|
| TBD | TBD | TBD | |

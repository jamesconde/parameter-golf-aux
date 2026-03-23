# Parameter Golf Leaderboard Analysis (Updated 2026-03-23)

## Current Top 5 (as of 2026-03-23)

| Rank | Entry Name | BPB | Author | Date | Key New Techniques |
|------|-----------|-----|--------|------|--------------------|
| 1 | 11L EMA + GPTQ-lite + warmdown3500 | 1.1228 | signalrush | 03-22 | GPTQ-lite clip search, EMA decay=0.997, warmdown3500, QAT@0.15 |
| 2 | 11L Partial RoPE + LN Scale + EMA + XSA4 | 1.1248 | jfprincz | 03-21 | Partial RoPE (16/64), layerwise LN scale |
| 3 | 11L XSA4 + EMA + Int6 MLP3x | 1.1271 | jfprincz | 03-20 | XSA on last 4 layers, EMA replacing SWA |
| 4 | 11L Efficient Partial XSA | 1.1307 | unnir | 03-20 | Efficient Partial XSA on deepest 3 layers |
| 5 | 10L Int5-MLP + BigramHash(10240) | 1.1428 | thwu1 | 03-20 | Mixed int5/int6, BigramHash(10240) |

**SOTA moved from 1.1428 → 1.1228 (0.020 BPB improvement) in 2 days.**

## New Techniques Since Initial Analysis (03-22)

- **XSA (Cross-Sequence Attention)**: Appears in 3 of top 4. Applied to last 3-4 layers.
- **EMA (Exponential Moving Average)**: Replaces/complements SWA. decay=0.997, every step.
- **GPTQ-lite**: Per-row optimal clip percentile search (5 candidates) for int6 quantization. Zero training cost.
- **Partial RoPE**: Only 16 of 64 dims get rotary embeddings.
- **Layerwise LN Scale**: 1/sqrt(layer_idx+1) normalization scaling.
- **Shared Value Embedding**: dim=128, applied to last 2 layers with per-layer scales.
- **Late QAT threshold**: QAT@0.15 (earlier fake quantization start).
- **11 layers now dominant** (all top 4 use 11L).

## Forking Decision (Updated)

**Re-forked from new #1 (signalrush, 1.1228 BPB, 2026-03-22).**

Rationale: 0.020 BPB gap to our previous fork is too large to ignore. The new #1 includes all previous techniques PLUS XSA, EMA, GPTQ-lite, Partial RoPE, LN Scale, and Shared Value Embedding.

**Still no loss function modifications in ANY submission.** Our approach remains the only one exploring this dimension.

## Previous Analysis (03-22, for reference)

### Comparison Matrix: Previous Top 5

| Rank | Entry Name | BPB | Layers | MLP | Quant | Eval Trick | Novel Modules | Loss Mods | Built On |
|------|-----------|-----|--------|-----|-------|------------|---------------|-----------|----------|
| 1 | 10L Int5-MLP + BigramHash(10240) | 1.1428 | 10 (5E+5D) | 3x (1536) | Int5 MLP / Int6 attn | Sliding stride=64 | SmearGate, BigramHash(10240), SWA(0.4), MagPrune | None | PR#162 |
| 2 | Int6 MLP3x + SmearGate + BigramHash | 1.1458 | 9 (4E+5D) | 3x (1536) | Int6 all blocks | Sliding stride=64 | SmearGate, BigramHash(4096), OrthoInit, SWA(0.5) | None | Baseline |
| 3 | 11L MLP3x + Int6 QAT | 1.1502 | 11 | 3x (1536) | Int6 STE QAT | Sliding stride=64 | None novel | None | Baseline + QAT |
| 4 | SmearGate + OrthoInit + Muon WD | 1.1556 | 9 (4E+5D) | 3x (1536) | Int6 STE QAT | Sliding stride=64 | SmearGate, BigramHash(4096), OrthoInit | None | Baseline |
| 5 | 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | 10 | 2.625x (1344) | Int6 STE QAT | Sliding stride=64 | None novel | None | Baseline |

### Key Questions (still valid)

**Q1: Single dominant recipe?** YES — even more converged now. All top entries share: 11L, Int6 QAT, 3x MLP, SmearGate, BigramHash, Muon+WD, sliding eval, zstd-22. Top entries add XSA + EMA + GPTQ-lite.

**Q4: Has ANYONE modified the loss function?** Still NO. Our approach remains genuinely novel and uncontested.

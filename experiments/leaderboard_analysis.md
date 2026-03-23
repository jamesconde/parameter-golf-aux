# Parameter Golf Leaderboard Analysis (2026-03-22)

## Comparison Matrix: Top 5 Merged Entries

| Rank | Entry Name | BPB | Layers | MLP | Quant | Eval Trick | Novel Modules | Loss Mods | Built On |
|------|-----------|-----|--------|-----|-------|------------|---------------|-----------|----------|
| 1 | 10L Int5-MLP + BigramHash(10240) | 1.1428 | 10 (5E+5D) | 3x (1536) | Int5 MLP / Int6 attn | Sliding stride=64 | SmearGate, BigramHash(10240), SWA(0.4), MagPrune | None | PR#162 (Raahil Shah) |
| 2 | Int6 MLP3x + SmearGate + BigramHash | 1.1458 | 9 (4E+5D) | 3x (1536) | Int6 all blocks | Sliding stride=64 | SmearGate, BigramHash(4096), OrthoInit, SWA(0.5) | None | Baseline |
| 3 | 11L MLP3x + Int6 QAT | 1.1502 | 11 | 3x (1536) | Int6 STE QAT | Sliding stride=64 | None novel | None | Baseline + QAT pattern |
| 4 | SmearGate + OrthoInit + Muon WD | 1.1556 | 9 (4E+5D) | 3x (1536) | Int6 STE QAT | Sliding stride=64 | SmearGate, BigramHash(4096), OrthoInit | None | Baseline |
| 5 | 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | 10 | 2.625x (1344) | Int6 STE QAT | Sliding stride=64 | None novel | None | Baseline |

## Divergent / Interesting Entries

| Entry | BPB | Track | Key Innovation |
|-------|-----|-------|---------------|
| LoRA TTT | 1.1928 | Record | Test-time training with rank-8 LoRA at eval. Orthogonal to all training mods. |
| 4-Hour Baseline | 1.2074 | Non-record (unlimited) | 172B tokens, 4 hours, standard arch. Shows diminishing returns vs arch innovation. |
| Warmdown-Quantization | 1.2154 | Record | Aggressive LR decay (warmdown=20000) tightens weight distributions for better post-quant. |

## Answers to Key Questions

### Q1: Is there a single dominant recipe?
**YES.** The top 5 all share the same core recipe:
- Int6 QAT (STE) + zstd-22 compression
- 3x MLP expansion (funded by int6 compression savings)
- Muon optimizer with WD ~0.04, momentum warmup 0.92→0.99
- Sliding window evaluation at stride=64
- FP16 tied embeddings
- U-Net skip connections (encoder/decoder)
- RoPE with QK-norm and logit softcap

The top 2 add SmearGate + BigramHash + OrthoInit + SWA on top of this core.

### Q2: Are there genuinely different approaches?
**One:** LoRA TTT (#9, 1.1928) is genuinely different — it adapts at eval time with LoRA. This is orthogonal to all training-time optimizations and could be stacked with the SOTA recipe.

No depth-recurrent, state-space, MoE, or novel tokenizer submissions exist yet.

### Q3: Techniques in lower entries NOT yet in #1?
- **LoRA TTT** (test-time training) — not in any top entry
- **11 layers** (#3 uses 11L, #1 uses 10L with int5/int6 split)
- **Overtone spectral init** (one entry) — not widely adopted

### Q4: Has ANYONE modified the loss function?
**NO.** Zero submissions modify the loss function. Every single entry uses standard `F.cross_entropy(logits.float(), targets, reduction="mean")`. **Our auxiliary loss approach is genuinely novel in this competition.**

### Q5: What has been tried and failed?
- Longer training (4h) shows diminishing returns vs architectural creativity
- SWA with high collection frequency (200→100→75 steps) — 50 steps is optimal
- Muon WD too low (0.01) or too high (0.05) — 0.04 is the sweet spot
- Non-STE quantization has much higher penalty than STE QAT

### Q6: Interesting outlier submissions?
- **LoRA TTT**: Could be stacked with our aux losses for a combined submission
- **Warmdown-Quantization**: Shows that training dynamics affect post-quant quality — our aux losses could similarly improve quantization-friendliness

## Forking Decision

**Fork #1 (thwu1, 1.1428 BPB).**

Rationale:
1. The field has fully converged on one recipe — fork the best version of it
2. #1 includes ALL techniques from #2 (SmearGate, BigramHash, OrthoInit, SWA) PLUS:
   - Mixed int5/int6 quantization (novel, extra layer funded by savings)
   - BigramHash(10240) vs (4096)
   - Magnitude pruning for better compression
   - SWA start_frac=0.4 (more selective than 0.5)
3. #1 is 1231 lines — well within manageable size for adding aux losses (~200-300 lines)
4. No entry modifies the loss function — our contribution fills this gap entirely
5. #1 builds on #2 (PR#162), so we get the full technique stack

**Our auxiliary losses are the only genuinely novel contribution possible in the loss function space.** Every other team is competing on architecture/quantization/eval tricks. We occupy an entirely uncontested dimension.

# Parameter Golf Research Summary — For Deep Review

## Competition Context

OpenAI's Parameter Golf: train the best language model that fits in 16MB (weights + code, compressed), trains in ≤10 minutes on 8×H100 GPUs, evaluated by bits-per-byte (BPB) on FineWeb validation set. Competition runs March 18 – April 30, 2026.

**Current leaderboard SOTA:** 1.1194 BPB (int6 quantized, 27M params, 11 layers, 512d)
**Ternary submission we're building on:** 1.1570 BPB (ternary {-1,0,+1}, 73.7M params, 10 layers, 768d)
**Gap:** 0.038 BPB

## Models We Have Worked With

### Model 1: Int6 SOTA (abaybektursun, 1.1194 BPB)
- **Architecture:** 11L, 512d, 8 heads, 4 KV heads (GQA), 3× MLP, LeakyReLU(0.5)², U-Net skip connections
- **Key techniques:** XSA on last 4 layers, Partial RoPE (16/64), LN Scale, Shared Value Embedding, EMA(0.997) + Tight SWA, GPTQ-lite int6 quantization, Late QAT@0.15, SmearGate, BigramHash(1536), OrthoInit, Parameter Banking + Parallel Muon, Legal Score-First TTT at eval, lzma compression
- **Quantization:** Int6 per-row for MLP/attention, Int8 for embeddings, GPTQ-lite clip search
- **Training:** ~7185 steps in 600s on 8×H100, 786K tokens/step
- **We re-forked 3 times** as the leaderboard evolved (thwu1 1.1428 → signalrush 1.1228 → abaybektursun 1.1194)
- **We trained a full 5-hour equivalent** on Colab (5871 steps, final sliding BPB: 1.1289)
- **We ran comprehensive error analysis** on this fully-trained model (see Track 1 below)

### Model 2: Ternary BitNet b1.58 (Ciprian-Florin Ifrim, 1.1570 BPB)
- **Architecture:** 10L, 768d, 8 heads, 4 KV heads (GQA), 4× relu² MLP, U-Net skip connections, factored embedding (254→768), polynomial softcap (deg 5, cap=10), YaRN 2048 positional encoding
- **Key techniques:** BitNet b1.58 ternary weights {-1,0,+1} with per-group absmean scaling (group=128), NeoMuon optimizer (3 Newton-Schulz steps), fused QKV, fused relu², FlashAttention-3, Z-loss regularization (1e-4), temperature scaling (T=0.90), FP8 storage for non-ternary params
- **Quantization:** Ternary ~1.6 bits/param, base-3 packing + LZMA compression
- **Training:** ~6530 steps in 600s on 8×H100, 524K tokens/step
- **73.7M params in 15.99MB** — 2.7× more parameters than int6 SOTA
- **The author ran 250+ experiments** with detailed ablation logs (RESULTS.md)
- **We trained 1-hour baseline** on Colab G4 GPU (1100 steps, val_bpb 1.2968 post-roundtrip)
- **We ran Phase 0 quantization error analysis** and geometric field ablation (see Track 2 below)

## Two Tracks of Investigation

### Track 1: Int6 SOTA — Auxiliary Losses — COMPLETED, NEGATIVE

We forked the #1 int6 submission and tested auxiliary loss modifications:

**What we tested:**
- Focal loss (gamma 0.5, 1.0, 2.0, 3.0)
- Inter-layer decorrelation (lambda 0.001 to 0.1)
- Representation rank loss (lambda 0.01 to 0.1)
- Unigram KL divergence (lambda 0.01 to 0.1)
- Scheduled label smoothing
- Gradient noise injection
- Sparse activation (k-WTA with STE)
- Top-K margin loss
- Close-wrong boost loss

**Result: ALL negative.** Every modification either hurt or was within noise of baseline. The model is already well-calibrated — CE is optimal for BPB.

**Error analysis of the fully trained int6 SOTA model (5-hour run, 1.1289 BPB sliding):**

| Category | % of Tokens | Mean Loss | Notes |
|----------|:-----------:|:---------:|-------|
| Correct (top-1) | 55.0% | low | Model predicts correctly |
| Wrong but in top-10 | 27.8% | 2.72 | Correct answer nearby but not ranked #1 |
| Wrong, not in top-10 | 17.3% | 5.50 | Model fundamentally can't predict these |

**Deep dive on the 17.3% "hopeless" tokens:**
- 46.5% are short tokens (1-2 chars) — BPE word-initial fragments
- 28.5% are common words — just hard to predict in context
- 20.5% are capitalized words — proper nouns
- Spread uniformly across positions and documents (not clustered)

**Key insight: The model's errors are dominated by word-boundary prediction with a 1024-token BPE vocabulary.** The first token of each word carries the "which word comes next?" decision, and a 27M parameter model can't memorize enough of web text to predict this for 17% of positions.

**Confidence calibration is near-perfect:** confidence bins map almost exactly to accuracy (0.3 conf → 34.5% acc, 0.9 conf → 97.8% acc). This means loss reweighting cannot help — the model already knows what it knows.

**Training dynamics:**
- 91.5% of learning happens in first 500 steps (rapid descent)
- Steps 500-3500: slow plateau with oscillating train loss
- Warmdown at 48% of wallclock: LR schedule is wallclock-based, auto-adapts

**Competition-wide findings (from Issue #140 live commentary):**
- ALL loss function modifications have failed across all competitors
- Label smoothing: tested and rejected ($500 systematic study)
- Focal loss: tested and rejected
- Selective token training: tested and rejected
- MTP: tested and rejected (compute overhead kills it)
- Standard CE is optimal for BPB in this regime

### Track 2: Ternary Model (geometric field + architecture) — IN PROGRESS

We're building on the ternary submission (73.7M params, {-1,0,+1} weights, 1.1570 BPB).

**Phase 0: Quantization Error Structure Analysis — COMPLETED**

Analyzed all 40 ternary weight matrices (4 types × 10 layers). Found:
- Row structure ratio: 7.20 (threshold was 2.0) — MASSIVE structure
- Column structure ratio: 6.82 — MASSIVE structure
- 100% of matrices exceed threshold in both dimensions
- Later layers have MORE structure (block 9 mlp.proj: RowStr=18.72)

**BUT the quantization tax is only 0.0013 BPB.** The structure exists but the STE already handles it well.

**Geometric Field G Ablation — COMPLETED, NEGATIVE**

Tested modulating weights before ternary STE with a position-dependent field G:
- Covariance signal (alpha): **Applied to 0 layers** (C_diag bug — compute_signals failed to collect input statistics due to dtype mismatch)
- Word-boundary signal (beta): Applied to 30 layers, **CATASTROPHICALLY harmful**:
  - beta=0.1: Q-tax 10x worse (0.0013 → 0.014 BPB)
  - beta=0.3: Q-tax 40x worse (0.0013 → 0.052 BPB)
  - beta=0.5: Q-tax 160x worse (0.0013 → 0.207 BPB)

**Root cause:** The ternary STE co-adapts continuous weights with quantization during training. G disrupts this co-adaptation. The model's weights are shaped to quantize well to {-1,0,+1}. G changes the quantization behavior, creating a mismatch that grows with G's amplitude.

**Control run baseline (ternary, 1 hour on G4 GPU):**
- 1100 steps in 3600s (3.29s/step without torch.compile)
- Step 500: val_bpb = 1.4589
- Step 1000: val_bpb = 1.3227
- Step 1100: val_bpb = 1.2955 (pre-quant), 1.2968 (post-roundtrip)
- Quantization tax: 0.0013 BPB
- Artifact: 15.98MB (slightly over 16MB budget at 1hr, fits at full 10min/8×H100)

## What We Now Understand About the Problem

### The int6 model (27M params)
1. **Well-calibrated** — loss reweighting can't help
2. **Capacity-limited** — 17.3% of tokens are fundamentally unpredictable at this param count
3. **Tokenizer-bottlenecked** — 46.5% of hardest tokens are BPE word-initial fragments
4. **Architecturally optimized** — the competition has tested every known technique over 250+ submissions

### The ternary model (73.7M params)
1. **More parameters but less information per parameter** — 118M total bits (ternary) vs 162M total bits (int6)
2. **Quantization tax is negligible** (0.0013 BPB) — the STE is already excellent
3. **The gap (0.038 BPB) is from computational expressiveness**, not rounding error
4. **Width over depth is critical** — 768d/10L beats 512d/25L (from the 250-run log)
5. **Step time is critical** — 22% overhead killed a -0.001 BPB gain from SmearModule
6. **Cross-group isolation is fatal** — Grouped MLP lost 0.03-0.06 BPB
7. **The STE co-adaptation is strong** — anything that disrupts the weight↔quantization relationship hurts severely (EMA: -0.12, G: up to -0.21)
8. **Weight freedom matters** — WD=0 at 4×MLP; the optimizer needs full freedom for ternary

### What each ternary weight actually computes
- Each weight is {-1, 0, +1} × scale (per group of 128)
- 0 = ignore this input entirely (gating)
- ±1 = include this input with positive or negative sign
- scale = one magnitude shared by 128 weights
- A ternary neuron computes: **signed selection + summation** of inputs
- relu² then squares the result, creating continuous positive values
- The ACTIVATION is where precision recovery happens — it's the only non-ternary computation

### The information flow bottleneck
```
Input (continuous) → TernaryMatmul (3 values per weight) → Activation (continuous) → next layer
```

The bottleneck is the TernaryMatmul step. Each layer reduces information from continuous to ~1.6 bits per connection, then the activation recovers some precision. **Making the activation more expressive is the cheapest way to recover information lost in ternary matmul.**

## Experiments Ready but Not Yet Run

### Experiment A: Ternary Residual
Same weight quantized twice with different group sizes. `y = ternary(W, g=128)·x + ε·(ternary(W, g=512)·x - ternary(W, g=128)·x)`. The residual patches where fine-group quantization is weakest. Zero extra params.

**Status:** Implemented but data path issue on Colab prevented execution. Need to delete stale crash logs and re-run.

### Experiment D: Structured Zero Placement
Bias the zero threshold per-column based on input importance. Important dims get fewer zeros (more ±1). Zero extra params.

**Status:** Implemented but same data path issue.

## Proposed Next Directions (Not Yet Implemented)

### 1. Learnable Activation Functions (KAN-lite)
Replace fixed relu² with a learned piecewise-linear or B-spline function per layer. ~128 bytes per layer = 1.3KB total. The model learns the optimal activation shape for ternary outputs. **Never tried in this competition.**

### 2. Asymmetric Ternary Values
Change {-1, 0, +1} to {-1, 0, +a} where `a` is learned per group. Costs 1 float per 128 weights. Breaks the symmetry assumption that some groups may not need.

### 3. Parametric Power Activation
`leaky_relu(x, s) ** p` where `p` is learned per layer. Generalizes relu² (p=2). Some layers may want different powers.

### 4. Ternary Factorization
`y = A·norm(B·x)` where both A, B are ternary. Intermediate is continuous → A operates on richer inputs. **CAUTION:** Grouped MLP failed badly (-0.03 to -0.06 BPB) from cross-group isolation. Factorization is mathematically different (intermediate shares info across all dims) but the precedent demands care.

### 5. Bit-Plane Decomposition
`W_eff = W₁·s₁ + W₂·s₂` (2 ternary planes). 9 effective levels per weight vs 3. Must keep full model width — shrinking dim is fatal. **Apply to MLP only**, keep attention standard.

## Hardware & Infrastructure

- **Colab GPUs tested:** T4 (16GB, too small for ternary), A100-40GB (tight, OOM at full batch), A100-80GB, G4/RTX PRO 6000 (95.6GB, best option), H100 (79.6GB, 18 CU/hr)
- **Ternary model needs ≥40GB VRAM** (73.7M params + activations)
- **No FlashAttention 3** on non-Hopper GPUs — we patched to FA2/SDPA fallback
- **No torch.compile** on Ada Lovelace — SM shared memory limit (99KB < 108KB needed)
- **H100 OpenAI credits:** $25 budget, saving for final 3-seed validation

## Key Code Artifacts

| File | Purpose |
|------|---------|
| `geometric_field/phase0_analysis.py` | Quantization error structure analysis |
| `geometric_field/compute_signals.py` | Word-boundary direction + input covariance |
| `geometric_field/geometric_field.py` | G modulation (monkey-patches TernaryLinear) |
| `geometric_field/ternary_residual.py` | Experiment A: two-scale residual quantization |
| `geometric_field/structured_zeros.py` | Experiment D: importance-based zero thresholds |
| `geometric_field/patch_ternary.py` | Patches 70K-line ternary script for non-Hopper GPUs |
| `scripts/error_analysis.py` | Per-token error analysis of trained models |
| `scripts/analyze_training_data.py` | FineWeb data statistics for loss design |
| `aux_losses/*.py` | 6 auxiliary loss implementations (all produced negative results) |

## What We're Looking For

Novel architectural modifications for the ternary model that:
1. **Don't shrink model width** (768d is critical — 512d loses badly)
2. **Add < 10% step time overhead** (step time is the binding constraint in 10 minutes)
3. **Don't disrupt STE co-adaptation** (the G experiment showed this is fragile)
4. **Exploit the continuous activation as the precision recovery point**
5. **Cost minimal artifact bytes** (budget is already at 15.98MB)
6. **Are genuinely novel** (the competition has tested every standard technique across 250+ submissions)

The most promising direction: **learnable activation functions** that compensate for ternary weight limitations by making the non-quantized parts of the computation more expressive.

# Parameter Golf — Novel Ideas Research

**Date:** 2026-04-07
**Context:** Research into which experimental directions remain genuinely novel in the OpenAI Parameter Golf competition, after our progressive growing experiment failed. Competition ends April 30, 2026.

## Competition State (as of 2026-04-07)

- **Merged SOTA:** 1.1147 BPB (PR #1019, abaybektursun, merged March 30)
- **Nothing merged since March 30** — maintainers are being selective
- **Pending frontier claims:** ~1.080 BPB (PR #1420 at 1.08014)
- **Major issue:** Tokenizer byte-accounting bug (PR #1143 lineage) invalidated several sub-1.10 claims
- **New rule:** Submissions <1.10 BPB hidden by default pending legality review
- **Competition heavily explored:** ~953 open + 433 closed PRs

## Our Experimental Track Record

- **Auxiliary losses (focal, decorrelation, rank, unigram KL, topk margin):** All failed
- **Activation experiments (parametric power, stochastic depth, GaugeReLU):** All failed (overhead fatal)
- **Geometric field modulation (ternary):** Catastrophic (+0.14 BPB)
- **Asymmetric ternary:** Untested
- **Progressive depth growing (7L → 11L):** Failed (+0.4 to +0.8 BPB worse). U-Net role change at transition is destructive.

## Brainstormed Directions

We identified 7 potentially novel dimensions to explore. The research below verifies novelty against:
- GitHub PRs on openai/parameter-golf (all 1386+ PRs)
- Issue #140 (live commentary thread)
- Recent PRs filed April 3-7 2026

### Verdict Summary

| # | Idea | Status | Evidence |
|---|------|--------|----------|
| 1 | Byte-level auxiliary head | **STILL NOVEL** | No auxiliary byte prediction head exists |
| 2 | Neural residual correction model | **STILL NOVEL** | Zero matches |
| 3 | Within-run NAS | **STILL NOVEL** | Only offline/manual search exists |
| 4 | Multi-neural ensemble at inference | **STILL NOVEL** | Only neural+n-gram blending done |
| 5 | **Online Hessian accumulation** | **STILL NOVEL (strongest angle)** | All Hessian PRs calibrate post-training |
| 6 | Advanced quantization (AQLM/QTIP/VPTQ/GuidedQuant) | **PARTIALLY TAKEN** | Lattice VQ tried (weak), others untried |
| 7 | Compression-aware training regularizer | **TAKEN** | PR #1385 (CAT) implemented April 5 |

## Detailed Findings

### 1. Byte-Level Auxiliary Head — STILL NOVEL

**The concept:** Add an auxiliary loss that directly penalizes byte-level prediction errors, on top of the main BPE token-level CE loss. The model is scored on byte-level BPB but trained on token-level CE, creating a theoretical gap.

**What exists:**
- **PR #1443 (hardik-bhadani-git)** "ByteJEPA" — **full** byte-level model (vocab=256), not auxiliary. 3-stage JEPA training, got 1.3496 BPB (non-competitive).
- **PR #1411 (Blakethefn)** "ByteEmbed" — **input-side only** 64-dim byte embedding pathway. No auxiliary prediction loss.
- **PR #1044 (greqone)** "H-Net" — full byte-level model, got 1.90 BPB (catastrophic).

**Failed auxiliary approaches in this space:**
- MTP (multi-token prediction) heads: PRs #212, #236, #375, #1031 — all neutral or worse
- Focal loss, KL from pre-quant: PR #481 — failed
- Label smoothing: PR #375 — failed

**Theoretical concern:** BPE → byte conversion is a fixed multiplier (`bits_per_token × tokens_per_byte`). It's unclear if an auxiliary head can actually change this ratio, or if it's a mathematical identity. Needs theoretical analysis before implementation.

**Reference paper:** arxiv 2410.09303 "Exact Byte-Level Probabilities from Tokenized Language Models"

### 2. Neural Residual Correction Model — STILL NOVEL

**The concept:** Train the main model for ~8 minutes. Freeze it. Train a tiny second neural model (1-2MB) for ~2 minutes that predicts and corrects the main model's systematic errors. At eval, combine both.

**What exists (different approaches):**
- **PR #232 (kellyvv)** "Error Correction Table" — pre-computes errors into a 2.87MB lookup table, not a neural model. Closed as illegal (used val data).
- **PR #1446** "gated Krylov residual correction" — correction INSIDE the Muon optimizer, not a separate model.

**Verdict:** A tiny second neural network specifically for error correction has not been attempted.

**Risk:** Freezing the main model and training a second one loses the benefit of continued gradient updates to the main model. Unclear if the gain exceeds the opportunity cost.

### 3. Within-Run Neural Architecture Search — STILL NOVEL

**The concept:** Spend the first 60 seconds of the 10-minute window training 10 architectural variants for 6 seconds each. Pick the winner. Train the winner for 9 minutes.

**What exists:**
- **Offline search** by many authors (PRs #141, #462, #823, #214, #1036, #748) — but these run hundreds of experiments externally.
- No PR automates architecture selection within the actual submission run.

**Risk:** Training variants for only 6 seconds each may not give reliable signal about which architecture will win at 600s. Early training dynamics don't always predict final performance.

### 4. Multi-Neural Ensemble at Inference — STILL NOVEL

**The concept:** Train 2-3 small independent neural models (9-12MB total, quantized), combine their predictions at eval time via averaging or learned gating.

**What exists (different approaches):**
- Neural + n-gram ensembles via Hedge algorithm: PRs #856, #909, #953, #1145 (the last is competitive at 1.0722-1.1109 BPB but uses n-gram experts, not independent neural models).
- **PR #1451 (davie2009kh)** MoE + BigramHash4096 — single-model MoE at 1.1180 BPB, not independent ensemble.
- SWA (Stochastic Weight Averaging) — within single run, not ensemble.

**Issue #140 note:** "MoE at scale is definitively unviable... optimal sparsity = 0 below ~500M params."

**Risk:** Smaller models almost never beat a single larger model of the same total parameter budget. Only hope is that the averaging cancels systematic errors of each sub-model.

### 5. Online Hessian Accumulation During Training — STRONGEST NOVEL ANGLE

**The concept:** Accumulate GPTQ Hessian matrices (`H = Σ X^T X`) during training forward passes, using spare VRAM. Skip the ~150s post-training calibration data generation phase. At end of training, run GPTQ directly on accumulated Hessians.

**Benefits:**
- **Saves ~150s** of post-training calibration → ~1800 extra training steps at ~83ms/step
- **Better calibration data** — real training data vs self-generated synthetic data
- **Cost:** ~11MB per GPU for Hessian storage (trivial vs 80GB VRAM)

**What exists (all post-training):**
- **PR #1412 (Robby955)** "Hessian-Aware SDClip" — uses GPTQ's Hessian diagonal to modulate clipping during **GPTQ calibration** (post-training).
- **PR #1446 (LauraGomezjurado)** "AR GPTQ int6" — calibrates Hessians on AR samples (temp=0.8), post-training.
- **PR #1433 (mtybadger)** "Codebooks" — Hessian-aware assignment for VQ, post-training.
- **PR #549, #1019 (abaybektursun)** current SOTA — AR self-gen Hessian GPTQ, post-training.

**Verdict:** Zero PRs accumulate Hessians during training. All current approaches treat calibration as a separate post-training stage.

**Implementation sketch:**
```python
# During training forward pass, in each target layer:
with torch.no_grad():
    X = layer_input  # [batch*seq, in_dim]
    H_accumulator += X.T @ X  # O(in_dim^2) storage per layer
# After training, feed accumulated H directly to GPTQ:
q, scale = quantize_int6_gptq(weight, hessian=H_accumulator, ...)
```

**Concerns:**
- Hessian accumulation adds minor overhead during training (~1 matmul per layer per step)
- Need to handle the distinction between pre-quant and post-quant Hessians
- Legal concern: training-time Hessians use training data, which might be considered "training data access during quantization" — but the quantization happens before evaluation, so likely legal

### 6. Advanced Quantization — PARTIALLY TAKEN

**Status of specific methods:**
- **Lattice VQ (QuIP#-style EP8):** **TRIED** by PR #1433 (mtybadger) "Codebooks" — got weak 1.2067 BPB. Author explicitly noted "AQLM was hard to optimize" and did not implement it.
- **AQLM:** **NOT TRIED** — noted as "hard to optimize" and "10-18x slower than GPTQ"
- **QTIP (Trellis):** **NOT TRIED** — 3-bit matches GPTQ 4-bit per paper
- **VPTQ:** **NOT TRIED** — beats GPTQ by 0.01-0.34 ppl per paper
- **GuidedQuant:** **NOT TRIED** — gradient-aware PTQ
- **EntQuant (entropy-coded weights):** **PARTIALLY DISCUSSED**
- **Rate-distortion optimal:** **NOT TRIED**

**Historical failures:**
- **PR #212 (mrdavtan)** K-means codebook K=256 — 87% lower reconstruction MSE BUT **25% larger artifact** (codebook indices compress worse than int6)
- **PR #1227** PAQ Logistic Mixing — **+292% BPB** catastrophic

**Speed concern is fatal:** Advanced methods like AQLM are 10-18x slower than GPTQ. The current GPTQ phase already takes ~150s. AQLM would take ~30 minutes, far exceeding the 10-minute budget.

### 7. Compression-Aware Training Regularizer — NOW TAKEN

**Status:** PR #1385 (korentomas, April 5) "Compressor-Aware Training (CAT)" implemented the exact idea.

**What CAT does:**
- **LZ77 Dictionary Matching Proxy:** multi-lag soft autocorrelation on serialized quantized weight byte streams
- **Entropy Proxy:** soft histogram Shannon entropy for Huffman/FSE compression

**Results (1× H100, 5 runs):**
- Control: 1.4374 BPB / 12.32 MB
- Combined CAT: 1.4465 BPB / 11.48 MB (−842 KB, **6.8% reduction at +0.009 BPB cost**)
- Entropy-strong: 1.5044 BPB / 9.81 MB (20% reduction at +0.067 BPB cost)

**Verdict:** The naive implementation gives modest size reduction at BPB cost. There's theoretical room for a smarter formulation that achieves compression without quality loss, but the current attempt suggests this is hard.

**Earlier related attempts:**
- **PR #934 (tuanaqeelbohoran)** MDL-T regularizer
- **PR #930 (lamb356)** Entropy-regularized QAT
- **PR #514 (sanjith3057)** Per-row weight range penalty
- **PR #1287 (dentity007)** Weight decay correlates with compressibility (R² ~0.99)
- **PR #609 (saml212)** Artifact size is the binding constraint

## Recently Filed PRs (April 3-7, 2026)

### Frontier Contenders (~1.08 BPB range)

- **PR #1420 (abaybektursun)** "Triple Loop + Fused Kernels + Parallel Residuals + N-gram Tilt" — **1.08014 BPB**
  - Triple depth recurrence (17 virtual from 11 physical layers)
  - Fused MLP kernel (Triton TMA + CUTLASS EVT) → +10% throughput
  - Parallel residuals on layers 7-10 (GPT-J style)
  - Eval-time N-gram tilt with causality verification
  - Negative results reported: E8 lattice VQ, entropy equalization, gauge symmetries

- **PR #1450 (andrewbaggio1)** "TMA Megakernel + Triple Loop + Parallel Residuals" — **1.08480 BPB**
  - Hopper-specific TMA megakernel (~384MB activation elimination)
  - +10.5% throughput
  - Same family as #1420 minus the n-gram tilt

- **PR #1437 (dexhunter)** "Diagnostic: SP8192 + Parallel Residuals" — **1.08091 BPB**
  - Discovered causality bug in n-gram kernel (within_hint/word_hint leaked metadata at position p instead of p-1)
  - Diagnostic/transparency submission

- **PR #1435 (AbhayAnandUCSD)** "11L Depth Recurrence + BigramHash + EMA 0.9965" — **1.0980 BPB**
  - Layers 4,5 repeat once (13 virtual from 11 physical), activated at step 3000
  - BigramHash at 1536×112
  - GPTQ int6 + Brotli compression

### Other Notable PRs

- **PR #1451 (davie2009kh)** "MoE + BigramHash4096" — 1.1180 BPB (first MoE exploration in the repo)
- **PR #1452 (bsisduck)** "TurboQuant + N-gram + TTT" — CLOSED, non-record
- **PR #1443 (hardik-bhadani-git)** "ByteJEPA" — 1.3496 BPB
- **PR #1411 (Blakethefn)** "Blueprint Stack + ByteEmbed" — 1.5568 BPB
- **PR #1385 (korentomas)** "Compressor-Aware Training (CAT)" — Took our idea #7

## Strategic Recommendations

### Priority 1: Online Hessian Accumulation
The cleanest novel idea with a clear value proposition:
- **Saves ~150s of calibration** → ~1800 extra training steps
- **Better calibration data** from real training samples
- **Trivial memory cost** (~11MB per GPU)
- **No model changes** — purely a training pipeline optimization
- **Zero precedent** — all Hessian work is post-training

### Priority 2: Byte-Level Auxiliary Head
Theoretical dark horse with zero precedent:
- **Risk:** May be a mathematical no-op due to BPE→byte multiplier being fixed
- **Reward:** Could address the train/eval mismatch no one else has touched
- **Recommendation:** Theoretical analysis before implementation

### Priority 3: Neural Residual Correction Model
Requires careful opportunity-cost analysis:
- **Pros:** Zero precedent, orthogonal to other approaches
- **Cons:** Takes training time from the main model
- **Recommendation:** Only if Hessian accumulation gives time budget headroom

### Avoid
- **Progressive growing:** Proven harmful for U-Net architectures (our experiment)
- **Compression-aware training:** Taken by PR #1385 (and results are weak)
- **Distillation:** PR #1029 definitive negative result
- **Pruning:** PR #1048 — zeroed weights hurt LZMA compression
- **Advanced quantization (AQLM/QTIP):** Too slow for 600s budget
- **Curriculum learning:** PR #212 — no effect; seq curriculum breaks SWA
- **Loss function modifications:** Issue #140 — "no loss function modification has helped in this competition"

## Competition Dimensions: Explored vs Untouched

| Dimension | Status |
|-----------|--------|
| Model architecture | Heavily explored |
| Quantization (uniform) | Heavily explored |
| Quantization (non-uniform) | Partially explored (lattice VQ weak, AQLM untried) |
| Optimizer (Muon variants) | Heavily explored |
| Compression codec | Heavily explored (LZMA, Brotli, zstd) |
| Test-time training | Heavily explored (SLOT, TTT, etc.) |
| Tokenizer | Fixed by competition rules |
| Loss functions | Heavily explored, all negative |
| Data ordering / curriculum | Explored, mostly negative |
| **Hessian accumulation during training** | **Uncontested** |
| **Byte-level auxiliary supervision** | **Uncontested** |
| **Independent neural ensembles** | **Uncontested** |

## Sources

- [OpenAI Parameter Golf Repository](https://github.com/openai/parameter-golf)
- [Issue #140 — Live AI Commentary](https://github.com/openai/parameter-golf/issues/140)
- [Current Merged SOTA PR #1019](https://github.com/openai/parameter-golf/pull/1019)
- [PR #1385 — CAT (compression-aware training)](https://github.com/openai/parameter-golf/pull/1385)
- [PR #1420 — Triple Loop + N-gram Tilt](https://github.com/openai/parameter-golf/pull/1420)
- [PR #1433 — Codebooks (EP8 lattice VQ)](https://github.com/openai/parameter-golf/pull/1433)
- [PR #1443 — ByteJEPA](https://github.com/openai/parameter-golf/pull/1443)
- [PR #1451 — MoE + BigramHash4096](https://github.com/openai/parameter-golf/pull/1451)
- Related paper: [Exact Byte-Level Probabilities from Tokenized Language Models (arxiv 2410.09303)](https://arxiv.org/abs/2410.09303)

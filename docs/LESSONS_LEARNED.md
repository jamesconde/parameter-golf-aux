# Parameter Golf: Lessons Learned

## Competition Summary

**OpenAI Parameter Golf** — train the best LM in 16MB, 10 minutes on 8×H100.
Competition: March 18 – April 30, 2026.

We worked on two models across ~10 days of experimentation:
- **Int6 SOTA** (27M params, 1.1147-1.1194 BPB) — auxiliary loss experiments
- **Ternary BitNet** (73.7M params, 1.1570 BPB) — architecture & quantization experiments

Total experiments run: ~40+ (across both models, multiple Colab sessions)

---

## The Hard Constraints (proven by data)

### 1. Step Time Is Everything
In a wallclock-limited competition, **every 1% of step time overhead costs ~11 training steps.** This is the single most important lesson.

| Experiment | Step Overhead | Steps Lost | BPB Impact |
|-----------|:------------:|:----------:|:----------:|
| Monkey-patched relu^p | +45% | -340 | +0.032 (worse) |
| GaugeReLU (atan2/cos/sin) | +72% | -460 | +0.14 (catastrophic) |
| Stochastic depth | -4.5% | +50 | +0.002 (neutral) |
| Geometric field G | +0.5% | -5 | neutral to harmful |

The parametric power activation actually showed **better per-step quality** (val_bpb 1.4446 vs 1.4619 at step 500) AND lower quantization tax (0.0010 vs 0.0014). But 340 fewer steps wiped out the gain entirely.

**Implication:** Any modification must be either (a) zero overhead or (b) compiled inline with torch.compile for kernel fusion. Monkey-patching breaks fusion.

### 2. The STE Co-Adaptation Is Fragile (Ternary Only)
The ternary model's continuous weights are shaped BY training to quantize well to {-1, 0, +1}. Anything that changes the quantization behavior disrupts this co-adaptation.

| Experiment | What It Changed | Q-Tax Impact |
|-----------|----------------|:------------:|
| Geometric field G (beta=0.1) | Modulated weights before STE | 10x worse |
| Geometric field G (beta=0.5) | More aggressive modulation | 160x worse |
| EMA (from submission log) | Averaged weights → more zeros | 92x worse (0.12 BPB) |

**Implication:** Don't touch the STE or the weight distribution. The model has optimized for its own quantization.

### 3. Cross-Entropy Is Optimal for BPB
We tested every loss modification the literature suggests:

| Loss Modification | Result | Why |
|------------------|--------|-----|
| Focal loss (gamma 0.5-3.0) | All worse | Model is well-calibrated; reweighting hurts |
| Label smoothing | Worse | Hurts perplexity (known); tested and failed by other competitors |
| Decorrelation loss | Worse | Model already uses capacity efficiently |
| Rank loss | Worse | Same |
| Unigram KL | Worse | Model already knows base rates |
| Gradient noise | Not viable | Muon already has noise characteristics |
| Selective token training | Worse (competition-wide) | Hardest tokens are genuinely unpredictable |
| Top-K margin loss | Worse | No gradient benefit over CE |
| Sparse activation (k-WTA) | Catastrophic | Broke GPTQ calibration + 35% slower |

**Confirmed across ALL competitors in Issue #140:** No loss function modification has helped in this competition. Standard CE is optimal for BPB.

### 4. The Model Is Well-Calibrated
From our error analysis of the fully-trained int6 SOTA:
- Confidence calibration is near-perfect (0.3 conf → 34.5% acc, 0.9 → 97.8%)
- This means the model already knows what it knows and doesn't know
- Loss reweighting CAN'T help because the model's uncertainty estimates are correct

### 5. Width Over Depth for Ternary
From the ternary submission's 250-run log:
- 768d/10L beats 512d/25L (faster steps → more training in 10 minutes)
- Any modification that narrows the model (bit-plane decomposition, factorization) is high risk
- 4× MLP beats 3× MLP for ternary (more width = better at this budget)

### 6. The Activation Is the Only Free Parameter
In ternary models:
- Weights: locked to {-1, 0, +1} × scale
- STE: fragile, don't touch
- Activation: continuous, full precision, the ONLY non-quantized computation

Making activations more expressive is theoretically the cheapest way to recover information lost in ternary matmul. But in practice, any activation that's more complex than relu² adds step time that costs more than it gains.

---

## What We Tried and Failed

### Track 1: Int6 SOTA — Loss Function Experiments

**Model:** abaybektursun's submission, 27M params, 11L/512d
**Approach:** Auxiliary losses to shape the weight landscape during training

| Experiment | Config | BPB Delta | Verdict |
|-----------|--------|:---------:|---------|
| Focal loss gamma=0.5 | USE_FOCAL_LOSS=1 | +0.001 | Within noise |
| Focal loss gamma=1.0 | FOCAL_GAMMA=1.0 | +0.002 | Worse |
| Focal loss gamma=2.0 | FOCAL_GAMMA=2.0 | +0.002 | Worse |
| Focal loss gamma=3.0 | FOCAL_GAMMA=3.0 | +0.003 | Worse, monotonically |
| Decorrelation lambda=0.001 | LAMBDA_DECORR=0.001 | +0.006 | Worse |
| Decorrelation lambda=0.01 | LAMBDA_DECORR=0.01 | +0.008 | Worse |
| Decorrelation lambda=0.05 | LAMBDA_DECORR=0.05 | +0.006 | Worse |
| Decorrelation lambda=0.1 | LAMBDA_DECORR=0.1 | +0.008 | Worse |
| Rank loss lambda=0.01 | LAMBDA_RANK=0.01 | +0.002 | Within noise |
| Rank loss lambda=0.05 | LAMBDA_RANK=0.05 | +0.002 | Worse |
| Rank loss lambda=0.1 | LAMBDA_RANK=0.1 | +0.003 | Worse |
| Unigram KL lambda=0.01 | LAMBDA_UNIGRAM=0.01 | +0.002 | Worse |
| Unigram KL lambda=0.05 | LAMBDA_UNIGRAM=0.05 | +0.004 | Worse |
| Unigram KL lambda=0.1 | LAMBDA_UNIGRAM=0.1 | +0.006 | Worse |
| Scheduled label smoothing | SCHED_LABEL_SMOOTHING=0.1 | Not run (research showed it fails) | — |
| Scheduled gradient noise | SCHED_GRAD_NOISE=0.005 | Not run (Muon already noisy) | — |
| Sparse activation 10% | ACTIVATION_SPARSITY=0.1 | +0.77 (catastrophic) | Broke GPTQ, +35% slower |
| Sparse activation 20% | ACTIVATION_SPARSITY=0.2 | +0.68 (catastrophic) | Same |

### Track 2: Ternary Model — Architecture & Quantization Experiments

**Model:** Ciprian-Florin Ifrim's submission, 73.7M params, 10L/768d

#### What the Ternary Author Already Tried and Failed (250+ runs, from RESULTS.md)

The ternary submission author (Ciprian-Florin Ifrim) conducted an exceptionally
thorough exploration. These failures informed our experiment design:

**Ternary-Incompatible Techniques (structurally broken):**

| Technique | Result | Root Cause |
|-----------|--------|-----------|
| EMA weight averaging | -0.12 BPB RT gap | Averaging pushes weights toward zero → ternary rounds to 0 → scale mismatch |
| TTT-LoRA | Harmful at convergence | LoRA delta corrupts RMSNorm-calibrated ternary representations |
| Ternary sigmoid prototypes | -0.077 BPB RT gap | Sigmoid membership needs continuous values; ternary collapses patterns |
| LM head SVD factorization | Over budget | SVD factors U,V need fp16 precision → costs more bytes than saved |

**Techniques That Hurt BPB:**

| Technique | BPB Impact | Reason |
|-----------|:----------:|--------|
| Grouped MLP (g=2) | -0.031 | Cross-group isolation kills information sharing |
| Grouped MLP (g=4) | -0.056 | Even worse isolation |
| MTP (1 head) | -0.006 | Model capacity too limited for auxiliary objectives |
| MTP (2 heads) | -0.006 | Same — confirmed post-optimizer-fix |
| BigramHash | -0.020 at convergence | fp16 hash table displaces ternary layer capacity |
| SmearModule | -0.001, +22% step time | Step cost not recoverable in 10-minute budget |
| Differential attention | -0.022 | Halved head_dim (96→48) insufficient at this scale |
| Skip weights zero-init | -0.010 | Decoder needs skip signal from step 0 |
| Batch size schedule | Harmful | Noisier gradients interfere with ternary STE convergence |
| FP4 storage | -0.026 to -0.029 RT gap | Even with QAT, FP4 precision is insufficient |
| 16 heads at 768d | Harmful | 48-dim head_dim insufficient for meaningful attention |
| Plain relu | Dominated by relu² | relu² is strictly better at zero cost |
| Leaky relu | Dominated by relu² | Same |
| Distillation (in-run) | Harmful | Train-from-scratch teacher always worse than supervised |
| AdamW for matrix params | Clearly worse | Muon is necessary for ternary weights |
| Depth recurrence | Harmful | Halves effective steps; OOM at DR=3 |
| Seq/batch schedule | Harmful | Recompile and step penalties dominate at 600s wallclock |

**Techniques That Worked:**

| Technique | BPB Impact | Notes |
|-----------|:----------:|-------|
| 8192 BPE vocabulary | -0.42 | Largest single win (vs 1024) |
| relu² activation | -0.024 vs relu | Free (no cost) |
| 4× MLP width | -0.008 vs 3× | Best within budget at 10L |
| Width over depth (768d/10L) | Significant | Faster steps (91ms vs 127ms) = more training |
| FP8 storage for non-ternary | Saves ~2.5MB | Halves fp_params, enables wider MLP |
| NeoMuon (3 Newton-Schulz steps) | -6ms/step | Equivalent quality, 190 extra training steps |
| Fused QKV + fused relu² | -4-6ms/step | ~180 extra training steps |
| FlashAttention-3 | -13ms/step | ~380 extra training steps |
| Z-loss regularization (1e-4) | Quality gain | Anchors logits, keeps STE gradients sharp |
| Temperature scaling (T=0.90) | Consistent | relu² logits slightly underconfident |
| Base-3 + LZMA compression | -39% vs int8+zlib | Ternary-specific compression |
| BITNET_GROUP_SIZE=128 | Same quality as 64 | Saves 0.69MB in artifact |
| EMBED_DIM=254 | -0.0004 BPB | 256-2 to fit code within byte budget |
| YaRN 2048 | Marginal gain | ROPE_BASE=5000 with YaRN retained |
| Sliding window eval (stride=16) | -0.025 vs chunked | Full context per scored token |
| Weight decay = 0.0 | Best for 10L 4×MLP | Wider MLP needs full weight freedom |

**Key Architectural Decisions:**

| Decision | Rationale |
|----------|-----------|
| 10L × 768d (not 25L × 512d) | Minimum viable depth at 768d; faster steps = more training |
| 4× MLP (not 3×) | Best BPB within budget at 10L |
| WD=0.0 (not 0.04) | Opposite to deep models — wider MLP needs full weight freedom |
| FP8 (not FP16) | Halves non-ternary params, enables wider architecture |
| EMBED_DIM=254 | 256-2 dims to fit artifact+code under 16,000,000 byte budget |
| GROUP_SIZE=128 | Same quality as 64; saves 0.69MB |

#### Phase 0: Quantization Error Structure
- Row structure ratio: 7.20 (massive, but per-group scale handles this)
- Column structure ratio: 6.82 (massive, exploitable in theory)
- **But Q-tax is only 0.0013 BPB** — the STE is already excellent

#### Geometric Field G (Experiment 0)

| Run | Alpha | Beta | G Range | RT BPB | Delta |
|-----|:-----:|:----:|--------:|:------:|:-----:|
| control_1 | 0 | 0 | — | 1.2968 | — |
| control_2 | 0 | 0 | — | 1.2984 | +0.0016 |
| cov_01 | 0.1 | 0 | — (0 layers!) | 1.2987 | +0.0011 |
| cov_03 | 0.3 | 0 | — (0 layers!) | 1.2990 | +0.0014 |
| cov_05 | 0.5 | 0 | — (0 layers!) | 1.2980 | +0.0004 |
| bnd_01 | 0 | 0.1 | [0.66, 1.11] | 1.3110 | +0.0134 |
| bnd_03 | 0 | 0.3 | [0.38, 1.36] | 1.3523 | +0.0547 |
| bnd_05 | 0 | 0.5 | [0.25, 1.74] | 1.5076 | +0.2100 |
| both_01 | 0.1 | 0.1 | [0.66, 1.11] | 1.3113 | +0.0137 |
| both_03 | 0.3 | 0.3 | [0.38, 1.36] | 1.3543 | +0.0567 |
| both_05 | 0.5 | 0.5 | [0.25, 1.74] | 1.4951 | +0.1975 |

**Note:** cov_* runs applied G to 0 layers (C_diag collection bug). The boundary signal was catastrophically harmful.

#### Activation Experiments

| Run | What | RT BPB | Delta | ms/step | Steps |
|-----|------|:------:|:-----:|:-------:|:-----:|
| act_control | Baseline | 1.2992 | — | 3302 | 1100 |
| act_stoch_02 | Stochastic depth 20% | 1.3008 | +0.002 | 3154 | 1150 |
| act_power | Parametric relu^p | 1.3315 | +0.032 | 4789 | 760 |
| act_power_stoch | Power + stoch depth | 1.3320 | +0.033 | 4553 | 800 |
| act_gauge | GaugeReLU | 1.4429 | +0.144 | 5693 | 640 |
| act_gauge_stoch | GaugeReLU + stoch | 1.4381 | +0.139 | 5403 | 670 |

---

## What Actually Moves the Needle (from competition winners)

Based on the leaderboard progression (1.2244 → 1.1147 over 2 weeks):

| Technique | BPB Gain | Category |
|-----------|:--------:|----------|
| Int6 QAT + 3× MLP + zstd | -0.074 | Quantization + architecture |
| SmearGate + BigramHash + OrthoInit | -0.003 | Input features |
| 11th layer + weight decay | -0.013 | Architecture |
| XSA (last 4 → all 11 layers) | -0.005 | Attention modification |
| EMA + SWA weight averaging | -0.003 | Training technique |
| LeakyReLU² activation | -0.003 | Activation (one-line change) |
| Partial RoPE + LN Scale | -0.002 | Position encoding |
| GPTQ-lite clip search | -0.001 | Better quantization |
| Legal TTT at eval | -0.003 | Eval-time adaptation |
| **Full Hessian GPTQ (self-gen calibration)** | **-0.005** | **Better quantization** |

**Pattern:** ALL successful improvements are architectural, quantization, or eval-time. ZERO loss function modifications succeeded.

---

## What We Built (Infrastructure)

Regardless of experimental results, we built significant infrastructure:

### Analysis Tools
- `scripts/error_analysis.py` — per-token error decomposition of trained models
- `scripts/analyze_training_data.py` — FineWeb data statistics for loss design
- `geometric_field/phase0_analysis.py` — quantization error structure analysis

### Experiment Framework
- `scripts/experiment_runner.py` — automated sweep with statistical testing
- GPU-specific sweep configs (A100 40/80GB, T4, G4)
- Live output streaming during experiments
- Resume-safe skip logic

### Ternary Modifications
- `geometric_field/geometric_field.py` — position-dependent weight modulation
- `geometric_field/ternary_residual.py` — two-scale quantization residual
- `geometric_field/structured_zeros.py` — importance-based zero thresholds
- `geometric_field/activation_experiments.py` — parametric power, stochastic depth, GaugeReLU
- `geometric_field/asymmetric_ternary.py` — learned pos/neg scale asymmetry
- `geometric_field/patch_ternary.py` — patches 70K-line ternary script for non-Hopper GPUs

### Notebooks (on Google Drive)
- `parameter_golf_experiments.ipynb` — int6 SOTA experiments
- `train_and_analyze.ipynb` — 5-hour training + error analysis
- `ternary_geometric_field.ipynb` — all ternary experiments (16 cells)
- `download_data.ipynb` — persistent data to Drive

---

## If We Had More Time

1. **Inline activation modification** — parametric power showed promise per-step but monkey-patching killed step time. Writing it directly into the training script with torch.compile would be a fair test.

2. **Asymmetric ternary** — implemented but not yet run. Zero step overhead, adds 9 effective values per group. Most promising remaining experiment.

3. **Full Hessian GPTQ for ternary** — the new SOTA's key innovation. The ternary model's Q-tax is only 0.0013, but full GPTQ might reduce it further or enable better weight distributions.

4. **Error analysis on the ternary model** — we only did error analysis on the int6 SOTA. Understanding WHERE the ternary model fails differently could inform ternary-specific modifications.

5. **Non-record submission writeup** — the competition explicitly welcomes interesting negative results. Our systematic elimination of loss function approaches, quantization error structure analysis, and activation experiments would make a compelling submission.

---

## Key Data Artifacts (on Google Drive)

| File | What |
|------|------|
| `ternary/final_model_raw_sd.pt` | Trained ternary model state dict (85 keys) |
| `ternary/results/phase0_results.json` | Quantization error structure (Scenario D, 7x structure) |
| `ternary/results/g_signals.pt` | Word-boundary direction + input covariance signals |
| `ternary/logs/geom_*.txt` | 11 geometric field ablation logs |
| `ternary/logs/act_*.txt` | 6 activation experiment logs |
| `results/error_analysis.json` | Per-token error analysis of int6 SOTA (1M tokens) |
| `logs/*.txt` | ~23 int6 experiment logs |

---

## The Leaderboard as of April 1, 2026

| Rank | Entry | BPB | Key Innovation |
|------|-------|-----|----------------|
| 1 | AR Self-Gen GPTQ + XSA-all | **1.1147** | Self-generated GPTQ calibration data |
| 2 | LeakyReLU² + Legal TTT | 1.1194 | LeakyReLU² + test-time training |
| 3 | EMA + GPTQ-lite | 1.1228 | GPTQ-lite clip search + EMA |
| 11 | Ternary (our base) | 1.1570 | 73.7M params at 1.6 bits each |
| — | Baseline | 1.2244 | 9L/512d, int8+zlib |

The competition is dominated by quantization innovations (GPTQ, int6 QAT) and architectural tweaks (XSA, BigramHash, LeakyReLU²). No loss function modifications have succeeded.

---

## In-Progress: Progressive Depth Growing (Int6 SOTA)

**Concept:** Start training with fewer layers (7L, ~55ms/step) for the first 33% of wallclock, then switch to full depth (11L, ~83ms/step). The shallow phase runs ~50% more steps in the same time, exposing the model to more data. At growth transition, dormant layer banks are restored to their initial values (weight decay would have shrunk them).

**Implementation:** `progressive_growing/patch_progressive.py` — patches the SOTA `train_gpt.py` with:
- `run_shallow_step()`: temporarily modifies `num_encoder_layers`/`num_decoder_layers` for forward pass
- Phase 1 uses uncompiled `base_model` (shallow), Phase 2 uses compiled `model` (full)
- Dormant bank restoration at growth transition (if muon_wd > 0)
- Env vars: `GROW_FRACTION` (0.0=disabled, 0.33=grow at 33%), `GROW_INITIAL_LAYERS` (default 7)

**Notebook:** `notebooks/progressive_growing.ipynb` — 4 experiments x 3 seeds

**Status:** Implemented, syntax-verified, awaiting Colab testing.

**Known risks:**
- Newton-Schulz orthogonalization processes full parameter banks including zero-gradient dormant slices — may produce small NS5 artifacts on dormant parameters (mitigated by bank restoration)
- EMA tracks decayed dormant layers during Phase 1 — will self-correct after growth
- VE layers (9-10) are not used during shallow Phase 1 (layers 0-6 only) — acceptable

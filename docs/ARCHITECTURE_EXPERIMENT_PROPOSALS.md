# Architecture Experiment Proposals — From Deep Review

## Source
These proposals come from a deep review of the full research summary, incorporating
all data from Phase 0 analysis, geometric field ablation, error analysis, and the
250-run ternary submission log.

## Core Constraint (proven by data)
**The activation is the only continuous computation. Making it more expressive is
the cheapest way to recover information lost in ternary matmul. Everything that
touches the weights or STE is off limits — the G experiment proved that
co-adaptation is fragile (0.0013 BPB tax → 0.21 BPB tax at beta=0.5).**

---

## Tier 1: High Feasibility, Directly Addresses the Bottleneck

### Experiment 1: Per-Layer Parametric Power Activation ("PowerNorm")

**Idea:** Replace fixed `relu²` with `leaky_relu(x, s) ** p` where `p` is learned per layer.

**Why:** Nobody asked whether exponent 2 is optimal everywhere. Encoder vs decoder
layers in the U-Net, and attention-following vs MLP-internal positions, may want
different powers. Some layers may want p=1.5 (gentler), others p=3.0 (more aggressive
sparsification).

**Cost:** 10 floats (1 per layer) = 20 bytes. ~0% step time overhead (one pow() call).

**Implementation:** ~10 lines. Initialize p=2.0 (known-good), high LR for power params.

**Risk:** Optimizer may not learn p well in 6500 steps. Mitigation: init at 2.0.

**Expected impact:** 0.001-0.002 BPB if layers genuinely want different exponents.

---

### Experiment 2: Stochastic Depth for Ternary Training

**Idea:** During training, randomly skip entire transformer blocks with depth-increasing
probability. Block 0 always runs, block 9 skipped 20% of the time. At eval, all
blocks run with outputs scaled by (1-drop_prob).

**Why:** Standard in vision transformers, never tried on ternary LLMs. Dual benefit:
regularization (prevents inter-layer co-adaptation, different from weight-STE
co-adaptation) AND training speedup (fewer layers per step = more total steps).

**Cost:** Zero artifact bytes. Zero eval cost. NEGATIVE training cost (faster steps).

**Math:** 10% average speedup = ~650 extra training steps in the 600s budget.

**Risk:** Ternary STE might interact badly with dropped blocks (scale factor
miscalibration if block frequently skipped). Mitigation: skip only middle blocks (2-7).

**Expected impact:** 0.002-0.005 BPB from extra steps + regularization. Stacks with
any activation modification.

---

## Tier 2: Medium Feasibility, High Novelty

### Experiment 3: Phase-Magnitude Activation ("GaugeReLU")

**Idea:** Treat adjacent pairs of hidden dimensions as complex numbers. Apply
phase-magnitude decomposition: nonlinearity on magnitude only, preserve phase.

```python
def gauge_relu(x):
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)  # (..., dim//2, 2)
    magnitude = (x_pairs[..., 0]**2 + x_pairs[..., 1]**2).sqrt()
    phase = torch.atan2(x_pairs[..., 1], x_pairs[..., 0])
    mag_activated = F.leaky_relu(magnitude, 0.01) ** 2
    out = torch.stack([mag_activated * phase.cos(),
                       mag_activated * phase.sin()], dim=-1)
    return out.reshape(x.shape)
```

**Why it's novel:** Nobody in ML is doing phase-aware activations in LLMs. relu²
applied independently destroys relative phase information between paired dimensions.
Ternary matmul produces outputs where sign patterns encode discrete phase — preserving
this through the nonlinearity could help the next layer's ternary matmul.

**Cost:** Zero extra parameters. 5-15% step time overhead (atan2, sqrt, cos, sin).

**Risk:** Pairing of dimensions is arbitrary. If pairing doesn't match the model's
internal structure, phase preservation is meaningless. Try multiple strategies.

**Expected impact:** High variance — 0.005+ BPB if phase matters, clean negative if not.

**Theoretical framing:** Gauge-equivariant activation preserving U(1) phase structure.
Connects to fiber bundle / topological defect framework.

---

### Experiment 4: Ternary Bit-Plane Decomposition (Two-Scale Matmul)

**Idea:** Replace one ternary matmul with two at different scales:
`y = (s₁ * T₁) @ x + (s₂ * T₂) @ x` → 9 effective levels per weight.

**CRITICAL CAUTION:** GroupedTernaryLinear lost 0.03-0.06 BPB from cross-group
isolation. Bit-plane is mathematically different (both planes operate on full input
then SUM) but the precedent demands extreme care.

**Cost:** Doubles matmul compute. Apply only to MLP, reduce from 4× to ~2.5× width.

**Risk:** Tight 16MB budget. Cross-group-like effects possible.

**Expected impact:** Potentially 0.01+ BPB if precision-count tradeoff favors this
point on the Pareto frontier. High risk.

---

## Tier 3: High Novelty, Uncertain Feasibility

### Experiment 5: Input-Dependent Activation Shape ("Neural Gain Modulation")

**Idea:** The activation function's power depends on the block's input context:

```python
def modulated_activation(x_intermediate, x_block_input):
    gain = torch.sigmoid(x_block_input @ gain_vector)  # (..., 1)
    power = 1.5 + gain * 1.5  # ranges from 1.5 to 3.0
    activated = F.leaky_relu(x_intermediate, 0.01).abs().pow(power) * x_intermediate.sign()
    return activated
```

**Why:** Different tokens need different activation shapes. The word-boundary analysis
showed 46.5% of errors are BPE fragments — these may need different gain than function
words. This is analogous to thalamic gain modulation in the brain.

**Cost:** 768 floats per block = 15KB total (~1% of budget). Negligible step time.

**Risk:** Gradient path through x^power(x) where power depends on x is complex.
Clamp power range and clip gradients.

**Expected impact:** 0.003-0.010 BPB if context-dependent activations help.

---

### Experiment 6: Activation Residual Learning ("Precision Recovery Network")

**Idea:** After each ternary matmul, add a tiny low-rank continuous correction:

```python
correction = h @ W_down @ W_up  # rank-4, continuous weights
h = h + alpha * correction       # alpha learned, init=0
```

**Why:** Adds continuous-precision computation at the exact bottleneck (between
ternary matmuls). Initialized to zero → model starts as exact baseline → gradually
learns to use correction. Avoids STE disruption because ternary weights are unmodified.

**CAUTION:** LoRA TTT failed catastrophically on ternary. But that was EVAL-TIME
adaptation within RMSNorm space. Training-time correction is different.

**Cost:** rank-4 on MLP hidden: 48KB per layer = 480KB total (3% of budget).
FP8 storage: 240KB. Negligible step time.

**Expected impact:** 0.005-0.015 BPB if it patches the ternary precision bottleneck.
Alpha-init-zero means it can only help (at worst alpha stays ≈0).

---

## Recommended Execution Order

### Must-Run (low risk, fast, stackable):
1. **Parametric power** — 10 lines, 20 bytes, ~0% overhead
2. **Stochastic depth** — simple, speeds up training, free regularization

### High-Novelty Bet (pick ONE):
3. **GaugeReLU** — strongest theoretical narrative, most novel
4. **Neural gain modulation** — strongest neuroscience connection

### Fallback:
- Submit parametric power results (positive or negative) as non-record entry
- The writeup connecting to cortical gain modulation is compelling regardless

### Do NOT attempt without more data:
5. **Bit-plane decomposition** — too risky given GroupedMLP precedent
6. **Precision recovery network** — LoRA TTT precedent and 480KB budget cost

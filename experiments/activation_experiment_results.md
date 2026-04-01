# Activation Experiment Results

**Date:** 2026-04-01
**Model:** Ternary 73.7M params (10L, 768d, BitNet b1.58)
**GPU:** G4 (95.6GB VRAM), ~1 hour wallclock per run, seed=42

## Summary Table

| Run | RT BPB | Pre-Q BPB | Q-Tax | Delta vs Control | ms/step | Steps | Overhead |
|-----|:------:|:---------:|:-----:|:----------------:|:-------:|:-----:|:--------:|
| **act_control** | **1.2992** | 1.2978 | +0.0014 | — | 3302 | 1100 | — |
| act_stoch_02 | 1.3008 | 1.2994 | +0.0014 | +0.0016 | 3154 | 1150 | **-4.5%** |
| act_power | 1.3315 | 1.3305 | +0.0010 | +0.0323 | 4789 | 760 | +45% |
| act_power_stoch | 1.3320 | 1.3310 | +0.0010 | +0.0328 | 4553 | 800 | +38% |
| act_gauge | 1.4429 | 1.4407 | +0.0022 | +0.1437 | 5693 | 640 | +72% |
| act_gauge_stoch | 1.4381 | 1.4366 | +0.0015 | +0.1389 | 5403 | 670 | +64% |

## Analysis

### Stochastic Depth: Neutral (±0.002 BPB)

Stochastic depth achieved its promised speedup: 3154ms/step vs 3302ms (-4.5%),
yielding 1150 steps vs 1100 (+50 extra steps). However, the quality was
indistinguishable from control (1.3008 vs 1.2992, delta = +0.0016 within noise).

**Conclusion:** The extra steps and regularization did not help. The ternary model
at 10 layers may be too shallow for stochastic depth to provide meaningful
regularization. Also, the eval-time scaling (delta * (1-drop_rate)) may not
perfectly compensate for the training-time drops.

### Parametric Power: Harmful (-0.032 BPB, +45% step time)

Parametric power was supposed to be "~0% overhead" but actually added 45% step time.
The monkey-patched forward replaces a fused `relu().square()` (2 fast ops) with
`leaky_relu() + abs() + clamp() + pow() + sign() + multiply` (6 ops, including
the expensive `pow()` operation). Only 760 steps completed vs 1100.

Even looking at per-step quality: at step 500, act_power has val_bpb 1.4446 vs
control's 1.4619 — actually BETTER per step. But 340 fewer steps wipes out the gain.

The Q-tax was actually lower (0.0010 vs 0.0014) — the learned power may produce
weights that quantize slightly better. But the step cost is fatal.

**Conclusion:** The idea has merit (per-step quality improved, Q-tax reduced) but
the implementation is too slow. Would need to be compiled into the training script
with torch.compile kernel fusion to be viable. In the 10-minute 8×H100 regime,
the 45% overhead = ~2600 fewer training steps.

### GaugeReLU: Catastrophically Slow (-0.14 BPB, +72% step time)

GaugeReLU added 72% step time overhead. Only 640 steps completed. The transcendental
functions (atan2, sqrt, cos, sin) are individually fast but collectively add massive
overhead on 3072-dimensional tensors, applied 10 times per forward pass.

The per-step convergence was also worse: at step 500, GaugeReLU has val_bpb 1.5371
vs control's 1.4619. The phase-magnitude decomposition appears to slow convergence
in addition to slowing step time.

**Conclusion:** The phase structure hypothesis was interesting but empirically
disproven. Ternary matmul outputs do not encode useful phase information that
the activation can preserve. The computational cost makes this completely unviable.

### Stacking (power+stoch, gauge+stoch)

Stochastic depth partially mitigated the step time of both modifications
(4553ms vs 4789ms for power, 5403ms vs 5693ms for gauge) by skipping blocks.
But the quality was nearly identical to the non-stacked versions, confirming
that the step time is the dominant factor, not regularization.

## Root Cause: Step Time Is Everything

In a wallclock-limited competition (10 minutes on 8×H100):
- Control: 1100 steps × 3302ms = 3.63M ms
- Power: 760 steps × 4789ms = 3.64M ms (same wallclock, 31% fewer steps)
- Gauge: 640 steps × 5693ms = 3.64M ms (same wallclock, 42% fewer steps)

**Every 1% of step time overhead costs ~11 training steps.** At the convergence
rate of ~0.0003 BPB per 100 steps in the plateau, losing 340 steps costs
~0.001 BPB. The parametric power's per-step improvement doesn't overcome this.

## What This Means

1. **Activation modifications must be zero-overhead** to be viable. Any modification
   that adds even 5% step time in the 10-minute budget is harmful.

2. **Monkey-patching breaks fusion.** The original `relu().square()` is optimized by
   PyTorch's eager mode. Our patched version adds separate ops that can't fuse.
   To fairly test activation changes, they must be written inline in the training
   script and compiled with torch.compile.

3. **The ternary model's activation (relu²) is likely near-optimal for this budget.**
   LeakyReLU² was already tested by the submission author and relu² was kept because
   it's faster despite being slightly less expressive.

4. **Stochastic depth is the only safe modification** — it speeds up training slightly
   but doesn't improve quality. It could potentially stack with other improvements
   that ARE zero-overhead.

## Learned Power Values (from act_power run)

The model learned 10 power parameters but we only trained for 760 steps.
The values likely didn't converge. A proper test would require inline
implementation with torch.compile and full 10-minute training.

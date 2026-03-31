# Geometric Field Ablation Results

**Date:** 2026-03-31
**Model:** Ternary 73.7M params (10L, 768d, 8192 BPE, BitNet b1.58)
**GPU:** G4 (95.6GB VRAM)
**Training:** 1 hour wallclock per run, ~1090-1100 steps, seed=42
**Evaluation:** ternary roundtrip BPB (post-quantization)

## Summary Table

| Run | Alpha | Beta | G Range | Steps | Pre-Q BPB | RT BPB | Q-Tax | Delta vs Control | Step Time |
|-----|:-----:|:----:|--------:|:-----:|:---------:|:------:|:-----:|:----------------:|:---------:|
| **control_1** | 0.0 | 0.0 | — | 1100 | 1.2955 | **1.2968** | +0.0013 | — | 3288.7ms |
| **control_2** | 0.0 | 0.0 | — | 1100 | 1.2971 | **1.2984** | +0.0013 | +0.0016 | 3288.7ms |
| cov_01 | 0.1 | 0.0 | — (0 layers!) | 1100 | 1.2972 | 1.2987 | +0.0015 | +0.0011 | 3288.6ms |
| cov_03 | 0.3 | 0.0 | — (0 layers!) | 1100 | 1.2974 | 1.2990 | +0.0016 | +0.0014 | 3288.7ms |
| cov_05 | 0.5 | 0.0 | — (0 layers!) | 1100 | 1.2966 | 1.2980 | +0.0014 | +0.0004 | 3288.5ms |
| bnd_01 | 0.0 | 0.1 | [0.66, 1.11] | 1090 | 1.2974 | 1.3110 | +0.0136 | +0.0134 | 3304.3ms |
| bnd_03 | 0.0 | 0.3 | [0.38, 1.36] | 1090 | 1.3004 | 1.3523 | +0.0519 | +0.0547 | 3304.6ms |
| **bnd_05** | **0.0** | **0.5** | **[0.25, 1.74]** | **1090** | **1.3004** | **1.5076** | **+0.2072** | **+0.2100** | **3303.6ms** |
| both_01 | 0.1 | 0.1 | [0.66, 1.11] | 1090 | 1.2984 | 1.3113 | +0.0129 | +0.0137 | 3304.4ms |
| both_03 | 0.3 | 0.3 | [0.38, 1.36] | 1090 | 1.2996 | 1.3543 | +0.0547 | +0.0567 | 3303.6ms |
| both_05 | 0.5 | 0.5 | [0.25, 1.74] | 1090 | 1.3019 | 1.4951 | +0.1932 | +0.1975 | 3303.8ms |

**Control mean:** 1.2976 BPB (roundtrip)
**Control variance:** ±0.0008 BPB (between control_1 and control_2)

## Key Observations

### 1. Covariance Signal (alpha) Applied to ZERO Layers

**CRITICAL BUG:** The cov_01/03/05 runs show "Geometric field applied to 0 layers" and
"alpha=0.0" in the G application logs despite GEOM_ALPHA being set. The C_diag signal
from compute_signals.py collected 0 layers (the forward pass dtype mismatch bug).
Without C_diag data, the covariance signal couldn't be applied.

**Result:** cov_* runs are effectively identical to controls (no G applied).
The tiny BPB differences (±0.001) are within noise.

### 2. Word-Boundary Signal (beta) INCREASES Quantization Tax

The boundary signal (delta_e) DID apply successfully to 30 layers.
However, it INCREASED the quantization tax dramatically:

| beta | G Range | Q-Tax | vs Control Q-Tax |
|:----:|--------:|:-----:|:----------------:|
| 0.0 | — | +0.0013 | baseline |
| 0.1 | [0.66, 1.11] | +0.0136 | 10.5x worse |
| 0.3 | [0.38, 1.36] | +0.0519 | 39.9x worse |
| 0.5 | [0.25, 1.74] | +0.2072 | 159.4x worse |

**The quantization tax scales superlinearly with beta.**
At beta=0.5, the Q-tax is 0.21 BPB — catastrophic.

### 3. Pre-Quantization BPB is Barely Affected

| beta | Pre-Q BPB | Delta |
|:----:|:---------:|:-----:|
| 0.0 | 1.2955-1.2974 | baseline |
| 0.1 | 1.2974 | +0.0019 |
| 0.3 | 1.3004 | +0.0049 |
| 0.5 | 1.3004 | +0.0049 |

The pre-quantization BPB is only slightly worse — the training converges
to similar quality. But the post-quantization BPB is MUCH worse because
G distorts the weight distribution that ternary STE was calibrated for.

### 4. Step Time Overhead from G

G adds ~16ms/step (3304 vs 3289ms) = 0.5% overhead. Negligible.

### 5. Zero Fraction Changes with G

Higher beta → higher zero fraction (0.307 → 0.317 at step 500).
G compresses some columns, making more weights round to zero.
This should improve compression but the quality loss dominates.

## Root Cause Analysis

**Why G hurts:** The ternary STE trains the model to work WITH the
quantization — the continuous weights are shaped so that ternary({-1,0,+1})
approximates them well. G changes the quantization behavior AFTER the model
learned to optimize for the standard quantization. The model can't compensate
fast enough in 1100 steps.

**Why the Q-tax explodes with beta:** G compresses word-boundary columns
(G < 1.0 for important dims). These columns now have SMALLER effective
magnitudes → their ternary values carry less information → but the model
still expects those dimensions to carry full information from training.
The mismatch between training dynamics and G-modulated quantization grows
with beta.

## Conclusion

**Geometric field G does NOT help the ternary model.** The approach was
based on the assumption that redistributing quantization error would improve
BPB. Instead, G INCREASES quantization error because it disrupts the
co-adaptation between continuous weights and ternary STE.

**The covariance signal was never tested** (0 layers due to a bug in
compute_signals.py). A re-run with working C_diag might show different
results — but given the boundary signal's catastrophic failure, the
overall approach is likely dead.

**This is a valuable negative result:**
- Phase 0 showed 7x structured quantization error → G SHOULD help in theory
- In practice, ternary STE co-adapts weights with quantization → G fights this
- The tighter the STE calibration (Q-tax = 0.0013), the more harmful G is
- This suggests the ternary model is ALREADY near-optimal for its quantization

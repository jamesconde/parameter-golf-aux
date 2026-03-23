# Citation Checklist for Submission README

## Framing Paragraph (from spec)
"We extend the selective language modeling paradigm (Z. Lin et al., 2024; Su et al., 2024) to the extreme-parameter-budget regime, where a reference model is unavailable. We introduce data-geometry-informed auxiliary losses based on a mixture entropy analysis of the training corpus (Shannon, 1951; Delétang et al., 2024), combining focal loss variants (T.-Y. Lin et al., 2017) with novel inter-layer decorrelation and representation rank penalties inspired by redundancy-reduction methods from self-supervised learning (Zbontar et al., 2021; Bardes et al., 2022). We show that kilobytes of loss function code can substitute for megabytes of model parameters."

## Essential Citations (must appear)
- Shannon, 1951 — entropy of English
- Delétang et al., 2024 — language modeling is compression (ICLR 2024)
- T.-Y. Lin et al., 2017 — focal loss (ICCV, Tsung-Yi Lin)
- Z. Lin et al., 2024 / Rho-1 — selective language modeling (NeurIPS 2024 Oral, Zhenghao Lin)
- Su et al., 2024 / MiLe Loss — focal loss for LM pretraining (NAACL 2024)
- Zbontar et al., 2021 / Barlow Twins — redundancy reduction (ICML 2021)
- Bardes et al., 2022 / VICReg — variance-invariance-covariance (ICLR 2022)
- Jordan et al. / modded-nanogpt — base recipe techniques
- thwu1 — forked Parameter Golf submission (1.1428 BPB)

## Important Citations (include if space permits)
- Ahmad et al., 2022; Dalm et al., 2024 — decorrelated backpropagation
- Papyan et al., 2020 — neural collapse
- Cover & King, 1978 — entropy estimation
- Roy & Vetterli, 2007 — effective rank definition
- Bengio et al., 2009 — curriculum learning
- Lepikhin et al., 2020 — MoE auxiliary losses
- Szegedy et al., 2016 — label smoothing

## Novelty Claims Per Loss

### Focal Loss
- **Borrowed:** (1-p)^gamma mechanism from T.-Y. Lin 2017
- **Novel:** Applying to extreme parameter-budget pretraining (16MB) where reference model (Rho-1) is prohibitive. Using model's OWN confidence as difficulty signal.

### Inter-Layer Decorrelation
- **Borrowed:** Redundancy-reduction principle from Barlow Twins/VICReg (SSL)
- **Novel:** Between-layer (not within-layer) decorrelation as training-time aux loss in dense autoregressive LM. "Morphogen-inspired layer differentiation" framing.

### Representation Rank
- **Borrowed:** Effective rank definition from Roy & Vetterli 2007, variance component from VICReg
- **Novel:** Applying effective-rank penalty to LM hidden states during pretraining under extreme parameter constraints.

### Unigram KL
- **Borrowed:** Structural similarity to label smoothing (Szegedy 2016) and knowledge distillation (Hinton 2015)
- **Novel:** Data-informed prior (actual marginal distribution, not uniform) with decay schedule (full strength early, zero by midpoint). "Distilling" a trivial statistical model.

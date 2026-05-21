"""LeWM-scRNA v2: latent-space world model for cellular perturbation prediction.

Pivot away from v1's joint encoder+decoder training. Key changes vs v1:

  - Encoder + predictor + action MLP train in pure latent space (Phase B).
  - Decoder is decoupled — trained AFTER the world model is frozen (Phase C),
    so it cannot shape the representation.
  - Augmentation-invariance loss (binomial-subsample views) added in Phase A.
  - InfoNCE contrastive centroid kept as the Phase B subspace-separation prior
    (baseline). MCR² is a future ablation, gated off by default.
  - ESM2 panel is PCA-reduced from 5120 -> 1280 dim for v2 (single-linear
    projection downstream). Real ESM2-650M is a future ablation (A10).

References:
  - Balestriero & LeCun 2025 (arXiv:2511.08544) — LeJEPA / SIGReg
  - Van Assel et al. NeurIPS 2025 (arXiv:2505.12477) — JE vs reconstruction
  - Li & Taylor-Weiner Dec 2025 — "Cells are NOT sentences"
  - Peebles & Xie 2023 (arXiv:2212.09748) — DiT / AdaLN-zero
  - Lin et al. 2023 — ESM2
  - Yu/Ma et al. 2020/2022 — MCR² (deferred ablation)
"""

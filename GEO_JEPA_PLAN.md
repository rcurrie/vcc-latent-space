# Geo-JEPA Roadmap

## Prior work — MCR²-based JEPA (geo_jepa_simple.py)

- MLP encoder (2000D → 512 → 512 → 128, L2-normalized) + MCR² (d×d eigvalsh) + JEPA predictor
- 7-room proxy dataset from MNIST [0,1,3,4,7,8,9] with MNAR dropout simulating scRNA-seq
- AdamW, batch 512, 40 epochs (~30s on MPS)
- **Phase A** — DR 1.78–2.29x, off-diagonal cosine ~0.02, clean UMAP separation
- Hard labels (known digit identity) used for Π in MCR² and for diagnostic coloring
- **Phase B** — TrajectoryPredictor with residual displacement, frozen encoder
  - Predicted DR ~1.25-1.37x vs actual DR ~1.9-2.3x
  - Predictions land in correct room neighborhood but lack precision
  - **Bottleneck**: frozen encoder compresses within-room variance; MLP predictor uses simple concatenation for perturbation conditioning

## Current — LeWM world model with SIGReg (lewm_scrna.py)

Inspired by [LeWorldModel](https://le-wm.github.io/) (Maes, Le Lidec, LeCun et al. 2026). Reframes the problem: perturbations are "actions" in a latent world model.

### Key design changes from geo_jepa_simple.py

| Aspect | MCR² approach | LeWM approach |
|--------|--------------|---------------|
| Regularizer | MCR² with hard labels | SIGReg (unsupervised) |
| Encoder output | L2-normalized (unit sphere) | BatchNorm projector (R^d) |
| Target encoder | EMA + stop-gradient | Stop-gradient only (no EMA) |
| Prediction loss | MSE on sphere / cosine sim | MSE in R^d |
| Perturbation conditioning | Concatenation + MLP | AdaLN (planned) |
| Encoder during pert training | Frozen | Joint training (planned) |

### Completed — homeostatic structure (Phase 1)

- MLP encoder + BatchNorm projector (no L2 norm) + JEPA gene-set masking
- SIGReg (Epps-Pulley test on 64 random 1D projections, Cramér-Wold theorem)
- **Result**: DR=2.37x, off-diag cosine ~0.47, 44s training, no labels used
- Separation weaker than MCR² (off-diag 0.47 vs 0.02) but achieved fully unsupervised
- SIGReg converges to ~-0.304 (near-Gaussian marginals confirmed)
- Representation sufficient for fate discrimination — proceed to perturbation prediction

### Next — perturbation prediction (Phase 2)

1. **AdaLN-conditioned predictor** — Adaptive Layer Normalization with zero-initialization at each predictor layer, conditioning on perturbation embedding. Replaces simple concatenation. Zero-init means predictor starts as identity ("predict no change"), learns deviations.
2. **Joint encoder+predictor training** — unlike frozen-encoder Phase B, train both together. SIGReg prevents collapse without freezing.
3. **Loss**: L_pred(trajectory) + λ_sigreg * SIGReg(Z). Optional MCR² overlay later.
4. **Evaluate**: compare trajectory DR against Phase B baseline (1.25-1.37x). Hypothesis: joint training + AdaLN will substantially improve this.

### Future extensions

5. **MCR² overlay** — add MCR² with discovered or known labels for structured subspace geometry on top of SIGReg
6. **Multi-step rollouts** — MPC-style planning for perturbation sequence optimization (CEM from LeWM)
7. **OOD detection** — digit 7 reserved as unseen fate target; measure coding-rate anomaly on predicted trajectories
8. **Validate on real biology** — Tabula Muris or PBMC 10x where cell-type labels exist for ground-truth comparison

## Architecture ideas (future)

- **Gene-set attention encoder** — replace MLP with GeneInteractionPrior (attention over pathway-grouped genes) from geo_jepa_mnist.py
- **ReduNet layers** — iterative compression; need stable d×d formulation (not Cholesky) for MPS
- **Soft Π** — replace hard labels with clustering head for MCR²; loss drives room discovery

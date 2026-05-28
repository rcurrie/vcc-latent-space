# latent-space

An exploration of latent-space world-model approaches for the [Virtual Cell Challenge 2025](https://virtualcellchallenge.org). What started as a LeJEPA-style latent prediction pipeline ended as a hybrid: a frozen SSL encoder paired with a small linear gene-space delta head — because that's what the data turned out to need.

_Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?_

## Where we landed

Final result on the never-touched 100-perturbation Test file (the leaderboard-comparable held-out set):

| variant | Test PDS | Test DES | Test MAE | val→test gap |
|---|---|---|---|---|
| v1 best (gene-space joint train + ESM2) | n/a | n/a | n/a | val=0.544 only |
| v2 latent pipeline (A1, no InfoNCE) | 0.528 | **0.084** | 0.017 | -0.043 (val-overfit) |
| **Hybrid simple-head (1.2M Linear action→gene)** | **0.538** | 0.060 | ~0.020 | **+0.002 (honest)** |

**Headline:** the hybrid simple-head — a 1.2M-param `Linear(ESM2_action_64 → gene_18080)` on top of a frozen SSL encoder — beat the full latent-space pipeline by +0.010 PDS on Test, with the *first positive val→test gap of the project* (meaning it generalizes honestly to perturbations we didn't iterate against). The trade is a smaller DES (0.060 vs 0.084) — the linear can't capture per-perturbation DEG specificity that the latent + decoder path got.

## Two robust empirical findings

**1. Latent-PDS is not a faithful proxy for gene-space PDS.** We measured a latent-PDS of 0.610 with the v5 centroid-only recipe and a Test gene-space PDS of 0.534. We measured a latent-PDS of 0.552 with v2 A1 and got Test gene-space PDS = 0.528. In our hands, *better latent-space discrimination did not produce better gene-space prediction*. Optimize at the metric you actually care about, not its latent-space proxy.

**2. Single-cell noise dominates the perturbation signal; centroids are where the signal lives.** A diagnostic on our trained encoder ([scripts/v2/diagnostic_gaussianity.py](scripts/v2/diagnostic_gaussianity.py)) found:

- Inter-perturbation variance (variance across perturbation centroids in latent space): **0.19**
- Intra-perturbation variance (variance within each pert's cells): **12.68**
- Ratio: **0.015**

Intra-class variance is ~67× the between-class signal. Single-cell prediction objectives spend almost all of their gradient fighting noise. Pseudo-bulk averaging removes that noise — and is exactly what the published VCC 2025 winners did (Arc Institute reported "purely AI-based approaches did not consistently outperform statistical baselines"). Our v5 centroid-only variant tried this in latent space and lifted latent-PDS to 0.610, but the gene-space decoder couldn't translate that gain. Going directly to pseudo-bulk gene-space delta prediction (hybrid simple-head) is what finally moved the Test number.

## Comments on latent vs gene space

We spent most of the project in pure latent space, motivated by:
- Li & Taylor-Weiner's "Cells are NOT sentences" argument that joint-embedding methods avoid the gene-level mean-collapse that reconstruction methods suffer on stochastic count data.
- The LeJEPA / SIGReg / MCR² family's clean theoretical grounding (isotropic-Gaussian manifolds, rate-distortion).
- The conceptual elegance of decoder-free representation learning.

The empirical verdict, from running this on a real biological benchmark:
- **Latent training does not solve the noise problem.** A "perfectly trained" latent space still has intra/inter variance ratio 0.015 — the noise is in the data, not the loss family.
- **The decoder is a one-way bottleneck.** Even when we lifted latent-PDS to 0.610 (v5), the gene-space decoder couldn't recover that improvement. Decoder capacity sweeps confirmed the bottleneck isn't decoder size (a 4× decoder *hurt* DES via per-cell reconstruction overfitting).
- **The actual transferable signal is from ESM2.** Our simple linear delta head from ESM2-derived actions, with no z_ctrl, no SSL encoder shaping, no anti-collapse priors, beat every latent-space variant we tried on Test PDS. The protein-language model's similarity structure is doing the OOD generalization work; the rest of the pipeline either didn't add value or hurt it.

What survived: the SSL-pretrained encoder (Phase A) is *still used* in the hybrid as a frozen feature extractor for the concat-head experiment, and the ESM2 → action projection is the same machinery v2 built. The decoder, perturbation predictor, MCR²-conditional, and the entire Phase B / Phase C structure did not survive.

## Final recipe — hybrid simple-head

```
                                       (frozen, training-only)
   x_control ─▶ Phase A SSL encoder ─▶ z_control_mean
                                           │
                                           │  (ignored in simple-head)
                                           ▼
   gene_idx ─▶ ProteinActionEmbedV2 ─▶ action ──┐
                  (PCA-1280 ESM2,                │
                   single linear → 64)            ▼
                                          Linear(64 → 18,080)        (trainable, zero-init)
                                                 │
                                                 ▼
                                          delta_x ─▶ x_pred = x_control_mean + delta_x
```

- **Frozen:** the Phase A SSL encoder (kept for the diagnostic and to enable other head variants; the simple-head ignores its output).
- **Trainable:** the ESM2-projection inside ProteinActionEmbedV2 (~80k params) + a single zero-initialized `Linear(64 → 18,080)` delta head (~1.16M params).
- **Training:** stratified sampler over training perturbations. Per batch, compute pseudo-bulk per-pert mean and pseudo-bulk control mean; loss is `MSE(predicted_delta, x_pert_mean − x_ctrl_mean)` per pert, averaged.
- **Inference:** pseudo-bulk. Average a sample of training controls; predict delta per test perturbation; add to control mean for the predicted perturbed centroid.
- **Convergence:** ~6 epochs, ~5 minutes on M4.

## Experiments and variations we tried

Not exhaustive, but the major branches:

**Anti-collapse priors in latent space (Phase A pretraining):**
- SIGReg only (K=64 cached projections) — *collapsed to rank ~2/256*
- SIGReg with fresh-resampled K=64 — *collapsed to rank ~1*
- SIGReg + VICReg variance-floor — *still collapsed (informational rank collapse)*
- SIGReg + VICReg variance + covariance — *worked, but is just VICReg with decorative SIGReg*
- SIGReg fresh K=1024 (the LeJEPA "more directions = stronger Cramér-Wold" claim) — *also collapsed; falsifies the claim for our data*
- **MCR² marginal-only (chosen)** — *single rate-distortion log-det handles both variance + decorrelation; rank=255/256*
- Pure invariance with no anti-collapse — *not seriously tried; would collapse*

**Phase B objectives (latent-space world model):**
- MSE in gene space (v1) — *mean collapse, PDS=0.500*
- Latent MSE + InfoNCE contrastive centroid — *lifted latent-PDS but hurt gene-space DES; dropped*
- Latent MSE + MCR²-conditional (A1, chosen) — *Test PDS=0.528*
- A3 augmentation sweep τ ∈ {0.3, 0.5, 0.7} — *τ=0.5 is the sweet spot*
- Delta-residual identity-init predictor (v3) — *mean collapse on the residual skip; Test PDS=0.519*
- Pseudo-bulk auxiliary loss alongside per-cell (v4) — *Test PDS=0.530 ≈ baseline*
- Centroid-only objective (v5) — *latent-PDS=0.610 (peak), Test gene PDS=0.534*

**Phase C decoder:**
- 2-block AdaLN-conditioned, hidden=1024 (chosen) — *Test PDS=0.528*
- 4-block AdaLN, hidden=2048 — *more params, ~0 PDS gain, −0.007 DES (overfit per-cell reconstruction)*

**Hybrid gene-space delta heads (final pivot):**
- 28.7M-param AdaLN delta head — *catastrophic mean collapse, Test PDS=0.498 (below chance)*
- 7M-param `Linear(z_ctrl + action → gene)` concat head — *val PDS=0.547 but Test PDS=0.515; z_ctrl train/test distribution shift caused val-overfit*
- **1.2M-param `Linear(action → gene)` simple head (chosen)** — *Test PDS=0.538, val→test gap +0.002*

**Other:**
- Population-risk gate optimizer (v1) — *killed 77% of params, hurt training, abandoned*
- JEPA masked-genes auxiliary in Phase A — *competed with MCR² (predictability vs spread), dropped*
- Cholesky-on-MPS for MCR² conditional — *4× slower than CPU slogdet at D=256; reverted*
- Gaussianity diagnostic on encoder output — *encoder output is more non-Gaussian than random projection; LeJEPA's worst-case failure mode applies to our data*

## What this project is and isn't

It is an honest exploration of latent-space SSL pretraining for biological perturbation prediction. The main result is that a sequence of theoretically clean latent-space objectives (LeJEPA, MCR², centroid-only) hit a Test-PDS ceiling around 0.53 on this benchmark, and that a much simpler ESM2-conditioned gene-space delta lifted Test PDS to 0.538 with no overfitting.

It is not a competitive VCC entry. The published VCC 2025 winners used pseudo-bulk + statistical features + larger foundation-model backbones (scFoundation, etc.) and produced gene-space PDS numbers we did not approach. Our scoring also uses full-panel L1 rather than the official cell-eval gene-set-restricted L1, so absolute comparisons require a small offset correction we didn't quantify.

See [docs/CURRENT_STATE_FOR_TUTOR.md](docs/CURRENT_STATE_FOR_TUTOR.md) for the long-form journey.

## References

- [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun 2025) — the isotropic-Gaussian SIGReg framework we started from.
- [LeJEPA Identifiability](http://klindtlab.github.io/lejepa-identifiability) (Klindt et al.) — proves LeJEPA's identifiability is Gaussian-specific; helped diagnose where our latent pipeline was overconstrained.
- [Joint Embedding vs Reconstruction](https://arxiv.org/abs/2505.12477) (Van Assel et al. NeurIPS 2025) — motivated binomial-subsample as the principled scRNA-seq augmentation.
- [Cells are NOT sentences](https://iamjli.substack.com) (Li & Taylor-Weiner 2025) — the latent-space pivot argument.
- MCR² (Yu et al. 2020, Ma et al. 2022) — the rate-distortion loss that survived our Phase A iterations.
- DiT / AdaLN-zero (Peebles & Xie 2023) — predictor and decoder conditioning.
- ESM2 (Lin et al. 2023) — the transferable action representation.
- LeWorldModel (Maes, Le Lidec, LeCun et al. 2026) — the original latent-prediction framing.
- [Arc Institute VCC 2025 Wrap-Up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up) — what the winners actually did.

## Run

```bash
# One-time setup
uv sync                                                  # Python 3.12, PyTorch 2.x with MPS
uv run python scripts/v2/build_esm2_panel_pca.py         # PCA-reduce ESM2 panel (~1 min)
uv run python scripts/v2/freeze_internal_val_split.py    # freeze 15-pert internal val

# Phase A: SSL encoder pretraining (~10 min on M4) — still used as frozen feature extractor
uv run python scripts/v2/run_phase_a.py --epochs 30

# Final result: hybrid simple-head (~5 min on M4)
uv run python scripts/v2/run_hybrid.py --epochs 10 --head-type simple

# The latent-space pipeline (kept for reproducibility of the comparison)
uv run python scripts/v2/run_phase_b.py --epochs 40 --contrastive-weight 0.0   # A1 recipe
uv run python scripts/v2/run_phase_c.py --epochs 20                            # decoder + Val scoring
uv run python scripts/v2/score_test.py                                         # Test scoring on the A1 stack

# Diagnostics
uv run python scripts/v2/diagnostic_gaussianity.py     # inter/intra variance, normality tests
uv run python scripts/v2/smoke_test.py                 # pipeline wiring
```

## Data

VCC 2025 data in `data/vcc/`:
- `adata_Training.h5ad` — 221k cells × 18,080 genes, 150 perturbations + 38k controls
- `adata_Validation.h5ad` — 50 disjoint perturbations (the dev set we made model decisions on)
- `adata_Test.h5ad` — 100 more disjoint perturbations (never touched until final scoring)
- `gene_esm2_panel.pt` — UCE-shipped ESM2-15B embeddings (5120-dim), PCA-reduced to 1280 by `scripts/v2/build_esm2_panel_pca.py`

v1 building blocks that v2 reuses (encoder, predictor, AdaLN block, SIGReg, contrastive centroid loss, data loader) live in `src/lewm/`. Earlier MNIST proxy exploration in `legacy/`. The v1 training/eval/scoring code was deleted in favor of v2 — recoverable via `git log`.

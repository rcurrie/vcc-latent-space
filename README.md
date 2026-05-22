# latent-space

A latent-space world model for the [Virtual Cell Challenge 2025](https://virtualcellchallenge.org). Treats CRISPRi perturbations as "actions" in a learned 256-dim latent space; predicts post-perturbation cell state in latent only; decodes to gene space only as a post-hoc evaluation step.

_Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?_

## Approach (v2)

Three phases. The world model is shaped entirely by latent-space objectives — the decoder cannot influence representation learning.

- **Phase A — encoder pretraining (controls only, ~10 min).** MLP encoder `R^18080 → R^256`. Trained on 38k non-targeting control cells with two losses: augmentation invariance (paired binomial-subsample views at τ=0.5) + MCR²-marginal rate-distortion (anti-collapse).
- **Phase B — world-model training (joint, no decoder, ~85 min).** Adds ProteinActionEmbedV2 (PCA-1280 ESM2-15B → single linear → 64-dim action) and an AdaLN-zero conditioned 4-layer transformer predictor. Losses: latent prediction MSE + invariance + MCR²-conditional (per-pert orthogonal subspaces).
- **Phase C — post-hoc decoder (frozen world model, ~22 min).** Trains a small AdaLN-conditioned decoder `(z, action) → x` for gene-space scoring. Cannot influence representation.

## Results

Three held-out splits, in increasing order of generalization difficulty / honesty:

| split | n perts | what it is | PDS | DES | MAE |
|---|---|---|---|---|---|
| internal-val (15 perts carved from training) | 15 | Phase B training-time monitor; latent-PDS only | 0.552 | — | — |
| `adata_Validation.h5ad` | 50 | **dev set; we made model-recipe decisions here** | 0.571 | 0.089 | 0.017 |
| `adata_Test.h5ad` | 100 | **never touched until final scoring — leaderboard-comparable** | **0.528** | **0.084** | **0.017** |

**The headline is the Test number: PDS=0.528, DES=0.084.** It is +0.028 above chance (0.500), a real OOD generalization signal. The Validation number (0.571) is +0.043 above Test because we used Validation for model selection (A1, A3, big-decoder ablations were all decided on val metrics), not because the model is genuinely that good on novel perturbations.

Total training: ~2 hours on M4.

Caveats on comparing to the published VCC 2025 leaderboard:

- The official scoring uses the [cell-eval](https://github.com/ArcInstitute/cell-eval) library with a gene-set-restricted L1 metric and cohort normalization. Our scoring uses full-panel L1 with our own implementation (`src/lewm/v2/eval_gene.py`). The two metrics aren't bit-identical — there's a known offset we haven't quantified.
- v1's reported PDS=0.544 was also on Validation, never on Test. A fair v1-vs-v2 comparison would re-score v1's checkpoint on Test; we have the checkpoint but the v1 eval module is in git history (pruned in commit `9a1eb45`) — recoverable.

## Approach lineage and ablations

The path to v2 A1 was a sequence of falsified hypotheses. Each row is one experiment we ran on the way. All `PDS`/`DES` numbers in this table are on `adata_Validation.h5ad` (50 perts) — *that's the set we made decisions on*. The headline Test number (0.528) above is the proper held-out score after the recipe was locked.

| variant | val PDS | val DES | takeaway |
|---|---|---|---|
| v1 baseline (gene-space joint + SIGReg) | 0.500 | 0.075 | mean collapse on stochastic count data |
| v1 + InfoNCE contrastive | 0.506 | 0.071 | in-dist OK, no OOD transfer |
| v1 + ESM2 protein actions | 0.544 | 0.076 | first real OOD signal — kept the idea, pivoted everything else |
| v2 SIGReg-only Phase A | (rank=2/256) | — | invariance-alone collapse; K projections not enough |
| v2 SIGReg + VICReg var+cov | (rank=249/256) | — | works, but recreates VICReg + decorative SIGReg |
| v2 SIGReg fresh K=1024 (var/cov off) | (rank=4/256) | — | falsifies "more directions = real Cramér-Wold pressure" claim |
| **v2 MCR²-marginal (no SIGReg, no VICReg)** | (rank=255/256) | — | single rate-distortion log-det handles variance + decorrelation |
| v2 + JEPA masked-genes aux | — | — | JEPA *competes* with MCR² (predictability vs spread); dropped |
| v2 Phase B with InfoNCE | 0.550 | 0.073 | latent-PDS=0.614 great, but decoder can't translate tight clusters |
| **v2 A1: drop InfoNCE** (chosen) | **0.571** | **0.089** | **best on val; Test=0.528 / 0.084** |
| A3 τ=0.3 (heavier aug) | 0.547 | 0.082 | aug too aggressive |
| A3 τ=0.7 (lighter aug) | 0.549 | 0.082 | aug too gentle; τ=0.5 is the sweet spot |
| Bigger decoder (4×2048) | 0.573 | 0.082 | decoder capacity not the bottleneck — bigger *hurts* DES |

**Two robust empirical findings from the journey:**

1. **Latent-PDS is not a faithful proxy for gene-space PDS.** Tighter latent clusters (InfoNCE) don't decode better. Optimize at the metric you care about.
2. **The simplest grounded recipe wins.** MCR²-conditional + invariance + pred. Three losses, all rate-distortion / information-theoretic. No SIGReg, no VICReg variance+covariance stack, no InfoNCE.

See [docs/CURRENT_STATE_FOR_TUTOR.md](docs/CURRENT_STATE_FOR_TUTOR.md) for a comprehensive backgrounder.

References:
- [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun 2025) — the SIGReg / isotropic-Gaussian framework whose strong claims we ended up falsifying.
- [Joint Embedding vs Reconstruction](https://arxiv.org/abs/2505.12477) (Van Assel et al. NeurIPS 2025) — why latent-space prediction beats reconstruction for noisy data; binomial subsample as the principled scRNA-seq augmentation.
- [Cells are NOT sentences](https://iamjli.substack.com) (Li & Taylor-Weiner 2025) — empirical argument for the latent-space pivot.
- MCR² (Yu et al. 2020, Ma et al. 2022).
- DiT / AdaLN-zero (Peebles & Xie 2023).
- ESM2 (Lin et al. 2023).
- LeWorldModel (Maes, Le Lidec, LeCun et al. 2026) — the original framing.

## Run

```bash
# One-time setup
uv sync                                                  # Python 3.12, PyTorch 2.x with MPS
uv run python scripts/v2/build_esm2_panel_pca.py         # PCA-reduce ESM2 panel (~1 min)
uv run python scripts/v2/freeze_internal_val_split.py    # freeze 15-pert internal val

# Full v2 pipeline (~2 hours on M4)
uv run python scripts/v2/run_phase_a.py --epochs 30
uv run python scripts/v2/run_phase_b.py --epochs 40 --contrastive-weight 0.0   # A1 recipe
uv run python scripts/v2/run_phase_c.py --epochs 20                            # trains decoder + scores Validation

# After everything else: score against the never-touched Test file (the honest number)
uv run python scripts/v2/score_test.py

# Smoke test
uv run python scripts/v2/smoke_test.py
```

## Data

VCC 2025 data in `data/vcc/`:
- `adata_Training.h5ad` — 221k cells × 18,080 genes, 150 perturbations + 38k controls
- `adata_Validation.h5ad` — 50 disjoint perturbations (the held-out scoring set)
- `adata_Test.h5ad` — 100 more disjoint perturbations
- `gene_esm2_panel.pt` — UCE-shipped ESM2-15B embeddings (5120-dim)

v1 building blocks that v2 reuses (encoder, predictor, AdaLN block, SIGReg, contrastive centroid loss, data loader) live in `src/lewm/`. Earlier MNIST proxy exploration in `legacy/`. The full v1 training/eval/scoring code (Phase 2 / 3.x recipes, population-risk gate experiment, plotting scripts) was deleted in favor of v2 — recoverable via `git log` if needed.

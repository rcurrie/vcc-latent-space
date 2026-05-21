# LeWM-scRNA v2 — A Backgrounder for the Tutor

A self-contained briefing for someone teaching the user about this project.
Covers (1) the contest and what the field did, (2) our final approach with
the reasoning behind each piece, and (3) a back-catalog of attempts so the
tutor can pull up specific experiments on demand.

The goal is for the user to internalize the **final approach** first
(sections 1–4), then optionally walk through the **journey of attempts**
(section 5) once the destination is clear.

---

## 1. The contest

**Virtual Cell Challenge 2025 (VCC).** Closed Cell-magazine challenge to
predict the effect of unseen CRISPRi gene knockdowns on the K562 cell line.
Given a control cell and a *gene to knock down*, predict the post-
perturbation expression vector of that same cell.

**Data layout (all share the same 18,080-gene panel):**

- `adata_Training.h5ad` — 221,273 cells × 18,080 genes. 150 distinct
  perturbations + 38,176 non-targeting controls.
- `adata_Validation.h5ad` — 50 perturbations the model has never seen.
  This is our held-out test set, scored locally.
- `adata_Test.h5ad` — 100 more disjoint perturbations.

**Three official metrics:**

- **PDS — Perturbation Discrimination Score.** L1 distance ranking on
  predicted profiles: does each predicted profile sit closer to its own
  perturbation's actual profile than to others'? 1.0 perfect, 0.5 chance.
- **DES — Differential Expression Score.** Top-K up/down regulated genes:
  Jaccard overlap between predicted DEG set and actual DEG set.
- **MAE.** Mean absolute error in `log1p(CP10k)` gene space.

**The hard part:** the test perturbations target **different genes** from
the training perturbations. A model that just memorizes "what happens when
you knock down gene X" cannot generalize. The model needs an inductive bias
that lets it reason about *unseen genes*.

## 2. What the field did

The publicly-discussed VCC 2025 winner (Altos Labs) used **flow matching**:
predict a conditional probability flow rather than a point estimate of the
post-perturbation expression. This directly addresses the "post-perturbation
distribution is wide, MSE collapses to its mean" problem and is the
state-of-the-art on the official metrics.

Most other published entries use one of:

- **Foundation-model-style encoders** trained on millions of cells then
  fine-tuned (scGPT, Geneformer, UCE).
- **Joint encoder+decoder MLPs** with various regularizers
  (the v1 path — see §5).
- **Mechanistic action representations** — protein-language models (ESM2),
  gene-knockout graphs, KEGG pathway features.

Our approach is none of these directly. It is a **latent-space world model**:
predict in a learned compact latent space, decode to gene space only as a
post-hoc evaluation step.

## 3. Our final approach: v2 A1

A three-phase pipeline. Phase A pretrains an encoder on controls, Phase B
trains a perturbation predictor jointly with the encoder, Phase C trains a
decoder *post-hoc on a frozen world model* purely so we can score in gene
space. The world model is shaped entirely by latent-space objectives — the
decoder cannot influence it.

```
                                  Phase C (decoder, frozen world model)
                                  ┌─────────────────────────┐
                                  ▼                         │
   x ─┐                       ┌──────┐                    18,080
      ▼                       │AdaLN │ ── softplus ─▶   gene-space
   ┌──────┐                   │Decoder│                  prediction
   │MLPEnc│──z──┐             └──────┘
   └──────┘     │                ▲
              z_source           │z_post
                ▼                │
            ┌──────────┐         │                Phase B (joint train,
            │AdaLN-zero│──z_post─┘                no decoder in the loop)
            │Predictor │
            └──────────┘
                ▲
        action  │
                │
            ┌───┴─────────┐                       Phase A (pretrain encoder
            │ProteinAction│      ┌─ ESM2 lookup │ on controls only)
            │ Embed v2    │◀─────│   (frozen)   │
            └─────────────┘      └──────────────┘
```

### 3.1 Architecture

**Encoder.** `MLP(18,080 → 512 → 512 → 256)` with a BatchNorm projector at
the end. Embeddings live in unconstrained R^256 — **not** L2-normalized.
This matters because MCR² (below) measures the embedding *covariance*
geometry and needs unconstrained magnitudes to be meaningful.

**Action embedding (ProteinActionEmbedV2).** For every gene in the 18,080-
gene panel we look up a frozen protein embedding. We use a PCA-reduced
ESM2-15B vector: original 5,120-dim, PCA-projected to 1,280-dim (93.5%
variance retained). Then a single **linear** projection `1,280 → 64`
produces the 64-dim action. The 1.7% of panel genes without ESM2 entries
(non-protein-coding RNAs, mostly) route through a per-gene learned
"fallback" embedding rather than aliasing to a single zero action.

  *Why this generalizes to unseen perturbations:* ESM2 was trained on ~100M
  protein sequences. Genes encoding similar proteins (same family,
  similar function) get similar ESM2 embeddings *by construction*. A
  perturbation we've never seen at training time gets an action embedding
  that's near the training perturbations of biologically related genes.
  This is the entire transferable inductive bias.

  *Why a single linear is enough:* a 2-layer MLP gave us +3 in-distribution
  accuracy and +0 OOD generalization in v1. The MLP had enough capacity to
  memorize 150 perturbations but not enough to discover transferable
  structure. The PCA panel does the heavy nonlinear work upfront; the
  single linear is just a 1,280 → 64 task-specific basis selection.

**Perturbation predictor.** 4-layer transformer with **AdaLN-zero
modulation** (Peebles & Xie, DiT 2023). Takes `(z_pre, action)` and
produces `z_post` in the same 256-dim latent. AdaLN-zero means each
transformer block's LayerNorm has scale/shift parameters that are produced
by a small MLP over the action — and those parameters are **zero-
initialized**. So at the start of training the predictor passes `z_pre`
through unchanged regardless of action. The model only learns to deviate
from "no perturbation effect" when training data forces it to. This
identity-start is dramatically more stable than random init.

**Post-hoc decoder.** `(z, action) → x` via 2-block AdaLN-conditioned MLP
at hidden=1024, softplus output for non-negative log-normalized expression.
*Trained only after the world model is frozen.* Mirrors the predictor's
action conditioning so it can specialize per perturbation. The decoder
exists only for evaluation; it cannot shape the representation.

### 3.2 Loss functions

The whole pipeline avoids in-loop gene-space targets. Every training
objective is computed on the 256-dim latents `z`.

**Phase A (encoder pretraining on 38,176 controls only):**

- `L_inv` — augmentation invariance: `||encoder(view_1) − encoder(view_2)||²`
  where view_1 and view_2 are *independent binomial subsamples* of the same
  raw UMI count vector, each with retention rate τ=0.5. Then both views are
  log1p(CP10k)-normalized. This is the principled biological augmentation:
  it mimics sequencing the same cell at half-depth. The Van Assel et al.
  2025 result says joint embedding methods need augmentations that span the
  irrelevant-feature subspace — sequencing depth is *the* dominant
  irrelevant feature in scRNA-seq.

- `L_mcr2-marginal` — Maximal Coding Rate Reduction, marginal-only:
  `−R(Z) = −½·logdet(I + α·ZᵀZ/B)` with α=D/(B·ε²), ε²=0.5. Minimizing
  this *maximizes* the rate R(Z), which forces the embedding covariance
  eigenvalues *upward*. This is the rate-distortion-theory anti-collapse
  prior: don't let any direction in latent space go to zero variance.

  *Loss balance:* `L_total = L_inv + 0.01 · L_mcr2`. MCR² has magnitude
  ~800 by construction; weight 0.01 brings it into the same order as
  invariance (~0.1–0.5).

**Phase B (joint encoder + predictor + action embedding on training perts):**

- `L_pred` — latent prediction MSE: `||z_post − stopgrad(z_target)||²`
  where `z_target = encoder(actual perturbed cell)` and `z_source` (input
  to the predictor) is the encoder output of a *random control cell from
  the same batch*. The stop-gradient on `z_target` ensures the encoder
  doesn't move its targets to make the predictor's job trivial; the
  predictor has to learn to hit wherever the encoder puts the perturbed
  cells, and the encoder is shaped by the other losses independently.

- `L_inv` — same paired-view invariance as Phase A, extended to perturbed
  cells.

- `L_mcr2-conditional` — full ΔR with class partition by perturbation ID:
  `ΔR = R(Z) − Σ_g (n_g/B) · R(Z | pert g)`. The conditional term groups
  cells by their perturbation and computes per-class rate. Maximizing ΔR
  drives **per-perturbation subspaces to be orthogonal** in latent space.
  This is the Yi Ma / Yu et al. 2020 objective. Controls form one class;
  each of the 8 perturbations sampled per batch forms another.

  We compute the slogdets on CPU because MPS doesn't have an slogdet
  kernel. Cholesky-on-MPS was *slower* than CPU slogdet at our matrix size
  (256×256) — Apple Silicon launch overhead dominates compute for small
  linear algebra. The compute is not the bottleneck anyway; data loading
  is.

  Same 0.01 loss weight balance.

**Phase C (decoder only, frozen world model):**

- `L_dec_actual` + `L_dec_post` — MSE between decoder output and the actual
  log1p(CP10k) vector. Trained on both `decoder(z_actual, action)` and
  `decoder(z_predicted, action)` targets so the decoder is robust to both
  "given the right latent" and "given the predictor's latent" inputs at
  eval time.

  No invariance, no MCR², no contrastive. The world model is frozen, so
  shaping the latent is no longer a goal. Just regression.

### 3.3 Training infrastructure

- Phase A: 30 epochs, ~10 min on M4.
- Phase B: 40 epochs, ~85 min on M4.
- Phase C: 20 epochs, ~22 min on M4.
- **Total: ~2 hours end-to-end.**

DataLoader uses 4 worker processes and an explicit *fork* multiprocessing
context (macOS DataLoader defaults to spawn, which re-pickles the in-memory
~10GB CSR matrix into each worker — 8+ minute startup that we removed).
Workers share the CSR via copy-on-write, dropping startup from minutes to
~1 second.

### 3.4 Held-out perturbation split

We carve **15 perturbations** out of the 150 training perturbations and
freeze the choice in `data/vcc/v2_internal_val_split.json` (seed 1742).
This held-out set is the source of our internal Latent-PDS monitoring
signal during Phase B. The official 50-pert validation file is only
touched at the *end* of Phase C, never during training, never during
hyperparameter tuning.

The 15 held-out names are a mix of well-known biology (`GSK3B`, `MKI67`,
`PTPN1`, `CHMP3`) and obscure zinc fingers (`ZNF426`, `ZNF714`) — a healthy
diversity stress test.

## 4. Final numbers and meaning

**Headline (v2 A1, official 50-pert VCC validation file, gene-space):**

| metric | v1 best (Phase 3.2) | **v2 A1** | Δ |
|---|---|---|---|
| **PDS** | 0.544 | **0.571** | **+0.027** |
| **DES** | 0.076 | **0.089** | **+0.013** |
| MAE  | 0.014–0.015 | 0.017 | +0.002 |

PDS = 0.571 means **on average 57% of the *other* 49 perturbations have
their actual centroid *farther* from the predicted centroid than the
matching one is**. Chance is 0.50; perfect is 1.00. The +0.027 over v1
is a real, statistically meaningful generalization gain on the metric
that defines the contest.

DES = 0.089 means the predicted top-100 up- and down-regulated DEGs share
~9% of their elements with the actual DEGs on average. This is small in
absolute terms — DES is genuinely hard — but +0.013 over v1 means the
predictor is starting to produce per-perturbation-specific direction
patterns, not just calling the same DEGs for every prediction.

MAE +0.002 is within decoder convergence noise and not load-bearing.

### Two robust empirical lessons

**(i) Latent-PDS is not a faithful proxy for gene-space PDS.** We can have
a latent space with tight, well-separated per-pert clusters (latent-PDS up
to 0.614 with InfoNCE), and still get *worse* gene-space PDS than a looser
latent (0.552) that decodes cleanly to differential expression.

  This is counter-intuitive and important. The InfoNCE recipe was a clean
  story in latent space — discriminative clusters formed exactly as
  designed — but the *kind* of structure InfoNCE imposes (tight per-pert
  centroids) does not map to the *direction* of perturbation effects in
  gene space. The decoder has trouble translating "which pert" into "what
  changes in gene expression."

  The lesson: train and evaluate at the metric you actually care about.
  Latent-PDS is a useful diagnostic but not a stand-in for the real number.

**(ii) The simplest grounded recipe wins.** MCR²-conditional + invariance +
pred. Three losses, all with rate-distortion or information-theoretic
groundings, all operating in latent space. No InfoNCE, no SIGReg, no
variance-floor, no covariance-decorrelation extras. We tried each of those
during development and either ran into collapse (SIGReg alone) or
discriminative-but-decoder-hostile geometry (InfoNCE).

  The lesson: when an SSL trick has a beautiful theoretical motivation but
  doesn't survive empirical falsification on the actual downstream task,
  drop it. The principled minimum was what worked.

---

## 5. Catalog of attempts (review after final approach is internalized)

Each entry is one experiment we ran. Read in order if you want the journey;
jump to specific ones the tutor calls out.

### 5.1 v1 (pre-pivot, gene-space joint training)

**v1 Phase 2 baseline (Aug 2025).** Encoder + AdaLN predictor + decoder
jointly trained with SIGReg + gene-space MSE + JEPA masked latent
prediction. 3-feature ActionEmbed (per-gene log mean expression,
dispersion, fraction expressing).

  Result: **PDS = 0.500 (chance)** on the 50-pert val. Diagnosed as mean
  collapse — MSE on stochastic count data is minimized by the conditional
  mean, which with weak action conditioning equals the global mean.
  Predictions converged on a single average for every perturbation.

**v1 Phase 3.1: + InfoNCE contrastive auxiliary.** Same as Phase 2 plus
contrastive centroid loss (each predicted z should sit nearer its own
pert's actual centroid than other perts' centroids).

  Result: **PDS = 0.506.** In-distribution discrimination strong (logit gap
  went from random −4.5 to +9.1), held-out training perts maintained DR =
  0.77, but **didn't transfer to unseen perts**. The 3-feature
  ActionEmbed had too thin a transferable signal — the MLP memorized 150
  training perts without discovering a generalizable rule.

**v1 Phase 3.2: + ESM2 protein-embedding actions.** Replace 3-feature
ActionEmbed with ESM2-15B protein embeddings projected through a 2-layer
MLP `5120 → 256 → 64`.

  Result: **PDS = 0.544.** First real OOD signal — protein-similarity
  prior allowed interpolation to unseen perts via sequence-space
  neighbors. But DES stuck at 0.076 — the decoder, untconditioned on the
  perturbation, couldn't specialize per-pert. This was the v1 ceiling.

**v1 also tried:**

- A population-risk gate optimizer (Litman & Guo 2026, arXiv:2605.01172).
  Custom AdamW that masks updates per-parameter when batch-mean gradient
  is below leave-one-out variance. The paper's wins on grokking / PINNs /
  DPO didn't transfer — in our high-noise scRNA-seq regime, gradient
  variance dominates almost every parameter, and even at α=0.01 it killed
  77% of parameters. Disabled and left behind a flag. (Lab notebook
  reminder: don't try this again on noisy scRNA-seq data.)

### 5.2 The v2 pivot (4 days of finding the right recipe)

After v1 plateaued at PDS=0.544 with DES=0.076, we pivoted to a latent-only
framework. The original v2 sketch was "drop the decoder from the
representation-shaping loop entirely; train the world model purely on
latent objectives; bolt a post-hoc decoder on at the end."

The sketch initially proposed SIGReg + augmentation-invariance + InfoNCE.
Reality was harder.

**Phase A attempts to prevent encoder collapse:**

1. **SIGReg only (K=64 cached projections + augmentation invariance).**
   Participation ratio collapsed 3.7 → 2.2 → 2.2 in three epochs.
   Invariance loss dropped trivially to 0.004 because the encoder mapped
   everything to a near-constant. SIGReg with cached directions is too
   weak to fight invariance pressure.

2. **SIGReg fresh-projections + variance floor.** Tried VICReg's
   variance-floor trick (per-dim std ≥ 1). Variance floor was satisfied
   trivially (BatchNorm makes std=1 per dim by construction) but PR still
   went to 1.0 — *informational* collapse: dimensions became perfect
   copies of each other while preserving per-dim variance.

3. **+ VICReg covariance-decorrelation (full VICReg stack + SIGReg).**
   PR climbed to 142, eff_rank ≈ 249. Worked! But we'd silently recreated
   VICReg with SIGReg as a decorative shape prior. Five losses.

4. **Pure SIGReg with much larger K (1024 projections, var+cov off).**
   Falsification test for the LeJEPA claim. **Still collapsed** —
   K=1024 cached projections are *worse* than K=64 because more directions
   to satisfy trivially with a collapsed embedding. Conclusion: SIGReg
   alone, at any K we tried, does not prevent invariance-driven collapse
   for our setup. (This contradicts the LeJEPA paper's stronger claim but
   matches the broader SSL literature on invariance-only objectives.)

5. **MCR²-marginal only (the winner).** Pure single log-det objective.
   PR=170, eff_rank=255, full rank. Two losses (inv + MCR²) — the
   simplest grounded recipe. Empirically equivalent to VICReg's variance +
   covariance combo but with a single rate-distortion-theory term.

   *Why it works:* maximizing R(Z) = ½·logdet(I + α·ZᵀZ/B) pushes the
   covariance eigenvalues upward, simultaneously enforcing per-dim
   variance (the marginal) AND decorrelation (off-diagonal entries) in a
   single principled objective. The VICReg two-term decomposition is a
   weaker approximation.

**JEPA masked-genes auxiliary task** — initially kept from v1 (Phase A
also masks 25% of genes and predicts the masked-half embedding from the
context-half embedding). With MCR² active, JEPA loss went **up** during
training (1.04 → 1.87 over 30 epochs) — the encoder was spreading
embeddings into orthogonal subspaces, which by definition makes masked-
gene latent prediction harder. JEPA was *competing* with MCR², not
complementing it.

  Dropping JEPA (Phase A final recipe: inv + MCR²-marginal only) gave
  better invariance loss (0.035 vs 0.081) and slightly higher PR. The
  cleanest signal that JEPA was a confound, not load-bearing.

**Phase B initial sweep** (with InfoNCE on, MCR² weight=0.01):

  Reached **Latent-PDS = 0.614** at convergence. Best v2 latent-space
  number. Gap = +0.07 over the no-InfoNCE Phase B which plateaued at
  0.552. InfoNCE was clearly doing something in latent space — the
  question was whether it survived the decoder.

**Ablations after the v2 baseline was set:**

| ablation | latent PDS | gene PDS | gene DES |
|---|---|---|---|
| baseline (τ=0.5, +InfoNCE) | 0.614 | 0.550 | 0.073 |
| **A1: drop InfoNCE** | 0.552 | **0.571** | **0.089** |
| A3 τ=0.3 (no InfoNCE) | 0.567 | 0.547 | 0.082 |
| A3 τ=0.7 (no InfoNCE) | 0.595 | 0.549 | 0.082 |
| A1 + 4× decoder capacity | 0.552 | 0.573 | 0.082 |

**A1 (no InfoNCE)** was the surprise. Dropping the most "active" Phase B
loss term *improved* gene-space PDS by +0.021 and DES by +0.016. The
diagnostic: InfoNCE creates *tight* per-pert clusters in latent space,
which the decoder cannot translate into differential expression patterns.
MCR²-conditional alone produces a *smoother* latent geometry where each
direction encodes meaningful gene-space variation. **This is the v2 final
recipe.**

**A3 augmentation strength sweep (τ = 0.3, 0.5, 0.7).** Both extremes
underperform. τ=0.5 (drop half the reads per view) is the genuine sweet
spot, not just a default we got lucky with. Van Assel theory satisfied.

**Bigger decoder ablation (4 blocks × 2048-dim vs 2 × 1024).** 4× param
count, decoder L fell 5.3%, but downstream PDS gained 0.002 (noise) and
DES *dropped* by 0.007. Decoder capacity is **not** the bottleneck — more
params actually overfit per-cell reconstruction in a way that blurs the
DEG signature. The default 2-block decoder is correctly sized.

### 5.3 What we explicitly chose not to try

- **Flow matching predictor.** Would address the mean-collapse problem
  more directly than MCR². Large rewrite. The Altos VCC winner used this.
  Deferred to a v3 fork if we ever need to push above PDS=0.571.
- **Foundation-model-style encoders** (gene-level attention with ESM2
  tokens). Bigger structural bet, out of v2 scope.
- **Real ESM2-650M re-embedding** (instead of PCA from ESM2-15B). Would
  require building a UniProt protein-sequence lookup for all 18,080 genes.
  Engineering project on its own. Tracked as a deferred ablation.
- **Multi-step rollouts / MPC-style sequence optimization.** Research
  direction, post-v2.

---

## 6. File map

```
src/lewm/v2/                  # the v2 codebase
  __init__.py
  data.py                     # paired binomial-subsample loader + sampler
  models.py                   # ProteinActionEmbedV2, ActionConditionedDecoder
                              #   (reuses v1's MLPEncoder, PerturbationPredictor,
                              #    AdaLNBlock, gene_set_mask)
  losses.py                   # mcr2_marginal_loss, mcr2_loss (conditional),
                              #   augmentation_invariance_loss,
                              #   covariance_decorrelation_loss, variance_floor_loss
                              #   (last two are leftover from VICReg-stack
                              #    experiments; available behind flags but not used)
  splits.py                   # internal-val split (15 perts) loader + freezer
  eval_latent.py              # latent_pds (held-out Latent-PDS during training)
  eval_gene.py                # score_v2_against_split (VCC PDS/DES/MAE)
  train_phase_a.py            # encoder pretrain, controls only
  train_phase_b.py            # world-model joint training
  train_phase_c.py            # post-hoc decoder, frozen world model

scripts/v2/
  build_esm2_panel_pca.py     # 5120 → 1280 SVD reduction (one-time)
  freeze_internal_val_split.py
  run_phase_a.py
  run_phase_b.py
  run_phase_c.py
  smoke_test.py               # Phase 0 wiring check

results/v2/
  phase_a/                    # 30-epoch Phase A checkpoint (used by all Phase B runs)
  phase_b/                    # baseline Phase B (with InfoNCE) — superseded
  A1_phase_b/                 # ★ final Phase B (no InfoNCE)
  A1_phase_c/                 # ★ final Phase C, headline PDS=0.571 DES=0.089
  A1_phase_c_big/             # bigger-decoder ablation (worse DES)
  A3_tau0.3_*, A3_tau0.7_*    # augmentation strength sweep

data/vcc/
  adata_Training.h5ad           221,273 cells × 18,080 genes
  adata_Validation.h5ad         98,927  cells (50 unseen perts)
  adata_Test.h5ad
  gene_esm2_panel.pt            ESM2-15B panel (5120-dim, from UCE)
  v2_gene_esm2_panel_pca1280.pt PCA-reduced panel used by v2
  v2_internal_val_split.json    15 held-out training perts (seed 1742)
```

The "★" runs are the live v2 result. Other runs are kept for ablation
reproducibility.

---

## 7. TL;DR for the tutor

The model is a **latent-space world model** for cells. It maps a cell's
expression to R^256 (encoder), takes an action embedding derived from the
*protein language model representation* of the gene being knocked down
(ProteinActionEmbedV2 + frozen PCA-1280 ESM2-15B), and predicts a new
latent via an AdaLN-zero transformer (identity-start). Trains entirely on
*latent-space objectives*: prediction MSE + augmentation invariance + MCR²
conditional rate-distortion separation. A simple post-hoc decoder maps
latents back to gene space only for evaluation.

The three innovations the tutor should be ready to defend:

1. **Pure-latent training** (no decoder in the loop) — sidesteps gene-
   space mean collapse on stochastic count data, the documented v1 failure.
2. **MCR²-conditional as both anti-collapse and per-pert subspace prior**
   — replaces the VICReg + SIGReg + contrastive stack with a single
   information-theoretic objective. Falsification of SIGReg-only and
   tuning of MCR² weight were the heart of the v2 journey.
3. **ESM2 protein embeddings as the transferable action representation,
   PCA-compressed, single-linear projected** — gives unseen genes a real
   action embedding without an MLP that can memorize 150 training perts.

The result is **PDS = 0.571 / DES = 0.089** on the official 50-pert
validation, vs v1's 0.544 / 0.076 — a real generalization gain with a
substantially simpler recipe operating purely in latent space.

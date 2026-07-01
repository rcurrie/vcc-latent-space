# VCC Latent-Space — Neighborhood Aggregation Experiment Plan

**For:** coding agent operating on `rcurrie/vcc-latent-space` (current `main`).
**One variable per step. Each step has a hard go/no-go gate. Stop at the first failed gate.**

---

## Base starting point (do NOT change)

- **Frozen SSL encoder** = Phase A, **MCR²-marginal** recipe (`scripts/v2/run_phase_a.py`), augmentation-invariant pretraining via binomial count-subsample, non-collapsed (rank 255/256). **Keep frozen. Its only job here is defining KNN neighborhoods.**
- **Current best model** = hybrid simple-head: `Linear(ESM2_action_64 → gene_18080)` predicting a delta added to the control pseudo-bulk mean (`scripts/v2/run_hybrid.py --head-type simple`). Baseline numbers: **Test PDS 0.538 / DES 0.060**.
- **Data:** `data/vcc/adata_{Training,Validation,Test}.h5ad`, ESM2 panel in `gene_esm2_panel.pt`. Building blocks in `src/lewm/`.
- **Thesis under test:** neighborhood size `k` is a bias/variance dial between per-cell (k=1) and global centroid (k=N). Mid-`k` overlapping-KNN pseudo-bulk + a DES-aware loss should **lift DES off 0.060 without sinking PDS below ~0.53.**

---

## S0 — Gate the whole thesis (no model work)

- **Do:** Trace the scorer (`scripts/v2/score_test.py` + cell-eval path). Determine whether DES is computed on a **population of predicted perturbed cells** (needs within-group spread) or on a **single mean profile**.
- **Gate:** If DES rewards population spread → continue. If it collapses to a mean before scoring → STOP and report; thesis can't pay off without also changing what inference emits.

## S1 — Headroom diagnostic (no model work)

- **Do:** Extend `diagnostic_gaussianity.py`. Compute cell-cycle phase via `sc.tl.score_genes_cell_cycle` (Tirosh markers). Report fraction of intra-perturbation variance (the 12.68) explained by phase.
- **Gate:** Record the number. If phase explains ~all of it, flag that H1 headroom is low and neighborhoods ≈ phase bins — proceed but set expectations. Not a kill.

## S2 — Build the primitive (infra only, no training)

- **Do:** New module `src/lewm/neighborhoods.py`. (a) Overlapping-KNN neighborhood construction on **frozen MCR² encoder** embeddings — lift the aggregation primitive from `milopy`, **NOT** Milo's NB-GLM differential-abundance test. (b) Count-corrected pseudo-bulk per neighborhood: **sum raw UMIs → CP10k → log1p** (never mean-of-logs).
- **Gate:** Unit test: pseudo-bulk of one all-cells "neighborhood" reproduces the existing global centroid within tolerance. If not, fix before proceeding.

## S3 — Matched delta targets (training-time)

- **Do:** Per perturbation, build the **set** {Δ_nb} = matched(perturbed nb) − (control nb). Match perturbed↔control neighborhoods by **cell-cycle phase** (cheap path only for now). Keep everything in gene space.
- **Gate:** Sanity: matched control distribution's phase composition aligns with perturbed; deltas are finite and non-degenerate.

## S4 — Spread-only test (the core experiment, MSE only)

- **Do:** Fork `run_hybrid.py` → `run_hybrid_nbhd.py`. `k=50`. Train the existing simple head on the **set of per-neighborhood deltas** (MSE only — **no rank loss yet**). Inference: apply predicted delta to control neighborhoods to emit a population. Score Val + Test.
- **Gate:** **DES moves up from 0.060 AND PDS ≥ ~0.53.** If DES moves on spread alone → thesis is alive, go to S5. If flat → recheck S0, then treat thesis as weak on H1 and stop before adding machinery.

## S5 — Sweep the dial

- **Do:** `k ∈ {20, 50, 100, 200}`, one run each, same everything else. Plot DES and PDS vs k.
- **Gate:** Expect a DES interior optimum (not monotonic to k=N). Pick best k for S6.

## S6 — Put DES in the loss

- **Do:** At best k, add a **DES-aware term** on high-|LFC| genes: differentiable surrogate only (soft-rank / Spearman approx, or sign-margin BCE). Tune its weight vs MSE.
- **Gate:** DES improves over S5 best **without** PDS dropping below baseline. Report final Val + Test PDS/DES/MAE vs the 0.538/0.060 baseline.

## S7 — Is SSL earning its keep? (ablation)

- **Do:** Repeat best config with neighborhoods defined on **plain PCA of log-normalized counts** (no SSL encoder).
- **Gate:** If MCR² neighborhoods don't beat PCA neighborhoods on DES, the frozen encoder is not load-bearing on H1 — record this; it's the main negative result if so.

---

### Guardrails
- Frozen encoder stays frozen; no decoder, no latent regularizer (avoid the one-way-decoder trap).
- One variable per step. Always report Val→Test gap, not Val alone.
- Every step writes its metrics to `results/nbhd/` for comparison against the 0.538/0.060 baseline.
